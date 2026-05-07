"""
Unified XAI Evaluation Script
==============================
Runs faithfulness (deletion AUC) and/or stability (overlap@k) evaluation
across any combination of XAI methods, models, and datasets.

Outputs CSVs to eval/results/ for visualization in a separate notebook.

Usage:
    python eval/run_evaluation.py --config eval/config.yaml

Or with inline overrides:
    python eval/run_evaluation.py --models Apollo-2B --methods lime ig --tasks yn --eval faithfulness
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import pandas as pd
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_llm_toolkit.wrapper import MedicalLLMWrapper, load_medical_llm
from medical_llm_toolkit.explainers.tokenshap.extensions import qa_extractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "Apollo-2B": "FreedomIntelligence/Apollo-2B",
    "MedGemma-4B": "google/medgemma-4b-it",
    "BioMistral-7B": "BioMistral/BioMistral-7B",
    "BioMedLM": "stanford-crfm/BioMedLM",
}

# Default params per method (matching your original evaluation)
METHOD_DEFAULTS = {
    "lime": {"n_samples": 100, "kernel_width": 0.75, "mask_token": ""},
    "ig": {"n_steps": 40, "baseline_type": "pad"},
    "tokenshap_default": {"max_combinations": 80, "sampling_ratio": 1.0, "vectorizer": "tfidf"},
    "tokenshap_correctness": {"max_combinations": 80, "sampling_ratio": 1.0, "vectorizer": "correctness", "correctness_mode": "prob"},
}

EVAL_DEFAULTS = {
    "rank_mode": "absolute",
    "stability_k": 10,
    "mask_token": "",
}


# ---------------------------------------------------------------------------
# Record builders - normalize each method's output to a common format:
#   {tokens, phi_vector, correct_label, static_suffix, question_text, ...}
# ---------------------------------------------------------------------------

def _strip_leading_question_header(words, phi, orig_pos):
    if not words:
        return words, phi, orig_pos
    if words[0] == "Question:":
        return words[1:], phi[1:], orig_pos[1:]
    if len(words) >= 2 and words[0] == "Question" and words[1] == ":":
        return words[2:], phi[2:], orig_pos[2:]
    return words, phi, orig_pos


def _find_question_slice_end(words):
    n = len(words)
    if n == 0:
        return 0
    for i in range(n - 2, -1, -1):
        if words[i] == "Answer" and words[i + 1] == "Choices:":
            return i
    option_labels = {"A.", "B.", "C.", "D.", "E.", "F."}
    last_opt_idx = None
    for i in range(n - 1, -1, -1):
        if words[i] in option_labels:
            last_opt_idx = i
            break
    if last_opt_idx is not None:
        for j in range(last_opt_idx, max(0, last_opt_idx - 12) - 1, -1):
            if words[j] in ("Answer", "Choices:"):
                return j
        return last_opt_idx
    return n


def build_record_lime(lime_out, prompt, row_id, correct_label):
    words_all = list(lime_out.get("words", []))
    attr_all = np.asarray(lime_out.get("word_attributions", []), dtype=float)

    end_idx = _find_question_slice_end(words_all)
    orig_positions = list(range(0, end_idx))
    words = words_all[:end_idx]
    phi_vec = attr_all[:end_idx]

    words, phi_vec, orig_positions = _strip_leading_question_header(words, phi_vec, orig_positions)

    q_text, static_suffix = qa_extractor(prompt)

    return {
        "row_id": row_id,
        "prompt": prompt,
        "tokens": words,
        "phi_vector": phi_vec.tolist(),
        "correct_label": correct_label,
        "question_text": q_text,
        "static_suffix": static_suffix,
        "rebuild_mode": "join",
    }


def _filter_special_tokens(tokens, offsets, phi):
    keep = [i for i, (a, b) in enumerate(offsets) if not (a == 0 and b == 0)]
    return (
        [tokens[i] for i in keep],
        [offsets[i] for i in keep],
        np.asarray(phi, dtype=float)[keep],
    )


def _find_substring_span(haystack, needle):
    start = haystack.find(needle)
    if start < 0:
        raise ValueError("Could not locate question_text inside prompt")
    return start, start + len(needle)


def build_record_ig(ig_out, prompt, row_id, correct_label, wrapper):
    q_text, static_suffix = qa_extractor(prompt)

    enc = wrapper.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    toks_full = wrapper.tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offsets_full = enc["offset_mapping"]
    phi_full = np.asarray(ig_out["attributions"], dtype=float)

    toks_full, offsets_full, phi_full = _filter_special_tokens(toks_full, offsets_full, phi_full)

    # Select question-only tokens by char span
    q0, q1 = _find_substring_span(prompt, q_text)
    sel = [i for i, (a, b) in enumerate(offsets_full) if not (b <= q0 or a >= q1)]
    tokens = [toks_full[i] for i in sel]
    offsets = [offsets_full[i] for i in sel]
    phi_vec = phi_full[sel]

    return {
        "row_id": row_id,
        "prompt": prompt,
        "tokens": tokens,
        "phi_vector": phi_vec.tolist(),
        "correct_label": correct_label,
        "question_text": q_text,
        "static_suffix": static_suffix,
        "token_offsets": offsets,
        "rebuild_mode": "offsets",
    }


def _parse_token_key(k):
    tok, pos = k.rsplit("_", 1)
    return tok, int(pos)


def build_record_tokenshap(token_shap_obj, prompt, row_id, correct_label):
    items = []
    for k, v in token_shap_obj.shapley_values.items():
        tok, pos = _parse_token_key(k)
        items.append((pos, tok, float(v)))
    items.sort(key=lambda x: x[0])

    tokens = [t for _, t, _ in items]
    phi_vec = np.array([phi for _, _, phi in items], dtype=float)

    q_text, static_suffix = qa_extractor(prompt)

    return {
        "row_id": row_id,
        "prompt": prompt,
        "tokens": tokens,
        "phi_vector": phi_vec.tolist(),
        "correct_label": correct_label,
        "question_text": q_text,
        "static_suffix": static_suffix,
        "rebuild_mode": "join",
    }


# ---------------------------------------------------------------------------
# Prompt reconstruction for deletion
# ---------------------------------------------------------------------------

def _rebuild_prompt_join(tokens, static_suffix, sep="\n\n"):
    q = " ".join(tokens).strip()
    s = (static_suffix or "").strip()
    return f"{q}{sep}{s}" if s else q


def _rebuild_prompt_offsets(full_prompt, question_text, static_suffix, token_offsets, del_idx, sep="\n\n"):
    q0, q1 = _find_substring_span(full_prompt, question_text)
    spans = []
    for i, (a, b) in enumerate(token_offsets):
        if i not in del_idx:
            continue
        if b <= q0 or a >= q1:
            continue
        spans.append((max(a, q0), min(b, q1)))

    if not spans:
        q_new = question_text
    else:
        spans.sort()
        merged = []
        for a, b in spans:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        q_new = full_prompt[q0:q1]
        for a, b in reversed(merged):
            ra, rb = a - q0, b - q0
            q_new = q_new[:ra] + q_new[rb:]
        q_new = q_new.strip()

    s = (static_suffix or "").strip()
    return f"{q_new}{sep}{s}" if s else q_new


def rebuild_prompt(record, del_idx_set, mask_token="", sep="\n\n"):
    tokens0 = record["tokens"]

    if record.get("rebuild_mode") == "offsets" and "token_offsets" in record:
        return _rebuild_prompt_offsets(
            record["prompt"], record["question_text"],
            record.get("static_suffix", ""), record["token_offsets"],
            del_idx_set, sep,
        )

    suffix = record.get("static_suffix", "")
    if mask_token == "":
        toks = [t for i, t in enumerate(tokens0) if i not in del_idx_set]
    else:
        toks = [mask_token if i in del_idx_set else t for i, t in enumerate(tokens0)]
    return _rebuild_prompt_join(toks, suffix, sep)


# ---------------------------------------------------------------------------
# Faithfulness: deletion curve → AUC
# ---------------------------------------------------------------------------

def deletion_curve(record, wrapper, rank_mode="absolute", mask_token="", sep="\n\n"):
    tokens0 = list(record["tokens"])
    phi = np.asarray(record["phi_vector"], dtype=float)
    correct_label = record["correct_label"]
    n = len(tokens0)

    if n != len(phi):
        raise ValueError(f"tokens ({n}) and phi_vector ({len(phi)}) length mismatch")

    k_list = sorted(set([0, 1, 2, 3, 5, 8, 10, min(15, n), min(20, n), n]))

    if rank_mode == "absolute":
        order = np.argsort(-np.abs(phi))
    elif rank_mode == "positive":
        idx_pool = np.where(phi > 0)[0]
        if len(idx_pool) == 0:
            order = np.argsort(-np.abs(phi))
        else:
            order = idx_pool[np.argsort(-phi[idx_pool])]
    else:
        raise ValueError("rank_mode must be 'absolute' or 'positive'")

    # Baseline
    baseline_prompt = rebuild_prompt(record, set(), mask_token, sep)
    wrapper.generate(prompt=baseline_prompt)
    p0 = float(wrapper.last_option_probs[correct_label])
    pred0 = wrapper.last_answer

    pks, preds = [], []

    for k in k_list:
        if k == 0:
            pks.append(p0)
            preds.append(pred0)
            continue

        del_idx = set(order[:k])
        prompt_k = rebuild_prompt(record, del_idx, mask_token, sep)
        wrapper.generate(prompt=prompt_k)
        pks.append(float(wrapper.last_option_probs[correct_label]))
        preds.append(wrapper.last_answer)

    xs = [k / n for k in k_list]
    auc_del = float(np.trapezoid(pks, xs))

    return {
        "row_id": record["row_id"],
        "baseline_p_correct": p0,
        "drop_at_last_k": float(p0 - pks[-1]),
        "auc_del": auc_del,
        "flip_at_last_k": preds[-1] != pred0,
    }


# ---------------------------------------------------------------------------
# Stability: overlap@k across paraphrases
# ---------------------------------------------------------------------------

def topk_tokens(record, k=10, rank_mode="absolute"):
    toks = list(record["tokens"])
    phi = np.asarray(record["phi_vector"], dtype=float)

    if rank_mode == "absolute":
        order = np.argsort(-np.abs(phi))
    elif rank_mode == "positive":
        idx_pool = np.where(phi > 0)[0]
        order = idx_pool[np.argsort(-phi[idx_pool])] if len(idx_pool) > 0 else np.argsort(-np.abs(phi))
    else:
        raise ValueError("rank_mode must be 'absolute' or 'positive'")

    k = min(k, len(order))
    top = {toks[i].strip().lower() for i in order[:k] if isinstance(toks[i], str) and toks[i].strip()}
    return top


def overlap_at_k(set_a, set_b, k):
    if k <= 0:
        return np.nan
    return len(set_a & set_b) / float(k)


# ---------------------------------------------------------------------------
# XAI method runners - produce a standardized record from a prompt
# ---------------------------------------------------------------------------

def run_lime(wrapper, prompt, correct_label, row_id, params):
    from medical_llm_toolkit.explainers.lime import MedicalLIME

    lime = MedicalLIME(
        wrapper,
        n_samples=params["n_samples"],
        kernel_width=params["kernel_width"],
        mask_token=params.get("mask_token", ""),
        verbose=False,
    )
    result = lime.analyze(prompt, target_class=correct_label, visualize=False)
    return build_record_lime(result, prompt, row_id, correct_label)


def run_ig(wrapper, prompt, correct_label, row_id, params):
    from medical_llm_toolkit.explainers.integrated_gradients import (
        MedicalIntegratedGradients,
        explain_medical_prediction,
    )

    result = explain_medical_prediction(
        wrapper=wrapper,
        prompt=prompt,
        target_class=correct_label,
        n_steps=params["n_steps"],
        visualize=False,
    )
    return build_record_ig(result, prompt, row_id, correct_label, wrapper)


def run_tokenshap(wrapper, prompt, correct_label, row_id, params):
    from medical_llm_toolkit.explainers.tokenshap.token_shap.token_shap import StringSplitter
    from medical_llm_toolkit.explainers.tokenshap.token_shap.base import TfidfTextVectorizer
    from medical_llm_toolkit.explainers.tokenshap.extensions.qa_tokenshap import QATokenSHAP
    from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions.correctness_value import CorrectnessValueFunction

    prev_mode = wrapper.mode
    wrapper.set_mode("answer_only")

    try:
        splitter = StringSplitter()

        if params.get("vectorizer") == "correctness":
            vec = CorrectnessValueFunction(
                correct_label=correct_label,
                mode=params.get("correctness_mode", "prob"),
            )
        else:
            vec = TfidfTextVectorizer()

        ts = QATokenSHAP(model=wrapper, splitter=splitter, vectorizer=vec, debug=False)
        ts.analyze(
            prompt,
            sampling_ratio=params["sampling_ratio"],
            max_combinations=params["max_combinations"],
            print_highlight_text=False,
        )
        return build_record_tokenshap(ts, prompt, row_id, correct_label)
    finally:
        wrapper.set_mode(prev_mode)


# Method dispatcher
METHOD_RUNNERS = {
    "lime": run_lime,
    "ig": run_ig,
    "tokenshap_default": run_tokenshap,
    "tokenshap_correctness": run_tokenshap,
}


# ---------------------------------------------------------------------------
# Main evaluation loops
# ---------------------------------------------------------------------------

def run_faithfulness(
    wrapper,
    df,
    method_name,
    method_params,
    task_type,
    rank_mode="absolute",
):
    runner = METHOD_RUNNERS[method_name]
    wrapper.set_task(task_type)
    wrapper.set_mode("answer_only")

    label_col = "answer_label_AB" if task_type == "yn" else "answer_label"
    results = []

    for i, row in df.iterrows():
        print(f"  [{method_name}] Q{i}/{len(df)}", end="\r")
        prompt = row["prompt_text"]
        correct_label = row[label_col]

        t0 = time.time()
        record = runner(wrapper, prompt, correct_label, i, method_params)
        explain_time = time.time() - t0

        t1 = time.time()
        curve = deletion_curve(record, wrapper, rank_mode=rank_mode)
        deletion_time = time.time() - t1

        curve["method"] = method_name
        curve["task_type"] = task_type
        curve["explain_time_s"] = round(explain_time, 2)
        curve["deletion_time_s"] = round(deletion_time, 2)
        results.append(curve)

    print()
    return pd.DataFrame(results)


def run_stability(
    wrapper,
    df,
    method_name,
    method_params,
    task_type,
    k=10,
    rank_mode="absolute",
):
    runner = METHOD_RUNNERS[method_name]
    wrapper.set_task(task_type)
    wrapper.set_mode("answer_only")

    label_col = "answer_label_AB" if task_type == "yn" else "answer_label"

    # Check if paraphrase columns exist
    para_cols = [c for c in df.columns if c.startswith("prompt_para_")]
    if not para_cols:
        print(f"  No paraphrase columns found in dataset. Skipping stability for {method_name}.")
        return pd.DataFrame()

    results = []

    for i, row in df.iterrows():
        print(f"  [{method_name}] Q{i}/{len(df)}", end="\r")
        prompt0 = row["prompt_text"]
        correct_label = row[label_col]

        # Prediction parity filter
        wrapper.generate(prompt=prompt0)
        pred0 = wrapper.last_answer

        paraphrase_preds = {}
        for pc in para_cols:
            wrapper.generate(prompt=row[pc])
            paraphrase_preds[pc] = wrapper.last_answer

        kept = {pc: (pred == pred0) for pc, pred in paraphrase_preds.items()}

        if not any(kept.values()):
            results.append({
                "row_id": i,
                "method": method_name,
                "task_type": task_type,
                "pred_original": pred0,
                "n_paraphrases": len(para_cols),
                "n_retained": 0,
                "mean_overlap": np.nan,
            })
            continue

        # Run on original
        record0 = runner(wrapper, prompt0, correct_label, i, method_params)
        top0 = topk_tokens(record0, k=k, rank_mode=rank_mode)

        overlaps = []
        for pc in para_cols:
            if not kept[pc]:
                continue
            rec_p = runner(wrapper, row[pc], correct_label, i, method_params)
            top_p = topk_tokens(rec_p, k=k, rank_mode=rank_mode)
            overlaps.append(overlap_at_k(top0, top_p, k))

        results.append({
            "row_id": i,
            "method": method_name,
            "task_type": task_type,
            "pred_original": pred0,
            "n_paraphrases": len(para_cols),
            "n_retained": sum(kept.values()),
            "mean_overlap": float(np.mean(overlaps)) if overlaps else np.nan,
        })

    print()
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified XAI Evaluation")
    p.add_argument("--models", nargs="+", default=["Apollo-2B"],
                   choices=list(MODELS.keys()), help="Models to evaluate")
    p.add_argument("--methods", nargs="+", default=["lime", "ig", "tokenshap_default", "tokenshap_correctness"],
                   choices=list(METHOD_RUNNERS.keys()), help="XAI methods to evaluate")
    p.add_argument("--tasks", nargs="+", default=["yn", "mcq"],
                   choices=["yn", "mcq"], help="Task types to evaluate")
    p.add_argument("--eval", nargs="+", default=["faithfulness"],
                   choices=["faithfulness", "stability"], help="Evaluation types")
    p.add_argument("--rank-mode", default="absolute", choices=["absolute", "positive"])
    p.add_argument("--stability-k", type=int, default=10)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Directory containing parquet files. Defaults to repo root.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for CSVs. Defaults to eval/results/")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_dataset(data_dir, task_type, eval_type):
    """Load the appropriate parquet file for the given task and eval type."""
    data_dir = Path(data_dir)

    if eval_type == "stability":
        # Stability needs paraphrased data
        fname = f"{task_type}_paraphrased.parquet"
    else:
        fname = f"{task_type}_sample.parquet"

    fpath = data_dir / fname
    if not fpath.exists():
        print(f"  WARNING: {fpath} not found, skipping.")
        return None

    df = pd.read_parquet(fpath).reset_index()

    # Ensure answer_label_AB column exists for yn tasks
    if task_type == "yn" and "answer_label_AB" not in df.columns:
        df["answer_label_AB"] = df["answer_label"].map({"yes": "A", "no": "B"})

    return df


def main():
    args = parse_args()

    # Resolve directories
    repo_root = Path(__file__).resolve().parent.parent.parent  # 298 Proj/
    data_dir = Path(args.data_dir) if args.data_dir else repo_root
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name} ({model_id})")
        print(f"{'='*60}")

        wrapper = load_medical_llm(
            model_id,
            device=args.device,
            torch_dtype=torch.float16,
            token=args.hf_token,
        )

        for task_type in args.tasks:
            for eval_type in args.eval:
                print(f"\n--- {eval_type} | {task_type} | {model_name} ---")

                df = load_dataset(data_dir, task_type, eval_type)
                if df is None:
                    continue

                for method_name in args.methods:
                    params = METHOD_DEFAULTS[method_name].copy()
                    print(f"\n  Method: {method_name} | Params: {params}")

                    if eval_type == "faithfulness":
                        result_df = run_faithfulness(
                            wrapper, df, method_name, params,
                            task_type, rank_mode=args.rank_mode,
                        )
                    elif eval_type == "stability":
                        result_df = run_stability(
                            wrapper, df, method_name, params,
                            task_type, k=args.stability_k,
                            rank_mode=args.rank_mode,
                        )

                    if result_df.empty:
                        continue

                    # Add model info
                    result_df["model"] = model_name

                    # Save
                    fname = f"{eval_type}_{task_type}_{method_name}_{model_name}_{timestamp}.csv"
                    out_path = output_dir / fname
                    result_df.to_csv(out_path, index=False)
                    print(f"  Saved: {out_path}")

                    # Print summary
                    if eval_type == "faithfulness":
                        print(f"  Mean AUC: {result_df['auc_del'].mean():.4f}")
                        print(f"  Mean baseline P(correct): {result_df['baseline_p_correct'].mean():.4f}")
                    elif eval_type == "stability":
                        valid = result_df.dropna(subset=["mean_overlap"])
                        retained = result_df["n_retained"].sum()
                        total = result_df["n_paraphrases"].sum()
                        print(f"  Retention rate: {retained}/{total} = {retained/total:.4f}" if total > 0 else "  No pairs")
                        print(f"  Mean overlap@{args.stability_k}: {valid['mean_overlap'].mean():.4f}" if len(valid) > 0 else "  No valid overlaps")

        # Clean up model
        del wrapper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
