"""
Run:
    python ELI5_updated.py

Before running, edit the CONFIG section below dataset path, model id, device, sample size if needed

Requirements:
    pip install -U pandas pyarrow numpy scikit-learn eli5 matplotlib

Also required in the same folder (or on PYTHONPATH):
    medical_llm_wrapper_fixed.py   (defines MedicalLLMWrapper)

Hugging Face token:
    - set an env var named HF_TOKEN (recommended):
        export HF_TOKEN="hf_..."
"""

import os, re, time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import eli5

SEED = 42
rng = np.random.default_rng(SEED)


#  CONFIG

DATA_PATH = "compiled_df.parquet"  
MODEL_ID = "BioMistral/BioMistral-7B" 
DEVICE = "cuda" 
N_SAMPLES = 1000  

USE_CONFIDENCE_GATE = True  
CONF_THRESHOLD = 0.65

TRAIN_SURROGATE = True 
SURROGATE_TOP = 30 
EXPLAIN_ERRORS = 5  

SEED = 42
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load data 
def load_compiled_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "question_type" in df.columns:
        qt = df["question_type"].astype(str).str.strip().str.lower()
        qt = qt.replace({"yn":"y/n", "yes/no":"y/n", "y\n":"y/n"})
        df["question_type_norm"] = qt

        print("\nquestion_type counts:")
        print(df["question_type_norm"].value_counts(dropna=False))
    else:
        df["question_type_norm"] = ""

    return df


def split_mcq_yn(df: pd.DataFrame):
    mcq = df[df["question_type_norm"].isin(["mcq","multiple choice","multiple-choice","mcq "])].copy()
    yn  = df[df["question_type_norm"].isin(["y/n","yes/no","yn"])].copy()

    if len(mcq)==0 and "options" in df.columns:
        mcq = df[df["options"].notna()].copy()
    if len(yn)==0:
        yn = df[df["answer_label"].astype(str).str.lower().isin(["yes","no"])].copy()

    print("MCQ rows:", len(mcq))
    print("YN  rows:", len(yn))

    # Clean label formats
    if len(mcq):
        mcq["answer_label"] = mcq["answer_label"].astype(str).str.strip().str.upper()
        mcq = mcq[mcq["answer_label"].isin(["A","B","C","D"])].copy()

    if len(yn):
        yn["answer_label"] = yn["answer_label"].astype(str).str.strip().str.lower()
        yn = yn[yn["answer_label"].isin(["yes","no"])].copy()

    if len(mcq):
        print("MCQ label dist:", Counter(mcq["answer_label"]))
    return mcq, yn


# Prompt builder
def parse_options_any_format(options_val):
    if options_val is None or (isinstance(options_val, float) and np.isnan(options_val)):
        return None

    if isinstance(options_val, (list, tuple)) and len(options_val) >= 4:
        return [str(options_val[i]).strip() for i in range(4)]

    if isinstance(options_val, dict):
        keys = [k.upper() for k in options_val.keys() if isinstance(k, str)]
        if all(k in keys for k in ["A","B","C","D"]):
            return [str(options_val["A"]).strip(), str(options_val["B"]).strip(),
                    str(options_val["C"]).strip(), str(options_val["D"]).strip()]
        try:
            return [str(options_val[i]).strip() for i in range(4)]
        except Exception:
            return None

    s = str(options_val).replace("\r", "\n")

    # Already A./B./C./D. formatted
    if re.search(r"\bA[\).:]\s", s, re.IGNORECASE) and re.search(r"\bB[\).:]\s", s, re.IGNORECASE):
        patt = r"(?:^|\n)\s*([ABCD])[\).:]\s*(.+?)(?=(?:\n\s*[ABCD][\).:]\s)|\Z)"
        chunks = re.findall(patt, s, flags=re.IGNORECASE | re.DOTALL)
        if chunks:
            d = {k.upper(): re.sub(r"\s+", " ", v).strip() for k, v in chunks}
            if all(k in d for k in ["A","B","C","D"]):
                return [d["A"], d["B"], d["C"], d["D"]]

    m = re.findall(r"'([^']+)'", s)
    if len(m) >= 4:
        return [m[0].strip(), m[1].strip(), m[2].strip(), m[3].strip()]
    m2 = re.findall(r"\"([^\"]+)\"", s)
    if len(m2) >= 4:
        return [m2[0].strip(), m2[1].strip(), m2[2].strip(), m2[3].strip()]

    return None


def render_mcq_prompt(question: str, options_val, prompt_text=None) -> str:
    # Use prompt_text only if it already contains explicit options
    if prompt_text is not None and str(prompt_text).strip():
        base = str(prompt_text).strip()
        if re.search(r"\bA[\).:]\s", base) and re.search(r"\bB[\).:]\s", base):
            return base.rstrip() + "\n\nReturn exactly one letter: A, B, C, or D."

    opts = parse_options_any_format(options_val)
    if opts is None:
        return f"Question:\n{question}\n\nOptions:\n{str(options_val)}\n\nReturn exactly one letter: A, B, C, or D."

    a,b,c,d = opts
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Answer Choices:\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n\n"
        f"Return exactly one letter: A, B, C, or D."
    )

def parse_answer_letter_strict(text: str):
    if text is None:
        return None
    t = str(text)

    m = re.search(r"Final:\s*([ABCD])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"Answer:\s*([ABCD])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    tail = t[-200:]
    m = re.search(r"\b([ABCD])\b", tail, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def reasoning_suffix_mcq():
    return (
        "\n\nGive a brief medical rationale (1-4 sentences). "
        "Then on a new line write exactly: Final: <A/B/C/D>"
    )


# Inference + plots 
def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def mcq_two_stage_predict(llm, prompt: str, use_confidence_gate=True, conf_threshold=0.65):
    raw_free, t_free = timed_call(
        llm.generate_free,
        prompt + reasoning_suffix_mcq()
    )
    pred_free = parse_answer_letter_strict(raw_free)

    raw_scored = pred_scored = None
    t_scored = None
    conf = None
    option_probs = None

    if use_confidence_gate:
        try: llm.set_task("mcq")
        except Exception: pass
        try: llm.set_mode("answer_only")
        except Exception: pass

        raw_scored, t_scored = timed_call(llm.generate, prompt)
        pred_scored = parse_answer_letter_strict(raw_scored)
        conf = getattr(llm, "last_confidence", None)
        option_probs = getattr(llm, "last_option_probs", None)

        if conf is not None and pred_scored in {"A","B","C","D"} and float(conf) >= conf_threshold:
            final_pred = pred_scored
            strategy = "scored_highconf"
        else:
            final_pred = pred_free if pred_free in {"A","B","C","D"} else pred_scored
            strategy = "free_or_lowconf"

        return {
            "pred": final_pred,
            "raw": raw_scored if strategy == "scored_highconf" else raw_free,
            "raw_free": raw_free,
            "pred_free": pred_free,
            "raw_scored": raw_scored,
            "pred_scored": pred_scored,
            "confidence": float(conf) if conf is not None else None,
            "option_probs": option_probs,
            "t_free_s": float(t_free),
            "t_scored_s": float(t_scored),
            "latency_s": float(t_free) + float(t_scored),
            "strategy": strategy,
        }

    return {
        "pred": pred_free,
        "raw": raw_free,
        "raw_free": raw_free,
        "pred_free": pred_free,
        "raw_scored": None,
        "pred_scored": None,
        "confidence": None,
        "option_probs": None,
        "t_free_s": float(t_free),
        "t_scored_s": None,
        "latency_s": float(t_free),
        "strategy": "free_only",
    }


def run_mcq_eval(df_mcq, llm, n=1000, use_confidence_gate=True, conf_threshold=0.65):
    sample = df_mcq.sample(n=min(n, len(df_mcq)), random_state=SEED) if n else df_mcq
    rows = []
    for _, r in sample.iterrows():
        prompt = render_mcq_prompt(str(r["question"]), r.get("options", None), r.get("prompt_text", None))
        out = mcq_two_stage_predict(llm, prompt, use_confidence_gate=use_confidence_gate, conf_threshold=conf_threshold)
        rows.append({
            "question": str(r["question"]),
            "options": r.get("options", None),
            "gold": str(r["answer_label"]).strip().upper(),
            **out,
        })
    return rows

def plot_confusion(y_true, y_pred, labels, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def plot_latency(rows, title="Latency (seconds)"):
    lat = [r.get("latency_s") for r in rows if r.get("latency_s") is not None]
    if not lat:
        print("No latency_s found.")
        return
    lat = np.array(lat, dtype=float)
    p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
    plt.figure(figsize=(6, 4))
    plt.hist(lat, bins=30)
    plt.title(f"{title} | p50={p50:.3f}s p95={p95:.3f}s mean={lat.mean():.3f}s")
    plt.xlabel("seconds")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

def plot_confidence_accuracy(rows, title="Confidence vs accuracy", n_bins=10):
    conf, corr = [], []
    for r in rows:
        c = r.get("confidence", None)
        p, g = r.get("pred"), r.get("gold")
        if c is None:
            continue
        if p not in {"A","B","C","D"} or g not in {"A","B","C","D"}:
            continue
        conf.append(float(c))
        corr.append(1.0 if p == g else 0.0)

    if not conf:
        print("No usable confidence values.")
        return

    conf = np.array(conf)
    corr = np.array(corr)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1

    xs, ys, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        xs.append((bins[b] + bins[b+1]) / 2)
        ys.append(corr[mask].mean())
        counts.append(mask.sum())

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("confidence bin center")
    plt.ylabel("accuracy in bin")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.bar(range(len(counts)), counts)
    plt.title(title + " (bin counts)")
    plt.xlabel("bin index (low→high confidence)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def evaluate_mcq_rows(mcq_rows):
    valid = [r for r in mcq_rows if r.get("pred") in {"A","B","C","D"} and r.get("gold") in {"A","B","C","D"}]
    y_true = [r["gold"] for r in valid]
    y_pred = [r["pred"] for r in valid]

    print("Pred distribution:", Counter(y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred), f"(n={len(valid)})")
    print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=["A","B","C","D"], zero_division=0))

    plot_confusion(y_true, y_pred, ["A","B","C","D"], "MCQ confusion matrix")
    plot_latency(valid, "MCQ latency")
    plot_confidence_accuracy(valid, "MCQ confidence vs accuracy")


# Surrogate + ELI5 
from IPython.display import display

STOP_PHRASES = [
    "Return exactly one letter", "Answer Choices", "Answer:", "Rationale:", "Final:",
    "Give a brief medical rationale", "Then on a new line write exactly"
]
def clean_for_surrogate(text: str) -> str:
    t = str(text or "")
    for p in STOP_PHRASES:
        t = t.replace(p, " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_surrogate_text_from_row(r):
    q = r["question"]
    opts = parse_options_any_format(r.get("options", None))
    if opts is None:
        return clean_for_surrogate(q + "\n" + str(r.get("options","")))
    a,b,c,d = opts
    return clean_for_surrogate(f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}")

def train_surrogate_mimic_llm(mcq_rows, top=25):
    X, y = [], []
    for r in mcq_rows:
        if r.get("pred") in {"A","B","C","D"}:
            X.append(build_surrogate_text_from_row(r))
            y.append(r["pred"])

    if len(set(y)) < 2:
        print("Not enough class variety to train surrogate.")
        return None

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95),
        LogisticRegression(max_iter=3000)
    )
    pipe.fit(Xtr, ytr)
    fidelity = pipe.score(Xte, yte)
    print(f"[Surrogate] Fidelity to LLM predictions: {fidelity:.3f} (n_test={len(yte)})")

    print("\n[Surrogate] Global weights:")
    display(eli5.show_weights(pipe, top=top))
    return pipe

def show_local_eli5(pipe, text, top=20):
    doc = clean_for_surrogate(text)
    display(eli5.show_prediction(pipe, doc, top=top))

def explain_some_errors(pipe, mcq_rows, n=3):
    shown = 0
    for r in mcq_rows:
        if r.get("pred") in {"A","B","C","D"} and r.get("gold") in {"A","B","C","D"} and r["pred"] != r["gold"]:
            print("\n--- ERROR CASE ---")
            print("Gold:", r["gold"], "| Pred:", r["pred"], "| Strategy:", r.get("strategy"))
            txt = build_surrogate_text_from_row(r)
            show_local_eli5(pipe, txt, top=20)
            shown += 1
            if shown >= n:
                break


# Main run 
def main():
    np.random.seed(SEED)

    if HF_TOKEN is None or not str(HF_TOKEN).strip():
        print("[Warning] HF_TOKEN env var not set. If your model is gated/private, set HF_TOKEN.")

    print("\n== Load data ==")
    df = load_compiled_df(DATA_PATH)
    mcq, yn = split_mcq_yn(df)

    if len(mcq) == 0:
        raise SystemExit("No MCQ rows found. Check DATA_PATH and dataset columns.")

    print("\n== Load medical wrapper ==")
    from medical_llm_wrapper_fixed import MedicalLLMWrapper

    llm = MedicalLLMWrapper(
        model_name=MODEL_ID,
        device=DEVICE,
        token=HF_TOKEN,
        torch_dtype=None,
    )

    # For accuracy evaluation: answer_only (fast)
    try:
        llm.set_mode("answer_only")
    except Exception:
        pass

    print("\n== Run evaluation ==")
    global mcq_rows 
    mcq_rows = run_mcq_eval(mcq, llm, n=N_SAMPLES, use_confidence_gate=USE_CONFIDENCE_GATE, conf_threshold=CONF_THRESHOLD)

    print("\n==============================")
    print("Base accuracy (FREE reasoning, no constraints)")
    base_free = [r for r in mcq_rows if r.get("pred_free") in {"A","B","C","D"} and r.get("gold") in {"A","B","C","D"}]
    if base_free:
        print("Accuracy:", accuracy_score([r["gold"] for r in base_free], [r["pred_free"] for r in base_free]), f"(n={len(base_free)})")
    else:
        print("No valid free predictions found.")

    print("\n==============================")
    print("Base accuracy (SCORED constrained answer_only, no confidence gate)")
    base_scored = [r for r in mcq_rows if r.get("pred_scored") in {"A","B","C","D"} and r.get("gold") in {"A","B","C","D"}]
    if base_scored:
        print("Accuracy:", accuracy_score([r["gold"] for r in base_scored], [r["pred_scored"] for r in base_scored]), f"(n={len(base_scored)})")
    else:
        print("No valid scored predictions found.")

    print("\n==============================")
    print("Final accuracy (selected strategy)")
    evaluate_mcq_rows(mcq_rows)

    if TRAIN_SURROGATE:
        print("\n== Train surrogate for explanations ==")
        mcq_pipe = train_surrogate_mimic_llm(mcq_rows, top=SURROGATE_TOP)
        if mcq_pipe is not None:
            if EXPLAIN_ERRORS > 0:
                explain_some_errors(mcq_pipe, mcq_rows, n=EXPLAIN_ERRORS)
        else:
            print("Skipping ELI5 explanations (surrogate not trained).")

    print("\nDone.")


if __name__ == "__main__":
    main()
