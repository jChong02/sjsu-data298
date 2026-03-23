
"""
Run:
    python ELI5_adapter.py

Before running, edit the CONFIG section below if needed

Requirements (example):
    pip install -U pandas pyarrow numpy scikit-learn eli5 matplotlib huggingface_hub

Also required:
    medical_llm_wrapper_fixed.py   (defines MedicalLLMWrapper)
    compiled_df.parquet
"""

import os, re, time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import eli5
from huggingface_hub import login
from medical_llm_wrapper_fixed import MedicalLLMWrapper

try:
    from IPython.display import display 
except Exception: 
    def display(x):
        print(x)


# CONFIG
SEED = 42
rng = np.random.default_rng(SEED)
DATA_PATH = "/data/compiled_df.parquet"
HF_TOKEN = os.environ.get('HF_TOKEN')
MODEL_ID = "BioMistral/BioMistral-7B"
DEVICE = "cuda"


# helper functions
def extract_abcd_from_options(options_text: str):
    if options_text is None:
        return None
    s = str(options_text)

    matches = re.findall(r"(?is)\b([ABCD])\s*[\.\)]\s*(.*?)(?=\n\s*[ABCD]\s*[\.\)]|\Z)", s)
    if not matches or len(matches) < 4:
        return None
    d = {k.upper(): v.strip() for k, v in matches}
    if all(k in d for k in ["A","B","C","D"]):
        return d
    return None

def render_mcq_prompt(question: str, options_text: str):
    opts = extract_abcd_from_options(options_text)
    if not opts:
        return (
            "You are a careful medical question-answering assistant.\n"
            "Choose the single best option.\n\n"
            f"Question: {question}\n"
            f"{options_text}\n"
            "Answer: "
        )
    return (
        "You are a careful medical question-answering assistant.\n"
        "Choose the single best option.\n\n"
        f"Question: {question}\n"
        "Answer Choices:\n"
        f"A. {opts['A']}\n"
        f"B. {opts['B']}\n"
        f"C. {opts['C']}\n"
        f"D. {opts['D']}\n"
        "Answer: "
    )

def render_yn_prompt(question: str):
    return (
        "You are a careful medical question-answering assistant.\n"
        "Answer Yes or No.\n"
        "Use A for Yes and B for No.\n\n"
        f"Question: {question}\n"
        "Answer: "
    )

def parse_mcq_answer_strict(text: str):
    if text is None:
        return None
    t = str(text).strip()

    m = re.search(r"(?im)^\s*Final:\s*([ABCD])\s*$", t)
    if m: return m.group(1).upper()

    m = re.search(r"(?im)^\s*Answer:\s*([ABCD])\s*$", t)
    if m: return m.group(1).upper()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.fullmatch(r"[ABCD]", ln, flags=re.IGNORECASE):
            return ln.upper()
    return None

def parse_yn_answer_strict(text: str):
    if text is None:
        return None
    t = str(text).strip()

    m = re.search(r"(?im)^\s*Final:\s*([AB])\s*$", t)
    if m: return m.group(1).upper()

    m = re.search(r"(?im)^\s*Answer:\s*([AB])\s*$", t)
    if m: return m.group(1).upper()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.fullmatch(r"[AB]", ln, flags=re.IGNORECASE):
            return ln.upper()

    return None

def normalize_gold_yn(val):
    g = str(val).strip().lower()
    if g in {"yes", "y", "true", "t", "1", "a"}:
        return "A"  # Yes
    if g in {"no", "n", "false", "f", "0", "b"}:
        return "B"  # No
    return None

def reasoning_suffix_mcq():
    return (
        "\n\nExplain briefly (1-3 sentences).\n"
        "Then output the final answer on the last line ONLY as:\n"
        "Final: A\n"
        "or Final: B\n"
        "or Final: C\n"
        "or Final: D\n"
        "Do not write anything after the Final line."
    )

def reasoning_suffix_yn():
    return (
        "\n\nExplain briefly (1-3 sentences).\n"
        "Then output the final answer on the last line ONLY as:\n"
        "Final: A   (A = Yes)\n"
        "or Final: B   (B = No)\n"
        "Do not write anything after the Final line."
    )

def eval_mcq_answer_only(df_mcq: pd.DataFrame, n=1000, seed=7):
    df_s = df_mcq.sample(n=min(n, len(df_mcq)), random_state=seed).reset_index(drop=True)
    golds, preds = [], []

    for r in df_s.to_dict("records"):
        gold = str(r["answer_label"]).strip().upper()
        if gold not in {"A","B","C","D"}:
            continue

        prompt = render_mcq_prompt(r["question"], r["options"])
        llm.set_task("mcq"); llm.set_mode("answer_only")
        out = llm.generate(prompt)
        pred = parse_mcq_answer_strict(out)

        if pred not in {"A","B","C","D"}:
            continue

        golds.append(gold); preds.append(pred)

    acc = accuracy_score(golds, preds) if golds else 0.0
    print(f"MCQ answer_only accuracy: {acc:.4f} (n={len(golds)}/{len(df_s)})")
    print(classification_report(golds, preds, labels=["A","B","C","D"]))
    return acc

def eval_yn_answer_only(df_yn: pd.DataFrame, n=600, seed=7):
    df_s = df_yn.sample(n=min(n, len(df_yn)), random_state=seed).reset_index(drop=True)
    golds, preds = [], []

    for r in df_s.to_dict("records"):
        gold = normalize_gold_yn(r["answer_label"])
        if gold not in {"A","B"}:
            continue

        prompt = render_yn_prompt(r["question"])
        llm.set_task("yn"); llm.set_mode("answer_only")
        out = llm.generate(prompt)
        pred = parse_yn_answer_strict(out)

        if pred not in {"A","B"}:
            continue

        golds.append(gold); preds.append(pred)

    acc = accuracy_score(golds, preds) if golds else 0.0
    print(f"Y/N answer_only accuracy: {acc:.4f} (n={len(golds)}/{len(df_s)})")
    print(classification_report(golds, preds, labels=["A","B"]))

    return acc

def clean_text(t: str):
    t = str(t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def mcq_surrogate_text(question: str, options_text: str):
    opts = extract_abcd_from_options(options_text)
    if not opts:
        return clean_text(question + " " + str(options_text))
    return clean_text(
        f"{question} "
        f"A {opts['A']} "
        f"B {opts['B']} "
        f"C {opts['C']} "
        f"D {opts['D']}"
    )

def yn_surrogate_text(question: str):
    return clean_text(question + " (A=Yes, B=No)")

def train_surrogate_return_parts(X_texts, y, title="", top=30, seed=7):
    if len(set(y)) < 2:
        print(f"[{title}] Not enough class variety.")
        return None, None, None

    Xtr, Xte, ytr, yte = train_test_split(
        X_texts, y, test_size=0.2, random_state=seed, stratify=y
    )

    tf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    clf = LogisticRegression(max_iter=4000)

    Xtr_t = tf.fit_transform(Xtr)
    clf.fit(Xtr_t, ytr)

    score = clf.score(tf.transform(Xte), yte)
    print(f"[{title}] heldout score: {score:.3f} (n_test={len(yte)})")

    pipe = make_pipeline(tf, clf)
    feat_names = tf.get_feature_names_out()
    try:
        display(eli5.show_weights(clf, top=top, feature_names=feat_names))
    except Exception as e:
        # fallback
        print("[eli5.show_weights] error:", e)

    return pipe, tf, clf

def eli5_explain_case_parts(vectorizer, classifier, text, top=20, display_html=True):
    X = vectorizer.transform([text])
    x_dense = X.toarray()[0]

    feature_names = vectorizer.get_feature_names_out()

    try:
        disp = eli5.show_prediction(classifier, x_dense, feature_names=feature_names, top=top)
        if display_html:
            display(disp)
        else:
            return disp
    except Exception as e:
        print("[eli5.show_prediction] error:", e)
        print("Falling back to top weighted features from classifier.coef_")
        coefs = classifier.coef_
        if coefs.ndim == 1 or coefs.shape[0] == 1:
            coef = coefs.ravel()
        else:
            coef = coefs.max(axis=0)
        top_idx_pos = np.argsort(-coef)[:top]
        top_idx_neg = np.argsort(coef)[:top]
        names = feature_names
        print("\nTop positive features:")
        for i in top_idx_pos:
            print(f"{coef[i]:+.3f}\t{names[i]}")
        print("\nTop negative features:")
        for i in top_idx_neg:
            print(f"{coef[i]:+.3f}\t{names[i]}")

def collect_mcq_for_surrogates(df_mcq, n=1500, seed=7):
    df_s = df_mcq.sample(n=min(n, len(df_mcq)), random_state=seed).reset_index(drop=True)
    rows = []
    for r in df_s.to_dict("records"):
        gold = str(r["answer_label"]).strip().upper()
        if gold not in {"A","B","C","D"}:
            continue

        prompt = render_mcq_prompt(r["question"], r["options"])
        llm.set_task("mcq"); llm.set_mode("answer_only")
        out = llm.generate(prompt)
        pred = parse_mcq_answer_strict(out)
        if pred not in {"A","B","C","D"}:
            continue

        rows.append({
            "text": mcq_surrogate_text(r["question"], r["options"]),
            "gold": gold,
            "pred": pred,
            "wrong": int(pred != gold),
            "question": r["question"],
            "options": r["options"],
        })
    return pd.DataFrame(rows)

def collect_yn_for_surrogates(df_yn, n=1500, seed=7):
    df_s = df_yn.sample(n=min(n, len(df_yn)), random_state=seed).reset_index(drop=True)
    rows = []
    for r in df_s.to_dict("records"):
        gold = normalize_gold_yn(r["answer_label"])
        if gold not in {"A","B"}:
            continue

        prompt = render_yn_prompt(r["question"])
        llm.set_task("yn"); llm.set_mode("answer_only")
        out = llm.generate(prompt)
        pred = parse_yn_answer_strict(out)
        if pred not in {"A","B"}:
            continue

        rows.append({
            "text": yn_surrogate_text(r["question"]),
            "gold": gold,
            "pred": pred,
            "wrong": int(pred != gold),
            "question": r["question"],
        })
    return pd.DataFrame(rows)

def explain_case(bundle, text, top=20):
    if bundle.get("mimic"):
        print("\n[ELI5] mimic_llm (why the model predicts what it predicts):")
        display(eli5.show_prediction(bundle["mimic"], text, top=top))
    if bundle.get("gold"):
        print("\n[ELI5] predict_gold (features associated with correct labels):")
        display(eli5.show_prediction(bundle["gold"], text, top=top))
    if bundle.get("error"):
        print("\n[ELI5] predict_error (features associated with failures):")
        display(eli5.show_prediction(bundle["error"], text, top=top))

def get_tf_clf_from_bundle(bundle, key):
    if bundle is None:
        return None, None

    part = bundle.get(key, None) if isinstance(bundle, dict) else bundle

    # If part is a dict with tf/clf
    if isinstance(part, dict):
        tf = part.get("tf", None)
        clf = part.get("clf", None)
        if tf is not None and clf is not None:
            return tf, clf

        pipe = part.get("pipe", None)
        if pipe is not None and hasattr(pipe, "named_steps"):
            tf = pipe.named_steps.get("tfidfvectorizer") or pipe.named_steps.get("tfidfvectorizer".replace("_",""))
            clf = pipe.named_steps.get("logisticregression") or pipe.named_steps.get("logisticregression".replace("_",""))
            return tf, clf

    # If part is a sklearn Pipeline directly
    if hasattr(part, "named_steps"):
        pipe = part
        tf = pipe.named_steps.get("tfidfvectorizer") or pipe.named_steps.get("tfidfvectorizer".replace("_",""))
        if tf is None:
            first = list(pipe.named_steps.items())[0][1]
            tf = first if hasattr(first, "vocabulary_") or hasattr(first, "get_feature_names_out") else None
        clf = pipe.named_steps.get("logisticregression") or pipe.named_steps.get("logisticregression".replace("_",""))
        if clf is None:
            last = list(pipe.named_steps.items())[-1][1]
            clf = last if hasattr(last, "coef_") else None
        return tf, clf

    return None, None

def show_wrong_cases_with_eli5(bundle, wrong_cases_df, which="gold", top=20):
    tf, clf = get_tf_clf_from_bundle(bundle, which)
    if tf is None or clf is None:
        print(f"[ERROR] Could not extract tf/clf for '{which}' from bundle. Inspect bundle keys:", list(bundle.keys()) if bundle else None)
        return

    for _, r in wrong_cases_df.iterrows():
        print("\n============================")
        print(f"{which.upper()} GOLD: {r.get('gold')} PRED: {r.get('pred')}")
        print("Q:", r.get("question"))
        text = r.get("text") or r.get("question")
        try:
            # eli5_explain_case_parts expects
            eli5_explain_case_parts(tf, clf, text, top=top, display_html=True)
        except Exception as e:
            print("[eli5_explain_case_parts] error:", e)
            # fallback
            try:
                X = tf.transform([text])
                x_dense = X.toarray()[0]
                feature_names = tf.get_feature_names_out()
                if hasattr(clf, "coef_"):
                    coefs = clf.coef_
                    if coefs.ndim == 1 or coefs.shape[0] == 1:
                        coef = coefs.ravel()
                        contrib = coef * x_dense
                        top_idx = np.argsort(-contrib)[:top]
                        print("\nTop contributing features (positive):")
                        for i in top_idx:
                            if x_dense[i] != 0:
                                print(f"{contrib[i]:+.3f}\t{feature_names[i]}")
                        neg_idx = np.argsort(contrib)[:top]
                        print("\nTop contributing features (negative):")
                        for i in neg_idx:
                            if x_dense[i] != 0:
                                print(f"{contrib[i]:+.3f}\t{feature_names[i]}")
                    else:
                        contribs = (coefs * x_dense).max(axis=0)
                        top_idx = np.argsort(-contribs)[:top]
                        print("\nTop features by max-class contribution:")
                        for i in top_idx:
                            print(f"{contribs[i]:+.3f}\t{feature_names[i]}")
                else:
                    print("Classifier has no coef_, cannot compute contributions.")
            except Exception as e2:
                print("Fallback failed:", e2)

def debug_inspect_bundle(bundle):
    print("BUNDLE KEYS:", list(bundle.keys()) if isinstance(bundle, dict) else "bundle not dict")
    for k in (bundle.keys() if isinstance(bundle, dict) else []):
        val = bundle[k]
        print(f"\n--- key: {k} ---")
        print("type:", type(val))
        # small repr
        rep = repr(val)
        print("repr:", rep[:400] + ("..." if len(rep) > 400 else ""))
        # show attributes that might help
        attrs = []
        for a in ("named_steps","steps","get_params","tf","clf","pipe"):
            if hasattr(val, a):
                attrs.append(a)
        if attrs:
            print("has attrs:", attrs)
        else:
            print("no obvious attrs")

def extract_tf_clf_from_obj(obj):
    # None
    if obj is None:
        return None, None

    if isinstance(obj, dict):
        tf = obj.get("tf") or obj.get("vectorizer") or obj.get("tfidf") or obj.get("tfidfvectorizer")
        clf = obj.get("clf") or obj.get("classifier") or obj.get("model") or obj.get("logisticregression")
        if tf is not None and clf is not None:
            return tf, clf
        pipe = obj.get("pipe")
        if pipe is not None:
            obj = pipe  # fall through

    if hasattr(obj, "named_steps"):
        ns = obj.named_steps
        tf = None; clf = None
        for name in ("tfidfvectorizer","tfidf","vectorizer","tfidf_vectorizer","tfidfvectoriser"):
            if name in ns:
                tf = ns[name]
                break
        for name in ("logisticregression","classifier","clf","logreg","logistic"):
            if name in ns:
                clf = ns[name]
                break

        if tf is None:
            try:
                first = list(ns.items())[0][1]
                if hasattr(first, "get_feature_names_out") or hasattr(first, "vocabulary_"):
                    tf = first
            except Exception:
                pass
        if clf is None:
            try:
                last = list(ns.items())[-1][1]
                if hasattr(last, "coef_") or hasattr(last, "decision_function"):
                    clf = last
            except Exception:
                pass
        if tf is not None and clf is not None:
            return tf, clf

    if hasattr(obj, "steps"):
        try:
            steps = obj.steps
            if len(steps) >= 2:
                first = steps[0][1]
                last = steps[-1][1]
                tf = first if (hasattr(first,"get_feature_names_out") or hasattr(first,"vocabulary_")) else None
                clf = last if (hasattr(last,"coef_") or hasattr(last,"decision_function")) else None
                if tf is not None and clf is not None:
                    return tf, clf
        except Exception:
            pass

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            a, b = obj[0], obj[1]
            if hasattr(a, "get_feature_names_out") and (hasattr(b, "coef_") or hasattr(b, "decision_function")):
                return a, b
            if len(obj) >= 3:
                a2, b2, c2 = obj[0], obj[1], obj[2]
                if hasattr(b2, "get_feature_names_out") and (hasattr(c2, "coef_") or hasattr(c2, "decision_function")):
                    return b2, c2
    if hasattr(obj, "get_feature_names_out") and hasattr(obj, "transform"):
        return obj, None
    if hasattr(obj, "coef_") or hasattr(obj, "decision_function"):
        return None, obj

    return None, None

def explain_with_fallback(tf, clf, text, top=20):
    try:
        if tf is None or clf is None:
            raise ValueError("tf or clf is None")
        eli5_explain_case_parts(tf, clf, text, top=top, display_html=True)
        return
    except Exception as e:
        print("[fallback explain] eli5_explain_case_parts failed:", e)
        try:
            X = tf.transform([text])
            x_dense = X.toarray()[0]
            feature_names = tf.get_feature_names_out()
            if hasattr(clf, "coef_"):
                coefs = clf.coef_
                if coefs.ndim == 1 or coefs.shape[0] == 1:
                    coef = coefs.ravel()
                    contrib = coef * x_dense
                    top_idx_pos = np.argsort(-contrib)[:top]
                    top_idx_neg = np.argsort(contrib)[:top]
                    print("\nTop positive contributions:")
                    for i in top_idx_pos:
                        if x_dense[i] != 0:
                            print(f"{contrib[i]:+.3f}\t{feature_names[i]}")
                    print("\nTop negative contributions:")
                    for i in top_idx_neg:
                        if x_dense[i] != 0:
                            print(f"{contrib[i]:+.3f}\t{feature_names[i]}")
                else:
                    # multiclass fallback
                    contribs = (coefs * x_dense).max(axis=0)
                    top_idx = np.argsort(-contribs)[:top]
                    print("\nTop features by max-class contribution:")
                    for i in top_idx:
                        print(f"{contribs[i]:+.3f}\t{feature_names[i]}")
            else:
                print("Classifier has no coef_; cannot compute contributions.")
        except Exception as e2:
            print("Fallback failed:", e2)


# MAIN
def main():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        login()

    df = pd.read_parquet(DATA_PATH)

    print("Rows:", len(df))

    print("Columns:", df.columns.tolist())

    qt = df["question_type"].astype(str).str.strip().str.lower()

    qt = qt.replace({"yn":"y/n", "yes/no":"y/n", "y\n":"y/n"})

    df["question_type_norm"] = qt

    print("\nquestion_type counts:")

    print(df["question_type_norm"].value_counts(dropna=False))

    assert "question" in df.columns

    assert "answer_label" in df.columns

    mcq = df[df["question_type_norm"].isin(["mcq","multiple choice","multiple-choice","mcq "])].copy()

    yn  = df[df["question_type_norm"].isin(["y/n","yes/no","yn"])].copy()

    if len(mcq)==0 and "options" in df.columns:
        mcq = df[df["options"].notna()].copy()

    if len(yn)==0:
        yn = df[df["answer_label"].astype(str).str.lower().isin(["yes","no"])].copy()

    print("MCQ rows:", len(mcq))

    print("YN  rows:", len(yn))

    if len(mcq):
        mcq["answer_label"] = mcq["answer_label"].astype(str).str.strip().str.upper()
        mcq = mcq[mcq["answer_label"].isin(["A","B","C","D"])].copy()

    if len(yn):
        yn["answer_label"] = yn["answer_label"].astype(str).str.strip().str.lower()
        yn = yn[yn["answer_label"].isin(["yes","no"])].copy()

    print("MCQ label dist:", Counter(mcq["answer_label"]) if len(mcq) else {})

    print("YN  label dist:", Counter(yn["answer_label"]) if len(yn) else {})

    ex = mcq.iloc[0]

    print(render_mcq_prompt(ex["question"], ex["options"])[:600])

    llm = MedicalLLMWrapper(
        model_name=MODEL_ID,
        device=DEVICE,
        token=HF_TOKEN,
        torch_dtype=None,
    )

    llm.set_mode("answer_only")

    llm.get_model_info()

    mcq_ans_acc = eval_mcq_answer_only(mcq, n=1000)

    yn_ans_acc  = eval_yn_answer_only(yn, n=600) if len(yn) else None

    mcq_bundle = None

    if len(mcq):
        mcq_sur = collect_mcq_for_surrogates(mcq, n=1500)
        print("MCQ surrogate rows:", len(mcq_sur))

        mcq_bundle = {
            "mimic": train_surrogate_return_parts(mcq_sur["text"], mcq_sur["pred"], title="MCQ mimic_llm", top=30),
            "gold":  train_surrogate_return_parts(mcq_sur["text"], mcq_sur["gold"], title="MCQ predict_gold", top=30),
            "error": train_surrogate_return_parts(mcq_sur["text"], mcq_sur["wrong"], title="MCQ predict_error", top=30),
        }

    yn_bundle = None

    if len(yn):
        yn_sur = collect_yn_for_surrogates(yn, n=1500)
        print("Y/N surrogate rows:", len(yn_sur))

        yn_bundle = {
            "mimic": train_surrogate_return_parts(yn_sur["text"], yn_sur["pred"], title="Y/N mimic_llm", top=30),
            "gold":  train_surrogate_return_parts(yn_sur["text"], yn_sur["gold"], title="Y/N predict_gold", top=30),
            "error": train_surrogate_return_parts(yn_sur["text"], yn_sur["wrong"], title="Y/N predict_error", top=30),
        }

    print("Inspecting mcq_bundle contents:")

    debug_inspect_bundle(mcq_bundle)

    try:
        wrong_df = mcq_sur[mcq_sur["wrong"] == 1].head(5)
    except Exception:
        try:
            wrong_df = wrong_cases
        except Exception:
            wrong_df = None

    if wrong_df is None or len(wrong_df) == 0:
        print("No wrong cases found to explain.")
    else:
        for which in ("mimic","gold","error"):
            print(f"\n\n=== Explanations for '{which}' ===")
            obj = mcq_bundle.get(which) if isinstance(mcq_bundle, dict) else None
            tf, clf = extract_tf_clf_from_obj(obj)
            if tf is None or clf is None:
                print(f"[WARN] could not extract tf/clf automatically for '{which}'. Attempting additional heuristics...")
                # try if mcq_bundle[which] is a Pipeline and pull first/last step
                val = mcq_bundle.get(which)
                try:
                    if hasattr(val, "named_steps"):
                        steps = val.named_steps
                        print("named_steps keys:", list(steps.keys()))
                    if hasattr(val, "steps"):
                        print("steps keys:", [s[0] for s in val.steps])
                except Exception:
                    pass
                # attempt to see if bundle contains tf/clf separately at top-level (unlikely)
                tf_alt = getattr(mcq_bundle, "tf", None) or mcq_bundle.get("tf") if isinstance(mcq_bundle, dict) else None
                clf_alt = getattr(mcq_bundle, "clf", None) or mcq_bundle.get("clf") if isinstance(mcq_bundle, dict) else None
                if tf_alt is not None and clf_alt is not None:
                    tf, clf = tf_alt, clf_alt

            if tf is None or clf is None:
                print(f"[ERROR] still no tf/clf for '{which}'. Repr of mcq_bundle['{which}'] shown above. Falling back to best-effort per-example contributions using any available classifier in bundle.")
                # try to find any classifier in the whole bundle dict
                found_tf, found_clf = None, None
                for k in (mcq_bundle.keys() if isinstance(mcq_bundle, dict) else []):
                    t_tmp, c_tmp = extract_tf_clf_from_obj(mcq_bundle[k])
                    if c_tmp is not None and found_clf is None:
                        found_tf, found_clf = t_tmp, c_tmp
                if found_clf is not None:
                    print(f"Using classifier found under key (first match).")
                    tf, clf = found_tf, found_clf
                else:
                    print("No classifier found anywhere in bundle. Can't compute contributions.")
                    tf, clf = None, None

            # Explain each wrong example with the extracted tf/clf
            for _, r in wrong_df.iterrows():
                print("\n-----------------------------")
                print("Q:", r.get("question"))
                text = r.get("text") or r.get("question")
                if tf is None or clf is None:
                    print("No tf/clf available — skipping eli5, printing raw text snippet:")
                    print(text[:400])
                else:
                    explain_with_fallback(tf, clf, text, top=20)


if __name__ == "__main__":
    main()
