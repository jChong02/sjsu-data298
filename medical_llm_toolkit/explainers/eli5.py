"""
ELI5 explainer for medical LLMs.

Wraps a pre-trained TF-IDF + Logistic Regression surrogate and uses the
`eli5` library to produce per-class feature-contribution HTML for any
prompt. Unlike LIME / IG / TokenSHAP, this is a *global* surrogate: it is
trained once on a corpus and then applied to any prompt by transforming
the prompt with the learned TF-IDF vocabulary. Explanations are per-class
contributions across the surrogate's vocabulary, not per-prompt-token
attributions.

Pre-trained surrogate bundles for the four preset models live in
``medical_llm_toolkit/eli5_surrogates/``. To train new bundles for a
custom model see ``notebooks/train_eli5_surrogates.ipynb``.

Bundle pickle schema (``schema_version=1``)::

    {
        "schema_version": 1,
        "model_id": str,
        "task_type": "mcq" | "yn",
        "n_samples_requested": int,
        "trained_at": iso8601 string,
        "vectorizer": sklearn TfidfVectorizer,
        "vocab_size": int,
        "n_train": int,
        "n_test": int,
        "surrogates": {
            "mimic" | "gold" | "error": {
                "clf": sklearn LogisticRegression,
                "labels": list[str],
                "heldout_score": float,
            },
            ...
        },
    }
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SUPPORTED_KINDS = ("mimic", "gold", "error")

KIND_DESCRIPTIONS: Dict[str, str] = {
    "mimic": "Predicts what the LLM outputs (model-explanation surrogate).",
    "gold":  "Predicts the true answer (dataset-level diagnostic; LLM-independent).",
    "error": "Predicts whether the LLM was wrong (failure-mode diagnostic).",
}


def model_id_to_filename(model_id: str, task_type: str) -> str:
    """Filename convention shared by the training notebook and the loader."""
    safe = model_id.replace("/", "__")
    return f"{safe}__{task_type}.pkl"


def default_surrogate_dir() -> Path:
    """Location of bundled pre-trained surrogates inside the package."""
    return Path(__file__).resolve().parent.parent / "eli5_surrogates"


class MedicalELI5:
    """ELI5 explainer wrapping a pre-trained surrogate bundle.

    Loaders:
        - ``MedicalELI5.from_bundle_path(path)`` — load a specific .pkl
        - ``MedicalELI5.from_disk(model_id, task_type, search_dir=None)``
          — look up a bundle in the surrogates directory; returns ``None``
          if no bundle exists for that ``(model_id, task_type)``.

    Main entry point:
        - ``.explain(prompt, kind, top)`` — returns a dict with native
          eli5 HTML, surrogate predictions, fidelity metrics, and an
          implied class prior derived from the surrogate's intercept
          term so callers can show an honest "above prior" delta.
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle
        self.model_id: str = bundle["model_id"]
        self.task_type: str = bundle["task_type"]
        self.vectorizer = bundle["vectorizer"]
        self.surrogates: Dict[str, Dict[str, Any]] = bundle["surrogates"]
        self.vocab_size: int = int(bundle.get("vocab_size", -1))
        self.n_train: int = int(bundle.get("n_train", -1))
        self.n_test: int = int(bundle.get("n_test", -1))
        self.trained_at: str = str(bundle.get("trained_at", ""))
        self.schema_version: int = int(bundle.get("schema_version", 1))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_bundle_path(cls, path: Path) -> "MedicalELI5":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(bundle)

    @classmethod
    def from_disk(
        cls,
        model_id: str,
        task_type: str,
        search_dir: Optional[Path] = None,
    ) -> Optional["MedicalELI5"]:
        """Look up a bundle by ``(model_id, task_type)``.

        Returns ``None`` if no matching bundle is found in
        ``search_dir`` (defaults to the packaged ``eli5_surrogates``
        directory).
        """
        if search_dir is None:
            search_dir = default_surrogate_dir()
        path = Path(search_dir) / model_id_to_filename(model_id, task_type)
        if not path.exists():
            return None
        return cls.from_bundle_path(path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def has_kind(self, kind: str) -> bool:
        return kind in self.surrogates

    def available_kinds(self) -> List[str]:
        """Return surrogate kinds present in the bundle, in canonical order."""
        return [k for k in SUPPORTED_KINDS if k in self.surrogates]

    def heldout_score(self, kind: str) -> Optional[float]:
        s = self.surrogates.get(kind)
        return None if s is None else float(s["heldout_score"])

    def labels(self, kind: str) -> List[str]:
        s = self.surrogates.get(kind)
        return [] if s is None else list(s["labels"])

    def implied_class_prior(self, kind: str) -> Dict[str, float]:
        """Compute the surrogate's implied class prior from its intercepts.

        - Multiclass: ``softmax(intercept_)`` per class.
        - Binary: ``sigmoid(intercept_)`` is ``P(class=1)``; the other
          class's probability is the complement.

        This approximates the majority-class baseline a "predict from
        intercept only" classifier would achieve, which is the right
        comparison point for the surrogate's heldout score.
        """
        s = self.surrogates.get(kind)
        if s is None:
            return {}
        clf = s["clf"]
        labels = list(s["labels"])
        intercepts = np.asarray(getattr(clf, "intercept_", []), dtype=float).ravel()
        if intercepts.size == 0:
            return {}
        # Binary LogReg in sklearn stores a single intercept for class 1.
        if intercepts.size == 1 and len(labels) == 2:
            p1 = float(1.0 / (1.0 + np.exp(-intercepts[0])))
            return {labels[0]: 1.0 - p1, labels[1]: p1}
        # Multiclass: softmax of per-class intercepts.
        if intercepts.size == len(labels):
            shifted = intercepts - intercepts.max()
            exp = np.exp(shifted)
            probs = exp / exp.sum()
            return {label: float(p) for label, p in zip(labels, probs)}
        return {}

    def majority_baseline(self, kind: str) -> Optional[float]:
        """Approximate majority-class baseline = ``max(implied_class_prior)``."""
        prior = self.implied_class_prior(kind)
        if not prior:
            return None
        return max(prior.values())

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    def explain(
        self,
        prompt: str,
        kind: str = "mimic",
        top: int = 20,
    ) -> Dict[str, Any]:
        """Run ELI5 on ``prompt`` using the chosen surrogate.

        Returns a dict with:
            kind: the surrogate kind used
            predicted_class: surrogate's top-class prediction
            class_probs: dict mapping each class label to its probability
            labels: the surrogate's class labels
            per_class_features: dict mapping class -> {"score", "proba",
                "bias", "features"} where ``features`` is a list of
                ``(name, weight, value)`` tuples sorted by absolute weight.
                The ``<BIAS>`` feature is split out into its own field.
            html: eli5.show_prediction HTML (kept for the optional raw view)
            heldout_score: held-out score from training
            majority_baseline: implied prior of the dominant class
            delta_above_prior: heldout_score - majority_baseline
            model_id, task_type: copied from the bundle
        """
        import eli5  # lazy import — avoids hard dependency at import time

        if kind not in self.surrogates:
            raise ValueError(
                f"Surrogate kind '{kind}' is not available for "
                f"{self.model_id} ({self.task_type}). "
                f"Available: {self.available_kinds()}"
            )

        s = self.surrogates[kind]
        clf = s["clf"]
        vec = self.vectorizer
        labels = [str(x) for x in s["labels"]]

        # Surrogate prediction + class probabilities for this prompt.
        X = vec.transform([prompt])
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            class_probs = {
                str(c): float(p)
                for c, p in zip(getattr(clf, "classes_", labels), probs)
            }
        else:
            class_probs = {}
        predicted_class = str(clf.predict(X)[0])

        # Pull a structured per-class breakdown via eli5.explain_prediction
        # (returns an Explanation object — much friendlier than the HTML).
        target_names = labels if len(labels) > 1 else None
        per_class_features: Dict[str, Dict[str, Any]] = {}
        html: str
        try:
            expl_obj = eli5.explain_prediction(
                clf, prompt, vec=vec, top=top, target_names=target_names
            )
            for target in expl_obj.targets:
                tname = str(target.target)
                bias_weight: Optional[float] = None
                features: List[tuple] = []
                # eli5 splits feature_weights into pos/neg lists already
                fw = target.feature_weights
                for f in list(fw.pos) + list(fw.neg):
                    if f.feature == "<BIAS>":
                        bias_weight = float(f.weight)
                        continue
                    features.append((
                        str(f.feature),
                        float(f.weight),
                        float(getattr(f, "value", 0.0) or 0.0),
                    ))
                features.sort(key=lambda t: abs(t[1]), reverse=True)
                per_class_features[tname] = {
                    "score": (float(target.score)
                              if target.score is not None else None),
                    "proba": (float(target.proba)
                              if target.proba is not None else None),
                    "bias": bias_weight,
                    "features": features,
                }

            # Keep the native HTML around for users who want the raw view.
            expl_html = eli5.show_prediction(
                clf, prompt, vec=vec, top=top, target_names=target_names
            )
            html = getattr(expl_html, "data", None) or str(expl_html)
        except Exception as exc:  # pragma: no cover — defensive
            per_class_features = {}
            html = f"<em>eli5.explain_prediction failed: {exc!s}</em>"

        score = float(s["heldout_score"])
        baseline = self.majority_baseline(kind)
        delta = (score - baseline) if baseline is not None else None

        return {
            "kind": kind,
            "html": html,
            "per_class_features": per_class_features,
            "predicted_class": predicted_class,
            "class_probs": class_probs,
            "labels": labels,
            "heldout_score": score,
            "majority_baseline": baseline,
            "delta_above_prior": delta,
            "model_id": self.model_id,
            "task_type": self.task_type,
        }

    def show_global_weights(self, kind: str = "mimic", top: int = 30) -> str:
        """Return ``eli5.show_weights`` HTML for the surrogate's global feature importances."""
        import eli5

        if kind not in self.surrogates:
            raise ValueError(
                f"Kind '{kind}' is not available; have {self.available_kinds()}"
            )
        s = self.surrogates[kind]
        labels = [str(x) for x in s["labels"]]
        target_names = labels if len(labels) > 1 else None
        expl = eli5.show_weights(
            s["clf"], vec=self.vectorizer, top=top, target_names=target_names
        )
        return getattr(expl, "data", None) or str(expl)

    def __repr__(self) -> str:  # pragma: no cover — convenience
        kinds = ",".join(self.available_kinds())
        return (
            f"MedicalELI5(model='{self.model_id}', task='{self.task_type}', "
            f"kinds=[{kinds}], vocab={self.vocab_size})"
        )
