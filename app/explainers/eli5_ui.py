"""ELI5 explainer UI component.

Loads a pre-trained surrogate bundle from
``medical_llm_toolkit/eli5_surrogates/`` based on the currently loaded
LLM and renders the native eli5 per-class feature-contribution HTML.

ELI5 is a *global* surrogate, not a per-prompt method, so it does not
participate in the cross-method Comparison tab - the main app filters
it out and shows a note explaining why.
"""

from typing import Any, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

from app.registry import ExplainerUI, register
from app.visualization import apply_plotly_theme, BAR_COLOR_POS, BAR_COLOR_NEG


# Kind labels shown in the radio control.
_KIND_LABELS = {
    "mimic": "Mimic - explains the LLM",
    "gold":  "Gold - dataset patterns (LLM-independent)",
    "error": "Error - failure-mode features",
}

# Help text shown next to the kind selector.
_KIND_HELP = (
    "**Mimic** trains the surrogate to predict what *this LLM* outputs - "
    "this is the actual model explanation.\n\n"
    "**Gold** trains the surrogate to predict the true answer - a dataset-level "
    "diagnostic, independent of the LLM.\n\n"
    "**Error** trains the surrogate to predict whether the LLM was wrong - "
    "useful for spotting failure-correlated features."
)


class ELI5UI(ExplainerUI):
    name = "eli5"
    display_name = "ELI5"
    description = (
        "Trains a TF-IDF + Logistic Regression surrogate on a QA corpus and "
        "uses ELI5 to highlight per-class feature contributions. Global "
        "surrogate - explains the model's overall behavior, not this specific "
        "prompt's per-token attributions."
    )
    supported_tasks = {"yn", "mcq"}

    # ------------------------------------------------------------------
    # Bundle lookup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(model_id: str, task_type: str) -> str:
        return f"_eli5_explainer__{model_id}__{task_type}"

    @classmethod
    def _get_explainer(cls, wrapper):
        """Load (and cache in session state) the MedicalELI5 explainer for the
        currently loaded LLM. Returns ``None`` if no bundle is found."""
        from medical_llm_toolkit.explainers.eli5 import MedicalELI5

        model_id = wrapper.model_name
        task_type = wrapper.task_type
        key = cls._cache_key(model_id, task_type)

        cached = st.session_state.get(key)
        if cached is not None:
            return cached

        explainer = MedicalELI5.from_disk(model_id, task_type)
        if explainer is not None:
            st.session_state[key] = explainer
        return explainer

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        wrapper = st.session_state.get("wrapper")
        if wrapper is None:
            st.info("Load a model in the sidebar to enable ELI5.")
            return {"_status": "no_model"}

        explainer = self._get_explainer(wrapper)
        if explainer is None:
            st.warning(
                f"No pre-trained ELI5 surrogate found for "
                f"`{wrapper.model_name}` ({wrapper.task_type}).\n\n"
                "Bundles ship with the four preset models. To train one for "
                "a custom model, run `notebooks/train_eli5_surrogates.ipynb`."
            )
            return {"_status": "no_bundle"}

        kinds = explainer.available_kinds()
        if not kinds:
            st.error("No surrogates in this bundle.")
            return {"_status": "no_kinds"}

        cols = st.columns([2, 1])
        with cols[0]:
            kind = st.radio(
                "Surrogate type",
                options=kinds,
                format_func=lambda k: _KIND_LABELS.get(k, k),
                key=f"{key_prefix}kind",
                help=_KIND_HELP,
            )
        with cols[1]:
            top = st.slider(
                "Top features",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                key=f"{key_prefix}top",
            )

        # Surface the kinds that *aren't* in the bundle so the user knows why.
        missing = [k for k in ("mimic", "gold", "error") if k not in kinds]
        if missing:
            st.caption(
                f"Skipped surrogate(s) for this model: **{', '.join(missing)}** "
                "- typically because the LLM produced only one class label "
                "across all training rows (collapse), so the surrogate had "
                "no variety to learn from."
            )

        return {"_status": "ok", "kind": kind, "top": int(top)}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        wrapper,
        prompt: str,
        target_class: Optional[str],
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        status = params.get("_status", "ok")
        if status != "ok":
            return {"_status": status,
                    "model_id": getattr(wrapper, "model_name", "?"),
                    "task_type": getattr(wrapper, "task_type", "?")}

        explainer = self._get_explainer(wrapper)
        if explainer is None:
            return {"_status": "no_bundle",
                    "model_id": wrapper.model_name,
                    "task_type": wrapper.task_type}

        result = explainer.explain(
            prompt, kind=params["kind"], top=params["top"]
        )
        result["_status"] = "ok"
        result["bundle_meta"] = {
            "trained_at": explainer.trained_at,
            "n_train": explainer.n_train,
            "n_test": explainer.n_test,
            "vocab_size": explainer.vocab_size,
            "available_kinds": explainer.available_kinds(),
        }
        try:
            result["global_html"] = explainer.show_global_weights(
                kind=params["kind"], top=30
            )
        except Exception:
            result["global_html"] = None
        return result

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render_results(self, result: Dict[str, Any]):
        status = result.get("_status", "ok")
        if status == "no_model":
            st.info("Load a model first.")
            return
        if status == "no_bundle":
            st.warning(
                f"No pre-trained ELI5 surrogate found for "
                f"`{result.get('model_id', '?')}` "
                f"({result.get('task_type', '?')}).\n\n"
                "Run `notebooks/train_eli5_surrogates.ipynb` to train one "
                "for this model."
            )
            return
        if status == "no_kinds":
            st.error("No surrogates available in the bundle.")
            return
        if status != "ok":
            st.error(f"Unknown ELI5 status: {status}")
            return

        # ---- Header metrics ----
        cols = st.columns(4)
        cols[0].metric("Surrogate", result["kind"])
        cols[1].metric("Predicted", result["predicted_class"])
        cols[2].metric("Heldout score", f"{result['heldout_score']:.3f}")
        baseline = result.get("majority_baseline")
        if baseline is not None:
            cols[3].metric("Majority baseline", f"{baseline:.3f}")

        # ---- Signal-strength banner ----
        delta = result.get("delta_above_prior")
        if delta is not None:
            if delta < 0.03:
                st.warning(
                    f"**Surrogate fidelity is at or below the majority-class "
                    f"baseline** (delta = {delta:+.3f}). The explanation mostly "
                    f"reflects this LLM's default class bias rather than "
                    f"content-driven reasoning. Interpret with caution."
                )
            elif delta < 0.10:
                st.caption(
                    f"Limited above-prior signal (delta = {delta:+.3f}). The "
                    f"surrogate captures only modest content-driven structure "
                    f"beyond the class prior."
                )

        # ---- Per-prompt class probabilities (predicted class highlighted) ----
        import plotly.graph_objects as go  # noqa: E402 - local to avoid import cost on cold paths

        predicted = result.get("predicted_class")
        if result.get("class_probs"):
            st.markdown("**Surrogate class probabilities for this prompt:**")
            probs = result["class_probs"]
            colors = [
                BAR_COLOR_POS if k == predicted else "rgba(232,153,35,0.30)"
                for k in probs.keys()
            ]
            fig = go.Figure(
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in probs.values()],
                    textposition="auto",
                    textfont=dict(color="#e8e0d4"),
                )
            )
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=20, t=10, b=0),
                yaxis_title="P(class)",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ---- Per-class feature contributions: tabs with themed bar charts ----
        per_class = result.get("per_class_features") or {}
        if per_class:
            st.markdown("**Per-class feature contributions:**")
            st.caption(
                "Each tab shows which n-gram features pushed the surrogate "
                "toward that class. Orange bars support the class; teal bars "
                "push against it. **BIAS** is the surrogate's intercept - "
                "the class's prior contribution before any prompt content. "
                "**Score** is the linear combination (bias + sum of feature "
                "contributions) before softmax."
            )

            labels = result.get("labels", list(per_class.keys()))
            ordered_labels = (
                [predicted] + [l for l in labels if l != predicted]
                if predicted in per_class else list(per_class.keys())
            )
            tab_titles = [
                (f"• {l}" if l == predicted else l) for l in ordered_labels
            ]
            class_tabs = st.tabs(tab_titles)

            for label, tab in zip(ordered_labels, class_tabs):
                info = per_class.get(label)
                if info is None:
                    continue
                with tab:
                    # Sub-metrics: P(class), score, bias
                    m_cols = st.columns(3)
                    if info.get("proba") is not None:
                        m_cols[0].metric(f"P({label})", f"{info['proba']:.3f}")
                    if info.get("score") is not None:
                        m_cols[1].metric("Score", f"{info['score']:+.3f}")
                    if info.get("bias") is not None:
                        m_cols[2].metric("BIAS", f"{info['bias']:+.3f}")

                    features = info.get("features") or []
                    if not features:
                        st.caption(
                            "No prompt-content features contributed beyond the "
                            "BIAS term for this class."
                        )
                        continue

                    names = [f[0] for f in features]
                    weights = [f[1] for f in features]
                    colors = [
                        BAR_COLOR_POS if w > 0 else BAR_COLOR_NEG
                        for w in weights
                    ]
                    fig = go.Figure(
                        go.Bar(
                            x=weights,
                            y=names,
                            orientation="h",
                            marker_color=colors,
                            text=[f"{w:+.3f}" for w in weights],
                            textposition="auto",
                            textfont=dict(color="#e8e0d4"),
                        )
                    )
                    fig.update_layout(
                        height=max(280, len(names) * 28),
                        margin=dict(l=0, r=20, t=10, b=0),
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="Contribution to score",
                    )
                    apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

        # ---- Raw eli5 HTML kept behind an expander for completeness ----
        if result.get("html"):
            with st.expander("Raw ELI5 table view (alternative layout)"):
                st.caption(
                    "Native eli5 HTML output. Useful if you prefer a tabular "
                    "view or want to see all classes at once. Light theme."
                )
                components.html(result["html"], height=600, scrolling=True)

        if result.get("global_html"):
            with st.expander("Global feature weights across the training corpus (raw view)"):
                components.html(result["global_html"], height=600, scrolling=True)

        # ---- Bundle metadata ----
        meta = result.get("bundle_meta", {})
        with st.expander("Surrogate metadata"):
            st.markdown(
                f"- **Model**: `{result['model_id']}`\n"
                f"- **Task**: `{result['task_type']}`\n"
                f"- **Trained at**: `{meta.get('trained_at', 'unknown')}`\n"
                f"- **Train rows**: {meta.get('n_train', '?')}\n"
                f"- **Test rows**: {meta.get('n_test', '?')}\n"
                f"- **Vocab size**: {meta.get('vocab_size', '?')}\n"
                f"- **Available surrogates**: "
                f"{', '.join(meta.get('available_kinds', [])) or 'none'}"
            )


register(ELI5UI())
