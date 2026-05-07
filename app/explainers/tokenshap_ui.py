"""TokenSHAP explainer UI component."""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional
from app.registry import ExplainerUI, register
from app.visualization import render_token_highlights_html, get_top_k_data, BAR_COLOR_POS, BAR_COLOR_NEG, apply_plotly_theme


def _strip_position_suffix(token_key: str) -> str:
    """Remove the trailing _N position suffix from TokenSHAP keys."""
    return token_key.rsplit("_", 1)[0]


# ---------------------------------------------------------------------------
# Cached loaders — expensive objects are shared across reruns and sessions
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_embedding_vectorizer(model_name: str, device: str):
    from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions import EmbeddingVectorizer
    return EmbeddingVectorizer(model_name=model_name, device=device)


@st.cache_resource
def _load_spacy_backend(model_name: str):
    from medical_llm_toolkit.explainers.tokenshap.extensions.splitters import SpaCyNERBackend
    return SpaCyNERBackend(model_name)


@st.cache_resource
def _load_hf_ner_backend(model_name: str, device: str):
    from medical_llm_toolkit.explainers.tokenshap.extensions.splitters import HuggingFaceNERBackend
    return HuggingFaceNERBackend(model_name, device=device)


# ---------------------------------------------------------------------------
# UI component
# ---------------------------------------------------------------------------

# Vectorizer options. Correctness-based and Hybrid options require a
# ground-truth A/B/C/D label and are hidden when the task is free response.
_VECTORIZER_OPTIONS_LABELED = [
    "TF-IDF (default)",
    "Correctness-aware (binary)",
    "Correctness-aware (prob)",
    "Embedding similarity",
    "Hybrid: correctness + embedding (binary)",
    "Hybrid: correctness + embedding (prob)",
]

_VECTORIZER_OPTIONS_FREE = [
    "TF-IDF (default)",
    "Embedding similarity",
]

_SPLITTER_OPTIONS = [
    "Word (default)",
    "Semantic — spaCy NER",
    "Semantic — HuggingFace NER",
]


def _current_task_type() -> str:
    """Return the task type of the currently-loaded wrapper, or '' if none."""
    wrapper = st.session_state.get("wrapper")
    if wrapper is None:
        return ""
    return getattr(wrapper, "task_type", "") or ""


class TokenShapUI(ExplainerUI):
    name = "tokenshap"
    display_name = "TokenSHAP"
    description = (
        "Computes Shapley values for each token by evaluating the model "
        "on random subsets of the input, measuring each token's fair contribution."
    )
    supported_tasks = {"yn", "mcq", "free"}

    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        is_free = _current_task_type() == "free"

        # Free task gets a narrower vectorizer list (correctness-based options
        # need a ground-truth letter label) and a lower default Max Combinations
        # because free generation produces ~200 tokens per LLM call vs 1 token
        # under constrained yn/mcq decoding — each combination is much slower.
        vec_options = _VECTORIZER_OPTIONS_FREE if is_free else _VECTORIZER_OPTIONS_LABELED
        default_max_comb = 30 if is_free else 100

        if is_free:
            st.caption(
                "Free-response mode uses plain TokenSHAP without the QA-aware "
                "perturbation. Correctness/Hybrid value functions are hidden "
                "because they require an A/B/C/D ground-truth label. Each LLM "
                "call generates a full free-text response, so keep "
                "**Max Combinations** low (≈30) — expect several minutes."
            )

        # --- Row 1: main controls ---
        cols = st.columns(4)

        with cols[0]:
            splitter = st.selectbox(
                "Splitter",
                options=_SPLITTER_OPTIONS,
                key=f"{key_prefix}splitter",
                help=(
                    "Word: whitespace split (default). "
                    "Semantic: groups named entities as single atomic tokens "
                    "so multi-word entities are not split across positions."
                ),
            )
        with cols[1]:
            vectorizer = st.selectbox(
                "Value Function",
                options=vec_options,
                key=f"{key_prefix}vectorizer",
                help=(
                    "TF-IDF: lexical response similarity. "
                    "Embedding: semantic response similarity. "
                    "Correctness-aware: impact on predicting the right answer. "
                    "Hybrid: blend of correctness and embedding similarity."
                ),
            )
        with cols[2]:
            sampling_ratio = st.slider(
                "Sampling Ratio",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                key=f"{key_prefix}sampling_ratio",
                help="Fraction of non-essential combinations to sample. 0 = essential only.",
            )
        with cols[3]:
            max_combinations = st.number_input(
                "Max Combinations",
                min_value=10, max_value=5000, value=default_max_comb, step=10,
                key=f"{key_prefix}max_combinations",
                help="Upper limit on total combinations evaluated.",
            )

        params: Dict[str, Any] = {
            "splitter":        splitter,
            "vectorizer":      vectorizer,
            "sampling_ratio":  sampling_ratio,
            "max_combinations": int(max_combinations),
        }

        # --- Row 2: conditional extension config ---
        needs_embedding = "Embedding" in vectorizer or "Hybrid" in vectorizer
        needs_ner       = "Semantic" in splitter

        if needs_embedding or needs_ner:
            st.markdown("**Extension settings**")
            ext_cols = st.columns(4)
            col_idx = 0

            if needs_embedding:
                with ext_cols[col_idx]:
                    params["embedding_model"] = st.text_input(
                        "Embedding model",
                        value="sentence-transformers/all-MiniLM-L6-v2",
                        key=f"{key_prefix}embedding_model",
                        help="Any HuggingFace encoder model ID.",
                    )
                col_idx += 1
                with ext_cols[col_idx]:
                    params["embedding_device"] = st.selectbox(
                        "Embedding device",
                        options=["cpu", "cuda"],
                        key=f"{key_prefix}embedding_device",
                    )
                col_idx += 1

            if "Hybrid" in vectorizer:
                with ext_cols[col_idx]:
                    params["alpha"] = st.slider(
                        "Alpha (correctness weight)",
                        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                        key=f"{key_prefix}alpha",
                        help="1.0 = pure correctness, 0.0 = pure embedding similarity.",
                    )
                col_idx += 1

            if needs_ner:
                with ext_cols[col_idx]:
                    if "spaCy" in splitter:
                        params["ner_model"] = st.text_input(
                            "spaCy model",
                            value="en_core_web_sm",
                            key=f"{key_prefix}ner_model",
                            help="Must be installed: python -m spacy download <model>",
                        )
                    else:
                        params["ner_model"] = st.text_input(
                            "HF NER model",
                            value="dslim/bert-base-NER",
                            key=f"{key_prefix}ner_model",
                            help="Any HuggingFace token-classification model ID.",
                        )
                col_idx += 1
                if "HuggingFace" in splitter and col_idx < 4:
                    with ext_cols[col_idx]:
                        params["ner_device"] = st.selectbox(
                            "NER device",
                            options=["cpu", "cuda"],
                            key=f"{key_prefix}ner_device",
                        )

        # Warn if a correctness-based vectorizer is chosen but no ground truth
        # will be available — the app passes ground_truth at run time, so we
        # surface the warning here rather than silently falling back.
        if ("Correctness" in vectorizer or "Hybrid" in vectorizer):
            st.info(
                "Correctness-aware and Hybrid value functions require a **ground truth label**. "
                "Make sure the correct answer is set in the main panel before running."
            )

        return params

    def run(
        self,
        wrapper,
        prompt: str,
        target_class: Optional[str],
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from medical_llm_toolkit.explainers.tokenshap.token_shap.token_shap import (
            StringSplitter, TokenSHAP,
        )
        from medical_llm_toolkit.explainers.tokenshap.extensions.qa_tokenshap import QATokenSHAP
        from medical_llm_toolkit.explainers.tokenshap.token_shap.base import TfidfTextVectorizer

        is_free = wrapper.task_type == "free"

        # Constrained yn/mcq tasks switch the wrapper into answer_only so each
        # generate call returns a deterministic single-letter prediction. For
        # free task the wrapper has no constrained answer space, so we keep
        # whatever mode the user has set; greedy decoding in answer_rationale
        # mode produces the same response across calls (no sampling), which
        # is what TokenSHAP's value function needs.
        prev_mode = wrapper.mode
        if not is_free:
            wrapper.set_mode("answer_only")

        try:
            # --- Splitter ---
            splitter_choice = params["splitter"]
            if "spaCy" in splitter_choice:
                from medical_llm_toolkit.explainers.tokenshap.extensions.splitters import SemanticSplitter
                ner = _load_spacy_backend(params.get("ner_model", "en_core_web_sm"))
                splitter = SemanticSplitter(ner)
            elif "HuggingFace" in splitter_choice:
                from medical_llm_toolkit.explainers.tokenshap.extensions.splitters import SemanticSplitter
                ner = _load_hf_ner_backend(
                    params.get("ner_model", "dslim/bert-base-NER"),
                    params.get("ner_device", "cpu"),
                )
                splitter = SemanticSplitter(ner)
            else:
                splitter = StringSplitter()

            # --- Vectorizer ---
            vec_choice = params["vectorizer"]
            needs_ground_truth = "Correctness" in vec_choice or "Hybrid" in vec_choice

            if needs_ground_truth and not ground_truth:
                st.error(
                    "This value function requires a ground truth label. "
                    "Please set the correct answer in the main panel and re-run."
                )
                return {}

            if vec_choice == "TF-IDF (default)":
                vectorizer = TfidfTextVectorizer()

            elif "Embedding similarity" in vec_choice:
                vectorizer = _load_embedding_vectorizer(
                    params.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    params.get("embedding_device", "cpu"),
                )

            elif vec_choice == "Correctness-aware (binary)":
                from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions import CorrectnessValueFunction
                vectorizer = CorrectnessValueFunction(correct_label=ground_truth, mode="binary")

            elif vec_choice == "Correctness-aware (prob)":
                from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions import CorrectnessValueFunction
                vectorizer = CorrectnessValueFunction(correct_label=ground_truth, mode="prob")

            elif "Hybrid" in vec_choice:
                from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions import HybridValueFunction
                emb = _load_embedding_vectorizer(
                    params.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    params.get("embedding_device", "cpu"),
                )
                mode = "binary" if "binary" in vec_choice else "prob"
                vectorizer = HybridValueFunction(
                    correct_label=ground_truth,
                    embedding_vectorizer=emb,
                    mode=mode,
                    alpha=params.get("alpha", 0.5),
                )

            else:
                vectorizer = TfidfTextVectorizer()

            # --- Run ---
            # QATokenSHAP keeps the answer-choices block static during
            # perturbation, which is the right thing for yn/mcq prompts. Free
            # prompts have no answer-choices block, so we use plain TokenSHAP
            # and let it perturb the entire prompt.
            if is_free:
                analyzer = TokenSHAP(
                    model=wrapper,
                    splitter=splitter,
                    vectorizer=vectorizer,
                    debug=False,
                )
            else:
                analyzer = QATokenSHAP(
                    model=wrapper,
                    splitter=splitter,
                    vectorizer=vectorizer,
                    debug=False,
                )
            results_df = analyzer.analyze(
                prompt,
                sampling_ratio=params["sampling_ratio"],
                max_combinations=params["max_combinations"],
            )

            shapley_values = analyzer.shapley_values
            tokens = [_strip_position_suffix(k) for k in shapley_values.keys()]
            scores = list(shapley_values.values())

            return {
                "tokens":          tokens,
                "attributions":    np.array(scores),
                "shapley_values":  shapley_values,
                "results_df":      results_df,
                "prediction":      wrapper.last_answer,
                "target_class":    target_class,
                "all_option_probs": wrapper.last_option_probs,
                "task_type":       wrapper.task_type,
            }

        finally:
            if not is_free:
                wrapper.set_mode(prev_mode)

    def render_results(self, result: Dict[str, Any]):
        if not result:
            return

        prediction = result.get("prediction", "")
        is_free = result.get("task_type") == "free"

        # Header metrics. yn/mcq predictions are a single letter so we show
        # them as a normal metric. Free predictions are full text — render
        # them as a code block above the metrics so the layout doesn't break.
        if is_free:
            st.markdown("**Generated response (baseline for Shapley comparison):**")
            display_text = str(prediction).strip() or "(empty response)"
            if len(display_text) > 600:
                st.code(display_text[:600] + "\n... [truncated]")
                with st.expander("Full response"):
                    st.code(display_text)
            else:
                st.code(display_text)
        else:
            cols = st.columns(3)
            cols[0].metric("Prediction", prediction or "N/A")
            cols[1].metric("Target Class", result.get("target_class", "N/A"))
            if result.get("all_option_probs"):
                target = result.get("target_class")
                prob = result["all_option_probs"].get(target, 0)
                cols[2].metric("P(target)", f"{prob:.4f}")

        # Token highlights
        st.markdown("**Token-level Shapley values:**")
        html = render_token_highlights_html(
            result["tokens"],
            result["attributions"],
        )
        st.markdown(html, unsafe_allow_html=True)

        # Top-k bar chart
        st.markdown("**Most influential tokens:**")
        labels, values = get_top_k_data(
            result["tokens"], result["attributions"], k=15
        )
        import plotly.graph_objects as go

        colors = [BAR_COLOR_POS if v > 0 else BAR_COLOR_NEG for v in values]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            height=max(300, len(labels) * 28),
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Shapley Value",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Option probabilities
        if result.get("all_option_probs"):
            st.markdown("**Option probabilities:**")
            probs = result["all_option_probs"]
            prob_fig = go.Figure(
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color="#e89923",
                )
            )
            prob_fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title="Probability",
            )
            apply_plotly_theme(prob_fig)
            st.plotly_chart(prob_fig, use_container_width=True)

        # Raw Shapley values table
        with st.expander("Raw Shapley Values"):
            import pandas as pd
            sv = result["shapley_values"]
            df = pd.DataFrame({
                "Token":         [_strip_position_suffix(k) for k in sv.keys()],
                "Shapley Value": list(sv.values()),
            })
            df = df.sort_values("Shapley Value", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)


register(TokenShapUI())
