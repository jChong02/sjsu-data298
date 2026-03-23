"""TokenSHAP explainer UI component."""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional
from app.registry import ExplainerUI, register
from app.visualization import render_token_highlights_html, get_top_k_data, BAR_COLOR_POS, BAR_COLOR_NEG, apply_plotly_theme


def _strip_position_suffix(token_key: str) -> str:
    """Remove the trailing _N position suffix from TokenSHAP keys."""
    return token_key.rsplit("_", 1)[0]


class TokenShapUI(ExplainerUI):
    name = "tokenshap"
    display_name = "TokenSHAP"
    description = (
        "Computes Shapley values for each token by evaluating the model "
        "on random subsets of the input, measuring each token's fair contribution."
    )
    supported_tasks = {"yn", "mcq"}

    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        cols = st.columns(3)
        with cols[0]:
            vectorizer = st.selectbox(
                "Value Function",
                options=["TF-IDF (default)", "Correctness-aware (binary)", "Correctness-aware (prob)"],
                key=f"{key_prefix}vectorizer",
                help=(
                    "TF-IDF: measures response similarity. "
                    "Correctness-aware: measures impact on prediction correctness."
                ),
            )
        with cols[1]:
            sampling_ratio = st.slider(
                "Sampling Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"{key_prefix}sampling_ratio",
                help="Fraction of non-essential combinations to sample. 0 = essential only.",
            )
        with cols[2]:
            max_combinations = st.number_input(
                "Max Combinations",
                min_value=10,
                max_value=5000,
                value=100,
                step=10,
                key=f"{key_prefix}max_combinations",
                help="Upper limit on total combinations evaluated.",
            )

        return {
            "vectorizer": vectorizer,
            "sampling_ratio": sampling_ratio,
            "max_combinations": max_combinations,
        }

    def run(
        self,
        wrapper,
        prompt: str,
        target_class: Optional[str],
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from medical_llm_toolkit.explainers.tokenshap.token_shap.token_shap import (
            StringSplitter,
        )
        from medical_llm_toolkit.explainers.tokenshap.extensions.qa_tokenshap import (
            QATokenSHAP,
        )
        from medical_llm_toolkit.explainers.tokenshap.token_shap.base import (
            TfidfTextVectorizer,
        )

        # Set wrapper to answer_only mode for TokenSHAP
        prev_mode = wrapper.mode
        wrapper.set_mode("answer_only")

        try:
            splitter = StringSplitter()

            # Select vectorizer
            vec_choice = params["vectorizer"]
            if vec_choice.startswith("Correctness-aware") and ground_truth:
                from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions.correctness_value import (
                    CorrectnessValueFunction,
                )
                mode = "binary" if "binary" in vec_choice else "prob"
                vectorizer = CorrectnessValueFunction(
                    correct_label=ground_truth, mode=mode
                )
            else:
                vectorizer = TfidfTextVectorizer()

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

            # Extract shapley values
            shapley_values = analyzer.shapley_values

            # Build token list and score list (strip position suffixes)
            tokens = [_strip_position_suffix(k) for k in shapley_values.keys()]
            scores = list(shapley_values.values())

            return {
                "tokens": tokens,
                "attributions": np.array(scores),
                "shapley_values": shapley_values,
                "results_df": results_df,
                "prediction": wrapper.last_answer,
                "target_class": target_class,
                "all_option_probs": wrapper.last_option_probs,
            }
        finally:
            wrapper.set_mode(prev_mode)

    def render_results(self, result: Dict[str, Any]):
        # Header metrics
        cols = st.columns(3)
        cols[0].metric("Prediction", result.get("prediction", "N/A"))
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
                "Token": [_strip_position_suffix(k) for k in sv.keys()],
                "Shapley Value": list(sv.values()),
            })
            df = df.sort_values("Shapley Value", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)


register(TokenShapUI())
