"""LIME explainer UI component."""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional
from app.registry import ExplainerUI, register
from app.visualization import render_token_highlights_html, get_top_k_data, BAR_COLOR_POS, BAR_COLOR_NEG, apply_plotly_theme


class LimeUI(ExplainerUI):
    name = "lime"
    display_name = "LIME"
    description = (
        "Perturbs the input at the word level and fits a local linear model "
        "to estimate each word's influence on the predicted answer."
    )
    supported_tasks = {"yn", "mcq"}

    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        cols = st.columns(3)
        with cols[0]:
            n_samples = st.number_input(
                "Perturbation Samples",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                key=f"{key_prefix}n_samples",
                help="More samples = more accurate but slower.",
            )
        with cols[1]:
            kernel_width = st.slider(
                "Kernel Width",
                min_value=0.1,
                max_value=2.0,
                value=0.75,
                step=0.05,
                key=f"{key_prefix}kernel_width",
                help="Controls locality. Smaller = tighter local fit.",
            )
        with cols[2]:
            mask_token = st.selectbox(
                "Mask Strategy",
                options=["Drop word", "[MASK]"],
                key=f"{key_prefix}mask_token",
                help="How masked words are replaced.",
            )

        return {
            "n_samples": n_samples,
            "kernel_width": kernel_width,
            "mask_token": "" if mask_token == "Drop word" else "[MASK]",
        }

    def run(
        self,
        wrapper,
        prompt: str,
        target_class: Optional[str],
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from medical_llm_toolkit.explainers.lime import MedicalLIME

        lime = MedicalLIME(
            wrapper,
            n_samples=params["n_samples"],
            kernel_width=params["kernel_width"],
            mask_token=params["mask_token"],
            verbose=False,
        )
        result = lime.analyze(prompt, target_class=target_class, visualize=False)
        return result

    def render_results(self, result: Dict[str, Any]):
        # Header metrics
        cols = st.columns(4)
        cols[0].metric("Prediction", result["prediction"])
        cols[1].metric("Target Class", result["target_class"])
        cols[2].metric("P(target)", f"{result['target_probability']:.4f}")
        cols[3].metric("R²", f"{result['r_squared']:.3f}")

        # Token highlights
        st.markdown("**Word-level attributions:**")
        html = render_token_highlights_html(
            result["words"],
            result["word_attributions"],
        )
        st.markdown(html, unsafe_allow_html=True)

        # Top-k bar chart
        st.markdown("**Most influential words:**")
        labels, values = get_top_k_data(
            result["words"], result["word_attributions"], k=15
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
            xaxis_title="Attribution Score",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Option probabilities
        if result.get("all_option_probs"):
            st.markdown("**Option probabilities (original prompt):**")
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


register(LimeUI())
