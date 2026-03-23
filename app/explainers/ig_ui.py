"""Integrated Gradients explainer UI component."""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional
from app.registry import ExplainerUI, register
from app.visualization import render_token_highlights_html, get_top_k_data, BAR_COLOR_POS, BAR_COLOR_NEG, apply_plotly_theme


class IntegratedGradientsUI(ExplainerUI):
    name = "ig"
    display_name = "Integrated Gradients"
    description = (
        "Computes gradient-based token attributions by integrating along "
        "an embedding path from a baseline to the actual input."
    )
    supported_tasks = {"yn", "mcq"}

    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        cols = st.columns(3)
        with cols[0]:
            n_steps = st.number_input(
                "Integration Steps",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key=f"{key_prefix}n_steps",
                help="More steps = more accurate but slower.",
            )
        with cols[1]:
            baseline_type = st.selectbox(
                "Baseline Type",
                options=["pad", "zero", "unk"],
                key=f"{key_prefix}baseline_type",
                help="Reference point for attribution. 'pad' is recommended.",
            )
        with cols[2]:
            convergence = st.checkbox(
                "Show Convergence",
                value=False,
                key=f"{key_prefix}convergence",
                help="Compute and display the convergence delta diagnostic.",
            )

        return {
            "n_steps": n_steps,
            "baseline_type": baseline_type,
            "return_convergence_delta": convergence,
        }

    def run(
        self,
        wrapper,
        prompt: str,
        target_class: Optional[str],
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from medical_llm_toolkit.explainers.integrated_gradients import (
            MedicalIntegratedGradients,
        )

        ig = MedicalIntegratedGradients(
            wrapper,
            n_steps=params["n_steps"],
            baseline_type=params["baseline_type"],
            verbose=False,
        )
        result = ig.attribute(
            prompt,
            target_class=target_class,
            return_convergence_delta=params["return_convergence_delta"],
        )
        return result

    def render_results(self, result: Dict[str, Any]):
        # Header metrics
        metric_cols = st.columns(4)
        metric_cols[0].metric("Prediction", result["prediction"])
        metric_cols[1].metric("Target Class", result["target_class"])
        metric_cols[2].metric("P(target)", f"{result['target_probability']:.4f}")

        if result.get("convergence_delta") is not None:
            metric_cols[3].metric("Convergence Δ", f"{result['convergence_delta']:.4f}")

        if result.get("nan_count", 0) > 0:
            st.warning(
                f"{result['nan_count']} gradient steps were skipped due to NaN values. "
                "Consider using float32 or a different baseline."
            )

        # Token highlights
        st.markdown("**Token-level attributions:**")
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
            xaxis_title="Attribution Score",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Convergence details
        if result.get("convergence_delta") is not None:
            with st.expander("Convergence Details"):
                st.write(f"**Expected sum** (f(input) - f(baseline)): {result['expected_sum']:.4f}")
                st.write(f"**Actual sum** (sum of attributions): {result['actual_sum']:.4f}")
                st.write(f"**Delta**: {result['convergence_delta']:.4f}")


register(IntegratedGradientsUI())
