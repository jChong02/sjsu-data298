"""Tree-of-Thought reasoning UI component."""

import streamlit as st
from typing import Dict, Any, Optional

from app.reasoning_registry import ReasoningUI, register
from app.visualization import apply_plotly_theme


class TreeOfThoughtUI(ReasoningUI):
    name = "tot"
    display_name = "Tree-of-Thought"
    description = (
        "Generates multiple candidate reasoning branches, scores each by mean "
        "log-probability under the model, and selects the most coherent one "
        "before answering."
    )
    supported_tasks = {"yn", "mcq", "free"}

    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        cols = st.columns(4)
        with cols[0]:
            n_branches = st.number_input(
                "Branches",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                key=f"{key_prefix}n_branches",
                help="How many candidate reasoning paths to sample. "
                     "More = better selection but linearly slower.",
            )
        with cols[1]:
            max_thought_tokens = st.number_input(
                "Max Thought Tokens",
                min_value=20,
                max_value=300,
                value=80,
                step=10,
                key=f"{key_prefix}max_thought_tokens",
                help="Cap on tokens per reasoning branch.",
            )
        with cols[2]:
            thought_temperature = st.slider(
                "Branch Temperature",
                min_value=0.1,
                max_value=1.5,
                value=0.8,
                step=0.05,
                key=f"{key_prefix}thought_temperature",
                help="Higher = more diverse branches.",
            )
        with cols[3]:
            rationale_max_tokens = st.number_input(
                "Rationale Max Tokens",
                min_value=20,
                max_value=400,
                value=150,
                step=10,
                key=f"{key_prefix}rationale_max_tokens",
                help="Cap on the final rationale length.",
            )

        return {
            "n_branches": int(n_branches),
            "max_thought_tokens": int(max_thought_tokens),
            "thought_temperature": float(thought_temperature),
            "rationale_max_tokens": int(rationale_max_tokens),
        }

    def run(
        self,
        wrapper,
        prompt: str,
        task_type: str,
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from medical_llm_toolkit.explainers.reasoning import TreeOfThoughtExtractor

        tot = TreeOfThoughtExtractor(
            wrapper,
            n_branches=params["n_branches"],
            max_thought_tokens=params["max_thought_tokens"],
            thought_temperature=params["thought_temperature"],
            rationale_max_tokens=params["rationale_max_tokens"],
            verbose=False,
        )
        return tot.extract(prompt, task_type=task_type)

    def render_results(self, result: Dict[str, Any], ground_truth: Optional[str] = None):
        is_free = result.get("task_type") == "free"
        n_branches = len(result.get("thoughts") or [])

        if is_free:
            cols = st.columns(2)
            cols[0].metric("Task Type", "Free Response")
            cols[1].metric("Branches", n_branches)
        else:
            cols = st.columns(4)
            cols[0].metric("Answer", result.get("answer") or "-")
            if result.get("confidence") is not None:
                cols[1].metric("Confidence", f"{result['confidence']:.4f}")
            if ground_truth:
                correct = result.get("answer") == ground_truth
                cols[2].metric("Ground Truth", ground_truth, delta="✓" if correct else "✗")
            cols[3].metric("Branches", n_branches)

        thoughts = result.get("thoughts") or []
        scores = result.get("scores") or []
        best_idx = result.get("best_idx", 0)

        # Score chart
        if thoughts:
            st.subheader("Branch Scores (mean log-probability)")
            import plotly.graph_objects as go

            colors = [
                "#e89923" if i == best_idx else "#4a4a52"
                for i in range(len(thoughts))
            ]
            fig = go.Figure(
                go.Bar(
                    x=[f"Branch {i+1}" for i in range(len(thoughts))],
                    y=scores,
                    marker_color=colors,
                    text=[f"{s:.3f}" for s in scores],
                    textposition="auto",
                    textfont=dict(color="#e8e0d4"),
                )
            )
            fig.update_layout(
                height=240,
                margin=dict(l=0, r=20, t=10, b=0),
                yaxis_title="Mean log-prob (higher = more coherent)",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Branch text
        st.subheader("Reasoning Branches")
        for i, (thought, score) in enumerate(zip(thoughts, scores)):
            is_best = i == best_idx
            label = f"Branch {i+1}  |  score: {score:.4f}"
            if is_best:
                label += "   ★ selected"
            with st.expander(label, expanded=is_best):
                st.markdown(thought or "*(empty)*")

        # Final rationale
        rationale = result.get("rationale")
        if rationale:
            st.subheader("Final Rationale")
            st.markdown(rationale)

        # Option probabilities
        option_probs = result.get("option_probs")
        if option_probs:
            st.subheader("Option Probabilities")
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Bar(
                    x=list(option_probs.keys()),
                    y=list(option_probs.values()),
                    marker_color="#e89923",
                    text=[f"{v:.3f}" for v in option_probs.values()],
                    textposition="auto",
                    textfont=dict(color="#e8e0d4"),
                )
            )
            fig.update_layout(
                height=240,
                margin=dict(l=0, r=20, t=10, b=0),
                yaxis_title="P(option)",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)


register(TreeOfThoughtUI())
