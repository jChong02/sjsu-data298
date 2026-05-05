"""Chain-of-Thought reasoning UI component."""

import streamlit as st
from typing import Dict, Any, Optional

from app.reasoning_registry import ReasoningUI, register
from app.visualization import apply_plotly_theme


class ChainOfThoughtUI(ReasoningUI):
    name = "cot"
    display_name = "Chain-of-Thought"
    description = (
        "Prompts the model to reason step-by-step before committing to an answer. "
        "Produces grounded reasoning rather than post-hoc rationalization."
    )
    supported_tasks = {"yn", "mcq", "free"}

    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        cols = st.columns(4)
        with cols[0]:
            strategy = st.selectbox(
                "Prompting Strategy",
                options=["structured", "zero_shot"],
                format_func=lambda x: {
                    "structured": "Structured (task-aware)",
                    "zero_shot": "Zero-shot (\"think step by step\")",
                }[x],
                key=f"{key_prefix}strategy",
                help="'Structured' uses MCQ/YN-specific templates. "
                     "'Zero-shot' uses a generic step-by-step prompt.",
            )
        with cols[1]:
            max_reasoning_tokens = st.number_input(
                "Max Reasoning Tokens",
                min_value=50,
                max_value=1024,
                value=350,
                step=25,
                key=f"{key_prefix}max_reasoning_tokens",
                help="Cap on how many tokens of reasoning to generate.",
            )
        with cols[2]:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.5,
                value=0.7,
                step=0.05,
                key=f"{key_prefix}temperature",
                help="Higher = more diverse reasoning. 0 = greedy.",
            )
        with cols[3]:
            top_p = st.slider(
                "Top-p",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key=f"{key_prefix}top_p",
                help="Nucleus sampling cutoff.",
            )

        return {
            "strategy": strategy,
            "max_reasoning_tokens": int(max_reasoning_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

    def run(
        self,
        wrapper,
        prompt: str,
        task_type: str,
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from medical_llm_toolkit.explainers.reasoning import ChainOfThoughtExtractor

        cot = ChainOfThoughtExtractor(
            wrapper,
            strategy=params["strategy"],
            max_reasoning_tokens=params["max_reasoning_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            verbose=False,
        )
        return cot.extract(prompt, task_type=task_type)

    def render_results(self, result: Dict[str, Any], ground_truth: Optional[str] = None):
        is_free = result.get("task_type") == "free"

        # Header metrics — for constrained tasks the answer is a single letter,
        # so a metric widget makes sense. For free responses the answer is prose,
        # which we render as its own block below.
        if is_free:
            cols = st.columns(2)
            cols[0].metric("Task Type", "Free Response")
            if result.get("confidence") is not None:
                cols[1].metric("Confidence", f"{result['confidence']:.4f}")
        else:
            cols = st.columns(4)
            cols[0].metric("Answer", result.get("answer") or "—")
            if result.get("confidence") is not None:
                cols[1].metric("Confidence", f"{result['confidence']:.4f}")
            if ground_truth:
                correct = result.get("answer") == ground_truth
                cols[2].metric("Ground Truth", ground_truth, delta="✓" if correct else "✗")
            cols[3].metric("Task Type", result.get("task_type", "—"))

        # Reasoning steps
        st.subheader("Reasoning Steps")
        steps = result.get("steps") or []
        if steps:
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}.** {step}")
        else:
            st.markdown(result.get("reasoning") or "(no reasoning produced)")

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

        # Raw reasoning text
        with st.expander("Raw model output", expanded=False):
            st.code(result.get("reasoning") or "")
            st.caption("Prompt sent to model:")
            st.code(result.get("cot_prompt") or "")


register(ChainOfThoughtUI())
