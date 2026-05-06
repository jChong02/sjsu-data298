"""
Medical LLM XAI Toolkit — Streamlit App

Run with:
    streamlit run app/main.py
"""

import sys
import os

# Ensure the repo root is on the path so medical_llm_toolkit is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np

# Register all explainers and reasoners on import
import app.explainers  # noqa: F401
import app.reasoning  # noqa: F401
from app.registry import get_all
from app.reasoning_registry import get_all as get_all_reasoners
from app.visualization import render_token_highlights_html, get_top_k_data, apply_plotly_theme

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical LLM XAI Toolkit",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — subtle warmth, better spacing, consistent feel
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Slightly warm dark background tint */
    .stApp {
        background-color: #1a1a20;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #16161c;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Headers — warm off-white with subtle accent */
    h1 { color: #e8e0d4 !important; }
    h2 { color: #d4cec4 !important; border-bottom: 2px solid #e8992322; padding-bottom: 8px; }
    h3 { color: #c8c2b8 !important; }

    /* Metric cards — subtle border and tint */
    [data-testid="stMetric"] {
        background: rgba(232, 153, 35, 0.04);
        border: 1px solid rgba(232, 153, 35, 0.12);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        color: #e8e0d4 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #999 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        color: #999;
    }
    .stTabs [aria-selected="true"] {
        color: #e89923 !important;
        border-bottom: 2px solid #e89923;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border-radius: 6px;
    }

    /* Buttons — accent color */
    .stButton > button[kind="primary"] {
        background-color: #e89923;
        border: none;
        color: #111;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #d4880f;
    }

    /* Text area and inputs — subtle border */
    .stTextArea textarea, .stTextInput input {
        border: 1px solid rgba(255,255,255,0.1) !important;
        background: rgba(255,255,255,0.03) !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------
EXAMPLE_QUESTIONS = {
    "yn": {
        "Custom": "",
        "Cholecystectomy safety": (
            "Is laparoscopic cholecystectomy not safe as a day case procedure?\n\n"
            "Answer Choices:\nA. Yes\nB. No"
        ),
        "Aspirin and cancer": (
            "Does aspirin reduce the risk of colorectal cancer?\n\n"
            "Answer Choices:\nA. Yes\nB. No"
        ),
    },
    "mcq": {
        "Custom": "",
        "Scurvy cause": (
            "Which vitamin deficiency causes scurvy?\n\n"
            "Answer Choices:\nA) Vitamin A\nB) Vitamin B12\nC) Vitamin C\nD) Vitamin D"
        ),
        "Pneumonia pathogen": (
            "What is the most common cause of community-acquired pneumonia?\n\n"
            "Answer Choices:\nA) Staphylococcus aureus\nB) Streptococcus pneumoniae\n"
            "C) Haemophilus influenzae\nD) Klebsiella pneumoniae"
        ),
    },
    "free": {
        "Custom": "",
        "Diabetes treatment": "What is the first-line treatment for type 2 diabetes?",
    },
}

PRESET_MODELS = {
    "Apollo 2B": "FreedomIntelligence/Apollo-2B",
    "MedGemma 4B": "google/medgemma-4b-it",
    "BioMistral 7B": "BioMistral/BioMistral-7B",
    "BioMedLM": "stanford-crfm/BioMedLM",
    "Custom": "",
}

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "wrapper" not in st.session_state:
    st.session_state.wrapper = None
if "model_status" not in st.session_state:
    st.session_state.model_status = "not_loaded"
if "generation_result" not in st.session_state:
    st.session_state.generation_result = None
if "explainer_results" not in st.session_state:
    st.session_state.explainer_results = {}
if "reasoning_results" not in st.session_state:
    st.session_state.reasoning_results = {}
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Attribution Methods"

# ---------------------------------------------------------------------------
# Sidebar — Model Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio(
        "Method Category",
        options=["Attribution Methods", "Reasoning Methods"],
        key="app_mode",
        help=(
            "Attribution methods (LIME, IG, TokenSHAP) explain WHICH input "
            "tokens drove the answer. Reasoning methods (CoT, ToT) elicit "
            "the model's step-by-step reasoning before answering."
        ),
        label_visibility="collapsed",
    )

    st.divider()

    st.header("Model Configuration")

    # HF token
    with st.expander("Hugging Face Token (for gated models)"):
        hf_token = st.text_input(
            "Token",
            type="password",
            key="hf_token",
            label_visibility="collapsed",
            help="Paste your token and press Enter to confirm.",
        )
        if hf_token:
            st.caption("Token set.")
        else:
            hf_token = None

    # Model selection
    model_choice = st.selectbox("Select Model", options=list(PRESET_MODELS.keys()))
    if model_choice == "Custom":
        model_id = st.text_input(
            "HuggingFace Model ID",
            placeholder="org/model-name",
            key="custom_model_id",
        )
    else:
        model_id = PRESET_MODELS[model_choice]

    # Task type
    task_type = st.selectbox(
        "Task Type",
        options=["yn", "mcq", "free"],
        format_func=lambda x: {"yn": "Yes / No", "mcq": "Multiple Choice", "free": "Free Response"}[x],
        help="Determines answer constraints and which XAI methods are available.",
    )

    # Device selection
    import torch as _torch
    _gpu_available = _torch.cuda.is_available()
    device_options = ["cuda", "cpu"] if _gpu_available else ["cpu"]
    device_choice = st.selectbox(
        "Device",
        options=device_options,
        help="GPU (cuda) is faster but requires VRAM. CPU works for smaller models.",
    )

    # Generation mode
    gen_mode = st.selectbox(
        "Generation Mode",
        options=["answer_rationale", "answer_only"],
        format_func=lambda x: {
            "answer_rationale": "Answer + Rationale",
            "answer_only": "Answer Only",
        }[x],
        help="'Answer + Rationale' generates a reasoning explanation. "
             "'Answer Only' returns just the predicted answer with confidence.",
    )

    st.divider()

    # Load / Clear buttons
    col_load, col_clear = st.columns(2)
    with col_load:
        load_clicked = st.button("Load Model", type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("Clear Model", use_container_width=True)

    # Handle load
    if load_clicked and model_id:
        st.session_state.model_status = "loading"
        st.session_state.generation_result = None
        st.session_state.explainer_results = {}
        st.session_state.reasoning_results = {}

        with st.spinner(f"Loading {model_id}..."):
            try:
                from medical_llm_toolkit.wrapper import MedicalLLMWrapper
                import torch

                wrapper = MedicalLLMWrapper(
                    model_id,
                    device=device_choice,
                    token=hf_token,
                )
                wrapper.set_task(task_type)
                wrapper.set_mode(gen_mode)

                st.session_state.wrapper = wrapper
                st.session_state.model_status = "ready"
            except Exception as e:
                st.session_state.model_status = "not_loaded"
                st.error(f"Failed to load model: {e}")

    # Handle clear
    if clear_clicked:
        if st.session_state.wrapper is not None:
            import gc
            import torch

            # Explicitly delete the model and tokenizer to free memory
            wrapper = st.session_state.wrapper
            if hasattr(wrapper, "model"):
                wrapper.model.cpu()  # move off GPU first
                del wrapper.model
            if hasattr(wrapper, "tokenizer"):
                del wrapper.tokenizer
            del wrapper
            st.session_state.wrapper = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        st.session_state.model_status = "not_loaded"
        st.session_state.generation_result = None
        st.session_state.explainer_results = {}
        st.session_state.reasoning_results = {}
        st.rerun()

    # Status indicator
    status = st.session_state.model_status
    if status == "ready" and st.session_state.wrapper is not None:
        st.success(f"Model ready: {st.session_state.wrapper.model_name}")
    elif status == "loading":
        st.info("Loading model...")
    else:
        st.warning("No model loaded")

    # Sync task type and gen mode if model is loaded
    if st.session_state.wrapper is not None:
        st.session_state.wrapper.set_task(task_type)
        st.session_state.wrapper.set_mode(gen_mode)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Medical LLM XAI Toolkit")
st.caption("Unified explainability for medical question-answering models")

# ---- Question Input ----
st.header("Question Input")

examples = EXAMPLE_QUESTIONS.get(task_type, {"Custom": ""})

example_choice = st.selectbox("Load Example Question", options=list(examples.keys()))
is_example = example_choice != "Custom"

if is_example:
    # Example selected — show as read-only display
    prompt = examples[example_choice]
    st.text_area(
        "Question Prompt",
        value=prompt,
        height=150,
        disabled=True,
    )
else:
    # Custom — editable text area
    prompt = st.text_area(
        "Enter Question Prompt",
        height=150,
        key="custom_prompt",
        placeholder="Type your question here...",
    )

ground_truth = None
if task_type in ("yn", "mcq"):
    gt_options = ["A", "B"] if task_type == "yn" else ["A", "B", "C", "D"]
    ground_truth = st.text_input(
        "Ground Truth Answer (optional)",
        placeholder=f"e.g., '{gt_options[0]}' or '{gt_options[-1]}'",
        key="ground_truth",
    )
    if ground_truth and ground_truth.strip().upper() not in gt_options:
        st.warning(f"Ground truth should be one of {gt_options}")
        ground_truth = None
    elif ground_truth:
        ground_truth = ground_truth.strip().upper()

# ---- Mode dispatch ----
st.divider()

if app_mode == "Attribution Methods":
    _show_attribution_section = True
    _show_reasoning_section = False
else:
    _show_attribution_section = False
    _show_reasoning_section = True

# ---- Generate (attribution mode only) ----
if _show_attribution_section:
    generate_clicked = st.button(
        "Generate",
        type="primary",
        disabled=st.session_state.model_status != "ready",
    )
else:
    generate_clicked = False

if generate_clicked and st.session_state.wrapper is not None:
    st.session_state.explainer_results = {}
    with st.spinner("Generating..."):
        try:
            wrapper = st.session_state.wrapper
            response = wrapper.generate(prompt)
            st.session_state.generation_result = {
                "response": response,
                "answer": wrapper.last_answer,
                "confidence": wrapper.last_confidence,
                "option_probs": wrapper.last_option_probs,
            }
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.session_state.generation_result = None

# ---- Display generation result (attribution mode only) ----
gen_result = st.session_state.generation_result
if _show_attribution_section and gen_result is not None:
    st.header("Model Output")

    col_answer, col_conf = st.columns([2, 1])
    with col_answer:
        st.markdown(f"```\n{gen_result['response']}\n```")
    with col_conf:
        if gen_result.get("confidence") is not None:
            st.metric("Confidence", f"{gen_result['confidence']:.4f}")
        if gen_result.get("option_probs"):
            import plotly.graph_objects as go

            probs = gen_result["option_probs"]
            fig = go.Figure(
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color="#e89923",
                    text=[f"{v:.3f}" for v in probs.values()],
                    textposition="auto",
                    textfont=dict(color="#e8e0d4"),
                )
            )
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=20, t=10, b=0),
                yaxis_title="P(option)",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Explanations ----
    if task_type in ("yn", "mcq"):
        st.header("Explanations")

        explainers = get_all()
        available = {
            name: exp
            for name, exp in explainers.items()
            if exp.is_available(task_type)
        }

        if not available:
            st.info("No explainer methods available for this task type.")
        else:
            # Determine target class for explainers
            target_class = gen_result.get("answer")
            if target_class and target_class not in ["A", "B", "C", "D"]:
                target_class = None

            tab_names = [exp.display_name for exp in available.values()]
            tab_names.append("Comparison")
            tabs = st.tabs(tab_names)

            for i, (method_name, explainer) in enumerate(available.items()):
                with tabs[i]:
                    st.caption(explainer.description)

                    # Config
                    with st.expander("Configuration", expanded=False):
                        params = explainer.render_config(f"{method_name}_")

                    # Run button
                    run_clicked = st.button(
                        f"Run {explainer.display_name}",
                        key=f"run_{method_name}",
                        type="primary",
                    )

                    if run_clicked:
                        with st.spinner(f"Running {explainer.display_name}..."):
                            try:
                                result = explainer.run(
                                    wrapper=st.session_state.wrapper,
                                    prompt=prompt,
                                    target_class=target_class,
                                    ground_truth=ground_truth,
                                    params=params,
                                )
                                st.session_state.explainer_results[method_name] = result
                            except Exception as e:
                                st.error(f"{explainer.display_name} failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                    # Display results if available
                    if method_name in st.session_state.explainer_results:
                        explainer.render_results(
                            st.session_state.explainer_results[method_name]
                        )

            # ---- Comparison tab ----
            with tabs[-1]:
                completed = st.session_state.explainer_results
                if len(completed) < 2:
                    st.info(
                        "Run at least 2 explainer methods to see a comparison. "
                        f"Currently completed: {len(completed)}"
                    )
                else:
                    st.subheader("Cross-Method Comparison")

                    # Shared legend (once, not per method)
                    from app.visualization import _POS_COLOR, _NEG_COLOR, _NEUTRAL_BG
                    pos_rgb = f"rgb({_POS_COLOR[0]},{_POS_COLOR[1]},{_POS_COLOR[2]})"
                    neg_rgb = f"rgb({_NEG_COLOR[0]},{_NEG_COLOR[1]},{_NEG_COLOR[2]})"
                    neu_rgb = f"rgb({_NEUTRAL_BG[0]},{_NEUTRAL_BG[1]},{_NEUTRAL_BG[2]})"
                    st.markdown(
                        f'<div style="font-size:12px;color:#999;display:flex;gap:16px;align-items:center;margin-bottom:16px;">'
                        f'<span style="background:{pos_rgb};padding:2px 10px;border-radius:3px;">&nbsp;</span> Supports'
                        f'<span style="background:{neg_rgb};padding:2px 10px;border-radius:3px;">&nbsp;</span> Against'
                        f'<span style="background:{neu_rgb};padding:2px 10px;border-radius:3px;color:#aaa;">&nbsp;</span> Neutral'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Side-by-side highlights (no per-column legend)
                    cols = st.columns(len(completed))
                    for col, (method_name, result) in zip(cols, completed.items()):
                        with col:
                            exp = available.get(method_name)
                            st.markdown(f"**{exp.display_name if exp else method_name}**")
                            tokens = result.get("words", result.get("tokens", []))
                            scores = result.get(
                                "word_attributions", result.get("attributions", [])
                            )
                            html = render_token_highlights_html(tokens, scores, show_legend=False)
                            st.markdown(html, unsafe_allow_html=True)

                    # Rank agreement analysis
                    st.subheader("Top Token Rank Agreement")
                    k = st.slider(
                        "Top-K tokens to compare",
                        min_value=3,
                        max_value=20,
                        value=10,
                        key="comparison_k",
                    )

                    all_top_k = {}
                    for method_name, result in completed.items():
                        tokens = result.get("words", result.get("tokens", []))
                        scores = np.array(
                            result.get(
                                "word_attributions",
                                result.get("attributions", []),
                            )
                        )
                        top_indices = np.argsort(np.abs(scores))[::-1][:k]
                        top_tokens = set()
                        for idx in top_indices:
                            t = tokens[idx] if idx < len(tokens) else f"[{idx}]"
                            top_tokens.add(t.replace("▁", " ").replace("Ġ", " ").strip())
                        all_top_k[method_name] = top_tokens

                    method_names = list(all_top_k.keys())
                    for i in range(len(method_names)):
                        for j in range(i + 1, len(method_names)):
                            m1, m2 = method_names[i], method_names[j]
                            e1 = available.get(m1)
                            e2 = available.get(m2)
                            n1 = e1.display_name if e1 else m1
                            n2 = e2.display_name if e2 else m2
                            overlap = all_top_k[m1] & all_top_k[m2]
                            pct = len(overlap) / k * 100
                            st.write(
                                f"**{n1}** vs **{n2}**: "
                                f"{len(overlap)}/{k} tokens overlap ({pct:.0f}%)"
                            )
                            if overlap:
                                st.caption(f"Shared: {', '.join(sorted(overlap))}")

                    # Consensus tokens (appear in all methods)
                    if len(all_top_k) >= 2:
                        consensus = set.intersection(*all_top_k.values())
                        if consensus:
                            st.success(
                                f"**Consensus tokens** (top-{k} in all methods): "
                                f"{', '.join(sorted(consensus))}"
                            )
                        else:
                            st.info(f"No tokens appear in the top-{k} of all methods.")

    elif task_type == "free":
        st.info(
            "XAI explanations are currently supported for Yes/No and "
            "Multiple Choice tasks only."
        )

# ---------------------------------------------------------------------------
# Reasoning Methods section
# ---------------------------------------------------------------------------
if _show_reasoning_section:
    st.header("Reasoning Methods")
    st.caption(
        "Each method runs its own generation pass. The model's predicted answer "
        "is produced as part of the reasoning, not from a separate Generate step."
    )

    if st.session_state.model_status != "ready" or st.session_state.wrapper is None:
        st.warning("Load a model first using the sidebar.")
    elif not (prompt and prompt.strip()):
        st.info("Enter or select a question above to run a reasoning method.")
    else:
        reasoners = get_all_reasoners()
        available_reasoners = {
            name: r for name, r in reasoners.items() if r.is_available(task_type)
        }

        if not available_reasoners:
            st.info("No reasoning methods available for this task type.")
        else:
            tab_names = [r.display_name for r in available_reasoners.values()]
            tabs = st.tabs(tab_names)

            for i, (method_name, reasoner) in enumerate(available_reasoners.items()):
                with tabs[i]:
                    st.caption(reasoner.description)

                    with st.expander("Configuration", expanded=False):
                        params = reasoner.render_config(f"{method_name}_")

                    run_clicked = st.button(
                        f"Run {reasoner.display_name}",
                        key=f"run_reasoning_{method_name}",
                        type="primary",
                    )

                    if run_clicked:
                        with st.spinner(f"Running {reasoner.display_name}..."):
                            try:
                                result = reasoner.run(
                                    wrapper=st.session_state.wrapper,
                                    prompt=prompt,
                                    task_type=task_type,
                                    ground_truth=ground_truth,
                                    params=params,
                                )
                                st.session_state.reasoning_results[method_name] = result
                            except Exception as e:
                                st.error(f"{reasoner.display_name} failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                    if method_name in st.session_state.reasoning_results:
                        reasoner.render_results(
                            st.session_state.reasoning_results[method_name],
                            ground_truth=ground_truth,
                        )
