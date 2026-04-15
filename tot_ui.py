"""
tot_ui.py

Gradio UI for the MedicalLLMWrapper Tree-of-Thought reasoning demo.

Usage in Colab:
    !pip install gradio -q
    !python /content/sjsu-data298/tot_ui.py

Usage locally:
    pip install gradio
    python tot_ui.py
"""

import sys
import os
import re
import math

# Add project root to path so medical_llm_wrapper can be found
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio not installed. Run: pip install gradio")

import torch
from medical_llm_wrapper import load_medical_llm

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_wrapper = None

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLES = {
    "MCQ — Lung mass (diagnosis)": """\
A 55-year-old patient presents with persistent cough, hemoptysis, and unintentional weight loss.
Chest X-ray shows a mass in the right upper lobe. What is the most likely diagnosis?

A) Tuberculosis
B) Lung cancer
C) Pneumonia
D) Pulmonary embolism

Answer:""",

    "MCQ — Heparin thrombocytopenia": """\
A patient on long-term heparin therapy develops thrombocytopenia.
Which condition should be suspected?

A) Disseminated intravascular coagulation
B) Heparin-induced thrombocytopenia
C) Immune thrombocytopenic purpura
D) Thrombotic thrombocytopenic purpura

Answer:""",

    "MCQ — Diabetic foot ulcer": """\
A 45-year-old man with type 2 diabetes presents with a foot ulcer that has not healed
in 3 weeks. His HbA1c is 10.2%. Which of the following is the most important initial step?

A) Prescribe oral antibiotics
B) Optimize glycemic control
C) Schedule amputation
D) Apply topical corticosteroids

Answer:""",

    "MCQ — Preeclampsia management": """\
A 28-year-old woman is 32 weeks pregnant and presents with sudden severe headache,
visual disturbances, and blood pressure of 160/110 mmHg. What is the most appropriate management?

A) Immediate cesarean section
B) Administer magnesium sulfate and antihypertensives
C) Prescribe bed rest and follow up in one week
D) Perform lumbar puncture to rule out meningitis

Answer:""",

    "Y/N — ACE inhibitors": """\
ACE inhibitors are contraindicated in patients with bilateral renal artery stenosis.

A) Yes
B) No

Answer:""",

    "Y/N — Metformin renal": """\
Metformin is contraindicated in patients with severe renal impairment.

A) Yes
B) No

Answer:""",
}

MODEL_OPTIONS = [
    "FreedomIntelligence/Apollo-2B",
    "google/medgemma-4b-it",
    "BioMistral/BioMistral-7B",
    "stanford-crfm/BioMedLM",
]

# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------

def load_model(model_name, hf_token, device):
    global _wrapper
    token = hf_token.strip() or None
    try:
        _wrapper = load_medical_llm(model_name, device=device, token=token)
        info = _wrapper.get_model_info()
        return (
            f"✓ Loaded: {model_name}\n"
            f"  Parameters : {info['num_parameters']:,}\n"
            f"  Dtype      : {info['dtype']}\n"
            f"  Device     : {info['device']}"
        )
    except Exception as e:
        _wrapper = None
        return f"❌ Failed to load model:\n{e}"


def _parse_response(response):
    """Extract Answer, Thought, and Rationale from a generated response string."""
    answer = re.search(r"Answer:\s*(.*?)(?:\n|$)", response)
    thought = re.search(r"Thought:\s*(.*?)(?=\nRationale:|\Z)", response, re.DOTALL)
    rationale = re.search(r"Rationale:\s*(.*)", response, re.DOTALL)
    return (
        answer.group(1).strip() if answer else response.strip(),
        thought.group(1).strip() if thought else "",
        rationale.group(1).strip() if rationale else "",
    )


def generate(prompt, task_type, strategy, n_branches, max_thought_tokens):
    if _wrapper is None:
        return "❌ No model loaded.", "", "", [], "Load a model first."

    if not prompt.strip():
        return "❌ Prompt is empty.", "", "", [], ""

    _wrapper.set_task(task_type)
    _wrapper.set_mode("answer_rationale")
    _wrapper.set_reasoning_strategy(strategy)

    if strategy == "tot":
        _wrapper.configure_tot(
            n_branches=int(n_branches),
            max_thought_tokens=int(max_thought_tokens)
        )

    try:
        response = _wrapper.generate(prompt)
    except Exception as e:
        return f"❌ Generation error:\n{e}", "", "", [], ""

    answer, thought, rationale = _parse_response(response)

    # Build branch table and score summary for ToT
    branch_rows = []
    score_md = ""

    if strategy == "tot" and _wrapper.last_thoughts:
        thoughts = _wrapper.last_thoughts
        scores = _wrapper.last_thought_scores
        best_idx = scores.index(max(scores))

        for i, (t, s) in enumerate(zip(thoughts, scores)):
            selected = "★" if i == best_idx else ""
            preview = (t[:130] + "…") if len(t) > 130 else t
            branch_rows.append([f"Branch {i + 1}", f"{s:.4f}", preview, selected])

        spread = max(scores) - min(scores)
        score_md = (
            f"**Best score:** `{max(scores):.4f}`  |  "
            f"**Worst score:** `{min(scores):.4f}`  |  "
            f"**Spread:** `{spread:.4f}`"
        )
    elif strategy == "cot":
        score_md = "Strategy is CoT — no branches generated."

    return answer, thought, rationale, branch_rows, score_md


def load_example(example_name):
    return EXAMPLES.get(example_name, "")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Tree-of-Thought Medical QA", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# Tree-of-Thought Medical QA\n"
        "Interactive demo for `MedicalLLMWrapper` — compare Chain-of-Thought and "
        "Tree-of-Thought reasoning on medical questions."
    )

    # --- Model setup ---
    gr.Markdown("## 1. Load Model")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=MODEL_OPTIONS,
            value=MODEL_OPTIONS[0],
            label="Model",
            scale=3,
        )
        hf_token_input = gr.Textbox(
            label="HF Token (gated models only)",
            placeholder="hf_xxxx…",
            type="password",
            scale=2,
        )
        device_radio = gr.Radio(
            choices=["cuda", "cpu"],
            value="cuda",
            label="Device",
            scale=1,
        )
        load_btn = gr.Button("Load Model", variant="primary", scale=1)

    model_status = gr.Textbox(
        label="Model status", interactive=False, lines=4
    )

    # --- Config + Prompt ---
    gr.Markdown("## 2. Configure & Prompt")
    with gr.Row():

        with gr.Column(scale=1):
            task_radio = gr.Radio(
                choices=["mcq", "yn", "free"],
                value="mcq",
                label="Task type",
            )
            strategy_radio = gr.Radio(
                choices=["tot", "cot"],
                value="tot",
                label="Reasoning strategy",
            )
            n_branches_slider = gr.Slider(
                minimum=2, maximum=6, step=1, value=3,
                label="n_branches  (ToT only)",
            )
            max_thought_slider = gr.Slider(
                minimum=40, maximum=150, step=10, value=80,
                label="max_thought_tokens  (ToT only)",
            )

        with gr.Column(scale=2):
            example_dropdown = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                label="Load an example prompt",
                value=None,
            )
            prompt_input = gr.Textbox(
                label="Medical question",
                placeholder="Paste or type your question here…",
                lines=10,
            )
            generate_btn = gr.Button("Generate", variant="primary")

    # --- Results ---
    gr.Markdown("## 3. Results")
    answer_output = gr.Textbox(label="Answer", interactive=False, lines=1)

    with gr.Tabs():
        with gr.Tab("Output"):
            thought_output = gr.Textbox(
                label="Thought  (ToT: best branch selected from all candidates)",
                interactive=False,
                lines=5,
            )
            rationale_output = gr.Textbox(
                label="Rationale  (generated after answer is decided)",
                interactive=False,
                lines=6,
            )

        with gr.Tab("Branch Explorer  (ToT)"):
            score_summary_md = gr.Markdown("Run a ToT generation to see branch scores.")
            branch_table = gr.Dataframe(
                headers=["Branch", "Score", "Reasoning preview", "Selected"],
                label="All reasoning branches",
                interactive=False,
                wrap=True,
            )

    # --- Event wiring ---
    load_btn.click(
        fn=load_model,
        inputs=[model_dropdown, hf_token_input, device_radio],
        outputs=[model_status],
    )

    example_dropdown.change(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[prompt_input],
    )

    generate_btn.click(
        fn=generate,
        inputs=[
            prompt_input,
            task_radio,
            strategy_radio,
            n_branches_slider,
            max_thought_slider,
        ],
        outputs=[
            answer_output,
            thought_output,
            rationale_output,
            branch_table,
            score_summary_md,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True)
