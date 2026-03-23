# Medical LLM XAI Toolkit

A unified explainability toolkit for medical question-answering language models. Combines multiple XAI methods (LIME, Integrated Gradients, TokenSHAP, ELI5) into a single interactive interface, enabling users to generate, compare, and analyze token-level explanations across methods.

## Features

- **Model-Agnostic Wrapper**: Standardized interface for any HuggingFace causal LLM with constrained generation and confidence extraction
- **Multiple XAI Methods**: LIME, Integrated Gradients, TokenSHAP (with QA-aware extensions), ELI5
- **Interactive UI**: Streamlit-based web app with per-method configuration, cross-method comparison, and consensus analysis
- **Unified Evaluation**: Single script for faithfulness (deletion AUC) and stability (overlap@k) evaluation across models and methods
- **Extensible**: Modular architecture — add new XAI methods without modifying core application code

## Project Structure

```
sjsu_data298/
├── medical_llm_toolkit/              # Core library
│   ├── __init__.py
│   ├── wrapper.py                    # MedicalLLMWrapper
│   └── explainers/
│       ├── lime.py                   # LIME adapter
│       ├── integrated_gradients.py   # IG adapter
│       └── tokenshap/
│           ├── token_shap/           # Upstream TokenSHAP (untouched)
│           └── extensions/           # QA-aware perturbation + correctness value function
│               ├── qa_tokenshap.py
│               ├── extractors.py
│               └── value_functions/
│                   └── correctness_value.py
├── app/                              # Streamlit UI
│   ├── main.py                       # App entry point
│   ├── registry.py                   # Explainer plugin system
│   ├── visualization.py             # Token highlights + chart helpers
│   └── explainers/                   # Per-method UI components
│       ├── lime_ui.py
│       ├── ig_ui.py
│       └── tokenshap_ui.py
├── eval/                             # Evaluation framework
│   └── run_evaluation.py             # Unified faithfulness + stability evaluation
├── notebooks/                        # Jupyter demos + experimentation
├── data/                             # Dataset files
├── setup.py
├── requirements.txt
└── run.bat                           # Windows launcher
```

## Quick Start

### Installation

```bash
git clone https://github.com/jChong02/sjsu-data298.git
cd sjsu-data298
pip install -e .
pip install streamlit plotly
```

### Launch the UI

```bash
# Windows
run.bat

# Or directly
streamlit run app/main.py
```

Then open `localhost:8501` in your browser.

### Programmatic Usage

```python
from medical_llm_toolkit.wrapper import MedicalLLMWrapper

# Load model
wrapper = MedicalLLMWrapper("FreedomIntelligence/Apollo-2B", device="cuda")
wrapper.set_task("yn")
wrapper.set_mode("answer_rationale")

# Generate
response = wrapper.generate(
    "Does aspirin reduce the risk of colorectal cancer?\n\n"
    "Answer Choices:\nA. Yes\nB. No"
)
print(response)
print(wrapper.last_option_probs)
```

```python
# LIME
from medical_llm_toolkit.explainers.lime import MedicalLIME

lime = MedicalLIME(wrapper, n_samples=100, kernel_width=0.75)
result = lime.analyze(prompt, target_class="A", visualize=True)
```

```python
# Integrated Gradients
from medical_llm_toolkit.explainers.integrated_gradients import MedicalIntegratedGradients

ig = MedicalIntegratedGradients(wrapper, n_steps=50)
result = ig.attribute(prompt, target_class="A")
```

```python
# TokenSHAP (with correctness-aware value function)
from medical_llm_toolkit.explainers.tokenshap.token_shap.token_shap import StringSplitter
from medical_llm_toolkit.explainers.tokenshap.extensions.qa_tokenshap import QATokenSHAP
from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions.correctness_value import CorrectnessValueFunction

wrapper.set_mode("answer_only")
vec = CorrectnessValueFunction(correct_label="A", mode="prob")
ts = QATokenSHAP(model=wrapper, splitter=StringSplitter(), vectorizer=vec)
results_df = ts.analyze(prompt, sampling_ratio=0.5, max_combinations=100)
```

## Evaluation

Run faithfulness and stability evaluation across models and methods:

```bash
# All methods on Apollo-2B, Y/N + MCQ faithfulness
python eval/run_evaluation.py --models Apollo-2B --eval-type faithfulness --tasks yn mcq

# Stability on Y/N
python eval/run_evaluation.py --models Apollo-2B --eval-type stability --tasks yn

# Cross-model comparison
python eval/run_evaluation.py --models Apollo-2B BioMistral-7B --eval-type faithfulness --tasks yn

# Specific methods only
python eval/run_evaluation.py --methods lime ig --tasks yn --eval-type faithfulness
```

Results are saved as CSVs in `eval/results/`.

## Supported Models

Tested with:
- **Apollo-2B** (FreedomIntelligence/Apollo-2B)
- **BioMistral-7B** (BioMistral/BioMistral-7B)
- **MedGemma-4B** (google/medgemma-4b-it) — requires HF token
- **BioMedLM** (stanford-crfm/BioMedLM)

Works with any HuggingFace causal language model.

## Adding a New XAI Method

1. Create `app/explainers/your_method_ui.py`:

```python
from app.registry import ExplainerUI, register

class YourMethodUI(ExplainerUI):
    name = "your_method"
    display_name = "Your Method"
    description = "One-line description."
    supported_tasks = {"yn", "mcq"}

    def render_config(self, key_prefix):
        # Streamlit widgets for method params
        return {"param": value}

    def run(self, wrapper, prompt, target_class, ground_truth, params):
        # Run your method, return result dict with 'tokens' and 'attributions'
        return {"tokens": [...], "attributions": [...]}

    def render_results(self, result):
        # Render with Streamlit
        pass

register(YourMethodUI())
```

2. Add `from . import your_method_ui` to `app/explainers/__init__.py`.

The new method automatically appears as a tab in the UI.

## Acknowledgments

- TokenSHAP: Karczmarz et al. — [TokenSHAP](https://github.com/KarczmarzJakub/TokenSHAP)
- LIME: Ribeiro et al. (2016)
- Integrated Gradients: Sundararajan et al. (2017)
- Built on HuggingFace Transformers
