# Medical LLM XAI Toolkit

A unified explainability and reasoning toolkit for medical question-answering language models. Combines four attribution methods (LIME, Integrated Gradients, TokenSHAP, ELI5) and two reasoning methods (Chain-of-Thought, Tree-of-Thought) into a single interactive Streamlit interface, enabling users to generate, compare, and analyze explanations across methods.

## Scope

The XAI and reasoning methods implemented here are **topic-agnostic**. LIME, Integrated Gradients, TokenSHAP, ELI5, Chain-of-Thought, and Tree-of-Thought all work on any HuggingFace causal LM and any QA prompt (none of them encode biomedical assumptions).

As an SJSU DATA 298 project, we chose to ground the work in **biomedical question-answering**, a high-stakes domain where interpretability matters. The following pieces of the repository are biomedical *by choice*, not by necessity:

- The bundled dataset (`data/compiled_df.parquet`, drawn from medical QA sources)
- The four preset models in the sidebar (Apollo-2B, BioMistral-7B, MedGemma-4B, BioMedLM)
- The pre-trained ELI5 surrogates shipped in `medical_llm_toolkit/eli5_surrogates/`
- The example prompts in the UI

To apply the same methods to a different domain, swap in any HuggingFace causal LM, feed the wrapper any QA prompt, and (for ELI5) train new surrogates on your own corpus with [`notebooks/train_eli5_surrogates.ipynb`](notebooks/train_eli5_surrogates.ipynb).

## Features

- **Model-agnostic wrapper** - standardized interface for any HuggingFace causal LLM with constrained generation, confidence extraction, and three task modes (Y/N, MCQ, free response).
- **Four attribution methods** - LIME, Integrated Gradients, TokenSHAP (with QA-aware perturbation, correctness/embedding/hybrid value functions, and semantic NER-based splitters), and ELI5 (TF-IDF + Logistic Regression surrogate with native per-class feature attributions).
- **Two reasoning methods** - Chain-of-Thought (CoT) and Tree-of-Thought (ToT) prompting strategies.
- **Pre-trained ELI5 surrogates** - bundles ship for the four preset LLMs so ELI5 explanations are instant; a notebook trains new ones for custom models.
- **Interactive Streamlit UI** - per-method configuration, cross-method token-overlap comparison, and a separate Reasoning Methods mode.
- **Unified evaluation framework** - single CLI script for faithfulness (deletion AUC) and stability (overlap@k) evaluation across models and methods.
- **Extensible plugin architecture** - add a new XAI method as a single `ExplainerUI` subclass; it auto-registers as a new tab.

## Project Structure

```
sjsu-data298/
├── medical_llm_toolkit/                # Core library
│   ├── __init__.py
│   ├── wrapper.py                      # MedicalLLMWrapper
│   ├── cli.py                          # `medxai` console-script launcher
│   ├── eli5_surrogates/                # Pre-trained ELI5 .pkl bundles (per preset model + task)
│   └── explainers/
│       ├── lime.py                     # MedicalLIME
│       ├── integrated_gradients.py     # MedicalIntegratedGradients
│       ├── eli5.py                     # MedicalELI5
│       ├── tokenshap/
│       │   ├── token_shap/             # Upstream TokenSHAP (untouched)
│       │   └── extensions/             # QA-aware perturbation + value functions + semantic splitter
│       │       ├── qa_tokenshap.py     # QATokenSHAP - keeps Answer Choices: static
│       │       ├── extractors.py       # qa_extractor - splits prompt into question + suffix
│       │       ├── value_functions/
│       │       │   ├── correctness_value.py     # binary or prob correctness payoff
│       │       │   ├── embedding_value.py        # cosine similarity over HF encoder embeddings
│       │       │   └── hybrid_value.py           # alpha-weighted blend of the above
│       │       └── splitters/
│       │           ├── semantic_splitter.py     # NER-based atomic-entity splitter
│       │           └── ner_backends.py          # spaCy / HuggingFace NER backends
│       └── reasoning/
│           ├── chain_of_thought.py     # ChainOfThoughtExtractor
│           └── tree_of_thought.py      # TreeOfThoughtExtractor
├── app/                                # Streamlit UI
│   ├── main.py                         # App entry point (mode toggle: Attribution / Reasoning)
│   ├── registry.py                     # Attribution-method plugin registry
│   ├── reasoning_registry.py           # Reasoning-method plugin registry
│   ├── visualization.py                # Token highlights + Plotly helpers
│   ├── explainers/                     # Per-method UI plugins
│   │   ├── lime_ui.py
│   │   ├── ig_ui.py
│   │   ├── tokenshap_ui.py
│   │   └── eli5_ui.py
│   └── reasoning/
│       ├── cot_ui.py
│       └── tot_ui.py
├── eval/
│   └── run_evaluation.py               # Unified faithfulness + stability evaluation
├── notebooks/
│   ├── demo_wrapper.ipynb              # Wrapper API walk-through
│   ├── demo_lime.ipynb
│   ├── demo_ig.ipynb
│   ├── demo_ELI5.ipynb
│   ├── medical_llm_wrapper_demo.ipynb  # Standalone end-to-end demo
│   └── train_eli5_surrogates.ipynb     # Train ELI5 .pkl bundles for any model
├── data/                               # Dataset parquet files
├── examples/basic_usage.py
├── pyproject.toml                      # Build config + dependencies (single source of truth)
├── LICENSE                             # MIT
└── run.bat                             # Windows convenience launcher (calls `medxai`)
```

## Quick start

### Installation

```bash
git clone https://github.com/jChong02/sjsu-data298.git
cd sjsu-data298
pip install -e .
```

`pyproject.toml` declares all runtime dependencies (torch, transformers, scikit-learn, eli5, streamlit, plotly, etc.) as a single source of truth, so `pip install -e .` is enough to install everything.

#### Optional extras

```bash
pip install -e ".[spacy]"                  # spaCy NER splitter for TokenSHAP
pip install -e ".[sentence-transformers]"  # SBERT backend for upstream paths
pip install -e ".[all]"                    # all optional extras
```

If you install the spaCy extra, you also need to download a model:

```bash
python -m spacy download en_core_web_sm
```

### Launch the UI

After install, the toolkit registers a cross-platform `medxai` console command:

```bash
medxai
```

Equivalent fallbacks:

```bash
# Windows convenience launcher
run.bat

# Or directly invoke streamlit
streamlit run app/main.py
```

Then open `localhost:8501` in your browser.

### Sidebar mode toggle

The sidebar lets you switch between **Attribution Methods** (LIME / IG / TokenSHAP / ELI5 + Comparison) and **Reasoning Methods** (CoT / ToT). The Generate button is part of the attribution flow; reasoning methods run their own generation pass.

## Programmatic usage

### Wrapper

```python
from medical_llm_toolkit import MedicalLLMWrapper

wrapper = MedicalLLMWrapper("FreedomIntelligence/Apollo-2B", device="cuda")
wrapper.set_task("yn")
wrapper.set_mode("answer_only")

response = wrapper.generate(
    "Does aspirin reduce the risk of colorectal cancer?\n\n"
    "Answer Choices:\nA. Yes\nB. No"
)
print(response, wrapper.last_confidence, wrapper.last_option_probs)
```

### LIME

```python
from medical_llm_toolkit import MedicalLIME

lime = MedicalLIME(wrapper, n_samples=500, kernel_width=0.75)
result = lime.analyze(prompt, target_class="A")
print(result["top_words"][:5])
```

### Integrated Gradients

```python
from medical_llm_toolkit import MedicalIntegratedGradients

ig = MedicalIntegratedGradients(wrapper, n_steps=50)
result = ig.attribute(prompt, target_class="A", return_convergence_delta=True)
print(result["convergence_delta"])
```

### TokenSHAP - three configurable extensions

`QATokenSHAP` keeps the answer-choices block static during perturbation. Three pluggable components let you adapt the analysis to QA tasks:

```python
from medical_llm_toolkit.explainers.tokenshap.token_shap.token_shap import StringSplitter
from medical_llm_toolkit.explainers.tokenshap.extensions.qa_tokenshap import QATokenSHAP
from medical_llm_toolkit.explainers.tokenshap.extensions.value_functions import (
    CorrectnessValueFunction, EmbeddingVectorizer, HybridValueFunction,
)
from medical_llm_toolkit.explainers.tokenshap.extensions.splitters import (
    SemanticSplitter, SpaCyNERBackend,
)

# (1) Correctness value function - measures contribution to the correct answer.
vec = CorrectnessValueFunction(correct_label="A", mode="prob")

# (2) Embedding similarity - semantic response similarity (use any HF encoder).
# vec = EmbeddingVectorizer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

# (3) Hybrid - alpha-blend of correctness and embedding similarity.
# vec = HybridValueFunction(
#     correct_label="A",
#     embedding_vectorizer=EmbeddingVectorizer(),
#     mode="prob",
#     alpha=0.5,
# )

# Optional: semantic NER-based splitter that groups multi-word medical entities
# as atomic tokens (so "myocardial infarction" stays as one token, not two).
splitter = SemanticSplitter(SpaCyNERBackend("en_core_web_sm"))
# Or fall back to plain whitespace splitting:
# splitter = StringSplitter()

wrapper.set_mode("answer_only")
ts = QATokenSHAP(model=wrapper, splitter=splitter, vectorizer=vec)
df = ts.analyze(prompt, sampling_ratio=0.5, max_combinations=100)
```

### ELI5

```python
from medical_llm_toolkit import MedicalELI5

# Auto-load the pre-trained bundle for this (model, task)
explainer = MedicalELI5.from_disk("FreedomIntelligence/Apollo-2B", "mcq")
result = explainer.explain(prompt, kind="mimic", top=20)
print(result["predicted_class"], result["heldout_score"], result["delta_above_prior"])

# Train new bundles for custom models - see notebooks/train_eli5_surrogates.ipynb
```

### Chain-of-Thought / Tree-of-Thought

```python
from medical_llm_toolkit.explainers.reasoning import ChainOfThoughtExtractor, TreeOfThoughtExtractor

cot = ChainOfThoughtExtractor(wrapper, n_thoughts=3)
out = cot.extract(prompt, task_type="mcq")
print(out["final_answer"], out["thoughts"])

tot = TreeOfThoughtExtractor(wrapper, n_thoughts=3, n_branches=2)
out = tot.extract(prompt, task_type="mcq")
print(out["best_thought"])
```

## Evaluation

```bash
# All XAI methods on Apollo-2B, faithfulness on yn + mcq
python eval/run_evaluation.py --models Apollo-2B --eval-type faithfulness --tasks yn mcq

# Stability on Y/N
python eval/run_evaluation.py --models Apollo-2B --eval-type stability --tasks yn

# Cross-model comparison on Y/N faithfulness
python eval/run_evaluation.py --models Apollo-2B BioMistral-7B --eval-type faithfulness --tasks yn
```

CSV results land in `eval/results/`.

## Supported models

Tested with the four presets that ship with pre-trained ELI5 surrogates:

| Preset | HF ID | Notes |
|---|---|---|
| Apollo 2B | `FreedomIntelligence/Apollo-2B` | Smallest, fastest |
| BioMistral 7B | `BioMistral/BioMistral-7B` | |
| MedGemma 4B | `google/medgemma-4b-it` | Gated - set `HF_TOKEN` env var |
| BioMedLM | `stanford-crfm/BioMedLM` | |

Any HuggingFace causal LM works for the wrapper, LIME, IG, TokenSHAP, and CoT/ToT. ELI5 needs a pre-trained surrogate bundle - for non-preset models, run [`notebooks/train_eli5_surrogates.ipynb`](notebooks/train_eli5_surrogates.ipynb).

## Adding a new XAI method

1. Create `app/explainers/your_method_ui.py`:

```python
from app.registry import ExplainerUI, register

class YourMethodUI(ExplainerUI):
    name = "your_method"
    display_name = "Your Method"
    description = "One-line description."
    supported_tasks = {"yn", "mcq"}      # or {"yn", "mcq", "free"}

    def render_config(self, key_prefix):
        return {"param": value}

    def run(self, wrapper, prompt, target_class, ground_truth, params):
        return {"tokens": [...], "attributions": [...]}

    def render_results(self, result):
        ...

register(YourMethodUI())
```

2. Add `from . import your_method_ui` to `app/explainers/__init__.py`.

The new method automatically appears as a tab in the UI. The Comparison tab picks it up if its result dict has `tokens` (or `words`) and `attributions` (or `word_attributions`); ELI5 is excluded from the comparison because its output shape is different.

## Team

**SJSU DATA 298A / 298B - Team 8** (Fall 2025 / Spring 2026)

- Jeff Chong
- Anne Ha
- Jiyoon Lee
- Matthew Leffler
- Nairui Liu

## Acknowledgments

- TokenSHAP - Karczmarz et al. ([upstream repo](https://github.com/GenAISHAP/TokenSHAP))
- LIME - Ribeiro et al. (2016)
- Integrated Gradients - Sundararajan et al. (2017)
- ELI5 - [eli5 library](https://github.com/TeamHG-Memex/eli5)
- Built on HuggingFace Transformers
