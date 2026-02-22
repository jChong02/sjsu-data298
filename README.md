# Medical LLM Interpretability Toolkit

A unified toolkit for explaining predictions from medical language models using multiple interpretability methods.

## Features

- **Model-Agnostic Wrapper**: Standardized interface for any HuggingFace medical LLM
- **Integrated Gradients**: Token-level attribution for model predictions
- **TokenSHAP**: SHAP values for medical QA tasks
- **Easy to Use**: Clean API with minimal boilerplate
- **Extensible**: Add new interpretability methods easily

## Installation

```bash
# Clone the repository
git clone https://github.com/jChong02/sjsu-data298.git
cd sjsu-data298

# Install in development mode
pip install -e .
```

## Project Structure

```
sjsu-data298/
├── TokenSHAP-QA/                        # TokenSHAP implementation
│   ├── token_shap/                      # Core TokenSHAP code
│   │   ├── base.py
│   │   ├── token_shap.py
│   │   ├── pixel_shap.py
│   │   ├── visualization.py
│   │   ├── image_utils.py
│   │   └── video_utils.py
│   └── tokenshap_extensions/            # QA-specific extensions
│       ├── extractors.py
│       ├── qa_tokenshap.py
│       └── value_functions/
│           └── correctness_aware.py
├── medical_llm_wrapper.py               # Model-agnostic wrapper
├── integrated_gradients.py              # IG base implementation
├── medical_ig_adapter.py                # IG adapter for medical wrapper
├── medical_lime_adapter.py              # LIME adapter for medical wrapper
├── medical_llm_wrapper_demo.ipynb       # Wrapper demo notebook
├── notebooks/                           # Jupyter demos
│   ├── demo_wrapper.ipynb
│   └── demo_ig.ipynb
├── examples/                            # Standalone examples
│   └── basic_usage.py
├── data/                                # Dataset files
│   ├── compiled_df.parquet
│   ├── mcq_df.parquet
│   └── yn_df.parquet
├── data-prep/                           # Data preparation notebooks
│   ├── preliminary_dataset_exploration.ipynb
│   └── preprocessing.ipynb
├── deprecated/                          # Deprecated code
├── requirements.txt                     # Python dependencies
└── setup.py                             # Package configuration
```

## Quick Start

### 1. Load a Medical LLM

```python
from medical_llm_wrapper import load_medical_llm

# Load model
wrapper = load_medical_llm(
    "FreedomIntelligence/Apollo-2B",
    device="cuda"
)

# Set task type
wrapper.set_task("mcq")  # or "yn" for Yes/No, "free" for open-ended
```

### 2. Get Predictions

```python
prompt = """Which vitamin deficiency causes scurvy?
A) Vitamin A
B) Vitamin B12
C) Vitamin C
D) Vitamin D

Answer:"""

answer = wrapper.generate(prompt)
print(f"Answer: {answer}")
```

### 3. Explain with Integrated Gradients

```python
from medical_ig_adapter import MedicalIntegratedGradients

ig = MedicalIntegratedGradients(wrapper)
result = ig.attribute(prompt, target_class="C")

print(f"Target probability: {result['target_probability']:.4f}")
print(f"Top tokens: {result['tokens'][:5]}")
```

### 4. Explain with TokenSHAP

```python
from tokenshap_adapter import MedicalTokenSHAP

shap = MedicalTokenSHAP(wrapper)
result = shap.explain(prompt, target_class="C", n_samples=100)

print(f"SHAP values: {result['shap_values']}")
```

### 5. Explain with LIME

```python
from medical_lime_adapter import MedicalLIME

lime = MedicalLIME(wrapper)

# Auto-detect predicted class as the target
result = lime.analyze(prompt)

# Or specify the class to explain, with color-coded visualization
result = lime.analyze(prompt, target_class="C", visualize=True)

print(f"Prediction: {result['prediction']}")
print(f"All option probs: {result['all_option_probs']}")  # {'A': 0.1, 'B': 0.1, 'C': 0.7, 'D': 0.1}
print(f"Top words: {result['top_words'][:5]}")            # [(word, score), ...] by |attribution|
print(f"Local model fit R²: {result['r_squared']:.3f}")
```

`MedicalLIME` perturbs the input prompt word-by-word, queries the model on each perturbation, and fits a local linear model to produce signed word-level attribution scores. Positive scores (red in visualization) indicate words that increase `P(target_class)`; negative scores (blue) indicate words that decrease it.

## Examples

See `examples/basic_usage.py` for a complete working example.

See `notebooks/` for interactive demos:
- `demo_wrapper.ipynb` - Wrapper functionality
- `demo_ig.ipynb` - Integrated Gradients explanations

## Supported Models

Works with any HuggingFace causal language model:
- **Medical LLMs**: MedGemma, BioMistral, BioMedLM, Apollo
- **General LLMs**: Llama, Mistral, GPT, etc.

## Citation

If you use this toolkit, please cite:

```bibtex
@software{medical_llm_toolkit,
  title = {Medical LLM Interpretability Toolkit},
  author = {DATA 298A Team},
  year = {2025},
  url = {https://github.com/jChong02/sjsu-data298}
}
```

## Contributing

This project integrates TokenSHAP from [TokenSHAP-QA](https://github.com/jChong02/TokenSHAP-QA).

## Acknowledgments

- TokenSHAP implementation from TokenSHAP-QA repository
- Built on HuggingFace Transformers
- Integrated Gradients based on Sundararajan et al. (2017)
