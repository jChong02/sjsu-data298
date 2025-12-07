# Medical LLM Interpretability Toolkit

A unified toolkit for explaining predictions from medical language models using multiple interpretability methods.

## 🚀 Features

- **Model-Agnostic Wrapper**: Standardized interface for any HuggingFace medical LLM
- **Integrated Gradients**: Token-level attribution for model predictions
- **TokenSHAP**: SHAP values for medical QA tasks
- **Easy to Use**: Clean API with minimal boilerplate
- **Extensible**: Add new interpretability methods easily

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/jChong02/sjsu-data298.git
cd sjsu-data298

# Install in development mode
pip install -e .
```

## 🏗️ Project Structure

```
sjsu-data298/
├── medical_llm_toolkit/          # Main package
│   ├── model_wrappers/           # LLM interface wrappers
│   │   ├── medical_llm_wrapper.py
│   │   └── tokenshap_adapter.py
│   ├── interpretability_methods/ # XAI implementations
│   │   ├── integrated_gradients.py
│   │   └── medical_ig_adapter.py
│   ├── token_shap/              # TokenSHAP (from TokenSHAP-QA)
│   ├── tokenshap_extensions/    # TokenSHAP extensions
│   └── utils/                   # Utilities
├── notebooks/                   # Jupyter demos
│   ├── demo_wrapper.ipynb
│   └── demo_ig.ipynb
├── examples/                    # Standalone examples
│   └── basic_usage.py
└── tests/                       # Unit tests

```

## 🎯 Quick Start

### 1. Load a Medical LLM

```python
from medical_llm_toolkit import load_medical_llm

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
from medical_llm_toolkit import MedicalIntegratedGradients

ig = MedicalIntegratedGradients(wrapper)
result = ig.attribute(prompt, target_class="C")

print(f"Target probability: {result['target_probability']:.4f}")
print(f"Top tokens: {result['tokens'][:5]}")
```

### 4. Explain with TokenSHAP

```python
from medical_llm_toolkit.model_wrappers import MedicalTokenSHAP

shap = MedicalTokenSHAP(wrapper)
result = shap.explain(prompt, target_class="C", n_samples=100)

print(f"SHAP values: {result['shap_values']}")
```

## 📚 Examples

See `examples/basic_usage.py` for a complete working example.

See `notebooks/` for interactive demos:
- `demo_wrapper.ipynb` - Wrapper functionality
- `demo_ig.ipynb` - Integrated Gradients explanations

## 🔧 Supported Models

Works with any HuggingFace causal language model:
- **Medical LLMs**: MedGemma, BioMistral, BioMedLM, Apollo
- **General LLMs**: Llama, Mistral, GPT, etc.

## 📖 Citation

If you use this toolkit, please cite:

```bibtex
@software{medical_llm_toolkit,
  title = {Medical LLM Interpretability Toolkit},
  author = {DATA 298A Team},
  year = {2025},
  url = {https://github.com/jChong02/sjsu-data298}
}
```

## 🤝 Contributing

This project integrates TokenSHAP from [TokenSHAP-QA](https://github.com/jChong02/TokenSHAP-QA).

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- TokenSHAP implementation from TokenSHAP-QA repository
- Built on HuggingFace Transformers
- Integrated Gradients based on Sundararajan et al. (2017)
