"""
Medical LLM Interpretability Toolkit

A unified toolkit for explaining medical language model predictions using
multiple interpretability methods including Integrated Gradients, TokenSHAP,
LIME, and ELLUI5.
"""

from .model_wrappers.medical_llm_wrapper import MedicalLLMWrapper, load_medical_llm
from .interpretability_methods.integrated_gradients import IntegratedGradients, IntegratedGradientsConfig
from .interpretability_methods.medical_ig_adapter import MedicalIntegratedGradients, explain_medical_prediction

__version__ = "0.1.0"

__all__ = [
    "MedicalLLMWrapper",
    "load_medical_llm",
    "IntegratedGradients",
    "IntegratedGradientsConfig",
    "MedicalIntegratedGradients",
    "explain_medical_prediction",
]
