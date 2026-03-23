"""
Medical LLM Interpretability Toolkit

A unified toolkit for explaining medical language model predictions using
multiple interpretability methods including Integrated Gradients, TokenSHAP,
and LIME.
"""

from .wrapper import MedicalLLMWrapper, load_medical_llm
from .explainers.lime import MedicalLIME, visualize_lime_attributions
from .explainers.integrated_gradients import MedicalIntegratedGradients, visualize_attributions
from .explainers.tokenshap.extensions import QATokenSHAP, qa_extractor

__version__ = "0.1.0"

__all__ = [
    "MedicalLLMWrapper",
    "load_medical_llm",
    "MedicalLIME",
    "visualize_lime_attributions",
    "MedicalIntegratedGradients",
    "visualize_attributions",
    "QATokenSHAP",
    "qa_extractor",
]
