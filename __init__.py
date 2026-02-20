"""Interpretability methods for explaining model predictions."""

from .integrated_gradients import IntegratedGradients, IntegratedGradientsConfig
from .medical_ig_adapter import MedicalIntegratedGradients, explain_medical_prediction
from .medical_lime_adapter import MedicalLIME, explain_with_lime, visualize_lime_attributions

__all__ = [
    "IntegratedGradients",
    "IntegratedGradientsConfig",
    "MedicalIntegratedGradients",
    "explain_medical_prediction",
    "MedicalLIME",
    "explain_with_lime",
    "visualize_lime_attributions",
]
