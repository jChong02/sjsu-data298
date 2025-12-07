"""Interpretability methods for explaining model predictions."""

from .integrated_gradients import IntegratedGradients, IntegratedGradientsConfig
from .medical_ig_adapter import MedicalIntegratedGradients, explain_medical_prediction

__all__ = [
    "IntegratedGradients",
    "IntegratedGradientsConfig",
    "MedicalIntegratedGradients",
    "explain_medical_prediction",
]
