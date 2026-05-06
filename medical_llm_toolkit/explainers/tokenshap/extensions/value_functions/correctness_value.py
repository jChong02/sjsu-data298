from typing import List
import numpy as np
from ...token_shap.base import TextVectorizer


class CorrectnessValueFunction(TextVectorizer):
    """
    A pluggable value function for correctness-based scoring in QATokenSHAP.

    Instead of measuring response similarity (like TF-IDF), this scores each
    token combination based on whether removing tokens changes the model's
    ability to predict the correct answer.
    """

    def __init__(self, correct_label: str, mode: str = "binary"):
        """
        Args:
            correct_label: Ground truth answer label ("A", "B", "Yes", etc.)
            mode:
                "binary" → 1.0 if predicted == correct_label else 0.0
                "prob"   → probability assigned to correct_label
        """
        self.correct_label = correct_label
        self.mode = mode

    def compute_payoff(self, pred: str, probs: dict, response: str = None) -> float:
        """
        Compute the correctness payoff for a single model prediction.

        Args:
            pred:  The model's predicted answer label (e.g., "A", "B").
            probs: Dict mapping answer labels to probabilities, or None.

        Returns:
            float score based on the configured mode.
        """
        if pred is None:
            raise RuntimeError(
                "Model wrapper did not store last_answer. "
                "Ensure your model.generate() sets self.last_answer."
            )

        if self.mode == "binary":
            return 1.0 if pred == self.correct_label else 0.0
        elif self.mode == "prob":
            if probs is None:
                return 1.0 if pred == self.correct_label else 0.0
            return probs.get(self.correct_label, 0.0)
        else:
            raise ValueError(f"Unknown correctness scoring mode: {self.mode!r}")

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        TokenSHAP requires a 2D array.
        Vector values are irrelevant for correctness scoring.
        """
        return np.zeros((len(texts), 1), dtype=np.float32)

    def calculate_similarity(self, base_vec: np.ndarray, comparison_vecs: np.ndarray) -> np.ndarray:
        """
        TokenSHAP still calls this method, so we return dummy values.
        Actual scoring happens via compute_payoff().
        """
        return np.zeros(comparison_vecs.shape[0], dtype=np.float32)
