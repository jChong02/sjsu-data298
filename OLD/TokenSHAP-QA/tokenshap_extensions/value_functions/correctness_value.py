from typing import List
import numpy as np
from ...token_shap.base import TextVectorizer


class CorrectnessValueFunction(TextVectorizer):
    """
    A pluggable value function that signals to QATokenSHAP that
    correctness-based scoring should be used instead of similarity.
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

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        TokenSHAP requires a 2D array.
        Vector values are irrelevant for correctness scoring.
        """
        return np.zeros((len(texts), 1), dtype=np.float32)

    def calculate_similarity(self, base_vec: np.ndarray, comparison_vecs: np.ndarray) -> np.ndarray:
        """
        This is never used directly because QATokenSHAP overrides similarity calculation.

        TokenSHAP still calls this method, so we return dummy values.
        """
        # Just return zeros – actual scoring happens in QATokenSHAP._compute_similarity
        return np.zeros(comparison_vecs.shape[0], dtype=np.float32)
