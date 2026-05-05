from typing import Optional, List

import numpy as np

from ...token_shap.base import TextVectorizer
from .embedding_value import EmbeddingVectorizer


class HybridValueFunction(TextVectorizer):
    """
    Combines correctness-based scoring with semantic embedding similarity.

    Triggers the correctness branch in QATokenSHAP (via the ``correct_label``
    attribute), then blends two signals per coalition:

      score = alpha * correctness + (1 - alpha) * embedding_similarity

    where:
      - ``correctness`` comes from CorrectnessValueFunction logic (binary or prob)
      - ``embedding_similarity`` is the cosine similarity of the coalition's model
        response to the baseline response in the EmbeddingVectorizer's semantic space

    This is useful when you want both a sharp correctness signal (did the answer
    flip?) and a smooth semantic signal (how much did the response drift?).

    Args:
        correct_label: Ground truth answer label ("A", "B", "Yes", etc.)
        embedding_vectorizer: An EmbeddingVectorizer instance. The user controls
            which model it uses (general or domain-specific).
        mode: "binary" (1/0 correctness) or "prob" (probability of correct label).
        alpha: Weight for the correctness component.
               alpha=1.0 → pure correctness (same as CorrectnessValueFunction)
               alpha=0.0 → pure embedding similarity
               alpha=0.5 → equal blend (default)
    """

    def __init__(
        self,
        correct_label: str,
        embedding_vectorizer: EmbeddingVectorizer,
        mode: str = "binary",
        alpha: float = 0.5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.correct_label = correct_label
        self.embedding_vectorizer = embedding_vectorizer
        self.mode = mode
        self.alpha = alpha
        self._baseline_embedding: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Baseline registration (called by QATokenSHAP before the payoff loop)
    # ------------------------------------------------------------------

    def set_baseline(self, baseline_response: str) -> None:
        """
        Encode and store the baseline response embedding.

        QATokenSHAP calls this at the start of _get_df_per_combination when
        it detects a ``set_baseline`` method on the vectorizer.
        """
        vecs = self.embedding_vectorizer.vectorize([baseline_response])
        self._baseline_embedding = vecs[0]

    # ------------------------------------------------------------------
    # Payoff computation
    # ------------------------------------------------------------------

    def compute_payoff(
        self,
        pred: str,
        probs: Optional[dict],
        response: Optional[str] = None,
    ) -> float:
        """
        Blend correctness and embedding similarity into a single payoff score.

        Args:
            pred:     Model's predicted answer label.
            probs:    Dict of label → probability, or None.
            response: The raw text response for this coalition (used for the
                      embedding component). If None or baseline not set, the
                      embedding component falls back to 0.

        Returns:
            Weighted blend in approximately [0, 1].
        """
        correctness = self._compute_correctness(pred, probs)
        embedding_sim = self._compute_embedding_similarity(response)
        return self.alpha * correctness + (1.0 - self.alpha) * embedding_sim

    def _compute_correctness(self, pred: str, probs: Optional[dict]) -> float:
        if pred is None:
            raise RuntimeError(
                "Model wrapper did not store last_answer. "
                "Ensure model.generate() sets self.last_answer."
            )
        if self.mode == "binary":
            return 1.0 if pred == self.correct_label else 0.0
        elif self.mode == "prob":
            if probs is None:
                return 1.0 if pred == self.correct_label else 0.0
            return float(probs.get(self.correct_label, 0.0))
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def _compute_embedding_similarity(self, response: Optional[str]) -> float:
        """Cosine similarity of response to baseline, scaled to [0, 1]."""
        if response is None or self._baseline_embedding is None:
            return 0.0

        vec = self.embedding_vectorizer.vectorize([response])[0]
        base_norm = self._baseline_embedding / (np.linalg.norm(self._baseline_embedding) + 1e-9)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        cosine = float(np.dot(base_norm, vec_norm))
        # Scale from [-1, 1] to [0, 1] so it's on the same range as correctness
        return (cosine + 1.0) / 2.0

    # ------------------------------------------------------------------
    # TextVectorizer interface (unused in correctness path, required by base)
    # ------------------------------------------------------------------

    def vectorize(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 1), dtype=np.float32)

    def calculate_similarity(
        self, base_vector: np.ndarray, comparison_vectors: np.ndarray
    ) -> np.ndarray:
        return np.zeros(comparison_vectors.shape[0], dtype=np.float32)
