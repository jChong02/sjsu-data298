from typing import List

import numpy as np

from ...token_shap.base import TextVectorizer


class EmbeddingVectorizer(TextVectorizer):
    """
    Semantic embedding vectorizer using any HuggingFace encoder model.

    Replaces TF-IDF cosine similarity with dense vector similarity, making the
    value function sensitive to synonym relationships and semantic paraphrases
    that bag-of-words approaches miss (e.g. "MI" ≈ "myocardial infarction").

    Plugs into the standard TokenSHAP similarity path — no changes to QATokenSHAP
    needed unless combined with CorrectnessValueFunction via HybridValueFunction.

    Args:
        model_name: Any HuggingFace encoder model ID.
                    Examples:
                      - "sentence-transformers/all-MiniLM-L6-v2"  (general)
                      - "neuml/pubmedbert-base-embeddings"         (biomedical)
                      - "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        device: "cpu", "cuda", "cuda:0", etc.
        batch_size: Number of texts to encode per forward pass.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for EmbeddingVectorizer. "
                "Install with: pip install transformers"
            )
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _mean_pool(token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        """Mean-pool token embeddings, ignoring padding positions."""
        import torch
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into dense vectors via mean-pooled last hidden states.

        Args:
            texts: List of strings (first element is baseline, rest are perturbations
                   when called from the standard TokenSHAP pipeline).

        Returns:
            np.ndarray of shape (len(texts), hidden_dim).
        """
        import torch

        if self._model is None:
            self._load()

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def calculate_similarity(
        self, base_vector: np.ndarray, comparison_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Cosine similarity between the baseline embedding and each comparison embedding.

        Returns:
            np.ndarray of shape (n_comparisons,) with values in [-1, 1].
        """
        base_norm = base_vector / (np.linalg.norm(base_vector) + 1e-9)
        comp_norms = comparison_vectors / (
            np.linalg.norm(comparison_vectors, axis=1, keepdims=True) + 1e-9
        )
        return comp_norms @ base_norm
