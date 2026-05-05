from abc import ABC, abstractmethod
from typing import List, Tuple


class NERBackend(ABC):
    """
    Abstract interface for named-entity recognition backends.

    A NERBackend receives raw text and returns the character-level spans of
    detected entities.  The SemanticSplitter uses these spans to group
    multi-word entities into single atomic tokens before Shapley analysis.

    All concrete implementations must return character offsets into the
    *original* input string — not token indices from any internal tokenizer.
    """

    @abstractmethod
    def extract_entities(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect entities in ``text`` and return their character spans.

        Args:
            text: The raw input string to analyse.

        Returns:
            List of (start, end) tuples where text[start:end] is an entity.
            Spans may be unsorted; SemanticSplitter will sort and merge them.
        """


class SpaCyNERBackend(NERBackend):
    """
    NER backend backed by a spaCy pipeline.

    Works with any spaCy model including domain-specific ones such as
    ``en_core_sci_sm`` (scispaCy, biomedical) or ``en_core_web_sm`` (general).

    Args:
        model_name: spaCy model to load (must be installed separately).
                    Examples:
                      - "en_core_web_sm"    (general English)
                      - "en_core_sci_sm"    (scispaCy biomedical, requires scispacy)
                      - "en_ner_bc5cdr_md"  (scispaCy diseases & chemicals)
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spacy is required for SpaCyNERBackend. "
                "Install with: pip install spacy"
            )
        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' is not installed. "
                f"Install with: python -m spacy download {model_name}"
            )

    def extract_entities(self, text: str) -> List[Tuple[int, int]]:
        doc = self._nlp(text)
        return [(ent.start_char, ent.end_char) for ent in doc.ents]


class HuggingFaceNERBackend(NERBackend):
    """
    NER backend backed by any HuggingFace token-classification model.

    Uses the ``transformers`` NER pipeline with ``aggregation_strategy``
    so that subword tokens are merged into whole-word or word-group entities
    and character offsets into the original string are returned.

    Args:
        model_name: HuggingFace model ID for a token-classification model.
                    Examples:
                      - "dslim/bert-base-NER"                (general)
                      - "allenai/scibert_scivocab_uncased"   (scientific)
                      - "pruas/BENT-PubMedBERT-NER-disease"  (biomedical diseases)
        device: "cpu", "cuda", "cuda:0", etc.
        aggregation_strategy: How subword tokens are merged.
                    "simple"  — majority-vote merge (default, recommended)
                    "first"   — use the first subword's label
                    "average" — average subword scores
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        aggregation_strategy: str = "simple",
    ):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceNERBackend. "
                "Install with: pip install transformers"
            )

        self._pipeline = pipeline(
            "ner",
            model=model_name,
            device=device,
            aggregation_strategy=aggregation_strategy,
        )

    def extract_entities(self, text: str) -> List[Tuple[int, int]]:
        results = self._pipeline(text)
        return [(int(r["start"]), int(r["end"])) for r in results]
