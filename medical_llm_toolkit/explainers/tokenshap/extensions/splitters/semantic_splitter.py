from typing import List, Tuple

from ...token_shap.token_shap import Splitter
from .ner_backends import NERBackend


class SemanticSplitter(Splitter):
    """
    A Splitter that groups multi-word named entities into single atomic tokens.

    Standard word-level splitting ("myocardial infarction" → two tokens) lets
    the Shapley analysis assign independent importance to each word, which is
    semantically meaningless for compound concepts. SemanticSplitter uses a
    pluggable NERBackend to detect entity boundaries and keeps each detected
    entity as one indivisible token; non-entity words are split by whitespace.

    The NERBackend is fully user-specified, so this splitter is not tied to any
    domain — swap in a general English model, a biomedical model, or any other
    HuggingFace token-classification model.

    Args:
        ner_backend: Any NERBackend implementation that returns character spans.

    Example::

        from extensions.splitters import SemanticSplitter, SpaCyNERBackend

        splitter = SemanticSplitter(SpaCyNERBackend("en_core_sci_sm"))
        tokens = splitter.split("The patient has myocardial infarction and hypertension.")
        # → ["The", "patient", "has", "myocardial infarction", "and", "hypertension", "."]
    """

    def __init__(self, ner_backend: NERBackend):
        self.ner_backend = ner_backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping or adjacent character spans.

        Spans must be sorted by start position before calling this method.
        """
        if not spans:
            return []
        merged = [list(spans[0])]
        for start, end in spans[1:]:
            if start < merged[-1][1]:
                # Overlapping: extend the current span if needed
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [tuple(s) for s in merged]

    # ------------------------------------------------------------------
    # Splitter interface
    # ------------------------------------------------------------------

    def split(self, text: str) -> List[str]:
        """
        Split ``text`` into tokens, treating each detected entity as one unit.

        Algorithm:
          1. Ask the NERBackend for entity character spans.
          2. Sort and merge overlapping spans.
          3. Walk through the text:
             - Text *between* entities is split by whitespace (word-level).
             - Text *inside* an entity span becomes a single token.
          4. Filter empty strings produced by leading/trailing/double whitespace.

        Returns:
            List of strings. Entities appear as single strings that may contain
            internal spaces (e.g. "myocardial infarction"). Non-entity words are
            single whitespace-delimited strings.
        """
        raw_spans = self.ner_backend.extract_entities(text)
        spans = self._merge_spans(sorted(raw_spans, key=lambda s: s[0]))

        tokens: List[str] = []
        cursor = 0

        for start, end in spans:
            # Words in the gap before this entity
            gap = text[cursor:start]
            tokens.extend(gap.split())

            # The entity itself as one atomic token
            entity = text[start:end].strip()
            if entity:
                tokens.append(entity)

            cursor = end

        # Remaining words after the last entity
        tokens.extend(text[cursor:].split())

        return [t for t in tokens if t]

    def join(self, tokens: List[str]) -> str:
        """
        Rejoin tokens with a single space.

        Multi-word entity tokens already carry their internal spaces, so a
        simple space-join reconstructs a well-formed sentence.

        Example::
            join(["The", "myocardial infarction", "was", "treated"])
            # → "The myocardial infarction was treated"
        """
        return " ".join(tokens)
