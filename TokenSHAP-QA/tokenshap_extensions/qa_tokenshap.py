# qa_tokenshap.py

from typing import Optional, Callable, List, Dict
from ..token_shap.base import ModelBase, TextVectorizer
from ..token_shap.token_shap import TokenSHAP, Splitter
from .extractors import qa_extractor

class QATokenSHAP(TokenSHAP):
    """
    Extension of TokenSHAP for structured question-answering prompts.
    Only the question segment is perturbed during Monte Carlo sampling,
    while the answer segment remains fixed.
    """
    def __init__(
        self,
        model: ModelBase,
        splitter: Splitter,
        vectorizer: Optional[TextVectorizer] = None,
        debug: bool = False,
        section_extractor: Optional[Callable[[str], tuple[str, str]]] = None,
        suffix_separator: str = "\n\n",
    ):
        """
        Initialize QATokenSHAP
        
        Args:
            model: Model to analyze
            splitter: Text splitter implementation
            vectorizer: Text vectorizer for calculating similarities
            debug: Enable debug output
            section_extractor: Function that splits a prompt into
                (variable_question, static_suffix). Defaults to qa_extractor.
        """
        super().__init__(model=model, splitter=splitter, vectorizer=vectorizer, debug=debug)

        if section_extractor is not None and not callable(section_extractor):
            raise TypeError("section_extractor must be callable.")
        
        self.section_extractor = section_extractor or qa_extractor
        self.suffix_separator = suffix_separator
        self._current_static_suffix = ""
    
    def _get_samples(self, content: str) -> List[str]:
        """
        Extract question tokens from structured prompt.
        The static suffix is cached for use in _prepare_combination_args.
        """
        question_text, static_suffix = self.section_extractor(content)        
        self._current_static_suffix = static_suffix
        
        tokens = self.splitter.split(question_text)
        
        q_display = question_text if len(question_text) <= 100 else f"{question_text[:100]}..."
        s_display = static_suffix if len(static_suffix) <= 100 else f"{static_suffix[:100]}..."
        
        self._debug_print(f"[QATokenSHAP] Extracting QA components")
        self._debug_print(f"[QATokenSHAP] Question: {q_display}")
        self._debug_print(f"[QATokenSHAP] Suffix: {s_display}")
        self._debug_print(f"[QATokenSHAP] Tokens extracted: {len(tokens)}")
            
        return tokens

    def _prepare_combination_args(self, combination: List[str], original_content: str) -> Dict[str, str]:
        """Prepare model input by reattaching the fixed suffix"""
        prompt = self.splitter.join(combination)
        if self._current_static_suffix:
            prompt = f"{prompt}{self.suffix_separator}{self._current_static_suffix}"
        return {"prompt": prompt}
    