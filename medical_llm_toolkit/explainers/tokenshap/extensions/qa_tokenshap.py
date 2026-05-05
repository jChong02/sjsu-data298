# qa_tokenshap.py

from typing import Optional, Callable, List, Dict, Tuple, Any
import pandas as pd

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
            suffix_separator: Separator placed between the perturbed question
                and the static suffix when reassembling the prompt.
        """
        super().__init__(model=model, splitter=splitter, vectorizer=vectorizer, debug=debug)

        if section_extractor is not None and not callable(section_extractor):
            raise TypeError("section_extractor must be callable.")

        self.section_extractor = section_extractor or qa_extractor
        self.suffix_separator = suffix_separator
        self._current_static_suffix = ""

        # Maps combination key → {"pred": str, "probs": dict|None}
        # Populated during _get_result_per_combination; consumed in _get_df_per_combination.
        self._combination_meta: Dict[str, Dict] = {}

        # Captured baseline metadata for the unperturbed (full) prompt.
        # Populated during _calculate_baseline; consumed in _get_df_per_combination.
        self._baseline_meta: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Prompt decomposition
    # ------------------------------------------------------------------

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

        self._debug_print("[QATokenSHAP] Extracting QA components")
        self._debug_print(f"[QATokenSHAP] Question: {q_display}")
        self._debug_print(f"[QATokenSHAP] Suffix: {s_display}")
        self._debug_print(f"[QATokenSHAP] Tokens extracted: {len(tokens)}")

        return tokens

    def _prepare_combination_args(self, combination: List[str], original_content: str) -> Dict[str, str]:
        """Prepare model input by reattaching the fixed suffix."""
        prompt = self.splitter.join(combination)
        if self._current_static_suffix:
            prompt = f"{prompt}{self.suffix_separator}{self._current_static_suffix}"
        return {"prompt": prompt}

    # ------------------------------------------------------------------
    # Baseline capture (critical fix)
    # ------------------------------------------------------------------

    def _calculate_baseline(self, content: Any, **kwargs) -> str:
        """
        Override baseline generation to snapshot model.last_answer and
        model.last_option_probs at the exact time the unperturbed prompt
        is evaluated.

        This avoids incorrectly assuming baseline correctness or reading stale
        state after the perturbation loop runs.
        """
        baseline_text = super()._calculate_baseline(content, **kwargs)

        self._baseline_meta = {
            "pred": getattr(self.model, "last_answer", None),
            "probs": getattr(self.model, "last_option_probs", None),
        }

        self._debug_print(
            "[QATokenSHAP] Captured BASELINE — "
            f"pred={self._baseline_meta['pred']}  probs={self._baseline_meta['probs']}"
        )

        return baseline_text

    # ------------------------------------------------------------------
    # Core override: capture per-combination predictions at call time
    # ------------------------------------------------------------------

    def _get_result_per_combination(
        self,
        content: Any,
        sampling_ratio: float,
        max_combinations: Optional[int] = 1000,
    ) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        """
        Override the base implementation to snapshot model.last_answer and
        model.last_option_probs immediately after each model.generate() call,
        before the next call can overwrite them.

        Snapshots are stored in self._combination_meta (keyed by combination
        key) and consumed later in _get_df_per_combination so that every row
        gets the prediction that was live at the time *that* combination ran.
        """
        self._combination_meta = {}
        captured_snapshots: List[Dict[str, Any]] = []

        original_generate = self.model.generate

        def _capturing_generate(**kwargs):
            response = original_generate(**kwargs)
            snapshot = {
                "pred": getattr(self.model, "last_answer", None),
                "probs": getattr(self.model, "last_option_probs", None),
            }
            captured_snapshots.append(snapshot)
            self._debug_print(
                f"[QATokenSHAP] Captured after generate — "
                f"pred={snapshot['pred']}  probs={snapshot['probs']}"
            )
            return response

        self.model.generate = _capturing_generate
        try:
            responses = super()._get_result_per_combination(
                content,
                sampling_ratio=sampling_ratio,
                max_combinations=max_combinations,
            )
        finally:
            self.model.generate = original_generate

        # responses is an ordered dict; the base class inserts keys in the
        # same order it calls model.generate, so zip is safe.
        for key, snapshot in zip(responses.keys(), captured_snapshots):
            self._combination_meta[key] = snapshot

        return responses

    # ------------------------------------------------------------------
    # Payoff helpers
    # ------------------------------------------------------------------

    def _compute_payoff_from_meta(self, meta: Dict[str, Any], response_text: str = None) -> float:
        """Compute a correctness payoff from one combination's captured metadata."""
        pred = meta.get("pred")
        probs = meta.get("probs")

        self._debug_print(f"[DEBUG PAYOFF] pred={pred}  probs={probs}  correct={self.vectorizer.correct_label}")

        return self.vectorizer.compute_payoff(pred, probs, response=response_text)

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _get_df_per_combination(self, responses, baseline_text):
        """
        Build the results DataFrame.

        For correctness-mode vectorizers, payoffs are looked up from:
          - self._baseline_meta for the unperturbed baseline
          - self._combination_meta for each perturbed combination

        so every row uses the prediction that was current at evaluation time,
        not stale model.last_* state.
        """
        df = pd.DataFrame(
            [(key.split("_")[0], response[0], response[1]) for key, response in responses.items()],
            columns=["Content", "Response", "Indexes"],
        )

        vec = self.vectorizer

        # ----------------------------
        # CASE 1: correctness scoring
        # ----------------------------
        if hasattr(vec, "correct_label"):
            if self._baseline_meta is None or self._baseline_meta.get("pred") is None:
                raise RuntimeError(
                    "Baseline metadata missing. QATokenSHAP must capture baseline pred/probs "
                    "during _calculate_baseline (unperturbed generate call)."
                )

            # Give hybrid vectorizers a chance to encode the baseline response
            # before the payoff loop starts.
            if hasattr(vec, "set_baseline"):
                vec.set_baseline(baseline_text)

            baseline_payoff = self._compute_payoff_from_meta(self._baseline_meta, response_text=baseline_text)
            self._debug_print(f"[QATokenSHAP] Baseline payoff: {baseline_payoff}")

            sims = []
            for key in responses:
                meta = self._combination_meta.get(key)
                if meta is None:
                    raise RuntimeError(
                        f"No captured metadata for combination key {key!r}. "
                        "This is an internal bug in QATokenSHAP."
                    )
                response_text_combo = responses[key][0]
                payoff = self._compute_payoff_from_meta(meta, response_text=response_text_combo)
                sims.append(payoff - baseline_payoff)

            df["Similarity"] = sims

            if self.debug:
                self._debug_print("=== Correctness-Based Similarity Scores ===")
                self._debug_print(str(df["Similarity"].tolist()))
                self._debug_print(f"Unique similarity values: {df['Similarity'].unique()}")

            return df

        # ----------------------------
        # CASE 2: fall back to TF-IDF / embedding similarity
        # ----------------------------
        return super()._get_df_per_combination(responses, baseline_text)