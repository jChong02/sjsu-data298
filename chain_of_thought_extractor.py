"""
chain_of_thought_extractor.py

Chain-of-Thought (CoT) reasoning extractor for medical LLMs.

Unlike the two-pass approach in MedicalLLMWrapper (answer then rationale),
CoT prompts the model to reason step-by-step BEFORE committing to an answer.
This produces grounded reasoning rather than post-hoc justification.

Compatible with any model loaded via MedicalLLMWrapper.

Usage:
    from medical_llm_wrapper import load_medical_llm
    from chain_of_thought_extractor import ChainOfThoughtExtractor

    wrapper = load_medical_llm("FreedomIntelligence/Apollo-2B")
    wrapper.set_task("mcq")

    cot = ChainOfThoughtExtractor(wrapper)
    result = cot.extract("Patient presents with crushing chest pain radiating to the left arm. "
                         "Most likely diagnosis? A) GERD B) MI C) Costochondritis D) PE")
    print(result["reasoning"])
    print("Answer:", result["answer"])
"""

import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CoT prompt templates
# ---------------------------------------------------------------------------

_ZERO_SHOT_TEMPLATE = (
    "{question}\n\n"
    "Let's think through this step by step before answering.\n\n"
    "Reasoning:"
)

_MCQ_TEMPLATE = (
    "{question}\n\n"
    "Think through each option carefully before selecting the best answer.\n\n"
    "Step-by-step reasoning:"
)

_YN_TEMPLATE = (
    "{question}\n\n"
    "Let's analyze the evidence step by step.\n\n"
    "Reasoning:"
)

_FREE_TEMPLATE = (
    "{question}\n\n"
    "Let's think through this carefully.\n\n"
    "Reasoning:"
)

# Patterns used to locate the final answer within generated text
_ANSWER_PATTERNS = [
    r"[Tt]herefore[,\s]+(?:the\s+)?(?:answer|correct\s+answer|best\s+answer)\s+is\s+([A-D])",
    r"[Tt]he\s+(?:answer|correct\s+answer|best\s+answer)\s+is\s+([A-D])",
    r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*([A-D])",
    r"[Aa]nswer\s*[:=]\s*([A-D])",
    r"\b([A-D])\s+is\s+(?:the\s+)?(?:most\s+likely|correct|best)",
    r"^([A-D])[.)]\s",   # Line starting with A) or A.
    r"\b([A-D])\b",      # Fallback: last standalone letter
]

_YN_ANSWER_PATTERNS = [
    r"[Tt]herefore[,\s]+(?:the\s+)?(?:answer|response)\s+is\s+(Yes|No|A|B)",
    r"[Tt]he\s+(?:answer|response)\s+is\s+(Yes|No|A|B)",
    r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*(Yes|No|A|B)",
    r"\b(Yes|No)\b",
]


class ChainOfThoughtExtractor:
    """
    Extracts Chain-of-Thought reasoning from medical LLMs.

    Two strategies are supported:
      - 'zero_shot': Appends "Let's think step by step" to the prompt.
      - 'structured': Uses task-specific prompts that guide the model to
                      consider each option before concluding.

    Attributes (after extract()):
        last_result (dict): Most recent extraction result.
    """

    def __init__(
        self,
        wrapper,
        strategy: str = "structured",
        max_reasoning_tokens: int = 350,
        temperature: float = 0.7,
        top_p: float = 0.9,
        verbose: bool = True,
    ):
        """
        Args:
            wrapper: MedicalLLMWrapper instance (model must already be loaded).
            strategy: 'zero_shot' or 'structured' prompting strategy.
            max_reasoning_tokens: Max tokens for the reasoning generation pass.
            temperature: Sampling temperature for reasoning generation.
            top_p: Nucleus sampling p for reasoning generation.
            verbose: Print progress messages.
        """
        if strategy not in {"zero_shot", "structured"}:
            raise ValueError(f"strategy must be 'zero_shot' or 'structured', got '{strategy}'")

        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = wrapper.device
        self.strategy = strategy
        self.max_reasoning_tokens = max_reasoning_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        self.last_result: Optional[Dict] = None
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, prompt: str, task_type: Optional[str] = None) -> Dict:
        """
        Generate Chain-of-Thought reasoning for a single prompt.

        Args:
            prompt: The medical question (may already contain A/B/C/D options).
            task_type: Override the wrapper's current task type ('yn', 'mcq',
                       'free'). If None, uses wrapper.task_type.

        Returns:
            Dictionary with keys:
                - 'reasoning'     : Full raw reasoning text from the model.
                - 'steps'         : Reasoning parsed into a list of step strings.
                - 'answer'        : Extracted final answer letter (or text).
                - 'confidence'    : Softmax probability of the answer token
                                   (None for 'free' tasks).
                - 'option_probs'  : Dict of per-option probabilities (mcq/yn only).
                - 'cot_prompt'    : The full prompt sent to the model.
                - 'task_type'     : Task type used.
        """
        effective_task = task_type or self.wrapper.task_type

        cot_prompt = self._build_cot_prompt(prompt, effective_task)
        reasoning_raw = self._generate_reasoning(cot_prompt)
        steps = self._parse_steps(reasoning_raw)

        answer, confidence, option_probs = self._extract_answer_with_confidence(
            prompt, reasoning_raw, effective_task
        )

        result = {
            "reasoning": reasoning_raw,
            "steps": steps,
            "answer": answer,
            "confidence": confidence,
            "option_probs": option_probs,
            "cot_prompt": cot_prompt,
            "task_type": effective_task,
        }
        self.last_result = result
        return result

    def batch_extract(
        self,
        prompts: List[str],
        task_types: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Extract CoT reasoning for a list of prompts.

        Args:
            prompts: List of question strings.
            task_types: Per-prompt task type overrides. If None, uses
                        wrapper.task_type for all prompts.
            show_progress: Print a progress counter.

        Returns:
            List of result dictionaries (same format as extract()).
        """
        if task_types is not None and len(task_types) != len(prompts):
            raise ValueError("task_types must have the same length as prompts")

        results = []
        for i, prompt in enumerate(prompts):
            if show_progress:
                print(f"[CoT {i+1}/{len(prompts)}] Extracting...", end="\r")
            tt = task_types[i] if task_types else None
            results.append(self.extract(prompt, task_type=tt))

        if show_progress:
            print(f"[CoT {len(prompts)}/{len(prompts)}] Complete!      ")

        return results

    def format_result(self, result: Optional[Dict] = None) -> str:
        """
        Return a human-readable string for a result dictionary.
        Uses self.last_result if result is None.
        """
        r = result or self.last_result
        if r is None:
            return "(no result yet)"

        lines = ["=" * 70, "Chain-of-Thought Reasoning", "=" * 70]

        if r["steps"]:
            for i, step in enumerate(r["steps"], 1):
                lines.append(f"  Step {i}: {step}")
        else:
            lines.append(f"  {r['reasoning']}")

        lines.append("-" * 70)
        lines.append(f"  Answer   : {r['answer']}")

        if r["confidence"] is not None:
            lines.append(f"  Confidence: {r['confidence']:.1%}")

        if r["option_probs"]:
            probs_str = "  | ".join(
                f"{k}: {v:.1%}" for k, v in r["option_probs"].items()
            )
            lines.append(f"  Options  : {probs_str}")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cot_prompt(self, prompt: str, task_type: str) -> str:
        """Select and fill the appropriate CoT prompt template."""
        prompt = prompt.strip()

        if self.strategy == "zero_shot":
            return _ZERO_SHOT_TEMPLATE.format(question=prompt)

        # structured: task-aware templates
        if task_type == "mcq":
            return _MCQ_TEMPLATE.format(question=prompt)
        elif task_type == "yn":
            return _YN_TEMPLATE.format(question=prompt)
        else:
            return _FREE_TEMPLATE.format(question=prompt)

    def _generate_reasoning(self, cot_prompt: str) -> str:
        """Run a single generation pass to produce reasoning text."""
        inputs = self.tokenizer(cot_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_reasoning_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=(
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                ),
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        reasoning = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reasoning = self._clean_reasoning(reasoning)
        return reasoning

    def _clean_reasoning(self, text: str) -> str:
        """Remove common generation artifacts from reasoning text."""
        # Stop at a new question block
        for stop_pattern in [
            r"\n\nQuestion:", r"\n\nQ:", r"\n\nA\)",
            r"\n\nAnswer:", r"\nFinal Answer",
        ]:
            parts = re.split(stop_pattern, text, maxsplit=1)
            text = parts[0]

        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _parse_steps(self, reasoning: str) -> List[str]:
        """
        Parse reasoning text into a list of discrete steps.

        Tries numbered lists first, then falls back to sentence splitting.
        """
        # Try numbered steps: "1.", "1)", "Step 1:"
        numbered = re.split(r"\n\s*(?:\d+[.)]\s*|[Ss]tep\s+\d+\s*[:.]\s*)", reasoning)
        numbered = [s.strip() for s in numbered if s.strip()]

        if len(numbered) > 1:
            return numbered

        # Try splitting on newlines that look like logical breaks
        lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
        if len(lines) > 1:
            return lines

        # Fallback: split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", reasoning)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [reasoning]

    def _extract_answer_with_confidence(
        self,
        original_prompt: str,
        reasoning: str,
        task_type: str,
    ) -> Tuple[str, Optional[float], Optional[Dict]]:
        """
        Extract the final answer from the reasoning text.

        For constrained tasks (yn/mcq), also computes a confidence score
        by scoring the answer token directly from the model.

        Returns:
            (answer_str, confidence_or_None, option_probs_or_None)
        """
        if task_type == "free":
            # For free tasks, the reasoning IS the answer; no confidence
            answer = self._extract_letter_from_text(reasoning, patterns=_ANSWER_PATTERNS)
            return answer or reasoning, None, None

        # --- Constrained tasks: score answer options directly ---
        if task_type == "yn":
            allowed_ids = self.wrapper.AB_IDS
            labels = ["A", "B"]
            patterns = _YN_ANSWER_PATTERNS
        else:  # mcq
            allowed_ids = self.wrapper.ABCD_IDS
            labels = ["A", "B", "C", "D"]
            patterns = _ANSWER_PATTERNS

        # Build a scoring prompt: original question + reasoning + "Therefore, Answer:"
        scoring_prompt = (
            f"{original_prompt.strip()}\n\n"
            f"Reasoning: {reasoning}\n\n"
            "Therefore, the answer is:"
        )

        inputs = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True,
            )

        # Logits for the next (answer) token
        logits = outputs.logits[0, -1, :]  # [vocab_size]
        allowed_logits = torch.tensor(
            [logits[tid].item() for tid in allowed_ids], dtype=torch.float32
        )

        if torch.isnan(allowed_logits).any() or torch.isinf(allowed_logits).all():
            # Fallback: extract from text
            answer = self._extract_letter_from_text(reasoning, patterns) or labels[0]
            return answer, None, None

        probs = F.softmax(allowed_logits, dim=0)
        option_probs = {label: probs[i].item() for i, label in enumerate(labels)}

        # Primary answer: highest-probability option
        best_idx = probs.argmax().item()
        model_answer = labels[best_idx]
        confidence = probs[best_idx].item()

        # Cross-check with text extraction; prefer model scoring but log mismatch
        text_answer = self._extract_letter_from_text(reasoning, patterns)
        if text_answer and text_answer != model_answer and self.verbose:
            print(
                f"[CoT] Note: text extracted '{text_answer}' but model scores "
                f"'{model_answer}' as highest probability — using model score."
            )

        return model_answer, confidence, option_probs

    @staticmethod
    def _extract_letter_from_text(
        text: str, patterns: List[str]
    ) -> Optional[str]:
        """
        Try each regex pattern in order; return the first match found.
        Uses the last occurrence for the fallback bare-letter pattern.
        """
        for pattern in patterns[:-1]:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()

        # Fallback: find all standalone option letters and take the last one
        matches = re.findall(patterns[-1], text)
        return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_chain_of_thought(
    wrapper,
    prompt: str,
    task_type: Optional[str] = None,
    strategy: str = "structured",
    max_reasoning_tokens: int = 350,
    print_result: bool = True,
) -> Dict:
    """
    One-liner CoT extraction.

    Args:
        wrapper: MedicalLLMWrapper instance.
        prompt: Medical question string.
        task_type: 'yn', 'mcq', or 'free'. Defaults to wrapper.task_type.
        strategy: 'zero_shot' or 'structured'.
        max_reasoning_tokens: Max tokens for reasoning generation.
        print_result: Pretty-print the result after extraction.

    Returns:
        Result dictionary (see ChainOfThoughtExtractor.extract).

    Example:
        >>> wrapper = load_medical_llm("FreedomIntelligence/Apollo-2B")
        >>> wrapper.set_task("mcq")
        >>> result = extract_chain_of_thought(
        ...     wrapper,
        ...     "A 45-year-old male presents with crushing chest pain. "
        ...     "Most likely diagnosis? A) GERD  B) MI  C) Costochondritis  D) PE"
        ... )
    """
    cot = ChainOfThoughtExtractor(
        wrapper,
        strategy=strategy,
        max_reasoning_tokens=max_reasoning_tokens,
    )
    result = cot.extract(prompt, task_type=task_type)

    if print_result:
        print(cot.format_result(result))

    return result
