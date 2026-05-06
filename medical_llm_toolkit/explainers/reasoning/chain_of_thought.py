"""
Chain-of-Thought (CoT) reasoning extractor for medical LLMs.

Unlike the two-pass approach in MedicalLLMWrapper (answer then rationale),
CoT prompts the model to reason step-by-step BEFORE committing to an answer,
producing grounded reasoning rather than post-hoc justification.

Compatible with any model loaded via medical_llm_toolkit.wrapper.MedicalLLMWrapper.

Usage:
    from medical_llm_toolkit.wrapper import MedicalLLMWrapper
    from medical_llm_toolkit.explainers.reasoning import ChainOfThoughtExtractor

    wrapper = MedicalLLMWrapper("FreedomIntelligence/Apollo-2B")
    wrapper.set_task("mcq")

    cot = ChainOfThoughtExtractor(wrapper)
    result = cot.extract(
        "Patient presents with crushing chest pain radiating to the left arm. "
        "Most likely diagnosis? A) GERD B) MI C) Costochondritis D) PE"
    )
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

_ANSWER_PATTERNS = [
    r"[Tt]herefore[,\s]+(?:the\s+)?(?:answer|correct\s+answer|best\s+answer)\s+is\s+([A-D])",
    r"[Tt]he\s+(?:answer|correct\s+answer|best\s+answer)\s+is\s+([A-D])",
    r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*([A-D])",
    r"[Aa]nswer\s*[:=]\s*([A-D])",
    r"\b([A-D])\s+is\s+(?:the\s+)?(?:most\s+likely|correct|best)",
    r"^([A-D])[.)]\s",
    r"\b([A-D])\b",
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
      - 'zero_shot': appends a generic "Let's think step by step" prompt.
      - 'structured': uses task-specific prompts that guide the model to
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

        Returns a dict with keys:
            reasoning, steps, answer, confidence, option_probs, cot_prompt, task_type
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
        prompt = prompt.strip()

        if self.strategy == "zero_shot":
            return _ZERO_SHOT_TEMPLATE.format(question=prompt)

        if task_type == "mcq":
            return _MCQ_TEMPLATE.format(question=prompt)
        elif task_type == "yn":
            return _YN_TEMPLATE.format(question=prompt)
        else:
            return _FREE_TEMPLATE.format(question=prompt)

    def _generate_reasoning(self, cot_prompt: str) -> str:
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
        return self._clean_reasoning(reasoning)

    def _clean_reasoning(self, text: str) -> str:
        for stop_pattern in [
            r"\n\nQuestion:", r"\n\nQ:", r"\n\nA\)",
            r"\n\nAnswer:", r"\nFinal Answer",
        ]:
            parts = re.split(stop_pattern, text, maxsplit=1)
            text = parts[0]

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _parse_steps(self, reasoning: str) -> List[str]:
        numbered = re.split(r"\n\s*(?:\d+[.)]\s*|[Ss]tep\s+\d+\s*[:.]\s*)", reasoning)
        numbered = [s.strip() for s in numbered if s.strip()]

        if len(numbered) > 1:
            return numbered

        lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
        if len(lines) > 1:
            return lines

        sentences = re.split(r"(?<=[.!?])\s+", reasoning)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [reasoning]

    def _extract_answer_with_confidence(
        self,
        original_prompt: str,
        reasoning: str,
        task_type: str,
    ) -> Tuple[str, Optional[float], Optional[Dict]]:
        if task_type == "free":
            # Free response: the reasoning IS the answer. Don't try to pull
            # an A/B/C/D out of it — the bare-letter fallback pattern would
            # spuriously match references like "Patient A" or "Type B".
            return reasoning, None, None

        if task_type == "yn":
            allowed_ids = self.wrapper.AB_IDS
            labels = ["A", "B"]
            patterns = _YN_ANSWER_PATTERNS
        else:
            allowed_ids = self.wrapper.ABCD_IDS
            labels = ["A", "B", "C", "D"]
            patterns = _ANSWER_PATTERNS

        scoring_prompt = (
            f"{original_prompt.strip()}\n\n"
            f"Reasoning: {reasoning}\n\n"
            "Therefore, the answer is:"
        )

        inputs = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        logits = outputs.logits[0, -1, :]
        allowed_logits = torch.tensor(
            [logits[tid].item() for tid in allowed_ids], dtype=torch.float32
        )

        if torch.isnan(allowed_logits).any() or torch.isinf(allowed_logits).all():
            answer = self._extract_letter_from_text(reasoning, patterns) or labels[0]
            return answer, None, None

        probs = F.softmax(allowed_logits, dim=0)
        option_probs = {label: probs[i].item() for i, label in enumerate(labels)}

        best_idx = probs.argmax().item()
        model_answer = labels[best_idx]
        confidence = probs[best_idx].item()

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
        for pattern in patterns[:-1]:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()

        matches = re.findall(patterns[-1], text)
        return matches[-1] if matches else None


def extract_chain_of_thought(
    wrapper,
    prompt: str,
    task_type: Optional[str] = None,
    strategy: str = "structured",
    max_reasoning_tokens: int = 350,
    print_result: bool = True,
) -> Dict:
    """
    One-liner CoT extraction. See ChainOfThoughtExtractor for parameter details.
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
