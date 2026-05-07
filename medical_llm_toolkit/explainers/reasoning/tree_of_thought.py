"""
Tree-of-Thought (ToT) reasoning extractor for medical LLMs.

Generates N candidate reasoning branches via sampling, scores each by mean
log-probability under the model (a coherence proxy), selects the highest-
scoring branch, then runs constrained answer generation conditioned on it.

Compatible with any model loaded via medical_llm_toolkit.wrapper.MedicalLLMWrapper.

Usage:
    from medical_llm_toolkit.wrapper import MedicalLLMWrapper
    from medical_llm_toolkit.explainers.reasoning import TreeOfThoughtExtractor

    wrapper = MedicalLLMWrapper("FreedomIntelligence/Apollo-2B")
    wrapper.set_task("mcq")

    tot = TreeOfThoughtExtractor(wrapper, n_branches=3, max_thought_tokens=80)
    result = tot.extract(
        "Patient on long-term heparin develops thrombocytopenia. "
        "Suspect? A) DIC B) HIT C) ITP D) TTP"
    )
    print(result["best_thought"])
    print("Answer:", result["answer"])
"""

import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import LogitsProcessorList

from medical_llm_toolkit.wrapper import EnforceAnswerThenRationaleProcessor


class TreeOfThoughtExtractor:
    """
    Tree-of-Thought reasoning extractor.

    Algorithm (per extract() call):
      1. Branch  - sample n_branches independent reasoning thoughts
      2. Score   - rank each by mean log-probability (model coherence proxy)
      3. Select  - keep the highest-scoring thought
      4. Solve   - run constrained answer generation conditioned on best thought
      5. Explain - generate final rationale

    Attributes (after extract()):
        last_result (dict): Most recent extraction result.
    """

    def __init__(
        self,
        wrapper,
        n_branches: int = 3,
        max_thought_tokens: int = 80,
        thought_temperature: float = 0.8,
        thought_top_p: float = 0.9,
        rationale_max_tokens: int = 150,
        rationale_temperature: float = 0.7,
        rationale_top_p: float = 0.9,
        verbose: bool = True,
    ):
        if n_branches < 1:
            raise ValueError(f"n_branches must be >= 1, got {n_branches}")

        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = wrapper.device

        self.n_branches = n_branches
        self.max_thought_tokens = max_thought_tokens
        self.thought_temperature = thought_temperature
        self.thought_top_p = thought_top_p
        self.rationale_max_tokens = rationale_max_tokens
        self.rationale_temperature = rationale_temperature
        self.rationale_top_p = rationale_top_p
        self.verbose = verbose

        self.last_result: Optional[Dict] = None
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, prompt: str, task_type: Optional[str] = None) -> Dict:
        """
        Run a full Tree-of-Thought pass for a single prompt.

        Returns a dict with keys:
            thoughts        - list of all generated thought branches
            scores          - list of mean log-prob scores (parallel to thoughts)
            best_idx        - index of the selected branch
            best_thought    - the selected branch's text
            answer          - final answer string
            confidence      - softmax probability of selected answer (mcq/yn only)
            option_probs    - per-option probabilities (mcq/yn only)
            rationale       - generated rationale (mcq/yn only)
            tot_prompt      - the thought-generation prompt
            task_type       - task type used
        """
        effective_task = task_type or self.wrapper.task_type

        thought_prompt = f"{prompt}\n\nLet me reason through this step by step:"

        # --- Step 1 & 2: Branch and Score ---
        thoughts: List[str] = []
        scores: List[float] = []
        for i in range(self.n_branches):
            thought = self._generate_thought_branch(
                thought_prompt, max_tokens=self.max_thought_tokens
            )
            score = self._score_thought_logprob(thought_prompt, thought)
            thoughts.append(thought)
            scores.append(score)
            if self.verbose:
                print(f"[ToT] Branch {i + 1}/{self.n_branches} | score: {score:.4f}")

        # --- Step 3: Select ---
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_thought = thoughts[best_idx]
        if self.verbose:
            print(f"[ToT] Selected branch {best_idx + 1} (score: {scores[best_idx]:.4f})")

        result: Dict = {
            "thoughts": thoughts,
            "scores": scores,
            "best_idx": best_idx,
            "best_thought": best_thought,
            "tot_prompt": thought_prompt,
            "task_type": effective_task,
            "answer": None,
            "confidence": None,
            "option_probs": None,
            "rationale": None,
        }

        if effective_task in {"yn", "mcq"}:
            answer, confidence, option_probs = self._solve_constrained(
                prompt, best_thought, effective_task
            )
            rationale = self._generate_rationale(prompt, best_thought, answer)

            result["answer"] = answer
            result["confidence"] = confidence
            result["option_probs"] = option_probs
            result["rationale"] = rationale
        else:
            # Free-form: best thought is the answer.
            result["answer"] = best_thought

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
                print(f"[ToT {i+1}/{len(prompts)}] Extracting...")
            tt = task_types[i] if task_types else None
            results.append(self.extract(prompt, task_type=tt))

        if show_progress:
            print(f"[ToT {len(prompts)}/{len(prompts)}] Complete!")

        return results

    def format_result(self, result: Optional[Dict] = None) -> str:
        r = result or self.last_result
        if r is None:
            return "(no result yet)"

        lines = ["=" * 70, "Tree-of-Thought Reasoning", "=" * 70]
        for i, (t, s) in enumerate(zip(r["thoughts"], r["scores"]), 1):
            marker = " *" if (i - 1) == r["best_idx"] else "  "
            lines.append(f"{marker} Branch {i} | score {s:.4f}")
            lines.append(f"     {t}")

        lines.append("-" * 70)
        lines.append(f"  Answer   : {r['answer']}")
        if r.get("confidence") is not None:
            lines.append(f"  Confidence: {r['confidence']:.1%}")
        if r.get("option_probs"):
            probs_str = "  | ".join(
                f"{k}: {v:.1%}" for k, v in r["option_probs"].items()
            )
            lines.append(f"  Options  : {probs_str}")
        if r.get("rationale"):
            lines.append(f"  Rationale: {r['rationale']}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_thought_branch(self, thought_prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(thought_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.thought_temperature,
                top_p=self.thought_top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        thought = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        for stop in ["\n\nAnswer:", "\n\nQuestion:", "Final Answer", "\n\nA)"]:
            thought = thought.split(stop)[0]
        return re.sub(r"\s{2,}", " ", thought).strip()

    def _score_thought_logprob(self, context: str, thought: str) -> float:
        full_text = context + thought
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        context_len = self.tokenizer(context, return_tensors="pt")["input_ids"].shape[1]

        labels = full_inputs["input_ids"].clone()
        labels[0, :context_len] = -100

        with torch.no_grad():
            outputs = self.model(**full_inputs, labels=labels)

        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float("-inf")
        return -loss.item()

    def _solve_constrained(
        self,
        prompt: str,
        best_thought: str,
        task_type: str,
    ) -> Tuple[str, Optional[float], Optional[Dict]]:
        if task_type == "yn":
            allowed_ids = self.wrapper.AB_IDS
            labels = ["A", "B"]
        else:
            allowed_ids = self.wrapper.ABCD_IDS
            labels = ["A", "B", "C", "D"]

        answer_prompt = (
            f"{prompt}\n\n"
            f"Reasoning: {best_thought}\n\n"
            "Based on the above reasoning, Answer:"
        )
        inputs = self.tokenizer(answer_prompt, return_tensors="pt").to(self.device)
        processors = LogitsProcessorList(
            [EnforceAnswerThenRationaleProcessor(allowed_ids)]
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                logits_processor=processors,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_token_id = outputs.sequences[0, inputs["input_ids"].shape[1]].item()
        answer = self.tokenizer.decode([gen_token_id], skip_special_tokens=True).strip()

        logits = outputs.scores[0][0]
        allowed_logits = torch.tensor(
            [logits[tid].item() for tid in allowed_ids], dtype=torch.float32
        )

        if torch.isnan(allowed_logits).any() or torch.isinf(allowed_logits).any():
            return answer, None, None

        probs = F.softmax(allowed_logits, dim=0)
        option_probs = {label: probs[i].item() for i, label in enumerate(labels)}

        try:
            gen_idx = list(allowed_ids).index(gen_token_id)
            confidence = probs[gen_idx].item()
        except ValueError:
            confidence = None

        return answer, confidence, option_probs

    def _generate_rationale(self, prompt: str, best_thought: str, answer: str) -> str:
        rationale_prompt = (
            f"{prompt}\n\n"
            f"Reasoning: {best_thought}\n\n"
            f"Based on the above reasoning, Answer: {answer}\n\n"
            "Rationale:"
        )
        inputs = self.tokenizer(rationale_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.rationale_max_tokens,
                do_sample=True,
                temperature=self.rationale_temperature,
                top_p=self.rationale_top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        rationale = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return self._clean_rationale(rationale)

    @staticmethod
    def _clean_rationale(rationale: str) -> str:
        rationale = rationale.split("Final Answer")[0]
        rationale = rationale.split("Question:")[0]
        rationale = rationale.split("\n\nAnswer:")[0]
        rationale = rationale.split("\n\nA)")[0]
        rationale = rationale.strip()
        return re.sub(r"\s{2,}", " ", rationale).strip()


def extract_tree_of_thought(
    wrapper,
    prompt: str,
    task_type: Optional[str] = None,
    n_branches: int = 3,
    max_thought_tokens: int = 80,
    print_result: bool = True,
) -> Dict:
    """
    One-liner ToT extraction. See TreeOfThoughtExtractor for parameter details.
    """
    tot = TreeOfThoughtExtractor(
        wrapper,
        n_branches=n_branches,
        max_thought_tokens=max_thought_tokens,
    )
    result = tot.extract(prompt, task_type=task_type)

    if print_result:
        print(tot.format_result(result))

    return result
