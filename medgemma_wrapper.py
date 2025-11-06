"""
medgemma_wrapper.py

Custom wrapper for Google's MedGemma-4B-IT model providing structured
'Answer:' + 'Rationale:' output for Yes/No, MCQ, and free-response
medical QA tasks.

Implements a lightweight constraint mechanism using Hugging Face
LogitsProcessors to enforce valid answer tokens
and generate concise rationales.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import re
import torch.nn.functional as F


class EnforceAnswerThenRationaleProcessor(LogitsProcessor):
    """
    LogitsProcessor that restricts the first generated token to a set
    of allowed token IDs (e.g., 'A', 'B', 'C', 'D') for controlled
    multiple-choice or Yes/No question answering.
    """
    def __init__(self, allowed_ids):
        super().__init__()
        self.allowed_ids = torch.tensor(allowed_ids)
        self.first_step = True

    def __call__(self, input_ids, scores):
        if self.first_step:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_ids.to(scores.device)] = scores[:, self.allowed_ids.to(scores.device)]
            self.first_step = False
            return mask
        return scores


class MedGemmaQAWrapper:
    """
    MedGemma QA wrapper for constrained answer + rationale generation.

    Supports two modes:
      - 'answer_rationale': (default) full reasoning output
      - 'answer_only'    : only generates answer token and confidence
    
    Supports three task types:
      - 'yn'  : Yes/No questions (forces first token A or B)
      - 'mcq' : Multiple-choice questions (forces A/B/C/D)
      - 'free': Unconstrained free-text answers

    Methods:
        set_task(task_type): set task type before generation
        generate(prompt): produce structured answer + rationale output

    Attributes (after generation):
        last_answer (str)
        last_confidence (float)
        last_option_probs (dict)
    """
    
    def __init__(self, model_name="google/medgemma-4b-it", device="cuda", token=None):
        """
        Load MedGemma model and tokenizer, precomputing token IDs
        for A/B/C/D choices.

        Args:
            model_name: Hugging Face model ID
            device: target device ('cuda' or 'cpu')
            token: Hugging Face auth token for gated models
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, token=token)
        self.device = torch.device(device)
        self.task_type = "free"
        self.mode = "answer_rationale"

        self.AB_IDS = [self.tokenizer.encode("A", add_special_tokens=False)[0],
                       self.tokenizer.encode("B", add_special_tokens=False)[0]]
        self.ABCD_IDS = [self.tokenizer.encode(ch, add_special_tokens=False)[0] for ch in ["A", "B", "C", "D"]]

        # runtime variables to store recent outputs
        self.last_answer = None
        self.last_confidence = None
        self.last_option_probs = None

    def set_task(self, task_type):
        """
        Set task type: 'yn', 'mcq', or 'free'.
        """
        self.task_type = task_type

    def set_mode(self, mode):
        """
        Set generation mode:
          - 'answer_rationale': generate both answer and rationale
          - 'answer_only'    : generate only answer (with confidence)
        """
        if mode not in {"answer_rationale", "answer_only"}:
            print(f"[Warning] Unknown mode '{mode}', defaulting to 'answer_rationale'")
            mode = "answer_rationale"
        self.mode = mode

    def generate(self, prompt):
        """
        Unified generate method.
        Behavior depends on self.mode:
          - 'answer_rationale': returns full answer + rationale
          - 'answer_only': returns only the answer and stores confidence internally
        """
        if self.mode == "answer_only":
            answer, conf, option_probs = self._generate_with_confidence(prompt)
            self.last_answer = answer
            self.last_confidence = conf
            self.last_option_probs = option_probs
            return f"Answer: {answer}"
        else:
            return self._generate_with_rationale(prompt)

    def _generate_with_rationale(self, prompt):
        """
        Generate a structured response consisting of:
            Answer: <letter or text>
            Rationale: <short explanation>
        """
        # Ensure prompt ends with 'Answer:' cue for constrained generation
        if self.task_type in {"yn", "mcq"} and not prompt.strip().endswith("Answer:"):
            prompt = prompt.rstrip() + "\n\nAnswer:"

        # Configure allowed IDs and generation limits
        if self.task_type == "yn":
            allowed_ids = self.AB_IDS
            max_new = 1
        elif self.task_type == "mcq":
            allowed_ids = self.ABCD_IDS
            max_new = 1
        else:
            allowed_ids = None
            max_new = 200

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        processors = LogitsProcessorList([EnforceAnswerThenRationaleProcessor(allowed_ids)]) if allowed_ids else LogitsProcessorList([])

        # First: constrained answer generation
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new, logits_processor=processors,
                                         pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        gen = output[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        # Second: rationale generation
        if self.task_type in {"yn", "mcq"}:
            reasoning_prompt = (
                f"{prompt}\n\nYou answered: {answer}.\n"
                "Write exactly one short paragraph beginning with 'Rationale:' "
                "and do not include any Final Answer or additional sections."
            )

            inputs2 = self.tokenizer(reasoning_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output2 = self.model.generate(**inputs2, max_new_tokens=125, do_sample=False, no_repeat_ngram_size=0,
                                              pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            rationale = self.tokenizer.decode(output2[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rationale = rationale.split("Final Answer")[0]
            rationale = rationale.split("Rationale for")[0]
            rationale = rationale.split("Rationale is")[0]
            rationale = rationale.split("Answer:")[0]
            rationale = re.sub(r"Rationale:\s*", "", rationale)  # remove leading repeated label
            rationale = re.sub(r"\s{2,}", " ", rationale).strip()

            self.last_answer = answer
            self.last_confidence = None
            self.last_option_probs = None

            return f"Answer: {answer}\nRationale: {rationale}"
        else:
            self.last_answer = answer
            return answer

    def generate_with_confidence(self, prompt):
        """
        Generate only the answer token and compute its confidence
        (softmax probability over allowed tokens).
        
        Returns:
            answer (str), confidence (float), option_probs (dict)
        """
        if self.task_type in {"yn", "mcq"} and not prompt.strip().endswith("Answer:"):
            prompt = prompt.rstrip() + "\n\nAnswer:"
        
        # Configure allowed tokens
        if self.task_type == "yn":
            allowed_ids = self.AB_IDS
            labels = ["A", "B"]
        elif self.task_type == "mcq":
            allowed_ids = self.ABCD_IDS
            labels = ["A", "B", "C", "D"]
        else:
            raise ValueError("Confidence extraction only supported for Y/N or MCQ.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        processors = LogitsProcessorList([EnforceAnswerThenRationaleProcessor(allowed_ids)])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                logits_processor=processors,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated token
        gen_token_id = outputs.sequences[0, inputs["input_ids"].shape[1]].item()
        answer = self.tokenizer.decode([gen_token_id], skip_special_tokens=True).strip()
        
        # Compute softmax over vocab for the first generated token
        logits = outputs.scores[0][0]  # [vocab_size]
        probs = F.softmax(logits, dim=-1)
        
        # Build option probabilities dictionary
        option_probs = {label: probs[token_id].item() 
                        for label, token_id in zip(labels, allowed_ids)}
        
        # Confidence = probability of the generated token
        confidence = probs[gen_token_id].item()
        
        return answer, confidence, option_probs
