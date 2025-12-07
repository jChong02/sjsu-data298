"""
medical_llm_wrapper.py

Model-agnostic wrapper for medical language models providing structured
'Answer:' + 'Rationale:' output for Yes/No, MCQ, and free-response
medical QA tasks.

Implements a lightweight constraint mechanism using Hugging Face
LogitsProcessors to enforce valid answer tokens and generate concise rationales.

Compatible with any HuggingFace causal language model including:
- Google MedGemma (medgemma-4b-it)
- FreedomIntelligence Apollo (Apollo-2B)
- BioMistral (BioMistral-7B)
- BioMedLM
- And other medical/general LLMs
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


class MedicalLLMWrapper:
    """
    Model-agnostic medical LLM wrapper for constrained answer + rationale generation.

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
        model_name (str)
        model_dtype (torch.dtype)
    """
    
    def __init__(self, model_name, device="cuda", token=None, torch_dtype=None):
        """
        Load any HuggingFace causal language model and tokenizer,
        precomputing token IDs for A/B/C/D choices.

        Args:
            model_name: Hugging Face model ID (e.g., "google/medgemma-4b-it", 
                       "FreedomIntelligence/Apollo-2B", "BioMistral/BioMistral-7B")
            device: target device ('cuda' or 'cpu')
            token: Hugging Face auth token for gated models (optional)
            torch_dtype: torch dtype for model (e.g., torch.float16, torch.float32)
                        If None, uses model's default. MedGemma requires float32.
        """
        print(f"[MedicalLLMWrapper] Loading model: {model_name}")
        
        # Handle MedGemma's float32 requirement automatically
        force_fp32_conversion = False
        if torch_dtype is None and "medgemma" in model_name.lower():
            torch_dtype = torch.float32
            force_fp32_conversion = True
            print(f"[MedicalLLMWrapper] Detected MedGemma - automatically using float32")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Load model with specified or auto-detected dtype
        # Use 'dtype' parameter (torch_dtype is deprecated in newer transformers)
        if torch_dtype is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device, 
                token=token,
                dtype=torch_dtype
            )
            self.model_dtype = torch_dtype
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device, 
                token=token
            )
            self.model_dtype = next(self.model.parameters()).dtype
        
        # Force fp32 conversion for MedGemma if loaded from cache in wrong dtype
        if force_fp32_conversion and self.model_dtype != torch.float32:
            print(f"[MedicalLLMWrapper] WARNING: Model loaded as {self.model_dtype}, converting to float32...")
            self.model = self.model.to(dtype=torch.float32)
            self.model_dtype = torch.float32
            print(f"[MedicalLLMWrapper] ✓ Converted to float32")
        
        self.device = torch.device(device if device != "auto" else "cuda")
        self.task_type = "free"
        self.mode = "answer_rationale"

        # Precompute token IDs for constrained generation
        # Handle different tokenizer behaviors
        self.AB_IDS = self._encode_options(["A", "B"])
        self.ABCD_IDS = self._encode_options(["A", "B", "C", "D"])

        # Runtime variables to store recent outputs
        self.last_answer = None
        self.last_confidence = None
        self.last_option_probs = None
        
        print(f"[MedicalLLMWrapper] ✓ Model loaded successfully")
        print(f"[MedicalLLMWrapper]   Device: {self.device}")
        print(f"[MedicalLLMWrapper]   Dtype: {self.model_dtype}")
        print(f"[MedicalLLMWrapper]   Option token IDs - AB: {self.AB_IDS}, ABCD: {self.ABCD_IDS}")

    def _encode_options(self, options):
        """
        Encode option letters to token IDs, handling various tokenizer behaviors.
        Some tokenizers may encode 'A' as multiple tokens or with special formatting.
        """
        ids = []
        for option in options:
            # Try different encoding strategies
            tokens = self.tokenizer.encode(option, add_special_tokens=False)
            if len(tokens) == 0:
                # Fallback: try with space
                tokens = self.tokenizer.encode(f" {option}", add_special_tokens=False)
            if len(tokens) > 0:
                ids.append(tokens[0])  # Take first token
            else:
                print(f"[Warning] Could not encode option '{option}'")
        return ids

    def set_task(self, task_type):
        """
        Set task type: 'yn', 'mcq', or 'free'.
        """
        if task_type not in {"yn", "mcq", "free"}:
            print(f"[Warning] Unknown task type '{task_type}', defaulting to 'free'")
            task_type = "free"
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
            answer, conf, option_probs = self.generate_with_confidence(prompt)
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
            output = self.model.generate(
                **inputs, 
                max_new_tokens=max_new, 
                logits_processor=processors,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        gen = output[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        # Second: rationale generation
        if self.task_type in {"yn", "mcq"}:
            reasoning_prompt = (
                f"{prompt} {answer}\n\n"
                "Rationale:"
            )

            inputs2 = self.tokenizer(reasoning_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output2 = self.model.generate(
                    **inputs2, 
                    max_new_tokens=150, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            rationale = self.tokenizer.decode(output2[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Clean up rationale text - remove common artifacts
            rationale = rationale.split("Final Answer")[0]
            rationale = rationale.split("Question:")[0]
            rationale = rationale.split("\n\nAnswer:")[0]
            rationale = rationale.split("\n\nA)")[0]
            rationale = rationale.strip()
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
            raise ValueError("Confidence extraction only supported for Y/N or MCQ tasks.")
        
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
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated token
        gen_token_id = outputs.sequences[0, inputs["input_ids"].shape[1]].item()
        answer = self.tokenizer.decode([gen_token_id], skip_special_tokens=True).strip()
        
        # Compute softmax ONLY over allowed tokens (not entire vocabulary)
        # This avoids NaN issues from -inf values set by LogitsProcessor
        logits = outputs.scores[0][0]  # [vocab_size]
        
        # Extract logits for allowed tokens only
        allowed_logits = torch.tensor([logits[token_id].item() for token_id in allowed_ids], 
                                      dtype=torch.float32)
        
        # Check for invalid logits
        if torch.isnan(allowed_logits).any() or torch.isinf(allowed_logits).any():
            print(f"[Warning] Invalid logits detected in confidence computation for {self.model_name}")
            option_probs = {label: float('nan') for label in labels}
            confidence = float('nan')
        else:
            # Compute softmax over allowed tokens only
            probs = F.softmax(allowed_logits, dim=0)
            
            # Build option probabilities dictionary
            option_probs = {label: probs[i].item() 
                            for i, label in enumerate(labels)}
            
            # Confidence = probability of the generated token
            gen_idx = allowed_ids.index(gen_token_id)
            confidence = probs[gen_idx].item()
        
        return answer, confidence, option_probs

    def batch_generate(self, prompts, show_progress=True):
        """
        Generate answers for multiple prompts.
        
        Args:
            prompts: list of prompt strings
            show_progress: whether to print progress
            
        Returns:
            list of generated responses (format depends on mode)
        """
        results = []
        for i, prompt in enumerate(prompts):
            if show_progress:
                print(f"[{i+1}/{len(prompts)}] Processing...", end="\r")
            results.append(self.generate(prompt))
        
        if show_progress:
            print(f"[{len(prompts)}/{len(prompts)}] Complete!     ")
        
        return results

    def get_model_info(self):
        """
        Return dictionary with model metadata.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.model_dtype),
            "task_type": self.task_type,
            "mode": self.mode,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "AB_token_ids": self.AB_IDS,
            "ABCD_token_ids": self.ABCD_IDS
        }

    def __repr__(self):
        return (f"MedicalLLMWrapper(model='{self.model_name}', "
                f"device='{self.device}', dtype={self.model_dtype}, "
                f"task='{self.task_type}', mode='{self.mode}')")


# Convenience function for quick model loading
def load_medical_llm(model_name, device="cuda", token=None, torch_dtype=None):
    """
    Quick helper to load a medical LLM wrapper.
    
    Args:
        model_name: HuggingFace model ID
        device: 'cuda', 'cpu', or 'auto'
        token: HF auth token for gated models
        torch_dtype: torch.float16, torch.float32, or None (auto)
    
    Returns:
        MedicalLLMWrapper instance
    
    Example:
        >>> wrapper = load_medical_llm("FreedomIntelligence/Apollo-2B")
        >>> wrapper.set_task("mcq")
        >>> response = wrapper.generate("What is the treatment for diabetes? A) Insulin B) Aspirin")
    """
    return MedicalLLMWrapper(model_name, device=device, token=token, torch_dtype=torch_dtype)
