"""
TokenSHAP adapter for MedicalLLMWrapper.

This module provides an interface between the TokenSHAP implementation
and the MedicalLLMWrapper, allowing TokenSHAP explanations for medical QA tasks.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

# Import TokenSHAP from the sibling package
from ..token_shap import TokenSHAP
from .medical_llm_wrapper import MedicalLLMWrapper


class MedicalTokenSHAP:
    """
    Adapter to use TokenSHAP with MedicalLLMWrapper.
    
    Provides a consistent interface for computing SHAP values on medical QA tasks.
    """
    
    def __init__(
        self,
        wrapper: MedicalLLMWrapper,
        n_samples: int = 100,
        verbose: bool = True
    ):
        """
        Initialize TokenSHAP with a medical LLM wrapper.
        
        Args:
            wrapper: MedicalLLMWrapper instance
            n_samples: Default number of SHAP samples
            verbose: Show progress information
        """
        self.wrapper = wrapper
        self.n_samples = n_samples
        self.verbose = verbose
        
        # Validate task type
        if self.wrapper.task_type not in ['yn', 'mcq']:
            raise ValueError(
                f"TokenSHAP requires task type 'yn' or 'mcq', "
                f"got '{self.wrapper.task_type}'. "
                f"Please call wrapper.set_task('mcq') or wrapper.set_task('yn') first."
            )
        
        # Initialize TokenSHAP with wrapper's model and tokenizer
        self.token_shap = TokenSHAP(
            model=wrapper.model,
            tokenizer=wrapper.tokenizer,
            device=wrapper.device
        )
    
    def explain(
        self,
        prompt: str,
        target_class: str,
        n_samples: Optional[int] = None
    ) -> Dict:
        """
        Compute SHAP values for a medical QA prompt.
        
        Args:
            prompt: Medical question text
            target_class: Answer choice to explain (e.g., 'A', 'B', 'C', 'D')
            n_samples: Number of SHAP samples (overrides default)
        
        Returns:
            Dictionary with:
                - 'tokens': List of token strings
                - 'shap_values': SHAP attribution values per token
                - 'prediction': Model's predicted answer
                - 'target_probability': Probability of target class
        """
        # Validate target class
        if target_class not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"target_class must be A/B/C/D, got {target_class}")
        
        # Check task type compatibility
        if self.wrapper.task_type == 'yn' and target_class not in ['A', 'B']:
            raise ValueError(f"For Y/N tasks, target_class must be A or B, got {target_class}")
        
        # Get target token ID
        if self.wrapper.task_type == 'yn':
            target_token_id = self.wrapper.AB_IDS[ord(target_class) - ord('A')]
        else:
            target_token_id = self.wrapper.ABCD_IDS[ord(target_class) - ord('A')]
        
        # Use provided n_samples or default
        samples = n_samples if n_samples is not None else self.n_samples
        
        # Compute SHAP values
        if self.verbose:
            print(f"Computing TokenSHAP with {samples} samples...")
        
        shap_values = self.token_shap.explain(
            text=prompt,
            target_token_id=target_token_id,
            n_samples=samples
        )
        
        # Get model prediction
        inputs = self.wrapper.tokenizer(prompt, return_tensors='pt').to(self.wrapper.device)
        
        with torch.no_grad():
            outputs = self.wrapper.model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            # Get probabilities over answer options
            if self.wrapper.task_type == 'yn':
                option_ids = self.wrapper.AB_IDS
                labels = ['A', 'B']
            else:
                option_ids = self.wrapper.ABCD_IDS
                labels = ['A', 'B', 'C', 'D']
            
            option_logits = torch.tensor([logits[i].item() for i in option_ids], 
                                        dtype=torch.float32)
            probs = F.softmax(option_logits, dim=0)
            
            pred_idx = probs.argmax().item()
            prediction = labels[pred_idx]
            target_prob = probs[ord(target_class) - ord('A')].item()
        
        # Get tokens
        tokens = self.wrapper.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return {
            'tokens': tokens,
            'shap_values': shap_values,
            'prediction': prediction,
            'target_probability': target_prob,
            'target_class': target_class
        }
    
    def explain_batch(
        self,
        prompts: List[str],
        target_classes: List[str],
        n_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Compute SHAP values for multiple prompts.
        
        Args:
            prompts: List of medical question texts
            target_classes: List of target classes (one per prompt)
            n_samples: Number of SHAP samples
        
        Returns:
            List of SHAP attribution dictionaries
        """
        if len(prompts) != len(target_classes):
            raise ValueError("prompts and target_classes must have same length")
        
        results = []
        for i, (prompt, target) in enumerate(zip(prompts, target_classes), 1):
            if self.verbose:
                print(f"\n[{i}/{len(prompts)}] Processing...")
            result = self.explain(prompt, target, n_samples)
            results.append(result)
        
        return results


def explain_with_tokenshap(
    wrapper: MedicalLLMWrapper,
    prompt: str,
    target_class: str,
    n_samples: int = 100
) -> Dict:
    """
    Convenience function to explain a prediction with TokenSHAP.
    
    Args:
        wrapper: MedicalLLMWrapper instance
        prompt: Medical question text
        target_class: Answer to explain ('A', 'B', 'C', or 'D')
        n_samples: Number of SHAP samples
    
    Returns:
        SHAP attribution dictionary
    """
    shap = MedicalTokenSHAP(wrapper, n_samples=n_samples)
    return shap.explain(prompt, target_class)
