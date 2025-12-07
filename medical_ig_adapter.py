"""
Medical Integrated Gradients - Works with MedicalLLMWrapper

Simple, clean implementation of Integrated Gradients for medical LLM interpretability.
Compatible with any model loaded through MedicalLLMWrapper.

Usage:
    from medical_llm_wrapper import load_medical_llm
    from medical_integrated_gradients import MedicalIntegratedGradients
    
    wrapper = load_medical_llm("google/medgemma-4b-it")
    ig = MedicalIntegratedGradients(wrapper)
    attributions = ig.attribute("Patient has chest pain and fever", target_class="A")
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm


class MedicalIntegratedGradients:
    """
    Integrated Gradients for medical LLMs using MedicalLLMWrapper.
    
    Computes token-level attribution scores for medical QA predictions.
    Works with Y/N and MCQ tasks.
    """
    
    def __init__(
        self,
        wrapper,
        n_steps: int = 50,
        baseline_type: str = 'pad',
        verbose: bool = True
    ):
        """
        Initialize IG explainer.
        
        Args:
            wrapper: MedicalLLMWrapper instance
            n_steps: Number of interpolation steps (more = more accurate)
            baseline_type: 'pad' (use pad token) or 'zero' (zero embeddings)
            verbose: Show progress bars
        """
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = wrapper.device
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.verbose = verbose
        
        # Set model to eval mode
        self.model.eval()
        
    def _get_baseline_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Create baseline embeddings.
        
        Args:
            embeddings: Input embeddings [batch, seq_len, embed_dim]
            
        Returns:
            Baseline embeddings of same shape
        """
        if self.baseline_type == 'zero':
            return torch.zeros_like(embeddings)
        elif self.baseline_type == 'pad':
            # Use pad token embedding as baseline
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            pad_embedding = self.model.get_input_embeddings()(
                torch.tensor([pad_id], device=self.device)
            )
            return pad_embedding.expand_as(embeddings)
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def _get_target_logit(
        self,
        embeddings: torch.Tensor,
        target_token_id: int
    ) -> torch.Tensor:
        """
        Forward pass to get logit for target token.
        
        Args:
            embeddings: Input embeddings [batch, seq_len, embed_dim]
            target_token_id: Token ID to get logit for (e.g., token for 'A')
            
        Returns:
            Logit for target token [batch]
        """
        with torch.set_grad_enabled(embeddings.requires_grad):
            outputs = self.model(inputs_embeds=embeddings)
            logits = outputs.logits[:, -1, :]  # Last token logits
            target_logit = logits[:, target_token_id]
        return target_logit
    
    def _compute_gradients(
        self,
        embeddings: torch.Tensor,
        target_token_id: int
    ) -> torch.Tensor:
        """
        Compute gradients of target logit w.r.t. embeddings.
        
        Args:
            embeddings: Input embeddings [batch, seq_len, embed_dim]
            target_token_id: Target token ID
            
        Returns:
            Gradients [batch, seq_len, embed_dim]
        """
        # Create a new tensor with gradients enabled
        embeddings_for_grad = embeddings.clone().detach().requires_grad_(True)
        
        # Forward pass
        target_logit = self._get_target_logit(embeddings_for_grad, target_token_id)
        
        # Backward pass
        self.model.zero_grad()
        target_logit.sum().backward()
        
        # Get gradients
        grads = embeddings_for_grad.grad.clone()
        
        return grads
    
    def attribute(
        self,
        prompt: str,
        target_class: str,
        return_convergence_delta: bool = False
    ) -> Dict:
        """
        Compute token attributions using Integrated Gradients.
        
        Args:
            prompt: Input prompt (e.g., "Patient has chest pain. Diagnosis? A) MI B) PE")
            target_class: Target answer class ('A', 'B', 'C', or 'D')
            return_convergence_delta: If True, return convergence diagnostic
            
        Returns:
            Dictionary with:
                - 'tokens': List of token strings
                - 'attributions': Attribution score per token (numpy array)
                - 'prediction': Model's predicted answer
                - 'target_probability': Probability of target class
                - 'convergence_delta': (optional) Convergence metric
        """
        # Validate target class
        if target_class not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"target_class must be A/B/C/D, got {target_class}")
        
        # Validate task type
        if self.wrapper.task_type not in ['yn', 'mcq']:
            raise ValueError(
                f"IG only supports 'yn' and 'mcq' tasks. "
                f"Current task type is '{self.wrapper.task_type}'. "
                f"Please call wrapper.set_task('mcq') or wrapper.set_task('yn') first."
            )
        
        # Get target token ID
        if self.wrapper.task_type == 'yn':
            if target_class not in ['A', 'B']:
                raise ValueError(f"For Y/N tasks, target_class must be A or B, got {target_class}")
            target_token_id = self.wrapper.AB_IDS[ord(target_class) - ord('A')]
        elif self.wrapper.task_type == 'mcq':
            target_token_id = self.wrapper.ABCD_IDS[ord(target_class) - ord('A')]
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        
        # Get embeddings (don't detach - we need gradients to flow)
        embed_layer = self.model.get_input_embeddings()
        input_embeddings = embed_layer(input_ids)
        
        # Get baseline
        baseline_embeddings = self._get_baseline_embeddings(input_embeddings)
        
        # Compute path
        path_embeddings = []
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)
        
        iterator = tqdm(alphas, desc="Computing IG", disable=not self.verbose)
        
        # Compute gradients along path
        all_grads = []
        for alpha in iterator:
            # Interpolate
            interpolated = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
            
            # Compute gradient
            grads = self._compute_gradients(interpolated, target_token_id)
            all_grads.append(grads)
        
        # Stack and average gradients
        all_grads = torch.stack(all_grads)  # [n_steps+1, batch, seq_len, embed_dim]
        avg_grads = all_grads.mean(dim=0)  # [batch, seq_len, embed_dim]
        
        # Multiply by input difference (IG formula)
        integrated_grads = (input_embeddings - baseline_embeddings) * avg_grads
        
        # Sum over embedding dimension to get per-token attribution
        token_attributions = integrated_grads.sum(dim=-1).squeeze(0)  # [seq_len]
        
        # Detach from graph since we're done computing gradients
        token_attributions = token_attributions.detach()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(inputs_embeds=input_embeddings)
            logits = outputs.logits[0, -1, :]
            
            # Get probabilities over answer options
            if self.wrapper.task_type == 'yn':
                option_ids = self.wrapper.AB_IDS
                labels = ['A', 'B']
            else:
                option_ids = self.wrapper.ABCD_IDS
                labels = ['A', 'B', 'C', 'D']
            
            option_logits = logits[option_ids]
            probs = F.softmax(option_logits, dim=0)
            
            pred_idx = probs.argmax().item()
            prediction = labels[pred_idx]
            target_prob = probs[ord(target_class) - ord('A')].item()
        
        result = {
            'tokens': tokens,
            'attributions': token_attributions.cpu().numpy(),
            'prediction': prediction,
            'target_probability': target_prob,
            'target_class': target_class
        }
        
        # Convergence diagnostic (optional)
        if return_convergence_delta:
            # Completeness check: sum of attributions should approximate
            # (f(x) - f(baseline))
            with torch.no_grad():
                f_input = self._get_target_logit(input_embeddings, target_token_id).item()
                f_baseline = self._get_target_logit(baseline_embeddings, target_token_id).item()
                expected_sum = f_input - f_baseline
                actual_sum = token_attributions.sum().item()
                delta = abs(expected_sum - actual_sum)
            
            result['convergence_delta'] = delta
            result['expected_sum'] = expected_sum
            result['actual_sum'] = actual_sum
        
        return result
    
    def attribute_batch(
        self,
        prompts: List[str],
        target_classes: List[str]
    ) -> List[Dict]:
        """
        Compute attributions for multiple prompts.
        
        Args:
            prompts: List of prompts
            target_classes: List of target classes (one per prompt)
            
        Returns:
            List of attribution dictionaries
        """
        if len(prompts) != len(target_classes):
            raise ValueError("prompts and target_classes must have same length")
        
        results = []
        for prompt, target in zip(prompts, target_classes):
            result = self.attribute(prompt, target)
            results.append(result)
        
        return results


def visualize_attributions(
    tokens: List[str],
    attributions: np.ndarray,
    prediction: str,
    target_class: str,
    title: Optional[str] = None,
    normalize: bool = True
):
    """
    Print colored attribution visualization.
    
    Args:
        tokens: List of token strings
        attributions: Attribution scores per token
        prediction: Model's prediction
        target_class: Target class being explained
        title: Optional title
        normalize: Normalize attributions to [0, 1]
    """
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
    print(f"  Prediction: {prediction} | Explaining: {target_class}")
    print("=" * 80)
    
    # Normalize
    if normalize:
        attr_min = attributions.min()
        attr_max = attributions.max()
        if attr_max - attr_min > 0:
            attributions = (attributions - attr_min) / (attr_max - attr_min)
    
    # Color scale
    def get_color(score):
        """Get color based on attribution score."""
        if score < 0:
            # Negative attribution (blue)
            intensity = min(abs(score), 1.0)
            return f"\033[48;2;{int(255*(1-intensity))};{int(255*(1-intensity))};255m"
        else:
            # Positive attribution (red)
            intensity = min(score, 1.0)
            return f"\033[48;2;255;{int(255*(1-intensity))};{int(255*(1-intensity))}m"
    
    reset = "\033[0m"
    
    # Print tokens with colors
    print("\n  ", end="")
    for token, score in zip(tokens, attributions):
        color = get_color(score)
        # Clean up token display
        display_token = token.replace('▁', ' ').replace('Ġ', ' ')
        print(f"{color}{display_token}{reset}", end="")
    
    print("\n\n  Legend: ", end="")
    print(f"\033[48;2;255;150;150mPositive (supports answer)\033[0m  ", end="")
    print(f"\033[48;2;150;150;255mNegative (against answer)\033[0m")
    print("=" * 80)


# Convenience function
def explain_medical_prediction(
    wrapper,
    prompt: str,
    target_class: str,
    n_steps: int = 50,
    visualize: bool = True
) -> Dict:
    """
    One-liner to explain a medical LLM prediction.
    
    Args:
        wrapper: MedicalLLMWrapper instance
        prompt: Input prompt
        target_class: Answer to explain ('A', 'B', 'C', or 'D')
        n_steps: Number of IG steps
        visualize: Print colored visualization
        
    Returns:
        Attribution dictionary
    """
    # Ensure wrapper is in the right mode
    if wrapper.task_type not in ['yn', 'mcq']:
        print(f"[Warning] Wrapper task type is '{wrapper.task_type}', auto-setting to 'mcq'")
        wrapper.set_task('mcq')
    
    ig = MedicalIntegratedGradients(wrapper, n_steps=n_steps)
    
    try:
        result = ig.attribute(prompt, target_class, return_convergence_delta=True)
    except Exception as e:
        print(f"\n[ERROR] Attribution failed: {e}")
        print(f"\nDebugging info:")
        print(f"  - Wrapper task type: {wrapper.task_type}")
        print(f"  - Wrapper mode: {wrapper.mode}")
        print(f"  - Target class: {target_class}")
        print(f"  - Model dtype: {wrapper.model_dtype}")
        raise
    
    if visualize:
        visualize_attributions(
            result['tokens'],
            result['attributions'],
            result['prediction'],
            result['target_class'],
            title=f"Integrated Gradients Explanation (Δ={result['convergence_delta']:.4f})"
        )
    
    return result
