"""
Model-Agnostic Integrated Gradients Implementation for Text Models

This module provides a flexible implementation of Integrated Gradients (Sundararajan et al., 2017)
for any HuggingFace transformer model. Supports causal language models including:
- Medical LLMs: MedGemma, BioMistral, BioMedLM
- General LLMs: Apollo, Mistral, Llama, GPT, etc.

Author: DATA 298A Project
Date: December 2025
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Dict, Optional, Callable
from tqdm import tqdm
import warnings


class IntegratedGradientsConfig:
    """Configuration for Integrated Gradients computation."""
    
    def __init__(
        self,
        n_steps: int = 50,
        batch_size: int = 8,
        baseline_type: str = 'zero',
        internal_batch_size: int = 1,
        verbose: bool = True,
        validate_convergence: bool = False,
        convergence_delta: float = 0.01
    ):
        """
        Args:
            n_steps: Number of interpolation steps (more = more accurate but slower)
            batch_size: Number of steps to process together
            baseline_type: Type of baseline ('zero', 'pad', 'unk', 'custom')
            internal_batch_size: Batch size for processing multiple prompts
            verbose: Whether to show progress bars
            validate_convergence: Whether to check if n_steps is sufficient
            convergence_delta: Threshold for convergence validation
        """
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.baseline_type = baseline_type
        self.internal_batch_size = internal_batch_size
        self.verbose = verbose
        self.validate_convergence = validate_convergence
        self.convergence_delta = convergence_delta


class IntegratedGradients:
    """
    Model-agnostic Integrated Gradients implementation for HuggingFace transformers.
    
    This class computes token-level attribution scores by integrating gradients
    along a path from a baseline input to the actual input.
    
    Compatible with any HuggingFace causal language model that supports:
    - .get_input_embeddings() method
    - inputs_embeds parameter in forward pass
    - .logits output
    
    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
        >>> ig = IntegratedGradients(model, tokenizer)
        >>> result = ig.attribute(prompt="Patient has fever", target_token="infection")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: Optional[str] = None,
        config: Optional[IntegratedGradientsConfig] = None
    ):
        """
        Initialize Integrated Gradients explainer.
        
        Args:
            model: HuggingFace model (must support inputs_embeds)
            tokenizer: Corresponding tokenizer
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            config: Configuration object (uses defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or IntegratedGradientsConfig()
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Validate model compatibility
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model is compatible with IG."""
        # Check for required methods
        if not hasattr(self.model, 'get_input_embeddings'):
            raise ValueError(
                "Model must have get_input_embeddings() method. "
                "Most HuggingFace models support this."
            )
        
        # Verify the model can accept inputs_embeds
        try:
            dummy_ids = torch.tensor([[1, 2, 3]], device=self.device)
            dummy_embeds = self.model.get_input_embeddings()(dummy_ids)
            _ = self.model(inputs_embeds=dummy_embeds)
        except Exception as e:
            raise ValueError(
                f"Model does not support inputs_embeds parameter: {e}"
            )
    
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input token IDs.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
        
        Returns:
            Embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.model.get_input_embeddings()(input_ids)
    
    def _forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        target_token_id: int,
        return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass using embeddings instead of token IDs.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            target_token_id: Token ID to compute logit for
            return_logits: If True, return full logits along with target logit
        
        Returns:
            Target logit value, or (target_logit, all_logits) if return_logits=True
        """
        # Simply pass embeddings through the model
        # If model is fp16 and embeddings are fp32, PyTorch will handle conversion
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits[:, -1, :]
        
        target_logit = logits[:, target_token_id]
        
        if return_logits:
            return target_logit, logits
        return target_logit
    
    def _create_baseline(
        self,
        input_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        baseline_type: str,
        custom_baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create baseline embeddings for IG computation.
        
        Args:
            input_embeddings: Original input embeddings
            input_ids: Original input token IDs
            baseline_type: Type of baseline ('zero', 'pad', 'unk', 'custom')
            custom_baseline: Custom baseline embeddings (required if baseline_type='custom')
        
        Returns:
            Baseline embeddings with same shape as input_embeddings
        """
        if baseline_type == 'zero':
            return torch.zeros_like(input_embeddings)
        
        elif baseline_type == 'pad':
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                warnings.warn("No pad token found, using 0")
                pad_token_id = 0
            baseline_ids = torch.full_like(input_ids, pad_token_id)
            with torch.no_grad():
                return self._get_embeddings(baseline_ids)
        
        elif baseline_type == 'unk':
            unk_token_id = self.tokenizer.unk_token_id
            if unk_token_id is None:
                warnings.warn("No unk token found, using 0")
                unk_token_id = 0
            baseline_ids = torch.full_like(input_ids, unk_token_id)
            with torch.no_grad():
                return self._get_embeddings(baseline_ids)
        
        elif baseline_type == 'custom':
            if custom_baseline is None:
                raise ValueError("custom_baseline must be provided when baseline_type='custom'")
            if custom_baseline.shape != input_embeddings.shape:
                raise ValueError(
                    f"Custom baseline shape {custom_baseline.shape} "
                    f"doesn't match input shape {input_embeddings.shape}"
                )
            return custom_baseline
        
        else:
            raise ValueError(
                f"Unknown baseline_type: {baseline_type}. "
                f"Choose from: 'zero', 'pad', 'unk', 'custom'"
            )
    
    def _compute_attributions(
        self,
        input_ids: torch.Tensor,
        target_token_id: int,
        baseline_type: str = 'zero',
        n_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        custom_baseline: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Core IG computation (internal method).
        
        Args:
            input_ids: Token IDs [1, seq_len]
            target_token_id: Token ID to attribute
            baseline_type: Baseline type
            n_steps: Number of steps (overrides config)
            batch_size: Batch size (overrides config)
            custom_baseline: Custom baseline if needed
        
        Returns:
            (attributions, tokens, metadata)
        """
        n_steps = n_steps or self.config.n_steps
        batch_size = batch_size or self.config.batch_size
        
        # Get input embeddings
        with torch.no_grad():
            input_embeddings = self._get_embeddings(input_ids)
        
        # Create baseline
        baseline_embeddings = self._create_baseline(
            input_embeddings, input_ids, baseline_type, custom_baseline
        )
        
        # Store on CPU to avoid memory issues
        input_embeddings_cpu = input_embeddings.cpu()
        baseline_embeddings_cpu = baseline_embeddings.cpu()
        
        # Accumulate gradients
        accumulated_gradients = torch.zeros_like(input_embeddings_cpu)
        nan_count = 0
        
        # Progress bar setup
        iterator = range(0, n_steps, batch_size)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Computing IG", leave=False)
        
        for i in iterator:
            batch_end = min(i + batch_size, n_steps)
            batch_alphas = torch.linspace(i / n_steps, batch_end / n_steps, batch_end - i)
            
            for alpha in batch_alphas:
                alpha_val = alpha.item()
                
                # Create interpolated embeddings on CPU
                interpolated_embeddings_cpu = (
                    baseline_embeddings_cpu + 
                    alpha_val * (input_embeddings_cpu - baseline_embeddings_cpu)
                )
                
                # Move to device and enable gradients
                interpolated_embeddings = interpolated_embeddings_cpu.to(self.device)
                interpolated_embeddings.requires_grad_(True)
                
                # Forward pass
                target_logit = self._forward_from_embeddings(
                    interpolated_embeddings,
                    target_token_id
                )
                
                # Check for NaN
                if torch.isnan(target_logit).any():
                    nan_count += 1
                    if nan_count <= 3:  # Only warn first few times
                        warnings.warn(f"NaN detected in forward pass at α={alpha_val:.4f}")
                    continue
                
                # Backward pass
                target_logit.backward()
                
                # Accumulate gradients
                if interpolated_embeddings.grad is not None:
                    grad_cpu = interpolated_embeddings.grad.detach().cpu()
                    
                    if torch.isnan(grad_cpu).any():
                        nan_count += 1
                        if nan_count <= 3:
                            warnings.warn(f"NaN in gradients at α={alpha_val:.4f}")
                        continue
                    
                    accumulated_gradients += grad_cpu
                
                # Cleanup
                self.model.zero_grad()
                del target_logit, interpolated_embeddings, interpolated_embeddings_cpu
            
            # Periodic cache clearing
            if torch.cuda.is_available() and i % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
        
        # Compute final attributions
        avg_gradients = accumulated_gradients / n_steps
        attributions = (input_embeddings_cpu - baseline_embeddings_cpu) * avg_gradients
        token_attributions = attributions.sum(dim=-1).squeeze(0).numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        
        # Metadata
        metadata = {
            'n_steps': n_steps,
            'baseline_type': baseline_type,
            'nan_count': nan_count,
            'max_attribution': float(np.max(np.abs(token_attributions))),
            'mean_attribution': float(np.mean(np.abs(token_attributions))),
            'sum_attribution': float(np.sum(token_attributions))
        }
        
        # Final cleanup
        del accumulated_gradients, avg_gradients, attributions, input_embeddings, baseline_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check for NaN in final result
        if np.isnan(token_attributions).any():
            warnings.warn(
                f"Final attributions contain {np.isnan(token_attributions).sum()} NaN values!"
            )
        
        return token_attributions, tokens, metadata
    
    def attribute(
        self,
        prompt: str,
        target_token: str,
        baseline_type: Optional[str] = None,
        n_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_metadata: bool = False,
        custom_baseline: Optional[torch.Tensor] = None
    ) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[str], Dict]]:
        """
        Compute attribution scores for each token in the prompt.
        
        This is the main public API for computing Integrated Gradients.
        
        Args:
            prompt: Input text to explain
            target_token: Token to compute attributions for (e.g., 'yes', 'A', 'infection')
            baseline_type: Baseline type ('zero', 'pad', 'unk', 'custom')
            n_steps: Number of integration steps (overrides config)
            batch_size: Batch size (overrides config)
            return_metadata: If True, return metadata dict as third element
            custom_baseline: Custom baseline embeddings (if baseline_type='custom')
        
        Returns:
            attributions: Array of attribution scores for each token
            tokens: List of token strings
            metadata: (optional) Dict with computation metadata
        
        Example:
            >>> attributions, tokens = ig.attribute(
            ...     prompt="Patient has diabetes",
            ...     target_token="insulin"
            ... )
        """
        baseline_type = baseline_type or self.config.baseline_type
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # Get target token ID
        target_tokens = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_tokens) == 0:
            raise ValueError(f"Target token '{target_token}' could not be tokenized")
        target_token_id = target_tokens[0]
        
        if len(target_tokens) > 1:
            warnings.warn(
                f"Target token '{target_token}' was tokenized into {len(target_tokens)} tokens. "
                f"Using first token: {self.tokenizer.decode([target_token_id])}"
            )
        
        # Compute attributions
        attributions, tokens, metadata = self._compute_attributions(
            input_ids=input_ids,
            target_token_id=target_token_id,
            baseline_type=baseline_type,
            n_steps=n_steps,
            batch_size=batch_size,
            custom_baseline=custom_baseline
        )
        
        if return_metadata:
            return attributions, tokens, metadata
        return attributions, tokens
    
    def attribute_multiple_targets(
        self,
        prompt: str,
        target_tokens: List[str],
        baseline_type: Optional[str] = None,
        n_steps: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """
        Compute attributions for multiple target tokens at once.
        
        Useful for comparing attributions across different answer options
        (e.g., A, B, C, D for multiple choice).
        
        Args:
            prompt: Input text
            target_tokens: List of target tokens to compute attributions for
            baseline_type: Baseline type
            n_steps: Number of steps
            batch_size: Batch size
        
        Returns:
            Dict mapping target_token -> (attributions, tokens)
        
        Example:
            >>> results = ig.attribute_multiple_targets(
            ...     prompt="What is the diagnosis?",
            ...     target_tokens=['A', 'B', 'C', 'D']
            ... )
        """
        results = {}
        
        iterator = target_tokens
        if self.config.verbose:
            iterator = tqdm(target_tokens, desc="Computing for multiple targets")
        
        for target in iterator:
            attributions, tokens = self.attribute(
                prompt=prompt,
                target_token=target,
                baseline_type=baseline_type,
                n_steps=n_steps,
                batch_size=batch_size
            )
            results[target] = (attributions, tokens)
        
        return results
    
    def save_attributions(
        self,
        filepath: str,
        attributions: np.ndarray,
        tokens: List[str],
        metadata: Optional[Dict] = None,
        prompt: Optional[str] = None,
        target_token: Optional[str] = None
    ):
        """
        Save attributions to file (JSON or NPZ format).
        
        Args:
            filepath: Path to save file (.json or .npz)
            attributions: Attribution scores
            tokens: Token strings
            metadata: Optional metadata dict
            prompt: Optional original prompt
            target_token: Optional target token
        """
        import json
        
        if filepath.endswith('.json'):
            data = {
                'attributions': attributions.tolist(),
                'tokens': tokens,
                'metadata': metadata or {},
                'prompt': prompt,
                'target_token': target_token
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif filepath.endswith('.npz'):
            np.savez(
                filepath,
                attributions=attributions,
                tokens=np.array(tokens),
                metadata=str(metadata or {}),
                prompt=prompt or '',
                target_token=target_token or ''
            )
        else:
            raise ValueError("Filepath must end with .json or .npz")


def load_model_for_ig(
    model_name: str,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    torch_dtype=torch.float16,
    **model_kwargs
):
    """
    Convenience function to load a model and tokenizer for IG.
    
    IMPORTANT: Some models (e.g., google/medgemma-4b-it) produce NaN values
    in float16. If you encounter all-zero attributions or NaN warnings,
    try loading with torch_dtype=torch.float32 instead.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load on
        trust_remote_code: Whether to trust remote code
        torch_dtype: Data type for model weights (use float32 if fp16 produces NaNs)
        **model_kwargs: Additional arguments for model loading
    
    Returns:
        (model, tokenizer) tuple
    
    Example:
        >>> # Standard loading (fp16)
        >>> model, tokenizer = load_model_for_ig("microsoft/BioGPT")
        >>> 
        >>> # If you get NaN issues, use fp32
        >>> model, tokenizer = load_model_for_ig(
        ...     "google/medgemma-4b-it",
        ...     torch_dtype=torch.float32
        ... )
        >>> ig = IntegratedGradients(model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warn about known problematic models
    if 'medgemma' in model_name.lower() and torch_dtype != torch.float32:
        warnings.warn(
            "\n" + "="*70 + "\n"
            "WARNING: MedGemma models REQUIRE float32 for Integrated Gradients!\n"
            "They produce NaN values in float16/bfloat16 during gradient computation.\n"
            "Automatically switching to torch_dtype=torch.float32\n"
            "="*70,
            UserWarning
        )
        torch_dtype = torch.float32  # Force fp32 for MedGemma
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device if device == 'auto' else None,
        **model_kwargs
    )
    
    if device != 'auto':
        model = model.to(device)
    
    return model, tokenizer
