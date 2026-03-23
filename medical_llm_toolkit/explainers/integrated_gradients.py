"""
Medical Integrated Gradients - Works with MedicalLLMWrapper

Integrated Gradients (Sundararajan et al., 2017) for medical LLM interpretability.
Compatible with any model loaded through MedicalLLMWrapper.

Usage:
    from medical_llm_toolkit import MedicalLLMWrapper, load_medical_llm
    from medical_llm_toolkit.explainers import MedicalIntegratedGradients

    wrapper = load_medical_llm("google/medgemma-4b-it")
    wrapper.set_task("mcq")
    ig = MedicalIntegratedGradients(wrapper)
    result = ig.attribute("Patient has chest pain and fever", target_class="A")
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import warnings


class MedicalIntegratedGradients:
    """
    Integrated Gradients for medical LLMs using MedicalLLMWrapper.

    Computes token-level attribution scores for medical QA predictions.
    Works with Y/N and MCQ tasks.

    Improvements over the basic implementation:
    - NaN detection and graceful skipping of corrupted gradient steps
    - CPU offloading of accumulated gradients to reduce VRAM usage
    - Periodic GPU cache clearing for long sequences
    - Support for 'zero', 'pad', 'unk', and 'custom' baseline types
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
            baseline_type: Baseline embedding type:
                'pad'    - pad token embedding (default)
                'zero'   - zero vector
                'unk'    - unknown token embedding
                'custom' - user-provided (pass via attribute())
            verbose: Show progress bars
        """
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = wrapper.device
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.verbose = verbose

        self.model.eval()

    def _get_baseline_embeddings(
        self,
        input_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        custom_baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create baseline embeddings.

        Args:
            input_embeddings: Input embeddings [batch, seq_len, embed_dim]
            input_ids: Input token IDs [batch, seq_len]
            custom_baseline: User-provided baseline (required when baseline_type='custom')

        Returns:
            Baseline embeddings of same shape
        """
        if self.baseline_type == 'zero':
            return torch.zeros_like(input_embeddings)

        elif self.baseline_type == 'pad':
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                warnings.warn("No pad token found, falling back to eos_token_id")
                pad_id = self.tokenizer.eos_token_id
            baseline_ids = torch.full_like(input_ids, pad_id)
            with torch.no_grad():
                return self.model.get_input_embeddings()(baseline_ids)

        elif self.baseline_type == 'unk':
            unk_id = self.tokenizer.unk_token_id
            if unk_id is None:
                warnings.warn("No unk token found, falling back to pad baseline")
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                baseline_ids = torch.full_like(input_ids, pad_id)
            else:
                baseline_ids = torch.full_like(input_ids, unk_id)
            with torch.no_grad():
                return self.model.get_input_embeddings()(baseline_ids)

        elif self.baseline_type == 'custom':
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
                f"Unknown baseline type: '{self.baseline_type}'. "
                f"Choose from: 'zero', 'pad', 'unk', 'custom'"
            )

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
            logits = outputs.logits[:, -1, :]
            target_logit = logits[:, target_token_id]
        return target_logit

    def attribute(
        self,
        prompt: str,
        target_class: str,
        return_convergence_delta: bool = False,
        custom_baseline: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute token attributions using Integrated Gradients.

        Gradients are accumulated on CPU to reduce VRAM usage. NaN values
        in forward or backward passes are detected and skipped with a warning.

        Args:
            prompt: Input prompt (e.g., "Patient has chest pain. Diagnosis? A) MI B) PE")
            target_class: Target answer class ('A', 'B', 'C', or 'D')
            return_convergence_delta: If True, return convergence diagnostic
            custom_baseline: Custom baseline embeddings (only used when baseline_type='custom')

        Returns:
            Dictionary with:
                - 'tokens': List of token strings
                - 'attributions': Attribution score per token (numpy array)
                - 'prediction': Model's predicted answer
                - 'target_class': Target class being explained
                - 'target_probability': Probability of target class
                - 'convergence_delta': (optional) Convergence metric
                - 'expected_sum': (optional) f(input) - f(baseline)
                - 'actual_sum': (optional) Sum of attributions
                - 'nan_count': Number of gradient steps skipped due to NaN
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

        # Get embeddings
        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            input_embeddings = embed_layer(input_ids)

        # Get baseline
        baseline_embeddings = self._get_baseline_embeddings(
            input_embeddings, input_ids, custom_baseline
        )

        # Store reference copies on CPU to reduce VRAM
        input_embeddings_cpu = input_embeddings.cpu()
        baseline_embeddings_cpu = baseline_embeddings.cpu()

        # Accumulate gradients on CPU
        accumulated_gradients = torch.zeros_like(input_embeddings_cpu)
        nan_count = 0
        alphas = torch.linspace(0, 1, self.n_steps + 1)

        iterator = tqdm(range(len(alphas)), desc="Computing IG", disable=not self.verbose)

        for step_idx in iterator:
            alpha_val = alphas[step_idx].item()

            # Interpolate on CPU, then move to device
            interpolated_cpu = (
                baseline_embeddings_cpu
                + alpha_val * (input_embeddings_cpu - baseline_embeddings_cpu)
            )
            interpolated = interpolated_cpu.to(self.device).requires_grad_(True)

            # Forward pass
            target_logit = self._get_target_logit(interpolated, target_token_id)

            # Check for NaN in forward pass
            if torch.isnan(target_logit).any():
                nan_count += 1
                if nan_count <= 3:
                    warnings.warn(f"NaN in forward pass at α={alpha_val:.4f}")
                self.model.zero_grad()
                del target_logit, interpolated, interpolated_cpu
                continue

            # Backward pass
            self.model.zero_grad()
            target_logit.sum().backward()

            # Check for NaN in gradients
            if interpolated.grad is not None:
                grad_cpu = interpolated.grad.detach().cpu()

                if torch.isnan(grad_cpu).any():
                    nan_count += 1
                    if nan_count <= 3:
                        warnings.warn(f"NaN in gradients at α={alpha_val:.4f}")
                    del target_logit, interpolated, interpolated_cpu, grad_cpu
                    continue

                accumulated_gradients += grad_cpu
                del grad_cpu

            # Cleanup per step
            del target_logit, interpolated, interpolated_cpu

            # Periodic GPU cache clearing
            if torch.cuda.is_available() and step_idx % 10 == 0:
                torch.cuda.empty_cache()

        if nan_count > 0:
            warnings.warn(
                f"Skipped {nan_count}/{self.n_steps + 1} gradient steps due to NaN values. "
                f"Results may be less accurate. Consider using float32 or a different baseline."
            )

        # Compute final attributions (IG formula)
        valid_steps = (self.n_steps + 1) - nan_count
        if valid_steps == 0:
            warnings.warn("All gradient steps produced NaN. Returning zero attributions.")
            token_attributions_np = np.zeros(input_ids.shape[1])
        else:
            avg_gradients = accumulated_gradients / valid_steps
            integrated_grads = (input_embeddings_cpu - baseline_embeddings_cpu) * avg_gradients
            token_attributions = integrated_grads.sum(dim=-1).squeeze(0)
            token_attributions_np = token_attributions.numpy()

        # Check for NaN in final result
        if np.isnan(token_attributions_np).any():
            nan_attr_count = int(np.isnan(token_attributions_np).sum())
            warnings.warn(f"Final attributions contain {nan_attr_count} NaN values!")

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Get model prediction (move input embeddings back to device for this)
        input_embeddings_device = input_embeddings_cpu.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs_embeds=input_embeddings_device)
            logits = outputs.logits[0, -1, :]

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
            'attributions': token_attributions_np,
            'prediction': prediction,
            'target_probability': target_prob,
            'target_class': target_class,
            'nan_count': nan_count,
        }

        # Convergence diagnostic (optional)
        if return_convergence_delta:
            baseline_embeddings_device = baseline_embeddings_cpu.to(self.device)
            with torch.no_grad():
                f_input = self._get_target_logit(input_embeddings_device, target_token_id).item()
                f_baseline = self._get_target_logit(baseline_embeddings_device, target_token_id).item()
                expected_sum = f_input - f_baseline
                actual_sum = float(token_attributions_np.sum())
                delta = abs(expected_sum - actual_sum)
            del baseline_embeddings_device

            result['convergence_delta'] = delta
            result['expected_sum'] = expected_sum
            result['actual_sum'] = actual_sum

        # Final cleanup
        del input_embeddings_device, accumulated_gradients, input_embeddings_cpu, baseline_embeddings_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            intensity = min(abs(score), 1.0)
            return f"\033[48;2;{int(255*(1-intensity))};{int(255*(1-intensity))};255m"
        else:
            intensity = min(score, 1.0)
            return f"\033[48;2;255;{int(255*(1-intensity))};{int(255*(1-intensity))}m"

    reset = "\033[0m"

    print("\n  ", end="")
    for token, score in zip(tokens, attributions):
        color = get_color(score)
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
