"""
Medical LIME - Works with MedicalLLMWrapper

LIME (Local Interpretable Model-agnostic Explanations) for medical LLM
interpretability. Perturbs the input prompt at the word level, queries the
model for each perturbation, and fits a local linear model to produce
word-level attribution scores.

Compatible with any model loaded through MedicalLLMWrapper.

Usage:
    from medical_llm_wrapper import load_medical_llm
    from medical_lime_adapter import MedicalLIME

    wrapper = load_medical_llm("google/medgemma-4b-it")
    wrapper.set_task("mcq")
    lime = MedicalLIME(wrapper)
    result = lime.attribute("Patient has chest pain and fever. A) MI B) PE", target_class="A")
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from tqdm import tqdm


class MedicalLIME:
    """
    LIME explainer for medical LLMs using MedicalLLMWrapper.

    Generates word-level attribution scores by randomly masking words in the
    prompt, querying the model on each perturbed input, then fitting a weighted
    ridge regression to find which words most influence the target class
    probability. Works with Y/N and MCQ tasks.
    """

    def __init__(
        self,
        wrapper,
        n_samples: int = 500,
        kernel_width: float = 0.75,
        mask_token: str = '',
        verbose: bool = True
    ):
        """
        Initialize LIME explainer.

        Args:
            wrapper: MedicalLLMWrapper instance
            n_samples: Number of perturbed samples to evaluate (more = more accurate)
            kernel_width: Kernel bandwidth controlling locality (relative to
                          normalised cosine distance in [0, 1]).
                          Smaller → tighter local fit; larger → more global.
            mask_token: String inserted when a word is masked. Empty string ('')
                        drops the word entirely; use '[MASK]' to replace instead.
            verbose: Show tqdm progress bar during sampling
        """
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = wrapper.device
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.mask_token = mask_token
        self.verbose = verbose

        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_option_probs(self, text: str) -> Dict[str, float]:
        """
        Run a single forward pass and return softmax probabilities over the
        allowed answer options (A/B or A/B/C/D).

        Args:
            text: Prompt text to evaluate

        Returns:
            Dict mapping option label to probability, e.g. {'A': 0.7, 'B': 0.3}
        """
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # [vocab_size]

        if self.wrapper.task_type == 'yn':
            option_ids = self.wrapper.AB_IDS
            labels = ['A', 'B']
        else:
            option_ids = self.wrapper.ABCD_IDS
            labels = ['A', 'B', 'C', 'D']

        option_logits = torch.tensor(
            [logits[tid].item() for tid in option_ids],
            dtype=torch.float32
        )
        probs = F.softmax(option_logits, dim=0)

        return {label: probs[i].item() for i, label in enumerate(labels)}

    def _perturb_text(self, words: List[str], mask: np.ndarray) -> str:
        """
        Reconstruct text from a binary word-presence mask.

        Args:
            words: Original word list
            mask: Float array with 1.0 = keep word, 0.0 = mask/remove

        Returns:
            Perturbed text string
        """
        parts = []
        for word, keep in zip(words, mask):
            if keep:
                parts.append(word)
            elif self.mask_token:
                parts.append(self.mask_token)
        return ' '.join(parts)

    def _map_words_to_tokens(
        self,
        prompt: str,
        words: List[str],
        word_attributions: np.ndarray
    ):
        """
        Map word-level attributions onto tokenizer tokens.

        Each tokenizer token inherits the attribution of the word it belongs to.
        Uses per-word token counts from isolated tokenization; this is a
        best-effort mapping and may drift slightly for unusual subword splits.

        Args:
            prompt: Original prompt string
            words: List of words (same as prompt.split())
            word_attributions: Per-word attribution scores

        Returns:
            (tokens, token_attributions): lists/arrays aligned by token index
        """
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        token_attributions = np.zeros(len(tokens))

        token_idx = 0
        for word_idx, word in enumerate(words):
            word_toks = self.tokenizer.encode(word, add_special_tokens=False)
            for _ in word_toks:
                if token_idx < len(tokens):
                    token_attributions[token_idx] = word_attributions[word_idx]
                    token_idx += 1

        return tokens, token_attributions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        prompt: str,
        target_class: str
    ) -> Dict:
        """
        Compute word-level attributions using LIME.

        Splits the prompt into words, generates n_samples random perturbations,
        evaluates P(target_class) for each, then fits a weighted ridge
        regression to produce per-word importance scores.

        Args:
            prompt: Input prompt (e.g., "Patient has chest pain. Diagnosis?
                    A) MI  B) PE  C) TB  D) Pneumonia  Answer:")
            target_class: Answer class to explain ('A', 'B', 'C', or 'D')

        Returns:
            Dictionary with:
                - 'words': List[str] - interpretable word segments
                - 'word_attributions': np.ndarray - per-word attribution scores
                - 'tokens': List[str] - tokenizer tokens
                - 'token_attributions': np.ndarray - per-token attribution scores
                - 'attributions': np.ndarray - alias for token_attributions
                  (compatible with visualize_attributions)
                - 'prediction': str - model's predicted answer ('A'/'B'/'C'/'D')
                - 'target_probability': float - P(target_class) on original prompt
                - 'target_class': str - the class being explained
                - 'intercept': float - linear model intercept
        """
        # --- Validate ---
        if target_class not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"target_class must be A/B/C/D, got {target_class}")

        if self.wrapper.task_type not in ['yn', 'mcq']:
            raise ValueError(
                f"LIME only supports 'yn' and 'mcq' tasks. "
                f"Current task type is '{self.wrapper.task_type}'. "
                f"Please call wrapper.set_task('mcq') or wrapper.set_task('yn') first."
            )

        if self.wrapper.task_type == 'yn' and target_class not in ['A', 'B']:
            raise ValueError(
                f"For Y/N tasks, target_class must be A or B, got {target_class}"
            )

        # --- Split into interpretable word-level components ---
        words = prompt.split()
        n_words = len(words)

        if n_words == 0:
            raise ValueError("Prompt is empty after splitting into words.")

        # --- Baseline: get prediction on the original (unperturbed) prompt ---
        original_probs = self._get_option_probs(prompt)

        labels = ['A', 'B'] if self.wrapper.task_type == 'yn' else ['A', 'B', 'C', 'D']
        prediction = max(labels, key=lambda l: original_probs.get(l, 0.0))
        target_prob = original_probs[target_class]

        # --- Generate binary perturbation masks ---
        # Row 0 is the all-ones mask (original prompt); remaining rows are random.
        rng = np.random.RandomState(42)
        masks = rng.randint(0, 2, size=(self.n_samples, n_words)).astype(float)
        masks[0] = 1.0  # Always evaluate the original

        # --- Evaluate model on perturbed inputs ---
        perturbed_probs = np.zeros(self.n_samples)

        iterator = tqdm(range(self.n_samples), desc="LIME sampling", disable=not self.verbose)
        for i in iterator:
            perturbed_text = self._perturb_text(words, masks[i])
            if not perturbed_text.strip():
                # All words removed — use 0.0 as fallback
                perturbed_probs[i] = 0.0
                continue
            try:
                probs = self._get_option_probs(perturbed_text)
                perturbed_probs[i] = probs.get(target_class, 0.0)
            except Exception:
                perturbed_probs[i] = 0.0

        # --- Compute cosine distance from the original (all-ones mask) ---
        # Normalise by sqrt(n_words) so distances stay in a comparable range
        # regardless of prompt length, then apply an exponential kernel.
        original_mask = np.ones(n_words)
        norms = np.linalg.norm(masks, axis=1)
        original_norm = np.linalg.norm(original_mask)
        # Cosine similarity; clip to avoid numerical issues
        cos_sim = np.clip(
            np.dot(masks, original_mask) / (norms * original_norm + 1e-10),
            -1.0, 1.0
        )
        distances = 1.0 - cos_sim  # cosine distance in [0, 1]
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # --- Fit weighted ridge regression ---
        regressor = Ridge(alpha=1.0, fit_intercept=True)
        regressor.fit(masks, perturbed_probs, sample_weight=weights)

        word_attributions = regressor.coef_   # shape: (n_words,)
        intercept = float(regressor.intercept_)

        # --- Map word attributions to tokenizer token attributions ---
        tokens, token_attributions = self._map_words_to_tokens(
            prompt, words, word_attributions
        )

        return {
            'words': words,
            'word_attributions': word_attributions,
            'tokens': tokens,
            'token_attributions': token_attributions,
            'attributions': token_attributions,   # alias for visualize_attributions
            'prediction': prediction,
            'target_probability': target_prob,
            'target_class': target_class,
            'intercept': intercept,
        }

    def attribute_batch(
        self,
        prompts: List[str],
        target_classes: List[str]
    ) -> List[Dict]:
        """
        Compute LIME attributions for multiple prompts.

        Args:
            prompts: List of input prompt strings
            target_classes: List of target classes (one per prompt)

        Returns:
            List of attribution dictionaries (same format as attribute())
        """
        if len(prompts) != len(target_classes):
            raise ValueError("prompts and target_classes must have the same length")

        return [
            self.attribute(prompt, target)
            for prompt, target in zip(prompts, target_classes)
        ]


# ---------------------------------------------------------------------------
# Visualization helper (word-level)
# ---------------------------------------------------------------------------

def visualize_lime_attributions(
    words: List[str],
    word_attributions: np.ndarray,
    prediction: str,
    target_class: str,
    title: Optional[str] = None,
    normalize: bool = True
):
    """
    Print a color-coded word attribution visualization for LIME results.

    Positive attributions (red) increase P(target_class);
    negative attributions (blue) decrease it.

    Args:
        words: List of word strings
        word_attributions: Attribution score per word
        prediction: Model's predicted answer
        target_class: Target class being explained
        title: Optional title line
        normalize: Normalize attributions to [0, 1] range before coloring
    """
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
    print(f"  Prediction: {prediction} | Explaining: {target_class}")
    print("=" * 80)

    attrs = word_attributions.copy()
    if normalize:
        attr_min, attr_max = attrs.min(), attrs.max()
        if attr_max - attr_min > 0:
            attrs = (attrs - attr_min) / (attr_max - attr_min)

    def get_color(score):
        if score < 0:
            intensity = min(abs(score), 1.0)
            return f"\033[48;2;{int(255*(1-intensity))};{int(255*(1-intensity))};255m"
        else:
            intensity = min(score, 1.0)
            return f"\033[48;2;255;{int(255*(1-intensity))};{int(255*(1-intensity))}m"

    reset = "\033[0m"
    print("\n  ", end="")
    for word, score in zip(words, attrs):
        color = get_color(score)
        print(f"{color} {word} {reset}", end="")

    print("\n\n  Legend: ", end="")
    print(f"\033[48;2;255;150;150mPositive (supports answer)\033[0m  ", end="")
    print(f"\033[48;2;150;150;255mNegative (against answer)\033[0m")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def explain_with_lime(
    wrapper,
    prompt: str,
    target_class: str,
    n_samples: int = 500,
    visualize: bool = True
) -> Dict:
    """
    One-liner to explain a medical LLM prediction using LIME.

    Args:
        wrapper: MedicalLLMWrapper instance
        prompt: Input prompt
        target_class: Answer to explain ('A', 'B', 'C', or 'D')
        n_samples: Number of LIME perturbation samples
        visualize: Print color-coded word visualization

    Returns:
        Attribution dictionary (see MedicalLIME.attribute)
    """
    if wrapper.task_type not in ['yn', 'mcq']:
        print(f"[Warning] Wrapper task type is '{wrapper.task_type}', auto-setting to 'mcq'")
        wrapper.set_task('mcq')

    lime_explainer = MedicalLIME(wrapper, n_samples=n_samples)

    try:
        result = lime_explainer.attribute(prompt, target_class)
    except Exception as e:
        print(f"\n[ERROR] LIME attribution failed: {e}")
        print(f"\nDebugging info:")
        print(f"  - Wrapper task type: {wrapper.task_type}")
        print(f"  - Wrapper mode: {wrapper.mode}")
        print(f"  - Target class: {target_class}")
        print(f"  - Model dtype: {wrapper.model_dtype}")
        raise

    if visualize:
        visualize_lime_attributions(
            result['words'],
            result['word_attributions'],
            result['prediction'],
            result['target_class'],
            title=f"LIME Explanation (target_prob={result['target_probability']:.4f})"
        )

    return result
