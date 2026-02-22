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

    # Main entry point — auto-selects predicted class as target
    result = lime.analyze(prompt)

    # Or specify the class to explain
    result = lime.analyze(prompt, target_class="B")

    # Useful computed fields
    print(result['prediction'])           # model's answer
    print(result['all_option_probs'])     # {'A': 0.1, 'B': 0.7, 'C': 0.1, 'D': 0.1}
    print(result['r_squared'])            # local model fit quality (0–1)
    print(result['top_words'][:5])        # [(word, score), ...] by |attribution|
    print(result['word_attributions'])    # numpy array — store, plot, compare
"""

from typing import Dict, List, Optional, Tuple

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

    Main entry point:
        result = lime.analyze(prompt)                  # auto-detects target
        result = lime.analyze(prompt, target_class='B')
        results = lime.analyze_batch(prompts)          # multiple prompts

    Result dict keys:
        words               List[str]          — word segments
        word_attributions   np.ndarray         — per-word signed scores
        tokens              List[str]          — tokenizer subword tokens
        token_attributions  np.ndarray         — per-token signed scores
        attributions        np.ndarray         — alias for token_attributions
        prediction          str                — model's top predicted class
        target_class        str                — class being explained
        target_probability  float              — P(target_class) on original
        all_option_probs    Dict[str, float]   — P(A/B/C/D) on original
        r_squared           float              — local linear model fit (0–1)
        intercept           float              — linear model intercept
        top_words           List[Tuple]        — (word, score) by |score| desc
        top_positive_words  List[Tuple]        — (word, score) supporting target
        top_negative_words  List[Tuple]        — (word, score) against target
        metadata            Dict               — run configuration
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
                          cosine distance in [0, 1]).
                          Smaller → tighter local fit; larger → more global.
            mask_token: String inserted when a word is masked. Empty string ('')
                        drops the word entirely; '[MASK]' replaces it.
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

    # Internal helpers

    def _get_option_probs(self, text: str) -> Dict[str, float]:
        """
        Run a single forward pass and return softmax probabilities over the
        allowed answer options (A/B for yn, A/B/C/D for mcq).

        The probability for each option is computed by extracting the logit for
        that token at the final position and applying softmax over the option
        subset — consistent with MedicalLLMWrapper.generate_with_confidence().

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
    ) -> Tuple[List[str], np.ndarray]:
        """
        Map word-level attributions onto tokenizer subword tokens.

        Each token inherits the attribution of the word it belongs to, using
        isolated per-word tokenization to count tokens per word.

        Args:
            prompt: Original prompt string
            words: List of words (same as prompt.split())
            word_attributions: Per-word attribution scores

        Returns:
            (tokens, token_attributions) aligned by token index
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

    def _validate(self, target_class: str) -> None:
        """
        Ensure the task type and target class are compatible.

        If wrapper.task_type is 'free' (the default after loading), it is
        automatically set to 'mcq' with a warning rather than raising an error.
        An invalid target_class value still raises ValueError.
        """
        if self.wrapper.task_type not in ['yn', 'mcq']:
            print(
                f"[MedicalLIME] Warning: wrapper task type is "
                f"'{self.wrapper.task_type}', auto-setting to 'mcq'."
            )
            self.wrapper.set_task('mcq')
        if target_class not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"target_class must be A/B/C/D, got '{target_class}'")
        if self.wrapper.task_type == 'yn' and target_class not in ['A', 'B']:
            raise ValueError(
                f"For Y/N tasks, target_class must be 'A' or 'B', got '{target_class}'"
            )


    # Public API

    def attribute(self, prompt: str, target_class: str) -> Dict:
        """
        Compute word-level LIME attributions for a single prompt.

        Splits the prompt into words, generates n_samples random binary
        perturbations (words kept or removed), evaluates P(target_class) for
        each via a direct forward pass, then fits a weighted ridge regression
        whose coefficients are the attribution scores.

        Args:
            prompt: Input prompt including answer options and 'Answer:' cue.
            target_class: Class to explain — 'A', 'B', 'C', or 'D'.

        Returns:
            Rich result dictionary (see class docstring for all keys).
        """
        self._validate(target_class)

        words = prompt.split()
        n_words = len(words)
        if n_words == 0:
            raise ValueError("Prompt is empty after splitting into words.")

        labels = ['A', 'B'] if self.wrapper.task_type == 'yn' else ['A', 'B', 'C', 'D']

        # --- Baseline: query original (unperturbed) prompt ---
        original_probs = self._get_option_probs(prompt)
        prediction = max(labels, key=lambda l: original_probs.get(l, 0.0))
        target_prob = original_probs[target_class]

        # --- Generate binary perturbation masks ---
        # Row 0 is always the original (all ones); remainder are random.
        rng = np.random.RandomState(42)
        masks = rng.randint(0, 2, size=(self.n_samples, n_words)).astype(float)
        masks[0] = 1.0

        # --- Evaluate model on each perturbed input ---
        perturbed_probs = np.zeros(self.n_samples)
        perturbed_probs[0] = target_prob  # original already known

        iterator = tqdm(
            range(1, self.n_samples),
            desc="LIME sampling",
            disable=not self.verbose
        )
        for i in iterator:
            perturbed_text = self._perturb_text(words, masks[i])
            if not perturbed_text.strip():
                perturbed_probs[i] = 0.0
                continue
            try:
                probs = self._get_option_probs(perturbed_text)
                perturbed_probs[i] = probs.get(target_class, 0.0)
            except Exception:
                perturbed_probs[i] = 0.0

        # --- Cosine-distance kernel weights ---
        # cos_sim(mask, all-ones) = (sum of mask) / (sqrt(n_kept) * sqrt(n_words))
        original_mask = np.ones(n_words)
        norms = np.linalg.norm(masks, axis=1)
        original_norm = float(np.linalg.norm(original_mask))
        cos_sim = np.clip(
            np.dot(masks, original_mask) / (norms * original_norm + 1e-10),
            -1.0, 1.0
        )
        distances = 1.0 - cos_sim                                   # in [0, 1]
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # --- Fit weighted ridge regression ---
        regressor = Ridge(alpha=1.0, fit_intercept=True)
        regressor.fit(masks, perturbed_probs, sample_weight=weights)

        word_attributions: np.ndarray = regressor.coef_             # (n_words,)
        intercept = float(regressor.intercept_)
        r_squared = float(regressor.score(masks, perturbed_probs, sample_weight=weights))

        # --- Map to tokenizer tokens ---
        tokens, token_attributions = self._map_words_to_tokens(
            prompt, words, word_attributions
        )

        # --- Derived ranked word lists ---
        # Sorted by absolute attribution magnitude (most important first)
        order_by_magnitude = np.argsort(np.abs(word_attributions))[::-1]
        top_words: List[Tuple[str, float]] = [
            (words[i], float(word_attributions[i])) for i in order_by_magnitude
        ]

        # Words that increase P(target_class), strongest first
        top_positive: List[Tuple[str, float]] = sorted(
            [(words[i], float(word_attributions[i]))
             for i in range(n_words) if word_attributions[i] > 0],
            key=lambda x: x[1], reverse=True
        )

        # Words that decrease P(target_class), most negative first
        top_negative: List[Tuple[str, float]] = sorted(
            [(words[i], float(word_attributions[i]))
             for i in range(n_words) if word_attributions[i] < 0],
            key=lambda x: x[1]
        )

        return {
            # Core word-level output
            'words': words,
            'word_attributions': word_attributions,
            # Token-level output (compatible with visualize_attributions)
            'tokens': tokens,
            'token_attributions': token_attributions,
            'attributions': token_attributions,
            # Prediction info
            'prediction': prediction,
            'target_class': target_class,
            'target_probability': target_prob,
            'all_option_probs': original_probs,          # P(A/B/C/D) on original
            # Linear model diagnostics
            'r_squared': r_squared,
            'intercept': intercept,
            # Pre-ranked word lists for downstream use
            'top_words': top_words,
            'top_positive_words': top_positive,
            'top_negative_words': top_negative,
            # Run configuration
            'metadata': {
                'n_samples': self.n_samples,
                'kernel_width': self.kernel_width,
                'mask_token': self.mask_token,
                'model_name': self.wrapper.model_name,
                'task_type': self.wrapper.task_type,
            },
        }

    def analyze(
        self,
        prompt: str,
        target_class: Optional[str] = None,
        visualize: bool = False
    ) -> Dict:
        """
        Main entry point for LIME explanations.

        Automatically selects the model's predicted class as the target if
        target_class is not specified. Returns the same rich result dict as
        attribute() — all values are plain Python / NumPy types so they can
        be stored, serialised, or fed into further computation.

        Args:
            prompt: Input prompt including answer options and 'Answer:' cue.
            target_class: Class to explain ('A'/'B'/'C'/'D'). If None, the
                          model's top predicted class is used automatically.
            visualize: Print color-coded word visualization after computing.

        Returns:
            Rich result dictionary (see class docstring for all keys).

        Example::

            lime = MedicalLIME(wrapper)
            result = lime.analyze(prompt)

            # Store and compute
            scores = result['word_attributions']           # np.ndarray
            print(result['top_words'][:5])                 # top-5 by |score|
            print(f"R² = {result['r_squared']:.3f}")
            print(f"All probs: {result['all_option_probs']}")
        """
        # Auto-set task type if still on the default 'free' — mirrors _validate()
        if self.wrapper.task_type not in ['yn', 'mcq']:
            print(
                f"[MedicalLIME] Warning: wrapper task type is "
                f"'{self.wrapper.task_type}', auto-setting to 'mcq'."
            )
            self.wrapper.set_task('mcq')

        # Auto-detect target class from model prediction
        if target_class is None:
            original_probs = self._get_option_probs(prompt)
            labels = ['A', 'B'] if self.wrapper.task_type == 'yn' else ['A', 'B', 'C', 'D']
            target_class = max(labels, key=lambda l: original_probs.get(l, 0.0))
            if self.verbose:
                print(f"[MedicalLIME] Auto-selected target class: {target_class}")

        result = self.attribute(prompt, target_class)

        if visualize:
            visualize_lime_attributions(
                result['words'],
                result['word_attributions'],
                result['prediction'],
                result['target_class'],
                title=(
                    f"LIME Explanation  |  R²={result['r_squared']:.3f}"
                    f"  |  P({result['target_class']})={result['target_probability']:.4f}"
                )
            )

        return result

    def analyze_batch(
        self,
        prompts: List[str],
        target_classes: Optional[List[Optional[str]]] = None
    ) -> List[Dict]:
        """
        Run analyze() over a list of prompts.

        Args:
            prompts: List of input prompt strings.
            target_classes: Optional list of target classes (one per prompt).
                            Use None entries or omit the list to auto-detect.

        Returns:
            List of result dictionaries, one per prompt.

        Example::

            results = lime.analyze_batch(prompts)
            all_scores = [r['word_attributions'] for r in results]
            predictions = [r['prediction'] for r in results]
        """
        resolved: List[Optional[str]] = (
            target_classes if target_classes is not None
            else [None] * len(prompts)
        )

        if len(prompts) != len(resolved):
            raise ValueError("prompts and target_classes must have the same length")

        return [
            self.analyze(prompt, tc)
            for prompt, tc in zip(prompts, resolved)
        ]

    def attribute_batch(
        self,
        prompts: List[str],
        target_classes: List[str]
    ) -> List[Dict]:
        """
        Compute attributions for multiple prompts with explicit target classes.

        Args:
            prompts: List of input prompt strings.
            target_classes: List of target classes (one per prompt, must be explicit).

        Returns:
            List of result dictionaries.
        """
        if len(prompts) != len(target_classes):
            raise ValueError("prompts and target_classes must have the same length")

        return [
            self.attribute(prompt, tc)
            for prompt, tc in zip(prompts, target_classes)
        ]


# Visualization

def visualize_lime_attributions(
    words: List[str],
    word_attributions: np.ndarray,
    prediction: str,
    target_class: str,
    title: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Print a color-coded word attribution visualization.

    Positive attributions (red) increase P(target_class);
    negative attributions (blue) decrease it.

    Args:
        words: List of word strings
        word_attributions: Attribution score per word
        prediction: Model's predicted answer
        target_class: Target class being explained
        title: Optional title line
        normalize: Normalize attributions to [-1, 1] before coloring
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

    def get_color(score: float) -> str:
        if score < 0:
            intensity = min(abs(score), 1.0)
            return f"\033[48;2;{int(255*(1-intensity))};{int(255*(1-intensity))};255m"
        intensity = min(score, 1.0)
        return f"\033[48;2;255;{int(255*(1-intensity))};{int(255*(1-intensity))}m"

    reset = "\033[0m"
    print("\n  ", end="")
    for word, score in zip(words, attrs):
        print(f"{get_color(score)} {word} {reset}", end="")

    print("\n\n  Legend: ", end="")
    print(f"\033[48;2;255;150;150mPositive (supports answer)\033[0m  ", end="")
    print(f"\033[48;2;150;150;255mNegative (against answer)\033[0m")
    print("=" * 80)


# Utility: convert result to DataFrame

def to_dataframe(result: Dict):
    """
    Convert a LIME result dictionary to a pandas DataFrame for analysis.

    Each row is one word with its attribution score and absolute magnitude.
    The DataFrame is sorted by absolute attribution (most important first).

    Args:
        result: Dictionary returned by MedicalLIME.analyze() or .attribute()

    Returns:
        pandas.DataFrame with columns: word, attribution, abs_attribution

    Example::

        result = lime.analyze(prompt)
        df = to_dataframe(result)
        print(df.head(10))          # top-10 most influential words
        df.to_csv("lime_scores.csv", index=False)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for to_dataframe(). Install it with: pip install pandas") from e

    df = pd.DataFrame({
        'word': result['words'],
        'attribution': result['word_attributions'].tolist(),
        'abs_attribution': np.abs(result['word_attributions']).tolist(),
    })
    return df.sort_values('abs_attribution', ascending=False).reset_index(drop=True)


# Utility: JSON serialization

def to_json_serializable(result: Dict) -> Dict:
    """
    Convert a LIME result dictionary to a fully JSON-serializable format.

    NumPy arrays (word_attributions, token_attributions, attributions) are
    converted to plain Python lists so the result can be passed directly to
    json.dumps() or stored with json.dump().

    Args:
        result: Dictionary returned by MedicalLIME.analyze() or .attribute()

    Returns:
        New dictionary with the same keys; NumPy arrays replaced by lists.

    Example::

        import json
        result = lime.analyze(prompt)
        serializable = to_json_serializable(result)
        with open("result.json", "w") as f:
            json.dump(serializable, f, indent=2)

        # Or inline
        json_str = json.dumps(to_json_serializable(result))
    """
    out = dict(result)
    for key in ('word_attributions', 'token_attributions', 'attributions'):
        if key in out and isinstance(out[key], np.ndarray):
            out[key] = out[key].tolist()
    return out


# Convenience one-liner

def explain_with_lime(
    wrapper,
    prompt: str,
    target_class: Optional[str] = None,
    n_samples: int = 500,
    visualize: bool = True
) -> Dict:
    """
    One-liner to explain a medical LLM prediction using LIME.

    Args:
        wrapper: MedicalLLMWrapper instance
        prompt: Input prompt
        target_class: Answer to explain ('A'/'B'/'C'/'D'), or None to auto-detect
        n_samples: Number of LIME perturbation samples
        visualize: Print color-coded word visualization

    Returns:
        Rich result dictionary (see MedicalLIME class docstring)
    """
    if wrapper.task_type not in ['yn', 'mcq']:
        print(f"[Warning] Wrapper task type is '{wrapper.task_type}', auto-setting to 'mcq'")
        wrapper.set_task('mcq')

    lime_explainer = MedicalLIME(wrapper, n_samples=n_samples)

    try:
        result = lime_explainer.analyze(prompt, target_class=target_class, visualize=visualize)
    except Exception as e:
        print(f"\n[ERROR] LIME attribution failed: {e}")
        print(f"  Wrapper task type : {wrapper.task_type}")
        print(f"  Wrapper mode      : {wrapper.mode}")
        print(f"  Target class      : {target_class}")
        print(f"  Model dtype       : {wrapper.model_dtype}")
        raise

    return result
