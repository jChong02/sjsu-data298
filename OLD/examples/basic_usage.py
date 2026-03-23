"""
Basic usage example for Medical LLM Toolkit.

Demonstrates how to use the wrapper and Integrated Gradients
to explain medical LLM predictions.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_llm_toolkit import load_medical_llm, MedicalIntegratedGradients


def main():
    print("=" * 80)
    print("Medical LLM Toolkit - Basic Usage Example")
    print("=" * 80)
    
    # Load model
    print("\n[1] Loading Apollo-2B model...")
    wrapper = load_medical_llm(
        "FreedomIntelligence/Apollo-2B",
        device="cuda",
        torch_dtype=torch.float16
    )
    
    # Set task
    wrapper.set_task("mcq")
    wrapper.set_mode("answer_only")
    
    # Medical question
    prompt = """Which vitamin deficiency causes scurvy?
A) Vitamin A
B) Vitamin B12
C) Vitamin C
D) Vitamin D

Answer:"""
    
    # Get prediction
    print("\n[2] Getting model prediction...")
    answer = wrapper.generate(prompt)
    print(f"\nModel's answer: {answer}")
    print(f"Confidence: {wrapper.last_confidence:.4f}")
    
    # Explain with Integrated Gradients
    print("\n[3] Computing Integrated Gradients explanation...")
    ig = MedicalIntegratedGradients(wrapper, n_steps=30)
    result = ig.attribute(prompt, target_class="C", return_convergence_delta=True)
    
    print(f"\n[4] Results:")
    print(f"  Target probability: {result['target_probability']:.4f}")
    print(f"  Convergence delta: {result['convergence_delta']:.6f}")
    
    print(f"\n  Top 5 most important tokens:")
    token_scores = list(zip(result['tokens'], result['attributions']))
    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (token, score) in enumerate(token_scores[:5], 1):
        clean_token = token.replace('▁', ' ').replace('Ġ', ' ').strip()
        if clean_token:
            print(f"    {i}. '{clean_token}': {score:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
