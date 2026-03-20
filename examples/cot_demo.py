"""
cot_demo.py

Demo script for ChainOfThoughtExtractor.

Tests CoT extraction across all three task types (mcq, yn, free) using
Apollo-2B by default. Swap in any model from MedicalLLMWrapper.

Run from the repo root:
    python examples/cot_demo.py

Optional: change MODEL_NAME below or pass --model on the command line.
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_llm_wrapper import load_medical_llm
from chain_of_thought_extractor import ChainOfThoughtExtractor, extract_chain_of_thought


# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------

MCQ_PROMPT = (
    "A 45-year-old male presents with sudden onset crushing chest pain radiating "
    "to the left arm and jaw, diaphoresis, and nausea. ECG shows ST elevation in "
    "leads II, III, and aVF. What is the most likely diagnosis?\n"
    "A) GERD\n"
    "B) Acute inferior myocardial infarction\n"
    "C) Costochondritis\n"
    "D) Pulmonary embolism"
)

YN_PROMPT = (
    "A patient with type 2 diabetes presents with a fasting blood glucose of "
    "280 mg/dL and HbA1c of 10.2%. Should insulin therapy be initiated?\n"
    "A) Yes\n"
    "B) No"
)

FREE_PROMPT = (
    "Explain the mechanism by which ACE inhibitors reduce blood pressure in "
    "patients with hypertension."
)


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_mcq(wrapper, strategy: str):
    print("\n" + "=" * 70)
    print(f"TEST 1 — MCQ  |  strategy='{strategy}'")
    print("=" * 70)
    print(f"Question:\n  {MCQ_PROMPT[:120]}...\n")

    wrapper.set_task("mcq")
    cot = ChainOfThoughtExtractor(wrapper, strategy=strategy)
    result = cot.extract(MCQ_PROMPT)

    print(cot.format_result(result))

    print("\nRaw result fields:")
    print(f"  answer        : {result['answer']}")
    print(f"  confidence    : {result['confidence']:.1%}" if result["confidence"] else "  confidence    : N/A")
    print(f"  num steps     : {len(result['steps'])}")
    print(f"  option_probs  : { {k: f'{v:.1%}' for k, v in result['option_probs'].items()} }")
    return result


def test_yn(wrapper, strategy: str):
    print("\n" + "=" * 70)
    print(f"TEST 2 — Y/N  |  strategy='{strategy}'")
    print("=" * 70)
    print(f"Question:\n  {YN_PROMPT[:120]}...\n")

    wrapper.set_task("yn")
    cot = ChainOfThoughtExtractor(wrapper, strategy=strategy)
    result = cot.extract(YN_PROMPT)

    print(cot.format_result(result))

    print("\nRaw result fields:")
    print(f"  answer        : {result['answer']}")
    print(f"  confidence    : {result['confidence']:.1%}" if result["confidence"] else "  confidence    : N/A")
    print(f"  option_probs  : { {k: f'{v:.1%}' for k, v in result['option_probs'].items()} }")
    return result


def test_free(wrapper, strategy: str):
    print("\n" + "=" * 70)
    print(f"TEST 3 — Free-text  |  strategy='{strategy}'")
    print("=" * 70)
    print(f"Question:\n  {FREE_PROMPT}\n")

    wrapper.set_task("free")
    cot = ChainOfThoughtExtractor(wrapper, strategy=strategy, max_reasoning_tokens=250)
    result = cot.extract(FREE_PROMPT)

    print(cot.format_result(result))

    print("\nRaw result fields:")
    print(f"  num steps     : {len(result['steps'])}")
    print(f"  confidence    : N/A (free task)")
    return result


def test_batch(wrapper, strategy: str):
    """Run batch extraction to verify the batch API works."""
    print("\n" + "=" * 70)
    print(f"TEST 4 — Batch extraction  |  strategy='{strategy}'")
    print("=" * 70)

    prompts = [
        MCQ_PROMPT,
        "Which organ produces insulin?\nA) Liver\nB) Pancreas\nC) Kidney\nD) Spleen",
    ]
    task_types = ["mcq", "mcq"]

    wrapper.set_task("mcq")
    cot = ChainOfThoughtExtractor(wrapper, strategy=strategy)
    results = cot.batch_extract(prompts, task_types=task_types)

    for i, r in enumerate(results, 1):
        print(f"\n  Batch item {i}:")
        print(f"    answer     : {r['answer']}")
        print(f"    confidence : {r['confidence']:.1%}" if r["confidence"] else "    confidence : N/A")

    return results


def test_convenience_fn(wrapper):
    """Test the extract_chain_of_thought one-liner helper."""
    print("\n" + "=" * 70)
    print("TEST 5 — extract_chain_of_thought() convenience function")
    print("=" * 70)

    wrapper.set_task("mcq")
    result = extract_chain_of_thought(
        wrapper,
        "Which vitamin deficiency causes scurvy?\n"
        "A) Vitamin A\nB) Vitamin B12\nC) Vitamin C\nD) Vitamin D",
        strategy="zero_shot",
        print_result=True,
    )
    return result


# ---------------------------------------------------------------------------
# Compare wrapper rationale vs CoT
# ---------------------------------------------------------------------------

def compare_with_wrapper(wrapper):
    """
    Side-by-side comparison: MedicalLLMWrapper rationale (post-hoc)
    vs ChainOfThoughtExtractor (reason-first).
    """
    print("\n" + "=" * 70)
    print("COMPARISON — Wrapper rationale  vs  CoT reasoning")
    print("=" * 70)

    question = (
        "A child presents with a salmon-colored rash, spiking fever, and arthritis. "
        "Most likely diagnosis?\n"
        "A) Kawasaki disease\n"
        "B) Systemic juvenile idiopathic arthritis\n"
        "C) Acute rheumatic fever\n"
        "D) Septic arthritis"
    )

    wrapper.set_task("mcq")

    # --- Wrapper: answer then rationale ---
    wrapper.set_mode("answer_rationale")
    wrapper_out = wrapper.generate(question)
    print("\n[Wrapper — post-hoc rationale]")
    print(wrapper_out)

    # --- CoT: reason first, then answer ---
    print("\n[CoT extractor — reason-first]")
    cot = ChainOfThoughtExtractor(wrapper, strategy="structured")
    result = cot.extract(question)
    print(cot.format_result(result))

    # Restore default mode
    wrapper.set_mode("answer_rationale")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CoT extractor demo")
    parser.add_argument(
        "--model",
        default="FreedomIntelligence/Apollo-2B",
        help="HuggingFace model ID to load",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--strategy",
        default="structured",
        choices=["structured", "zero_shot"],
        help="CoT prompting strategy",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Model dtype (use float32 for MedGemma)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Chain-of-Thought Extractor — Demo")
    print("=" * 70)
    print(f"  Model    : {args.model}")
    print(f"  Device   : {args.device}")
    print(f"  Strategy : {args.strategy}")
    print(f"  Dtype    : {args.dtype}")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print("\n[Loading model...]")
    wrapper = load_medical_llm(args.model, device=args.device, torch_dtype=dtype)

    # Run tests
    test_mcq(wrapper, args.strategy)
    test_yn(wrapper, args.strategy)
    test_free(wrapper, args.strategy)
    test_batch(wrapper, args.strategy)
    test_convenience_fn(wrapper)
    compare_with_wrapper(wrapper)

    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
