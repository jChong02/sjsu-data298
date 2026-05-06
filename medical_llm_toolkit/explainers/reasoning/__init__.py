"""
Reasoning-elicitation methods for medical LLMs.

These are not XAI / attribution methods. They prompt the model to reason
explicitly before answering, producing reasoning text + a final answer
rather than per-token attribution scores.

Available extractors:
    - ChainOfThoughtExtractor : single linear reasoning chain
    - TreeOfThoughtExtractor  : N candidate reasoning branches, scored and selected
"""

from medical_llm_toolkit.explainers.reasoning.chain_of_thought import (
    ChainOfThoughtExtractor,
    extract_chain_of_thought,
)
from medical_llm_toolkit.explainers.reasoning.tree_of_thought import (
    TreeOfThoughtExtractor,
    extract_tree_of_thought,
)

__all__ = [
    "ChainOfThoughtExtractor",
    "extract_chain_of_thought",
    "TreeOfThoughtExtractor",
    "extract_tree_of_thought",
]
