"""
Reasoning UI registry — plugin system for reasoning-elicitation methods
(Chain-of-Thought, Tree-of-Thought).

Parallel to app/registry.py (which handles attribution-based XAI methods).
Reasoning methods are NOT XAI methods: they produce reasoning text + a
final answer, not per-token attribution scores. They live in their own
registry so the UI can render them with a different result schema.

To add a new reasoning method:
1. Create a new file in app/reasoning/
2. Subclass ReasoningUI
3. Call register(YourReasoner()) in that file
4. Import that file in app/reasoning/__init__.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


_REGISTRY: Dict[str, "ReasoningUI"] = {}


class ReasoningUI(ABC):
    """
    Base class for reasoning-method UI components.

    Each reasoner must define:
        name            — unique identifier (e.g., "cot")
        display_name    — shown in UI tabs (e.g., "Chain-of-Thought")
        description     — one-line summary for tooltips
        supported_tasks — set of task types this method supports

    Result dicts produced by run() are method-specific and consumed only by
    the same reasoner's render_results(). Common keys include 'answer',
    'reasoning' / 'best_thought', 'confidence', 'option_probs'.
    """

    name: str = ""
    display_name: str = ""
    description: str = ""
    supported_tasks: set = {"yn", "mcq", "free"}

    @abstractmethod
    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        """Render Streamlit config widgets. Return a dict of parameter values."""
        pass

    @abstractmethod
    def run(
        self,
        wrapper,
        prompt: str,
        task_type: str,
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the reasoner and return a result dict."""
        pass

    @abstractmethod
    def render_results(self, result: Dict[str, Any], ground_truth: Optional[str] = None):
        """Render results using Streamlit components."""
        pass

    def is_available(self, task_type: str) -> bool:
        return task_type in self.supported_tasks


def register(reasoner: ReasoningUI):
    _REGISTRY[reasoner.name] = reasoner


def get_all() -> Dict[str, "ReasoningUI"]:
    return _REGISTRY


def get(name: str) -> Optional["ReasoningUI"]:
    return _REGISTRY.get(name)
