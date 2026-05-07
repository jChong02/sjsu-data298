"""
Explainer registry - plugin system for XAI methods.

To add a new method:
1. Create a new file in app/explainers/
2. Subclass ExplainerUI
3. Call registry.register(YourExplainer()) in that file
4. Import that file in app/explainers/__init__.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

_REGISTRY: Dict[str, "ExplainerUI"] = {}


class ExplainerUI(ABC):
    """
    Base class for explainer UI components.

    Each explainer must define:
        name         - unique identifier (e.g., "lime")
        display_name - shown in UI tabs (e.g., "LIME")
        description  - one-line summary for tooltips
        supported_tasks - set of task types this method supports (e.g., {"yn", "mcq"})
    """

    name: str = ""
    display_name: str = ""
    description: str = ""
    supported_tasks: set = {"yn", "mcq"}

    @abstractmethod
    def render_config(self, key_prefix: str) -> Dict[str, Any]:
        """
        Render Streamlit config widgets for this method.
        Return a dict of parameter values.
        key_prefix is used to namespace widget keys (e.g., "lime_").
        """
        pass

    @abstractmethod
    def run(
        self,
        wrapper,
        prompt: str,
        target_class: Optional[str],
        ground_truth: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the explainer. Return a result dict.
        The result dict should contain at minimum:
            'tokens' or 'words' - list of text segments
            'attributions' - scores per segment (list or numpy array)
        Additional keys are method-specific.
        """
        pass

    @abstractmethod
    def render_results(self, result: Dict[str, Any]):
        """Render results using Streamlit components."""
        pass

    def is_available(self, task_type: str) -> bool:
        """Check if this method supports the current task type."""
        return task_type in self.supported_tasks


def register(explainer: ExplainerUI):
    """Register an explainer instance."""
    _REGISTRY[explainer.name] = explainer


def get_all() -> Dict[str, "ExplainerUI"]:
    """Return all registered explainers."""
    return _REGISTRY


def get(name: str) -> Optional["ExplainerUI"]:
    """Get a specific explainer by name."""
    return _REGISTRY.get(name)
