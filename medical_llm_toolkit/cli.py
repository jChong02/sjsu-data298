"""Console-script entry point for the LLM XAI Toolkit Streamlit app.

Provides a cross-platform ``medxai`` command (defined in pyproject.toml's
``[project.scripts]``) that launches ``app/main.py`` via Streamlit. After
``pip install -e .``, run::

    medxai

This is equivalent to the legacy Windows-only ``run.bat`` and ``streamlit
run app/main.py`` commands.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_app_main() -> Path | None:
    """Locate ``app/main.py`` relative to the installed package.

    For ``pip install -e .`` (editable install from a clone), ``__file__``
    points into the source tree, so walking up from the package directory
    reaches ``app/main.py`` at the repo root. For a non-editable wheel
    install, ``app/`` does not ship inside the package - see the README
    "Install from source" note.
    """
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent / "app" / "main.py",  # repo root (editable install)
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    """Launch the Streamlit app."""
    try:
        import streamlit.web.cli as stcli
    except ImportError as exc:
        sys.exit(
            f"Streamlit is not installed: {exc}\n"
            "Run `pip install -e .` from the repository root to install "
            "all dependencies."
        )

    app_main = _find_app_main()
    if app_main is None:
        sys.exit(
            "Could not locate `app/main.py`. The medxai command expects to "
            "be run after `pip install -e .` from a clone of the toolkit "
            "repository. Alternatively, run `streamlit run app/main.py` "
            "directly from the repo root."
        )

    sys.argv = ["streamlit", "run", str(app_main)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
