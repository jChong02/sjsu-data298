"""
Shared visualization utilities for the Streamlit app.
Renders token-level attributions as highlighted HTML and bar charts.
"""

import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Color palette - teal/orange, colorblind-friendly, less "alarming" than red
# ---------------------------------------------------------------------------
# Supports answer (positive attribution): warm orange-amber
# Against answer (negative attribution): cool teal-blue
# Neutral (near-zero): subtle warm gray

_POS_COLOR = (232, 153, 35)   # amber-orange
_NEG_COLOR = (56, 163, 165)   # teal
_NEUTRAL_BG = (58, 58, 66)    # dark warm gray (for near-zero tokens)


def _score_to_colors(score: float) -> tuple:
    """
    Map a normalized score in [-1, 1] to (background_rgb, text_color).
    Near-zero scores get a neutral gray rather than washed-out white.
    """
    abs_score = min(abs(score), 1.0)

    # Below a threshold, use neutral background
    if abs_score < 0.08:
        bg = f"rgb({_NEUTRAL_BG[0]},{_NEUTRAL_BG[1]},{_NEUTRAL_BG[2]})"
        return bg, "#aaa"

    if score >= 0:
        # Interpolate from neutral → full orange
        r = int(_NEUTRAL_BG[0] + (_POS_COLOR[0] - _NEUTRAL_BG[0]) * abs_score)
        g = int(_NEUTRAL_BG[1] + (_POS_COLOR[1] - _NEUTRAL_BG[1]) * abs_score)
        b = int(_NEUTRAL_BG[2] + (_POS_COLOR[2] - _NEUTRAL_BG[2]) * abs_score)
    else:
        # Interpolate from neutral → full teal
        r = int(_NEUTRAL_BG[0] + (_NEG_COLOR[0] - _NEUTRAL_BG[0]) * abs_score)
        g = int(_NEUTRAL_BG[1] + (_NEG_COLOR[1] - _NEUTRAL_BG[1]) * abs_score)
        b = int(_NEUTRAL_BG[2] + (_NEG_COLOR[2] - _NEUTRAL_BG[2]) * abs_score)

    # Perceived luminance for text contrast
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = "#111" if luminance > 150 else "#f0f0f0"

    return f"rgb({r},{g},{b})", text_color


def _normalize_scores(scores, method: str = "symmetric") -> np.ndarray:
    """
    Normalize attribution scores.

    method='symmetric' : scale to [-1, 1] by dividing by max absolute value
    method='minmax'    : scale to [0, 1]
    """
    scores = np.array(scores, dtype=float)
    if len(scores) == 0:
        return scores

    if method == "symmetric":
        max_abs = np.max(np.abs(scores))
        if max_abs > 0:
            return scores / max_abs
        return scores
    elif method == "minmax":
        smin, smax = scores.min(), scores.max()
        if smax - smin > 0:
            return (scores - smin) / (smax - smin)
        return np.zeros_like(scores)
    return scores


def render_token_highlights_html(
    tokens: List[str],
    scores,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> str:
    """
    Render tokens with colored backgrounds based on attribution scores.
    Returns an HTML string for use with st.markdown(unsafe_allow_html=True).
    """
    scores = _normalize_scores(scores, method="symmetric")

    html_parts = []
    if title:
        html_parts.append(f'<div style="margin-bottom:8px;font-weight:600;">{title}</div>')

    html_parts.append(
        '<div style="line-height:2.4;font-family:\'JetBrains Mono\',\'Fira Code\',monospace;font-size:13px;">'
    )
    for token, score in zip(tokens, scores):
        bg, text_color = _score_to_colors(score)
        display = token.replace("▁", " ").replace("Ġ", " ")
        html_parts.append(
            f'<span style="background:{bg};color:{text_color};padding:3px 6px;margin:2px;'
            f'border-radius:4px;display:inline-block;border:1px solid rgba(255,255,255,0.06);">'
            f'{display}</span>'
        )
    html_parts.append("</div>")

    if show_legend:
        pos_rgb = f"rgb({_POS_COLOR[0]},{_POS_COLOR[1]},{_POS_COLOR[2]})"
        neg_rgb = f"rgb({_NEG_COLOR[0]},{_NEG_COLOR[1]},{_NEG_COLOR[2]})"
        neu_rgb = f"rgb({_NEUTRAL_BG[0]},{_NEUTRAL_BG[1]},{_NEUTRAL_BG[2]})"
        html_parts.append(
            f'<div style="margin-top:10px;font-size:12px;color:#999;display:flex;gap:16px;align-items:center;">'
            f'<span style="background:{pos_rgb};padding:2px 10px;border-radius:3px;color:#111;">'
            f'&nbsp;</span> <span>Supports</span>'
            f'<span style="background:{neg_rgb};padding:2px 10px;border-radius:3px;color:#111;">'
            f'&nbsp;</span> <span>Against</span>'
            f'<span style="background:{neu_rgb};padding:2px 10px;border-radius:3px;color:#aaa;">'
            f'&nbsp;</span> <span>Neutral</span>'
            f'</div>'
        )

    return "\n".join(html_parts)


# Bar chart color constants (for plotly)
BAR_COLOR_POS = "#e89923"   # amber-orange
BAR_COLOR_NEG = "#38a3a5"   # teal

# Shared plotly layout for dark theme consistency
_GRID_COLOR = "rgba(255,255,255,0.06)"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#999", family="'Inter', sans-serif"),
)


def apply_plotly_theme(fig):
    """Apply shared dark theme to a plotly figure, preserving existing axis settings."""
    fig.update_layout(**PLOTLY_THEME)
    fig.update_xaxes(gridcolor=_GRID_COLOR)
    fig.update_yaxes(gridcolor=_GRID_COLOR)
    return fig


def get_top_k_data(
    tokens: List[str],
    scores,
    k: int = 15,
) -> Tuple[List[str], List[float]]:
    """
    Get top-k tokens by absolute attribution score.
    Returns (labels, values) sorted by absolute value descending.
    """
    scores = np.array(scores, dtype=float)
    indices = np.argsort(np.abs(scores))[::-1][:k]

    labels = []
    values = []
    for idx in indices:
        token = tokens[idx].replace("▁", " ").replace("Ġ", " ").strip()
        if not token:
            token = f"[pos {idx}]"
        labels.append(token)
        values.append(float(scores[idx]))

    return labels, values
