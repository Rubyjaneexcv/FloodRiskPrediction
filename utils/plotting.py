"""
Shared Plotly styling so every chart in the app looks consistent.

Centralises the font, colour palette, margins and grid styling that would
otherwise be copy-pasted into each view.
"""

from __future__ import annotations

from typing import Any

import config

FONT_FAMILY = "Inter, sans-serif"
GRID_COLOR = "#EDF1F7"
TEXT_COLOR = config.THEME.text
MUTED = config.THEME.text_muted
PRIMARY = config.THEME.primary

# Ordered low -> high risk colours (matches config.RISK_LEVELS).
RISK_COLORWAY = [lvl.color for lvl in config.RISK_LEVELS]


def style_figure(fig: Any, height: int | None = None, showlegend: bool = False) -> Any:
    """Apply the shared visual template to a Plotly figure in place."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
        colorway=[PRIMARY],
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    if height:
        fig.update_layout(height=height)
    return fig
