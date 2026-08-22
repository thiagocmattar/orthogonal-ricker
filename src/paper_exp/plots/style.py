"""Shared visual defaults for diagnostics and paper figures.

Scientific cohorts, reductions, labels, and panel choices belong to the
figure-family module that owns them. This module contains presentation-only
tokens so plots use a consistent, colorblind-safe visual language.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLORBLIND_SAFE_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)
SERIES_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "h")


@dataclass(frozen=True)
class SeriesStyle:
    """Presentation identity for one series, independent of its label."""

    color: str
    marker: str
    linestyle: str = "-"
    linewidth: float = 1.4


def series_style(index: int) -> SeriesStyle:
    """Return a deterministic color-and-marker pair for a series index."""

    if index < 0:
        raise ValueError("Series index must be nonnegative.")
    color_index = index % len(COLORBLIND_SAFE_COLORS)
    marker_index = (
        color_index + index // len(COLORBLIND_SAFE_COLORS)
    ) % len(SERIES_MARKERS)
    linestyles = ("-", "--", "-.", ":")
    style_block = index // (len(COLORBLIND_SAFE_COLORS) * len(SERIES_MARKERS))
    return SeriesStyle(
        color=COLORBLIND_SAFE_COLORS[color_index],
        marker=SERIES_MARKERS[marker_index],
        linestyle=linestyles[style_block % len(linestyles)],
    )


PLOT_STYLE = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE_COLORS),
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Paper figures are authored on a white canvas at their final size. Tight
# bounding boxes are disabled because cropping would change the validated
# canvas dimensions and make PDF MediaBoxes depend on annotation placement.
PAPER_STYLE = {
    **PLOT_STYLE,
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#B0B0B0",
    "legend.facecolor": "white",
    "legend.edgecolor": "#B0B0B0",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.bbox": None,
}
