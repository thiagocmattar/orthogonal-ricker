"""Central visual contract for repository paper figures.

Report 04 is the current reference. Figure-specific layouts and scientific
cohorts remain with their owning family; reusable colors, markers, typography,
and export defaults live here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLORBLIND_SAFE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
ADAMW_COLOR = "#000000"
ADAMW_LINEWIDTH = 2.6
ADAMW_MARKER_SCALE = 1.35
DEFAULT_SERIES_LINEWIDTH = 1.2

REPORT04_METHOD_COLORS = {
    "GELU AdamW": "#000000",
    "MLP-ReLU AdamW": "#0072B2",
    "MLP-ReLU OL1": "#E69F00",
    "Three-ReLU AdamW": "#009E73",
    "Three-ReLU OR": "#D55E00",
    "Three-ReLU L1N": "#56B4E9",
    "Three-ReLU OL1": "#CC79A7",
}
REPORT04_METHOD_MARKERS = {
    "GELU AdamW": "D",
    "MLP-ReLU AdamW": "o",
    "MLP-ReLU OL1": "s",
    "Three-ReLU AdamW": "^",
    "Three-ReLU OR": "X",
    "Three-ReLU L1N": "v",
    "Three-ReLU OL1": "P",
}

PLOT_STYLE = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE_COLORS),
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
