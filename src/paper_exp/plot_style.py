"""Central visual contract for repository paper figures.

Report 04 is the current reference. Figure-specific layouts and scientific
cohorts remain with their owning family; reusable colors, markers, typography,
and export defaults live here.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SeriesStyle:
    """Visual identity for one scientific series, independent of its label."""

    color: str
    marker: str
    linestyle: str = "-"
    linewidth: float = 1.4


REPORT04_METHOD_LABELS = {
    "gelu_adamw": "GELU AdamW",
    "mlp_relu_adamw": "MLP-ReLU AdamW",
    "mlp_relu_ol1": "MLP-ReLU OL1",
    "three_relu_adamw": "Three-ReLU AdamW",
    "three_relu_rn": "Three-ReLU RN",
    "three_relu_or": "Three-ReLU OR",
    "three_relu_l1n": "Three-ReLU L1N",
    "three_relu_ol1": "Three-ReLU OL1",
}
REPORT04_METHOD_IDS = {label: method_id for method_id, label in REPORT04_METHOD_LABELS.items()}
REPORT04_METHOD_STYLES = {
    "gelu_adamw": SeriesStyle("#000000", "D", "-", 1.7),
    "mlp_relu_adamw": SeriesStyle("#0072B2", "o", "--"),
    "mlp_relu_ol1": SeriesStyle("#E69F00", "s", "-."),
    "three_relu_adamw": SeriesStyle("#009E73", "^", "-", 1.6),
    "three_relu_rn": SeriesStyle("#6F4C9B", "h", ":"),
    "three_relu_or": SeriesStyle("#D55E00", "X", "--"),
    "three_relu_l1n": SeriesStyle("#56B4E9", "v", "-."),
    "three_relu_ol1": SeriesStyle("#CC79A7", "P", ":"),
}


def report04_method_style(method_id_or_label: str) -> SeriesStyle:
    """Resolve a Report 04 style from a stable ID or its display label."""

    method_id = REPORT04_METHOD_IDS.get(method_id_or_label, method_id_or_label)
    try:
        return REPORT04_METHOD_STYLES[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown Report 04 method style: {method_id_or_label!r}") from exc


# Compatibility maps for the existing renderers and external callers.
REPORT04_METHOD_COLORS = {
    label: report04_method_style(method_id).color
    for method_id, label in REPORT04_METHOD_LABELS.items()
}
REPORT04_METHOD_MARKERS = {
    label: report04_method_style(method_id).marker
    for method_id, label in REPORT04_METHOD_LABELS.items()
}
REPORT04_METHOD_LINESTYLES = {
    label: report04_method_style(method_id).linestyle
    for method_id, label in REPORT04_METHOD_LABELS.items()
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

# Report 04 figures are authored at their final two-column width.  Disabling
# tight bounding boxes is deliberate: otherwise annotations outside an axes can
# silently change the PDF MediaBox and defeat publication-size checks.
REPORT04_PLOT_STYLE = {
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
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "savefig.bbox": None,
}
