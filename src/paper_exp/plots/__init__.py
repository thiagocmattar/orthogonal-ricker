"""Focused plotting package for explicit saved experiment artifacts."""

from .activation_histograms import build_activation_histograms
from .clipping import build_clipping_frontier
from .dispatch import PLOT_KINDS, plot_artifact
from .propagation import build_activation_propagation
from .run_diagnostics import build_run_diagnostics
from .weight_histograms import build_weight_histograms

__all__ = [
    "PLOT_KINDS",
    "build_activation_histograms",
    "build_activation_propagation",
    "build_clipping_frontier",
    "build_run_diagnostics",
    "build_weight_histograms",
    "plot_artifact",
]
