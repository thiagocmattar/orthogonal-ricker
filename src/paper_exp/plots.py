"""Stable plotting facade, paper dispatch, and legacy figure families.

The CLI and existing imports enter here. Shared presentation tokens live in
``plot_style.py``; reusable pure helpers live in ``plot_common.py``; Report 04
constants, reductions, and renderers live together in ``plot_report04.py``.
Older families remain here until they can be moved with the same contract and
visual-parity coverage.

Report 04 (figures 79--90) is the current visual baseline. Its data-backed
selectors require a coherent terminal run envelope; the architecture-only
figures 87 and 90 do not depend on a complete training cohort.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from paper_exp.plot_common import (
    _finite,
    _histogram_center_window_mass,
    _histogram_density,
    _histogram_layer,
    _histogram_method,
    _trimmed_decimal_tick,
)
from paper_exp.plot_api import (
    DOUBLE_COLUMN_WIDTH_INCHES,
    REPORT04_PUBLICATION_PROFILE,
    GridLayout,
    PublicationProfile,
    export_figure,
    publish_staged_outputs,
)
from paper_exp.plot_catalog import REPORT04_FIGURES, REPORT05_FIGURES
from paper_exp.plot_report04 import (
    POST_LAYERNORM_RELU_PROPAGATION_EXPERIMENT,
    PROPAGATION_ACTIVATION_ROWS,
    PROPAGATION_MATMUL_ROWS,
    REPORT04_BLOCK_PRODUCTS_PER_TOKEN,
    REPORT04_BLOCK_SIZE,
    REPORT04_CLIPPING_RUNS,
    REPORT04_CLIPPING_SITES,
    REPORT04_HIDDEN_SIZE,
    REPORT04_HISTOGRAM_METHOD_LABELS,
    REPORT04_INPUT_HISTOGRAM_EXPERIMENT,
    REPORT04_INTERMEDIATE_SIZE,
    REPORT04_JOINT_CLIPPING_RUNS,
    REPORT04_LM_HEAD_PRODUCTS_PER_TOKEN,
    REPORT04_MLP_HISTOGRAM_EXPERIMENT,
    REPORT04_MODEL_PRODUCTS_PER_TOKEN,
    REPORT04_NUM_LAYERS,
    REPORT04_PYTHIA_FAMILY,
    REPORT04_RN_PROPAGATION_EXPERIMENT,
    REPORT04_RN_TRAINING_RUNS,
    REPORT04_TARGET_MODEL_FRACTION,
    REPORT04_TARGET_PRODUCTS_PER_TOKEN,
    REPORT04_TRAINING_RUNS,
    REPORT04_VOCAB_SIZE,
    _load_report04_parameter_series,
    _plot_post_layernorm_relu_propagation_heatmaps,
    _plot_post_layernorm_relu_zero_product_heatmaps,
    _plot_report04_activation_densities,
    _plot_report04_activation_heatmaps,
    _plot_report04_activation_weight_densities,
    _plot_report04_joint_compute_frontier,
    _plot_report04_layernorm_parameters,
    _plot_report04_learning_diagnostics,
    _plot_report04_parameter_diagnostics,
    _plot_report04_pythia_family_compute_ceiling,
    _plot_report04_site_clipping_frontiers,
    _plot_report04_three_relu_architecture,
)
from paper_exp.plot_report05 import (
    REPORT05_ARCHITECTURE_FAMILIES,
    REPORT05_GATE_HISTOGRAM_EXPERIMENT,
    REPORT05_INPUT_HISTOGRAM_EXPERIMENT,
    REPORT05_MLP_HISTOGRAM_EXPERIMENT,
    REPORT05_PINNED_RUN_IDS,
    REPORT05_PROPAGATION_EXPERIMENT,
    REPORT05_TRAINING_RUNS,
    _plot_report05_architecture_schematic,
    _plot_report05_validation_learning_curves,
)
from paper_exp.plot_report05_clipping import (
    REPORT05_ACTIVE_CLIPPING_SITES,
    _plot_report05_model_matmul_frontiers,
    _plot_report05_site_clipping_frontiers,
    _reduce_report05_model_matmul_frontiers,
    _reduce_report05_site_clipping_frontiers,
)
from paper_exp.plot_report05_diagnostics import (
    REPORT05_HEATMAP_PROFILE,
    _plot_report05_one_relu_distributions,
    _plot_report05_one_relu_propagation_heatmaps,
    _plot_report05_six_relu_post_distributions,
    _plot_report05_six_relu_post_propagation_heatmaps,
    _plot_report05_six_relu_pre_distributions,
    _plot_report05_six_relu_pre_propagation_heatmaps,
    _plot_report05_three_relu_distributions,
    _plot_report05_three_relu_propagation_heatmaps,
)
from paper_exp.plot_style import (
    ADAMW_COLOR,
    ADAMW_LINEWIDTH,
    ADAMW_MARKER_SCALE,
    COLORBLIND_SAFE_COLORS,
    DEFAULT_SERIES_LINEWIDTH,
    PLOT_STYLE,
    REPORT04_METHOD_COLORS,
    REPORT04_METHOD_LINESTYLES,
    REPORT04_METHOD_MARKERS,
    REPORT04_PLOT_STYLE,
)
from paper_exp.run import CORE_RUN_ARTIFACTS
from paper_exp.utils import read_json

PRESSURE_SHORT_RUNS = [
    ("AdamW baseline", "03-pythia-14m-minipile-random-full-10min"),
    ("Ricker naive", "08-pythia-14m-minipile-ricker-naive-short"),
    ("L1 naive", "09-pythia-14m-minipile-l1-naive-short"),
    ("Orthogonal Ricker", "10-pythia-14m-minipile-orthogonal-ricker-short"),
    ("Orthogonal L1", "11-pythia-14m-minipile-orthogonal-l1-short"),
]

FIXED_STEP_SWEEP_NAME = "pressure_fixed_step_v1"
FIXED_STEP_ROLE_LABELS = {
    "adamw": "AdamW",
    "ricker_naive": "Ricker naive",
    "orthogonal_ricker": "Orthogonal Ricker",
    "l1_naive": "L1 naive",
    "orthogonal_l1": "Orthogonal L1",
}
FIXED_STEP_ROLE_MARKERS = {
    "adamw": "D",
    "ricker_naive": "o",
    "orthogonal_ricker": "s",
    "l1_naive": "^",
    "orthogonal_l1": "P",
}
FIXED_STEP_METHOD_FIGURES = (
    (8, "ricker_naive", "naive-ricker", "Naive Ricker"),
    (9, "l1_naive", "naive-l1", "Naive L1"),
    (10, "orthogonal_ricker", "orthogonal-ricker", "Orthogonal Ricker"),
    (11, "orthogonal_l1", "orthogonal-l1", "Orthogonal L1"),
)
SELECTED_CLIPPING_FRONTIER_CONFIGS = (12, 29, 33, 18, 25)
HIGH_PRESSURE_EXPANSION_CONFIGS = tuple([12, *range(35, 49)])
HIGH_PRESSURE_LEARNING_FIGURES = (
    (17, (12, *range(35, 40)), "rn", "High-pressure RN Learning Curves", "AdamW plus RN configs 35-39"),
    (18, (12, *range(40, 45)), "or", "High-pressure OR Learning Curves", "AdamW plus OR configs 40-44"),
    (19, (12, 45, 46, 47, 48), "l1", "High-pressure L1N/OL1 Learning Curves", "AdamW plus L1N/OL1 configs 45-48"),
)
HIGH_PRESSURE_OR_L1_NORM_CONFIGS = tuple([12, *range(40, 49)])
SELECTED_ACTIVATION_HISTOGRAM_EXPERIMENT = "49-pythia-14m-pressure-fixed-2048-selected-activation-histograms"
FULL_PASS_HIGH_PRESSURE_ACTIVATION_HISTOGRAM_EXPERIMENT = (
    "60-pythia-14m-minipile-full-pass-high-pressure-activation-histograms"
)
FULL_PASS_HIGH_PRESSURE_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "63-pythia-14m-minipile-full-pass-high-pressure-residual-stream-histograms"
)
FULL_PASS_HIGH_PRESSURE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "64-pythia-14m-minipile-full-pass-high-pressure-attention-output-histograms"
)
FULL_PASS_HIGH_PRESSURE_WEIGHT_HISTOGRAM_EXPERIMENT = (
    "61-pythia-14m-minipile-full-pass-high-pressure-weight-histograms"
)
FULL_PASS_HIGH_PRESSURE_ATTENTION_WEIGHT_HISTOGRAM_EXPERIMENT = (
    "62-pythia-14m-minipile-full-pass-high-pressure-attention-weight-histograms"
)
FULL_PASS_ALL_SITE_MLP_HISTOGRAM_EXPERIMENT = (
    "67-pythia-14m-minipile-full-pass-all-site-pressure-mlp-activation-histograms"
)
FULL_PASS_ALL_SITE_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "68-pythia-14m-minipile-full-pass-all-site-pressure-residual-stream-histograms"
)
FULL_PASS_ALL_SITE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "69-pythia-14m-minipile-full-pass-all-site-pressure-attention-output-histograms"
)
FULL_PASS_MLP_RESIDUAL_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "74-pythia-14m-minipile-full-pass-mlp-residual-pressure-attention-output-histograms"
)
FULL_PASS_MLP_RESIDUAL_MLP_HISTOGRAM_EXPERIMENT = (
    "75-pythia-14m-minipile-full-pass-mlp-residual-pressure-mlp-activation-histograms"
)
FULL_PASS_MLP_RESIDUAL_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "76-pythia-14m-minipile-full-pass-mlp-residual-pressure-residual-stream-histograms"
)
FULL_PASS_RELU_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "82-pythia-14m-minipile-relu-completed-attention-output-histograms"
)
FULL_PASS_RELU_MLP_HISTOGRAM_EXPERIMENT = (
    "83-pythia-14m-minipile-relu-completed-mlp-activation-histograms"
)
FULL_PASS_RELU_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "84-pythia-14m-minipile-relu-completed-residual-stream-histograms"
)
FULL_PASS_RELU_MLP_INPUT_HISTOGRAM_EXPERIMENT = (
    "85-pythia-14m-minipile-relu-completed-mlp-input-histograms"
)
RELU_SITE_SCOPE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "94-pythia-14m-minipile-relu-site-scope-attention-output-histograms"
)
RELU_SITE_SCOPE_MLP_HISTOGRAM_EXPERIMENT = (
    "95-pythia-14m-minipile-relu-site-scope-mlp-activation-histograms"
)
RELU_SITE_SCOPE_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "96-pythia-14m-minipile-relu-site-scope-residual-stream-histograms"
)
RELU_SITE_SCOPE_MLP_INPUT_HISTOGRAM_EXPERIMENT = (
    "97-pythia-14m-minipile-relu-site-scope-mlp-input-histograms"
)
STATUS_UPDATE_COUPLING_METHODS = (
    ("AdamW", "AdamW"),
    ("L1N w5", "L1N w5"),
    ("RN w1 c0.05 s0.05", "RN w1 c0.05 s0.05"),
)
STATUS_UPDATE_RELU_COUPLING_METHODS = (
    ("AdamW ReLU", "AdamW ReLU"),
    ("RN ReLU", "RN ReLU"),
    ("OR ReLU", "OR ReLU"),
    ("L1N ReLU", "L1N ReLU"),
    ("OL1 ReLU", "OL1 ReLU"),
)
STATUS_UPDATE_COUPLING_SITE_SPECS = (
    ("mlp_hiddens", "MLP hiddens", "mlp_hiddens.layer_3", (-0.08, 0.75)),
    ("residual_streams", "Residual streams", "residual_streams.layer_3", (-0.35, 0.35)),
    ("attention_outputs", "Attention outputs", "attention_outputs.layer_3", (-0.35, 0.35)),
)
STATUS_UPDATE_RELU_COUPLING_SITE_SPECS = (
    ("mlp_inputs", "Post-LN MLP inputs", "mlp_inputs.layer_3", (-4.0, 4.0)),
    ("mlp_hiddens", "MLP hiddens", "mlp_hiddens.layer_3", (-0.08, 0.75)),
    ("residual_streams", "Residual streams", "residual_streams.layer_3", (-0.35, 0.35)),
    ("attention_outputs", "Attention outputs", "attention_outputs.layer_3", (-0.35, 0.35)),
)
STATUS_UPDATE_COUPLING_WEIGHT_RUNS = (
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("L1N w5", "58-pythia-14m-minipile-l1-naive-full-pass-w5"),
    ("RN w1 c0.05 s0.05", "57-pythia-14m-minipile-ricker-naive-full-pass-w1-c0p05-s0p05"),
)
STATUS_UPDATE_RELU_WEIGHT_RUNS = (
    ("AdamW ReLU", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("RN ReLU", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR ReLU", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N ReLU", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
    ("OL1 ReLU", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
)
STATUS_UPDATE_PRESSURE_WEIGHT_GROUPS = (
    (
        "mlp_hidden_input",
        "MLP hiddens",
        re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$"),
        (-0.25, 0.25),
    ),
    (
        "residual_write",
        "Residual streams",
        re.compile(r"^gpt_neox\.layers\.\d+\.(?:mlp\.dense_4h_to_h|attention\.dense)\.weight$"),
        (-0.75, 0.75),
    ),
    (
        "attention_qkv",
        "Attention",
        re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"),
        (-0.75, 0.75),
    ),
)
STATUS_UPDATE_MLP_WEIGHT_LAYER_GROUPS = (
    (
        "mlp_layer_0",
        "MLP layer 0",
        re.compile(r"^gpt_neox\.layers\.0\.mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$"),
        (-0.25, 0.25),
    ),
    (
        "mlp_layer_3",
        "MLP layer 3",
        re.compile(r"^gpt_neox\.layers\.3\.mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$"),
        (-0.25, 0.25),
    ),
    (
        "mlp_layer_5",
        "MLP layer 5",
        re.compile(r"^gpt_neox\.layers\.5\.mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$"),
        (-0.25, 0.25),
    ),
)
FULL_PASS_SELECTED_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("L1N w0.5", "51-pythia-14m-minipile-l1-naive-full-pass-w0p5"),
    ("OL1 w0.5", "52-pythia-14m-minipile-orthogonal-l1-full-pass-w0p5"),
    ("RN w0.1 c0.05 s0.025", "53-pythia-14m-minipile-ricker-naive-full-pass-w0p1-c0p05-s0p025"),
    ("OR w0.1 c0.05 s0.025", "54-pythia-14m-minipile-orthogonal-ricker-full-pass-w0p1-c0p05-s0p025"),
]
FULL_PASS_HIGH_PRESSURE_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("RN w1 c0.05 s0.05", "57-pythia-14m-minipile-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR w1 c0.05 s0.05", "56-pythia-14m-minipile-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N w5", "58-pythia-14m-minipile-l1-naive-full-pass-w5"),
    ("OL1 w5", "59-pythia-14m-minipile-orthogonal-l1-full-pass-w5"),
]
FULL_PASS_ALL_SITE_PRESSURE_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("OR all-site w1 c0.05 s0.05", "65-pythia-14m-minipile-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05"),
    ("OL1 all-site w5", "66-pythia-14m-minipile-orthogonal-l1-all-site-full-pass-w5"),
]
FULL_PASS_MLP_RESIDUAL_PRESSURE_RUNS = [
    ("RN MLP+res w1 c0.05 s0.05", "70-pythia-14m-minipile-ricker-naive-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("OR MLP+res w1 c0.05 s0.05", "71-pythia-14m-minipile-orthogonal-ricker-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("L1N MLP+res w5", "72-pythia-14m-minipile-l1-naive-mlp-residual-full-pass-w5"),
    ("OL1 MLP+res w5", "73-pythia-14m-minipile-orthogonal-l1-mlp-residual-full-pass-w5"),
]
FULL_PASS_RELU_COMPLETED_RUNS = [
    ("AdamW ReLU", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("RN ReLU w1 c0.05 s0.05", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR ReLU w1 c0.05 s0.05", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N ReLU w5", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
    ("OL1 ReLU w5", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
]
STATUS_UPDATE_CLIPPING_SITES = (
    ("mlp_hiddens", "MLP hiddens"),
    ("residual_streams", "Residual streams"),
    ("attention_outputs", "Attention outputs"),
)
STATUS_UPDATE_ALL_SITE_FRONTIER_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("OR all-site", "65-pythia-14m-minipile-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05"),
    ("OL1 all-site", "66-pythia-14m-minipile-orthogonal-l1-all-site-full-pass-w5"),
]
STATUS_UPDATE_MLP_RESIDUAL_FRONTIER_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("RN MLP+res", "70-pythia-14m-minipile-ricker-naive-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("OR MLP+res", "71-pythia-14m-minipile-orthogonal-ricker-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("L1N MLP+res", "72-pythia-14m-minipile-l1-naive-mlp-residual-full-pass-w5"),
    ("OL1 MLP+res", "73-pythia-14m-minipile-orthogonal-l1-mlp-residual-full-pass-w5"),
]
STATUS_UPDATE_RELU_FRONTIER_RUNS = [
    ("AdamW ReLU", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("RN ReLU", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR ReLU", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N ReLU", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
    ("OL1 ReLU", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
]
RELU_SITE_SCOPE_TRAINING_GROUPS = (
    (
        "MLP-only",
        (
            ("AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            ("RN", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
            ("OR", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
            ("L1N", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
            ("OL1", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
        ),
    ),
    (
        "MLP+residual",
        (
            ("AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            ("RN", "86-pythia-14m-minipile-relu-ricker-naive-mlp-residual-full-pass-w1-c0p05-s0p05"),
            ("OR", "87-pythia-14m-minipile-relu-orthogonal-ricker-mlp-residual-full-pass-w1-c0p05-s0p05"),
            ("L1N", "88-pythia-14m-minipile-relu-l1-naive-mlp-residual-full-pass-w5"),
            ("OL1", "89-pythia-14m-minipile-relu-orthogonal-l1-mlp-residual-full-pass-w5"),
        ),
    ),
    (
        "All-site",
        (
            ("AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            ("RN", "90-pythia-14m-minipile-relu-ricker-naive-all-site-full-pass-w1-c0p05-s0p05"),
            ("OR", "91-pythia-14m-minipile-relu-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05"),
            ("L1N", "92-pythia-14m-minipile-relu-l1-naive-all-site-full-pass-w5"),
            ("OL1", "93-pythia-14m-minipile-relu-orthogonal-l1-all-site-full-pass-w5"),
        ),
    ),
)
RELU_SITE_SCOPE_HISTOGRAM_GROUPS = (
    (
        "MLP-only",
        (
            ("AdamW", "AdamW"),
            ("RN", "MLP-only RN"),
            ("OR", "MLP-only OR"),
            ("L1N", "MLP-only L1N"),
            ("OL1", "MLP-only OL1"),
        ),
    ),
    (
        "MLP+residual",
        (
            ("AdamW", "AdamW"),
            ("RN", "MLP+res RN"),
            ("OR", "MLP+res OR"),
            ("L1N", "MLP+res L1N"),
            ("OL1", "MLP+res OL1"),
        ),
    ),
    (
        "All-site",
        (
            ("AdamW", "AdamW"),
            ("RN", "All-site RN"),
            ("OR", "All-site OR"),
            ("L1N", "All-site L1N"),
            ("OL1", "All-site OL1"),
        ),
    ),
)
RELU_SITE_SCOPE_REPORT_METHODS = (
    ("AdamW", "AdamW"),
    ("MLP-only RN", "MLP-only RN"),
    ("MLP-only OR", "MLP-only OR"),
    ("MLP-only L1N", "MLP-only L1N"),
    ("MLP-only OL1", "MLP-only OL1"),
    ("MLP+res RN", "MLP+res RN"),
    ("MLP+res OR", "MLP+res OR"),
    ("MLP+res L1N", "MLP+res L1N"),
    ("MLP+res OL1", "MLP+res OL1"),
    ("All-site RN", "All-site RN"),
    ("All-site OR", "All-site OR"),
    ("All-site L1N", "All-site L1N"),
    ("All-site OL1", "All-site OL1"),
)
RELU_SITE_SCOPE_DENSITY_SITE_SPECS = (
    ("mlp_inputs", "Post-LN MLP inputs", "mlp_inputs.layer_3", (-4.0, 4.0)),
    ("mlp_hiddens", "MLP hiddens", "mlp_hiddens.layer_3", (-0.08, 0.75)),
    ("residual_streams", "Residual streams", "residual_streams.layer_3", (-0.35, 0.35)),
    ("attention_outputs", "Attention outputs", "attention_outputs.layer_3", (-0.35, 0.35)),
)
RELU_SITE_SCOPE_NEAR_ZERO_SITES = (
    ("mlp_hiddens", "MLP hiddens"),
    ("residual_streams", "Residual streams"),
    ("attention_outputs", "Attention outputs"),
)


def generate_plots(
    *,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(results_dir)
    outputs = _generate_known_paper_figures(results_path, figures_path, save_png=save_png)
    if outputs:
        return outputs

    rows = collect_numeric_metrics(results_path)
    output_pdf = figures_path / "01-results-summary.pdf"
    if rows:
        _plot_metric_summary(rows, output_pdf)
    else:
        _plot_empty_placeholder(output_pdf)

    outputs = [output_pdf]
    if save_png:
        output_png = output_pdf.with_suffix(".png")
        if rows:
            _plot_metric_summary(rows, output_png)
        else:
            _plot_empty_placeholder(output_png)
        outputs.append(output_png)

    return outputs


def _generate_known_paper_figures(results_path: Path, figures_path: Path, *, save_png: bool) -> list[Path]:
    outputs: list[Path] = []

    diagnostic_run = _latest_run_with(
        results_path / "03-pythia-14m-minipile-random-full-10min",
        "events.jsonl",
    )
    if diagnostic_run is not None:
        output_pdf = figures_path / "01-pythia-14m-minipile-random-full-10min-diagnostics.pdf"
        outputs.extend(generate_run_diagnostics(run_dir=diagnostic_run, output=output_pdf, save_png=save_png))

    clipping_run = _latest_run_with(
        results_path / "03-pythia-14m-minipile-random-full-10min-clipping-sweep",
        "clipping_frontier.jsonl",
    )
    if clipping_run is not None:
        output_pdf = figures_path / "02-pythia-14m-minipile-clipping-frontier-smoke.pdf"
        outputs.extend(generate_clipping_frontier(run_dir=clipping_run, output=output_pdf, save_png=save_png))

    pressure_runs = _latest_labeled_runs(results_path, PRESSURE_SHORT_RUNS, "events.jsonl")
    if len(pressure_runs) >= 2:
        output_pdf = figures_path / "03-pythia-14m-pressure-short-learning-curves.pdf"
        outputs.extend(generate_pressure_comparison(runs=pressure_runs, output=output_pdf, save_png=save_png))

    clipping_runs = _latest_labeled_runs(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in PRESSURE_SHORT_RUNS[1:]],
        "clipping_frontier.jsonl",
    )
    if len(clipping_runs) >= 2:
        output_pdf = figures_path / "04-pythia-14m-pressure-short-clipping-frontiers.pdf"
        outputs.extend(generate_clipping_comparison(runs=clipping_runs, output=output_pdf, save_png=save_png))

    fixed_step_rows = _load_fixed_step_sweep_rows(results_path)
    if len(fixed_step_rows) >= 2:
        output_pdf = figures_path / "05-pythia-14m-pressure-fixed-2048-summary.pdf"
        outputs.extend(
            generate_fixed_step_sweep_summary(rows=fixed_step_rows, output=output_pdf, save_png=save_png)
        )

        output_pdf = figures_path / "06-pythia-14m-pressure-fixed-2048-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_learning_curves(rows=fixed_step_rows, output=output_pdf, save_png=save_png)
        )

    fixed_step_clipping = _load_fixed_step_clipping_series(results_path, fixed_step_rows)
    if len(fixed_step_clipping) >= 2:
        output_pdf = figures_path / "07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_clipping_frontiers(
                series=fixed_step_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for index, role, slug, role_label in FIXED_STEP_METHOD_FIGURES:
        role_rows = _select_fixed_step_rows_for_role(fixed_step_rows, role)
        if len(role_rows) < 2:
            continue
        output_pdf = figures_path / f"{index:02d}-pythia-14m-pressure-fixed-2048-{slug}-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_role_learning_curves(
                rows=role_rows,
                role_label=role_label,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for offset, role, slug, role_label in FIXED_STEP_METHOD_FIGURES:
        role_series = _select_fixed_step_clipping_for_role(fixed_step_clipping, role)
        if len(role_series) < 2:
            continue
        output_pdf = figures_path / f"{offset + 4:02d}-pythia-14m-pressure-fixed-2048-{slug}-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_role_clipping_frontiers(
                series=role_series,
                role_label=role_label,
                output=output_pdf,
                save_png=save_png,
            )
        )

    selected_clipping = _select_fixed_step_clipping_by_config_indices(
        fixed_step_clipping,
        SELECTED_CLIPPING_FRONTIER_CONFIGS,
    )
    if len(selected_clipping) == len(SELECTED_CLIPPING_FRONTIER_CONFIGS):
        output_pdf = figures_path / "16-pythia-14m-pressure-fixed-2048-selected-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_selected_clipping_frontiers(
                series=selected_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for index, config_indices, slug, title, scope_note in HIGH_PRESSURE_LEARNING_FIGURES:
        high_pressure_rows = _select_fixed_step_rows_by_config_indices(
            fixed_step_rows,
            config_indices,
        )
        if len(high_pressure_rows) < 2:
            continue
        output_pdf = figures_path / f"{index:02d}-pythia-14m-pressure-fixed-2048-high-pressure-{slug}-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_learning_curves(
                rows=high_pressure_rows,
                output=output_pdf,
                title=title,
                scope_note=scope_note,
                save_png=save_png,
            )
        )

    high_pressure_clipping = _select_fixed_step_clipping_by_config_indices(
        fixed_step_clipping,
        HIGH_PRESSURE_EXPANSION_CONFIGS,
    )
    if len(high_pressure_clipping) >= 2:
        output_pdf = figures_path / "20-pythia-14m-pressure-fixed-2048-high-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_clipping_frontiers(
                series=high_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    high_pressure_or_l1_rows = _select_fixed_step_rows_by_config_indices(
        fixed_step_rows,
        HIGH_PRESSURE_OR_L1_NORM_CONFIGS,
    )
    if len(high_pressure_or_l1_rows) >= 2:
        output_pdf = figures_path / "21-pythia-14m-pressure-fixed-2048-or-l1-weight-norms.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_weight_norms(
                rows=high_pressure_or_l1_rows,
                output=output_pdf,
                save_png=save_png,
            )
        )

    activation_histogram_run = _latest_run_with(
        results_path / SELECTED_ACTIVATION_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if activation_histogram_run is not None:
        output_pdf = figures_path / "22-pythia-14m-pressure-fixed-2048-selected-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=activation_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_runs = _latest_labeled_runs(results_path, FULL_PASS_SELECTED_RUNS, "events.jsonl")
    if len(full_pass_runs) >= 2:
        output_pdf = figures_path / "28-pythia-14m-minipile-full-pass-selected-gradient-diagnostics.pdf"
        outputs.extend(
            generate_full_pass_gradient_diagnostics(
                runs=full_pass_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )
        output_pdf = figures_path / "30-pythia-14m-minipile-full-pass-pressure-dominance.pdf"
        outputs.extend(
            generate_full_pass_pressure_dominance(
                runs=full_pass_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_rms_clipping = _latest_labeled_runs(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in FULL_PASS_SELECTED_RUNS],
        "clipping_frontier.jsonl",
    )
    full_pass_rms_clipping = [
        (label, run)
        for label, run in full_pass_rms_clipping
        if _is_rms_clipping_run(run)
    ]
    if len(full_pass_rms_clipping) >= 2:
        output_pdf = figures_path / "29-pythia-14m-minipile-full-pass-selected-rms-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_rms_clipping_frontiers(
                runs=full_pass_rms_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_runs = _latest_labeled_runs(results_path, FULL_PASS_HIGH_PRESSURE_RUNS, "events.jsonl")
    if len(full_pass_high_pressure_runs) >= 2:
        output_pdf = figures_path / "31-pythia-14m-minipile-full-pass-high-pressure-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_high_pressure_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "32-pythia-14m-minipile-full-pass-high-pressure-weight-norms.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_weight_norms(
                runs=full_pass_high_pressure_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "34-pythia-14m-minipile-full-pass-high-pressure-gradient-diagnostics.pdf"
        outputs.extend(
            generate_full_pass_gradient_diagnostics(
                runs=full_pass_high_pressure_runs,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass High-pressure Gradient Diagnostics",
                subtitle=(
                    "n={n} runs; {events} train log events; "
                    "pressure panels omit AdamW where no pressure is applied"
                ),
            )
        )

    full_pass_high_pressure_clipping = _latest_labeled_runs_filtered(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in FULL_PASS_HIGH_PRESSURE_RUNS],
        "clipping_frontier.jsonl",
        predicate=lambda run: not _is_rms_clipping_run(run),
    )
    if len(full_pass_high_pressure_clipping) >= 2:
        output_pdf = figures_path / "33-pythia-14m-minipile-full-pass-high-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=full_pass_high_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_all_site_clipping = _latest_labeled_runs_filtered(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep-all-sites") for label, experiment_id in FULL_PASS_HIGH_PRESSURE_RUNS],
        "clipping_frontier.jsonl",
        predicate=_is_all_site_clipping_run,
    )
    if len(full_pass_high_pressure_all_site_clipping) >= 2:
        output_pdf = figures_path / "40-pythia-14m-minipile-full-pass-high-pressure-all-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=full_pass_high_pressure_all_site_clipping,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass High-pressure All-site Post-hoc Clipping Frontiers",
                subtitle=(
                    "absolute thresholds applied to MLP hiddens, attention outputs, and residual streams; "
                    "validation-loss axis zoomed"
                ),
            )
        )

    full_pass_high_pressure_site_frontiers = (
        (
            49,
            "mlp_hiddens",
            "MLP Hiddens",
            "Full-pass MLP-pressure High-pressure MLP-only Post-hoc Clipping Frontiers",
            "pressure trained only on MLP hiddens; clipping thresholds applied only to MLP hiddens",
        ),
        (
            50,
            "residual_streams",
            "Residual Streams",
            "Full-pass MLP-pressure High-pressure Residual-only Post-hoc Clipping Frontiers",
            "pressure trained only on MLP hiddens; clipping thresholds applied only to residual streams",
        ),
        (
            51,
            "attention_outputs",
            "Attention Outputs",
            "Full-pass MLP-pressure High-pressure Attention-only Post-hoc Clipping Frontiers",
            "pressure trained only on MLP hiddens; clipping thresholds applied only to attention outputs",
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_high_pressure_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_HIGH_PRESSURE_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = (
            figures_path
            / f"{figure_index:02d}-pythia-14m-minipile-full-pass-high-pressure-{slug}-clipping-frontiers.pdf"
        )
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=f"{subtitle}; validation-loss axis zoomed",
            )
        )

    full_pass_all_site_pressure_runs = _latest_labeled_runs(
        results_path,
        FULL_PASS_ALL_SITE_PRESSURE_RUNS,
        "events.jsonl",
    )
    if len(full_pass_all_site_pressure_runs) >= 2:
        output_pdf = figures_path / "41-pythia-14m-minipile-full-pass-all-site-pressure-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_all_site_pressure_runs,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass All-site Pressure Learning Curves",
                subtitle="AdamW baseline plus all-site OR and OL1; one MiniPile token-cache pass per pressure run",
            )
        )

    full_pass_all_site_pressure_clipping = _latest_labeled_runs_filtered(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep-all-sites") for label, experiment_id in FULL_PASS_ALL_SITE_PRESSURE_RUNS],
        "clipping_frontier.jsonl",
        predicate=_is_all_site_clipping_run,
    )
    if len(full_pass_all_site_pressure_clipping) >= 2:
        output_pdf = figures_path / "42-pythia-14m-minipile-full-pass-all-site-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=full_pass_all_site_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass All-site Pressure Post-hoc Clipping Frontiers",
                subtitle=(
                    "absolute thresholds applied to MLP hiddens, attention outputs, and residual streams; "
                    "validation-loss axis zoomed"
                ),
            )
        )

    full_pass_all_site_pressure_site_frontiers = (
        (
            46,
            "mlp_hiddens",
            "MLP Hiddens",
            "Full-pass All-site Pressure MLP-only Post-hoc Clipping Frontiers",
            "absolute thresholds applied only to MLP hiddens; validation-loss axis zoomed",
        ),
        (
            47,
            "residual_streams",
            "Residual Streams",
            "Full-pass All-site Pressure Residual-only Post-hoc Clipping Frontiers",
            "absolute thresholds applied only to residual streams; validation-loss axis zoomed",
        ),
        (
            48,
            "attention_outputs",
            "Attention Outputs",
            "Full-pass All-site Pressure Attention-only Post-hoc Clipping Frontiers",
            "absolute thresholds applied only to attention outputs; validation-loss axis zoomed",
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_all_site_pressure_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_ALL_SITE_PRESSURE_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = (
            figures_path
            / f"{figure_index:02d}-pythia-14m-minipile-full-pass-all-site-pressure-{slug}-clipping-frontiers.pdf"
        )
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=subtitle,
            )
        )

    full_pass_mlp_residual_pressure_runs = _latest_labeled_runs(
        results_path,
        FULL_PASS_MLP_RESIDUAL_PRESSURE_RUNS,
        "events.jsonl",
    )
    if len(full_pass_mlp_residual_pressure_runs) >= 2:
        output_pdf = figures_path / "52-pythia-14m-minipile-full-pass-mlp-residual-pressure-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_mlp_residual_pressure_runs,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass MLP+Residual Pressure Learning Curves",
                subtitle=(
                    "pressure applied to MLP hiddens and residual streams; "
                    "one MiniPile token-cache pass per run"
                ),
            )
        )

    full_pass_mlp_residual_site_frontiers = (
        (
            53,
            "attention_outputs",
            "Attention Outputs",
            "Full-pass MLP+Residual Pressure Attention-only Post-hoc Clipping Frontiers",
            (
                "pressure trained on MLP hiddens and residual streams; "
                "clipping thresholds applied only to attention outputs; validation-loss axis zoomed"
            ),
        ),
        (
            54,
            "residual_streams",
            "Residual Streams",
            "Full-pass MLP+Residual Pressure Residual-only Post-hoc Clipping Frontiers",
            (
                "pressure trained on MLP hiddens and residual streams; "
                "clipping thresholds applied only to residual streams; validation-loss axis zoomed"
            ),
        ),
        (
            55,
            "mlp_hiddens",
            "MLP Hiddens",
            "Full-pass MLP+Residual Pressure MLP-only Post-hoc Clipping Frontiers",
            (
                "pressure trained on MLP hiddens and residual streams; "
                "clipping thresholds applied only to MLP hiddens; validation-loss axis zoomed"
            ),
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_mlp_residual_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_MLP_RESIDUAL_PRESSURE_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = (
            figures_path
            / f"{figure_index:02d}-pythia-14m-minipile-full-pass-mlp-residual-pressure-{slug}-clipping-frontiers.pdf"
        )
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=subtitle,
            )
        )

    full_pass_high_pressure_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_ACTIVATION_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_high_pressure_histogram_run is not None:
        output_pdf = figures_path / "35-pythia-14m-minipile-full-pass-high-pressure-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_high_pressure_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_weight_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_WEIGHT_HISTOGRAM_EXPERIMENT,
        "weight_histograms.json",
    )
    if full_pass_high_pressure_weight_histogram_run is not None:
        output_pdf = figures_path / "36-pythia-14m-minipile-full-pass-high-pressure-weight-histograms.pdf"
        outputs.extend(
            generate_weight_histogram_grid(
                run_dir=full_pass_high_pressure_weight_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_attention_weight_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_ATTENTION_WEIGHT_HISTOGRAM_EXPERIMENT,
        "weight_histograms.json",
    )
    if full_pass_high_pressure_attention_weight_histogram_run is not None:
        output_pdf = figures_path / "37-pythia-14m-minipile-full-pass-high-pressure-attention-weight-histograms.pdf"
        outputs.extend(
            generate_weight_histogram_grid(
                run_dir=full_pass_high_pressure_attention_weight_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_high_pressure_residual_histogram_run is not None:
        output_pdf = figures_path / "38-pythia-14m-minipile-full-pass-high-pressure-residual-stream-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_high_pressure_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_high_pressure_attention_output_histogram_run is not None:
        output_pdf = figures_path / "39-pythia-14m-minipile-full-pass-high-pressure-attention-output-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_high_pressure_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_all_site_mlp_histogram_run = _latest_run_with(
        results_path / FULL_PASS_ALL_SITE_MLP_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_all_site_mlp_histogram_run is not None:
        output_pdf = figures_path / "43-pythia-14m-minipile-full-pass-all-site-pressure-mlp-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_all_site_mlp_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_all_site_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_ALL_SITE_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_all_site_residual_histogram_run is not None:
        output_pdf = figures_path / "44-pythia-14m-minipile-full-pass-all-site-pressure-residual-stream-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_all_site_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_all_site_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_ALL_SITE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_all_site_attention_output_histogram_run is not None:
        output_pdf = figures_path / "45-pythia-14m-minipile-full-pass-all-site-pressure-attention-output-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_all_site_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_mlp_residual_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_MLP_RESIDUAL_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_mlp_residual_attention_output_histogram_run is not None:
        output_pdf = (
            figures_path
            / "56-pythia-14m-minipile-full-pass-mlp-residual-pressure-attention-output-histograms.pdf"
        )
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_mlp_residual_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_mlp_residual_mlp_histogram_run = _latest_run_with(
        results_path / FULL_PASS_MLP_RESIDUAL_MLP_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_mlp_residual_mlp_histogram_run is not None:
        output_pdf = (
            figures_path
            / "57-pythia-14m-minipile-full-pass-mlp-residual-pressure-mlp-activation-histograms.pdf"
        )
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_mlp_residual_mlp_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_mlp_residual_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_MLP_RESIDUAL_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_mlp_residual_residual_histogram_run is not None:
        output_pdf = (
            figures_path
            / "58-pythia-14m-minipile-full-pass-mlp-residual-pressure-residual-stream-histograms.pdf"
        )
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_mlp_residual_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_relu_completed_runs = _latest_labeled_runs(
        results_path,
        FULL_PASS_RELU_COMPLETED_RUNS,
        "events.jsonl",
    )
    if len(full_pass_relu_completed_runs) >= 2:
        output_pdf = figures_path / "59-pythia-14m-minipile-relu-completed-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_relu_completed_runs,
                output=output_pdf,
                save_png=save_png,
                title="ReLU Completed Method Learning Curves",
                subtitle=(
                    "completed ReLU runs: AdamW, RN, OR, L1N, and OL1; "
                    "one MiniPile token-cache pass per run"
                ),
            )
        )

    full_pass_relu_site_frontiers = (
        (
            60,
            "attention_outputs",
            "Attention Outputs",
            "ReLU Completed Methods Attention-only Post-hoc Clipping Frontiers",
            (
                "completed ReLU runs; clipping thresholds applied only to attention outputs; "
                "validation-loss axis zoomed"
            ),
        ),
        (
            61,
            "residual_streams",
            "Residual Streams",
            "ReLU Completed Methods Residual-only Post-hoc Clipping Frontiers",
            (
                "completed ReLU runs; clipping thresholds applied only to residual streams; "
                "validation-loss axis zoomed"
            ),
        ),
        (
            62,
            "mlp_hiddens",
            "MLP Hiddens",
            "ReLU Completed Methods MLP-only Post-hoc Clipping Frontiers",
            (
                "completed ReLU runs; clipping thresholds applied only to MLP hiddens; "
                "validation-loss axis zoomed"
            ),
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_relu_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_RELU_COMPLETED_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = figures_path / f"{figure_index:02d}-pythia-14m-minipile-relu-completed-{slug}-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=subtitle,
            )
        )

    full_pass_relu_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_RELU_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_relu_attention_output_histogram_run is not None:
        output_pdf = figures_path / "63-pythia-14m-minipile-relu-completed-attention-output-densities.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_relu_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_relu_mlp_histogram_run = _latest_run_with(
        results_path / FULL_PASS_RELU_MLP_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_relu_mlp_histogram_run is not None:
        output_pdf = figures_path / "64-pythia-14m-minipile-relu-completed-mlp-activation-densities.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_relu_mlp_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_relu_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_RELU_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_relu_residual_histogram_run is not None:
        output_pdf = figures_path / "65-pythia-14m-minipile-relu-completed-residual-stream-densities.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_relu_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report_coupling_histogram_runs = {
        "mlp_hiddens": _latest_run_with(
            results_path / FULL_PASS_HIGH_PRESSURE_ACTIVATION_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "residual_streams": _latest_run_with(
            results_path / FULL_PASS_HIGH_PRESSURE_RESIDUAL_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "attention_outputs": _latest_run_with(
            results_path / FULL_PASS_HIGH_PRESSURE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(report_coupling_histogram_runs.values()):
        output_pdf = figures_path / "66-status-update-gelu-mlp-only-coupling-density-comparison.pdf"
        outputs.extend(
            generate_status_update_coupling_density_comparison(
                histogram_runs=report_coupling_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report_relu_coupling_histogram_runs = {
        "mlp_inputs": _latest_run_with(
            results_path / FULL_PASS_RELU_MLP_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / FULL_PASS_RELU_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "residual_streams": _latest_run_with(
            results_path / FULL_PASS_RELU_RESIDUAL_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "attention_outputs": _latest_run_with(
            results_path / FULL_PASS_RELU_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(report_relu_coupling_histogram_runs.values()):
        output_pdf = figures_path / "73-status-update-relu-coupling-density-comparison.pdf"
        outputs.extend(
            generate_status_update_coupling_density_comparison(
                histogram_runs=report_relu_coupling_histogram_runs,
                output=output_pdf,
                save_png=save_png,
                methods=STATUS_UPDATE_RELU_COUPLING_METHODS,
                site_specs=STATUS_UPDATE_RELU_COUPLING_SITE_SPECS,
                title="ReLU MLP-only Pressure Coupling Diagnostic",
                subtitle=(
                    "rows are completed ReLU AdamW, RN, OR, L1N, and OL1"
                ),
            )
        )

    report_pressure_weight_runs = _latest_labeled_runs(
        results_path,
        list(STATUS_UPDATE_COUPLING_WEIGHT_RUNS),
        "checkpoints/final/model.safetensors",
    )
    if report_pressure_weight_runs:
        output_pdf = figures_path / "70-status-update-gelu-mlp-only-pressure-weight-diagnostic.pdf"
        outputs.extend(
            generate_status_update_pressure_weight_diagnostic(
                runs=report_pressure_weight_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report_relu_weight_runs = _latest_labeled_runs(
        results_path,
        list(STATUS_UPDATE_RELU_WEIGHT_RUNS),
        "checkpoints/final/model.safetensors",
    )
    if report_relu_weight_runs:
        output_pdf = figures_path / "71-status-update-relu-pressure-weight-diagnostic.pdf"
        outputs.extend(
            generate_status_update_pressure_weight_diagnostic(
                runs=report_relu_weight_runs,
                output=output_pdf,
                save_png=save_png,
                title="ReLU Completed Pressure Weight Diagnostic",
                subtitle=(
                    "rows are completed AdamW, RN, OR, L1N, and OL1 ReLU runs; "
                    "columns aggregate final-checkpoint weights across all six layers"
                ),
                footnote=(
                    "Shaded band marks |weight| <= 0.01. "
                    "Residual-stream column uses MLP and attention output-projection weights."
                ),
            )
        )

        output_pdf = figures_path / "72-status-update-relu-mlp-weight-density-comparison.pdf"
        outputs.extend(
            generate_status_update_pressure_weight_diagnostic(
                runs=report_relu_weight_runs,
                output=output_pdf,
                save_png=save_png,
                group_specs=STATUS_UPDATE_MLP_WEIGHT_LAYER_GROUPS,
                title="ReLU Completed MLP Dense Weight Diagnostic",
                subtitle=(
                    "rows are completed AdamW, RN, OR, L1N, and OL1 ReLU runs; "
                    "columns show representative MLP dense layers"
                ),
                footnote="Shaded band marks |weight| <= 0.01. Biases excluded.",
            )
        )

    report_all_site_clipping = _latest_status_update_site_clipping_runs(
        results_path,
        STATUS_UPDATE_ALL_SITE_FRONTIER_RUNS,
    )
    if _has_status_update_site_clipping_runs(report_all_site_clipping):
        output_pdf = figures_path / "67-status-update-all-site-pressure-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_status_update_site_clipping_frontiers(
                site_runs=report_all_site_clipping,
                output=output_pdf,
                save_png=save_png,
                title="All-site Pressure: Site-specific Post-hoc Clipping",
                subtitle="one panel per clipped activation family; AdamW baseline included",
            )
        )

    report_mlp_residual_clipping = _latest_status_update_site_clipping_runs(
        results_path,
        STATUS_UPDATE_MLP_RESIDUAL_FRONTIER_RUNS,
    )
    if _has_status_update_site_clipping_runs(report_mlp_residual_clipping):
        output_pdf = figures_path / "68-status-update-mlp-residual-pressure-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_status_update_site_clipping_frontiers(
                site_runs=report_mlp_residual_clipping,
                output=output_pdf,
                save_png=save_png,
                title="MLP+Residual Pressure: Site-specific Post-hoc Clipping",
                subtitle="pressure trained on MLP hiddens and residual streams; AdamW baseline included",
            )
        )

    report_relu_clipping = _latest_status_update_site_clipping_runs(
        results_path,
        STATUS_UPDATE_RELU_FRONTIER_RUNS,
    )
    if _has_status_update_site_clipping_runs(report_relu_clipping):
        output_pdf = figures_path / "69-status-update-relu-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_status_update_site_clipping_frontiers(
                site_runs=report_relu_clipping,
                output=output_pdf,
                save_png=save_png,
                title="ReLU Completed Runs: Site-specific Post-hoc Clipping",
                subtitle="completed AdamW/RN/OR/L1N/OL1 ReLU runs",
            )
        )

    relu_site_scope_training_groups = _latest_grouped_labeled_runs(
        results_path,
        RELU_SITE_SCOPE_TRAINING_GROUPS,
        "events.jsonl",
    )
    if relu_site_scope_training_groups and all(len(runs) >= 2 for _scope, runs in relu_site_scope_training_groups):
        output_pdf = figures_path / "74-pythia-14m-minipile-relu-site-scope-learning-curves.pdf"
        outputs.extend(
            generate_relu_site_scope_learning_curves(
                grouped_runs=relu_site_scope_training_groups,
                output=output_pdf,
                save_png=save_png,
            )
        )

    relu_site_scope_clipping = _latest_relu_site_scope_site_clipping_groups(results_path)
    if _has_relu_site_scope_site_clipping_runs(relu_site_scope_clipping):
        output_pdf = figures_path / "75-status-update-relu-site-scope-clipping-frontiers.pdf"
        outputs.extend(
            generate_relu_site_scope_clipping_frontiers(
                grouped_site_runs=relu_site_scope_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    relu_site_scope_histogram_runs = {
        "mlp_inputs": _latest_run_with(
            results_path / RELU_SITE_SCOPE_MLP_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / RELU_SITE_SCOPE_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "residual_streams": _latest_run_with(
            results_path / RELU_SITE_SCOPE_RESIDUAL_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "attention_outputs": _latest_run_with(
            results_path / RELU_SITE_SCOPE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(relu_site_scope_histogram_runs.values()):
        output_pdf = figures_path / "76-status-update-relu-site-scope-activation-density-comparison.pdf"
        outputs.extend(
            generate_relu_site_scope_activation_density_comparison(
                histogram_runs=relu_site_scope_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "78-status-update-relu-site-scope-near-zero-heatmaps.pdf"
        outputs.extend(
            generate_relu_site_scope_near_zero_heatmaps(
                histogram_runs=relu_site_scope_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    relu_site_scope_weight_groups = _latest_grouped_labeled_runs(
        results_path,
        RELU_SITE_SCOPE_TRAINING_GROUPS,
        "checkpoints/final/model.safetensors",
    )
    if relu_site_scope_weight_groups and all(len(runs) >= 2 for _scope, runs in relu_site_scope_weight_groups):
        output_pdf = figures_path / "77-status-update-relu-site-scope-pressure-weight-diagnostic.pdf"
        outputs.extend(
            generate_relu_site_scope_pressure_weight_diagnostic(
                grouped_runs=relu_site_scope_weight_groups,
                output=output_pdf,
                save_png=save_png,
            )
        )

    outputs.extend(
        generate_report04_figures(
            results_path,
            figures_path,
            save_png=save_png,
            strict=False,
        )
    )

    return outputs


class Report04InputError(RuntimeError):
    """Raised when the strict Report 04 suite cannot resolve every input."""


class Report05InputError(RuntimeError):
    """Raised when the strict Report 05 suite cannot resolve every input."""


_REPORT05_METHODS = ("AdamW", "OR", "OL1")
_REPORT05_STANDARD_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=8.8,
    min_text_points=8.0,
)
_REPORT05_THREE_RELU_SITE_SUFFIX = {
    "attention_inputs": "report04-attention-inputs",
    "mlp_inputs": "report04-mlp-inputs",
    "mlp_hiddens": "report04-mlp-hiddens",
}
_REPORT05_SIX_RELU_SITE_SUFFIX = {
    "attention_inputs": "r05s-a",
    "mlp_inputs": "r05s-m",
    "mlp_hiddens": "r05s-h",
    "query_gate_outputs": "r05s-q",
    "key_gate_outputs": "r05s-k",
    "value_gate_outputs": "r05s-v",
}


def generate_report05_figures(
    results_dir: str | Path,
    figures_dir: str | Path,
    save_png: bool = False,
    strict: bool = True,
) -> list[Path]:
    """Generate the complete Report 05 suite from its pinned training cohort.

    Strict mode resolves every saved input before rendering and publishes the
    staged suite atomically. ``strict=False`` is an explicit exploratory mode:
    it renders only complete figure families and does not weaken the exact-run
    selection used by any family that is present.
    """

    if not strict:
        return _generate_report05_figures_in_place(
            results_dir,
            figures_dir,
            save_png=save_png,
            strict=False,
        )

    figures_path = Path(figures_dir)
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".report05-stage-",
        dir=figures_path.parent,
    ) as staging_directory:
        staged_outputs = _generate_report05_figures_in_place(
            results_dir,
            staging_directory,
            save_png=save_png,
            strict=True,
        )
        figures_path.mkdir(parents=True, exist_ok=True)
        final_outputs = [figures_path / path.name for path in staged_outputs]
        publish_staged_outputs(dict(zip(final_outputs, staged_outputs, strict=True)))
    return final_outputs


def _generate_report05_figures_in_place(
    results_dir: str | Path,
    figures_dir: str | Path,
    *,
    save_png: bool,
    strict: bool,
) -> list[Path]:
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)

    training_events = _pinned_report05_training_runs(results_path, "events.jsonl")
    training_checkpoints = _pinned_report05_training_runs(
        results_path,
        "checkpoints/final/model.safetensors",
    )
    propagation_run = _latest_run_with(
        results_path / REPORT05_PROPAGATION_EXPERIMENT,
        "activation_propagation.json",
        require_complete_run=True,
    )
    histogram_runs = {
        "inputs": _latest_run_with(
            results_path / REPORT05_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
            require_complete_run=True,
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / REPORT05_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
            require_complete_run=True,
        ),
        "gates": _latest_run_with(
            results_path / REPORT05_GATE_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
            require_complete_run=True,
        ),
    }
    site_clipping_runs = _report05_site_clipping_runs(results_path)
    joint_clipping_runs = _report05_joint_clipping_runs(results_path)

    issues = _report05_input_issues(
        training_events=training_events,
        training_checkpoints=training_checkpoints,
        propagation_run=propagation_run,
        histogram_runs=histogram_runs,
        site_clipping_runs=site_clipping_runs,
        joint_clipping_runs=joint_clipping_runs,
    )
    if strict and issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise Report05InputError(
            "Report 05 input preflight failed; no figures were generated:\n"
            f"{details}\n"
            "Use --allow-partial to generate only complete figure families."
        )

    outputs: list[Path] = []

    def output(number: int) -> Path:
        entry = next(entry for entry in REPORT05_FIGURES if entry.number == number)
        return figures_path / entry.filename

    outputs.extend(
        generate_report05_architecture_schematic(
            output=output(91),
            save_png=save_png,
        )
    )

    training_complete = len(training_events) == len(REPORT05_TRAINING_RUNS)
    checkpoints_complete = len(training_checkpoints) == len(REPORT05_TRAINING_RUNS)
    histograms_complete = all(histogram_runs.values())
    site_clipping_complete = _report05_site_clipping_complete(site_clipping_runs)
    joint_clipping_complete = _report05_joint_clipping_complete(joint_clipping_runs)

    if training_complete:
        outputs.extend(
            generate_report05_learning_curves(
                runs=training_events,
                output=output(92),
                save_png=save_png,
            )
        )

    if propagation_run is not None:
        propagation_generators = (
            (93, generate_report05_one_relu_propagation),
            (94, generate_report05_three_relu_propagation),
            (95, generate_report05_six_relu_pre_propagation),
            (96, generate_report05_six_relu_post_propagation),
        )
        for number, generator in propagation_generators:
            outputs.extend(
                generator(
                    run_dir=propagation_run,
                    output=output(number),
                    save_png=save_png,
                )
            )

    if histograms_complete and checkpoints_complete:
        distribution_generators = (
            (97, generate_report05_one_relu_distributions),
            (98, generate_report05_three_relu_distributions),
            (99, generate_report05_six_relu_pre_distributions),
            (100, generate_report05_six_relu_post_distributions),
        )
        for number, generator in distribution_generators:
            outputs.extend(
                generator(
                    histogram_runs=histogram_runs,
                    runs=training_checkpoints,
                    output=output(number),
                    save_png=save_png,
                )
            )

    if site_clipping_complete:
        outputs.extend(
            generate_report05_site_clipping_frontiers(
                site_runs=site_clipping_runs,
                output=output(101),
                save_png=save_png,
            )
        )

    if joint_clipping_complete:
        outputs.extend(
            generate_report05_model_compute_frontiers(
                joint_runs=joint_clipping_runs,
                output=output(102),
                save_png=save_png,
            )
        )

    return outputs


def _pinned_report05_training_runs(
    results_path: Path,
    artifact_name: str,
) -> list[tuple[str, Path]]:
    """Resolve only the declared Report 05 run IDs, never a newer sibling."""

    selected: list[tuple[str, Path]] = []
    for label, experiment_id in REPORT05_TRAINING_RUNS:
        run_id = REPORT05_PINNED_RUN_IDS[experiment_id]
        run_dir = results_path / experiment_id / run_id
        if (run_dir / artifact_name).is_file() and _has_coherent_terminal_manifest(run_dir):
            selected.append((label, run_dir))
    return selected


def _report05_architecture_runs() -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        architecture_id: tuple(
            (method, experiment_id)
            for method, (_label, experiment_id) in zip(
                _REPORT05_METHODS,
                family_runs,
                strict=True,
            )
        )
        for architecture_id, _architecture_label, family_runs in REPORT05_ARCHITECTURE_FAMILIES
    }


def _report05_site_clipping_runs(
    results_path: Path,
) -> dict[str, dict[str, list[tuple[str, Path]]]]:
    selected: dict[str, dict[str, list[tuple[str, Path]]]] = {}
    for architecture_id, method_runs in _report05_architecture_runs().items():
        selected[architecture_id] = {}
        for site in REPORT05_ACTIVE_CLIPPING_SITES[architecture_id]:
            experiments: list[tuple[str, str]] = []
            for method, experiment_id in method_runs:
                if architecture_id == "one_relu":
                    suffix = "report05-exact-joint"
                elif architecture_id == "three_relu":
                    suffix = _REPORT05_THREE_RELU_SITE_SUFFIX[site]
                else:
                    suffix = _REPORT05_SIX_RELU_SITE_SUFFIX[site]
                experiments.append(
                    (method, f"{experiment_id}-clipping-sweep-{suffix}")
                )
            selected[architecture_id][site] = _latest_labeled_runs(
                results_path,
                experiments,
                "clipping_frontier.jsonl",
                require_complete_run=True,
            )
    return selected


def _report05_joint_clipping_runs(
    results_path: Path,
) -> dict[str, list[tuple[str, Path]]]:
    return {
        architecture_id: _latest_labeled_runs(
            results_path,
            [
                (
                    method,
                    f"{experiment_id}-clipping-sweep-report05-exact-joint",
                )
                for method, experiment_id in method_runs
            ],
            "clipping_frontier.jsonl",
            require_complete_run=True,
        )
        for architecture_id, method_runs in _report05_architecture_runs().items()
    }


def _report05_site_clipping_complete(
    runs: dict[str, dict[str, list[tuple[str, Path]]]],
) -> bool:
    return all(
        len(runs.get(architecture_id, {}).get(site, [])) == len(_REPORT05_METHODS)
        for architecture_id, sites in REPORT05_ACTIVE_CLIPPING_SITES.items()
        for site in sites
    )


def _report05_joint_clipping_complete(
    runs: dict[str, list[tuple[str, Path]]],
) -> bool:
    return all(
        len(runs.get(architecture_id, [])) == len(_REPORT05_METHODS)
        for architecture_id in _report05_architecture_runs()
    )


def _report05_input_issues(
    *,
    training_events: list[tuple[str, Path]],
    training_checkpoints: list[tuple[str, Path]],
    propagation_run: Path | None,
    histogram_runs: dict[str, Path | None],
    site_clipping_runs: dict[str, dict[str, list[tuple[str, Path]]]],
    joint_clipping_runs: dict[str, list[tuple[str, Path]]],
) -> list[str]:
    issues: list[str] = []
    for role, selected in (
        ("training events", training_events),
        ("final checkpoints", training_checkpoints),
    ):
        selected_labels = {label for label, _run in selected}
        missing = [
            f"{label} ({experiment_id}/{REPORT05_PINNED_RUN_IDS[experiment_id]})"
            for label, experiment_id in REPORT05_TRAINING_RUNS
            if label not in selected_labels
        ]
        if missing:
            issues.append(f"{role}: missing " + ", ".join(missing))
    if propagation_run is None:
        issues.append(
            f"activation propagation: missing completed {REPORT05_PROPAGATION_EXPERIMENT}"
        )
    for role, run in histogram_runs.items():
        if run is None:
            issues.append(f"activation histograms ({role}): missing completed diagnostic run")
    for architecture_id, sites in REPORT05_ACTIVE_CLIPPING_SITES.items():
        for site in sites:
            selected_methods = {
                method
                for method, _run in site_clipping_runs.get(architecture_id, {}).get(site, [])
            }
            missing = [method for method in _REPORT05_METHODS if method not in selected_methods]
            if missing:
                issues.append(
                    f"site clipping {architecture_id}/{site}: missing " + ", ".join(missing)
                )
    for architecture_id in _report05_architecture_runs():
        selected_methods = {
            method for method, _run in joint_clipping_runs.get(architecture_id, [])
        }
        missing = [method for method in _REPORT05_METHODS if method not in selected_methods]
        if missing:
            issues.append(
                f"exact joint clipping {architecture_id}: missing " + ", ".join(missing)
            )
    return issues


def generate_report04_figures(
    results_dir: str | Path,
    figures_dir: str | Path,
    save_png: bool = False,
    strict: bool = True,
    write_provenance: bool = False,
    include_rn: bool = False,
) -> list[Path]:
    """Generate the Report 04 figure suite from coherent terminal runs.

    Selection is resolved for every figure family before rendering starts. The
    default is the published pre-RN cohort; ``include_rn=True`` adds run 105 and
    switches propagation figures to run 106. In strict mode, any missing cohort
    member or training/checkpoint mismatch is reported together. A complete
    strict suite is rendered in a sibling staging directory and promoted with
    rollback only after every figure succeeds, so a renderer failure cannot
    mix old and new artifacts. ``strict=False`` keeps the historical paper
    dispatcher behavior: complete families are rendered and incomplete
    families are skipped. Input provenance is opt-in for direct callers and is
    only valid for a complete strict suite.
    """

    if write_provenance and not strict:
        raise ValueError("Report 04 input provenance requires strict=True.")

    if not strict:
        return _generate_report04_figures_in_place(
            results_dir,
            figures_dir,
            save_png=save_png,
            strict=False,
            write_provenance=False,
            include_rn=include_rn,
        )

    figures_path = Path(figures_dir)
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".report04-stage-",
        dir=figures_path.parent,
    ) as staging_directory:
        staged_outputs = _generate_report04_figures_in_place(
            results_dir,
            staging_directory,
            save_png=save_png,
            strict=True,
            write_provenance=write_provenance,
            include_rn=include_rn,
        )
        figures_path.mkdir(parents=True, exist_ok=True)
        final_outputs = [figures_path / path.name for path in staged_outputs]
        publish_staged_outputs(dict(zip(final_outputs, staged_outputs, strict=True)))
    return final_outputs


def _generate_report04_figures_in_place(
    results_dir: str | Path,
    figures_dir: str | Path,
    save_png: bool = False,
    strict: bool = True,
    write_provenance: bool = False,
    include_rn: bool = False,
) -> list[Path]:
    """Resolve and render one Report 04 cohort into a single destination."""

    if write_provenance and not strict:
        raise ValueError("Report 04 input provenance requires strict=True.")

    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    training_specs = REPORT04_RN_TRAINING_RUNS if include_rn else REPORT04_TRAINING_RUNS
    propagation_experiment = (
        REPORT04_RN_PROPAGATION_EXPERIMENT
        if include_rn
        else POST_LAYERNORM_RELU_PROPAGATION_EXPERIMENT
    )
    cohort = "rn-comparison" if include_rn else "published-pre-rn"

    report04_training_runs = _latest_labeled_runs(
        results_path,
        list(training_specs),
        "events.jsonl",
        require_complete_run=True,
    )
    report04_histogram_runs = {
        "inputs": _latest_run_with(
            results_path / REPORT04_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
            require_complete_run=True,
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / REPORT04_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
            require_complete_run=True,
        ),
    }

    report04_site_clipping_runs: dict[str, list[tuple[str, Path]]] = {}
    for site, _site_label in REPORT04_CLIPPING_SITES:
        suffix = site.replace("_", "-")
        experiments = [
            (label, f"{experiment_id}-clipping-sweep-report04-{suffix}")
            for label, experiment_id in REPORT04_CLIPPING_RUNS
        ]
        report04_site_clipping_runs[site] = _latest_labeled_runs(
            results_path,
            experiments,
            "clipping_frontier.jsonl",
            require_complete_run=True,
        )

    report04_joint_clipping_runs = _latest_labeled_runs(
        results_path,
        [
            (label, f"{experiment_id}-clipping-sweep-report04-joint")
            for label, experiment_id in REPORT04_JOINT_CLIPPING_RUNS
        ],
        "clipping_frontier.jsonl",
        require_complete_run=True,
    )
    report04_parameter_runs = _latest_labeled_runs(
        results_path,
        list(training_specs),
        "checkpoints/final/model.safetensors",
        require_complete_run=True,
    )
    (
        report04_histogram_parameter_runs,
        histogram_parameter_issues,
    ) = _report04_histogram_checkpoint_runs(
        results_path,
        report04_histogram_runs,
    )
    propagation_run = _latest_run_with(
        results_path / propagation_experiment,
        "activation_propagation.json",
        require_complete_run=True,
    )

    if strict:
        input_issues = _report04_input_issues(
            training_specs=training_specs,
            propagation_experiment=propagation_experiment,
            training_runs=report04_training_runs,
            histogram_runs=report04_histogram_runs,
            site_clipping_runs=report04_site_clipping_runs,
            joint_clipping_runs=report04_joint_clipping_runs,
            parameter_runs=report04_parameter_runs,
            histogram_parameter_issues=histogram_parameter_issues,
            propagation_run=propagation_run,
        )
        if input_issues:
            details = "\n".join(f"- {issue}" for issue in input_issues)
            raise Report04InputError(
                "Report 04 input preflight failed; no figures were generated:\n"
                f"{details}\n"
                "Use --allow-partial to generate only complete figure families."
            )

    outputs: list[Path] = []
    training_complete = len(report04_training_runs) == len(training_specs)
    histograms_complete = all(report04_histogram_runs.values())
    site_clipping_complete = all(
        len(runs) == len(REPORT04_CLIPPING_RUNS)
        for runs in report04_site_clipping_runs.values()
    )
    joint_clipping_complete = (
        len(report04_joint_clipping_runs) == len(REPORT04_JOINT_CLIPPING_RUNS)
    )
    parameters_complete = len(report04_parameter_runs) == len(training_specs)
    histogram_parameters_complete = (
        len(report04_histogram_parameter_runs) == len(REPORT04_HISTOGRAM_METHOD_LABELS)
        and not histogram_parameter_issues
    )

    if training_complete:
        outputs.extend(
            generate_report04_learning_diagnostics(
                runs=report04_training_runs,
                training_specs=training_specs,
                output=(
                    figures_path
                    / "79-pythia-14m-minipile-post-layernorm-relu-learning-diagnostics.pdf"
                ),
                save_png=save_png,
            )
        )

    if histograms_complete:
        outputs.extend(
            generate_report04_activation_heatmaps(
                histogram_runs=report04_histogram_runs,
                output=(
                    figures_path
                    / "80-pythia-14m-minipile-post-layernorm-relu-activation-heatmaps.pdf"
                ),
                save_png=save_png,
            )
        )
        outputs.extend(
            generate_report04_activation_densities(
                histogram_runs=report04_histogram_runs,
                output=(
                    figures_path
                    / "81-pythia-14m-minipile-post-layernorm-relu-activation-densities.pdf"
                ),
                save_png=save_png,
            )
        )

    if site_clipping_complete:
        outputs.extend(
            generate_report04_site_clipping_frontiers(
                site_runs=report04_site_clipping_runs,
                output=(
                    figures_path
                    / "82-pythia-14m-minipile-post-layernorm-relu-site-clipping-frontiers.pdf"
                ),
                save_png=save_png,
            )
        )

    if joint_clipping_complete:
        outputs.extend(
            generate_report04_joint_compute_frontier(
                runs=report04_joint_clipping_runs,
                output=(
                    figures_path
                    / "83-pythia-14m-minipile-post-layernorm-relu-joint-compute-frontier.pdf"
                ),
                save_png=save_png,
            )
        )

    if parameters_complete:
        outputs.extend(
            generate_report04_parameter_diagnostics(
                runs=report04_parameter_runs,
                output=(
                    figures_path
                    / "84-pythia-14m-minipile-post-layernorm-relu-parameter-diagnostics.pdf"
                ),
                save_png=save_png,
            )
        )
        outputs.extend(
            generate_report04_layernorm_parameters(
                runs=report04_parameter_runs,
                output=(
                    figures_path
                    / "89-pythia-14m-minipile-post-layernorm-relu-layernorm-parameters.pdf"
                ),
                save_png=save_png,
            )
        )
    if histograms_complete and histogram_parameters_complete:
        outputs.extend(
            generate_report04_activation_weight_densities(
                histogram_runs=report04_histogram_runs,
                runs=report04_histogram_parameter_runs,
                output=(
                    figures_path
                    / "88-pythia-14m-minipile-post-layernorm-relu-activation-weight-densities.pdf"
                ),
                save_png=save_png,
            )
        )

    if propagation_run is not None:
        outputs.extend(
            generate_post_layernorm_relu_propagation_heatmaps(
                run_dir=propagation_run,
                output=(
                    figures_path
                    / "85-pythia-14m-minipile-post-layernorm-relu-zero-propagation-heatmaps.pdf"
                ),
                save_png=save_png,
            )
        )
        outputs.extend(
            generate_post_layernorm_relu_zero_product_heatmaps(
                run_dir=propagation_run,
                output=(
                    figures_path
                    / "86-pythia-14m-minipile-post-layernorm-relu-zero-product-propagation-heatmaps.pdf"
                ),
                save_png=save_png,
            )
        )

    report04_specific_training_ids = {
        experiment_id
        for _label, experiment_id in training_specs
        if experiment_id.startswith(("98-", "99-", "103-", "104-", "105-"))
    }
    report04_context_available = (
        any(
            run.parent.name in report04_specific_training_ids
            for _label, run in report04_training_runs
        )
        or any(report04_histogram_runs.values())
        or any(report04_site_clipping_runs.values())
        or bool(report04_joint_clipping_runs)
        or propagation_run is not None
    )
    if report04_context_available:
        outputs.extend(
            generate_report04_three_relu_architecture(
                output=(
                    figures_path
                    / "87-pythia-14m-minipile-three-relu-architecture-compute-map.pdf"
                ),
                save_png=save_png,
            )
        )
        outputs.extend(
            generate_report04_pythia_family_compute_ceiling(
                output=(
                    figures_path
                    / "90-pythia-family-three-relu-model-compute-ceilings.pdf"
                ),
                save_png=save_png,
            )
        )

    if write_provenance:
        provenance_path = figures_path / "report04-provenance.json"
        provenance = _build_report04_provenance(
            results_path=results_path,
            cohort=cohort,
            propagation_experiment=propagation_experiment,
            training_runs=report04_training_runs,
            histogram_runs=report04_histogram_runs,
            site_clipping_runs=report04_site_clipping_runs,
            joint_clipping_runs=report04_joint_clipping_runs,
            parameter_runs=report04_parameter_runs,
            histogram_parameter_runs=report04_histogram_parameter_runs,
            propagation_run=propagation_run,
            output_paths=outputs,
        )
        _write_report04_provenance(provenance_path, provenance)
        outputs.append(provenance_path)

    return outputs


def _build_report04_provenance(
    *,
    results_path: Path,
    cohort: str,
    propagation_experiment: str,
    training_runs: list[tuple[str, Path]],
    histogram_runs: dict[str, Path | None],
    site_clipping_runs: dict[str, list[tuple[str, Path]]],
    joint_clipping_runs: list[tuple[str, Path]],
    parameter_runs: list[tuple[str, Path]],
    histogram_parameter_runs: list[tuple[str, Path]],
    propagation_run: Path | None,
    output_paths: list[Path],
) -> dict[str, Any]:
    """Describe the selected strict-suite cohort and exact portable inputs."""

    if any(run is None for run in histogram_runs.values()) or propagation_run is None:
        raise ValueError("Report 04 provenance requires a complete strict input cohort.")

    sha256_cache: dict[Path, str] = {}
    training_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=run_dir,
            artifact="events.jsonl",
            role="training_events",
            label=label,
            sha256_cache=sha256_cache,
        )
        for label, run_dir in training_runs
    ]
    histogram_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=histogram_runs[label],
            artifact="activation_histograms.json",
            role="activation_histograms",
            label=label,
            sha256_cache=sha256_cache,
        )
        for label in ("inputs", "mlp_hiddens")
        if histogram_runs[label] is not None
    ]
    site_clipping_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=run_dir,
            artifact="clipping_frontier.jsonl",
            role="site_clipping_frontier",
            label=label,
            site=site,
            sha256_cache=sha256_cache,
        )
        for site, _site_label in REPORT04_CLIPPING_SITES
        for label, run_dir in site_clipping_runs[site]
    ]
    joint_clipping_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=run_dir,
            artifact="clipping_frontier.jsonl",
            role="joint_clipping_frontier",
            label=label,
            sha256_cache=sha256_cache,
        )
        for label, run_dir in joint_clipping_runs
    ]
    parameter_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=run_dir,
            artifact="checkpoints/final/model.safetensors",
            role="final_checkpoint",
            label=label,
            sha256_cache=sha256_cache,
        )
        for label, run_dir in parameter_runs
    ]
    histogram_parameter_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=run_dir,
            artifact="checkpoints/final/model.safetensors",
            role="histogram_source_checkpoint",
            label=label,
            sha256_cache=sha256_cache,
        )
        for label, run_dir in histogram_parameter_runs
    ]
    propagation_inputs = [
        _report04_provenance_input(
            results_path=results_path,
            run_dir=propagation_run,
            artifact="activation_propagation.json",
            role="activation_propagation",
            label=propagation_experiment,
            sha256_cache=sha256_cache,
        )
    ]

    figure_inputs = {
        79: training_inputs,
        80: histogram_inputs,
        81: histogram_inputs,
        82: site_clipping_inputs,
        83: joint_clipping_inputs,
        84: parameter_inputs,
        85: propagation_inputs,
        86: propagation_inputs,
        87: [],
        88: [*histogram_inputs, *histogram_parameter_inputs],
        89: parameter_inputs,
        90: [],
    }
    catalog_numbers = {entry.number for entry in REPORT04_FIGURES}
    if set(figure_inputs) != catalog_numbers:
        raise ValueError("Report 04 provenance input mapping does not match the figure catalog.")

    outputs_by_name = {path.name: path for path in output_paths}
    missing_outputs = [
        entry.filename
        for entry in REPORT04_FIGURES
        if entry.filename not in outputs_by_name
    ]
    if missing_outputs:
        raise ValueError(
            "Report 04 provenance requires every catalog PDF; missing: "
            + ", ".join(missing_outputs)
        )

    figure_outputs = {
        entry.number: [
            {
                "filename": filename,
                "sha256": _report04_artifact_sha256(outputs_by_name[filename], sha256_cache),
                "size_bytes": outputs_by_name[filename].stat().st_size,
            }
            for filename in (
                entry.filename,
                Path(entry.filename).with_suffix(".png").name,
            )
            if filename in outputs_by_name
        ]
        for entry in REPORT04_FIGURES
    }

    return {
        "schema_version": 3,
        "suite": "report04",
        "cohort": cohort,
        "selection_policy": (
            "latest coherent terminal run containing the required artifact under each "
            "declared experiment ID; figure 88 checkpoints use the exact config_id/run_id "
            "recorded consistently by both histogram artifacts"
        ),
        "terminal_run_policy": (
            "config.yaml, metrics.json, predictions.jsonl, and matching manifest config_id/run_id; "
            "manifest status absent or completed"
        ),
        "figures": [
            {
                "number": entry.number,
                "filename": entry.filename,
                "plot_type": entry.plot_type,
                "required_artifact_kinds": list(entry.required_artifact_kinds),
                "public_wrapper": entry.public_wrapper,
                "embedded_in_report": entry.embedded_in_report,
                "inputs": list(figure_inputs[entry.number]),
                "outputs": figure_outputs[entry.number],
            }
            for entry in REPORT04_FIGURES
        ],
    }


def _report04_provenance_input(
    *,
    results_path: Path,
    run_dir: Path,
    artifact: str,
    role: str,
    label: str,
    sha256_cache: dict[Path, str],
    site: str | None = None,
) -> dict[str, Any]:
    relative_run = run_dir.relative_to(results_path)
    artifact_path = run_dir / artifact
    config_path = run_dir / "config.yaml"
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    input_record = {
        "role": role,
        "label": label,
        "experiment_id": run_dir.parent.name,
        "run_id": run_dir.name,
        "run_dir": relative_run.as_posix(),
        "artifact": artifact,
        "artifact_path": (relative_run / artifact).as_posix(),
        "sha256": _report04_artifact_sha256(artifact_path, sha256_cache),
        "run_config": {
            "artifact_path": (relative_run / "config.yaml").as_posix(),
            "sha256": _report04_artifact_sha256(config_path, sha256_cache),
        },
        "run_manifest": {
            "artifact_path": (relative_run / "manifest.json").as_posix(),
            "sha256": _report04_artifact_sha256(manifest_path, sha256_cache),
            "git_commit": manifest.get("git_commit"),
            "git_dirty": manifest.get("git_dirty"),
        },
    }
    if site is not None:
        input_record["site"] = site
    return input_record


def _report04_artifact_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def _write_report04_provenance(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write stable UTF-8 JSON with platform-independent newlines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _report04_histogram_checkpoint_runs(
    results_path: Path,
    histogram_runs: dict[str, Path | None],
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Resolve Figure 88 checkpoints from the histogram source metadata.

    The activation and weight distributions must describe the same saved
    models. Histogram artifacts therefore pin Figure 88 to their recorded
    ``config_id``/``run_id`` pairs instead of the latest run in each training
    experiment.
    """

    issue_prefix = "histogram_matched_checkpoints (figure 88)"
    source_by_role: dict[str, dict[str, tuple[str, str]]] = {}
    issues: list[str] = []

    for role in ("inputs", "mlp_hiddens"):
        run_dir = histogram_runs.get(role)
        if run_dir is None:
            continue
        try:
            payload = read_json(run_dir / "activation_histograms.json")
        except (OSError, UnicodeError, ValueError):
            issues.append(
                f"{issue_prefix}: could not read source metadata from the {role} histogram"
            )
            continue
        methods = payload.get("methods") if isinstance(payload, dict) else None
        if not isinstance(methods, list):
            issues.append(
                f"{issue_prefix}: the {role} histogram has no methods source metadata"
            )
            continue

        role_sources: dict[str, tuple[str, str]] = {}
        for label in REPORT04_HISTOGRAM_METHOD_LABELS:
            matches = [
                method
                for method in methods
                if isinstance(method, dict) and method.get("label") == label
            ]
            if len(matches) != 1:
                qualifier = "no" if not matches else "duplicate"
                issues.append(
                    f"{issue_prefix}: {qualifier} source metadata for {label} "
                    f"in the {role} histogram"
                )
                continue
            config_id = matches[0].get("config_id")
            run_id = matches[0].get("run_id")
            identifiers = (config_id, run_id)
            if any(
                not isinstance(identifier, str)
                or not identifier
                or identifier in {".", ".."}
                or "/" in identifier
                or "\\" in identifier
                for identifier in identifiers
            ):
                issues.append(
                    f"{issue_prefix}: invalid config_id/run_id for {label} "
                    f"in the {role} histogram"
                )
                continue
            role_sources[label] = (config_id, run_id)
        source_by_role[role] = role_sources

    expected_experiment_by_label = dict(REPORT04_TRAINING_RUNS)
    selected: list[tuple[str, Path]] = []
    for label in REPORT04_HISTOGRAM_METHOD_LABELS:
        role_identities = [
            (role, sources[label])
            for role, sources in source_by_role.items()
            if label in sources
        ]
        if not role_identities:
            continue
        distinct_identities = {identity for _role, identity in role_identities}
        if len(distinct_identities) != 1:
            details = ", ".join(
                f"{role}={config_id}/{run_id}"
                for role, (config_id, run_id) in role_identities
            )
            issues.append(
                f"{issue_prefix}: histogram sources disagree for {label} ({details})"
            )
            continue

        config_id, run_id = role_identities[0][1]
        expected_experiment = expected_experiment_by_label.get(label)
        if config_id != expected_experiment:
            issues.append(
                f"{issue_prefix}: {label} records experiment {config_id}, "
                f"expected {expected_experiment}"
            )
            continue
        run_dir = results_path / config_id / run_id
        checkpoint = run_dir / "checkpoints/final/model.safetensors"
        if not _has_coherent_terminal_manifest(run_dir) or not checkpoint.is_file():
            issues.append(
                f"{issue_prefix}: missing coherent source checkpoint for "
                f"{label} ({config_id}/{run_id})"
            )
            continue
        selected.append((label, run_dir))

    return selected, issues


def _report04_input_issues(
    *,
    training_specs: tuple[tuple[str, str], ...],
    propagation_experiment: str,
    training_runs: list[tuple[str, Path]],
    histogram_runs: dict[str, Path | None],
    site_clipping_runs: dict[str, list[tuple[str, Path]]],
    joint_clipping_runs: list[tuple[str, Path]],
    parameter_runs: list[tuple[str, Path]],
    histogram_parameter_issues: list[str],
    propagation_run: Path | None,
) -> list[str]:
    """Return deterministic, human-readable strict-suite preflight issues."""

    issues: list[str] = []
    missing_training = _missing_report04_labeled_runs(training_specs, training_runs)
    if missing_training:
        issues.append(
            "training_events (figure 79): missing coherent events.jsonl for "
            + ", ".join(missing_training)
        )

    missing_histograms = [
        f"{role} ({experiment_id})"
        for role, experiment_id in (
            ("inputs", REPORT04_INPUT_HISTOGRAM_EXPERIMENT),
            ("mlp_hiddens", REPORT04_MLP_HISTOGRAM_EXPERIMENT),
        )
        if histogram_runs.get(role) is None
    ]
    if missing_histograms:
        issues.append(
            "activation_histograms (figures 80, 81, 88): missing coherent "
            "activation_histograms.json for " + ", ".join(missing_histograms)
        )

    missing_sites: list[str] = []
    for site, _site_label in REPORT04_CLIPPING_SITES:
        missing = _missing_report04_labeled_runs(
            REPORT04_CLIPPING_RUNS,
            site_clipping_runs.get(site, []),
        )
        if missing:
            missing_sites.append(f"{site}: {', '.join(missing)}")
    if missing_sites:
        issues.append(
            "site_clipping_frontiers (figure 82): missing coherent clipping_frontier.jsonl for "
            + "; ".join(missing_sites)
        )

    missing_joint = _missing_report04_labeled_runs(
        REPORT04_JOINT_CLIPPING_RUNS,
        joint_clipping_runs,
    )
    if missing_joint:
        issues.append(
            "joint_clipping_frontiers (figure 83): missing coherent clipping_frontier.jsonl for "
            + ", ".join(missing_joint)
        )

    missing_parameters = _missing_report04_labeled_runs(
        training_specs,
        parameter_runs,
    )
    if missing_parameters:
        issues.append(
            "final_checkpoints (figures 84, 89): missing coherent "
            "checkpoints/final/model.safetensors for " + ", ".join(missing_parameters)
        )

    issues.extend(histogram_parameter_issues)

    if propagation_run is None:
        issues.append(
            "activation_propagation (figures 85, 86): missing coherent "
            f"activation_propagation.json for {propagation_experiment}"
        )

    if not missing_training and not missing_parameters:
        event_runs = dict(training_runs)
        checkpoint_runs = dict(parameter_runs)
        mismatched = [
            label
            for label, _experiment_id in training_specs
            if event_runs[label] != checkpoint_runs[label]
        ]
        if mismatched:
            issues.append(
                "training_checkpoint_alignment (figures 79, 84, 89): events and final "
                "checkpoint selected from different runs for " + ", ".join(mismatched)
            )

    return issues


def _missing_report04_labeled_runs(
    expected: tuple[tuple[str, str], ...],
    selected: list[tuple[str, Path]],
) -> list[str]:
    selected_labels = {label for label, _run in selected}
    return [
        f"{label} ({experiment_id})"
        for label, experiment_id in expected
        if label not in selected_labels
    ]


def _latest_run_with(
    experiment_dir: Path,
    artifact_name: str,
    *,
    require_complete_run: bool = False,
) -> Path | None:
    if not experiment_dir.exists():
        return None
    candidates = [
        path
        for path in sorted(experiment_dir.iterdir())
        if (path / artifact_name).exists()
        and (not require_complete_run or _has_coherent_terminal_manifest(path))
    ]
    return candidates[-1] if candidates else None


def _has_coherent_terminal_manifest(run_dir: Path) -> bool:
    if not all((run_dir / artifact).is_file() for artifact in CORE_RUN_ARTIFACTS):
        return False
    try:
        manifest = read_json(run_dir / "manifest.json")
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(manifest, dict):
        return False
    if manifest.get("config_id") != run_dir.parent.name:
        return False
    if manifest.get("run_id") != run_dir.name:
        return False
    status = manifest.get("status")
    return status is None or status == "completed"


def generate_run_diagnostics(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = _read_jsonl(run_path / "events.jsonl")
    metrics = read_json(run_path / "metrics.json")
    train_events = [event for event in events if event.get("event") == "train"]
    validation_events = [event for event in events if event.get("event") == "validation"]
    if not train_events:
        raise ValueError(f"No train events found in {run_path / 'events.jsonl'}")

    _plot_run_diagnostics(train_events, validation_events, metrics, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_run_diagnostics(train_events, validation_events, metrics, png_path)
        outputs.append(png_path)
    return outputs


def generate_clipping_frontier(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frontier_path = run_path / "clipping_frontier.jsonl"
    rows = _read_jsonl(frontier_path)
    if not rows:
        raise ValueError(f"No clipping frontier rows found in {frontier_path}")

    _plot_clipping_frontier(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_clipping_frontier(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_pressure_comparison(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_pressure_series(runs)
    if not series:
        raise ValueError("No pressure comparison runs with train events were found.")

    _plot_pressure_comparison(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_pressure_comparison(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_clipping_comparison(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No clipping comparison runs were found.")

    _plot_clipping_comparison(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_clipping_comparison(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_gradient_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    title: str = "Full-pass Selected Methods Gradient Diagnostics",
    subtitle: str | None = None,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No full-pass train events were found.")

    _plot_full_pass_gradient_diagnostics(series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_gradient_diagnostics(series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_full_pass_rms_clipping_frontiers(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No RMS clipping comparison runs were found.")

    _plot_full_pass_rms_clipping_frontiers(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_rms_clipping_frontiers(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_pressure_dominance(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No full-pass train events were found.")

    _plot_full_pass_pressure_dominance(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_pressure_dominance(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_high_pressure_learning_curves(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    title: str = "Full-pass High-pressure Learning Curves",
    subtitle: str = "one MiniPile token-cache pass per run",
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No high-pressure full-pass train events were found.")

    _plot_full_pass_high_pressure_learning_curves(series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_high_pressure_learning_curves(series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_full_pass_high_pressure_weight_norms(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No high-pressure full-pass weight-norm events were found.")

    _plot_full_pass_high_pressure_weight_norms(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_high_pressure_weight_norms(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_high_pressure_clipping_frontiers(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    title: str = "Full-pass High-pressure Post-hoc Clipping Frontiers",
    subtitle: str = "absolute thresholds, no RMS normalization; validation-loss axis zoomed",
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No high-pressure full-pass clipping frontier runs were found.")

    _plot_full_pass_high_pressure_clipping_frontiers(series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_high_pressure_clipping_frontiers(series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_status_update_coupling_density_comparison(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
    methods: tuple[tuple[str, str], ...] = STATUS_UPDATE_COUPLING_METHODS,
    site_specs: tuple[tuple[str, str, str, tuple[float, float]], ...] = STATUS_UPDATE_COUPLING_SITE_SPECS,
    title: str = "GELU MLP-only Pressure Coupling Diagnostic",
    subtitle: str = "rows are AdamW, L1N w5, and RN w1 c0.05 s0.05",
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        site: read_json(Path(run_dir) / "activation_histograms.json")
        for site, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_status_update_coupling_density_comparison(
        payloads,
        output_path,
        methods=methods,
        site_specs=site_specs,
        title=title,
        subtitle=subtitle,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_status_update_coupling_density_comparison(
            payloads,
            png_path,
            methods=methods,
            site_specs=site_specs,
            title=title,
            subtitle=subtitle,
        )
        outputs.append(png_path)
    return outputs


def generate_status_update_pressure_weight_diagnostic(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    group_specs: tuple[tuple[str, str, Any, tuple[float, float]], ...] = STATUS_UPDATE_PRESSURE_WEIGHT_GROUPS,
    title: str = "GELU MLP-only Pressure Weight Diagnostic",
    subtitle: str = (
        "rows are AdamW, L1N w5, and RN w1 c0.05 s0.05; "
        "columns aggregate final-checkpoint weights across all six layers"
    ),
    footnote: str = (
        "Shaded band marks |weight| <= 0.01. "
        "Residual-stream column uses MLP and attention output-projection weights."
    ),
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_status_update_pressure_weight_groups(runs, group_specs=group_specs)
    _plot_status_update_pressure_weight_diagnostic(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        footnote=footnote,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_status_update_pressure_weight_diagnostic(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            footnote=footnote,
        )
        outputs.append(png_path)
    return outputs


def generate_status_update_site_clipping_frontiers(
    *,
    site_runs: dict[str, list[tuple[str, str | Path]]],
    output: str | Path,
    title: str,
    subtitle: str,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    site_series = {site: _load_clipping_series(runs) for site, runs in site_runs.items()}
    _plot_status_update_site_clipping_frontiers(site_series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_status_update_site_clipping_frontiers(site_series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_learning_curves(
    *,
    grouped_runs: list[tuple[str, list[tuple[str, str | Path]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_series = [(scope, _load_event_series(runs)) for scope, runs in grouped_runs]
    grouped_series = [(scope, series) for scope, series in grouped_series if series]
    if not grouped_series:
        raise ValueError("No ReLU site-scope learning-curve runs were found.")

    _plot_relu_site_scope_learning_curves(grouped_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_learning_curves(grouped_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_clipping_frontiers(
    *,
    grouped_site_runs: list[tuple[str, dict[str, list[tuple[str, str | Path]]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_site_series = [
        (scope, {site: _load_clipping_series(runs) for site, runs in site_runs.items()})
        for scope, site_runs in grouped_site_runs
    ]
    grouped_site_series = [(scope, site_series) for scope, site_series in grouped_site_series if site_series]
    if not grouped_site_series:
        raise ValueError("No ReLU site-scope clipping frontier runs were found.")

    _plot_relu_site_scope_clipping_frontiers(grouped_site_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_clipping_frontiers(grouped_site_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_activation_density_comparison(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        site: read_json(Path(run_dir) / "activation_histograms.json")
        for site, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_relu_site_scope_activation_density_comparison(payloads, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_activation_density_comparison(payloads, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_pressure_weight_diagnostic(
    *,
    grouped_runs: list[tuple[str, list[tuple[str, str | Path]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_series = [
        (scope, _load_status_update_pressure_weight_groups(runs, group_specs=STATUS_UPDATE_PRESSURE_WEIGHT_GROUPS))
        for scope, runs in grouped_runs
    ]
    grouped_series = [(scope, series) for scope, series in grouped_series if series]
    if not grouped_series:
        raise ValueError("No ReLU site-scope pressure-weight runs were found.")

    _plot_relu_site_scope_pressure_weight_diagnostic(grouped_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_pressure_weight_diagnostic(grouped_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_near_zero_heatmaps(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        site: read_json(Path(run_dir) / "activation_histograms.json")
        for site, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_relu_site_scope_near_zero_heatmaps(payloads, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_near_zero_heatmaps(payloads, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_learning_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    training_specs: tuple[tuple[str, str], ...] = REPORT04_TRAINING_RUNS,
) -> list[Path]:
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No report-04 learning-diagnostic runs were found.")
    return export_figure(
        lambda: _plot_report04_learning_diagnostics(series, training_specs),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_activation_heatmaps(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
    layout: GridLayout | None = None,
) -> list[Path]:
    payloads = {
        key: read_json(Path(run_dir) / "activation_histograms.json")
        for key, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    return export_figure(
        lambda: _plot_report04_activation_heatmaps(payloads, layout=layout),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_activation_densities(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    payloads = {
        key: read_json(Path(run_dir) / "activation_histograms.json")
        for key, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    return export_figure(
        lambda: _plot_report04_activation_densities(payloads),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_site_clipping_frontiers(
    *,
    site_runs: dict[str, list[tuple[str, str | Path]]],
    output: str | Path,
    save_png: bool = False,
    layout: GridLayout | None = None,
) -> list[Path]:
    site_series = {site: _load_clipping_series(runs) for site, runs in site_runs.items()}
    return export_figure(
        lambda: _plot_report04_site_clipping_frontiers(site_series, layout=layout),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_joint_compute_frontier(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No report-04 joint clipping runs were found.")
    return export_figure(
        lambda: _plot_report04_joint_compute_frontier(series),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_parameter_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    series = _load_report04_parameter_series(runs)
    if not series:
        raise ValueError("No report-04 parameter checkpoints were found.")
    return export_figure(
        lambda: _plot_report04_parameter_diagnostics(series),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_activation_weight_densities(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    payloads = {
        key: read_json(Path(run_dir) / "activation_histograms.json")
        for key, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    series = _load_report04_parameter_series(runs)
    if not series:
        raise ValueError("No report-04 parameter checkpoints were found.")
    return export_figure(
        lambda: _plot_report04_activation_weight_densities(payloads, series),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_layernorm_parameters(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    series = _load_report04_parameter_series(runs)
    if not series:
        raise ValueError("No report-04 parameter checkpoints were found.")
    return export_figure(
        lambda: _plot_report04_layernorm_parameters(series),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_three_relu_architecture(
    *,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    return export_figure(
        _plot_report04_three_relu_architecture,
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report04_pythia_family_compute_ceiling(
    *,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    return export_figure(
        _plot_report04_pythia_family_compute_ceiling,
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_report05_architecture_schematic(
    *,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    """Export the four-case Pythia-14M ReLU architecture ladder."""

    return export_figure(
        _plot_report05_architecture_schematic,
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=_REPORT05_STANDARD_PROFILE,
    )


def generate_report05_learning_curves(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    series = _load_event_series(runs)
    if len(series) != len(REPORT05_TRAINING_RUNS):
        raise ValueError("Report 05 learning curves require all 13 pinned training runs.")
    return export_figure(
        lambda: _plot_report05_validation_learning_curves(series),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=_REPORT05_STANDARD_PROFILE,
    )


def _generate_report05_propagation(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool,
    renderer: Any,
) -> list[Path]:
    payload = read_json(Path(run_dir) / "activation_propagation.json")
    return export_figure(
        lambda: renderer(payload),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT05_HEATMAP_PROFILE,
    )


def generate_report05_one_relu_propagation(
    *, run_dir: str | Path, output: str | Path, save_png: bool = False
) -> list[Path]:
    return _generate_report05_propagation(
        run_dir=run_dir,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_one_relu_propagation_heatmaps,
    )


def generate_report05_three_relu_propagation(
    *, run_dir: str | Path, output: str | Path, save_png: bool = False
) -> list[Path]:
    return _generate_report05_propagation(
        run_dir=run_dir,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_three_relu_propagation_heatmaps,
    )


def generate_report05_six_relu_pre_propagation(
    *, run_dir: str | Path, output: str | Path, save_png: bool = False
) -> list[Path]:
    return _generate_report05_propagation(
        run_dir=run_dir,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_six_relu_pre_propagation_heatmaps,
    )


def generate_report05_six_relu_post_propagation(
    *, run_dir: str | Path, output: str | Path, save_png: bool = False
) -> list[Path]:
    return _generate_report05_propagation(
        run_dir=run_dir,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_six_relu_post_propagation_heatmaps,
    )


def _generate_report05_distributions(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool,
    renderer: Any,
) -> list[Path]:
    payloads = {
        role: read_json(Path(run_dir) / "activation_histograms.json")
        for role, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    if len(payloads) != 3:
        raise ValueError("Report 05 distributions require input, MLP-hidden, and gate histograms.")
    weight_series = _load_report04_parameter_series(runs)
    if len(weight_series) != len(REPORT05_TRAINING_RUNS):
        raise ValueError("Report 05 distributions require all 13 pinned final checkpoints.")
    return export_figure(
        lambda: renderer(payloads, weight_series),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=_REPORT05_STANDARD_PROFILE,
    )


def generate_report05_one_relu_distributions(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    return _generate_report05_distributions(
        histogram_runs=histogram_runs,
        runs=runs,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_one_relu_distributions,
    )


def generate_report05_three_relu_distributions(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    return _generate_report05_distributions(
        histogram_runs=histogram_runs,
        runs=runs,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_three_relu_distributions,
    )


def generate_report05_six_relu_pre_distributions(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    return _generate_report05_distributions(
        histogram_runs=histogram_runs,
        runs=runs,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_six_relu_pre_distributions,
    )


def generate_report05_six_relu_post_distributions(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    return _generate_report05_distributions(
        histogram_runs=histogram_runs,
        runs=runs,
        output=output,
        save_png=save_png,
        renderer=_plot_report05_six_relu_post_distributions,
    )


def generate_report05_site_clipping_frontiers(
    *,
    site_runs: dict[str, dict[str, list[tuple[str, str | Path]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    site_sweeps = {
        architecture_id: {
            site: _load_clipping_series(runs)
            for site, runs in architecture_runs.items()
        }
        for architecture_id, architecture_runs in site_runs.items()
    }
    reduced = _reduce_report05_site_clipping_frontiers(site_sweeps)
    return export_figure(
        lambda: _plot_report05_site_clipping_frontiers(reduced),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=_REPORT05_STANDARD_PROFILE,
    )


def generate_report05_model_compute_frontiers(
    *,
    joint_runs: dict[str, list[tuple[str, str | Path]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    joint_sweeps = {
        architecture_id: _load_clipping_series(runs)
        for architecture_id, runs in joint_runs.items()
    }
    reduced = _reduce_report05_model_matmul_frontiers(joint_sweeps)
    return export_figure(
        lambda: _plot_report05_model_matmul_frontiers(reduced),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=_REPORT05_STANDARD_PROFILE,
    )


def generate_post_layernorm_relu_propagation_heatmaps(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
    layout: GridLayout | None = None,
) -> list[Path]:
    run_path = Path(run_dir)
    payload = read_json(run_path / "activation_propagation.json")
    return export_figure(
        lambda: _plot_post_layernorm_relu_propagation_heatmaps(payload, layout=layout),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_post_layernorm_relu_zero_product_heatmaps(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
    layout: GridLayout | None = None,
) -> list[Path]:
    run_path = Path(run_dir)
    payload = read_json(run_path / "activation_propagation.json")
    return export_figure(
        lambda: _plot_post_layernorm_relu_zero_product_heatmaps(payload, layout=layout),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )


def generate_fixed_step_sweep_summary(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_fixed_step_sweep_summary(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_sweep_summary(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_learning_curves(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = _select_representative_fixed_step_rows(rows)
    _plot_fixed_step_learning_curves(selected_rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(selected_rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_series = _select_representative_clipping_series(series)
    _plot_fixed_step_clipping_frontiers(selected_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(selected_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_role_learning_curves(
    *,
    rows: list[dict[str, Any]],
    role_label: str,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{role_label} Fixed-step Learning Curves"
    subtitle = f"n={len(rows)} runs: AdamW plus all {role_label} settings"
    _plot_fixed_step_learning_curves(
        rows,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(
            rows,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_role_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    role_label: str,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{role_label} Post-hoc Clipping Frontiers"
    total_points = sum(len(item.get("rows", [])) for item in series)
    subtitle = (
        f"n={len(series)} runs, {total_points} sweep points: "
        f"AdamW plus all {role_label} settings; validation-loss axis zoomed"
    )
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_selected_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(len(item.get("rows", [])) for item in series)
    title = "Selected Post-hoc Clipping Frontiers"
    subtitle = f"n={len(series)} runs, {total_points} sweep points; validation-loss axis zoomed"
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_learning_curves(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    title: str,
    scope_note: str,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = sorted({int(row["tokens_seen"]) for row in rows if row.get("tokens_seen")})
    token_note = f"{tokens[0]:,} tokens/run" if len(tokens) == 1 else "fixed token budget"
    subtitle = f"n={len(rows)} runs: {scope_note}; {token_note}"
    _plot_fixed_step_learning_curves(
        rows,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
        legend_ncol=2,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(
            rows,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
            legend_ncol=2,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(len(item.get("rows", [])) for item in series)
    title = "High-pressure Post-hoc Clipping Frontiers"
    subtitle = f"n={len(series)} runs, {total_points} sweep points: AdamW plus configs 35-48"
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
        legend_ncol=3,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
            legend_ncol=3,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_weight_norms(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_fixed_step_high_pressure_weight_norms(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_high_pressure_weight_norms(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_activation_histogram_grid(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "activation_histograms.json")
    _plot_activation_histogram_grid(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_activation_histogram_grid(payload, png_path)
        outputs.append(png_path)
    return outputs


def generate_weight_histogram_grid(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "weight_histograms.json")
    _plot_weight_histogram_grid(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_weight_histogram_grid(payload, png_path)
        outputs.append(png_path)
    return outputs


def collect_numeric_metrics(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*/*/metrics.json")):
        metrics = read_json(metrics_path)
        manifest_path = metrics_path.parent / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}

        for metric_name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            rows.append(
                {
                    "experiment_name": manifest.get("experiment_name", metrics_path.parent.parent.name),
                    "config_id": manifest.get("config_id", metrics_path.parent.parent.name),
                    "run_id": manifest.get("run_id", metrics_path.parent.name),
                    "metric_name": metric_name,
                    "value": numeric_value,
                }
            )
    return rows


def _latest_labeled_runs(
    results_path: Path,
    experiments: list[tuple[str, str]],
    artifact_name: str,
    *,
    require_complete_run: bool = False,
) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for label, experiment_id in experiments:
        run = _latest_run_with(
            results_path / experiment_id,
            artifact_name,
            require_complete_run=require_complete_run,
        )
        if run is not None:
            runs.append((label, run))
    return runs


def _latest_grouped_labeled_runs(
    results_path: Path,
    grouped_experiments: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    artifact_name: str,
) -> list[tuple[str, list[tuple[str, Path]]]]:
    groups: list[tuple[str, list[tuple[str, Path]]]] = []
    for scope, experiments in grouped_experiments:
        groups.append((scope, _latest_labeled_runs(results_path, list(experiments), artifact_name)))
    return groups


def _latest_labeled_runs_filtered(
    results_path: Path,
    experiments: list[tuple[str, str]],
    artifact_name: str,
    *,
    predicate: Any,
) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for label, experiment_id in experiments:
        experiment_dir = results_path / experiment_id
        if not experiment_dir.exists():
            continue
        candidates = [
            path
            for path in sorted(experiment_dir.iterdir())
            if (path / artifact_name).exists() and predicate(path)
        ]
        if candidates:
            runs.append((label, candidates[-1]))
    return runs


def _latest_relu_site_scope_site_clipping_groups(
    results_path: Path,
) -> list[tuple[str, dict[str, list[tuple[str, Path]]]]]:
    return [
        (scope, _latest_status_update_site_clipping_runs(results_path, list(experiments)))
        for scope, experiments in RELU_SITE_SCOPE_TRAINING_GROUPS
    ]


def _latest_status_update_site_clipping_runs(
    results_path: Path,
    experiments: list[tuple[str, str]],
) -> dict[str, list[tuple[str, Path]]]:
    site_runs: dict[str, list[tuple[str, Path]]] = {}
    for site, _site_label in STATUS_UPDATE_CLIPPING_SITES:
        site_suffix = site.replace("_", "-")
        site_runs[site] = _latest_labeled_runs_filtered(
            results_path,
            [(label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}") for label, experiment_id in experiments],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
    return site_runs


def _has_status_update_site_clipping_runs(site_runs: dict[str, list[tuple[str, Path]]]) -> bool:
    return all(len(site_runs.get(site, [])) >= 2 for site, _site_label in STATUS_UPDATE_CLIPPING_SITES)


def _has_relu_site_scope_site_clipping_runs(
    grouped_site_runs: list[tuple[str, dict[str, list[tuple[str, Path]]]]],
) -> bool:
    return bool(grouped_site_runs) and all(
        _has_status_update_site_clipping_runs(site_runs) for _scope, site_runs in grouped_site_runs
    )


def _load_pressure_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        events_path = run_path / "events.jsonl"
        metrics_path = run_path / "metrics.json"
        if not events_path.exists() or not metrics_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if not train_events:
            continue
        series.append(
            {
                "label": label,
                "run_dir": run_path,
                "train_events": train_events,
                "validation_events": validation_events,
                "metrics": read_json(metrics_path),
            }
        )
    return series


def _load_clipping_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        frontier_path = run_path / "clipping_frontier.jsonl"
        if not frontier_path.exists():
            continue
        rows = [
            row
            for row in _read_jsonl(frontier_path)
            if row.get("achieved_sparsity") is not None
            and row.get("validation_loss") is not None
            and math.isfinite(float(row["achieved_sparsity"]))
            and math.isfinite(float(row["validation_loss"]))
        ]
        if rows:
            series.append({"label": label, "run_dir": run_path, "rows": rows})
    return series


def _load_event_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        events_path = run_path / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if train_events:
            series.append(
                {
                    "label": label,
                    "run_dir": run_path,
                    "train_events": train_events,
                    "validation_events": validation_events,
                }
            )
    return series


def _is_rms_clipping_run(run_dir: str | Path) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    return bool(manifest.get("rms_multipliers"))


def _is_all_site_clipping_run(run_dir: str | Path) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    expected_sites = {"mlp_hiddens", "attention_outputs", "residual_streams"}
    return manifest.get("clipping_sweep_suffix") == "all-sites" and set(manifest.get("clipping_sites", [])) == expected_sites


def _is_single_site_clipping_run(run_dir: str | Path, site: str) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    expected_suffix = "sites-" + site.replace("_", "-")
    return manifest.get("clipping_sweep_suffix") == expected_suffix and manifest.get("clipping_sites") == [site]


def _load_fixed_step_sweep_rows(results_path: Path) -> list[dict[str, Any]]:
    latest_by_config: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(results_path.glob("*/*/metrics.json")):
        run_path = metrics_path.parent
        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        sweep = manifest.get("sweep") or {}
        if sweep.get("name") != FIXED_STEP_SWEEP_NAME:
            continue
        metrics = read_json(metrics_path)
        if int(metrics.get("calibration/optimizer_steps", -1)) != int(sweep.get("fixed_steps", -1)):
            continue

        config_id = str(manifest.get("config_id", run_path.parent.name))
        pressure = manifest.get("activation_pressure") or {}
        row = {
            "config_index": _config_index(config_id),
            "config_id": config_id,
            "run_id": manifest.get("run_id", run_path.name),
            "run_sequence": int(manifest.get("run_sequence", 0)),
            "run_dir": run_path,
            "role": sweep.get("role", "unknown"),
            "label": _fixed_step_label(sweep.get("role", "unknown"), pressure),
            "short_label": _fixed_step_short_label(sweep.get("role", "unknown"), pressure),
            "pressure": pressure,
            "train_loss": metrics.get("calibration/train_loss_final"),
            "train_loss_mean": metrics.get("calibration/train_loss_mean"),
            "validation_loss": metrics.get("calibration/validation_loss_final"),
            "validation_loss_best": metrics.get("calibration/validation_loss_best"),
            "tokens_seen": metrics.get("calibration/tokens_seen"),
            "tokens_per_second": metrics.get("calibration/tokens_per_second"),
            "wall_seconds": metrics.get("calibration/wall_seconds_total"),
            "peak_gpu_memory_mb": metrics.get("calibration/peak_gpu_memory_mb"),
            "final_model_size_mb": metrics.get("checkpoint/final_size_mb"),
            "mlp_weight_norm_final": metrics.get("calibration/mlp_weight_norm_final"),
            "near_zero_k01": metrics.get("final/activation/near_zero_mass/k1em02"),
            "near_zero_k03": metrics.get("final/activation/near_zero_mass/k3em02"),
            "pressure_loss": metrics.get("final/pressure_loss"),
        }
        previous = latest_by_config.get(config_id)
        if previous is None or row["run_sequence"] >= previous["run_sequence"]:
            latest_by_config[config_id] = row
    return sorted(latest_by_config.values(), key=lambda row: row["config_index"])


def _load_fixed_step_clipping_series(results_path: Path, training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_by_config = {row["config_id"]: row for row in training_rows}
    latest_by_config: dict[str, dict[str, Any]] = {}
    for frontier_path in sorted(results_path.glob("*-clipping-sweep/*/clipping_frontier.jsonl")):
        run_path = frontier_path.parent
        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        source_sweep = manifest.get("source_sweep") or {}
        if source_sweep.get("name") != FIXED_STEP_SWEEP_NAME:
            continue
        source_run = manifest.get("source_run")
        if not source_run:
            continue
        source_config_id = Path(source_run).parent.name
        training_row = training_by_config.get(source_config_id)
        if training_row is None:
            continue
        rows = [
            row
            for row in _read_jsonl(frontier_path)
            if row.get("achieved_sparsity") is not None
            and row.get("validation_loss") is not None
            and math.isfinite(float(row["achieved_sparsity"]))
            and math.isfinite(float(row["validation_loss"]))
        ]
        if not rows:
            continue
        series = {
            "config_id": source_config_id,
            "run_sequence": int(manifest.get("run_sequence", 0)),
            "run_dir": run_path,
            "training": training_row,
            "label": training_row["label"],
            "short_label": training_row["short_label"],
            "role": training_row["role"],
            "rows": rows,
        }
        previous = latest_by_config.get(source_config_id)
        if previous is None or series["run_sequence"] >= previous["run_sequence"]:
            latest_by_config[source_config_id] = series
    return sorted(latest_by_config.values(), key=lambda item: item["training"]["config_index"])


def _plot_run_diagnostics(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 8.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.42, wspace=0.48)

    ax_loss = fig.add_subplot(grid[0, :])
    ax_grad = fig.add_subplot(grid[1, 0])
    ax_weight = fig.add_subplot(grid[1, 1])
    ax_stats = fig.add_subplot(grid[2, :])

    train_tokens = [event["tokens_seen"] for event in train_events]
    train_loss = [event["train_loss"] for event in train_events]
    ax_loss.plot(train_tokens, train_loss, marker="o", markersize=2.5, linewidth=1.3, label="train")
    if validation_events:
        val_tokens = [event["tokens_seen"] for event in validation_events]
        val_loss = [event["validation_loss"] for event in validation_events]
        ax_loss.plot(val_tokens, val_loss, marker="s", markersize=3.0, linewidth=1.3, label="validation")
    ax_loss.set_title("Loss vs Tokens")
    ax_loss.set_xlabel("Tokens seen")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)

    grad_events = [event for event in train_events if event.get("grad_norm") is not None]
    ax_grad.plot(
        [event["tokens_seen"] for event in grad_events],
        [event["grad_norm"] for event in grad_events],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    ax_grad.set_title("Gradient Norm")
    ax_grad.set_xlabel("Tokens seen")
    ax_grad.set_ylabel("L2 norm")

    weight_events = [event for event in train_events if event.get("weight_norm") is not None]
    ax_weight.plot(
        [event["tokens_seen"] for event in weight_events],
        [event["weight_norm"] for event in weight_events],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    ax_weight.set_title("Weight Norm")
    ax_weight.set_xlabel("Tokens seen")
    ax_weight.set_ylabel("L2 norm")
    ax_weight.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_weight.tick_params(axis="y", labelsize=7)

    tokens_per_step = metrics.get("calibration/tokens_per_step")
    logged_step_rates = [
        tokens_per_step / event["step_wall_seconds"]
        for event in train_events
        if tokens_per_step and event.get("step_wall_seconds")
    ]
    peak_logged_tokens_per_second = max(logged_step_rates) if logged_step_rates else None
    avg_tokens_per_second = metrics.get("calibration/tokens_per_second")
    if peak_logged_tokens_per_second is not None and avg_tokens_per_second is not None:
        peak_tokens_per_second = max(peak_logged_tokens_per_second, avg_tokens_per_second)
    else:
        peak_tokens_per_second = peak_logged_tokens_per_second or avg_tokens_per_second
    stats = [
        ("Peak tokens/s", peak_tokens_per_second, "tok/s"),
        ("Average tokens/s", avg_tokens_per_second, "tok/s"),
        ("Peak GPU allocated", metrics.get("calibration/peak_gpu_memory_mb"), "MB"),
        ("Peak GPU reserved", metrics.get("calibration/peak_gpu_reserved_mb"), "MB"),
        ("Train wall time", metrics.get("calibration/wall_seconds_train"), "sec"),
        ("Total wall time", metrics.get("calibration/wall_seconds_total"), "sec"),
        ("Final model size", metrics.get("checkpoint/final_size_mb"), "MB"),
    ]
    ax_stats.axis("off")
    ax_stats.set_title("Run Statistics")
    _draw_stats_panel(ax_stats, stats)

    fig.suptitle("Pythia-14M MiniPile Run Diagnostics", y=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _final_mlp_weight_norm_from_checkpoint(run_path: Path) -> float | None:
    checkpoint_path = run_path / "checkpoints" / "final" / "model.safetensors"
    if not checkpoint_path.exists():
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    total = 0.0
    tensor_count = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".mlp." not in key or not key.endswith(".weight"):
                continue
            tensor = handle.get_tensor(key).float()
            param_norm = tensor.norm(2).item()
            total += param_norm * param_norm
            tensor_count += 1
    if tensor_count == 0:
        return None
    return total**0.5


def _load_status_update_pressure_weight_groups(
    runs: list[tuple[str, str | Path]],
    *,
    group_specs: tuple[tuple[str, str, Any, tuple[float, float]], ...],
) -> list[dict[str, Any]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Pressure weight diagnostics require torch and safetensors.") from exc

    bins = 180
    series: list[dict[str, Any]] = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        checkpoint_path = run_path / "checkpoints" / "final" / "model.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing final checkpoint for {label}: {checkpoint_path}")

        groups: list[dict[str, Any]] = []
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for group_id, group_label, pattern, (range_min, range_max) in group_specs:
                values = []
                tensor_names = []
                for key in keys:
                    if pattern.match(key) is None:
                        continue
                    values.append(handle.get_tensor(key).detach().float().reshape(-1))
                    tensor_names.append(key)
                if not values:
                    raise ValueError(f"No tensors matched pressure weight group {group_id!r} for {label}.")

                flat = torch.cat(values)
                finite_values = flat[torch.isfinite(flat)]
                counts = torch.histc(finite_values, bins=bins, min=float(range_min), max=float(range_max)).cpu().double()
                total = int(flat.numel())
                width = (float(range_max) - float(range_min)) / bins
                densities = [float(count) / total / width if total and width > 0.0 else 0.0 for count in counts.tolist()]
                centers = [
                    float(range_min) + (index + 0.5) * width
                    for index in range(bins)
                ]
                underflow = int((finite_values < float(range_min)).sum().detach().cpu())
                overflow = int((finite_values > float(range_max)).sum().detach().cpu())
                groups.append(
                    {
                        "id": group_id,
                        "label": group_label,
                        "centers": centers,
                        "densities": densities,
                        "range": (float(range_min), float(range_max)),
                        "total": total,
                        "tensor_count": len(tensor_names),
                        "underflow": underflow,
                        "overflow": overflow,
                    }
                )
        series.append({"label": label, "run_dir": str(run_path), "groups": groups})

    return series




def _format_stat_value(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_stats_panel(ax: Any, stats: list[tuple[str, Any, str]]) -> None:
    visible_stats = [(label, value, unit) for label, value, unit in stats if value is not None]
    columns = 3
    rows = (len(visible_stats) + columns - 1) // columns
    for index, (label, value, unit) in enumerate(visible_stats):
        row = index // columns
        column = index % columns
        x = (column + 0.5) / columns
        y = 0.74 - row * (0.58 / max(rows - 1, 1))
        ax.text(x, y + 0.08, label, transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.text(
            x,
            y - 0.07,
            f"{_format_stat_value(float(value))} {unit}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )


def _plot_pressure_comparison(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(7.4, 8.4))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    token_limit = _short_run_token_limit(series)
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        label = item["label"]
        train_events = _events_up_to(item["train_events"], token_limit)
        validation_events = _events_up_to(item["validation_events"], token_limit)

        ax_train.plot(
            [event["tokens_seen"] for event in train_events],
            [event["train_loss"] for event in train_events],
            marker="o",
            markersize=_marker_size(label, 2.2),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

        if validation_events:
            ax_val.plot(
                [event["tokens_seen"] for event in validation_events],
                [event["validation_loss"] for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.5),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                [event["tokens_seen"] for event in near_zero_events],
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                [event["tokens_seen"] for event in pressure_events],
                [event["pressure_loss"] for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen")
    ax_train.set_ylabel("Task loss")

    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen")
    ax_val.set_ylabel("Loss")

    ax_near_zero.set_title("MLP Hidden Near-zero Mass")
    ax_near_zero.set_xlabel("Tokens seen")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    if not ax_near_zero.lines:
        ax_near_zero.text(0.5, 0.5, "No activation-pressure metrics found.", ha="center", va="center")

    ax_pressure.set_title("Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure loss metrics found.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 3),
            frameon=False,
        )
    fig.suptitle("Pythia-14M Short Pressure Pretraining Checks", y=0.995)
    fig.text(
        0.5,
        0.945,
        f"n={len(series)} runs; AdamW baseline is shown only through the shared short-run token horizon",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_clipping_comparison(series: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    all_losses: list[float] = []
    total_points = 0
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        loss = [float(row["validation_loss"]) for row in rows]
        total_points += len(rows)
        all_losses.extend(loss)
        ax.plot(
            sparsity,
            loss,
            marker="o",
            markersize=_marker_size(item["label"], 3.5),
            linewidth=_line_width(item["label"]),
            color=colors[item["label"]],
            label=item["label"],
        )

    ax.set_title("Post-hoc MLP Activation Clipping Frontiers")
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False, fontsize=8)

    if all_losses and min(all_losses) > 0.0:
        loss_span = max(all_losses) - min(all_losses)
        margin = max(loss_span * 0.15, 1e-4)
        ax.set_ylim(min(all_losses) - margin, max(all_losses) + margin)
        axis_note = "validation-loss axis zoomed"
    else:
        axis_note = "validation-loss axis starts at zero"

    ax.text(
        0.99,
        0.02,
        f"n={len(series)} runs, {total_points} sweep points; {axis_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_gradient_diagnostics(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Full-pass Selected Methods Gradient Diagnostics",
    subtitle: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8.4, 7.5))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_task = fig.add_subplot(grid[0, 0])
    ax_pressure = fig.add_subplot(grid[0, 1])
    ax_weighted = fig.add_subplot(grid[1, 0])
    ax_ratio = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    total_events = 0

    for item in series:
        label = item["label"]
        train_events = item["train_events"]
        total_events += len(train_events)
        task_norms = [
            float(event.get("pressure/task_gradient_norm", event.get("grad_norm")))
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        task_tokens = [
            float(event["tokens_seen"]) / 1_000_000.0
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        if task_norms:
            ax_task.plot(
                task_tokens,
                task_norms,
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        pressure_events = [
            event
            for event in train_events
            if event.get("pressure/pressure_gradient_norm") is not None
            and event.get("pressure/task_gradient_norm") is not None
        ]
        if not pressure_events:
            continue
        pressure_tokens = [float(event["tokens_seen"]) / 1_000_000.0 for event in pressure_events]
        pressure_norms = [float(event["pressure/pressure_gradient_norm"]) for event in pressure_events]
        weighted_pressure_norms = [
            float(event.get("pressure_weight", 1.0)) * float(event["pressure/pressure_gradient_norm"])
            for event in pressure_events
        ]
        weighted_ratios = [
            (
                float(event.get("pressure_weight", 1.0))
                * float(event["pressure/pressure_gradient_norm"])
                / (float(event["pressure/task_gradient_norm"]) + 1e-12)
            )
            for event in pressure_events
        ]

        ax_pressure.plot(
            pressure_tokens,
            pressure_norms,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )
        ax_weighted.plot(
            pressure_tokens,
            weighted_pressure_norms,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )
        ax_ratio.plot(
            pressure_tokens,
            weighted_ratios,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax_task.set_title("Task Gradient Norm")
    ax_task.set_xlabel("Tokens seen (millions)")
    ax_task.set_ylabel("L2 norm")
    ax_task.set_ylim(bottom=0.0)

    ax_pressure.set_title("Pressure Gradient Norm")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Raw pressure L2 norm")
    ax_pressure.set_ylim(bottom=0.0)
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_weighted.set_title("Weighted Pressure Gradient Norm")
    ax_weighted.set_xlabel("Tokens seen (millions)")
    ax_weighted.set_ylabel("weight * pressure grad norm")
    ax_weighted.set_ylim(bottom=0.0)
    if not ax_weighted.lines:
        ax_weighted.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_ratio.set_title("Weighted Pressure / Task Gradient")
    ax_ratio.set_xlabel("Tokens seen (millions)")
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(bottom=0.0)
    if not ax_ratio.lines:
        ax_ratio.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    handles, labels = ax_task.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.992)
    if subtitle is None:
        subtitle = (
            "n={n} runs; {events} train log events; "
            "pressure panels omit AdamW where no pressure is applied"
        )
    fig.text(
        0.5,
        0.948,
        subtitle.format(n=len(series), events=total_events),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.89, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_rms_clipping_frontiers(series: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    total_points = sum(len(item.get("rows", [])) for item in series)
    all_losses: list[float] = []
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        label = str(item["label"])
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(label, 3.4),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    _zoom_loss_axis(ax, all_losses)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass RMS-normalized Post-hoc Clipping Frontiers", y=0.975)
    fig.text(
        0.5,
        0.935,
        (
            f"n={len(series)} runs, {total_points} sweep points; "
            "threshold = multiplier * RMS(A) per captured tensor/pass; validation-loss axis zoomed"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.83, bottom=0.25, left=0.11, right=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_pressure_dominance(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_task = fig.add_subplot(grid[0, 1])
    ax_ratio = fig.add_subplot(grid[1, 0])
    ax_phase = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    pressure_method_count = 0

    for item in series:
        label = str(item["label"])
        color = colors[label]
        train_events = item["train_events"]
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        task_events = [
            event
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        pressure_events = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None
            and event.get("pressure/task_gradient_norm") is not None
            and event.get("pressure/pressure_gradient_norm") is not None
            and event.get("pressure_weight") is not None
        ]

        if near_zero_events:
            near_zero_tokens = _tokens_millions(near_zero_events)
            near_zero_values = [
                100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events
            ]
            ax_near_zero.plot(
                near_zero_tokens,
                near_zero_values,
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
            peak_index = max(range(len(near_zero_events)), key=lambda index: near_zero_values[index])
            ax_near_zero.scatter(
                [near_zero_tokens[peak_index]],
                [near_zero_values[peak_index]],
                marker="*",
                s=_scatter_size(label, 70.0),
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )

        if task_events:
            ax_task.plot(
                _tokens_millions(task_events),
                [
                    float(event.get("pressure/task_gradient_norm", event.get("grad_norm")))
                    for event in task_events
                ],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        if not pressure_events:
            continue
        pressure_method_count += 1
        ratio_tokens = _tokens_millions(pressure_events)
        ratio_values = [
            100.0
            * float(event["pressure_weight"])
            * float(event["pressure/pressure_gradient_norm"])
            / (float(event["pressure/task_gradient_norm"]) + 1e-12)
            for event in pressure_events
        ]
        near_zero_values = [
            100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in pressure_events
        ]
        ax_ratio.plot(
            ratio_tokens,
            ratio_values,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        ax_phase.plot(
            ratio_values,
            near_zero_values,
            marker="o",
            markersize=_marker_size(label, 1.7),
            linewidth=_line_width(label),
            color=color,
            alpha=0.92,
            label=label,
        )
        peak_index = max(range(len(near_zero_values)), key=lambda index: near_zero_values[index])
        ax_phase.scatter(
            [ratio_values[peak_index]],
            [near_zero_values[peak_index]],
            marker="*",
            s=_scatter_size(label, 78.0),
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        ax_phase.scatter(
            [ratio_values[-1]],
            [near_zero_values[-1]],
            marker="X",
            s=_scatter_size(label, 48.0),
            color=color,
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_task.set_title("Task Gradient Norm")
    ax_task.set_xlabel("Tokens seen (millions)")
    ax_task.set_ylabel("L2 norm")
    ax_task.set_ylim(bottom=0.0)

    ax_ratio.set_title("Weighted Pressure / Task Gradient")
    ax_ratio.set_xlabel("Tokens seen (millions)")
    ax_ratio.set_ylabel("ratio (%)")
    ax_ratio.set_ylim(bottom=0.0)
    if not ax_ratio.lines:
        ax_ratio.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_phase.set_title("Near-zero Mass vs Pressure Ratio")
    ax_phase.set_xlabel("weighted pressure/task gradient (%)")
    ax_phase.set_ylabel("|activation| <= 0.01 (%)")
    ax_phase.set_ylim(bottom=0.0)
    ax_phase.set_xlim(left=0.0)
    if not ax_phase.lines:
        ax_phase.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass Pressure Dominance Timing Diagnostic", y=0.992)
    fig.text(
        0.5,
        0.948,
        (
            f"n={len(series)} runs; {pressure_method_count} pressure methods; "
            "stars mark peak near-zero mass, X marks final pressure log event"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.89, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_high_pressure_learning_curves(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    total_events = 0

    for item in series:
        label = str(item["label"])
        color = colors[label]
        train_events = sorted(item["train_events"], key=lambda event: int(event.get("tokens_seen", 0)))
        validation_events = sorted(item["validation_events"], key=lambda event: int(event.get("tokens_seen", 0)))
        total_events += len(train_events)

        ax_train.plot(
            _tokens_millions(train_events),
            [float(event["train_loss"]) for event in train_events],
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        if validation_events:
            ax_val.plot(
                _tokens_millions(validation_events),
                [float(event["validation_loss"]) for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                _tokens_millions(pressure_events),
                [float(event["pressure_loss"]) for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen (millions)")
    ax_train.set_ylabel("Task loss")

    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen (millions)")
    ax_val.set_ylabel("Loss")

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_pressure.set_title("Auxiliary Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-loss series found.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.99)
    fig.text(
        0.5,
        0.935,
        (
            f"n={len(series)} completed runs; {total_events} train log events; "
            f"{subtitle}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_high_pressure_weight_norms(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_global_tokens = fig.add_subplot(grid[0, 1])
    ax_global_near_zero = fig.add_subplot(grid[1, 0])
    ax_mlp_near_zero = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    global_fit_x: list[float] = []
    global_fit_y: list[float] = []
    mlp_fit_x: list[float] = []
    mlp_fit_y: list[float] = []

    for item in series:
        label = str(item["label"])
        color = colors[label]
        marker = _method_marker(label)
        train_events = sorted(item["train_events"], key=lambda event: int(event.get("tokens_seen", 0)))
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        weight_events = [event for event in train_events if event.get("weight_norm") is not None]
        paired_global = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("weight_norm") is not None
        ]
        paired_mlp = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("mlp_weight_norm") is not None
        ]

        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if weight_events:
            ax_global_tokens.plot(
                _tokens_millions(weight_events),
                [float(event["weight_norm"]) for event in weight_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if paired_global:
            final_event = paired_global[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["weight_norm"])
            global_fit_x.append(final_x)
            global_fit_y.append(final_y)
            ax_global_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 38.0),
                color=color,
                alpha=0.92,
                linewidths=0.0,
                label=label,
                zorder=2,
            )
        if paired_mlp:
            final_event = paired_mlp[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["mlp_weight_norm"])
            mlp_fit_x.append(final_x)
            mlp_fit_y.append(final_y)
            ax_mlp_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 38.0),
                color=color,
                alpha=0.92,
                linewidths=0.0,
                label=label,
                zorder=2,
            )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_global_tokens.set_title("Global Weight Norm")
    ax_global_tokens.set_xlabel("Tokens seen (millions)")
    ax_global_tokens.set_ylabel("L2 norm")
    ax_global_tokens.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax_global_near_zero.set_title("Final Global Norm vs Near-zero Mass")
    ax_global_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_global_near_zero.set_ylabel("Global L2 norm")
    ax_global_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_global_near_zero.collections:
        ax_global_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_global_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_global_near_zero, global_fit_x, global_fit_y, loc="lower right")

    ax_mlp_near_zero.set_title("Final MLP Norm vs Near-zero Mass")
    ax_mlp_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_mlp_near_zero.set_ylabel("MLP weight L2 norm")
    ax_mlp_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_mlp_near_zero.collections:
        ax_mlp_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_mlp_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    else:
        ax_mlp_near_zero.text(0.5, 0.5, "No MLP weight norms found.", ha="center", va="center")
    _add_linear_fit_annotation(ax_mlp_near_zero, mlp_fit_x, mlp_fit_y, loc="upper right")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

    tokens = sorted(
        {
            int(event["tokens_seen"])
            for item in series
            for event in item["train_events"]
            if event.get("tokens_seen")
        }
    )
    token_note = f"up to {tokens[-1]:,} tokens/run" if tokens else "one MiniPile token-cache pass per run"
    fig.suptitle("Full-pass High-pressure Weight Norm Diagnostics", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(series)} completed runs; {token_note}; lower panels use final train log event",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_high_pressure_clipping_frontiers(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    total_points = sum(len(item.get("rows", [])) for item in series)
    all_losses: list[float] = []
    colors = _series_colors([str(item["label"]) for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        label = str(item["label"])
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(label, 3.2),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    _zoom_loss_axis(ax, all_losses)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.975)
    fig.text(
        0.5,
        0.93,
        (
            f"n={len(series)} runs, {total_points} sweep points; "
            f"{subtitle}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.83, bottom=0.27, left=0.11, right=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_status_update_coupling_density_comparison(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    methods: tuple[tuple[str, str], ...],
    site_specs: tuple[tuple[str, str, str, tuple[float, float]], ...],
    title: str,
    subtitle: str,
) -> None:
    missing = [site for site, _title, _layer, _xlim in site_specs if site not in payloads]
    if missing:
        raise ValueError(f"Missing histogram payloads for report density comparison: {missing}")

    labels = [label for _prefix, label in methods]
    colors = _series_colors(labels)
    site_density: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    site_y_limits: dict[str, tuple[float, float]] = {}

    for site, _site_title, layer_name, _xlim in site_specs:
        payload = payloads[site]
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
        per_method: dict[str, tuple[list[float], list[float]]] = {}
        positive_densities: list[float] = []
        max_density = 0.0
        for method_prefix, method_label in methods:
            method = _histogram_method(payload, method_prefix)
            if method is None:
                raise ValueError(f"Missing method {method_prefix!r} in histogram payload for {site}.")
            layer = _histogram_layer(method, layer_name)
            densities = _histogram_density(layer, edges)
            per_method[method_label] = (centers, densities)
            positive_densities.extend(value for value in densities if value > 0.0)
            max_density = max(max_density, max(densities, default=0.0))
        site_density[site] = per_method
        y_min = max(min(positive_densities) * 0.7, max_density * 1e-4, 1e-6) if positive_densities else 1e-6
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        site_y_limits[site] = (y_min, y_max)

    row_count = len(methods)
    col_count = len(site_specs)
    fig_width = max(10.8, 3.25 * col_count)
    fig_height = max(7.2, 1.65 * row_count + 1.5)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    for row_index, (_method_prefix, method_label) in enumerate(methods):
        for col_index, (site, site_title, _layer_name, xlim) in enumerate(site_specs):
            ax = axes[row_index][col_index]
            centers, densities = site_density[site][method_label]
            color = colors[method_label]
            y_min, y_max = site_y_limits[site]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
            ax.fill_between(centers, y_min, y_values, step="mid", color=color, alpha=0.24, linewidth=0)
            ax.step(centers, y_values, where="mid", color=color, linewidth=_line_width(method_label))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            ax.set_xlim(*xlim)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            if row_index == 0:
                ax.set_title(site_title, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(method_label, rotation=0, ha="right", va="center", labelpad=18, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    eval_tokens = int(next(iter(payloads.values())).get("validation_tokens") or 0)
    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.95,
        f"{subtitle}; layer 3 shown for each activation family; {eval_tokens:,} validation tokens",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Activation value", y=0.052, fontsize=9)
    fig.supylabel("Probability density (log scale)", x=0.008, fontsize=9)
    fig.text(
        0.5,
        0.018,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.91, bottom=0.095, hspace=0.34, wspace=0.14)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_status_update_pressure_weight_diagnostic(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
) -> None:
    if not series:
        raise ValueError("No pressure weight diagnostic series were found.")

    labels = [str(item["label"]) for item in series]
    colors = _series_colors(labels)
    group_labels = [group["label"] for group in series[0]["groups"]]
    group_y_limits: dict[str, tuple[float, float]] = {}
    for group_label in group_labels:
        positive_values = [
            float(value)
            for item in series
            for group in item["groups"]
            if group["label"] == group_label
            for value in group["densities"]
            if value > 0.0
        ]
        max_density = max(positive_values, default=1.0)
        y_min = max(min(positive_values) * 0.7, max_density * 1e-5, 1e-8) if positive_values else 1e-8
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        group_y_limits[group_label] = (y_min, y_max)

    row_count = len(series)
    col_count = len(group_labels)
    fig_width = max(10.8, 3.25 * col_count)
    fig_height = max(7.0, 1.6 * row_count + 1.5)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    for row_index, item in enumerate(series):
        method_label = str(item["label"])
        groups_by_label = {group["label"]: group for group in item["groups"]}
        for col_index, group_label in enumerate(group_labels):
            ax = axes[row_index][col_index]
            group = groups_by_label[group_label]
            centers = [float(value) for value in group["centers"]]
            densities = [float(value) for value in group["densities"]]
            y_min, y_max = group_y_limits[group_label]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
            color = colors[method_label]
            ax.fill_between(centers, y_min, y_values, step="mid", color=color, alpha=0.24, linewidth=0)
            ax.step(centers, y_values, where="mid", color=color, linewidth=_line_width(method_label))
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(*group["range"])
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            if row_index == 0:
                ax.set_title(group_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(method_label, rotation=0, ha="right", va="center", labelpad=18, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.947,
        subtitle,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Weight value", y=0.052, fontsize=9)
    fig.supylabel("Probability density (log scale)", x=0.008, fontsize=9)
    fig.text(
        0.5,
        0.018,
        footnote,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.905, bottom=0.095, hspace=0.32, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_status_update_site_clipping_frontiers(
    site_series: dict[str, list[dict[str, Any]]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=False)
    labels = [
        str(item["label"])
        for site, _site_label in STATUS_UPDATE_CLIPPING_SITES
        for item in site_series.get(site, [])
    ]
    if not labels:
        raise ValueError("No site-specific clipping series were found for the status update figure.")
    colors = _series_colors(labels)
    legend_handles: dict[str, Any] = {}
    total_points = 0

    for ax, (site, site_label) in zip(axes, STATUS_UPDATE_CLIPPING_SITES, strict=True):
        series = site_series.get(site, [])
        all_losses: list[float] = []
        for item in series:
            rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
            sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
            losses = [float(row["validation_loss"]) for row in rows]
            all_losses.extend(losses)
            total_points += len(rows)
            label = str(item["label"])
            (line,) = ax.plot(
                sparsity,
                losses,
                marker=_method_marker(label),
                markersize=_marker_size(label, 3.0),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )
            legend_handles.setdefault(label, line)

        ax.set_title(site_label, fontsize=10)
        ax.set_xlabel("Exact zeros after clipping (%)")
        ax.grid(True, alpha=0.25)
        if all_losses:
            _zoom_loss_axis(ax, all_losses)
        ax.tick_params(axis="both", labelsize=8)
    axes[0].set_ylabel("Validation loss")

    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.925,
        f"{subtitle}; {total_points} sweep points across the plotted panels; validation-loss axes zoomed",
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.81, bottom=0.28, wspace=0.26)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_learning_curves(
    grouped_series: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    row_count = len(grouped_series)
    fig, axes = plt.subplots(row_count, 2, figsize=(11.0, max(6.8, 2.1 * row_count + 1.4)), sharex="col")
    if row_count == 1:
        axes = [axes]

    labels = [
        str(item["label"])
        for _scope, series in grouped_series
        for item in series
    ]
    colors = _series_colors(labels)
    legend_handles: dict[str, Any] = {}
    token_max = 0

    for row_index, (scope, series) in enumerate(grouped_series):
        ax_loss = axes[row_index][0]
        ax_near_zero = axes[row_index][1]
        loss_values: list[float] = []

        for item in series:
            label = str(item["label"])
            color = colors[label]
            validation_events = sorted(
                item["validation_events"],
                key=lambda event: int(event.get("tokens_seen", 0)),
            )
            near_zero_events = sorted(
                [
                    event
                    for event in item["train_events"]
                    if event.get("activation/near_zero_mass/k1em02") is not None
                ],
                key=lambda event: int(event.get("tokens_seen", 0)),
            )
            if validation_events:
                token_max = max(token_max, max(int(event["tokens_seen"]) for event in validation_events))
                loss_values.extend(float(event["validation_loss"]) for event in validation_events)
                (line,) = ax_loss.plot(
                    _tokens_millions(validation_events),
                    [float(event["validation_loss"]) for event in validation_events],
                    marker=_method_marker(label),
                    markersize=_marker_size(label, 2.8),
                    linewidth=_line_width(label),
                    color=color,
                    label=label,
                )
                legend_handles.setdefault(label, line)
            if near_zero_events:
                token_max = max(token_max, max(int(event["tokens_seen"]) for event in near_zero_events))
                ax_near_zero.plot(
                    _tokens_millions(near_zero_events),
                    [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                    marker=_method_marker(label),
                    markersize=_marker_size(label, 2.4),
                    linewidth=_line_width(label),
                    color=color,
                    label=label,
                )

        if row_index == 0:
            ax_loss.set_title("Validation loss", fontsize=10)
            ax_near_zero.set_title("Logged near-zero mass", fontsize=10)
        ax_loss.set_ylabel(f"{scope}\nloss", fontsize=8)
        ax_near_zero.set_ylabel("|a| <= 0.01 (%)", fontsize=8)
        ax_near_zero.set_ylim(0.0, 100.0)
        if loss_values:
            _zoom_loss_axis(ax_loss, loss_values)
        if row_index == row_count - 1:
            ax_loss.set_xlabel("Tokens seen (millions)")
            ax_near_zero.set_xlabel("Tokens seen (millions)")
        ax_loss.tick_params(axis="both", labelsize=8)
        ax_near_zero.tick_params(axis="both", labelsize=8)

    fig.suptitle("ReLU Site-scope Learning Curves", y=0.985)
    token_note = f"up to {token_max:,} tokens/run" if token_max else "one MiniPile token-cache pass per run"
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; AdamW baseline repeated in each row; "
            f"{token_note}; near-zero mass is the run's logged pressure-site aggregate"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.88, bottom=0.16, hspace=0.3, wspace=0.24)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_clipping_frontiers(
    grouped_site_series: list[tuple[str, dict[str, list[dict[str, Any]]]]],
    output_path: Path,
) -> None:
    row_count = len(grouped_site_series)
    col_count = len(STATUS_UPDATE_CLIPPING_SITES)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(11.4, max(7.1, 2.15 * row_count + 1.35)),
        sharex=True,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    labels = [
        str(item["label"])
        for _scope, site_series in grouped_site_series
        for site, _site_label in STATUS_UPDATE_CLIPPING_SITES
        for item in site_series.get(site, [])
    ]
    if not labels:
        raise ValueError("No ReLU site-scope clipping series were found.")
    colors = _series_colors(labels)
    legend_handles: dict[str, Any] = {}
    total_points = 0

    for row_index, (scope, site_series) in enumerate(grouped_site_series):
        for col_index, (site, site_label) in enumerate(STATUS_UPDATE_CLIPPING_SITES):
            ax = axes[row_index][col_index]
            all_losses: list[float] = []
            for item in site_series.get(site, []):
                rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
                sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
                losses = [float(row["validation_loss"]) for row in rows]
                all_losses.extend(losses)
                total_points += len(rows)
                label = str(item["label"])
                (line,) = ax.plot(
                    sparsity,
                    losses,
                    marker=_method_marker(label),
                    markersize=_marker_size(label, 2.6),
                    linewidth=_line_width(label),
                    color=colors[label],
                    label=label,
                )
                legend_handles.setdefault(label, line)
            if row_index == 0:
                ax.set_title(site_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\nvalidation loss", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == row_count - 1:
                ax.set_xlabel("Exact zeros after clipping (%)")
            if all_losses:
                _zoom_loss_axis(ax, all_losses)
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("ReLU Site-scope Post-hoc Clipping Frontiers", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; columns clip one activation family at a time; "
            f"{total_points} sweep points; validation-loss axes zoomed"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.88, bottom=0.17, hspace=0.28, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_activation_density_comparison(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    missing = [site for site, _title, _layer, _xlim in RELU_SITE_SCOPE_DENSITY_SITE_SPECS if site not in payloads]
    if missing:
        raise ValueError(f"Missing ReLU site-scope histogram payloads: {missing}")

    method_labels = [label for _scope, methods in RELU_SITE_SCOPE_HISTOGRAM_GROUPS for label, _prefix in methods]
    colors = _series_colors(method_labels)
    site_density: dict[tuple[str, str], dict[str, tuple[list[float], list[float]]]] = {}
    site_y_limits: dict[str, tuple[float, float]] = {}

    for site, _site_title, layer_name, _xlim in RELU_SITE_SCOPE_DENSITY_SITE_SPECS:
        payload = payloads[site]
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
        positive_densities: list[float] = []
        max_density = 0.0
        for scope, methods in RELU_SITE_SCOPE_HISTOGRAM_GROUPS:
            per_method: dict[str, tuple[list[float], list[float]]] = {}
            for method_label, method_prefix in methods:
                method = _histogram_method(payload, method_prefix)
                if method is None:
                    raise ValueError(f"Missing method {method_prefix!r} in histogram payload for {site}.")
                layer = _histogram_layer(method, layer_name)
                densities = _histogram_density(layer, edges)
                per_method[method_label] = (centers, densities)
                positive_densities.extend(value for value in densities if value > 0.0)
                max_density = max(max_density, max(densities, default=0.0))
            site_density[(scope, site)] = per_method
        y_min = max(min(positive_densities) * 0.7, max_density * 1e-4, 1e-6) if positive_densities else 1e-6
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        site_y_limits[site] = (y_min, y_max)

    row_count = len(RELU_SITE_SCOPE_HISTOGRAM_GROUPS)
    col_count = len(RELU_SITE_SCOPE_DENSITY_SITE_SPECS)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(12.4, max(7.4, 2.1 * row_count + 1.35)),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    legend_handles: dict[str, Any] = {}
    for row_index, (scope, methods) in enumerate(RELU_SITE_SCOPE_HISTOGRAM_GROUPS):
        for col_index, (site, site_title, _layer_name, xlim) in enumerate(RELU_SITE_SCOPE_DENSITY_SITE_SPECS):
            ax = axes[row_index][col_index]
            y_min, y_max = site_y_limits[site]
            for method_label, _method_prefix in methods:
                centers, densities = site_density[(scope, site)][method_label]
                y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
                (line,) = ax.step(
                    centers,
                    y_values,
                    where="mid",
                    color=colors[method_label],
                    linewidth=_line_width(method_label),
                    label=method_label,
                )
                legend_handles.setdefault(method_label, line)
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(*xlim)
            if row_index == 0:
                ax.set_title(site_title, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\ndensity", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == row_count - 1:
                ax.set_xlabel("Activation value")
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    eval_tokens = int(next(iter(payloads.values())).get("validation_tokens") or 0)
    fig.suptitle("ReLU Site-scope Activation Density Comparison", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; columns show layer 3 for each activation family; "
            f"{eval_tokens:,} validation tokens; probability density is log-scaled"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.text(
        0.5,
        0.055,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.17, hspace=0.28, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_pressure_weight_diagnostic(
    grouped_series: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    group_labels = [group["label"] for group in grouped_series[0][1][0]["groups"]]
    labels = [str(item["label"]) for _scope, series in grouped_series for item in series]
    colors = _series_colors(labels)
    group_y_limits: dict[str, tuple[float, float]] = {}
    for group_label in group_labels:
        positive_values = [
            float(value)
            for _scope, series in grouped_series
            for item in series
            for group in item["groups"]
            if group["label"] == group_label
            for value in group["densities"]
            if value > 0.0
        ]
        max_density = max(positive_values, default=1.0)
        y_min = max(min(positive_values) * 0.7, max_density * 1e-5, 1e-8) if positive_values else 1e-8
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        group_y_limits[group_label] = (y_min, y_max)

    row_count = len(grouped_series)
    col_count = len(group_labels)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(11.4, max(7.1, 2.1 * row_count + 1.35)),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    legend_handles: dict[str, Any] = {}
    for row_index, (scope, series) in enumerate(grouped_series):
        for col_index, group_label in enumerate(group_labels):
            ax = axes[row_index][col_index]
            y_min, y_max = group_y_limits[group_label]
            for item in series:
                method_label = str(item["label"])
                groups_by_label = {group["label"]: group for group in item["groups"]}
                group = groups_by_label[group_label]
                centers = [float(value) for value in group["centers"]]
                densities = [float(value) for value in group["densities"]]
                y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
                (line,) = ax.step(
                    centers,
                    y_values,
                    where="mid",
                    color=colors[method_label],
                    linewidth=_line_width(method_label),
                    label=method_label,
                )
                legend_handles.setdefault(method_label, line)
                ax.set_xlim(*group["range"])
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            if row_index == 0:
                ax.set_title(group_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\ndensity", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == row_count - 1:
                ax.set_xlabel("Weight value")
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("ReLU Site-scope Pressure Weight Diagnostic", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; columns aggregate final-checkpoint weights across all six layers; "
            "probability density is log-scaled"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.text(
        0.5,
        0.055,
        (
            "Shaded band marks |weight| <= 0.01. "
            "Residual-stream column uses MLP and attention output-projection weights."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.17, hspace=0.28, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_near_zero_heatmaps(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    missing = [site for site, _site_label in RELU_SITE_SCOPE_NEAR_ZERO_SITES if site not in payloads]
    if missing:
        raise ValueError(f"Missing ReLU near-zero histogram payloads: {missing}")

    row_labels = [label for label, _prefix in RELU_SITE_SCOPE_REPORT_METHODS]
    matrices: dict[str, list[list[float]]] = {}
    for site, _site_label in RELU_SITE_SCOPE_NEAR_ZERO_SITES:
        payload = payloads[site]
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        site_matrix: list[list[float]] = []
        for _method_label, method_prefix in RELU_SITE_SCOPE_REPORT_METHODS:
            method = _histogram_method(payload, method_prefix)
            if method is None:
                raise ValueError(f"Missing method {method_prefix!r} in histogram payload for {site}.")
            site_matrix.append(
                [
                    100.0
                    * _histogram_center_window_mass(
                        _histogram_layer(method, f"{site}.layer_{layer_index}"),
                        edges,
                        threshold=0.01,
                    )
                    for layer_index in range(6)
                ]
            )
        matrices[site] = site_matrix

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.2), sharex=True)
    image = None
    for ax, (site, site_label) in zip(axes, RELU_SITE_SCOPE_NEAR_ZERO_SITES, strict=True):
        matrix = matrices[site]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
        ax.set_title(site_label, fontsize=10)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=6)
        ax.set_xticks(range(6), [f"L{index}" for index in range(6)], fontsize=8)
        for boundary in (0.5, 4.5, 8.5):
            ax.axhline(boundary, color="white", linewidth=0.9, alpha=0.8)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if value >= 55.0 else "black",
                )
    axes[-1].set_xlabel("Transformer layer")

    fig.suptitle("ReLU Site-scope Near-zero Mass by Layer Type", y=0.985)
    eval_tokens = int(next(iter(payloads.values())).get("validation_tokens") or 0)
    fig.text(
        0.5,
        0.952,
        (
            "Entries are histogram-estimated |activation| <= 0.01 percentages by layer; "
            f"{eval_tokens:,} validation tokens"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.18, right=0.86, top=0.91, bottom=0.08, hspace=0.33)
    if image is not None:
        colorbar_axis = fig.add_axes([0.885, 0.18, 0.018, 0.64])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Near-zero mass (%)", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_sweep_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    plottable = [
        row
        for row in rows
        if _finite(row.get("validation_loss"))
        and _finite(row.get("near_zero_k01"))
        and _finite(row.get("near_zero_k03"))
    ]
    if not plottable:
        raise ValueError("No fixed-step sweep rows with validation loss and activation metrics were found.")

    fig = plt.figure(figsize=(7.4, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)
    ax_k01 = fig.add_subplot(grid[0, 0])
    ax_k03 = fig.add_subplot(grid[1, 0])

    roles = _ordered_roles(plottable)
    colors = _series_colors([FIXED_STEP_ROLE_LABELS.get(role, role) for role in roles])
    baseline = next((row for row in plottable if row["role"] == "adamw"), None)

    for role in roles:
        role_rows = [row for row in plottable if row["role"] == role]
        label = FIXED_STEP_ROLE_LABELS.get(role, role)
        color = colors[label]
        marker = FIXED_STEP_ROLE_MARKERS.get(role, "o")
        ax_k01.scatter(
            [100.0 * float(row["near_zero_k01"]) for row in role_rows],
            [float(row["validation_loss"]) for row in role_rows],
            marker=marker,
            s=_scatter_size(label, 42),
            color=color,
            label=label,
            alpha=0.9,
        )
        ax_k03.scatter(
            [100.0 * float(row["near_zero_k03"]) for row in role_rows],
            [float(row["validation_loss"]) for row in role_rows],
            marker=marker,
            s=_scatter_size(label, 42),
            color=color,
            label=label,
            alpha=0.9,
        )

    for row in _fixed_step_annotation_rows(plottable):
        ax_k01.annotate(
            row["short_label"],
            (100.0 * float(row["near_zero_k01"]), float(row["validation_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
        ax_k03.annotate(
            row["short_label"],
            (100.0 * float(row["near_zero_k03"]), float(row["validation_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    for ax, threshold in [(ax_k01, 0.01), (ax_k03, 0.03)]:
        ax.set_xlabel(f"MLP hidden mass with |activation| <= {threshold:g} (%)")
        ax.set_ylabel("Final validation loss")
        if baseline is not None:
            ax.axhline(
                float(baseline["validation_loss"]),
                color=ADAMW_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
            )
        _zoom_loss_axis(ax, [float(row["validation_loss"]) for row in plottable])

    ax_k01.set_title("Fixed-step Tradeoff at Threshold 0.01")
    ax_k03.set_title("Fixed-step Tradeoff at Threshold 0.03")
    handles, labels = ax_k01.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    tokens = sorted({int(row["tokens_seen"]) for row in plottable if row.get("tokens_seen")})
    token_note = f"{tokens[0]:,} tokens/run" if len(tokens) == 1 else "fixed token budget"
    fig.suptitle("Pythia-14M MiniPile Fixed-step Pressure Screen", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(plottable)} runs; {token_note}; validation-loss axes zoomed",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.17)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_learning_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Representative Fixed-step Learning Curves",
    subtitle: str | None = None,
    use_short_labels: bool = False,
    legend_ncol: int = 2,
) -> None:
    series = []
    for row in rows:
        events_path = Path(row["run_dir"]) / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if train_events:
            series.append({**row, "train_events": train_events, "validation_events": validation_events})
    if not series:
        raise ValueError("No fixed-step sweep learning curves were found.")

    fig = plt.figure(figsize=(7.6, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    labels_for_colors = [_plot_label(item, use_short_labels=use_short_labels) for item in series]
    colors = _series_colors(labels_for_colors)

    for item in series:
        label = _plot_label(item, use_short_labels=use_short_labels)
        color = colors[label]
        train_events = item["train_events"]
        validation_events = item["validation_events"]
        ax_train.plot(
            _tokens_millions(train_events),
            [event["train_loss"] for event in train_events],
            marker="o",
            markersize=_marker_size(label, 2.0),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        if validation_events:
            ax_val.plot(
                _tokens_millions(validation_events),
                [event["validation_loss"] for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.4),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                _tokens_millions(pressure_events),
                [event["pressure_loss"] for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen (millions)")
    ax_train.set_ylabel("Task loss")
    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen (millions)")
    ax_val.set_ylabel("Loss")
    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_pressure.set_title("Auxiliary Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-loss series selected.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_ncol,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.99)
    subtitle = subtitle or f"n={len(series)} selected runs: AdamW plus best validation-loss run per pressure family"
    fig.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_high_pressure_weight_norms(rows: list[dict[str, Any]], output_path: Path) -> None:
    series = []
    for row in rows:
        events_path = Path(row["run_dir"]) / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        if not train_events:
            continue
        final_mlp_weight_norm = row.get("mlp_weight_norm_final")
        if final_mlp_weight_norm is None:
            final_mlp_weight_norm = _final_mlp_weight_norm_from_checkpoint(Path(row["run_dir"]))
        series.append({**row, "train_events": train_events, "final_mlp_weight_norm": final_mlp_weight_norm})
    if not series:
        raise ValueError("No high-pressure OR/L1 weight-norm series were found.")

    fig = plt.figure(figsize=(8.2, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_global_tokens = fig.add_subplot(grid[0, 1])
    ax_global_near_zero = fig.add_subplot(grid[1, 0])
    ax_mlp_near_zero = fig.add_subplot(grid[1, 1])

    labels_for_colors = [_plot_label(item, use_short_labels=True) for item in series]
    colors = _series_colors(labels_for_colors)
    mlp_time_series_count = 0
    mlp_final_count = 0
    global_fit_x: list[float] = []
    global_fit_y: list[float] = []
    mlp_fit_x: list[float] = []
    mlp_fit_y: list[float] = []

    for item in series:
        label = _plot_label(item, use_short_labels=True)
        color = colors[label]
        marker = FIXED_STEP_ROLE_MARKERS.get(str(item.get("role")), "o")
        train_events = item["train_events"]
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        weight_events = [event for event in train_events if event.get("weight_norm") is not None]
        paired_global = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("weight_norm") is not None
        ]
        paired_mlp = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("mlp_weight_norm") is not None
        ]

        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if weight_events:
            ax_global_tokens.plot(
                _tokens_millions(weight_events),
                [float(event["weight_norm"]) for event in weight_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if paired_global:
            final_event = paired_global[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["weight_norm"])
            global_fit_x.append(final_x)
            global_fit_y.append(final_y)
            ax_global_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 28.0),
                color=color,
                alpha=0.9,
                linewidths=0.0,
                zorder=2,
            )
        if paired_mlp:
            mlp_time_series_count += 1
            xs = [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in paired_mlp]
            ys = [float(event["mlp_weight_norm"]) for event in paired_mlp]
            mlp_fit_x.extend(xs)
            mlp_fit_y.extend(ys)
            ax_mlp_near_zero.scatter(
                xs,
                ys,
                marker=marker,
                s=_scatter_size(label, 12.0),
                color=color,
                alpha=0.72,
                linewidths=0.0,
                zorder=2,
            )
        elif item.get("final_mlp_weight_norm") is not None:
            final_event = paired_global[-1] if paired_global else None
            if final_event is not None:
                final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
                final_y = float(item["final_mlp_weight_norm"])
                mlp_fit_x.append(final_x)
                mlp_fit_y.append(final_y)
                mlp_final_count += 1
                ax_mlp_near_zero.scatter(
                    [final_x],
                    [final_y],
                    marker=marker,
                    s=_scatter_size(label, 28.0),
                    color=color,
                    alpha=0.9,
                    linewidths=0.0,
                    zorder=2,
                )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")

    ax_global_tokens.set_title("Global Weight Norm")
    ax_global_tokens.set_xlabel("Tokens seen (millions)")
    ax_global_tokens.set_ylabel("L2 norm")
    ax_global_tokens.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax_global_near_zero.set_title("Final Global Norm vs Near-zero Mass")
    ax_global_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_global_near_zero.set_ylabel("Global L2 norm")
    ax_global_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_global_near_zero.collections:
        ax_global_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_global_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_global_near_zero, global_fit_x, global_fit_y, loc="lower right")

    mlp_title_prefix = "MLP Weight Norm"
    if mlp_time_series_count == 0 and mlp_final_count > 0:
        mlp_title_prefix = "Final MLP Weight Norm"
    if mlp_title_prefix == "Final MLP Weight Norm":
        ax_mlp_near_zero.set_title("Final MLP Norm vs Near-zero Mass")
    else:
        ax_mlp_near_zero.set_title(f"{mlp_title_prefix} vs Near-zero Mass")
    ax_mlp_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_mlp_near_zero.set_ylabel("MLP weight L2 norm")
    ax_mlp_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if not ax_mlp_near_zero.lines and not ax_mlp_near_zero.collections:
        ax_mlp_near_zero.text(0.5, 0.5, "No MLP weight norms found.", ha="center", va="center")
    elif mlp_time_series_count == 0 and mlp_final_count > 0:
        ax_mlp_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.\nFuture runs log MLP norm per step.",
            transform=ax_mlp_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_mlp_near_zero, mlp_fit_x, mlp_fit_y, loc="upper right")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=7,
        )

    tokens = sorted(
        {
            int(event["tokens_seen"])
            for item in series
            for event in item["train_events"]
            if event.get("tokens_seen")
        }
    )
    token_note = f"up to {tokens[-1]:,} tokens/run" if tokens else "fixed token budget"
    fig.suptitle("High-pressure OR and L1 Weight Norm Diagnostics", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(series)} runs: AdamW, OR configs 40-44, and L1N/OL1 configs 45-48; {token_note}",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_activation_histogram_grid(payload: dict[str, Any], output_path: Path) -> None:
    methods = payload.get("methods", [])
    edges = [float(value) for value in payload.get("bin_edges", [])]
    if not methods or len(edges) < 2:
        raise ValueError("Activation histogram payload has no plottable methods or bin edges.")

    layer_names = [layer["name"] for layer in methods[0].get("layers", [])]
    if not layer_names:
        raise ValueError("Activation histogram payload has no layer histograms.")
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    labels = [str(method["label"]) for method in methods]
    colors = _series_colors(labels)

    positive_values: list[float] = []
    probability_by_method_layer: list[list[list[float]]] = []
    overflow_notes: list[str] = []
    for method in methods:
        method_layers: list[list[float]] = []
        method_overflow = 0.0
        method_total = 0.0
        layers_by_name = {layer["name"]: layer for layer in method.get("layers", [])}
        for layer_name in layer_names:
            layer = layers_by_name[layer_name]
            total = float(layer.get("total") or sum(layer.get("counts", [])) or 1.0)
            counts = [float(value) for value in layer["counts"]]
            probabilities = [count / total for count in counts]
            positive_values.extend(value for value in probabilities if value > 0.0)
            method_layers.append(probabilities)
            method_overflow += float(layer.get("underflow", 0)) + float(layer.get("overflow", 0))
            method_total += total
        probability_by_method_layer.append(method_layers)
        overflow_fraction = method_overflow / method_total if method_total else 0.0
        if overflow_fraction > 0.001:
            overflow_notes.append(f"{method['label']}: {100.0 * overflow_fraction:.2f}% outside range")

    rows = len(methods)
    cols = len(layer_names)
    fig_width = max(12.0, 2.05 * cols)
    if rows <= 3:
        fig_height = max(9.0, 2.05 * rows + 2.6)
        title_y = 0.99
        subtitle_y = 0.958
        top_margin = 0.895
        bottom_margin = 0.13
        xlabel_y = 0.055
        footnote_y = 0.018
        hspace = 0.22
    else:
        fig_height = max(14.0, 1.34 * rows + 2.6)
        title_y = 0.997
        subtitle_y = 0.982
        top_margin = 0.955
        bottom_margin = 0.065
        xlabel_y = 0.035
        footnote_y = 0.02
        hspace = 0.18
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    y_min = max(min(positive_values) * 0.6, 1e-9) if positive_values else 1e-9
    y_max = max(positive_values) * 2.0 if positive_values else 1.0
    site_scope = _activation_histogram_scope(payload, layer_names)
    x_min = min(edges)
    x_max = _activation_histogram_x_max(site_scope, edges)
    for row_index, method in enumerate(methods):
        label = str(method["label"])
        color = colors[label]
        for col_index, layer_name in enumerate(layer_names):
            ax = axes[row_index][col_index]
            probabilities = probability_by_method_layer[row_index][col_index]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in probabilities]
            ax.fill_between(
                centers,
                y_min,
                y_values,
                step="mid",
                color=color,
                alpha=0.26,
                linewidth=0,
            )
            ax.step(centers, y_values, where="mid", color=color, linewidth=0.75, alpha=0.95)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.07, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.45, alpha=0.5)
            if row_index == 0:
                ax.set_title(_activation_histogram_layer_title(layer_name), fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{label}\nprobability", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    eval_tokens = int(payload.get("validation_tokens") or 0)
    eval_sequences = int(payload.get("validation_sequences") or 0)
    overflow_note = ""
    if overflow_notes:
        overflow_note = "; outside plotted range: " + ", ".join(overflow_notes[:3])
        if len(overflow_notes) > 3:
            overflow_note += ", ..."
    x_note = ""
    if x_max < max(edges):
        x_note = f"; x-axis shown to {x_max:g}"
    title = str(payload.get("plot_title") or _activation_histogram_default_title(site_scope))
    fig.suptitle(title, y=title_y, fontsize=14)
    fig.text(
        0.5,
        subtitle_y,
        (
            f"n={len(methods)} checkpoints; {eval_sequences:,} validation blocks; "
            f"{eval_tokens:,} validation tokens; y-axis is log probability per bin"
            f"{x_note}{overflow_note}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Activation value", y=xlabel_y, fontsize=10)
    fig.supylabel("Probability per bin (log scale)", x=0.006, fontsize=10)
    fig.text(
        0.5,
        footnote_y,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=top_margin,
        bottom=bottom_margin,
        hspace=hspace,
        wspace=0.08,
    )
    fig.savefig(output_path)
    plt.close(fig)


def _activation_histogram_scope(payload: dict[str, Any], layer_names: list[str]) -> str:
    sites = payload.get("sites") or []
    if sites:
        return str(sites[0])
    first_layer = layer_names[0] if layer_names else ""
    if first_layer.startswith("residual_streams."):
        return "residual_streams"
    if first_layer.startswith("attention_outputs."):
        return "attention_outputs"
    return "mlp_hiddens"


def _activation_histogram_x_max(site_scope: str, edges: list[float]) -> float:
    if site_scope == "mlp_hiddens":
        return min(max(edges), 1.0)
    return max(edges)


def _activation_histogram_layer_title(layer_name: str) -> str:
    return (
        layer_name.replace("mlp_hiddens.layer_", "MLP ")
        .replace("residual_streams.layer_", "H ")
        .replace("attention_outputs.layer_", "Attn out ")
    )


def _activation_histogram_default_title(site_scope: str) -> str:
    if site_scope == "residual_streams":
        return "Selected Method Residual Stream Distributions"
    if site_scope == "attention_outputs":
        return "Selected Method Attention Output Distributions"
    return "Selected Method MLP Activation Distributions"


def _plot_weight_histogram_grid(payload: dict[str, Any], output_path: Path) -> None:
    methods = payload.get("methods", [])
    edges = [float(value) for value in payload.get("bin_edges", [])]
    if not methods or len(edges) < 2:
        raise ValueError("Weight histogram payload has no plottable methods or bin edges.")

    layer_names = [layer["name"] for layer in methods[0].get("layers", [])]
    if not layer_names:
        raise ValueError("Weight histogram payload has no layer histograms.")
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    labels = [str(method["label"]) for method in methods]
    colors = _series_colors(labels)

    positive_values: list[float] = []
    probability_by_method_layer: list[list[list[float]]] = []
    overflow_notes: list[str] = []
    for method in methods:
        method_layers: list[list[float]] = []
        method_overflow = 0.0
        method_total = 0.0
        layers_by_name = {layer["name"]: layer for layer in method.get("layers", [])}
        for layer_name in layer_names:
            layer = layers_by_name[layer_name]
            total = float(layer.get("total") or sum(layer.get("counts", [])) or 1.0)
            counts = [float(value) for value in layer["counts"]]
            probabilities = [count / total for count in counts]
            positive_values.extend(value for value in probabilities if value > 0.0)
            method_layers.append(probabilities)
            method_overflow += float(layer.get("underflow", 0)) + float(layer.get("overflow", 0))
            method_total += total
        probability_by_method_layer.append(method_layers)
        overflow_fraction = method_overflow / method_total if method_total else 0.0
        if overflow_fraction > 0.001:
            overflow_notes.append(f"{method['label']}: {100.0 * overflow_fraction:.2f}% outside range")

    rows = len(methods)
    cols = len(layer_names)
    fig_width = max(12.0, 2.05 * cols)
    fig_height = max(14.0, 1.34 * rows + 2.6)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    y_min = max(min(positive_values) * 0.6, 1e-9) if positive_values else 1e-9
    y_max = max(positive_values) * 2.0 if positive_values else 1.0
    x_min = min(edges)
    x_max = max(edges)
    for row_index, method in enumerate(methods):
        label = str(method["label"])
        color = colors[label]
        for col_index, layer_name in enumerate(layer_names):
            ax = axes[row_index][col_index]
            probabilities = probability_by_method_layer[row_index][col_index]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in probabilities]
            ax.fill_between(
                centers,
                y_min,
                y_values,
                step="mid",
                color=color,
                alpha=0.26,
                linewidth=0,
            )
            ax.step(centers, y_values, where="mid", color=color, linewidth=0.75, alpha=0.95)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.07, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.45, alpha=0.5)
            if row_index == 0:
                ax.set_title(_weight_histogram_layer_title(layer_name), fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{label}\nprobability", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    overflow_note = ""
    if overflow_notes:
        overflow_note = "; outside plotted range: " + ", ".join(overflow_notes[:3])
        if len(overflow_notes) > 3:
            overflow_note += ", ..."
    scope = str(payload.get("scope") or "mlp_weights")
    if scope == "attention_weights":
        scope_note = "attention QKV and output dense weights only"
        default_title = "Selected Method Attention Weight Distributions"
    else:
        scope_note = "MLP dense weights only"
        default_title = "Selected Method MLP Weight Distributions"
    title = str(payload.get("plot_title") or default_title)
    fig.suptitle(title, y=0.997, fontsize=14)
    fig.text(
        0.5,
        0.982,
        (
            f"n={len(methods)} checkpoints; {scope_note}; "
            f"biases excluded; y-axis is log probability per bin{overflow_note}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Weight value", y=0.035, fontsize=10)
    fig.supylabel("Probability per bin (log scale)", x=0.006, fontsize=10)
    fig.text(
        0.5,
        0.02,
        "Shaded band marks |weight| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.955, bottom=0.065, hspace=0.18, wspace=0.08)
    fig.savefig(output_path)
    plt.close(fig)


def _weight_histogram_layer_title(layer_name: str) -> str:
    return layer_name.replace("mlp_weights.layer_", "MLP ").replace("attention_weights.layer_", "Attn ")


def _plot_fixed_step_clipping_frontiers(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Post-hoc Clipping Frontiers After Fixed-step Pretraining",
    subtitle: str | None = None,
    use_short_labels: bool = False,
    legend_ncol: int | None = None,
) -> None:
    if not series:
        raise ValueError("No fixed-step clipping frontier series were found.")

    figsize = (7.0, 5.4) if legend_ncol is not None else (7.0, 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    labels_for_colors = [_plot_label(item, use_short_labels=use_short_labels) for item in series]
    colors = _series_colors(labels_for_colors)
    all_losses: list[float] = []
    total_points = 0
    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        total_points += len(rows)
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(_plot_label(item, use_short_labels=use_short_labels), 3.2),
            linewidth=_line_width(_plot_label(item, use_short_labels=use_short_labels)),
            color=colors[_plot_label(item, use_short_labels=use_short_labels)],
            label=_plot_label(item, use_short_labels=use_short_labels),
        )

    ax.set_title(title)
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    if legend_ncol is None:
        ax.legend(frameon=False, fontsize=8)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=legend_ncol,
                frameon=False,
                fontsize=7,
            )
    _zoom_loss_axis(ax, all_losses)
    ax.text(
        0.99,
        0.02,
        subtitle or f"n={len(series)} selected runs, {total_points} sweep points; validation-loss axis zoomed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    if legend_ncol is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(bottom=0.34)
    fig.savefig(output_path)
    plt.close(fig)


def _short_run_token_limit(series: list[dict[str, Any]]) -> int | None:
    pressure_maxima = [
        max((int(event["tokens_seen"]) for event in item["train_events"]), default=0)
        for item in series
        if item["label"] != "AdamW baseline"
    ]
    pressure_maxima = [value for value in pressure_maxima if value > 0]
    if pressure_maxima:
        return max(pressure_maxima)
    all_maxima = [max((int(event["tokens_seen"]) for event in item["train_events"]), default=0) for item in series]
    all_maxima = [value for value in all_maxima if value > 0]
    return max(all_maxima) if all_maxima else None


def _events_up_to(events: list[dict[str, Any]], token_limit: int | None) -> list[dict[str, Any]]:
    if token_limit is None:
        return events
    return [event for event in events if int(event.get("tokens_seen", 0)) <= token_limit]


def _series_colors(labels: list[str]) -> dict[str, str]:
    unique_labels = list(dict.fromkeys(labels))
    colors: dict[str, str] = {}
    color_index = 0
    for label in unique_labels:
        if _is_adamw_label(label):
            colors[label] = ADAMW_COLOR
            continue
        colors[label] = COLORBLIND_SAFE_COLORS[color_index % len(COLORBLIND_SAFE_COLORS)]
        color_index += 1
    return colors


def _is_adamw_label(label: str) -> bool:
    return label.lower().startswith("adamw")


def _line_width(label: str) -> float:
    return ADAMW_LINEWIDTH if _is_adamw_label(label) else DEFAULT_SERIES_LINEWIDTH


def _marker_size(label: str, default: float) -> float:
    return default * ADAMW_MARKER_SCALE if _is_adamw_label(label) else default


def _scatter_size(label: str, default: float) -> float:
    return default * ADAMW_MARKER_SCALE if _is_adamw_label(label) else default


def _method_marker(label: str) -> str:
    normalized = label.lower()
    if normalized.startswith("adamw"):
        return "D"
    if normalized.startswith("rn"):
        return "o"
    if normalized.startswith("or"):
        return "s"
    if normalized.startswith("l1n"):
        return "^"
    if normalized.startswith("ol1"):
        return "P"
    return "o"


def _ordered_roles(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["adamw", "ricker_naive", "orthogonal_ricker", "l1_naive", "orthogonal_l1"]
    present = {str(row["role"]) for row in rows}
    ordered = [role for role in preferred if role in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _fixed_step_label(role: str, pressure: dict[str, Any]) -> str:
    role_label = FIXED_STEP_ROLE_LABELS.get(role, role)
    if role == "adamw":
        return role_label
    weight = pressure.get("weight")
    if "ricker" in role:
        return (
            f"{role_label} "
            f"w={_compact_number(weight)}, c={_compact_number(pressure.get('ricker_c'))}, "
            f"s={_compact_number(pressure.get('ricker_sigma'))}"
        )
    if "l1" in role:
        return f"{role_label} w={_compact_number(weight)}"
    return role_label


def _fixed_step_short_label(role: str, pressure: dict[str, Any]) -> str:
    if role == "adamw":
        return "AdamW"
    prefix = {
        "ricker_naive": "RN",
        "orthogonal_ricker": "OR",
        "l1_naive": "L1N",
        "orthogonal_l1": "OL1",
    }.get(role, role)
    weight = _compact_number(pressure.get("weight"))
    if "ricker" in role:
        return f"{prefix} w{weight} c{_compact_number(pressure.get('ricker_c'))} s{_compact_number(pressure.get('ricker_sigma'))}"
    return f"{prefix} w{weight}"


def _compact_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):g}"


def _config_index(config_id: str) -> int:
    first = config_id.split("-", 1)[0]
    try:
        return int(first)
    except ValueError:
        return 9999


def _fixed_step_annotation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected[baseline["config_id"]] = baseline
    for role in _ordered_roles(rows):
        role_rows = [row for row in rows if row["role"] == role and _finite(row.get("validation_loss"))]
        if role == "adamw" or not role_rows:
            continue
        sparse = max(role_rows, key=lambda row: float(row.get("near_zero_k03") or 0.0))
        selected[sparse["config_id"]] = sparse
    return sorted(selected.values(), key=lambda row: row["config_index"])


def _select_representative_fixed_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    for role in ("ricker_naive", "orthogonal_ricker", "l1_naive", "orthogonal_l1"):
        role_rows = [row for row in rows if row["role"] == role and _finite(row.get("validation_loss"))]
        if role_rows:
            selected.append(min(role_rows, key=lambda row: float(row["validation_loss"])))
    return selected


def _select_fixed_step_rows_for_role(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    selected.extend(sorted((row for row in rows if row["role"] == role), key=lambda row: row["config_index"]))
    return selected


def _select_fixed_step_rows_by_config_indices(
    rows: list[dict[str, Any]],
    config_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_index = {int(row["config_index"]): row for row in rows}
    return [by_index[index] for index in config_indices if index in by_index]


def _select_representative_clipping_series(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_rows = [item["training"] for item in series]
    selected_configs = {row["config_id"] for row in _select_representative_fixed_step_rows(training_rows)}
    return [item for item in series if item["config_id"] in selected_configs]


def _select_fixed_step_clipping_for_role(series: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((item for item in series if item["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    selected.extend(sorted((item for item in series if item["role"] == role), key=lambda item: item["training"]["config_index"]))
    return selected


def _select_fixed_step_clipping_by_config_indices(
    series: list[dict[str, Any]],
    config_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_index = {int(item["training"]["config_index"]): item for item in series}
    return [by_index[index] for index in config_indices if index in by_index]


def _tokens_millions(events: list[dict[str, Any]]) -> list[float]:
    return [float(event["tokens_seen"]) / 1_000_000.0 for event in events]


def _plot_label(item: dict[str, Any], *, use_short_labels: bool) -> str:
    return str(item.get("short_label") if use_short_labels else item.get("label"))


def _add_linear_fit_annotation(ax: Any, x_values: list[float], y_values: list[float], *, loc: str) -> None:
    points = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=False)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(points) < 2:
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    if x_var <= 0.0:
        return
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / x_var
    intercept = y_mean - slope * x_mean
    residual_sum = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    total_sum = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0

    x_line = [min(xs), max(xs)]
    y_line = [slope * x + intercept for x in x_line]
    ax.plot(
        x_line,
        y_line,
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="_nolegend_",
        zorder=1,
    )

    positions = {
        "upper right": (0.98, 0.97, "right", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
    }
    x_pos, y_pos, ha, va = positions.get(loc, positions["upper right"])
    ax.text(
        x_pos,
        y_pos,
        f"slope={slope:.3g}/pp\nR2={r2:.2f}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def _zoom_loss_axis(ax: Any, values: list[float]) -> None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return
    low = min(finite_values)
    high = max(finite_values)
    span = high - low
    margin = max(span * 0.15, 1e-4)
    ax.set_ylim(low - margin, high + margin)


def _plot_clipping_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    finite_rows = [
        row
        for row in rows
        if row.get("achieved_sparsity") is not None
        and row.get("validation_loss") is not None
        and math.isfinite(float(row["achieved_sparsity"]))
        and math.isfinite(float(row["validation_loss"]))
    ]
    if not finite_rows:
        raise ValueError("No finite clipping frontier points to plot.")

    finite_rows = sorted(finite_rows, key=lambda row: float(row["achieved_sparsity"]))
    sparsity = [100.0 * float(row["achieved_sparsity"]) for row in finite_rows]
    loss = [float(row["validation_loss"]) for row in finite_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(sparsity, loss, marker="o", linewidth=1.5, markersize=4.5)
    ax.set_title("Post-hoc Activation Clipping Frontier")
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")

    for index, (point_x, point_y, row) in enumerate(zip(sparsity, loss, finite_rows, strict=True)):
        label = _clipping_label(row)
        ax.annotate(
            label,
            (point_x, point_y),
            textcoords="offset points",
            xytext=_clipping_label_offset(index, len(finite_rows)),
            fontsize=7,
        )

    loss_span = max(loss) - min(loss)
    if min(loss) > 0.0:
        margin = max(loss_span * 0.15, 1e-4)
        ax.set_ylim(min(loss) - margin, max(loss) + margin)
        axis_note = "validation-loss axis zoomed"
    else:
        axis_note = "validation-loss axis starts at zero"

    eval_tokens = sorted({int(row["validation_tokens"]) for row in finite_rows if row.get("validation_tokens")})
    token_note = f"; {eval_tokens[0]:,} validation tokens/point" if len(eval_tokens) == 1 else ""
    ax.text(
        0.99,
        0.02,
        f"n={len(finite_rows)} sweep points{token_note}; {axis_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _clipping_label(row: dict[str, Any]) -> str:
    if row.get("threshold") is not None:
        return f"t={float(row['threshold']):g}"
    if row.get("quantile") is not None:
        return f"q={float(row['quantile']):g}"
    if row.get("rms_multiplier") is not None:
        return f"r={float(row['rms_multiplier']):g}"
    return str(row.get("mode", "clip"))


def _clipping_label_offset(index: int, total: int) -> tuple[int, int]:
    if index == 0:
        return (6, -16)
    if index == total - 1:
        return (-36, 6)
    return (6, 8 + 5 * (index % 2))


def _plot_metric_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    metric_name = sorted({row["metric_name"] for row in rows})[0]
    selected = [row for row in rows if row["metric_name"] == metric_name]
    labels = [f"{row['config_id']}\n{row['run_id'][:8]}" for row in selected]
    values = [row["value"] for row in selected]

    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_title("Result Summary")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Run")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if values and min(values) >= 0:
        ax.set_ylim(bottom=0)
    ax.text(0.99, 0.98, f"n={len(selected)} runs", transform=ax.transAxes, ha="right", va="top")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_empty_placeholder(output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "No numeric results found.\nRun make smoke or make baseline first.",
        ha="center",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
