"""Report 05 cohorts, endpoint reductions, and first figure renderers.

This module owns the scientific presentation choices for the post-QKV ReLU
comparison.  The stable public wrappers, result selection, catalog entries,
and CLI wiring remain in :mod:`paper_exp.plots` and are deliberately not added
by this first slice.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_exp.plot_api import DOUBLE_COLUMN_WIDTH_INCHES
from paper_exp.plot_style import SeriesStyle, report04_method_style


REPORT05_STOCK_RUN = (
    "GELU AdamW",
    "50-pythia-14m-minipile-adamw-full-pass",
)
REPORT05_ONE_RELU_RUNS = (
    ("MLP-ReLU AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    (
        "MLP-ReLU OR",
        "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
    ),
    (
        "MLP-ReLU OL1",
        "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5",
    ),
)
REPORT05_THREE_RELU_RUNS = (
    (
        "Three-ReLU AdamW",
        "98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass",
    ),
    (
        "Three-ReLU OR",
        "103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
    ),
    (
        "Three-ReLU OL1",
        "99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5",
    ),
)
REPORT05_PRE_RUNS = (
    (
        "Six-ReLU PRE AdamW",
        "107-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-adamw-full-pass",
    ),
    (
        "Six-ReLU PRE OR",
        "108-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
    ),
    (
        "Six-ReLU PRE OL1",
        "109-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-l1-full-pass-w5",
    ),
)
REPORT05_POST_RUNS = (
    (
        "Six-ReLU POST AdamW",
        "110-pythia-14m-minipile-post-qkv-relu-qk-post-rope-adamw-full-pass",
    ),
    (
        "Six-ReLU POST OR",
        "111-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
    ),
    (
        "Six-ReLU POST OL1",
        "112-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-l1-full-pass-w5",
    ),
)

REPORT05_METHOD_ORDER = ("AdamW", "OR", "OL1")
REPORT05_ARCHITECTURE_FAMILIES = (
    ("one_relu", "One-ReLU (MLP hidden)", REPORT05_ONE_RELU_RUNS),
    ("three_relu", "Three-ReLU", REPORT05_THREE_RELU_RUNS),
    ("six_relu_pre", "Six-ReLU PRE", REPORT05_PRE_RUNS),
    ("six_relu_post", "Six-ReLU POST", REPORT05_POST_RUNS),
)
REPORT05_TRAINING_RUNS = (
    REPORT05_STOCK_RUN,
    *(
        run
        for _family_id, _family_label, family_runs in REPORT05_ARCHITECTURE_FAMILIES
        for run in family_runs
    ),
)
REPORT05_PINNED_RUN_IDS = {
    "50-pythia-14m-minipile-adamw-full-pass": "001-20260629-161507-ef4ddaed",
    "77-pythia-14m-minipile-relu-adamw-full-pass": "004-20260705-144930-dd498fa9",
    "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05": (
        "001-20260705-232549-d0737405"
    ),
    "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5": (
        "001-20260707-015527-fe1ee962"
    ),
    "98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass": (
        "001-20260710-135015-5802fe11"
    ),
    "103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05": (
        "001-20260711-132226-055ae84b"
    ),
    "99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5": (
        "001-20260710-171732-f48a0bcc"
    ),
    "107-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-adamw-full-pass": (
        "001-20260716-110737-3a2e785c"
    ),
    "108-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05": (
        "001-20260716-175845-bee05387"
    ),
    "109-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-l1-full-pass-w5": (
        "001-20260717-024030-35516363"
    ),
    "110-pythia-14m-minipile-post-qkv-relu-qk-post-rope-adamw-full-pass": (
        "001-20260716-143107-dd964d74"
    ),
    "111-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05": (
        "001-20260716-222739-974927ab"
    ),
    "112-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-l1-full-pass-w5": (
        "001-20260717-064252-89acf88a"
    ),
}
REPORT05_PROPAGATION_EXPERIMENT = (
    "114-pythia-14m-minipile-report05-relu-architecture-ladder-activation-propagation"
)
REPORT05_INPUT_HISTOGRAM_EXPERIMENT = (
    "115-pythia-14m-minipile-post-qkv-relu-input-histograms"
)
REPORT05_MLP_HISTOGRAM_EXPERIMENT = (
    "116-pythia-14m-minipile-post-qkv-relu-mlp-hidden-histograms"
)
REPORT05_GATE_HISTOGRAM_EXPERIMENT = (
    "117-pythia-14m-minipile-post-qkv-relu-gate-output-histograms"
)
REPORT05_HIDDEN_SIZE = 128
REPORT05_VOCAB_SIZE = 50_304
REPORT05_ZERO_SITE_STAGES = {
    "z_a": "attention_input_relu",
    "z_m": "mlp_input_relu",
    "z_h": "mlp_hidden_relu",
    "z_q_gate": "query_gate_output",
    "z_k_gate": "key_gate_output",
    "z_v_gate": "value_gate_output",
    "z_q": "query_qk_input",
    "z_k": "key_qk_input",
    "z_v": "value_pv_input",
}
REPORT05_ENDPOINT_TABLE_COLUMNS = (
    "Architecture",
    "Method",
    "Validation loss",
    "Delta vs stock",
)


_STOCK_STYLE = SeriesStyle("#4D4D4D", "D", "--", 1.25)
_METHOD_STYLES = {
    "AdamW": report04_method_style("Three-ReLU AdamW"),
    "OR": report04_method_style("Three-ReLU OR"),
    "OL1": report04_method_style("Three-ReLU OL1"),
}
_GATE_COLOR = "#CC79A7"
_GATE_FILL = "#FBE3F0"
_PROJECTION_COLOR = "#56B4E9"
_PROJECTION_FILL = "#DDF1FA"
_DENSE_FILL = "#EEEEEE"
_RESIDUAL_COLOR = "#333333"


def _report05_series_by_label(
    series: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the exact Report 05 cohort indexed by stable display label."""

    required_labels = [label for label, _experiment_id in REPORT05_TRAINING_RUNS]
    matches = {
        label: [item for item in series if str(item.get("label")) == label]
        for label in required_labels
    }
    missing = [label for label in required_labels if not matches[label]]
    duplicates = [label for label in required_labels if len(matches[label]) > 1]
    issues = []
    if missing:
        issues.append("missing " + ", ".join(missing))
    if duplicates:
        issues.append("duplicate " + ", ".join(duplicates))
    if issues:
        raise ValueError("Invalid Report 05 training cohort: " + "; ".join(issues) + ".")
    return {label: matches[label][0] for label in required_labels}


def _report05_validation_points(item: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Normalize and sort the finite validation points for one saved run."""

    points: list[dict[str, Any]] = []
    for event in item.get("validation_events", []):
        if not isinstance(event, dict):
            continue
        try:
            tokens_seen = float(event["tokens_seen"])
            validation_loss = float(event["validation_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(tokens_seen)
            or tokens_seen < 0.0
            or not math.isfinite(validation_loss)
        ):
            continue
        step_value = event.get("step")
        try:
            step = int(step_value) if step_value is not None else -1
        except (TypeError, ValueError):
            step = -1
        validation_tokens_value = event.get("validation_tokens")
        try:
            validation_tokens = (
                int(validation_tokens_value)
                if validation_tokens_value is not None
                else None
            )
        except (TypeError, ValueError):
            validation_tokens = None
        points.append(
            {
                "step": step,
                "tokens_seen": int(tokens_seen),
                "validation_loss": validation_loss,
                "validation_tokens": validation_tokens,
            }
        )
    points.sort(key=lambda point: (point["tokens_seen"], point["step"]))
    if not points:
        raise ValueError(
            f"No finite validation points for Report 05 series {item.get('label')!r}."
        )
    return tuple(points)


def _report05_endpoint_rows(
    series: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Extract ordered, numeric endpoint rows without reading or writing files."""

    by_label = _report05_series_by_label(series)
    stock_label, stock_experiment = REPORT05_STOCK_RUN
    stock_endpoint = _report05_validation_points(by_label[stock_label])[-1]
    stock_loss = float(stock_endpoint["validation_loss"])

    rows: list[dict[str, Any]] = [
        {
            "architecture_id": "stock",
            "architecture": "Stock (GELU)",
            "method": "AdamW",
            "label": stock_label,
            "experiment_id": stock_experiment,
            **stock_endpoint,
            "delta_vs_stock": 0.0,
        }
    ]
    for family_id, family_label, family_runs in REPORT05_ARCHITECTURE_FAMILIES:
        for method, (label, experiment_id) in zip(
            REPORT05_METHOD_ORDER,
            family_runs,
            strict=True,
        ):
            endpoint = _report05_validation_points(by_label[label])[-1]
            validation_loss = float(endpoint["validation_loss"])
            rows.append(
                {
                    "architecture_id": family_id,
                    "architecture": family_label,
                    "method": method,
                    "label": label,
                    "experiment_id": experiment_id,
                    **endpoint,
                    "delta_vs_stock": validation_loss - stock_loss,
                }
            )
    return tuple(rows)


def _report05_endpoint_table(
    series: list[dict[str, Any]],
) -> tuple[tuple[str, str, float, float], ...]:
    """Return the compact validation-loss table used by the first report slice."""

    return tuple(
        (
            str(row["architecture"]),
            str(row["method"]),
            float(row["validation_loss"]),
            float(row["delta_vs_stock"]),
        )
        for row in _report05_endpoint_rows(series)
    )


def _report05_propagation_rows(
    payload: dict[str, Any],
    *,
    validation_losses: dict[str, float] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Pool exact-zero sites and direct logical products for the fixed cohort.

    ``z_q``, ``z_k``, and ``z_v`` refer to the actual operands entering QK or
    PV. For PRE-RoPE gates this intentionally measures Q/K after RoPE.
    Unavailable gates remain ``None`` rather than being reported as zero.
    """

    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("Report 05 propagation payload has no methods list.")
    by_config: dict[str, dict[str, Any]] = {}
    for method in methods:
        if not isinstance(method, dict):
            continue
        config_id = str(method.get("config_id", ""))
        if config_id in by_config:
            raise ValueError(f"Duplicate Report 05 propagation method: {config_id}")
        by_config[config_id] = method

    required_ids = [experiment_id for _label, experiment_id in REPORT05_TRAINING_RUNS]
    missing = [config_id for config_id in required_ids if config_id not in by_config]
    if missing:
        raise ValueError(
            "Report 05 propagation payload is missing: " + ", ".join(missing)
        )

    validation_tokens = int(payload.get("validation_tokens", 0))
    if validation_tokens <= 0:
        raise ValueError("Report 05 propagation payload has no validation tokens.")

    rows: list[dict[str, Any]] = []
    family_by_config = {
        REPORT05_STOCK_RUN[1]: ("stock", "Stock (GELU)", "AdamW"),
        **{
            config_id: (family_id, family_label, method_name)
            for family_id, family_label, family_runs in REPORT05_ARCHITECTURE_FAMILIES
            for method_name, (_label, config_id) in zip(
                REPORT05_METHOD_ORDER, family_runs, strict=True
            )
        },
    }
    for config_id in required_ids:
        method = by_config[config_id]
        architecture_id, architecture, method_name = family_by_config[config_id]
        activation_rows = method.get("activations", [])
        matmul_rows = [
            row
            for row in method.get("matmuls", [])
            if isinstance(row, dict) and bool(row.get("available", True))
        ]
        block_zero_count = sum(int(row["zero_count"]) for row in matmul_rows)
        block_product_count = sum(int(row["total"]) for row in matmul_rows)
        if block_product_count <= 0:
            raise ValueError(f"No matmul products for Report 05 method {config_id}.")
        model_product_count = block_product_count + (
            validation_tokens * REPORT05_HIDDEN_SIZE * REPORT05_VOCAB_SIZE
        )
        row: dict[str, Any] = {
            "config_id": config_id,
            "config": int(config_id.split("-", 1)[0]),
            "label": str(method.get("label", config_id)),
            "architecture_id": architecture_id,
            "architecture": architecture,
            "method": method_name,
            "validation_loss": (
                float(validation_losses[config_id])
                if validation_losses is not None and config_id in validation_losses
                else None
            ),
            "r_block": block_zero_count / block_product_count,
            "r_model": block_zero_count / model_product_count,
            "block_zero_product_count": block_zero_count,
            "block_product_count": block_product_count,
            "model_product_count": model_product_count,
        }
        for alias, stage in REPORT05_ZERO_SITE_STAGES.items():
            selected = [
                item
                for item in activation_rows
                if isinstance(item, dict)
                and item.get("name") == stage
                and bool(item.get("available", True))
            ]
            total = sum(int(item["total"]) for item in selected)
            zero_count = sum(int(item["zero_count"]) for item in selected)
            row[alias] = zero_count / total if total else None
        rows.append(row)
    return tuple(rows)


def _plot_report05_architecture_schematic() -> Figure:
    """Render Figure 91: four Pythia-14M parallel-residual gate paths."""

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 8.6),
    )
    cases = (
        {
            "title": "(a) One-ReLU (MLP hidden)",
            "pressure": r"OR/OL1 targets: $h$",
            "post_ln_relu": False,
            "qkv_stage": "split Q,K,V\nRoPE(Q,K)",
            "qkv_gate": False,
        },
        {
            "title": "(b) Three-ReLU",
            "pressure": r"OR/OL1 targets: $a,m,h$",
            "post_ln_relu": True,
            "qkv_stage": "split Q,K,V\nRoPE(Q,K)",
            "qkv_gate": False,
        },
        {
            "title": "(c) Six-ReLU PRE",
            "pressure": r"OR/OL1 targets: $q,k,v$ only",
            "post_ln_relu": True,
            "qkv_stage": "split -> ReLU(Q,K,V)\nRoPE(Q,K)",
            "qkv_gate": True,
        },
        {
            "title": "(d) Six-ReLU POST",
            "pressure": r"OR/OL1 targets: $q,k,v$ only",
            "post_ln_relu": True,
            "qkv_stage": "split -> RoPE(Q,K)\nReLU(Q,K); ReLU(V)",
            "qkv_gate": True,
        },
    )
    for ax, case in zip(axes, cases, strict=True):
        _draw_report05_architecture_case(ax, **case)

    fig.suptitle(
        "Pythia-14M ReLU Architecture Ladder",
        x=0.5,
        y=0.982,
        fontsize=11.0,
    )
    fig.text(
        0.5,
        0.952,
        (
            r"Parallel residual in every case: $H_{l+1}=H_l+O_l+M_l$; "
            r"$d=128$, 4 heads, $d_h=32$, partial RoPE on 8 coordinates"
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )
    fig.text(
        0.5,
        0.060,
        (
            r"Gate aliases: $a=$ attention_inputs, $m=$ mlp_inputs, "
            r"$h=$ mlp_hiddens, $q/k/v=$ post-QKV gate outputs."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Magenta boxes create exact zeros; blue boxes are learned projections.\n"
            "Stock AdamW replaces the hidden ReLU with GELU and has no other gates."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.910,
        bottom=0.105,
        hspace=0.30,
    )
    return fig


def _draw_report05_architecture_case(
    ax: Axes,
    *,
    title: str,
    pressure: str,
    post_ln_relu: bool,
    qkv_stage: str,
    qkv_gate: bool,
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=9.0, weight="bold", pad=5)
    ax.text(0.99, 0.965, pressure, ha="right", va="top", fontsize=8.0)

    # Draw the residual path first so branch boxes and arrows remain dominant.
    _architecture_arrow(ax, (0.095, 0.50), (0.945, 0.50), width=1.15)
    _architecture_box(
        ax,
        0.01,
        0.415,
        0.085,
        0.17,
        r"$H_l$" + "\nresidual",
        edgecolor=_RESIDUAL_COLOR,
        weight="bold",
    )

    layer_norm_text = r"ReLU$\circ$LN$_a$" if post_ln_relu else r"LN$_a$"
    layer_norm_fill = _GATE_FILL if post_ln_relu else _DENSE_FILL
    layer_norm_edge = _GATE_COLOR if post_ln_relu else "#777777"
    _architecture_box(
        ax,
        0.13,
        0.665,
        0.13,
        0.16,
        layer_norm_text,
        facecolor=layer_norm_fill,
        edgecolor=layer_norm_edge,
        linewidth=1.6 if post_ln_relu else 1.0,
        weight="bold" if post_ln_relu else "normal",
    )
    _architecture_box(
        ax,
        0.29,
        0.64,
        0.14,
        0.21,
        "QKV\n128 -> 384",
        facecolor=_PROJECTION_FILL,
        edgecolor=_PROJECTION_COLOR,
        linewidth=1.5,
        weight="bold",
    )
    _architecture_box(
        ax,
        0.46,
        0.625,
        0.225,
        0.24,
        qkv_stage,
        facecolor=_GATE_FILL if qkv_gate else _DENSE_FILL,
        edgecolor=_GATE_COLOR if qkv_gate else "#777777",
        linewidth=1.6 if qkv_gate else 1.0,
        weight="bold" if qkv_gate else "normal",
    )
    _architecture_box(
        ax,
        0.715,
        0.64,
        0.12,
        0.21,
        "QK -> P\nPV",
        facecolor=_DENSE_FILL,
    )
    _architecture_box(
        ax,
        0.855,
        0.665,
        0.07,
        0.16,
        r"$W_O$" + "\n128 -> 128",
        facecolor=_PROJECTION_FILL,
        edgecolor=_PROJECTION_COLOR,
        linewidth=1.4,
    )

    mlp_norm_text = r"ReLU$\circ$LN$_m$" if post_ln_relu else r"LN$_m$"
    _architecture_box(
        ax,
        0.13,
        0.19,
        0.13,
        0.16,
        mlp_norm_text,
        facecolor=layer_norm_fill,
        edgecolor=layer_norm_edge,
        linewidth=1.6 if post_ln_relu else 1.0,
        weight="bold" if post_ln_relu else "normal",
    )
    _architecture_box(
        ax,
        0.31,
        0.165,
        0.15,
        0.21,
        r"$W_1$" + "\n128 -> 512",
        facecolor=_PROJECTION_FILL,
        edgecolor=_PROJECTION_COLOR,
        linewidth=1.5,
        weight="bold",
    )
    _architecture_box(
        ax,
        0.51,
        0.19,
        0.13,
        0.16,
        r"ReLU$_h$",
        facecolor=_GATE_FILL,
        edgecolor=_GATE_COLOR,
        linewidth=1.6,
        weight="bold",
    )
    _architecture_box(
        ax,
        0.69,
        0.165,
        0.16,
        0.21,
        r"$W_2$" + "\n512 -> 128",
        facecolor=_PROJECTION_FILL,
        edgecolor=_PROJECTION_COLOR,
        linewidth=1.5,
        weight="bold",
    )
    _architecture_box(
        ax,
        0.945,
        0.405,
        0.05,
        0.19,
        r"$+$" + "\n" + r"$H_{l+1}$",
        edgecolor=_RESIDUAL_COLOR,
        weight="bold",
    )

    _architecture_arrow(ax, (0.095, 0.50), (0.13, 0.745))
    _architecture_arrow(ax, (0.26, 0.745), (0.29, 0.745))
    _architecture_arrow(ax, (0.43, 0.745), (0.46, 0.745))
    _architecture_arrow(
        ax,
        (0.685, 0.745),
        (0.715, 0.745),
        color=_GATE_COLOR if qkv_gate else _RESIDUAL_COLOR,
        width=1.6 if qkv_gate else 1.1,
    )
    _architecture_arrow(ax, (0.835, 0.745), (0.855, 0.745))
    _architecture_arrow(ax, (0.925, 0.745), (0.945, 0.55))

    _architecture_arrow(ax, (0.095, 0.50), (0.13, 0.27))
    _architecture_arrow(ax, (0.26, 0.27), (0.31, 0.27))
    _architecture_arrow(ax, (0.46, 0.27), (0.51, 0.27))
    _architecture_arrow(
        ax,
        (0.64, 0.27),
        (0.69, 0.27),
        color=_GATE_COLOR,
        width=1.6,
    )
    _architecture_arrow(ax, (0.85, 0.27), (0.945, 0.45))


def _architecture_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "#666666",
    linewidth: float = 1.0,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=8.0,
        weight=weight,
        linespacing=1.05,
    )


def _architecture_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = _RESIDUAL_COLOR,
    width: float = 1.1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _plot_report05_validation_learning_curves(
    series: list[dict[str, Any]],
) -> Figure:
    """Render Figure 92: four matched architecture rows on shared axes."""

    by_label = _report05_series_by_label(series)
    endpoints = _report05_endpoint_rows(series)
    stock_label, _stock_experiment = REPORT05_STOCK_RUN
    stock_points = _report05_validation_points(by_label[stock_label])

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 8.4),
        sharex=True,
        sharey=True,
    )
    panel_pressure_labels = (
        r"OR/OL1 targets: $h$",
        r"OR/OL1 targets: $a,m,h$",
        r"OR/OL1 targets: $q,k,v$ only",
        r"OR/OL1 targets: $q,k,v$ only",
    )
    all_losses: list[float] = [
        float(point["validation_loss"])
        for item in by_label.values()
        for point in _report05_validation_points(item)
    ]

    for panel_index, (
        ax,
        (_family_id, family_label, family_runs),
        pressure_label,
    ) in enumerate(
        zip(axes, REPORT05_ARCHITECTURE_FAMILIES, panel_pressure_labels, strict=True)
    ):
        _plot_validation_curve(
            ax,
            stock_points,
            label="Stock GELU AdamW",
            style=_STOCK_STYLE,
        )
        for method, (label, _experiment_id) in zip(
            REPORT05_METHOD_ORDER,
            family_runs,
            strict=True,
        ):
            _plot_validation_curve(
                ax,
                _report05_validation_points(by_label[label]),
                label=method,
                style=_METHOD_STYLES[method],
            )
        ax.set_title(
            f"({chr(ord('a') + panel_index)}) {family_label} - {pressure_label}",
            loc="left",
            fontsize=9.0,
            pad=4,
        )
        ax.set_ylabel("Val. loss")
        ax.grid(True, alpha=0.22)

    x_max = max(
        point["tokens_seen"]
        for item in by_label.values()
        for point in _report05_validation_points(item)
    ) / 1e9
    axes[-1].set_xlim(0.0, max(1.52, x_max * 1.015))
    axes[-1].set_xticks((0.0, 0.5, 1.0, 1.5))
    loss_min = min(all_losses)
    loss_max = max(all_losses)
    loss_padding = max(0.05, 0.035 * (loss_max - loss_min))
    axes[-1].set_ylim(loss_min - loss_padding, loss_max + loss_padding)

    legend_handles = [
        Line2D(
            [],
            [],
            color=_STOCK_STYLE.color,
            marker=_STOCK_STYLE.marker,
            linestyle=_STOCK_STYLE.linestyle,
            linewidth=_STOCK_STYLE.linewidth,
            label="Stock GELU AdamW",
        ),
        *[
            Line2D(
                [],
                [],
                color=_METHOD_STYLES[method].color,
                marker=_METHOD_STYLES[method].marker,
                linestyle=_METHOD_STYLES[method].linestyle,
                linewidth=_METHOD_STYLES[method].linewidth,
                label=method,
            )
            for method in REPORT05_METHOD_ORDER
        ],
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8.0,
    )
    validation_tokens = max(
        (
            int(row["validation_tokens"])
            for row in endpoints
            if row["validation_tokens"] is not None
        ),
        default=0,
    )
    fig.suptitle(
        "Validation Learning Curves Across the ReLU Architecture Ladder",
        x=0.5,
        y=0.985,
        fontsize=11.0,
    )
    fig.text(
        0.5,
        0.957,
        (
            "One seed per run; fixed 22,762-step / 1.492B-token budget; "
            + (
                f"{validation_tokens:,} validation tokens per point"
                if validation_tokens
                else "full-validation checkpoints"
            )
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )
    fig.supxlabel("Tokens seen (billions)", x=0.54, y=0.052, fontsize=9.0)
    fig.text(
        0.5,
        0.012,
        (
            "Stock GELU AdamW is repeated in every panel as the common reference.\n"
            "Curves are single-run observations; no uncertainty bands."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.875,
        bottom=0.105,
        hspace=0.18,
    )
    return fig


def _plot_validation_curve(
    ax: Axes,
    points: tuple[dict[str, Any], ...],
    *,
    label: str,
    style: SeriesStyle,
) -> None:
    marker_stride = max(1, len(points) // 7)
    ax.plot(
        [float(point["tokens_seen"]) / 1e9 for point in points],
        [float(point["validation_loss"]) for point in points],
        color=style.color,
        marker=style.marker,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        markersize=3.2,
        markevery=marker_stride,
        label=label,
    )
