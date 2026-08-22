"""Explicit diagnostic plotting from one saved run artifact.

This module deliberately has no experiment registry, run discovery, or paper
cohort selection. Callers name the exact run directory and artifact kind.
Plan-specific paper figure families should compose the shared loader/reducer/
renderer/export contract described in docs/plotting.md.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import textwrap
from typing import Any, Callable
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from paper_exp.plot_api import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    DOUBLE_COLUMN_WIDTH_INCHES,
    export_figure,
)
from paper_exp.plot_common import (
    finite_number,
    histogram_density,
    histogram_nonzero_density,
)
from paper_exp.plot_style import PAPER_STYLE, series_style
from paper_exp.utils import read_json


PLOT_KINDS = (
    "run",
    "clipping",
    "activation-histograms",
    "weight-histograms",
    "activation-propagation",
)
_PLOT_INPUT_FILES = {
    "run": ("config.yaml", "manifest.json", "events.jsonl", "metrics.json"),
    "clipping": ("config.yaml", "manifest.json", "clipping_frontier.jsonl"),
    "activation-histograms": (
        "config.yaml",
        "manifest.json",
        "activation_histograms.json",
    ),
    "weight-histograms": (
        "config.yaml",
        "manifest.json",
        "weight_histograms.json",
    ),
    "activation-propagation": (
        "config.yaml",
        "manifest.json",
        "activation_propagation.json",
    ),
}


def plot_artifact(
    *,
    kind: str,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    """Render one explicit saved artifact to PDF and, optionally, PNG."""

    run_path = Path(run_dir)
    handlers: dict[str, tuple[Callable[[Path], Any], Callable[[Any], Figure]]] = {
        "run": (_load_run_diagnostics, build_run_diagnostics),
        "clipping": (_load_clipping_frontier, build_clipping_frontier),
        "activation-histograms": (
            lambda path: _load_versioned_mapping(
                path / "activation_histograms.json",
                expected_version=2,
            ),
            build_activation_histograms,
        ),
        "weight-histograms": (
            lambda path: _load_versioned_mapping(
                path / "weight_histograms.json",
                expected_version=1,
            ),
            build_weight_histograms,
        ),
        "activation-propagation": (
            lambda path: _load_versioned_mapping(
                path / "activation_propagation.json",
                expected_version=4,
            ),
            build_activation_propagation,
        ),
    }
    if kind not in handlers:
        choices = ", ".join(PLOT_KINDS)
        raise ValueError(f"Unknown plot kind {kind!r}; expected one of: {choices}.")
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    input_paths = _plot_input_paths(kind, run_path)

    loader, builder = handlers[kind]
    payload = loader(run_path)
    outputs = export_figure(
        lambda: builder(payload),
        output,
        save_png=save_png,
        style=PAPER_STYLE,
        profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
    )
    provenance_path = _write_plot_provenance(
        kind=kind,
        run_path=run_path,
        input_paths=input_paths,
        outputs=outputs,
        artifact_schema_version=(
            payload.get("schema_version") if isinstance(payload, dict) else None
        ),
    )
    return [*outputs, provenance_path]


def build_run_diagnostics(payload: dict[str, Any]) -> Figure:
    """Build a model-agnostic run diagnostic from events and terminal metrics."""

    train_events = payload["train_events"]
    validation_events = payload["validation_events"]
    metrics = payload["metrics"]
    manifest = payload["manifest"]

    figure = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 6.3))
    grid = figure.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    loss_axis = figure.add_subplot(grid[0, :])
    gradient_axis = figure.add_subplot(grid[1, 0])
    stats_axis = figure.add_subplot(grid[1, 1])

    loss_axis.plot(
        [row["tokens_seen"] for row in train_events],
        [row["train_loss"] for row in train_events],
        label="train",
        marker="o",
        markersize=2.5,
        linewidth=1.2,
    )
    if validation_events:
        loss_axis.plot(
            [row["tokens_seen"] for row in validation_events],
            [row["validation_loss"] for row in validation_events],
            label="validation",
            marker="s",
            markersize=3.0,
            linewidth=1.2,
        )
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Tokens seen")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend(frameon=False)

    plotted_norm = False
    for key, label, style_index in (
        ("grad_norm", "gradient", 0),
        ("weight_norm", "weight", 1),
    ):
        rows = [row for row in train_events if finite_number(row.get(key))]
        if not rows:
            continue
        style = series_style(style_index)
        gradient_axis.plot(
            [row["tokens_seen"] for row in rows],
            [row[key] for row in rows],
            label=label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markersize=2.5,
            linewidth=style.linewidth,
        )
        plotted_norm = True
    gradient_axis.set_title("Optimization norms")
    gradient_axis.set_xlabel("Tokens seen")
    gradient_axis.set_ylabel("L2 norm")
    if plotted_norm:
        gradient_axis.legend(frameon=False)
    else:
        gradient_axis.text(
            0.5,
            0.5,
            "No gradient or weight norms logged",
            transform=gradient_axis.transAxes,
            ha="center",
            va="center",
        )

    stats_axis.axis("off")
    stats_axis.set_title("Recorded run statistics")
    metric_prefix = "training" if manifest.get("mode") == "pretrain" else "calibration"
    statistic_keys = (
        (f"{metric_prefix}/tokens_per_second", "Average tokens/s"),
        (f"{metric_prefix}/peak_gpu_memory_mb", "Peak GPU allocated (MB)"),
        (f"{metric_prefix}/peak_gpu_reserved_mb", "Peak GPU reserved (MB)"),
        (f"{metric_prefix}/wall_seconds_train", "Train wall time (s)"),
        (f"{metric_prefix}/wall_seconds_total", "Total wall time (s)"),
        ("checkpoint/final_size_mb", "Final checkpoint (MB)"),
    )
    statistic_lines = [
        f"{label}: {_format_number(metrics[key])}"
        for key, label in statistic_keys
        if finite_number(metrics.get(key))
    ]
    statistic_lines.extend(
        (
            f"Train points: {len(train_events)}",
            f"Validation points: {len(validation_events)}",
        )
    )
    stats_axis.text(
        0.02,
        0.94,
        "\n".join(statistic_lines),
        transform=stats_axis.transAxes,
        ha="left",
        va="top",
        linespacing=1.5,
    )

    identity = " / ".join(
        str(value)
        for value in (manifest.get("config_id"), manifest.get("run_id"))
        if value
    )
    figure.suptitle(f"Run diagnostics{f' - {identity}' if identity else ''}", y=0.995)
    return figure


def build_clipping_frontier(rows: list[dict[str, Any]]) -> Figure:
    """Build an exact-sparsity versus validation-loss clipping frontier."""

    if not rows:
        raise ValueError("Clipping artifact contains no frontier points.")
    validated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Clipping row {index} must be a JSON object.")
        achieved = row.get("achieved_sparsity")
        loss = row.get("validation_loss")
        if not finite_number(achieved) or not 0.0 <= float(achieved) <= 1.0:
            raise ValueError(
                f"Clipping row {index} achieved_sparsity must be finite and in [0, 1]."
            )
        if not finite_number(loss):
            raise ValueError(f"Clipping row {index} validation_loss must be finite.")
        validation_tokens = row.get("validation_tokens")
        if validation_tokens is not None and (
            isinstance(validation_tokens, bool)
            or not isinstance(validation_tokens, int)
            or validation_tokens <= 0
        ):
            raise ValueError(
                f"Clipping row {index} validation_tokens must be a positive integer."
            )
        validated_rows.append(row)
    validated_rows.sort(key=lambda row: float(row["achieved_sparsity"]))

    sparsity = [100.0 * float(row["achieved_sparsity"]) for row in validated_rows]
    losses = [float(row["validation_loss"]) for row in validated_rows]
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.4))
    style = series_style(0)
    axis.plot(
        sparsity,
        losses,
        color=style.color,
        marker=style.marker,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        markersize=4.5,
    )
    axis.set_title("Post-hoc activation clipping frontier")
    axis.set_xlabel("Achieved exact-zero activation sparsity (%)")
    axis.set_ylabel("Validation loss")
    axis.set_xlim(left=0.0)

    for index, (x_value, y_value, row) in enumerate(
        zip(sparsity, losses, validated_rows, strict=True)
    ):
        axis.annotate(
            _clipping_label(row),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(5, 6 if index % 2 == 0 else -12),
            fontsize=8,
        )

    span = max(losses) - min(losses)
    margin = max(0.15 * span, 1e-4)
    if min(losses) > 0.0:
        axis.set_ylim(min(losses) - margin, max(losses) + margin)
        scale_note = "validation-loss axis is zoomed"
    else:
        axis.set_ylim(bottom=0.0)
        scale_note = "validation-loss axis starts at zero"
    token_counts = {
        row["validation_tokens"]
        for row in validated_rows
        if row.get("validation_tokens") is not None
    }
    token_note = (
        f"; {next(iter(token_counts)):,} validation tokens/point"
        if len(token_counts) == 1
        else ""
    )
    axis.text(
        0.99,
        0.02,
        f"n={len(validated_rows)} sweep points{token_note}; {scale_note}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout()
    return figure


def build_activation_histograms(payload: dict[str, Any]) -> Figure:
    """Build pooled conditional densities with a separate exact-zero atom."""

    edges, methods = _validated_histogram_payload(payload)
    centers = _bin_centers(edges)
    reductions = [
        _pooled_histogram(method, edges, separate_zero=True)
        for method in methods
    ]

    figure, (atom_axis, density_axis) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.1),
        gridspec_kw={"width_ratios": (1.0, 2.4)},
    )
    labels = [str(method.get("label") or method.get("config_id") or "unnamed") for method in methods]
    atom_values = [item["zero_fraction"] for item in reductions]
    atom_axis.bar(
        range(len(labels)),
        [100.0 * value for value in atom_values],
        color=[series_style(index).color for index in range(len(labels))],
    )
    atom_axis.set_title("Exact-zero atom")
    atom_axis.set_ylabel("Exact zeros (%)")
    atom_axis.set_xticks(range(len(labels)))
    atom_axis.set_xticklabels(labels, rotation=35, ha="right")
    atom_axis.set_ylim(bottom=0.0)

    for index, (label, reduction) in enumerate(zip(labels, reductions, strict=True)):
        style = series_style(index)
        density_axis.plot(
            centers,
            reduction["density"],
            label=label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markevery=max(1, len(centers) // 10),
            markersize=3.0,
            linewidth=style.linewidth,
        )
    density_axis.set_title("Conditional density given nonzero")
    density_axis.set_xlabel("Activation value")
    density_axis.set_ylabel("Density")
    if any(value > 0.0 for item in reductions for value in item["density"]):
        density_axis.set_yscale("log")
    density_axis.legend(frameon=False)
    figure.suptitle(str(payload.get("plot_title") or "Activation distributions"))
    figure.text(
        0.5,
        0.01,
        (
            f"n={len(methods)} checkpoints; integer counts pooled across layers before "
            "normalization; out-of-range mass remains in the denominator"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    return figure


def build_weight_histograms(payload: dict[str, Any]) -> Figure:
    """Build pooled weight densities from one saved diagnostic artifact."""

    edges, methods = _validated_histogram_payload(payload)
    centers = _bin_centers(edges)
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.2))
    for index, method in enumerate(methods):
        reduction = _pooled_histogram(method, edges, separate_zero=False)
        label = str(method.get("label") or method.get("config_id") or "unnamed")
        style = series_style(index)
        axis.plot(
            centers,
            reduction["density"],
            label=label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markevery=max(1, len(centers) // 10),
            markersize=3.0,
            linewidth=style.linewidth,
        )
    axis.set_title(str(payload.get("plot_title") or "Weight distributions"))
    axis.set_xlabel("Weight value")
    axis.set_ylabel("Density over stored total")
    has_in_range_mass = any(
        value > 0.0
        for line in axis.lines
        for value in line.get_ydata()
    )
    if has_in_range_mass:
        axis.set_yscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No in-range weight mass was recorded",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.02,
        (
            f"n={len(methods)} checkpoints; integer counts pooled across layers; "
            "out-of-range mass remains in the denominator"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout()
    return figure


def build_activation_propagation(payload: dict[str, Any]) -> Figure:
    """Build pooled activation and matmul exact-zero heatmaps."""

    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Activation propagation payload has no methods.")
    groups = (
        (
            "activations",
            payload.get("activation_stage_order"),
            payload.get("activation_stage_labels"),
            "Activation exact zeros (%)",
        ),
        (
            "matmuls",
            payload.get("matmul_stage_order"),
            payload.get("matmul_stage_labels"),
            "Logical zero\nproducts (%)",
        ),
    )
    if any(
        not isinstance(order, list)
        or not order
        or not isinstance(labels, dict)
        for _key, order, labels, _title in groups
    ):
        raise ValueError("Activation propagation payload is missing an explicit stage order.")

    method_labels = [
        str(method.get("label") or method.get("config_id") or "unnamed")
        for method in methods
    ]
    longest_order = max(len(order) for _key, order, _labels, _title in groups)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            DOUBLE_COLUMN_WIDTH_INCHES,
            min(8.6, max(5.0, 1.8 + 0.32 * longest_order)),
        ),
        gridspec_kw={"width_ratios": (2.2, 1.0)},
    )
    for axis, (row_key, order, labels, title) in zip(axes, groups, strict=True):
        matrix = np.asarray(
            [_pooled_fraction_row(method, row_key, order) for method in methods],
            dtype=float,
        ).T
        percent_matrix = np.ma.masked_invalid(100.0 * matrix)
        colormap = plt.get_cmap("cividis").with_extremes(bad="#D9D9D9")
        image = axis.imshow(
            percent_matrix,
            vmin=0.0,
            vmax=100.0,
            cmap=colormap,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xticks(range(len(method_labels)))
        axis.set_yticks(range(len(order)))
        axis.set_xticklabels(
            method_labels,
            rotation=35,
            ha="right",
        )
        axis.set_yticklabels(
            [
                textwrap.fill(str(labels.get(str(stage), stage)), width=28)
                for stage in order
            ]
        )
        axis.grid(False)
        if matrix.size <= 80:
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    if math.isfinite(value):
                        axis.text(
                            column_index,
                            row_index,
                            f"{100.0 * value:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value < 0.55 else "black",
                        )
                    else:
                        axis.text(
                            column_index,
                            row_index,
                            "N/A",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#333333",
                        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
        colorbar.set_label("Percent")

    figure.suptitle(
        str(payload.get("plot_title") or "Exact-zero activation propagation"),
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"n={len(methods)} checkpoints; integer zero and total counts pooled across "
            "layers; logical opportunity is not realized kernel speedup"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    return figure


def _load_run_diagnostics(run_path: Path) -> dict[str, Any]:
    events = _read_jsonl(run_path / "events.jsonl")
    metrics = _load_mapping(run_path / "metrics.json")
    manifest_path = run_path / "manifest.json"
    manifest = _load_mapping(manifest_path) if manifest_path.is_file() else {}

    train_events = [
        row
        for row in events
        if row.get("event") == "train"
        and finite_number(row.get("tokens_seen"))
        and finite_number(row.get("train_loss"))
    ]
    if not train_events:
        raise ValueError(f"No finite train events found in {run_path / 'events.jsonl'}.")
    validation_events = [
        row
        for row in events
        if row.get("event") == "validation"
        and finite_number(row.get("tokens_seen"))
        and finite_number(row.get("validation_loss"))
    ]
    return {
        "train_events": train_events,
        "validation_events": validation_events,
        "metrics": metrics,
        "manifest": manifest,
    }


def _load_clipping_frontier(run_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(run_path / "clipping_frontier.jsonl")
    if not rows:
        raise ValueError(f"No clipping rows found in {run_path / 'clipping_frontier.jsonl'}.")
    return rows


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {path}")
    return payload


def _load_versioned_mapping(
    path: Path,
    *,
    expected_version: int,
) -> dict[str, Any]:
    payload = _load_mapping(path)
    actual_version = payload.get("schema_version")
    if actual_version != expected_version:
        raise ValueError(
            f"Unsupported schema_version in {path}: expected {expected_version}, "
            f"found {actual_version!r}."
        )
    return payload


def _plot_input_paths(kind: str, run_path: Path) -> list[Path]:
    paths = [run_path / name for name in _PLOT_INPUT_FILES[kind]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Plot inputs are incomplete; missing: {rendered}")
    manifest = _load_mapping(run_path / "manifest.json")
    if (
        manifest.get("config_id") != run_path.parent.name
        or manifest.get("run_id") != run_path.name
    ):
        raise ValueError(f"Run manifest identity does not match its directory: {run_path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"Plot input run is not completed: {run_path}")
    return paths


def _write_plot_provenance(
    *,
    kind: str,
    run_path: Path,
    input_paths: list[Path],
    outputs: list[Path],
    artifact_schema_version: Any,
) -> Path:
    manifest = _load_mapping(run_path / "manifest.json")
    sidecar = Path(outputs[0]).with_suffix(".provenance.json")
    payload = {
        "schema_version": 1,
        "plot_kind": kind,
        "artifact_schema_version": artifact_schema_version,
        "source": {
            "config_id": manifest.get("config_id"),
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status"),
            "git_commit": manifest.get("git_commit"),
            "source_run": manifest.get("source_run"),
            "source_runs": manifest.get("source_runs"),
            "source_checkpoint": manifest.get("source_checkpoint"),
            "source_checkpoints": manifest.get("source_checkpoints"),
        },
        "inputs": [
            {
                "path": _relative_path(path, anchor=sidecar.parent),
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": _relative_path(path, anchor=sidecar.parent),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    temporary = sidecar.with_name(f".{sidecar.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, *, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), start=anchor.resolve())).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Plot provenance cannot express a portable path across volumes: {path}"
        ) from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _validated_histogram_payload(
    payload: dict[str, Any],
) -> tuple[list[float], list[dict[str, Any]]]:
    raw_edges = payload.get("bin_edges")
    methods = payload.get("methods")
    if not isinstance(raw_edges, list) or len(raw_edges) < 2:
        raise ValueError("Histogram payload has no usable bin edges.")
    edges = [float(value) for value in raw_edges]
    if any(not math.isfinite(value) for value in edges) or any(
        right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)
    ):
        raise ValueError("Histogram edges must be finite and strictly increasing.")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Histogram payload has no methods.")
    if not all(isinstance(method, dict) for method in methods):
        raise ValueError("Every histogram method must be a JSON object.")
    return edges, methods


def _pooled_histogram(
    method: dict[str, Any],
    edges: list[float],
    *,
    separate_zero: bool,
) -> dict[str, Any]:
    layers = method.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"Histogram method {method.get('label')!r} has no layers.")
    counts = [0] * (len(edges) - 1)
    total = 0
    exact_zeros = 0
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Histogram layers must be JSON objects.")
        raw_counts = layer.get("counts")
        if not isinstance(raw_counts, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_counts
        ):
            raise ValueError("Histogram layer counts must be nonnegative integers.")
        if len(raw_counts) != len(counts):
            raise ValueError("Histogram layers do not share the stored bin edges.")
        counts = [
            pooled + value
            for pooled, value in zip(counts, raw_counts, strict=True)
        ]
        raw_total = layer.get("total")
        if (
            isinstance(raw_total, bool)
            or not isinstance(raw_total, int)
            or raw_total < sum(raw_counts)
        ):
            raise ValueError(
                "Histogram layer total must be an integer covering every in-range count."
            )
        total += raw_total
        threshold_hits = layer.get("threshold_hits") or {}
        raw_exact_zeros = threshold_hits.get("0", 0)
        if (
            isinstance(raw_exact_zeros, bool)
            or not isinstance(raw_exact_zeros, int)
            or raw_exact_zeros < 0
            or raw_exact_zeros > raw_total
        ):
            raise ValueError("Histogram exact-zero counts must be valid nonnegative integers.")
        exact_zeros += raw_exact_zeros

    pooled_layer = {
        "counts": counts,
        "total": total,
        "threshold_hits": {"0": exact_zeros},
    }
    if separate_zero:
        density, zero_fraction = histogram_nonzero_density(pooled_layer, edges)
    else:
        density = histogram_density(pooled_layer, edges)
        zero_fraction = 0.0
    return {"density": density, "zero_fraction": zero_fraction}


def _pooled_fraction_row(
    method: dict[str, Any],
    row_key: str,
    order: list[Any],
) -> list[float]:
    rows = method.get(row_key)
    if not isinstance(rows, list):
        raise ValueError(f"Propagation method {method.get('label')!r} is missing {row_key}.")
    counts = {str(stage): [0, 0] for stage in order}
    seen = {str(stage): False for stage in order}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Propagation rows must be JSON objects.")
        name = str(row.get("name"))
        if name not in counts:
            continue
        seen[name] = True
        zero_count = row.get("zero_count")
        total = row.get("total")
        if zero_count is None and total is None:
            continue
        if (
            zero_count is None
            or total is None
            or isinstance(zero_count, bool)
            or not isinstance(zero_count, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or zero_count < 0
            or total <= 0
            or zero_count > total
        ):
            raise ValueError(f"Invalid integer propagation counts for stage {name!r}.")
        counts[name][0] += zero_count
        counts[name][1] += total
    missing = [name for name, was_seen in seen.items() if not was_seen]
    if missing:
        raise ValueError(
            f"Propagation method {method.get('label')!r} is missing stages: {', '.join(missing)}."
        )
    return [
        zeros / total if total > 0 else math.nan
        for zeros, total in counts.values()
    ]


def _bin_centers(edges: list[float]) -> list[float]:
    return [
        0.5 * (left + right)
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]


def _clipping_label(row: dict[str, Any]) -> str:
    if finite_number(row.get("threshold")):
        return f"t={float(row['threshold']):g}"
    if finite_number(row.get("quantile")):
        return f"q={float(row['quantile']):g}"
    if finite_number(row.get("rms_multiplier")):
        return f"r={float(row['rms_multiplier']):g}"
    return str(row.get("mode") or "clip")


def _format_number(value: Any) -> str:
    numeric = float(value)
    if abs(numeric) >= 1000.0:
        return f"{numeric:,.0f}"
    return f"{numeric:.4g}"
