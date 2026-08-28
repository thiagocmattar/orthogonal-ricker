"""Deterministic A2 spillover reductions and paper-figure suite.

This module deliberately pins the accepted seed-0 A2 cohort and its one
post-hoc activation-histogram run.  It does not discover attempts or infer a
latest run.  Site summaries are count-first: threshold hits and denominators
are summed before division, while pooled RMS follows the reviewed
finite-count-weighted second-moment reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml
from matplotlib.figure import Figure

from paper_exp.config import validate_diagnostic_config, validate_training_config
from paper_exp.design import complete_config_sha256
from paper_exp.utils import read_json

from .export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    DOUBLE_COLUMN_WIDTH_INCHES,
    export_figure,
    publish_staged_outputs,
)
from .histograms import bin_centers, histogram_layer, histogram_nonzero_density
from .style import PAPER_STYLE, series_style


TRANCHE_ID = "02-a2-l1-screen"
DIAGNOSTIC_CONFIG_ID = "018-a2-activation-histograms"
DIAGNOSTIC_RUN_ID = "001-20260828-082044-a031175f"
DIAGNOSTIC_GIT_COMMIT = "a0f86e057b0f67e3a2726b9cb6e352d8f8914176"
RESPONSE_STEM = "01-a2-spillover-response"
DISTRIBUTION_STEM = "02-a2-layer5-distributions"
GENERATOR_PATH = "src/paper_exp/plots/a2_spillover.py"

SITES = ("h", "a", "m", "q_post", "k_post", "v")
SITE_LABELS = {
    "h": r"$h$",
    "a": r"$a$",
    "m": r"$m$",
    "q_post": r"$q_{post}$",
    "k_post": r"$k_{post}$",
    "v": r"$v$",
}
LAYERS = tuple(range(6))
THRESHOLDS = (0.0, 0.01, 0.1)
THRESHOLD_KEYS = ("0", "0.01", "0.1")
EXPECTED_BINS = 3_200
EXPECTED_RANGE = (-16.0, 16.0)
EXPECTED_VALIDATION_SEQUENCES = 152
EXPECTED_VALIDATION_TOKENS = 311_296
EXPECTED_VALIDATION_BATCHES = 38
EXPECTED_SELECTION_HASH = (
    "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
)
EXPECTED_TRAINING_GIT_COMMIT = "7a573432ca45cf184395551c7f45110e27552244"
EXPECTED_IMPLEMENTATION_ID = "a1_pretraining_v1"
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "778955b0319dc27e39201153e55c491350f90e0317e0a7b6ae6c7590fa7cfd17"
)
EXPECTED_SCHEDULE_HASH = (
    "35da3f6aa891a2248407344715e4c75e99cb518b17119a8e66004466a823a21c"
)
EXPECTED_STEPS = 5_691
EXPECTED_TOKENS_PER_STEP = 262_144
EXPECTED_TRAINING_TOKENS = 1_491_861_504
EXPECTED_LAYER_TOTALS = {
    "h": EXPECTED_VALIDATION_TOKENS * 512,
    "a": EXPECTED_VALIDATION_TOKENS * 128,
    "m": EXPECTED_VALIDATION_TOKENS * 128,
    "q_post": EXPECTED_VALIDATION_TOKENS * 128,
    "k_post": EXPECTED_VALIDATION_TOKENS * 128,
    "v": EXPECTED_VALIDATION_TOKENS * 128,
}


@dataclass(frozen=True)
class A2Source:
    """One exact accepted A2 pretraining source."""

    config_id: str
    run_id: str
    label: str
    lambda_value: float
    config_sha256: str
    condition_fingerprint: str
    group_id: str


A2_SOURCES = (
    A2Source(
        "012-a2-relu-control",
        "001-20260827-150809-2eb832f6",
        "ReLU control",
        0.0,
        "9ea7b92e8ce1deb79b83eee695f58985c6c2bfe5d0add2994547b7fcee0eece8",
        "6d1901448dcf3c2736c66970bd218e8bdb3107f8937e8dfb5b9e040099263172",
        "A2-relu-control",
    ),
    A2Source(
        "013-a2-l1-1e-1",
        "001-20260827-150808-8117d1fe",
        "L1 lambda=0.1",
        0.1,
        "9c3da8cb8c4d3b9f808f12c098c3e3c64fb193e2bad15da193b5365d7cd84e4f",
        "43f9c08acb04b58be8b54fbe9b6268893e14e742379fe897a90f6d3864331b84",
        "A2-l1-screen",
    ),
    A2Source(
        "014-a2-l1-5e-1",
        "001-20260827-173546-360c077f",
        "L1 lambda=0.5",
        0.5,
        "e5a20814b97cfed1b550c8f53e512fb5c92acf549451240f47e491498bdde154",
        "c1185d23ec47335c5e34f0c9ff0fc8eec6d51f4c6e7c29c9a9c16fd62f326021",
        "A2-l1-screen",
    ),
    A2Source(
        "015-a2-l1-1",
        "001-20260827-193752-3fbbd6c0",
        "L1 lambda=1",
        1.0,
        "a89b5dc89d34850935449f53155f6d83d8d3077fe6e66e491def71ded884b5c3",
        "e9a14f4f94e1c34b64cd423cdf7874bbaf85abf9d155b528b0e29894017261e9",
        "A2-l1-screen",
    ),
    A2Source(
        "016-a2-l1-2",
        "001-20260827-220532-79995961",
        "L1 lambda=2",
        2.0,
        "c498d426fcc7ee133f5497923d84a66777c2f645c825bcb6c607096fcccc8c82",
        "a85bbf26d6790f2a1ac3e6c89fa2665f47508bcbe1effc2fe3fa3d778657fc63",
        "A2-l1-screen",
    ),
    A2Source(
        "017-a2-l1-5",
        "001-20260828-000829-0959f855",
        "L1 lambda=5",
        5.0,
        "99da3a160c26700a5ad611a1f6d1597814f0c19433cdb20fdbb7ad5a0a8751af",
        "c7ca2a1a3a65464091f8ea3b2dca0484adbc7993a7f44c70708244e0ba98df25",
        "A2-l1-screen",
    ),
)


@dataclass(frozen=True)
class SiteReduction:
    """Count-first site summary pooled across all transformer layers."""

    config_id: str
    label: str
    lambda_value: float
    site: str
    total: int
    finite: int
    exact_zero_hits: int
    near_zero_0p01_hits: int
    near_zero_0p1_hits: int
    underflow: int
    overflow: int
    exact_zero_fraction: float
    near_zero_0p01_fraction: float
    near_zero_0p1_fraction: float
    pooled_rms: float


@dataclass(frozen=True)
class LossPoint:
    """Final complete-selection validation result for one A2 cell."""

    config_id: str
    run_id: str
    label: str
    lambda_value: float
    final_validation_loss: float


@dataclass(frozen=True)
class A2SpilloverData:
    """Validated fixed-cohort data ready for rendering."""

    reductions: tuple[SiteReduction, ...]
    losses: tuple[LossPoint, ...]
    bin_edges: tuple[float, ...]
    methods: tuple[dict[str, Any], ...]
    inputs: tuple[dict[str, Any], ...]


def reduce_site_layers(
    layers: Sequence[dict[str, Any]],
    *,
    source: A2Source,
    site: str,
    bin_count: int = EXPECTED_BINS,
) -> SiteReduction:
    """Pool one site's layer rows by integer counts and finite second moments."""

    if site not in SITES:
        raise ValueError(f"Unsupported A2 site: {site!r}.")
    expected_names = tuple(f"{site}.layer_{layer}" for layer in LAYERS)
    named = {str(layer.get("name")): layer for layer in layers}
    if len(named) != len(layers):
        raise ValueError(f"Duplicate activation-histogram layer for {source.config_id}.")
    selected = []
    for name in expected_names:
        row = named.get(name)
        if row is None:
            raise ValueError(f"Missing {name} for {source.config_id}.")
        _validate_layer_row(row, site=site, bin_count=bin_count)
        selected.append(row)

    total = sum(int(row["total"]) for row in selected)
    finite = sum(int(row["finite"]) for row in selected)
    hits = tuple(
        sum(int(row["threshold_hits"][key]) for row in selected)
        for key in THRESHOLD_KEYS
    )
    if total <= 0 or finite <= 0:
        raise ValueError(f"A2 site {site} has no finite observations: {source.config_id}.")
    rms_square_sum = sum(
        int(row["finite"]) * float(row["rms"]) ** 2 for row in selected
    )
    return SiteReduction(
        config_id=source.config_id,
        label=source.label,
        lambda_value=source.lambda_value,
        site=site,
        total=total,
        finite=finite,
        exact_zero_hits=hits[0],
        near_zero_0p01_hits=hits[1],
        near_zero_0p1_hits=hits[2],
        underflow=sum(int(row["underflow"]) for row in selected),
        overflow=sum(int(row["overflow"]) for row in selected),
        exact_zero_fraction=hits[0] / total,
        near_zero_0p01_fraction=hits[1] / total,
        near_zero_0p1_fraction=hits[2] / total,
        pooled_rms=math.sqrt(rms_square_sum / finite),
    )


def load_a2_spillover(
    repository: str | Path | None = None,
) -> A2SpilloverData:
    """Load the exact accepted A2 cohort and diagnostic without discovery."""

    root = _repository_root(repository)
    losses: list[LossPoint] = []
    inputs: list[dict[str, Any]] = []
    for source in A2_SOURCES:
        loss, source_inputs = _load_training_source(root, source)
        losses.append(loss)
        inputs.extend(source_inputs)

    payload, diagnostic_inputs = _load_diagnostic(root)
    inputs.extend(diagnostic_inputs)
    methods = tuple(payload["methods"])
    reductions = tuple(
        reduce_site_layers(
            method["layers"],
            source=source,
            site=site,
            bin_count=EXPECTED_BINS,
        )
        for source, method in zip(A2_SOURCES, methods, strict=True)
        for site in SITES
    )
    return A2SpilloverData(
        reductions=reductions,
        losses=tuple(losses),
        bin_edges=tuple(float(value) for value in payload["bin_edges"]),
        methods=methods,
        inputs=tuple(inputs),
    )


def build_a2_spillover_response_figure(data: A2SpilloverData) -> Figure:
    """Plot primary near-zero and supporting RMS responses versus control."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.25),
        sharex=True,
    )
    x = tuple(range(len(A2_SOURCES)))
    tick_labels = ("control", "0.1", "0.5", "1", "2", "5")
    for site_index, site in enumerate(SITES):
        rows = _site_rows(data.reductions, site)
        control = rows[0]
        near_zero_change = [
            100.0 * (row.near_zero_0p01_fraction - control.near_zero_0p01_fraction)
            for row in rows
        ]
        rms_change = [
            100.0 * (row.pooled_rms / control.pooled_rms - 1.0) for row in rows
        ]
        style = series_style(site_index)
        for axis, values in zip(axes, (near_zero_change, rms_change), strict=True):
            axis.plot(
                x,
                values,
                label=SITE_LABELS[site],
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                linewidth=1.4,
                markersize=4.2,
            )
    axes[0].set_title(r"Near-zero response, $n_s(0.01)$")
    axes[0].set_ylabel("Change from control (percentage points)")
    axes[1].set_title("Activation-scale response")
    axes[1].set_ylabel("Pooled RMS change from control (%)")
    for axis in axes:
        axis.axhline(0.0, color="#666666", linewidth=0.8, zorder=0)
        axis.set_xticks(x)
        axis.set_xticklabels(tick_labels)
        axis.set_xlabel(r"L1 coefficient $\lambda$")
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(SITES),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle("Pythia-14M A2 spillover response (seed 0)")
    figure.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.25, wspace=0.30)
    return figure


def build_a2_layer5_distributions_figure(data: A2SpilloverData) -> Figure:
    """Plot deepest-layer control/L1-lambda-1 distributions at all six sites."""

    control = data.methods[0]
    lambda_one = data.methods[3]
    centers = bin_centers(list(data.bin_edges))
    figure = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 8.55))
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(0.72, 2.15, 0.72, 2.15),
        hspace=0.44,
        wspace=0.28,
    )
    density_handles = []
    density_labels = ("ReLU control", r"L1 $\lambda=1$")
    for index, site in enumerate(SITES):
        pair_row = 0 if index < 3 else 2
        column = index % 3
        atom_axis = figure.add_subplot(grid[pair_row, column])
        density_axis = figure.add_subplot(grid[pair_row + 1, column])
        atom_axis.set_title(SITE_LABELS[site], pad=2)
        for method_index, method in enumerate((control, lambda_one)):
            layer = histogram_layer(method, f"{site}.layer_5")
            density, zero_fraction = histogram_nonzero_density(
                layer, list(data.bin_edges)
            )
            style = series_style(method_index)
            bar = atom_axis.bar(
                method_index,
                100.0 * zero_fraction,
                color=style.color,
                width=0.62,
            )
            height = float(bar[0].get_height())
            atom_axis.text(
                method_index,
                max(height + 2.0, 2.0),
                _format_percent(height),
                ha="center",
                va="bottom",
                fontsize=8,
            )
            (line,) = density_axis.plot(
                centers,
                density,
                color=style.color,
                linestyle=style.linestyle,
                linewidth=1.2,
                label=density_labels[method_index],
            )
            if index == 0:
                density_handles.append(line)
        atom_axis.set_ylim(0.0, 108.0)
        atom_axis.set_xticks((0, 1))
        atom_axis.set_xticklabels(("control", r"$\lambda=1$"))
        atom_axis.grid(False)
        atom_axis.yaxis.grid(True, alpha=0.20)
        if column == 0:
            atom_axis.set_ylabel("Exact zeros (%)")
        else:
            atom_axis.tick_params(labelleft=False)
        density_axis.set_xlim(EXPECTED_RANGE)
        density_axis.set_yscale("log")
        density_axis.grid(False)
        density_axis.yaxis.grid(True, alpha=0.20)
        if pair_row == 2:
            density_axis.set_xlabel("Activation value")
        if column == 0:
            density_axis.set_ylabel("Nonzero density")
        else:
            density_axis.tick_params(labelleft=False)
    figure.legend(
        density_handles,
        density_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.suptitle("Pythia-14M layer 5 activation distributions (seed 0)")
    figure.subplots_adjust(left=0.16, right=0.985, top=0.95, bottom=0.085)
    return figure


def build_a2_response_markdown(data: A2SpilloverData) -> str:
    """Build the response caption and exact count-first reduction tables."""

    lines = [
        "# A2 spillover response",
        "",
        (
            "**Figure caption.** Seed-0 Pythia-14M response to h-only naive L1 "
            "pressure relative to the matched ReLU-only A1-H control. The left "
            "panel is the primary count-first change in near-zero mass "
            "`n_s(0.01)`; the right panel is the supporting finite-count-weighted "
            "pooled RMS change."
        ),
        "",
        (
            "This is a single-seed directional screen (n = 1 per condition). It "
            "does not estimate seed uncertainty or support robustness, functional-"
            "compensation, compute-reduction, or speedup claims."
        ),
        "",
        "## Final validation loss",
        "",
        "| Condition | Config / run | Final validation loss | Change from control |",
        "| --- | --- | ---: | ---: |",
    ]
    control_loss = data.losses[0].final_validation_loss
    for point in data.losses:
        lines.append(
            f"| {point.label} | `{point.config_id}` / `{point.run_id}` | "
            f"{point.final_validation_loss:.6f} | "
            f"{point.final_validation_loss - control_loss:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Count-first activation summaries",
            "",
            (
                "`z`, `n(.01)`, and `n(.1)` are sums of integer hits divided by "
                "sums of totals across all six layers. RMS is "
                "`sqrt(sum(finite * rms^2) / sum(finite))`."
            ),
            "",
            (
                "| Condition | Site | z | n(.01) | Δn(.01), pp | n(.1) | "
                "Pooled RMS | ΔRMS, % |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    controls = {row.site: row for row in _source_rows(data.reductions, A2_SOURCES[0])}
    for source in A2_SOURCES:
        for row in _source_rows(data.reductions, source):
            control = controls[row.site]
            lines.append(
                f"| {source.label} | `{row.site}` | {row.exact_zero_fraction:.8f} | "
                f"{row.near_zero_0p01_fraction:.8f} | "
                f"{100.0 * (row.near_zero_0p01_fraction - control.near_zero_0p01_fraction):+.4f} | "
                f"{row.near_zero_0p1_fraction:.8f} | {row.pooled_rms:.8f} | "
                f"{100.0 * (row.pooled_rms / control.pooled_rms - 1.0):+.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_a2_distribution_markdown(data: A2SpilloverData) -> str:
    """Build the layer-5 distribution caption and explicit tail disclosure."""

    lines = [
        "# A2 layer-5 activation distributions",
        "",
        (
            "**Figure caption.** Deepest-transformer-layer (`layer_5`) activation "
            "distributions for the seed-0 ReLU-only control and h-only L1 at "
            "lambda 1. Each site shows the exact-zero atom separately from the "
            "conditional density among nonzero activations."
        ),
        "",
        (
            "The curves show the complete stored histogram range `[-16, 16]`. "
            "Underflow and overflow are not drawn but remain in the density "
            "normalization denominator; the table reports their combined mass. "
            "Accordingly, a curve need not integrate to one over the displayed "
            "range. No nonfinite activations were accepted by the loader."
        ),
        "",
        (
            "This is a seed-0 descriptive contrast (n = 1 per condition), not a "
            "seed-robustness or population claim."
        ),
        "",
        "| Condition | Site | Exact-zero atom | Outside stored range |",
        "| --- | --- | ---: | ---: |",
    ]
    for source_index in (0, 3):
        source = A2_SOURCES[source_index]
        method = data.methods[source_index]
        for site in SITES:
            layer = histogram_layer(method, f"{site}.layer_5")
            total = int(layer["total"])
            zero = int(layer["threshold_hits"]["0"]) / total
            tail = (int(layer["underflow"]) + int(layer["overflow"])) / total
            lines.append(
                f"| {source.label} | `{site}` | {zero:.8%} | {tail:.8%} |"
            )
    lines.append("")
    return "\n".join(lines)


def generate_a2_spillover_suite(
    repository: str | Path | None = None,
) -> tuple[Path, ...]:
    """Validate inputs and atomically publish both A2 figure packages."""

    root = _repository_root(repository)
    data = load_a2_spillover(root)
    figures_dir = root / "experiments" / TRANCHE_ID / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stems = (RESPONSE_STEM, DISTRIBUTION_STEM)
    final_paths = tuple(
        figures_dir / f"{stem}.{suffix}"
        for stem in stems
        for suffix in ("pdf", "png", "md", "provenance.json")
    )
    with tempfile.TemporaryDirectory(prefix=".a2-spillover.", dir=figures_dir) as temp:
        staging = Path(temp)
        staged = {path: staging / path.name for path in final_paths}
        response_pdf = staged[figures_dir / f"{RESPONSE_STEM}.pdf"]
        distribution_pdf = staged[figures_dir / f"{DISTRIBUTION_STEM}.pdf"]
        export_figure(
            lambda: build_a2_spillover_response_figure(data),
            response_pdf,
            save_png=True,
            style=PAPER_STYLE,
            profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
        )
        export_figure(
            lambda: build_a2_layer5_distributions_figure(data),
            distribution_pdf,
            save_png=True,
            style=PAPER_STYLE,
            profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
        )
        _write_text(staged[figures_dir / f"{RESPONSE_STEM}.md"], build_a2_response_markdown(data))
        _write_text(
            staged[figures_dir / f"{DISTRIBUTION_STEM}.md"],
            build_a2_distribution_markdown(data),
        )
        for stem in stems:
            suite_outputs = tuple(
                figures_dir / f"{stem}.{suffix}" for suffix in ("pdf", "png", "md")
            )
            provenance = _build_provenance(
                root=root,
                stem=stem,
                data=data,
                staged_outputs=tuple(staged[path] for path in suite_outputs),
                final_outputs=suite_outputs,
            )
            _write_json(staged[figures_dir / f"{stem}.provenance.json"], provenance)
        publish_staged_outputs({path: staged[path] for path in final_paths})
    return final_paths


def _load_training_source(
    root: Path, source: A2Source
) -> tuple[LossPoint, tuple[dict[str, Any], ...]]:
    recipe_path = root / "experiments" / TRANCHE_ID / "run" / f"{source.config_id}.yaml"
    run_dir = root / "experiments" / TRANCHE_ID / "raw" / source.config_id / source.run_id
    paths = {
        "recipe_config": recipe_path,
        "saved_config": run_dir / "config.yaml",
        "manifest": run_dir / "manifest.json",
        "metrics": run_dir / "metrics.json",
        "checkpoint": run_dir / "checkpoints" / "final" / "model.safetensors",
    }
    _require_files(paths.values(), context=f"A2 source {source.config_id}")
    recipe = _load_yaml(paths["recipe_config"])
    saved = _load_yaml(paths["saved_config"])
    if saved != recipe:
        raise ValueError(f"Saved config does not match tracked recipe: {run_dir}")
    validate_training_config(recipe)
    if complete_config_sha256(recipe) != source.config_sha256:
        raise ValueError(f"A2 config SHA-256 mismatch: {run_dir}")
    _validate_source_config(recipe, source, run_dir)
    manifest = _load_json(paths["manifest"])
    metrics = _load_json(paths["metrics"])
    _validate_source_manifest(manifest, source, run_dir)
    loss = _validate_source_metrics(metrics, run_dir)
    point = LossPoint(
        config_id=source.config_id,
        run_id=source.run_id,
        label=source.label,
        lambda_value=source.lambda_value,
        final_validation_loss=loss,
    )
    return point, tuple(
        _input_record(root, path, role=role, source=source)
        for role, path in paths.items()
    )


def _load_diagnostic(root: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    recipe_path = root / "experiments" / TRANCHE_ID / "run" / f"{DIAGNOSTIC_CONFIG_ID}.yaml"
    run_dir = (
        root / "experiments" / TRANCHE_ID / "raw" / DIAGNOSTIC_CONFIG_ID / DIAGNOSTIC_RUN_ID
    )
    paths = {
        "diagnostic_recipe": recipe_path,
        "diagnostic_saved_config": run_dir / "config.yaml",
        "diagnostic_manifest": run_dir / "manifest.json",
        "diagnostic_metrics": run_dir / "metrics.json",
        "activation_histograms": run_dir / "activation_histograms.json",
    }
    _require_files(paths.values(), context="A2 diagnostic")
    recipe = _load_yaml(paths["diagnostic_recipe"])
    saved = _load_yaml(paths["diagnostic_saved_config"])
    if saved != recipe:
        raise ValueError(f"Saved diagnostic config does not match tracked recipe: {run_dir}")
    validate_diagnostic_config(recipe, "activation_histograms")
    _validate_diagnostic_config(recipe, run_dir)
    manifest = _load_json(paths["diagnostic_manifest"])
    metrics = _load_json(paths["diagnostic_metrics"])
    payload = _load_json(paths["activation_histograms"])
    _validate_diagnostic_manifest(manifest, run_dir)
    _validate_diagnostic_metrics(metrics, run_dir)
    _validate_diagnostic_payload(payload, run_dir)
    return payload, tuple(
        _input_record(root, path, role=role, source=None)
        for role, path in paths.items()
    )


def _validate_source_config(config: dict[str, Any], source: A2Source, run_dir: Path) -> None:
    identity = _mapping(config.get("identity"), "config.identity", run_dir)
    run = _mapping(config.get("run"), "config.run", run_dir)
    model = _mapping(config.get("model"), "config.model", run_dir)
    training = _mapping(config.get("training"), "config.training", run_dir)
    validation = _mapping(config.get("validation"), "config.validation", run_dir)
    pressure = _mapping(config.get("activation_pressure"), "config.activation_pressure", run_dir)
    expected_pressure = {
        "enabled": source.lambda_value > 0.0,
        "method": "l1_naive" if source.lambda_value > 0.0 else "none",
        "sites": ["h"],
        "weight": source.lambda_value,
        "step_budget": None,
    }
    checks = {
        "group": identity.get("group_id") == source.group_id,
        "fingerprint": identity.get("condition_fingerprint") == source.condition_fingerprint,
        "implementation": identity.get("training_implementation_id") == EXPECTED_IMPLEMENTATION_ID,
        "seed": run.get("seed") == run.get("model_initialization_seed") == run.get("data_order_seed") == 0,
        "schedule": run.get("training_schedule_hash") == EXPECTED_SCHEDULE_HASH,
        "model": model.get("initialization") == "random" and model.get("topology_id") == "A1-H",
        "site gate": model.get("site_gate") == {"operator": "relu"},
        "training": training.get("max_steps") == EXPECTED_STEPS and float(training.get("learning_rate", math.nan)) == 0.064,
        "batch": training.get("micro_batch_size") == 16 and training.get("gradient_accumulation_steps") == 8,
        "validation": validation.get("partition") == "selection" and validation.get("partition_hash") == EXPECTED_SELECTION_HASH and validation.get("eval_batches") is None,
        "pressure": all(pressure.get(key) == value for key, value in expected_pressure.items()),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"A2 source config mismatch ({', '.join(failed)}): {run_dir}")


def _validate_source_manifest(
    manifest: dict[str, Any], source: A2Source, run_dir: Path
) -> None:
    expected = {
        "status": "completed",
        "mode": "pretrain",
        "tranche_id": TRANCHE_ID,
        "config_id": source.config_id,
        "run_id": source.run_id,
        "git_commit": EXPECTED_TRAINING_GIT_COMMIT,
        "git_dirty": False,
        "config_sha256": source.config_sha256,
        "condition_fingerprint": source.condition_fingerprint,
        "case_group_id": source.group_id,
        "training_implementation_id": EXPECTED_IMPLEMENTATION_ID,
        "seed": 0,
        "model_initialization_seed": 0,
        "data_order_seed": 0,
        "training_schedule_hash": EXPECTED_SCHEDULE_HASH,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_SELECTION_HASH,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    checkpoint = manifest.get("checkpoint")
    model = manifest.get("model")
    training = manifest.get("training")
    validation = (manifest.get("tokenized_data") or {}).get("validation")
    if not isinstance(checkpoint, dict) or checkpoint.get("saved") is not True or checkpoint.get("path") != "checkpoints/final":
        mismatches.append("checkpoint")
    if not isinstance(model, dict) or model.get("initial_parameter_sha256") != EXPECTED_INITIAL_PARAMETER_SHA256 or model.get("loaded_checkpoint_weights") is not False:
        mismatches.append("model")
    if not isinstance(training, dict) or any(
        training.get(key) != value
        for key, value in {
            "completed_steps": EXPECTED_STEPS,
            "max_steps": EXPECTED_STEPS,
            "tokens_per_step": EXPECTED_TOKENS_PER_STEP,
            "stopped_by_operational_wall_time_limit": False,
        }.items()
    ):
        mismatches.append("training")
    if not isinstance(validation, dict) or any(
        validation.get(key) != value
        for key, value in {
            "partition": "selection",
            "source_document_indices_sha256": EXPECTED_SELECTION_HASH,
            "tokens_sha256": "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19",
        }.items()
    ):
        mismatches.append("validation cache")
    if mismatches:
        raise ValueError(f"A2 manifest provenance mismatch ({', '.join(mismatches)}): {run_dir}")


def _validate_source_metrics(metrics: dict[str, Any], run_dir: Path) -> float:
    expected = {
        "training/optimizer_steps": EXPECTED_STEPS,
        "training/planned_optimizer_steps": EXPECTED_STEPS,
        "training/tokens_per_step": EXPECTED_TOKENS_PER_STEP,
        "training/tokens_seen": EXPECTED_TRAINING_TOKENS,
        "training/validation_loss_final_step": EXPECTED_STEPS,
        "training/validation_tokens_final": EXPECTED_VALIDATION_TOKENS,
        "training/validation_sequences_final": EXPECTED_VALIDATION_SEQUENCES,
        "training/validation_available_complete_blocks": EXPECTED_VALIDATION_SEQUENCES,
        "training/validation_batches_final": EXPECTED_VALIDATION_BATCHES,
        "training/validation_partition": "selection",
        "training/validation_partition_hash": EXPECTED_SELECTION_HASH,
        "training/training_schedule_hash": EXPECTED_SCHEDULE_HASH,
    }
    mismatches = [key for key, value in expected.items() if metrics.get(key) != value]
    if mismatches:
        raise ValueError(f"A2 metric budget mismatch ({', '.join(mismatches)}): {run_dir}")
    if metrics.get("training/validation_complete_block_coverage") is not True or metrics.get("training/wall_time_limit_reached") is not False:
        raise ValueError(f"A2 validation or terminal coverage mismatch: {run_dir}")
    loss = metrics.get("training/validation_loss_final")
    if isinstance(loss, bool) or not isinstance(loss, int | float) or not math.isfinite(float(loss)):
        raise ValueError(f"A2 final validation loss is not finite: {run_dir}")
    return float(loss)


def _validate_diagnostic_config(config: dict[str, Any], run_dir: Path) -> None:
    diagnostic = _mapping(config.get("activation_histograms"), "activation_histograms", run_dir)
    validation = _mapping(config.get("validation"), "validation", run_dir)
    selected = diagnostic.get("selected_runs")
    expected_selected = [
        {
            "label": source.label,
            "tranche_id": TRANCHE_ID,
            "config_id": source.config_id,
            "run_id": source.run_id,
        }
        for source in A2_SOURCES
    ]
    checks = {
        "selected runs": selected == expected_selected,
        "sites": diagnostic.get("sites") == list(SITES),
        "thresholds": diagnostic.get("thresholds") == list(THRESHOLDS),
        "bins": diagnostic.get("bins") == EXPECTED_BINS,
        "range": (float(diagnostic.get("range_min", math.nan)), float(diagnostic.get("range_max", math.nan))) == EXPECTED_RANGE,
        "partition": validation.get("partition") == "selection" and validation.get("partition_hash") == EXPECTED_SELECTION_HASH,
        "coverage": validation.get("eval_batches") is None and validation.get("batch_size") == 4,
        "seed": (config.get("run") or {}).get("seed") == 0,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"A2 diagnostic recipe mismatch ({', '.join(failed)}): {run_dir}")


def _validate_diagnostic_manifest(manifest: dict[str, Any], run_dir: Path) -> None:
    expected = {
        "status": "completed",
        "mode": "activation-histograms",
        "tranche_id": TRANCHE_ID,
        "config_id": DIAGNOSTIC_CONFIG_ID,
        "run_id": DIAGNOSTIC_RUN_ID,
        "git_commit": DIAGNOSTIC_GIT_COMMIT,
        "git_dirty": False,
        "seed": 0,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_SELECTION_HASH,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    expected_runs = [
        f"experiments/{TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
        for source in A2_SOURCES
    ]
    expected_checkpoints = [f"{path}/checkpoints/final" for path in expected_runs]
    if manifest.get("source_runs") != expected_runs:
        mismatches.append("source_runs")
    if manifest.get("source_checkpoints") != expected_checkpoints:
        mismatches.append("source_checkpoints")
    diagnostic = manifest.get("activation_histograms")
    if not isinstance(diagnostic, dict) or any(
        diagnostic.get(key) != value
        for key, value in {
            "bins": EXPECTED_BINS,
            "range_min": EXPECTED_RANGE[0],
            "range_max": EXPECTED_RANGE[1],
            "sites": list(SITES),
            "thresholds": list(THRESHOLDS),
            "eval_batches": None,
            "batch_size": 4,
            "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        }.items()
    ):
        mismatches.append("activation_histograms")
    if mismatches:
        raise ValueError(f"A2 diagnostic manifest mismatch ({', '.join(mismatches)}): {run_dir}")


def _validate_diagnostic_metrics(metrics: dict[str, Any], run_dir: Path) -> None:
    expected = {
        "activation_histograms/methods": len(A2_SOURCES),
        "activation_histograms/layers": len(SITES) * len(LAYERS),
        "activation_histograms/bins": EXPECTED_BINS,
        "activation_histograms/range_min": EXPECTED_RANGE[0],
        "activation_histograms/range_max": EXPECTED_RANGE[1],
        "activation_histograms/validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
        "activation_histograms/validation_tokens": EXPECTED_VALIDATION_TOKENS,
    }
    mismatches = [key for key, value in expected.items() if metrics.get(key) != value]
    if mismatches:
        raise ValueError(f"A2 diagnostic metric mismatch ({', '.join(mismatches)}): {run_dir}")


def _validate_diagnostic_payload(payload: dict[str, Any], run_dir: Path) -> None:
    expected = {
        "schema_version": 3,
        "bins": EXPECTED_BINS,
        "range_min": EXPECTED_RANGE[0],
        "range_max": EXPECTED_RANGE[1],
        "sites": list(SITES),
        "thresholds": list(THRESHOLDS),
        "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
        "validation_tokens": EXPECTED_VALIDATION_TOKENS,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    edges = payload.get("bin_edges")
    if not isinstance(edges, list) or len(edges) != EXPECTED_BINS + 1:
        mismatches.append("bin_edges")
    else:
        width = (EXPECTED_RANGE[1] - EXPECTED_RANGE[0]) / EXPECTED_BINS
        if any(
            not _close(float(value), EXPECTED_RANGE[0] + index * width)
            for index, value in enumerate(edges)
        ):
            mismatches.append("bin_edges")
    methods = payload.get("methods")
    if not isinstance(methods, list) or len(methods) != len(A2_SOURCES):
        mismatches.append("methods")
    else:
        for source, method in zip(A2_SOURCES, methods, strict=True):
            if not isinstance(method, dict) or any(
                method.get(key) != value
                for key, value in {
                    "label": source.label,
                    "config_id": source.config_id,
                    "run_id": source.run_id,
                    "batches": EXPECTED_VALIDATION_BATCHES,
                    "source_run": f"experiments/{TRANCHE_ID}/raw/{source.config_id}/{source.run_id}",
                    "source_checkpoint": f"experiments/{TRANCHE_ID}/raw/{source.config_id}/{source.run_id}/checkpoints/final",
                }.items()
            ):
                mismatches.append(f"method:{source.config_id}")
                continue
            layers = method.get("layers")
            expected_layer_names = {
                f"{site}.layer_{layer}" for site in SITES for layer in LAYERS
            }
            if not isinstance(layers, list) or {str(row.get("name")) for row in layers if isinstance(row, dict)} != expected_layer_names:
                mismatches.append(f"layers:{source.config_id}")
                continue
            for site in SITES:
                reduce_site_layers(
                    layers,
                    source=source,
                    site=site,
                    bin_count=EXPECTED_BINS,
                )
    if mismatches:
        raise ValueError(f"A2 diagnostic payload mismatch ({', '.join(mismatches)}): {run_dir}")


def _validate_layer_row(row: dict[str, Any], *, site: str, bin_count: int) -> None:
    name = str(row.get("name"))
    expected_total = EXPECTED_LAYER_TOTALS[site]
    integer_fields = ("total", "finite", "nonfinite", "in_range", "underflow", "overflow")
    for field in integer_fields:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid {field} for histogram row {name}.")
    if row["total"] != expected_total or row["finite"] + row["nonfinite"] != row["total"]:
        raise ValueError(f"Invalid full-coverage total for histogram row {name}.")
    if row["nonfinite"] != 0:
        raise ValueError(f"Nonfinite activations are unsupported for A2 figure evidence: {name}.")
    counts = row.get("counts")
    if not isinstance(counts, list) or len(counts) != bin_count or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
    ):
        raise ValueError(f"Invalid histogram counts for row {name}.")
    if sum(counts) != row["in_range"] or row["in_range"] + row["underflow"] + row["overflow"] != row["finite"]:
        raise ValueError(f"Histogram count envelope is inconsistent for row {name}.")
    raw_hits = row.get("threshold_hits")
    if not isinstance(raw_hits, dict) or set(raw_hits) != set(THRESHOLD_KEYS):
        raise ValueError(f"Histogram threshold schema mismatch for row {name}.")
    hits = tuple(raw_hits[key] for key in THRESHOLD_KEYS)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in hits) or not (0 <= hits[0] <= hits[1] <= hits[2] <= row["total"]):
        raise ValueError(f"Histogram threshold counts are invalid for row {name}.")
    fractions = row.get("threshold_fractions")
    if not isinstance(fractions, dict) or set(fractions) != set(THRESHOLD_KEYS) or any(
        not _close(float(fractions[key]), hits[index] / row["total"])
        for index, key in enumerate(THRESHOLD_KEYS)
    ):
        raise ValueError(f"Histogram threshold fractions disagree with counts: {name}.")
    rms = row.get("rms")
    if isinstance(rms, bool) or not isinstance(rms, int | float) or not math.isfinite(float(rms)) or float(rms) < 0.0:
        raise ValueError(f"Invalid RMS for histogram row {name}.")


def _site_rows(reductions: Sequence[SiteReduction], site: str) -> tuple[SiteReduction, ...]:
    rows = tuple(row for row in reductions if row.site == site)
    if tuple(row.config_id for row in rows) != tuple(source.config_id for source in A2_SOURCES):
        raise ValueError(f"A2 reduction cohort is incomplete for site {site}.")
    return rows


def _source_rows(
    reductions: Sequence[SiteReduction], source: A2Source
) -> tuple[SiteReduction, ...]:
    rows = tuple(row for row in reductions if row.config_id == source.config_id)
    if tuple(row.site for row in rows) != SITES:
        raise ValueError(f"A2 reduction site order is incomplete for {source.config_id}.")
    return rows


def _build_provenance(
    *,
    root: Path,
    stem: str,
    data: A2SpilloverData,
    staged_outputs: Sequence[Path],
    final_outputs: Sequence[Path],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "figure_id": stem,
        "evidence_level": "seed-0 directional screen",
        "claim_boundary": (
            "One seed per condition; no seed uncertainty, robustness, functional "
            "compensation, compute-reduction, or speedup claim."
        ),
        "cohort": {
            "tranche_id": TRANCHE_ID,
            "seed": 0,
            "cell_count": len(A2_SOURCES),
            "sites": list(SITES),
            "lambda_values": [source.lambda_value for source in A2_SOURCES],
            "diagnostic_config_id": DIAGNOSTIC_CONFIG_ID,
            "diagnostic_run_id": DIAGNOSTIC_RUN_ID,
            "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        },
        "reduction": {
            "threshold_fractions": "sum(layer hits) / sum(layer totals)",
            "pooled_rms": "sqrt(sum(layer finite * layer rms^2) / sum(layer finite))",
            "primary_threshold": 0.01,
            "distribution_layer": 5 if stem == DISTRIBUTION_STEM else None,
            "histogram_range": list(EXPECTED_RANGE),
            "exact_zero_atom_separate": stem == DISTRIBUTION_STEM,
        },
        "losses": [asdict(point) for point in data.losses],
        "site_reductions": [asdict(row) for row in data.reductions],
        "inputs": list(data.inputs),
        "generator": {
            "path": GENERATOR_PATH,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": [
            {
                "path": _relative(final_path, root),
                "sha256": _sha256(staged_path),
                "size_bytes": staged_path.stat().st_size,
            }
            for staged_path, final_path in zip(staged_outputs, final_outputs, strict=True)
        ],
    }


def _input_record(
    root: Path,
    path: Path,
    *,
    role: str,
    source: A2Source | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _relative(path, root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if source is not None:
        record.update({"config_id": source.config_id, "run_id": source.run_id})
    return record


def _require_files(paths: Sequence[Path], *, context: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{context} is missing required files: {', '.join(missing)}")


def _mapping(value: Any, field: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {field} to be a mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _format_percent(value: float) -> str:
    if value == 0.0:
        return "0"
    if value < 0.01:
        return f"{value:.1e}%"
    return f"{value:.1f}%"


def _close(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _repository_root(repository: str | Path | None) -> Path:
    return Path(repository).resolve() if repository is not None else Path(__file__).resolve().parents[3]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"A2 provenance path is outside the repository: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the pinned A2 spillover figure suite.")
    parser.add_argument("--repository", type=Path, default=None)
    arguments = parser.parse_args(argv)
    for output in generate_a2_spillover_suite(arguments.repository):
        print(output)


if __name__ == "__main__":
    main()
