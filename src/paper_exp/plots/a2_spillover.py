"""Deterministic A2 spillover reductions and paper-figure suite.

This module deliberately pins the accepted seed-0 A2 cohort and its post-hoc
activation-histogram and activation-propagation runs.  It does not discover
attempts or infer a latest run.  Site summaries are count-first: threshold
hits and denominators are summed before division, while pooled RMS follows the
reviewed finite-count-weighted second-moment reduction.  Logical-product
opportunities are loaded only from exact operation-level integer counters.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper_exp.config import validate_diagnostic_config, validate_training_config
from paper_exp.design import complete_config_sha256
from paper_exp.diagnostics.logical_products import LOGICAL_MATMUL_STAGES
from paper_exp.utils import read_json

from .export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    DOUBLE_COLUMN_WIDTH_INCHES,
    export_figure,
    publish_staged_outputs,
)
from .histograms import histogram_layer
from .style import PAPER_STYLE


TRANCHE_ID = "02-a2-l1-screen"
DIAGNOSTIC_CONFIG_ID = "018-a2-activation-histograms"
DIAGNOSTIC_RUN_ID = "001-20260828-082044-a031175f"
DIAGNOSTIC_GIT_COMMIT = "a0f86e057b0f67e3a2726b9cb6e352d8f8914176"
PROPAGATION_CONFIG_ID = "019-a2-activation-propagation"
PROPAGATION_RUN_ID: str | None = "001-20260828-110533-6ac813e6"
PROPAGATION_GIT_COMMIT: str | None = (
    "96621bcb73f74933f95b8b5fcd9a63ec2e15e3ff"
)
RESPONSE_STEM = "01-a2-spillover-response"
LAYERWISE_DISTRIBUTION_STEM = "02-a2-layerwise-distributions"
POOLED_DISTRIBUTION_STEM = "03-a2-site-distributions"
OBSOLETE_STEMS = ("02-a2-layer5-distributions",)
GENERATOR_PATH = "src/paper_exp/plots/a2_spillover.py"

SITES = ("h", "a", "m", "q_post", "k_post", "v")
DENSITY_SITES = ("h", "m", "a", "q_post", "k_post", "v")
ATTENTION_SITES = ("a", "q_post", "k_post", "v")
SITE_LABELS = {
    "h": r"$h$",
    "a": r"$a$",
    "m": r"$m$",
    "q_post": r"$q_{post}$",
    "k_post": r"$k_{post}$",
    "v": r"$v$",
}
LAYERS = tuple(range(6))
LAYERWISE_SOURCE_INDICES = (0, 3)
POOLED_SOURCE_INDICES = (0, 4, 5)
DENSITY_REBIN_FACTOR = 16
DENSITY_WINDOWS = {
    "h": (0.0, 8.0),
    "m": (-4.0, 4.0),
    "a": (-4.0, 4.0),
    "q_post": (-16.0, 16.0),
    "k_post": (-16.0, 16.0),
    "v": (-4.0, 4.0),
}
DENSITY_STYLE_BY_SOURCE = {
    0: ("#4D4D4D", "-"),
    3: ("#009E73", ":"),
    4: ("#0072B2", "--"),
    5: ("#D55E00", "-."),
}
LAYERWISE_LABELS = ("ReLU control", r"L1 $\lambda=1$")
POOLED_LABELS = ("Control", r"L1 $\lambda=2$", r"L1 $\lambda=5$")
THRESHOLDS = (0.0, 0.01, 0.1)
THRESHOLD_KEYS = ("0", "0.01", "0.1")
EXPECTED_BINS = 3_200
EXPECTED_RANGE = (-16.0, 16.0)
EXPECTED_VALIDATION_SEQUENCES = 152
EXPECTED_VALIDATION_TOKENS = 311_296
EXPECTED_VALIDATION_BATCHES = 38
EXPECTED_VALIDATION_CACHE_TOKENS = 311_739
EXPECTED_TRAILING_VALIDATION_TOKENS = 443
EXPECTED_BLOCK_SIZE = 2_048
EXPECTED_PROPAGATION_EXECUTION = {
    "requested_device": "cuda",
    "requested_precision": "bfloat16",
    "resolved_device": "cuda",
    "resolved_precision": "bfloat16",
}
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
class GroupReduction:
    """Count-first near-zero summary over an explicit set of measured sites."""

    config_id: str
    label: str
    lambda_value: float
    sites: tuple[str, ...]
    total: int
    near_zero_0p01_hits: int
    near_zero_0p01_fraction: float


@dataclass(frozen=True)
class LogicalOpportunityPoint:
    """Exact operation-level logical-product opportunity for one A2 cell."""

    config_id: str
    run_id: str
    label: str
    lambda_value: float
    block_zero_product_count: int
    block_product_count: int
    lm_head_product_count: int
    model_product_count: int
    R_block: float
    R_model: float


@dataclass(frozen=True)
class DensityReduction:
    """Exact rebinned density with zero and omitted mass kept explicit."""

    edges: tuple[float, ...]
    density: tuple[float, ...]
    total: int
    exact_zero_hits: int
    nonzero_total: int
    outside_stored_hits: int
    outside_display_hits: int
    exact_zero_fraction: float
    outside_display_fraction_nonzero: float | None


@dataclass(frozen=True)
class A2SpilloverData:
    """Validated fixed-cohort data ready for rendering."""

    reductions: tuple[SiteReduction, ...]
    losses: tuple[LossPoint, ...]
    bin_edges: tuple[float, ...]
    methods: tuple[dict[str, Any], ...]
    logical_opportunities: tuple[LogicalOpportunityPoint, ...]
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


def reduce_site_group(
    reductions: Sequence[SiteReduction],
    *,
    source: A2Source,
    sites: Sequence[str],
) -> GroupReduction:
    """Pool near-zero hits count-first over an explicit measured-site set."""

    site_tuple = tuple(sites)
    if not site_tuple or len(set(site_tuple)) != len(site_tuple):
        raise ValueError("Grouped A2 sites must be nonempty and unique.")
    unsupported = tuple(site for site in site_tuple if site not in SITES)
    if unsupported:
        raise ValueError(f"Unsupported grouped A2 sites: {unsupported!r}.")
    source_rows = {row.site: row for row in _source_rows(reductions, source)}
    selected = tuple(source_rows[site] for site in site_tuple)
    total = sum(row.total for row in selected)
    hits = sum(row.near_zero_0p01_hits for row in selected)
    if total <= 0:
        raise ValueError(f"Grouped A2 sites have no observations: {source.config_id}.")
    return GroupReduction(
        config_id=source.config_id,
        label=source.label,
        lambda_value=source.lambda_value,
        sites=site_tuple,
        total=total,
        near_zero_0p01_hits=hits,
        near_zero_0p01_fraction=hits / total,
    )


def reduce_density_layers(
    layers: Sequence[dict[str, Any]],
    edges: Sequence[float],
    *,
    display_window: tuple[float, float],
    rebin_factor: int = DENSITY_REBIN_FACTOR,
) -> DensityReduction:
    """Pool integer histograms, remove the zero atom, then rebin exactly."""

    numeric_edges = tuple(float(value) for value in edges)
    if len(numeric_edges) < 2 or any(
        not math.isfinite(value) for value in numeric_edges
    ) or any(
        right <= left
        for left, right in zip(numeric_edges[:-1], numeric_edges[1:], strict=True)
    ):
        raise ValueError("Density edges must be finite and strictly increasing.")
    if (
        isinstance(rebin_factor, bool)
        or not isinstance(rebin_factor, int)
        or rebin_factor <= 0
    ):
        raise ValueError("Density rebin factor must be a positive integer.")
    bin_count = len(numeric_edges) - 1
    if bin_count % rebin_factor:
        raise ValueError("Density bin count must be divisible by the rebin factor.")

    counts = [0] * bin_count
    total = 0
    exact_zero_hits = 0
    outside_stored_hits = 0
    if not layers:
        raise ValueError("Density reduction requires at least one histogram layer.")
    for layer in layers:
        raw_counts = layer.get("counts")
        if not isinstance(raw_counts, list) or len(raw_counts) != bin_count or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_counts
        ):
            raise ValueError("Density counts must be nonnegative integers matching the edges.")
        raw_total = layer.get("total")
        underflow = layer.get("underflow")
        overflow = layer.get("overflow")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (raw_total, underflow, overflow)
        ):
            raise ValueError("Density totals and tails must be nonnegative integers.")
        if sum(raw_counts) + underflow + overflow != raw_total:
            raise ValueError("Density stored counts and tails must exactly cover the total.")
        threshold_hits = layer.get("threshold_hits")
        raw_zero = threshold_hits.get("0") if isinstance(threshold_hits, dict) else None
        if (
            isinstance(raw_zero, bool)
            or not isinstance(raw_zero, int)
            or raw_zero < 0
            or raw_zero > raw_total
        ):
            raise ValueError("Density exact-zero hits must be valid integer counts.")
        counts = [left + right for left, right in zip(counts, raw_counts, strict=True)]
        total += raw_total
        exact_zero_hits += raw_zero
        outside_stored_hits += underflow + overflow

    zero_bin = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(numeric_edges[:-1], numeric_edges[1:], strict=True)
            )
            if left <= 0.0 < right
            or (index == bin_count - 1 and right == 0.0)
        ),
        None,
    )
    if exact_zero_hits:
        if zero_bin is None:
            raise ValueError("Density range excludes zero despite an exact-zero atom.")
        if counts[zero_bin] < exact_zero_hits:
            raise ValueError("Density exact-zero atom exceeds its containing bin.")
        counts[zero_bin] -= exact_zero_hits

    nonzero_total = total - exact_zero_hits
    if sum(counts) + outside_stored_hits != nonzero_total:
        raise ValueError("Density nonzero mass is not conserved before rebinning.")
    rebinned_counts = tuple(
        sum(counts[index : index + rebin_factor])
        for index in range(0, bin_count, rebin_factor)
    )
    rebinned_edges = tuple(numeric_edges[::rebin_factor])
    if len(rebinned_edges) != len(rebinned_counts) + 1:
        rebinned_edges = (*rebinned_edges, numeric_edges[-1])
    lower, upper = (float(value) for value in display_window)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Density display window must be finite and increasing.")
    start = _matching_edge_index(rebinned_edges, lower)
    stop = _matching_edge_index(rebinned_edges, upper)
    if start is None or stop is None or start >= stop:
        raise ValueError("Density display window must align with rebinned edges.")
    displayed_counts = rebinned_counts[start:stop]
    displayed_edges = rebinned_edges[start : stop + 1]
    outside_display_hits = (
        outside_stored_hits
        + sum(rebinned_counts[:start])
        + sum(rebinned_counts[stop:])
    )
    density = tuple(
        count / nonzero_total / (right - left) if nonzero_total else 0.0
        for count, left, right in zip(
            displayed_counts,
            displayed_edges[:-1],
            displayed_edges[1:],
            strict=True,
        )
    )
    if sum(displayed_counts) + outside_display_hits != nonzero_total:
        raise ValueError("Density nonzero mass is not conserved after display cropping.")
    return DensityReduction(
        edges=displayed_edges,
        density=density,
        total=total,
        exact_zero_hits=exact_zero_hits,
        nonzero_total=nonzero_total,
        outside_stored_hits=outside_stored_hits,
        outside_display_hits=outside_display_hits,
        exact_zero_fraction=exact_zero_hits / total,
        outside_display_fraction_nonzero=(
            outside_display_hits / nonzero_total if nonzero_total else None
        ),
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
    logical_opportunities, propagation_inputs = _load_propagation_diagnostic(root)
    inputs.extend(propagation_inputs)
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
        logical_opportunities=logical_opportunities,
        inputs=tuple(inputs),
    )


def _density_reduction(
    data: A2SpilloverData,
    *,
    source_index: int,
    site: str,
    layers: Sequence[int],
) -> DensityReduction:
    selected = tuple(
        histogram_layer(data.methods[source_index], f"{site}.layer_{layer}")
        for layer in layers
    )
    return reduce_density_layers(
        selected,
        data.bin_edges,
        display_window=DENSITY_WINDOWS[site],
        rebin_factor=DENSITY_REBIN_FACTOR,
    )


def _density_column_limits(
    data: A2SpilloverData,
    *,
    source_indices: Sequence[int],
    pooled: bool,
) -> dict[str, float]:
    limits: dict[str, float] = {}
    layer_groups: tuple[tuple[int, ...], ...] = (
        (LAYERS,) if pooled else tuple((layer,) for layer in LAYERS)
    )
    for site in DENSITY_SITES:
        maximum = 0.0
        for source_index in source_indices:
            for layers in layer_groups:
                curve = _density_reduction(
                    data,
                    source_index=source_index,
                    site=site,
                    layers=layers,
                )
                maximum = max(maximum, *curve.density)
        limits[site] = 1.08 * maximum if maximum > 0.0 else 1.0
    return limits


def build_a2_spillover_response_figure(data: A2SpilloverData) -> Figure:
    """Plot m-site against measured-attention near-zero response."""

    m_rows = _site_rows(data.reductions, "m")
    attention_rows = tuple(
        reduce_site_group(data.reductions, source=source, sites=ATTENTION_SITES)
        for source in A2_SOURCES
    )
    m_control = m_rows[0].near_zero_0p01_fraction
    attention_control = attention_rows[0].near_zero_0p01_fraction
    points = tuple(
        (
            100.0 * (m_rows[index].near_zero_0p01_fraction - m_control),
            100.0
            * (attention_rows[index].near_zero_0p01_fraction - attention_control),
            A2_SOURCES[index],
        )
        for index in range(1, len(A2_SOURCES))
    )
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 3.60))
    axis.axhline(0.0, color="#888888", linewidth=0.8, zorder=0)
    axis.axvline(0.0, color="#888888", linewidth=0.8, zorder=0)
    axis.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=48,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    offsets = ((8, -12), (8, 6), (8, 6), (-5, 10), (-46, 10))
    for (x_value, y_value, source), offset in zip(points, offsets, strict=True):
        axis.annotate(
            rf"$\lambda={source.lambda_value:g}$",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(-2.5, 33.2)
    axis.set_ylim(-0.9, 7.45)
    axis.set_xticks((-2.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0))
    axis.set_yticks((-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    axis.set_xlabel(r"$m$ near-zero response, $\Delta n_m(0.01)$ (pp)")
    axis.set_ylabel(
        "Attention near-zero response,\n"
        r"$\Delta n_A(0.01)$ (pp)"
    )
    axis.grid(False)
    axis.xaxis.grid(True, alpha=0.18)
    axis.yaxis.grid(True, alpha=0.18)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.155, right=0.98, top=0.96, bottom=0.20)
    return figure


def build_a2_layerwise_distributions_figure(data: A2SpilloverData) -> Figure:
    """Plot control/lambda-1 densities for every layer and site."""

    figure, axes = plt.subplots(
        len(LAYERS),
        len(DENSITY_SITES),
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 8.65),
        sharex="col",
        sharey="col",
        squeeze=False,
    )
    limits = _density_column_limits(
        data,
        source_indices=LAYERWISE_SOURCE_INDICES,
        pooled=False,
    )
    for column, site in enumerate(DENSITY_SITES):
        title = SITE_LABELS[site] + (r"$^{\dagger}$" if site == "k_post" else "")
        axes[0, column].set_title(title, pad=4)
        for row, layer in enumerate(LAYERS):
            axis = axes[row, column]
            curves = tuple(
                _density_reduction(
                    data,
                    source_index=source_index,
                    site=site,
                    layers=(layer,),
                )
                for source_index in LAYERWISE_SOURCE_INDICES
            )
            for source_index, curve in zip(
                LAYERWISE_SOURCE_INDICES,
                curves,
                strict=True,
            ):
                if curve.nonzero_total == 0:
                    continue
                color, linestyle = DENSITY_STYLE_BY_SOURCE[source_index]
                axis.stairs(
                    curve.density,
                    curve.edges,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.20,
                )
            if site == "h":
                values = " / ".join(
                    _format_compact_percentage(curve.exact_zero_fraction)
                    for curve in curves
                )
                axis.text(
                    0.98,
                    0.95,
                    r"$P_0$ (%)" + "\n" + values,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 1.0,
                    },
                )
            if site == "k_post":
                values = " / ".join(
                    _format_compact_percentage(curve.outside_display_fraction_nonzero)
                    for curve in curves
                )
                axis.text(
                    0.98,
                    0.95,
                    "Out (%)\n" + values,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 1.0,
                    },
                )
            axis.set_xlim(DENSITY_WINDOWS[site])
            axis.set_ylim(0.0, limits[site])
            axis.grid(False)
            axis.yaxis.grid(True, alpha=0.14)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            axis.tick_params(labelbottom=row == len(LAYERS) - 1, labelleft=row == 0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column == 0:
                axis.set_ylabel(f"Layer {layer}", labelpad=8)
    handles = tuple(
        Line2D(
            (0,),
            (0,),
            color=DENSITY_STYLE_BY_SOURCE[source_index][0],
            linestyle=DENSITY_STYLE_BY_SOURCE[source_index][1],
            linewidth=1.4,
            label=LAYERWISE_LABELS[index],
        )
        for index, source_index in enumerate(LAYERWISE_SOURCE_INDICES)
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.998),
    )
    figure.text(
        0.012,
        0.50,
        r"Conditional density ($x\ne0$)",
        ha="center",
        va="center",
        rotation="vertical",
    )
    figure.supxlabel("Activation value", y=0.012)
    figure.subplots_adjust(
        left=0.115,
        right=0.975,
        top=0.935,
        bottom=0.075,
        wspace=0.44,
        hspace=0.23,
    )
    return figure


def build_a2_site_distributions_figure(data: A2SpilloverData) -> Figure:
    """Plot within-site, across-layer pooled densities for three conditions."""

    figure, axes = plt.subplots(
        len(POOLED_SOURCE_INDICES),
        len(DENSITY_SITES),
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 5.15),
        sharex="col",
        sharey="col",
        squeeze=False,
    )
    limits = _density_column_limits(
        data,
        source_indices=POOLED_SOURCE_INDICES,
        pooled=True,
    )
    row_labels = POOLED_LABELS
    for column, site in enumerate(DENSITY_SITES):
        title = SITE_LABELS[site] + (r"$^{\dagger}$" if site == "k_post" else "")
        axes[0, column].set_title(title, pad=4)
        for row, source_index in enumerate(POOLED_SOURCE_INDICES):
            axis = axes[row, column]
            color, _ = DENSITY_STYLE_BY_SOURCE[source_index]
            curve = _density_reduction(
                data,
                source_index=source_index,
                site=site,
                layers=LAYERS,
            )
            if curve.nonzero_total:
                axis.stairs(
                    curve.density,
                    curve.edges,
                    fill=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.12,
                )
                axis.stairs(
                    curve.density,
                    curve.edges,
                    color=color,
                    linewidth=1.25,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "All mass at x=0",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            annotation = None
            if site == "h":
                annotation = "$P_0$ " + _format_probability(
                    curve.exact_zero_fraction
                )
            elif site == "k_post":
                annotation = "Out " + _format_probability(
                    curve.outside_display_fraction_nonzero
                )
            if annotation is not None:
                axis.text(
                    0.93,
                    0.94,
                    annotation,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.80,
                        "pad": 1.0,
                    },
                )
            axis.set_xlim(DENSITY_WINDOWS[site])
            axis.set_ylim(0.0, limits[site])
            axis.grid(False)
            axis.yaxis.grid(True, alpha=0.14)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            axis.tick_params(
                labelbottom=row == len(POOLED_SOURCE_INDICES) - 1,
                labelleft=row == 0,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column == 0:
                axis.set_ylabel(
                    row_labels[row],
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=5,
                )
    figure.text(
        0.012,
        0.50,
        r"Conditional density ($x\ne0$)",
        ha="center",
        va="center",
        rotation="vertical",
    )
    figure.supxlabel("Activation value", y=0.018)
    figure.subplots_adjust(
        left=0.155,
        right=0.975,
        top=0.92,
        bottom=0.11,
        wspace=0.44,
        hspace=0.25,
    )
    return figure


def build_a2_response_markdown(data: A2SpilloverData) -> str:
    """Build the spillover-plane caption and compact measured-results table."""

    controls = {row.site: row for row in _source_rows(data.reductions, A2_SOURCES[0])}
    attention = tuple(
        reduce_site_group(data.reductions, source=source, sites=ATTENTION_SITES)
        for source in A2_SOURCES
    )
    attention_control = attention[0]
    opportunities = _logical_opportunity_rows(data.logical_opportunities)
    opportunity_control = opportunities[0]
    control_loss = data.losses[0].final_validation_loss
    lines = [
        "# A2 spillover response",
        "",
        (
            "**Figure caption.** Near-zero response outside the pressured `h` site "
            "for seed-0 Pythia-14M. Each point is one positive L1 coefficient. "
            "Both axes are percentage-point changes from the matched ReLU control. "
            "The measured-attention aggregate is count-first over "
            "`A = {a, q_post, k_post, v}` and all six layers: "
            "`n_A(0.01) = sum(hits) / sum(total)`."
        ),
        "",
        (
            "Across the tested coefficients, lambda <= 1 increases attention-site "
            "near-zero mass while leaving `m` near its control value. Lambda 2 and "
            "5 instead produce a large, layer-local near-zero response at `m`, "
            "while the pooled attention response returns near control. These are "
            "descriptive response ranges, not a causal compensation or phase-boundary "
            "claim."
        ),
        "",
        "## Quality and activation response",
        "",
        (
            "All `n(0.01)` values are percentages; deltas are percentage points. "
            "RMS is pooled from finite-count-weighted second moments across layers. "
            "`R_block` and `R_model` are exact-zero logical-product opportunities, "
            "not removed FLOPs or measured speedups."
        ),
        "",
        (
            "| Condition | Final selection val. loss | Delta loss | R_block (%) | "
            "R_model (%) | Delta R_model (pp) | n_h | Delta n_h | n_m | "
            "Delta n_m | n_A | Delta n_A | RMS h | RMS m |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for source, loss, attention_row, opportunity in zip(
        A2_SOURCES, data.losses, attention, opportunities, strict=True
    ):
        source_rows = {row.site: row for row in _source_rows(data.reductions, source)}
        h_row = source_rows["h"]
        m_row = source_rows["m"]
        lines.append(
            f"| {_condition_label(source)} | {loss.final_validation_loss:.6f} | "
            f"{loss.final_validation_loss - control_loss:+.6f} | "
            f"{100.0 * opportunity.R_block:.6f} | "
            f"{100.0 * opportunity.R_model:.6f} | "
            f"{100.0 * (opportunity.R_model - opportunity_control.R_model):+.6f} | "
            f"{100.0 * h_row.near_zero_0p01_fraction:.4f} | "
            f"{100.0 * (h_row.near_zero_0p01_fraction - controls['h'].near_zero_0p01_fraction):+.4f} | "
            f"{100.0 * m_row.near_zero_0p01_fraction:.4f} | "
            f"{100.0 * (m_row.near_zero_0p01_fraction - controls['m'].near_zero_0p01_fraction):+.4f} | "
            f"{100.0 * attention_row.near_zero_0p01_fraction:.4f} | "
            f"{100.0 * (attention_row.near_zero_0p01_fraction - attention_control.near_zero_0p01_fraction):+.4f} | "
            f"{h_row.pooled_rms:.5f} | {m_row.pooled_rms:.5f} |"
        )
    r_model_values = tuple(point.R_model for point in opportunities[1:])
    r_model_deltas = tuple(
        value - opportunity_control.R_model for value in r_model_values
    )
    response_shape = (
        "non-monotonic"
        if any(
            left > right
            for left, right in zip(r_model_values, r_model_values[1:])
        )
        else "monotonic non-decreasing"
    )
    sign_groups = (
        (
            "below",
            tuple(
                source
                for source, delta in zip(
                    A2_SOURCES[1:], r_model_deltas, strict=True
                )
                if delta < 0
            ),
        ),
        (
            "equal to",
            tuple(
                source
                for source, delta in zip(
                    A2_SOURCES[1:], r_model_deltas, strict=True
                )
                if delta == 0
            ),
        ),
        (
            "above",
            tuple(
                source
                for source, delta in zip(
                    A2_SOURCES[1:], r_model_deltas, strict=True
                )
                if delta > 0
            ),
        ),
    )
    sign_summary = "; ".join(
        f"{_lambda_values(sources)} "
        f"{'is' if len(sources) == 1 else 'are'} {relation} control"
        for relation, sources in sign_groups
        if sources
    )
    maximum_index = max(
        range(1, len(opportunities)), key=lambda index: opportunities[index].R_model
    )
    maximum_source = A2_SOURCES[maximum_index]
    maximum_opportunity = opportunities[maximum_index]
    maximum_loss_delta = data.losses[maximum_index].final_validation_loss - control_loss
    all_l1_losses_worse = all(
        loss.final_validation_loss > control_loss for loss in data.losses[1:]
    )
    lines.extend(
        [
            "",
            (
                "`R_block` counts exact-zero operand products across QKV, valid-causal "
                "QK and PV, Wo, W1, and W2. `R_model` keeps the same numerator and "
                "adds the dense LM-head products to the denominator. Diagnostic `019` "
                f"used BF16 eager attention at `T = {EXPECTED_BLOCK_SIZE:,}` over "
                f"{EXPECTED_VALIDATION_SEQUENCES} complete validation blocks "
                f"({EXPECTED_VALIDATION_TOKENS:,} tokens); "
                f"{EXPECTED_TRAILING_VALIDATION_TOKENS} trailing tokens from the "
                f"{EXPECTED_VALIDATION_CACHE_TOKENS:,}-token cache were excluded."
            ),
            "",
            (
                f"The seed-0 `R_model` response is {response_shape}: {sign_summary}. "
                f"The largest observed value among the tested L1 coefficients is "
                f"lambda {maximum_source.lambda_value:g} at "
                f"{100.0 * maximum_opportunity.R_model:.6f}% "
                f"({100.0 * (maximum_opportunity.R_model - opportunity_control.R_model):+.6f} pp), "
                f"paired with a {maximum_loss_delta:+.6f} validation-loss change. "
                + (
                    "Every L1 cell has higher final validation loss than control."
                    if all_l1_losses_worse
                    else (
                        "Not every L1 cell has higher final validation loss than "
                        "control."
                    )
                )
                + (
                    " This is descriptive single-seed evidence, not a speedup or "
                    "compute-reduction claim."
                )
            ),
            "",
            "## Complete per-site activation response",
            "",
            (
                "For every site, `n_s(epsilon) = #(|x| <= epsilon) / #elements`; "
                "`z_s = #(x = 0) / #elements`. Delta values are relative to the "
                "matched ReLU control. `a` is the attention-branch LayerNorm output "
                "feeding fused W_QKV; `m` is the MLP-branch LayerNorm output feeding "
                "W1; `h` is the post-activation MLP hidden state feeding W2; "
                "`q_post` and `k_post` are the post-RoPE Q/K operands of QK^T; and "
                "`v` is the V operand of PV."
            ),
            "",
            "| Condition | Site | z_s (%) | n_s(0.01) (%) | Delta n_s(0.01) (pp) | n_s(0.1) (%) | RMS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in A2_SOURCES:
        source_rows = {row.site: row for row in _source_rows(data.reductions, source)}
        for site in DENSITY_SITES:
            row = source_rows[site]
            control = controls[site]
            lines.append(
                f"| {_condition_label(source)} | `{site}` | "
                f"{_format_precise_percentage(row.exact_zero_fraction)} | "
                f"{100.0 * row.near_zero_0p01_fraction:.4f} | "
                f"{100.0 * (row.near_zero_0p01_fraction - control.near_zero_0p01_fraction):+.4f} | "
                f"{100.0 * row.near_zero_0p1_fraction:.4f} | "
                f"{row.pooled_rms:.5f} |"
            )
    lines.extend(
        [
            "",
            "## Localization of the high-lambda m response",
            "",
            (
                "The pooled lambda-2/lambda-5 `m` response is driven by the three "
                "observed layer cells below. Their exact-zero mass remains zero: "
                "this is near-zero collapse at the measured `m` port, not exact "
                "zeros or whole-model collapse. In the same cells, the pressured "
                "`h` port has 100% exact-zero mass; this co-location is descriptive "
                "and does not establish causality."
            ),
            "",
            "| Condition | Layer | z_h (%) | z_m (%) | n_m(0.01) (%) | RMS m |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source_index, layer in ((4, 4), (5, 0), (5, 1)):
        source = A2_SOURCES[source_index]
        row = histogram_layer(data.methods[source_index], f"m.layer_{layer}")
        _validate_layer_row(
            row,
            site="m",
            bin_count=EXPECTED_BINS,
        )
        h_row = histogram_layer(data.methods[source_index], f"h.layer_{layer}")
        _validate_layer_row(
            h_row,
            site="h",
            bin_count=EXPECTED_BINS,
        )
        total = int(row["total"])
        hits = row["threshold_hits"]
        h_total = int(h_row["total"])
        h_hits = h_row["threshold_hits"]
        lines.append(
            f"| {_condition_label(source)} | {layer} | "
            f"{100.0 * int(h_hits['0']) / h_total:.4f} | "
            f"{100.0 * int(hits['0']) / total:.4f} | "
            f"{100.0 * int(hits['0.01']) / total:.4f} | "
            f"{float(row['rms']):.3e} |"
        )
    lines.extend(
        [
            "",
            (
                "Logical opportunities use the complete selection partition and "
                "operation-level integer counts from diagnostic `019`, including "
                "actual post-RoPE Q/K operands and valid causal QK/PV pairs. The "
                "near-zero `m` response is not counted unless an operand is exactly "
                "zero; activation near-zero fractions must not be relabeled as "
                "logical-product opportunity."
            ),
            "",
            (
                "This is a single-seed directional screen (`n = 1` per condition). "
                "It does not estimate seed uncertainty or support robustness, "
                "compute-reduction, or runtime-speedup claims."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_a2_layerwise_distribution_markdown(data: A2SpilloverData) -> str:
    """Build the layerwise-atlas caption and material atom/tail disclosures."""

    lines = [
        "# A2 layerwise activation distributions",
        "",
        (
            "**Figure caption.** Layer-resolved activation densities for the ReLU "
            "control and h-only L1 at lambda 1. This is the predeclared A2 "
            "distribution comparison; all six layers are shown, including the "
            "required deepest-layer row. "
            "Each panel is a density conditional on `x != 0`; exact-zero atoms are "
            "removed from every density before adjacent-bin rebinning. Material `h` "
            "atoms are printed in-panel; all-site values are tabulated in the "
            "spillover-response companion and recorded exactly in provenance. Curves "
            "use exact count-preserving 0.16-wide bins, linear density, and the same "
            "x and y scales within each site column. No KDE or interpolation is used."
        ),
        "In-panel pairs follow legend order: ReLU control / L1 lambda 1.",
        "",
        (
            "Displayed windows are `h: [0, 8]`, `m/a/v: [-4, 4]`, and "
            "`q_post/k_post: [-16, 16]`. Out-of-window mass remains in the "
            "conditional-density denominator. The dagger on `k_post` marks material "
            "stored-range tails; reported values are below."
        ),
        "",
        "## Exact-zero mass at h",
        "",
        "| Layer | Control | lambda 1 |",
        "| ---: | ---: | ---: |",
    ]
    for layer in LAYERS:
        curves = tuple(
            _density_reduction(
                data,
                source_index=source_index,
                site="h",
                layers=(layer,),
            )
            for source_index in LAYERWISE_SOURCE_INDICES
        )
        lines.append(
            f"| {layer} | "
            + " | ".join(f"{curve.exact_zero_fraction:.6%}" for curve in curves)
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "No all-atom `h` cell occurs in this control-versus-lambda-1 "
                "comparison."
            ),
            "",
            "## k_post mass outside the stored range, conditional on x != 0",
            "",
            "| Layer | Control | lambda 1 |",
            "| ---: | ---: | ---: |",
        ]
    )
    for layer in LAYERS:
        curves = tuple(
            _density_reduction(
                data,
                source_index=source_index,
                site="k_post",
                layers=(layer,),
            )
            for source_index in LAYERWISE_SOURCE_INDICES
        )
        lines.append(
            f"| {layer} | "
            + " | ".join(
                _format_probability(curve.outside_display_fraction_nonzero)
                for curve in curves
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Maximum omitted conditional mass by site",
            "",
            "| Site | Maximum | Condition / layer |",
            "| --- | ---: | --- |",
        ]
    )
    for site in DENSITY_SITES:
        candidates = []
        for source_index in LAYERWISE_SOURCE_INDICES:
            for layer in LAYERS:
                curve = _density_reduction(
                    data,
                    source_index=source_index,
                    site=site,
                    layers=(layer,),
                )
                if curve.outside_display_fraction_nonzero is not None:
                    candidates.append(
                        (curve.outside_display_fraction_nonzero, source_index, layer)
                    )
        maximum, source_index, layer = max(candidates)
        lines.append(
            f"| `{site}` | {_format_probability(maximum)} | "
            f"{_condition_label(A2_SOURCES[source_index])} / layer {layer} |"
        )
    lines.extend(
        [
            "",
            (
                "The control-versus-lambda-1 atlas is the predeclared distribution "
                "view. The separate pooled control/lambda-2/lambda-5 figure shows "
                "the larger tested-coefficient shapes; the response companion "
                "quantifies their layer-local `m` near-zero collapse."
            ),
            "",
            (
                "This is seed-0 descriptive evidence (`n = 1` per condition), not "
                "a uniform-across-layer, causal, compute, or speedup claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_a2_site_distribution_markdown(data: A2SpilloverData) -> str:
    """Build the across-layer pooled-site caption and mass table."""

    lines = [
        "# A2 across-layer pooled site distributions",
        "",
        (
            "**Figure caption.** Per-site activation distributions after pooling "
            "integer histogram counts across all six layers within each site. Rows "
            "are the ReLU control, h-only L1 lambda 2, and h-only L1 lambda 5. This "
            "high-coefficient comparison is a post-hoc descriptive view selected "
            "after inspecting A2; it does not replace the predeclared control-versus-"
            "lambda-1 deepest-layer comparison. "
            "Densities are conditional on `x != 0`; material exact-zero mass at "
            "`h` and omitted tail mass at `k_post` are printed in-panel. The table "
            "reports both quantities for every condition and site."
        ),
        "",
        (
            "The reduction, rebinning, and windows match the layerwise atlas. "
            "Within this figure, x and y scales are shared down each site column. "
            "Counts are never pooled across sites. The dagger on `k_post` flags "
            "stored-range tails."
        ),
        "",
        "| Condition | Site | P(x = 0) | Outside window given x != 0 |",
        "| --- | --- | ---: | ---: |",
    ]
    for source_index in POOLED_SOURCE_INDICES:
        source = A2_SOURCES[source_index]
        for site in DENSITY_SITES:
            curve = _density_reduction(
                data,
                source_index=source_index,
                site=site,
                layers=LAYERS,
            )
            lines.append(
                f"| {_condition_label(source)} | `{site}` | "
                f"{_format_probability(curve.exact_zero_fraction)} | "
                f"{_format_probability(curve.outside_display_fraction_nonzero)} |"
            )
    lines.extend(
        [
            "",
            (
                "This aggregation is element-weighted within a site and remains a "
                "seed-0 descriptive view. It is not a logical-product or runtime "
                "measurement."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate_a2_spillover_suite(
    repository: str | Path | None = None,
) -> tuple[Path, ...]:
    """Validate inputs and atomically publish the three A2 figure packages."""

    root = _repository_root(repository)
    data = load_a2_spillover(root)
    figures_dir = root / "experiments" / TRANCHE_ID / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stems = (
        RESPONSE_STEM,
        LAYERWISE_DISTRIBUTION_STEM,
        POOLED_DISTRIBUTION_STEM,
    )
    final_paths = tuple(
        figures_dir / f"{stem}.{suffix}"
        for stem in stems
        for suffix in ("pdf", "png", "md", "provenance.json")
    )
    with tempfile.TemporaryDirectory(prefix=".a2-spillover.", dir=figures_dir) as temp:
        staging = Path(temp)
        staged = {path: staging / path.name for path in final_paths}
        builders = {
            RESPONSE_STEM: build_a2_spillover_response_figure,
            LAYERWISE_DISTRIBUTION_STEM: build_a2_layerwise_distributions_figure,
            POOLED_DISTRIBUTION_STEM: build_a2_site_distributions_figure,
        }
        markdown_builders = {
            RESPONSE_STEM: build_a2_response_markdown,
            LAYERWISE_DISTRIBUTION_STEM: build_a2_layerwise_distribution_markdown,
            POOLED_DISTRIBUTION_STEM: build_a2_site_distribution_markdown,
        }
        for stem in stems:
            export_figure(
                lambda stem=stem: builders[stem](data),
                staged[figures_dir / f"{stem}.pdf"],
                save_png=True,
                style=PAPER_STYLE,
                profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
            )
            _write_text(
                staged[figures_dir / f"{stem}.md"],
                markdown_builders[stem](data),
            )
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
    for obsolete_stem in OBSOLETE_STEMS:
        for suffix in ("pdf", "png", "md", "provenance.json"):
            obsolete = figures_dir / f"{obsolete_stem}.{suffix}"
            if obsolete.is_file():
                obsolete.unlink()
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


def _load_propagation_diagnostic(
    root: Path,
) -> tuple[tuple[LogicalOpportunityPoint, ...], tuple[dict[str, Any], ...]]:
    run_id, git_commit = _pinned_propagation_identity()
    recipe_path = (
        root
        / "experiments"
        / TRANCHE_ID
        / "run"
        / f"{PROPAGATION_CONFIG_ID}.yaml"
    )
    run_dir = (
        root
        / "experiments"
        / TRANCHE_ID
        / "raw"
        / PROPAGATION_CONFIG_ID
        / run_id
    )
    paths = {
        "propagation_recipe": recipe_path,
        "propagation_saved_config": run_dir / "config.yaml",
        "propagation_manifest": run_dir / "manifest.json",
        "propagation_metrics": run_dir / "metrics.json",
        "activation_propagation": run_dir / "activation_propagation.json",
    }
    _require_files(paths.values(), context="A2 activation-propagation diagnostic")
    recipe = _load_yaml(paths["propagation_recipe"])
    saved = _load_yaml(paths["propagation_saved_config"])
    if saved != recipe:
        raise ValueError(
            f"Saved propagation config does not match tracked recipe: {run_dir}"
        )
    validate_diagnostic_config(recipe, "activation_propagation")
    _validate_propagation_config(recipe, run_dir)
    manifest = _load_json(paths["propagation_manifest"])
    metrics = _load_json(paths["propagation_metrics"])
    payload = _load_json(paths["activation_propagation"])
    _validate_propagation_manifest(
        manifest,
        run_dir,
        run_id=run_id,
        git_commit=git_commit,
    )
    points = _validate_propagation_payload(payload, run_dir)
    _validate_propagation_metrics(metrics, run_dir, points=points)
    return points, tuple(
        _input_record(root, path, role=role, source=None)
        for role, path in paths.items()
    )


def _pinned_propagation_identity() -> tuple[str, str]:
    run_id = PROPAGATION_RUN_ID
    git_commit = PROPAGATION_GIT_COMMIT
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(
            "A2 activation-propagation run 019 has not been pinned; set "
            "PROPAGATION_RUN_ID only after accepting one exact completed attempt."
        )
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise RuntimeError(
            "A2 activation-propagation Git identity has not been pinned to one "
            "40-character lowercase commit."
        )
    return run_id, git_commit


def _expected_selected_runs() -> list[dict[str, Any]]:
    return [
        {
            "label": source.label,
            "tranche_id": TRANCHE_ID,
            "config_id": source.config_id,
            "run_id": source.run_id,
        }
        for source in A2_SOURCES
    ]


def _expected_source_paths() -> tuple[list[str], list[str]]:
    runs = [
        f"experiments/{TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
        for source in A2_SOURCES
    ]
    return runs, [f"{path}/checkpoints/final" for path in runs]


def _validate_propagation_config(config: dict[str, Any], run_dir: Path) -> None:
    diagnostic = _mapping(
        config.get("activation_propagation"), "activation_propagation", run_dir
    )
    validation = _mapping(config.get("validation"), "validation", run_dir)
    checks = {
        "selected runs": diagnostic.get("selected_runs") == _expected_selected_runs(),
        "partition": (
            validation.get("partition") == "selection"
            and validation.get("partition_hash") == EXPECTED_SELECTION_HASH
        ),
        "coverage": (
            validation.get("eval_batches") is None
            and validation.get("batch_size") == 4
        ),
        "seed": (config.get("run") or {}).get("seed") == 0,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(
            f"A2 propagation recipe mismatch ({', '.join(failed)}): {run_dir}"
        )


def _validate_propagation_manifest(
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    run_id: str,
    git_commit: str,
) -> None:
    expected = {
        "status": "completed",
        "mode": "activation-propagation",
        "tranche_id": TRANCHE_ID,
        "config_id": PROPAGATION_CONFIG_ID,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": False,
        "seed": 0,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_SELECTION_HASH,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    expected_runs, expected_checkpoints = _expected_source_paths()
    if manifest.get("source_runs") != expected_runs:
        mismatches.append("source_runs")
    if manifest.get("source_checkpoints") != expected_checkpoints:
        mismatches.append("source_checkpoints")
    diagnostic = manifest.get("activation_propagation")
    diagnostic_expected = {
        "selected_runs": _expected_selected_runs(),
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "eval_batches": None,
        "batch_size": 4,
        "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
        "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "validation_cache_tokens": EXPECTED_VALIDATION_CACHE_TOKENS,
        "trailing_tokens_excluded": EXPECTED_TRAILING_VALIDATION_TOKENS,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_SELECTION_HASH,
        "complete_named_partition": True,
        "execution": EXPECTED_PROPAGATION_EXECUTION,
    }
    if not isinstance(diagnostic, dict) or any(
        diagnostic.get(key) != value
        for key, value in diagnostic_expected.items()
    ):
        mismatches.append("activation_propagation")
    validation = (manifest.get("tokenized_data") or {}).get("validation")
    if not isinstance(validation, dict) or any(
        validation.get(key) != value
        for key, value in {
            "partition": "selection",
            "source_document_indices_sha256": EXPECTED_SELECTION_HASH,
            "tokens_sha256": (
                "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19"
            ),
        }.items()
    ):
        mismatches.append("validation cache")
    if mismatches:
        raise ValueError(
            f"A2 propagation manifest mismatch ({', '.join(mismatches)}): {run_dir}"
        )


def _validate_propagation_metrics(
    metrics: dict[str, Any],
    run_dir: Path,
    *,
    points: Sequence[LogicalOpportunityPoint],
) -> None:
    expected = {
        "activation_propagation/methods": len(A2_SOURCES),
        "activation_propagation/layers": len(LAYERS),
        "activation_propagation/matmul_stages": len(LOGICAL_MATMUL_STAGES),
        "activation_propagation/validation_batches": EXPECTED_VALIDATION_BATCHES,
        "activation_propagation/validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
        "activation_propagation/validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "activation_propagation/validation_cache_tokens": EXPECTED_VALIDATION_CACHE_TOKENS,
        "activation_propagation/trailing_tokens_excluded": EXPECTED_TRAILING_VALIDATION_TOKENS,
        "activation_propagation/validation_partition": "selection",
        "activation_propagation/validation_partition_hash": EXPECTED_SELECTION_HASH,
    }
    mismatches = [key for key, value in expected.items() if metrics.get(key) != value]
    for source, point in zip(A2_SOURCES, points, strict=True):
        prefix = f"activation_propagation/endpoint/{source.config_id}"
        for name, expected_value in (
            ("R_block", point.R_block),
            ("R_model", point.R_model),
        ):
            value = metrics.get(f"{prefix}/{name}")
            if not _finite_fraction(value) or not _close(
                float(value), expected_value
            ):
                mismatches.append(f"{prefix}/{name}")
    if mismatches:
        raise ValueError(
            f"A2 propagation metric mismatch ({', '.join(mismatches)}): {run_dir}"
        )


def _validate_propagation_payload(
    payload: dict[str, Any], run_dir: Path
) -> tuple[LogicalOpportunityPoint, ...]:
    expected = {
        "schema_version": 5,
        "validation_batches": EXPECTED_VALIDATION_BATCHES,
        "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
        "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "validation_cache_tokens": EXPECTED_VALIDATION_CACHE_TOKENS,
        "trailing_tokens_excluded": EXPECTED_TRAILING_VALIDATION_TOKENS,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_SELECTION_HASH,
        "complete_named_partition": True,
        "block_size": EXPECTED_BLOCK_SIZE,
        "batch_size": 4,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "matmul_stage_order": list(LOGICAL_MATMUL_STAGES),
        "execution": EXPECTED_PROPAGATION_EXECUTION,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    methods = payload.get("methods")
    points: list[LogicalOpportunityPoint] = []
    if not isinstance(methods, list) or len(methods) != len(A2_SOURCES):
        mismatches.append("methods")
    else:
        expected_runs, expected_checkpoints = _expected_source_paths()
        for source, source_run, source_checkpoint, method in zip(
            A2_SOURCES,
            expected_runs,
            expected_checkpoints,
            methods,
            strict=True,
        ):
            if not isinstance(method, dict) or any(
                method.get(key) != value
                for key, value in {
                    "label": source.label,
                    "config_id": source.config_id,
                    "run_id": source.run_id,
                    "source_run": source_run,
                    "source_checkpoint": source_checkpoint,
                    "source_manifest_status": "completed",
                    "num_layers": len(LAYERS),
                    "batches": EXPECTED_VALIDATION_BATCHES,
                }.items()
            ):
                mismatches.append(f"method:{source.config_id}")
                continue
            architecture = method.get("architecture")
            if not isinstance(architecture, dict) or any(
                architecture.get(key) != value
                for key, value in {
                    "topology_id": "A1-H",
                    "active_sites": ["h"],
                    "num_layers": len(LAYERS),
                    "sequence_length": EXPECTED_BLOCK_SIZE,
                }.items()
            ):
                mismatches.append(f"architecture:{source.config_id}")
                continue
            try:
                points.append(_logical_opportunity_point(source, method["endpoint"]))
            except (KeyError, TypeError, ValueError) as error:
                mismatches.append(f"endpoint:{source.config_id}:{error}")
    if mismatches:
        raise ValueError(
            f"A2 propagation payload mismatch ({', '.join(mismatches)}): {run_dir}"
        )
    return tuple(points)


def _logical_opportunity_point(
    source: A2Source, endpoint: dict[str, Any]
) -> LogicalOpportunityPoint:
    if not isinstance(endpoint, dict):
        raise ValueError("endpoint must be a mapping")
    if endpoint.get("validation_sequences") != EXPECTED_VALIDATION_SEQUENCES or endpoint.get(
        "validation_tokens"
    ) != EXPECTED_VALIDATION_TOKENS:
        raise ValueError("endpoint validation coverage differs")
    integer_names = (
        "block_zero_product_count",
        "block_product_count",
        "lm_head_product_count",
        "model_product_count",
    )
    integers: dict[str, int] = {}
    for name in integer_names:
        value = endpoint.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {name}")
        integers[name] = value
    if (
        integers["block_product_count"] <= 0
        or integers["lm_head_product_count"] <= 0
        or integers["block_zero_product_count"] > integers["block_product_count"]
        or integers["model_product_count"]
        != integers["block_product_count"] + integers["lm_head_product_count"]
    ):
        raise ValueError("inconsistent endpoint denominators")
    per_operation = endpoint.get("per_operation")
    if not isinstance(per_operation, dict) or set(per_operation) != set(
        LOGICAL_MATMUL_STAGES
    ):
        raise ValueError("operation coverage differs")
    operation_zero_total = 0
    operation_product_total = 0
    for name in LOGICAL_MATMUL_STAGES:
        row = per_operation[name]
        if not isinstance(row, dict):
            raise ValueError(f"invalid operation {name}")
        zero_count = row.get("zero_product_count")
        product_count = row.get("product_count")
        fraction = row.get("zero_product_fraction")
        if (
            isinstance(zero_count, bool)
            or not isinstance(zero_count, int)
            or isinstance(product_count, bool)
            or not isinstance(product_count, int)
            or not 0 <= zero_count <= product_count
            or product_count <= 0
            or not _close(float(fraction), zero_count / product_count)
        ):
            raise ValueError(f"invalid operation counters for {name}")
        operation_zero_total += zero_count
        operation_product_total += product_count
    if (
        operation_zero_total != integers["block_zero_product_count"]
        or operation_product_total != integers["block_product_count"]
    ):
        raise ValueError("operation counters do not sum to the block endpoint")
    r_block = endpoint.get("R_block")
    r_model = endpoint.get("R_model")
    if not _finite_fraction(r_block) or not _finite_fraction(r_model):
        raise ValueError("invalid R_block/R_model")
    if not _close(
        float(r_block),
        integers["block_zero_product_count"] / integers["block_product_count"],
    ) or not _close(
        float(r_model),
        integers["block_zero_product_count"] / integers["model_product_count"],
    ):
        raise ValueError("logical-opportunity fractions disagree with counters")
    return LogicalOpportunityPoint(
        config_id=source.config_id,
        run_id=source.run_id,
        label=source.label,
        lambda_value=source.lambda_value,
        R_block=float(r_block),
        R_model=float(r_model),
        **integers,
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


def _logical_opportunity_rows(
    rows: Sequence[LogicalOpportunityPoint],
) -> tuple[LogicalOpportunityPoint, ...]:
    ordered = tuple(rows)
    expected = tuple((source.config_id, source.run_id) for source in A2_SOURCES)
    realized = tuple((row.config_id, row.run_id) for row in ordered)
    if realized != expected:
        raise ValueError("A2 logical-opportunity cohort is incomplete or out of order.")
    return ordered


def _build_provenance(
    *,
    root: Path,
    stem: str,
    data: A2SpilloverData,
    staged_outputs: Sequence[Path],
    final_outputs: Sequence[Path],
) -> dict[str, Any]:
    density_stems = {LAYERWISE_DISTRIBUTION_STEM, POOLED_DISTRIBUTION_STEM}
    reduction: dict[str, Any] = {
        "threshold_fractions": "sum(layer hits) / sum(layer totals)",
        "pooled_rms": "sqrt(sum(layer finite * layer rms^2) / sum(layer finite))",
        "primary_threshold": 0.01,
    }
    if stem == RESPONSE_STEM:
        reduction.update(
            {
                "x": "100 * (n_m(0.01, lambda) - n_m(0.01, control))",
                "y": "100 * (n_A(0.01, lambda) - n_A(0.01, control))",
                "attention_sites": list(ATTENTION_SITES),
                "attention_pool": "sum(site-and-layer hits) / sum(site-and-layer totals)",
                "logical_product_metric": {
                    "R_block": "block_zero_product_count / block_product_count",
                    "R_model": "block_zero_product_count / model_product_count",
                    "delta_R_model": "R_model(condition) - R_model(control)",
                    "block_operations": list(LOGICAL_MATMUL_STAGES),
                    "model_denominator_extension": (
                        "dense hidden-to-vocabulary LM head"
                    ),
                    "attention_implementation": "eager",
                    "future_causal_positions_excluded": True,
                    "execution": dict(EXPECTED_PROPAGATION_EXECUTION),
                    "block_size": EXPECTED_BLOCK_SIZE,
                    "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
                    "validation_tokens": EXPECTED_VALIDATION_TOKENS,
                    "validation_cache_tokens": EXPECTED_VALIDATION_CACHE_TOKENS,
                    "trailing_tokens_excluded": EXPECTED_TRAILING_VALIDATION_TOKENS,
                    "unit": "exact-zero logical-product opportunity",
                    "not_a_speedup": True,
                },
            }
        )
    elif stem in density_stems:
        pooled = stem == POOLED_DISTRIBUTION_STEM
        source_indices = (
            POOLED_SOURCE_INDICES if pooled else LAYERWISE_SOURCE_INDICES
        )
        reduction.update(
            {
                "density_sources": [
                    A2_SOURCES[index].config_id for index in source_indices
                ],
                "density_sites": list(DENSITY_SITES),
                "layers": "pooled within site" if pooled else list(LAYERS),
                "histogram_range": list(EXPECTED_RANGE),
                "display_windows": {
                    site: list(DENSITY_WINDOWS[site]) for site in DENSITY_SITES
                },
                "rebin_factor": DENSITY_REBIN_FACTOR,
                "rebinned_width": (
                    (EXPECTED_RANGE[1] - EXPECTED_RANGE[0])
                    / EXPECTED_BINS
                    * DENSITY_REBIN_FACTOR
                ),
                "density_normalization": (
                    "rebinned count / ((total - exact_zero_hits) * bin width); "
                    "stored and cropped tails remain in the denominator"
                ),
                "exact_zero_atom_separate": True,
                "no_kde_or_interpolation": True,
                "comparison_status": (
                    "post-hoc descriptive high-coefficient view"
                    if pooled
                    else "predeclared control-versus-lambda-1 comparison"
                ),
                "panels": _density_panel_summaries(
                    data,
                    pooled=pooled,
                    source_indices=source_indices,
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported A2 figure stem for provenance: {stem}.")
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
            "propagation_config_id": PROPAGATION_CONFIG_ID,
            "propagation_run_id": PROPAGATION_RUN_ID,
            "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        },
        "reduction": reduction,
        "losses": [asdict(point) for point in data.losses],
        "logical_opportunities": [
            asdict(point) for point in _logical_opportunity_rows(data.logical_opportunities)
        ],
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


def _density_panel_summaries(
    data: A2SpilloverData,
    *,
    pooled: bool,
    source_indices: Sequence[int],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    layer_groups: tuple[tuple[int, ...], ...] = (
        (LAYERS,) if pooled else tuple((layer,) for layer in LAYERS)
    )
    for source_index in source_indices:
        source = A2_SOURCES[source_index]
        for site in DENSITY_SITES:
            for layers in layer_groups:
                curve = _density_reduction(
                    data,
                    source_index=source_index,
                    site=site,
                    layers=layers,
                )
                summaries.append(
                    {
                        "config_id": source.config_id,
                        "lambda": source.lambda_value,
                        "site": site,
                        "layers": list(layers),
                        "total": curve.total,
                        "exact_zero_hits": curve.exact_zero_hits,
                        "nonzero_total": curve.nonzero_total,
                        "outside_stored_hits": curve.outside_stored_hits,
                        "outside_display_hits": curve.outside_display_hits,
                        "exact_zero_fraction": curve.exact_zero_fraction,
                        "outside_display_fraction_nonzero": (
                            curve.outside_display_fraction_nonzero
                        ),
                    }
                )
    return summaries


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


def _finite_fraction(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _lambda_values(sources: Sequence[A2Source]) -> str:
    values = [f"lambda {source.lambda_value:g}" for source in sources]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return " and ".join(values)
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _matching_edge_index(edges: Sequence[float], value: float) -> int | None:
    return next(
        (
            index
            for index, edge in enumerate(edges)
            if math.isclose(edge, value, rel_tol=0.0, abs_tol=1e-10)
        ),
        None,
    )


def _format_probability(value: float | None) -> str:
    if value is None:
        return "n/a"
    percentage = 100.0 * value
    if percentage == 0.0:
        return "0%"
    if percentage == 100.0:
        return "100%"
    if percentage < 0.001:
        return "<0.001%"
    if percentage < 0.1:
        return f"{percentage:.3f}%"
    if percentage < 10.0:
        return f"{percentage:.2f}%"
    if percentage > 99.99:
        return ">99.99%"
    if percentage >= 99.95:
        return f"{percentage:.2f}%"
    return f"{percentage:.1f}%"


def _format_compact_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    percentage = 100.0 * value
    if percentage == 0.0:
        return "0"
    if percentage == 100.0:
        return "100"
    if percentage < 0.001:
        return "<.001"
    if percentage < 0.01:
        return "<.01"
    if percentage < 10.0:
        return f"{percentage:.2f}"
    if percentage < 99.95:
        return f"{percentage:.1f}"
    if percentage > 99.99:
        return ">99.99"
    return f"{percentage:.2f}"


def _format_precise_percentage(value: float) -> str:
    percentage = 100.0 * value
    if percentage == 0.0:
        return "0.0000"
    if percentage < 0.0001:
        return "<0.0001"
    return f"{percentage:.4f}"


def _condition_label(source: A2Source) -> str:
    return "Control" if source.lambda_value == 0.0 else f"lambda {source.lambda_value:g}"


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
