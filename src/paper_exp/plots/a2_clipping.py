"""Pinned A2 post-hoc clipping frontier and companion report.

The paper recipe in this module owns one exact six-checkpoint cohort and one
numbered clipping diagnostic.  It never discovers a latest run.  The accepted
diagnostic must provide every source/cutoff pair, exact logical-product integer
counts, and a same-sweep ``t = 0`` reference for each checkpoint.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from paper_exp.config import validate_diagnostic_config
from paper_exp.diagnostics.clipping import _checkpoint_content_identity
from paper_exp.diagnostics.clipping_frontier import validate_clipping_frontier_rows

from .a2_spillover import (
    A2_SOURCES,
    A2Source,
    EXPECTED_TRAINING_GIT_COMMIT,
    _close,
    _expected_selected_runs,
    _input_record,
    _load_json,
    _load_training_source,
    _load_yaml,
    _mapping,
    _relative,
    _repository_root,
    _require_files,
    _sha256,
    _write_json,
    _write_text,
)
from .dispatch import _read_jsonl
from .export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    DOUBLE_COLUMN_WIDTH_INCHES,
    export_figure,
    publish_staged_outputs,
)
from .style import COLORBLIND_SAFE_COLORS, PAPER_STYLE


TRANCHE_ID = "02-a2-l1-screen"
DIAGNOSTIC_CONFIG_ID = "020-a2-posthoc-clipping-frontier"
# Filled only after an accepted diagnostic has been verified.  Paper plotting
# must fail closed before that exact immutable run and its launch commit exist.
DIAGNOSTIC_RUN_ID: str | None = None
DIAGNOSTIC_GIT_COMMIT: str | None = None
DIAGNOSTIC_ARTIFACT_SHA256: str | None = None
FIGURE_STEM = "04-a2-posthoc-clipping-frontier"
GENERATOR_PATH = "src/paper_exp/plots/a2_clipping.py"

ZERO_REFERENCE_GIT_COMMIT = "96621bcb73f74933f95b8b5fcd9a63ec2e15e3ff"
ZERO_REFERENCE_ARTIFACT_SHA256 = (
    "709599e0e68abe8350a720e6a37f392f19aadaf42fb681d28308b62db44cf3d9"
)
ZERO_REFERENCE = {
    "tranche_id": TRANCHE_ID,
    "config_id": "019-a2-activation-propagation",
    "run_id": "001-20260828-110533-6ac813e6",
    "git_commit": ZERO_REFERENCE_GIT_COMMIT,
    "artifact_sha256": ZERO_REFERENCE_ARTIFACT_SHA256,
}
ZERO_REFERENCE_AUDIT_SCOPE = (
    "Only zero-threshold per-operation integer counts and operation/block/"
    "LM-head/model denominators are compared with diagnostic 019. Validation "
    "losses and all paired deltas come solely from this diagnostic's own "
    "zero-threshold rows."
)

CLIPPING_MODE = "threshold"
SITES = ("a", "m", "h", "q_post", "k_post", "v")
THRESHOLDS = (0.0, 0.01, 0.03, 0.10, 0.30)
EXPECTED_POINT_COUNT = len(A2_SOURCES) * len(THRESHOLDS)
EXPECTED_VALIDATION_SEQUENCES = 152
EXPECTED_VALIDATION_BATCHES = 38
EXPECTED_VALIDATION_TOKENS = 311_296
EXPECTED_CACHE_TOKENS = 311_739
EXPECTED_BLOCK_SIZE = 2_048
EXPECTED_CACHE_SHA256 = (
    "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19"
)
EXPECTED_PARTITION_HASH = (
    "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
)
SITE_WIDTHS = {
    "a": 128,
    "m": 128,
    "h": 512,
    "q_post": 128,
    "k_post": 128,
    "v": 128,
}
EXPECTED_SITE_COUNTS = {
    site: EXPECTED_VALIDATION_TOKENS * 6 * width
    for site, width in SITE_WIDTHS.items()
}

_CURVE_LINESTYLES: tuple[Any, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
)
_THRESHOLD_MARKERS = ("o", "s", "^", "D", "X")


@dataclass(frozen=True)
class A2ClippingPoint:
    """One validated source/cutoff observation and its paired reductions."""

    config_id: str
    run_id: str
    label: str
    lambda_value: float
    threshold: float
    validation_loss: float
    delta_validation_loss: float
    block_zero_product_count: int
    block_product_count: int
    model_product_count: int
    R_block: float
    R_model: float
    delta_R_model: float
    nondominated: bool


@dataclass(frozen=True)
class A2ClippingData:
    """Complete plot-ready cohort plus deterministic evidence records."""

    points: tuple[A2ClippingPoint, ...]
    diagnostic_run_id: str
    diagnostic_git_commit: str
    source_evidence: tuple[dict[str, Any], ...]
    inputs: tuple[dict[str, Any], ...]


def reduce_a2_clipping_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[A2ClippingPoint, ...]:
    """Validate and reduce the exact source-major, cutoff-major A2 grid."""

    row_list = list(rows)
    validate_clipping_frontier_rows(
        row_list,
        selected_runs=_expected_selected_runs(),
        thresholds=list(THRESHOLDS),
        sites=list(SITES),
        expected_validation_batches=EXPECTED_VALIDATION_BATCHES,
        expected_validation_sequences=EXPECTED_VALIDATION_SEQUENCES,
        expected_validation_tokens=EXPECTED_VALIDATION_TOKENS,
    )

    parsed: list[dict[str, Any]] = []
    for row_index, (source, threshold) in enumerate(
        (
            (source, threshold)
            for source in A2_SOURCES
            for threshold in THRESHOLDS
        )
    ):
        row = row_list[row_index]
        loss = float(row["validation_loss"])
        if loss <= 0.0:
            raise ValueError(f"A2 clipping row {row_index} validation_loss must be positive.")
        if row["site_activation_count"] != EXPECTED_SITE_COUNTS:
            raise ValueError(
                f"A2 clipping row {row_index} has incorrect site element coverage."
            )
        block_zero_product_count = int(row["block_zero_product_count"])
        block_product_count = int(row["block_matmul_product_count"])
        model_product_count = int(row["model_matmul_product_count"])
        parsed.append(
            {
                "source": source,
                "threshold": threshold,
                "validation_loss": loss,
                "block_zero_product_count": block_zero_product_count,
                "block_product_count": block_product_count,
                "model_product_count": model_product_count,
                "R_block": block_zero_product_count / block_product_count,
                "R_model": block_zero_product_count / model_product_count,
            }
        )

    baselines = {
        row["source"].config_id: row
        for row in parsed
        if _close(row["threshold"], 0.0)
    }
    if set(baselines) != {source.config_id for source in A2_SOURCES}:
        raise ValueError("A2 clipping artifact lacks one t = 0 baseline per source.")

    preliminary: list[A2ClippingPoint] = []
    for row in parsed:
        source = row["source"]
        baseline = baselines[source.config_id]
        preliminary.append(
            A2ClippingPoint(
                config_id=source.config_id,
                run_id=source.run_id,
                label=source.label,
                lambda_value=source.lambda_value,
                threshold=row["threshold"],
                validation_loss=row["validation_loss"],
                delta_validation_loss=(
                    row["validation_loss"] - baseline["validation_loss"]
                ),
                block_zero_product_count=row["block_zero_product_count"],
                block_product_count=row["block_product_count"],
                model_product_count=row["model_product_count"],
                R_block=row["R_block"],
                R_model=row["R_model"],
                delta_R_model=row["R_model"] - baseline["R_model"],
                nondominated=False,
            )
        )

    flags = _nondominated_flags(preliminary)
    return tuple(
        A2ClippingPoint(**{**asdict(point), "nondominated": flag})
        for point, flag in zip(preliminary, flags, strict=True)
    )


def build_a2_clipping_figure(data: A2ClippingData) -> Figure:
    """Render absolute and within-checkpoint A2 clipping frontiers."""

    _validate_reduced_cohort(data.points)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.10),
    )
    absolute_axis, delta_axis = axes

    for source_index, source in enumerate(A2_SOURCES):
        points = _source_points(data.points, source)
        color = COLORBLIND_SAFE_COLORS[source_index]
        linestyle = _CURVE_LINESTYLES[source_index]
        absolute_axis.plot(
            [100.0 * point.R_model for point in points],
            [point.validation_loss for point in points],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            zorder=2,
        )
        delta_axis.plot(
            [100.0 * point.delta_R_model for point in points],
            [point.delta_validation_loss for point in points],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            zorder=2,
        )
        for threshold_index, point in enumerate(points):
            marker = _THRESHOLD_MARKERS[threshold_index]
            facecolor = "white" if threshold_index == 0 else color
            marker_size = 39 if threshold_index == 0 else 31
            for axis_object, x_value, y_value in (
                (
                    absolute_axis,
                    100.0 * point.R_model,
                    point.validation_loss,
                ),
                (
                    delta_axis,
                    100.0 * point.delta_R_model,
                    point.delta_validation_loss,
                ),
            ):
                if threshold_index == 0 and axis_object is delta_axis:
                    continue
                axis_object.scatter(
                    [x_value],
                    [y_value],
                    marker=marker,
                    s=marker_size,
                    facecolor=facecolor,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=3,
                )

    # All paired baselines are exactly the same point.  Show that common
    # anchor once without jitter rather than letting drawing order imply that
    # one checkpoint owns the origin.
    delta_axis.scatter(
        [0.0],
        [0.0],
        marker=_THRESHOLD_MARKERS[0],
        s=43,
        facecolor="white",
        edgecolor="#333333",
        linewidth=1.1,
        zorder=4,
    )

    absolute_axis.set_title("(a) Absolute frontier", loc="left")
    absolute_axis.set_xlabel(r"Model-wide logical opportunity, $R_{model}$ (%)")
    absolute_axis.set_ylabel("Validation loss")
    absolute_axis.set_xlim(left=0.0)

    delta_axis.set_title("(b) Within-checkpoint change", loc="left")
    delta_axis.set_xlabel(r"Change in $R_{model}$ (pp)")
    delta_axis.set_ylabel("Change in validation loss")
    delta_axis.axhline(0.0, color="#888888", linewidth=0.8, zorder=0)
    delta_axis.axvline(0.0, color="#888888", linewidth=0.8, zorder=0)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
        axis.xaxis.grid(True, alpha=0.18)
        axis.yaxis.grid(True, alpha=0.18)

    all_losses = [point.validation_loss for point in data.points]
    _set_padded_limits(
        absolute_axis,
        values=all_losses,
        axis_name="y",
        include_zero=False,
    )
    _include_zero(
        delta_axis,
        values=[100.0 * point.delta_R_model for point in data.points],
        axis_name="x",
    )
    _include_zero(
        delta_axis,
        values=[point.delta_validation_loss for point in data.points],
        axis_name="y",
    )

    condition_handles = [
        Line2D(
            [0],
            [0],
            color=COLORBLIND_SAFE_COLORS[index],
            linestyle=_CURVE_LINESTYLES[index],
            linewidth=1.6,
            label=_condition_label(source),
        )
        for index, source in enumerate(A2_SOURCES)
    ]
    threshold_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=marker,
            linestyle="none",
            markerfacecolor="white" if index == 0 else "#777777",
            markeredgecolor="#555555",
            markersize=5.6,
            label=f"t = {threshold:g}",
        )
        for index, (threshold, marker) in enumerate(
            zip(THRESHOLDS, _THRESHOLD_MARKERS, strict=True)
        )
    ]
    figure.legend(
        handles=[*condition_handles, *threshold_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.0,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.90,
        bottom=0.275,
        wspace=0.29,
    )
    return figure


def build_a2_clipping_markdown(data: A2ClippingData) -> str:
    """Build the self-contained caption and complete 30-point result table."""

    _validate_reduced_cohort(data.points)
    lines = [
        "# A2 post-hoc clipping frontier",
        "",
        "**Figure caption.** The six curves are the accepted Pythia-14M A2 "
        "checkpoints: the ReLU control and h-only L1 pressure at lambda "
        "`{0.1, 0.5, 1, 2, 5}`. At evaluation only, one common absolute cutoff "
        "`t` is applied jointly at `{a, m, h, q_post, k_post, v}` in every layer: "
        "an activation is set to exactly zero when `abs(x) <= t`. Panel (a) "
        "shows validation loss against observed model-wide logical opportunity "
        "`R_model`; panel (b) subtracts each checkpoint's own same-sweep `t = 0` "
        "reference from both quantities. Marker shape identifies the common "
        "cutoff. Panel (a) uses a disclosed zoomed validation-loss axis while "
        "its `R_model` axis starts at zero. All tested points are shown; no point "
        "is selected as a winner.",
        "",
        "The clipped ports are `a` (attention-branch layer-normalization output "
        "entering Q/K/V projections), `m` (MLP-branch layer-normalization output "
        "entering W1), `h` (MLP hidden activation entering W2), `q_post` and "
        "`k_post` (post-RoPE queries and keys entering QK-transpose), and `v` "
        "(values entering PV). `R_block` is the number of products with at least "
        "one exact-zero operand, pooled over the six transformer-block operation "
        "families, divided by all products in those operations. `R_model` keeps "
        "that numerator and divides by the block products plus the dense LM-head "
        "products. Evaluation uses eager attention and the sources' BF16 policy.",
        "",
        "Each point evaluates the complete A2 selection partition (152 sequences, "
        "311,296 tokens, sequence length 2,048) with exact operation-level "
        "logical-product counting. `R_block` and `R_model` are potentially "
        "avoidable logical products, not measured speedup, removed FLOPs, latency, "
        "energy, or throughput. This is seed-0 (`n = 1`) descriptive evidence on "
        "a finite cutoff grid; it provides neither seed uncertainty nor a "
        "continuous/global optimum. Joint clipping measures checkpoint-level "
        "thresholdability and does not causally isolate spillover or any one site. "
        "It is not a trained `A6-POST` topology or a Phase-B threshold gate.",
        "",
        "A tested point is marked nondominated when no other tested point has both "
        "lower-or-equal validation loss and higher-or-equal `R_model`, with at "
        "least one strict inequality. This descriptive flag is not a selection.",
        "",
        "## Complete tested grid",
        "",
        "| Condition | lambda | t | Validation loss | Delta loss | R_block (%) | "
        "R_model (%) | Delta R_model (pp) | Nondominated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for point in data.points:
        lambda_value = "—" if point.lambda_value == 0.0 else f"{point.lambda_value:g}"
        lines.append(
            "| "
            f"{_condition_label(_source_for_config(point.config_id))} | "
            f"{lambda_value} | {point.threshold:g} | "
            f"{point.validation_loss:.6f} | "
            f"{point.delta_validation_loss:+.6f} | "
            f"{100.0 * point.R_block:.6f} | "
            f"{100.0 * point.R_model:.6f} | "
            f"{100.0 * point.delta_R_model:+.6f} | "
            f"{'yes' if point.nondominated else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Paired changes are defined only within one checkpoint's clipping sweep:",
            "",
            "```text",
            "Delta loss(t) = validation_loss(t) - validation_loss(0)",
            "Delta R_model(t) = R_model(t) - R_model(0)",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def load_a2_clipping(repository: str | Path | None = None) -> A2ClippingData:
    """Load the exact accepted A2 clipping diagnostic, or fail before discovery."""

    if (
        DIAGNOSTIC_RUN_ID is None
        or DIAGNOSTIC_GIT_COMMIT is None
        or DIAGNOSTIC_ARTIFACT_SHA256 is None
    ):
        raise RuntimeError(
            "A2 clipping paper evidence is not pinned: set the exact accepted "
            "DIAGNOSTIC_RUN_ID, DIAGNOSTIC_GIT_COMMIT, and "
            "DIAGNOSTIC_ARTIFACT_SHA256 after diagnostic 020 completes and is "
            "verified."
        )
    root = _repository_root(repository)
    recipe_path = (
        root
        / "experiments"
        / TRANCHE_ID
        / "run"
        / f"{DIAGNOSTIC_CONFIG_ID}.yaml"
    )
    run_dir = (
        root
        / "experiments"
        / TRANCHE_ID
        / "raw"
        / DIAGNOSTIC_CONFIG_ID
        / DIAGNOSTIC_RUN_ID
    )
    paths = {
        "diagnostic_recipe": recipe_path,
        "diagnostic_saved_config": run_dir / "config.yaml",
        "diagnostic_manifest": run_dir / "manifest.json",
        "diagnostic_metrics": run_dir / "metrics.json",
        "diagnostic_predictions": run_dir / "predictions.jsonl",
        "clipping_frontier": run_dir / "clipping_frontier.jsonl",
    }
    _require_files(paths.values(), context="A2 clipping diagnostic")
    recipe = _load_yaml(recipe_path)
    saved = _load_yaml(paths["diagnostic_saved_config"])
    if recipe != saved:
        raise ValueError(f"A2 clipping saved config does not match its recipe: {run_dir}")
    validate_diagnostic_config(recipe, "clipping_frontier")
    _validate_recipe(recipe, recipe_path)
    manifest = _load_json(paths["diagnostic_manifest"])
    _validate_manifest(manifest, run_dir=run_dir, recipe=recipe)
    _validate_metrics(_load_json(paths["diagnostic_metrics"]), run_dir=run_dir)
    if _sha256(paths["clipping_frontier"]) != DIAGNOSTIC_ARTIFACT_SHA256:
        raise ValueError(
            f"A2 clipping artifact SHA-256 mismatch: {paths['clipping_frontier']}"
        )
    rows = _read_jsonl(paths["clipping_frontier"])
    if _read_jsonl(paths["diagnostic_predictions"]) != rows:
        raise ValueError(
            f"A2 clipping predictions do not match the cohort artifact: {run_dir}"
        )
    points = reduce_a2_clipping_rows(rows)

    source_evidence, source_inputs = _validate_source_evidence(
        root=root,
        rows=rows,
        diagnostic_manifest=manifest,
    )
    reference_inputs = _validate_zero_reference(root=root, manifest=manifest)
    diagnostic_inputs = tuple(
        _input_record(root, path, role=role, source=None)
        for role, path in paths.items()
    )
    return A2ClippingData(
        points=points,
        diagnostic_run_id=DIAGNOSTIC_RUN_ID,
        diagnostic_git_commit=DIAGNOSTIC_GIT_COMMIT,
        source_evidence=source_evidence,
        inputs=(*diagnostic_inputs, *source_inputs, *reference_inputs),
    )


def generate_a2_clipping_figure(
    repository: str | Path | None = None,
) -> tuple[Path, ...]:
    """Atomically generate the fixed A2 clipping PDF/PNG/report/provenance."""

    root = _repository_root(repository)
    data = load_a2_clipping(root)
    figures_dir = root / "experiments" / TRANCHE_ID / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_paths = tuple(
        figures_dir / f"{FIGURE_STEM}.{suffix}"
        for suffix in ("pdf", "png", "md", "provenance.json")
    )
    with tempfile.TemporaryDirectory(prefix=".a2-clipping.", dir=figures_dir) as temp:
        staging = Path(temp)
        staged = {path: staging / path.name for path in final_paths}
        export_figure(
            lambda: build_a2_clipping_figure(data),
            staged[figures_dir / f"{FIGURE_STEM}.pdf"],
            save_png=True,
            style=PAPER_STYLE,
            profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
        )
        _write_text(
            staged[figures_dir / f"{FIGURE_STEM}.md"],
            build_a2_clipping_markdown(data),
        )
        suite_outputs = final_paths[:3]
        provenance = _build_provenance(
            root=root,
            data=data,
            staged_outputs=tuple(staged[path] for path in suite_outputs),
            final_outputs=suite_outputs,
        )
        _write_json(
            staged[figures_dir / f"{FIGURE_STEM}.provenance.json"],
            provenance,
        )
        publish_staged_outputs({path: staged[path] for path in final_paths})
    return final_paths


def _nondominated_flags(points: Sequence[A2ClippingPoint]) -> tuple[bool, ...]:
    flags = []
    for point in points:
        dominated = any(
            other.validation_loss <= point.validation_loss
            and other.R_model >= point.R_model
            and (
                other.validation_loss < point.validation_loss
                or other.R_model > point.R_model
            )
            for other in points
        )
        flags.append(not dominated)
    return tuple(flags)


def _validate_reduced_cohort(points: Sequence[A2ClippingPoint]) -> None:
    expected = tuple(
        (source.config_id, source.run_id, threshold)
        for source in A2_SOURCES
        for threshold in THRESHOLDS
    )
    actual = tuple((point.config_id, point.run_id, point.threshold) for point in points)
    if len(points) != EXPECTED_POINT_COUNT or any(
        left[:2] != right[:2] or not _close(left[2], right[2])
        for left, right in zip(actual, expected, strict=True)
    ):
        raise ValueError("A2 clipping reduced cohort is incomplete or out of order.")
    for source in A2_SOURCES:
        baseline = _source_points(points, source)[0]
        if not _close(baseline.delta_validation_loss, 0.0) or not _close(
            baseline.delta_R_model, 0.0
        ):
            raise ValueError(f"A2 clipping baseline delta is not zero: {source.config_id}.")


def _source_points(
    points: Sequence[A2ClippingPoint], source: A2Source
) -> tuple[A2ClippingPoint, ...]:
    selected = tuple(point for point in points if point.config_id == source.config_id)
    if len(selected) != len(THRESHOLDS):
        raise ValueError(f"A2 clipping source is incomplete: {source.config_id}.")
    return selected


def _validate_recipe(config: Mapping[str, Any], path: Path) -> None:
    frontier = _mapping(config.get("clipping_frontier"), "clipping_frontier", path)
    expected_runs = [
        {
            "label": source.label,
            "tranche_id": TRANCHE_ID,
            "config_id": source.config_id,
            "run_id": source.run_id,
        }
        for source in A2_SOURCES
    ]
    expected = {
        "selected_runs": expected_runs,
        "mode": CLIPPING_MODE,
        "thresholds": list(THRESHOLDS),
        "sites": list(SITES),
        "measure_zero_products": True,
        "zero_threshold_reference": ZERO_REFERENCE,
    }
    mismatches = [key for key, value in expected.items() if frontier.get(key) != value]
    validation = _mapping(config.get("validation"), "validation", path)
    validation_expected = {
        "enabled": True,
        "batch_size": 4,
        "eval_batches": None,
        "partition": "selection",
        "partition_hash": EXPECTED_PARTITION_HASH,
        "tokens_sha256": EXPECTED_CACHE_SHA256,
    }
    mismatches.extend(
        f"validation.{key}"
        for key, value in validation_expected.items()
        if validation.get(key) != value
    )
    output = _mapping(config.get("output"), "output", path)
    if output.get("dir") != f"experiments/{TRANCHE_ID}/raw":
        mismatches.append("output.dir")
    if mismatches:
        raise ValueError(
            f"A2 clipping diagnostic recipe mismatch ({', '.join(mismatches)}): {path}"
        )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    recipe: Mapping[str, Any],
) -> None:
    expected = {
        "tranche_id": TRANCHE_ID,
        "config_id": DIAGNOSTIC_CONFIG_ID,
        "run_id": DIAGNOSTIC_RUN_ID,
        "mode": "clipping-frontier",
        "status": "completed",
        "git_commit": DIAGNOSTIC_GIT_COMMIT,
        "git_dirty": False,
        "seed": 0,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if run_dir.parent.name != DIAGNOSTIC_CONFIG_ID or run_dir.name != DIAGNOSTIC_RUN_ID:
        mismatches.append("path identity")
    frontier = _mapping(manifest.get("clipping_frontier"), "clipping_frontier", run_dir)
    recipe_frontier = recipe["clipping_frontier"]
    for key in (
        "selected_runs",
        "mode",
        "thresholds",
        "sites",
        "measure_zero_products",
    ):
        if frontier.get(key) != recipe_frontier.get(key):
            mismatches.append(f"clipping_frontier.{key}")
    coverage = {
        "zero_threshold_reference": {
            **ZERO_REFERENCE,
            "source_run": (
                f"experiments/{TRANCHE_ID}/raw/"
                f"{ZERO_REFERENCE['config_id']}/{ZERO_REFERENCE['run_id']}"
            ),
            "source_artifact": (
                f"experiments/{TRANCHE_ID}/raw/"
                f"{ZERO_REFERENCE['config_id']}/{ZERO_REFERENCE['run_id']}/"
                "activation_propagation.json"
            ),
            "source_artifact_sha256": ZERO_REFERENCE_ARTIFACT_SHA256,
            "audit_scope": ZERO_REFERENCE_AUDIT_SCOPE,
        },
        "attention_implementation": "eager",
        "eval_batches": None,
        "batch_size": 4,
        "validation_sequences": EXPECTED_VALIDATION_SEQUENCES,
        "validation_batches": EXPECTED_VALIDATION_BATCHES,
        "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "validation_cache_tokens": EXPECTED_CACHE_TOKENS,
        "trailing_tokens_excluded": EXPECTED_CACHE_TOKENS - EXPECTED_VALIDATION_TOKENS,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_PARTITION_HASH,
        "complete_named_partition": True,
        "execution": {
            "requested_device": "cuda",
            "requested_precision": "bfloat16",
            "resolved_device": "cuda",
            "resolved_precision": "bfloat16",
        },
    }
    mismatches.extend(
        f"clipping_frontier.{key}"
        for key, value in coverage.items()
        if frontier.get(key) != value
    )
    if mismatches:
        raise ValueError(
            f"A2 clipping diagnostic manifest mismatch ({', '.join(mismatches)}): {run_dir}"
        )


def _validate_metrics(metrics: Mapping[str, Any], *, run_dir: Path) -> None:
    expected = {
        "clipping_frontier/sources": len(A2_SOURCES),
        "clipping_frontier/cutoffs": len(THRESHOLDS),
        "clipping_frontier/points": EXPECTED_POINT_COUNT,
        "clipping_frontier/validation_batches_per_point": EXPECTED_VALIDATION_BATCHES,
        "clipping_frontier/validation_sequences_per_point": (
            EXPECTED_VALIDATION_SEQUENCES
        ),
        "clipping_frontier/validation_tokens_per_point": EXPECTED_VALIDATION_TOKENS,
        "clipping_frontier/total_point_tokens": (
            EXPECTED_POINT_COUNT * EXPECTED_VALIDATION_TOKENS
        ),
    }
    mismatches = [key for key, value in expected.items() if metrics.get(key) != value]
    wall_seconds = metrics.get("clipping_frontier/wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, int | float)
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0.0
    ):
        mismatches.append("clipping_frontier/wall_seconds")
    if mismatches:
        raise ValueError(
            f"A2 clipping metrics mismatch ({', '.join(mismatches)}): {run_dir}"
        )


def _validate_zero_reference(
    *, root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    reference_dir = (
        root
        / "experiments"
        / TRANCHE_ID
        / "raw"
        / ZERO_REFERENCE["config_id"]
        / ZERO_REFERENCE["run_id"]
    )
    paths = {
        "zero_reference_manifest": reference_dir / "manifest.json",
        "zero_reference_artifact": reference_dir / "activation_propagation.json",
    }
    _require_files(paths.values(), context="A2 clipping zero-threshold reference")
    reference_manifest = _load_json(paths["zero_reference_manifest"])
    expected_manifest = {
        "tranche_id": ZERO_REFERENCE["tranche_id"],
        "config_id": ZERO_REFERENCE["config_id"],
        "run_id": ZERO_REFERENCE["run_id"],
        "mode": "activation-propagation",
        "status": "completed",
        "git_commit": ZERO_REFERENCE_GIT_COMMIT,
        "git_dirty": False,
    }
    mismatches = [
        key for key, value in expected_manifest.items()
        if reference_manifest.get(key) != value
    ]
    artifact_sha256 = _sha256(paths["zero_reference_artifact"])
    if artifact_sha256 != ZERO_REFERENCE_ARTIFACT_SHA256:
        mismatches.append("activation_propagation SHA-256")
    recorded = _mapping(
        _mapping(manifest.get("clipping_frontier"), "clipping_frontier", reference_dir).get(
            "zero_threshold_reference"
        ),
        "clipping_frontier.zero_threshold_reference",
        reference_dir,
    )
    if recorded.get("source_artifact_sha256") != artifact_sha256:
        mismatches.append("recorded zero-reference SHA-256")
    if mismatches:
        raise ValueError(
            f"A2 clipping zero-reference mismatch ({', '.join(mismatches)}): "
            f"{reference_dir}"
        )
    return tuple(
        _input_record(root, path, role=role, source=None)
        for role, path in paths.items()
    )


def _validate_source_evidence(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    diagnostic_manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    evidence: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    source_runs: list[str] = []
    source_checkpoints: list[str] = []
    source_statuses: list[str] = []
    source_checkpoint_contents: list[dict[str, Any]] = []
    validation_metadata: dict[str, Any] | None = None
    for source_index, source in enumerate(A2_SOURCES):
        row = rows[source_index * len(THRESHOLDS)]
        source_run = (
            root
            / "experiments"
            / TRANCHE_ID
            / "raw"
            / source.config_id
            / source.run_id
        )
        _, accepted_inputs = _load_training_source(root, source)
        manifest = _load_json(source_run / "manifest.json")
        launch_expected = {
            "status": "completed",
            "git_commit": EXPECTED_TRAINING_GIT_COMMIT,
            "git_dirty": False,
            "config_sha256": source.config_sha256,
            "condition_fingerprint": source.condition_fingerprint,
        }
        launch_mismatches = [
            key for key, value in launch_expected.items()
            if manifest.get(key) != value
        ]
        if launch_mismatches:
            raise ValueError(
                "A2 clipping source launch provenance mismatch "
                f"({', '.join(launch_mismatches)}): {source.config_id}."
            )

        expected_run_path = f"experiments/{TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
        expected_checkpoint_path = f"{expected_run_path}/checkpoints/final"
        if row.get("source_run") != expected_run_path:
            raise ValueError(f"A2 clipping source_run mismatch: {source.config_id}.")
        if row.get("source_checkpoint") != expected_checkpoint_path:
            raise ValueError(f"A2 clipping checkpoint path mismatch: {source.config_id}.")

        checkpoint_path = source_run / "checkpoints" / "final"
        checkpoint_identity = _checkpoint_content_identity(checkpoint_path)
        if row.get("source_checkpoint_content") != checkpoint_identity:
            raise ValueError(f"A2 clipping checkpoint content mismatch: {source.config_id}.")
        cache_identity = _mapping(
            row.get("source_validation_cache"),
            "source_validation_cache",
            source_run,
        )
        current_validation = _validate_cache_identity(
            cache_identity,
            manifest=manifest,
            source=source,
        )
        if validation_metadata is None:
            validation_metadata = current_validation
        elif current_validation != validation_metadata:
            raise ValueError("A2 clipping source validation manifests are not identical.")
        source_runs.append(expected_run_path)
        source_checkpoints.append(expected_checkpoint_path)
        source_statuses.append(str(manifest["status"]))
        source_checkpoint_contents.append(checkpoint_identity)
        evidence.append(
            {
                "label": source.label,
                "tranche_id": TRANCHE_ID,
                "config_id": source.config_id,
                "run_id": source.run_id,
                "source_run": expected_run_path,
                "source_checkpoint": expected_checkpoint_path,
                "source_checkpoint_content": checkpoint_identity,
                "source_validation_cache": dict(cache_identity),
                "manifest_status": manifest["status"],
                "git_commit": manifest["git_commit"],
                "git_dirty": manifest["git_dirty"],
                "config_sha256": manifest["config_sha256"],
                "condition_fingerprint": manifest["condition_fingerprint"],
            }
        )
        inputs.extend(accepted_inputs)
        recorded_checkpoint_paths = {
            item["path"] for item in accepted_inputs if item["role"] == "checkpoint"
        }
        inputs.extend(
            _input_record(root, path, role="source_checkpoint_file", source=source)
            for path in sorted(checkpoint_path.rglob("*"))
            if path.is_file() and _relative(path, root) not in recorded_checkpoint_paths
        )
    expected_manifest_sources = {
        "source_runs": source_runs,
        "source_checkpoints": source_checkpoints,
        "source_manifest_statuses": source_statuses,
        "source_checkpoint_contents": source_checkpoint_contents,
        "tokenized_data": {"validation": validation_metadata},
    }
    manifest_mismatches = [
        key for key, value in expected_manifest_sources.items()
        if diagnostic_manifest.get(key) != value
    ]
    if manifest_mismatches:
        raise ValueError(
            "A2 clipping diagnostic source manifest mismatch "
            f"({', '.join(manifest_mismatches)})."
        )
    return tuple(evidence), tuple(inputs)


def _validate_cache_identity(
    cache: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    source: A2Source,
) -> dict[str, Any]:
    expected = {
        "dtype": "int32",
        "tokens": EXPECTED_CACHE_TOKENS,
        "tokens_bytes": EXPECTED_CACHE_TOKENS * 4,
        "tokens_sha256": EXPECTED_CACHE_SHA256,
        "block_size": EXPECTED_BLOCK_SIZE,
        "split": "validation",
        "max_documents": 500,
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "source_document_indices_sha256": EXPECTED_PARTITION_HASH,
    }
    mismatches = [key for key, value in expected.items() if cache.get(key) != value]
    recorded = _mapping(
        _mapping(manifest.get("tokenized_data"), "tokenized_data", Path(source.config_id)).get(
            "validation"
        ),
        "tokenized_data.validation",
        Path(source.config_id),
    )
    mismatches.extend(
        f"manifest.{key}" for key, value in expected.items() if recorded.get(key) != value
    )
    if not isinstance(cache.get("tokens_path"), str) or not cache["tokens_path"].strip():
        mismatches.append("tokens_path")
    if mismatches:
        raise ValueError(
            f"A2 clipping validation-cache mismatch ({', '.join(mismatches)}): "
            f"{source.config_id}."
        )
    return recorded


def _build_provenance(
    *,
    root: Path,
    data: A2ClippingData,
    staged_outputs: Sequence[Path],
    final_outputs: Sequence[Path],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "figure_id": FIGURE_STEM,
        "generator": {
            "path": GENERATOR_PATH,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "diagnostic": {
            "tranche_id": TRANCHE_ID,
            "config_id": DIAGNOSTIC_CONFIG_ID,
            "run_id": data.diagnostic_run_id,
            "git_commit": data.diagnostic_git_commit,
            "artifact_sha256": DIAGNOSTIC_ARTIFACT_SHA256,
        },
        "sources": list(data.source_evidence),
        "inputs": list(data.inputs),
        "reduction": {
            "mode": CLIPPING_MODE,
            "rule": "set x to exactly zero when abs(x) <= t",
            "sites": list(SITES),
            "thresholds": list(THRESHOLDS),
            "point_order": "source-major, then ascending threshold",
            "absolute_x": "100 * block_zero_product_count / model_product_count",
            "absolute_y": "same-sweep validation_loss(t)",
            "delta_x": "100 * (R_model(t) - R_model(0))",
            "delta_y": "validation_loss(t) - validation_loss(0)",
            "nondominated": (
                "no tested point has lower-or-equal validation loss and "
                "higher-or-equal R_model with at least one strict inequality"
            ),
            "validation_sequences_per_point": EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens_per_point": EXPECTED_VALIDATION_TOKENS,
            "points": [asdict(point) for point in data.points],
        },
        "outputs": [
            {
                "path": _relative(final, root),
                "sha256": _sha256(staged),
                "size_bytes": staged.stat().st_size,
            }
            for staged, final in zip(staged_outputs, final_outputs, strict=True)
        ],
    }


def _condition_label(source: A2Source) -> str:
    return (
        "Control"
        if source.lambda_value == 0.0
        else rf"L1 $\lambda = {source.lambda_value:g}$"
    )


def _source_for_config(config_id: str) -> A2Source:
    try:
        return next(source for source in A2_SOURCES if source.config_id == config_id)
    except StopIteration as error:
        raise ValueError(f"Unknown A2 clipping source config: {config_id}") from error


def _set_padded_limits(
    axis_object: Any,
    *,
    values: Sequence[float],
    axis_name: str,
    include_zero: bool,
) -> None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("Cannot set figure limits from an empty finite sequence.")
    lower = min(finite)
    upper = max(finite)
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    margin = max(0.08 * span, 1e-6)
    setter = axis_object.set_xlim if axis_name == "x" else axis_object.set_ylim
    setter(lower - margin, upper + margin)


def _include_zero(axis_object: Any, *, values: Sequence[float], axis_name: str) -> None:
    _set_padded_limits(
        axis_object,
        values=values,
        axis_name=axis_name,
        include_zero=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the pinned A2 post-hoc clipping frontier."
    )
    parser.add_argument("--repository", type=Path, default=None)
    arguments = parser.parse_args(argv)
    for output in generate_a2_clipping_figure(arguments.repository):
        print(output)


if __name__ == "__main__":
    main()
