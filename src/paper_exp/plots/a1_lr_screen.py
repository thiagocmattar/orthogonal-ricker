"""Deterministic table and curve for the pinned A1 learning-rate screen."""

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

from paper_exp.config import validate_training_config
from paper_exp.design import complete_config_sha256
from paper_exp.utils import read_json

from .export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    DOUBLE_COLUMN_WIDTH_INCHES,
    export_figure,
    publish_staged_outputs,
)
from .style import (
    COLORBLIND_SAFE_COLORS,
    PAPER_STYLE,
    SERIES_MARKERS,
    SeriesStyle,
    series_style,
)


TRANCHE_ID = "01-a1-lr-screen"
GROUP_ID = "A1-lr-screen"
OUTPUT_STEM = "01-a1-learning-rate-screen"
PROGRESS_OUTPUT_STEM = "02-a1-training-progress"
GENERATOR_PATH = "src/paper_exp/plots/a1_lr_screen.py"
EXPECTED_STEPS = 1_526
EXPECTED_TOKENS_PER_STEP = 262_144
EXPECTED_TRAINING_TOKENS = 400_031_744
EXPECTED_LOG_EVERY = 10
EXPECTED_WARMUP_STEPS = 16
EXPECTED_TRAIN_LOG_STEPS = (
    1,
    *range(EXPECTED_LOG_EVERY, EXPECTED_STEPS, EXPECTED_LOG_EVERY),
    EXPECTED_STEPS,
)
EXPECTED_VALIDATION_STEPS = (1, 191, 382, 573, 764, 955, 1_146, 1_337, 1_526)
EXPECTED_VALIDATION_TOKENS = 311_296
EXPECTED_VALIDATION_SEQUENCES = 152
EXPECTED_VALIDATION_BATCHES = 38
EXPECTED_SELECTION_HASH = (
    "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
)
EXPECTED_SCHEDULE_HASH = (
    "5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c"
)
EXPECTED_IMPLEMENTATION_ID = "a1_pretraining_v1"


@dataclass(frozen=True)
class A1Source:
    """One reviewed, pinned A1 source and its terminal classification."""

    config_id: str
    run_id: str
    learning_rate: float
    git_commit: str
    config_sha256: str
    condition_fingerprint: str
    terminal_status: str = "completed"
    case_class: str = "eligible"
    evidence_status: str = "valid"


A1_SOURCES = (
    A1Source(
        config_id="001-a1-lr-5e-4",
        run_id="001-20260825-191155-6b7376de",
        learning_rate=5e-4,
        git_commit="276da7cd8e9142da48b95e12b46a99d61367ca8f",
        config_sha256=(
            "b6e2468dd7be8e687cb84e61537e7791601239ea9279dcaeea1e0b4d6cd1e5e5"
        ),
        condition_fingerprint=(
            "592e1b1e08c806b6d780482bfac65cf829cd56a852f98f5eb9d3128baba73fd7"
        ),
    ),
    A1Source(
        config_id="002-a1-lr-1e-3",
        run_id="001-20260825-191154-b9299c46",
        learning_rate=1e-3,
        git_commit="276da7cd8e9142da48b95e12b46a99d61367ca8f",
        config_sha256=(
            "8085202e2c488aa227ae8e39d837bd85798f821c8a76d1b4200ed9e0e00dcfa0"
        ),
        condition_fingerprint=(
            "2c6fb1f003cc0fcb560bdeea10ec53cbdf1fbaa39bbb9b0f1ac60cc47f850a4a"
        ),
    ),
    A1Source(
        config_id="003-a1-lr-2e-3",
        run_id="001-20260825-195141-f842c400",
        learning_rate=2e-3,
        git_commit="276da7cd8e9142da48b95e12b46a99d61367ca8f",
        config_sha256=(
            "3bae75d2f91ad378c09a3acf035457488622692fde1363ee314111542515e86e"
        ),
        condition_fingerprint=(
            "f32e3d49385078a9e8eaf50498ea12cb30657f1d41762255f12b8b2e47a6116f"
        ),
    ),
    A1Source(
        config_id="004-a1-lr-4e-3",
        run_id="001-20260826-123606-46e7454f",
        learning_rate=4e-3,
        git_commit="f235081239cd67831684e4174531992af4253e9c",
        config_sha256=(
            "701afdbdcade83bdd878a30b65683825fb27c15038e24e4f6426f30658f1680d"
        ),
        condition_fingerprint=(
            "7742e7219fb40ee55adc4a42d87c00de6790eb7a5b3f5ff9643f85a137b9dd01"
        ),
    ),
    A1Source(
        config_id="005-a1-lr-8e-3",
        run_id="001-20260826-135546-928279bb",
        learning_rate=8e-3,
        git_commit="a4ddaa5c9897224a9285afae09d2d9c6b07b3cec",
        config_sha256=(
            "9383ccc371a86e64fff2beb02057f15ff84256bf505ce1449e0ff3df5d2945ea"
        ),
        condition_fingerprint=(
            "9cc6a74400386deee36fe706aac1967fd79facd156d7e2ca7cacee56a2a22167"
        ),
    ),
    A1Source(
        config_id="006-a1-lr-1p6e-2",
        run_id="001-20260826-174611-04b42898",
        learning_rate=1.6e-2,
        git_commit="d4105722516958df6e9c3cc43b20d6bfd4619d0f",
        config_sha256=(
            "ca66f1c07c009a6801ac5ccf038d9d0e1f9cfc320affb40517a744e04ec8b751"
        ),
        condition_fingerprint=(
            "088d78d630a4c292211ae4038a9f7ea8be8a824778758c15a161149e4b9891dc"
        ),
    ),
    A1Source(
        config_id="007-a1-lr-3p2e-2",
        run_id="001-20260826-182559-bb05a50c",
        learning_rate=3.2e-2,
        git_commit="d4105722516958df6e9c3cc43b20d6bfd4619d0f",
        config_sha256=(
            "24b93ac2435717c653f4849eea597698020a4b9abd8a71fcd5ca2fc600ea543d"
        ),
        condition_fingerprint=(
            "97e04832ee14e7929778e7e38237a8005b6028ef4435212714aa9aa7546eed29"
        ),
    ),
    A1Source(
        config_id="008-a1-lr-6p4e-2",
        run_id="001-20260826-190546-4df1c441",
        learning_rate=6.4e-2,
        git_commit="d4105722516958df6e9c3cc43b20d6bfd4619d0f",
        config_sha256=(
            "9e6f56a930024b5bf67fa34a9bd94ce3a1903f99a149249cf1b5946a5a2d597e"
        ),
        condition_fingerprint=(
            "336adaf43bde4494c90d1f6a7f11d5a4aca7804a60ee75fded15a670a9f6f89e"
        ),
    ),
    A1Source(
        config_id="009-a1-lr-1p28e-1",
        run_id="001-20260826-221407-812e78f4",
        learning_rate=1.28e-1,
        git_commit="4e5e93e64d979004f2fd2e2a5b7aab275b088e0d",
        config_sha256=(
            "8a7da2c097435968e03704fa09921a8345dea0c14a20a9e9ed717c980b05f259"
        ),
        condition_fingerprint=(
            "0b5c10665bc83355fe365e91b1aeae156f970a96dc995a4cffbd80581426a001"
        ),
    ),
    A1Source(
        config_id="010-a1-lr-2p56e-1",
        run_id="001-20260826-225355-07a74682",
        learning_rate=2.56e-1,
        git_commit="4e5e93e64d979004f2fd2e2a5b7aab275b088e0d",
        config_sha256=(
            "4edfcfbca86a72e0ef0809e743ae62157b4759518dd2b7705a8ba8d5f8805cfc"
        ),
        condition_fingerprint=(
            "9a541bb3b14196b96a9d130c7eb2d3c360beaa94ff4d828bb13aee93b10fec18"
        ),
    ),
    A1Source(
        config_id="011-a1-lr-5p12e-1",
        run_id="001-20260826-233349-87400e7d",
        learning_rate=5.12e-1,
        git_commit="4e5e93e64d979004f2fd2e2a5b7aab275b088e0d",
        config_sha256=(
            "7424beae00383dc0396712992cc38d372f2b7bf6de2226bf30b574a8fb7a7ae1"
        ),
        condition_fingerprint=(
            "9782a13261d41f817ad99b72ac4efd2addc43bfe51de452ac9673cdfcf15faaa"
        ),
    ),
)


@dataclass(frozen=True)
class A1Point:
    """Validated plot-ready value from one pinned source."""

    config_id: str
    run_id: str
    learning_rate: float
    final_validation_loss: float
    seed: int
    optimizer_steps: int
    training_tokens: int
    validation_tokens: int
    terminal_status: str
    case_class: str
    evidence_status: str


@dataclass(frozen=True)
class A1ProgressTrace:
    """Validated logged training progress from one pinned A1 source."""

    config_id: str
    run_id: str
    peak_learning_rate: float
    validation_steps: tuple[int, ...]
    validation_tokens_seen: tuple[int, ...]
    validation_losses: tuple[float, ...]
    learning_rate_steps: tuple[int, ...]
    learning_rate_tokens_seen: tuple[int, ...]
    effective_learning_rates: tuple[float, ...]


def load_a1_lr_points(
    repository: str | Path | None = None,
) -> tuple[tuple[A1Point, ...], tuple[dict[str, Any], ...]]:
    """Load and validate the exact eleven-cell cohort without run discovery."""

    root = _repository_root(repository)
    _validate_source_registry()
    points: list[A1Point] = []
    inputs: list[dict[str, Any]] = []
    for source in A1_SOURCES:
        point, source_inputs = _load_source(root, source)
        points.append(point)
        inputs.extend(source_inputs)
    ordered = tuple(sorted(points, key=lambda point: point.learning_rate))
    if tuple(point.config_id for point in ordered) != tuple(
        source.config_id for source in A1_SOURCES
    ):
        raise ValueError("Pinned A1 sources are not in strictly increasing LR order.")
    return ordered, tuple(inputs)


def load_a1_training_progress(
    repository: str | Path | None = None,
) -> tuple[tuple[A1ProgressTrace, ...], tuple[dict[str, Any], ...]]:
    """Load the exact A1 event streams and their accepted evidence envelopes."""

    root = _repository_root(repository)
    _validate_source_registry()
    traces: list[A1ProgressTrace] = []
    inputs: list[dict[str, Any]] = []
    for source in A1_SOURCES:
        point, source_inputs = _load_source(root, source)
        trace, events_input = _load_progress_trace(root, source, point)
        traces.append(trace)
        inputs.extend((*source_inputs, events_input))
    ordered = tuple(sorted(traces, key=lambda trace: trace.peak_learning_rate))
    if tuple(trace.config_id for trace in ordered) != tuple(
        source.config_id for source in A1_SOURCES
    ):
        raise ValueError("Pinned A1 progress sources are not in increasing LR order.")
    return ordered, tuple(inputs)


def select_a1_point(points: Sequence[A1Point]) -> A1Point:
    """Select minimum final loss, breaking an exact tie toward lower LR."""

    if not points:
        raise ValueError("A1 selection requires at least one point.")
    for point in points:
        if not math.isfinite(point.final_validation_loss):
            raise ValueError(f"Non-finite A1 validation loss: {point.config_id}.")
        if not math.isfinite(point.learning_rate) or point.learning_rate <= 0.0:
            raise ValueError(f"Invalid A1 learning rate: {point.config_id}.")
    return min(
        points,
        key=lambda point: (point.final_validation_loss, point.learning_rate),
    )


def build_a1_lr_figure(points: Sequence[A1Point]) -> Figure:
    """Build the complete eleven-cell fixed-horizon A1 curve."""

    ordered = tuple(sorted(points, key=lambda point: point.learning_rate))
    learning_rates = [point.learning_rate for point in ordered]
    losses = [point.final_validation_loss for point in ordered]

    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.4))
    sweep_style = series_style(0)
    axis.plot(
        learning_rates,
        losses,
        color=sweep_style.color,
        marker=sweep_style.marker,
        linestyle=sweep_style.linestyle,
        linewidth=1.6,
        markersize=5.5,
        zorder=2,
    )
    axis.set_title("Pythia-14M A1 learning-rate screen (400M training tokens)")
    axis.set_xlabel(r"Peak learning rate (log$_2$ scale)")
    axis.set_ylabel("Final validation loss")
    axis.set_xscale("log", base=2)
    axis.set_xticks(learning_rates)
    axis.set_xticklabels(
        tuple(_math_lr_label(value) for value in learning_rates),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    axis.tick_params(axis="x", labelsize=8, pad=2)
    axis.minorticks_off()

    log_values = [math.log2(value) for value in learning_rates]
    log_span = max(log_values) - min(log_values)
    axis.set_xlim(
        2 ** (min(log_values) - 0.08 * log_span),
        2 ** (max(log_values) + 0.08 * log_span),
    )
    loss_span = max(losses) - min(losses)
    loss_margin = max(0.12 * loss_span, 0.05)
    axis.set_ylim(min(losses) - loss_margin, max(losses) + loss_margin)
    axis.grid(False)
    axis.yaxis.grid(True, alpha=0.25)

    for point in ordered:
        axis.annotate(
            f"{point.final_validation_loss:.3f}",
            (point.learning_rate, point.final_validation_loss),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.20)
    return figure


def build_a1_training_progress_figure(
    traces: Sequence[A1ProgressTrace],
) -> Figure:
    """Build the shared-legend A1 loss and effective-LR progress panels."""

    ordered = tuple(sorted(traces, key=lambda trace: trace.peak_learning_rate))
    if len(ordered) != len(A1_SOURCES):
        raise ValueError("A1 training progress requires the complete pinned cohort.")

    figure, (loss_axis, learning_rate_axis) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.0),
        sharex=True,
    )
    legend_handles = []
    for index, trace in enumerate(ordered):
        _validate_progress_trace_for_rendering(trace)
        style = _progress_series_style(index)
        label = _math_lr_label(trace.peak_learning_rate)
        validation_tokens_millions = [
            value / 1_000_000 for value in trace.validation_tokens_seen
        ]
        learning_rate_tokens_millions = [
            value / 1_000_000 for value in trace.learning_rate_tokens_seen
        ]
        (loss_line,) = loss_axis.plot(
            validation_tokens_millions,
            trace.validation_losses,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            markersize=3.4,
            markeredgewidth=0.5,
            label=label,
            alpha=0.95,
        )
        learning_rate_axis.plot(
            learning_rate_tokens_millions,
            trace.effective_learning_rates,
            color=style.color,
            marker=style.marker,
            markevery=(index * 3, 18),
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            markersize=3.0,
            markeredgewidth=0.5,
            alpha=0.95,
        )
        legend_handles.append(loss_line)

    loss_axis.set_title("(a) Learning trajectories")
    loss_axis.set_xlabel("Cumulative training tokens (millions)")
    loss_axis.set_ylabel("Validation loss")
    learning_rate_axis.set_title("(b) Learning-rate schedules")
    learning_rate_axis.set_xlabel("Cumulative training tokens (millions)")
    learning_rate_axis.set_ylabel(r"Effective learning rate (log$_2$ scale)")
    learning_rate_axis.set_yscale("log", base=2)

    final_tokens_millions = EXPECTED_TRAINING_TOKENS / 1_000_000
    for axis in (loss_axis, learning_rate_axis):
        axis.set_xlim(0.0, final_tokens_millions * 1.02)
        axis.set_xticks((0.0, 100.0, 200.0, 300.0, 400.0))
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.25)

    all_losses = [loss for trace in ordered for loss in trace.validation_losses]
    loss_span = max(all_losses) - min(all_losses)
    loss_margin = max(0.04 * loss_span, 0.05)
    loss_axis.set_ylim(min(all_losses) - loss_margin, max(all_losses) + loss_margin)

    all_learning_rates = [
        rate for trace in ordered for rate in trace.effective_learning_rates
    ]
    minimum_log2 = math.log2(min(all_learning_rates))
    maximum_log2 = math.log2(max(all_learning_rates))
    log2_margin = max(0.04 * (maximum_log2 - minimum_log2), 0.25)
    learning_rate_axis.set_ylim(
        2 ** (minimum_log2 - log2_margin),
        2 ** (maximum_log2 + log2_margin),
    )
    first_tick_exponent = math.ceil(minimum_log2 - log2_margin)
    last_tick_exponent = math.floor(maximum_log2 + log2_margin)
    tick_stride = max(
        1,
        math.ceil((last_tick_exponent - first_tick_exponent + 1) / 8),
    )
    tick_exponents = tuple(
        range(first_tick_exponent, last_tick_exponent + 1, tick_stride)
    )
    learning_rate_axis.set_yticks(tuple(2**value for value in tick_exponents))
    learning_rate_axis.set_yticklabels(
        tuple(rf"$2^{{{value}}}$" for value in tick_exponents)
    )

    figure.suptitle("Pythia-14M A1 training progress (400M training tokens)")
    legend_order = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5)
    ordered_legend_handles = tuple(legend_handles[index] for index in legend_order)
    figure.legend(
        ordered_legend_handles,
        tuple(handle.get_label() for handle in ordered_legend_handles),
        title="Peak learning rate",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.45,
        columnspacing=1.15,
        fontsize=8,
        title_fontsize=8,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.82,
        bottom=0.31,
        wspace=0.30,
    )
    return figure


def build_a1_training_progress_caption() -> str:
    """Return the self-contained companion caption for figure 02."""

    return "\n".join(
        (
            "# A1 training progress",
            "",
            (
                "**Figure caption.** Pythia-14M validation-loss trajectories "
                "(left) and logged effective learning-rate schedules (right) "
                "over the fixed 400,031,744-token A1 horizon. Each line is one "
                "peak learning rate, with the same color, marker, and line style "
                "in both panels. Validation loss is measured on the fixed "
                "selection partition at nine scheduled evaluations. Effective "
                "learning rate is recorded at the first update, every ten "
                "updates, and the final update; the right y-axis uses a base-2 "
                "logarithmic scale. Each rate has one randomly initialized run "
                "with seed 0 (n = 1), so no uncertainty estimate is available."
            ),
            "",
        )
    )


def build_a1_lr_table(points: Sequence[A1Point]) -> str:
    """Return the complete human-readable A1 result table."""

    ordered = tuple(sorted(points, key=lambda point: point.learning_rate))
    selected = select_a1_point(ordered)
    selected_is_upper_boundary = selected == ordered[-1]
    rows = [
        "# A1 learning-rate screen",
        "",
        (
            "**Figure caption.** Final validation loss versus peak learning rate "
            "for randomly initialized Pythia-14M after 1,526 optimizer updates "
            "(400,031,744 training tokens). Each point is one run with seed 0; "
            "the line connects tested rates in ascending order, the x-axis uses "
            "a base-2 logarithmic scale, and point labels report the final loss. "
            "Lower is better. With n = 1 per learning rate, no uncertainty "
            "estimate is available."
        ),
        "",
        (
            "| Config | Pinned run | Peak LR | Seed | Updates | Training tokens | "
            "Final validation loss | Terminal status | Eligibility | Evidence | Selection |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for point in ordered:
        selection = "selected"
        if point == selected and selected_is_upper_boundary:
            selection = "selected (upper tested boundary)"
        elif point != selected:
            selection = "not selected"
        rows.append(
            f"| `{point.config_id}` | `{point.run_id}` | `{_plain_lr_label(point.learning_rate)}` "
            f"| {point.seed} | {point.optimizer_steps:,} | {point.training_tokens:,} "
            f"| {point.final_validation_loss:.6f} | {point.terminal_status} "
            f"| {point.case_class} | {point.evidence_status} | {selection} |"
        )
    rows.extend(
        [
            "",
            (
                "Selection rule: lowest finite final validation loss; an exact tie "
                "favors the lower peak learning rate."
            ),
            "",
            (
                f"Selected peak learning rate: `{_plain_lr_label(selected.learning_rate)}`. "
                + (
                    "It is the upper tested boundary, so this is only the best tested "
                    "rate at the fixed 400M-token horizon, not a global or "
                    "horizon-independent optimum."
                    if selected_is_upper_boundary
                    else "It lies inside the tested range."
                )
            ),
            "",
            (
                "Validation losses are displayed to six decimals; full saved values and "
                "input hashes are retained in the provenance sidecar."
            ),
            "",
        ]
    )
    return "\n".join(rows)


def generate_a1_lr_screen(
    repository: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Validate sources and atomically publish PDF, PNG, table, and provenance."""

    root = _repository_root(repository)
    points, inputs = load_a1_lr_points(root)
    selected = select_a1_point(points)
    figures_dir = root / "experiments" / TRANCHE_ID / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_pdf = figures_dir / f"{OUTPUT_STEM}.pdf"
    final_png = figures_dir / f"{OUTPUT_STEM}.png"
    final_table = figures_dir / f"{OUTPUT_STEM}.md"
    final_provenance = figures_dir / f"{OUTPUT_STEM}.provenance.json"

    with tempfile.TemporaryDirectory(prefix=f".{OUTPUT_STEM}.", dir=figures_dir) as temp:
        staging = Path(temp)
        staged_pdf = staging / final_pdf.name
        staged_png = staging / final_png.name
        staged_table = staging / final_table.name
        staged_provenance = staging / final_provenance.name
        export_figure(
            lambda: build_a1_lr_figure(points),
            staged_pdf,
            save_png=True,
            style=PAPER_STYLE,
            profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
        )
        _write_text(staged_table, build_a1_lr_table(points))
        provenance = _build_provenance(
            root=root,
            points=points,
            inputs=inputs,
            selected=selected,
            staged_outputs=(staged_pdf, staged_png, staged_table),
            final_outputs=(final_pdf, final_png, final_table),
        )
        _write_json(staged_provenance, provenance)
        publish_staged_outputs(
            {
                final_pdf: staged_pdf,
                final_png: staged_png,
                final_table: staged_table,
                final_provenance: staged_provenance,
            }
        )
    return final_pdf, final_png, final_table, final_provenance


def generate_a1_training_progress(
    repository: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Validate and atomically publish the A1 training-progress figure."""

    root = _repository_root(repository)
    traces, inputs = load_a1_training_progress(root)
    figures_dir = root / "experiments" / TRANCHE_ID / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_pdf = figures_dir / f"{PROGRESS_OUTPUT_STEM}.pdf"
    final_png = figures_dir / f"{PROGRESS_OUTPUT_STEM}.png"
    final_caption = figures_dir / f"{PROGRESS_OUTPUT_STEM}.md"
    final_provenance = figures_dir / f"{PROGRESS_OUTPUT_STEM}.provenance.json"

    with tempfile.TemporaryDirectory(
        prefix=f".{PROGRESS_OUTPUT_STEM}.", dir=figures_dir
    ) as temp:
        staging = Path(temp)
        staged_pdf = staging / final_pdf.name
        staged_png = staging / final_png.name
        staged_caption = staging / final_caption.name
        staged_provenance = staging / final_provenance.name
        export_figure(
            lambda: build_a1_training_progress_figure(traces),
            staged_pdf,
            save_png=True,
            style=PAPER_STYLE,
            profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
        )
        _write_text(staged_caption, build_a1_training_progress_caption())
        provenance = _build_progress_provenance(
            root=root,
            traces=traces,
            inputs=inputs,
            staged_outputs=(staged_pdf, staged_png, staged_caption),
            final_outputs=(final_pdf, final_png, final_caption),
        )
        _write_json(staged_provenance, provenance)
        publish_staged_outputs(
            {
                final_pdf: staged_pdf,
                final_png: staged_png,
                final_caption: staged_caption,
                final_provenance: staged_provenance,
            }
        )
    return final_pdf, final_png, final_caption, final_provenance


def _load_source(
    root: Path,
    source: A1Source,
) -> tuple[A1Point, tuple[dict[str, Any], ...]]:
    config_path = root / "experiments" / TRANCHE_ID / "run" / f"{source.config_id}.yaml"
    run_dir = (
        root
        / "experiments"
        / TRANCHE_ID
        / "raw"
        / source.config_id
        / source.run_id
    )
    paths = {
        "recipe_config": config_path,
        "config": run_dir / "config.yaml",
        "manifest": run_dir / "manifest.json",
        "metrics": run_dir / "metrics.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required A1 input does not exist: {path}")

    recipe = _load_yaml_mapping(paths["recipe_config"])
    config = _load_yaml_mapping(paths["config"])
    if config != recipe:
        raise ValueError(f"Saved config does not match its tracked recipe: {run_dir}")
    validate_training_config(config)
    manifest = _load_json_mapping(paths["manifest"])
    metrics = _load_json_mapping(paths["metrics"])
    _validate_config(source, config, run_dir)
    _validate_manifest(source, manifest, run_dir)
    validation_loss = _validate_metrics(metrics, run_dir)

    point = A1Point(
        config_id=source.config_id,
        run_id=source.run_id,
        learning_rate=source.learning_rate,
        final_validation_loss=validation_loss,
        seed=0,
        optimizer_steps=EXPECTED_STEPS,
        training_tokens=EXPECTED_TRAINING_TOKENS,
        validation_tokens=EXPECTED_VALIDATION_TOKENS,
        terminal_status=source.terminal_status,
        case_class=source.case_class,
        evidence_status=source.evidence_status,
    )
    inputs = tuple(
        {
            "config_id": source.config_id,
            "run_id": source.run_id,
            "role": role,
            "path": _relative(path, root),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for role, path in paths.items()
    )
    return point, inputs


def _load_progress_trace(
    root: Path,
    source: A1Source,
    point: A1Point,
) -> tuple[A1ProgressTrace, dict[str, Any]]:
    run_dir = (
        root
        / "experiments"
        / TRANCHE_ID
        / "raw"
        / source.config_id
        / source.run_id
    )
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"Required A1 progress input does not exist: {events_path}")
    rows = _load_jsonl_mappings(events_path)
    train_rows: list[tuple[int, int, float]] = []
    validation_rows: list[tuple[int, int, float]] = []
    previous_key: tuple[int, int] | None = None
    for row_index, row in enumerate(rows, start=1):
        event = row.get("event")
        if event not in {"train", "validation"}:
            raise ValueError(
                f"Unexpected A1 progress event at row {row_index}: {events_path}"
            )
        step = _positive_event_integer(row, "step", row_index, events_path)
        tokens_seen = _positive_event_integer(
            row, "tokens_seen", row_index, events_path
        )
        if step > EXPECTED_STEPS or tokens_seen != step * EXPECTED_TOKENS_PER_STEP:
            raise ValueError(
                f"A1 progress token-step mismatch at row {row_index}: {events_path}"
            )
        event_rank = 0 if event == "train" else 1
        current_key = (step, event_rank)
        if previous_key is not None and current_key <= previous_key:
            raise ValueError(f"A1 progress events are not strictly ordered: {events_path}")
        previous_key = current_key
        if event == "train":
            learning_rate = _finite_event_number(
                row, "learning_rate", row_index, events_path
            )
            if learning_rate <= 0.0:
                raise ValueError(
                    "A1 effective learning rate must be positive at row "
                    f"{row_index}: {events_path}"
                )
            train_rows.append((step, tokens_seen, learning_rate))
        else:
            validation_loss = _finite_event_number(
                row, "validation_loss", row_index, events_path
            )
            validation_rows.append((step, tokens_seen, validation_loss))

    train_steps = tuple(row[0] for row in train_rows)
    validation_steps = tuple(row[0] for row in validation_rows)
    if train_steps != EXPECTED_TRAIN_LOG_STEPS:
        raise ValueError(f"A1 learning-rate log cadence mismatch: {events_path}")
    if validation_steps != EXPECTED_VALIDATION_STEPS:
        raise ValueError(f"A1 validation cadence mismatch: {events_path}")
    if not math.isclose(
        validation_rows[-1][2],
        point.final_validation_loss,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"A1 final validation event disagrees with metrics: {events_path}")

    metrics = _load_json_mapping(run_dir / "metrics.json")
    final_learning_rate = metrics.get("training/learning_rate_final")
    if (
        isinstance(final_learning_rate, bool)
        or not isinstance(final_learning_rate, int | float)
        or not math.isfinite(float(final_learning_rate))
        or float(final_learning_rate) <= 0.0
    ):
        raise ValueError(f"A1 final learning-rate metric is invalid: {run_dir}")
    if not math.isclose(
        train_rows[-1][2],
        float(final_learning_rate),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"A1 final learning-rate event disagrees with metrics: {events_path}")
    if not math.isclose(
        train_rows[0][2],
        source.learning_rate / EXPECTED_WARMUP_STEPS,
        rel_tol=1e-12,
        abs_tol=0.0,
    ) or not math.isclose(
        train_rows[-1][2],
        source.learning_rate * 0.1,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(f"A1 logged learning-rate endpoints mismatch: {events_path}")

    trace = A1ProgressTrace(
        config_id=source.config_id,
        run_id=source.run_id,
        peak_learning_rate=source.learning_rate,
        validation_steps=validation_steps,
        validation_tokens_seen=tuple(row[1] for row in validation_rows),
        validation_losses=tuple(row[2] for row in validation_rows),
        learning_rate_steps=train_steps,
        learning_rate_tokens_seen=tuple(row[1] for row in train_rows),
        effective_learning_rates=tuple(row[2] for row in train_rows),
    )
    events_input = {
        "config_id": source.config_id,
        "run_id": source.run_id,
        "role": "events",
        "path": _relative(events_path, root),
        "sha256": _sha256(events_path),
        "size_bytes": events_path.stat().st_size,
    }
    return trace, events_input


def _load_jsonl_mappings(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at row {row_index}: {path}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at row {row_index}: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"A1 progress event stream is empty: {path}")
    return tuple(rows)


def _positive_event_integer(
    row: dict[str, Any],
    field: str,
    row_index: int,
    path: Path,
) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"A1 progress {field} must be a positive integer at row "
            f"{row_index}: {path}"
        )
    return value


def _finite_event_number(
    row: dict[str, Any],
    field: str,
    row_index: int,
    path: Path,
) -> float:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"A1 progress {field} must be finite at row {row_index}: {path}"
        )
    return float(value)


def _validate_progress_trace_for_rendering(trace: A1ProgressTrace) -> None:
    validation_lengths = {
        len(trace.validation_steps),
        len(trace.validation_tokens_seen),
        len(trace.validation_losses),
    }
    learning_rate_lengths = {
        len(trace.learning_rate_steps),
        len(trace.learning_rate_tokens_seen),
        len(trace.effective_learning_rates),
    }
    if validation_lengths != {len(EXPECTED_VALIDATION_STEPS)}:
        raise ValueError(f"Incomplete A1 validation trace: {trace.config_id}.")
    if learning_rate_lengths != {len(EXPECTED_TRAIN_LOG_STEPS)}:
        raise ValueError(f"Incomplete A1 learning-rate trace: {trace.config_id}.")
    if any(not math.isfinite(value) for value in trace.validation_losses):
        raise ValueError(f"Non-finite A1 validation trace: {trace.config_id}.")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in trace.effective_learning_rates
    ):
        raise ValueError(f"Invalid A1 learning-rate trace: {trace.config_id}.")


def _progress_series_style(index: int) -> SeriesStyle:
    """Return eleven high-contrast identities without yellow on white."""

    if not 0 <= index < len(A1_SOURCES):
        raise ValueError("A1 progress style index is outside the pinned cohort.")
    if index < 7:
        color_index = index
        marker_index = index
        linestyle = "-"
    else:
        repeated_index = index - 7
        color_index = repeated_index
        marker_index = repeated_index + 1
        linestyle = "--"
    return SeriesStyle(
        color=COLORBLIND_SAFE_COLORS[color_index],
        marker=SERIES_MARKERS[marker_index],
        linestyle=linestyle,
        linewidth=1.2,
    )


def _validate_source_registry() -> None:
    if len(A1_SOURCES) != 11:
        raise ValueError("The A1 figure registry must contain exactly eleven sources.")
    if len({source.config_id for source in A1_SOURCES}) != len(A1_SOURCES):
        raise ValueError("The A1 figure registry contains duplicate config IDs.")
    if len({source.run_id for source in A1_SOURCES}) != len(A1_SOURCES):
        raise ValueError("The A1 figure registry contains duplicate run IDs.")
    expected_prefixes = tuple(f"{index:03d}" for index in range(1, 12))
    prefixes = tuple(source.config_id.split("-", 1)[0] for source in A1_SOURCES)
    if prefixes != expected_prefixes:
        raise ValueError("The A1 figure registry must cover config prefixes 001-011.")
    learning_rates = tuple(source.learning_rate for source in A1_SOURCES)
    if any(
        not math.isclose(current, previous * 2.0, rel_tol=0.0, abs_tol=0.0)
        for previous, current in zip(
            learning_rates[:-1], learning_rates[1:], strict=True
        )
    ):
        raise ValueError("The A1 figure registry must preserve the factor-two LR grid.")
    for source in A1_SOURCES:
        if (
            source.terminal_status != "completed"
            or source.case_class != "eligible"
            or source.evidence_status != "valid"
        ):
            raise ValueError(
                f"A1 source is not completed, eligible, and valid: {source.config_id}."
            )


def _validate_config(
    source: A1Source,
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    identity = _mapping(config.get("identity"), "config.identity", run_dir)
    run = _mapping(config.get("run"), "config.run", run_dir)
    training = _mapping(config.get("training"), "config.training", run_dir)
    preprocessing = _mapping(
        config.get("preprocessing"), "config.preprocessing", run_dir
    )
    validation = _mapping(config.get("validation"), "config.validation", run_dir)
    if identity.get("group_id") != GROUP_ID:
        raise ValueError(f"A1 case-group mismatch: {run_dir}")
    if identity.get("condition_fingerprint") != source.condition_fingerprint:
        raise ValueError(f"A1 condition fingerprint mismatch: {run_dir}")
    if identity.get("training_implementation_id") != EXPECTED_IMPLEMENTATION_ID:
        raise ValueError(f"A1 training implementation mismatch: {run_dir}")
    if complete_config_sha256(config) != source.config_sha256:
        raise ValueError(f"A1 config SHA-256 mismatch: {run_dir}")
    if run.get("seed") != 0 or run.get("model_initialization_seed") != 0:
        raise ValueError(f"A1 seed mismatch: {run_dir}")
    if run.get("data_order_seed") != 0:
        raise ValueError(f"A1 data-order seed mismatch: {run_dir}")
    if run.get("training_schedule_hash") != EXPECTED_SCHEDULE_HASH:
        raise ValueError(f"A1 training schedule mismatch: {run_dir}")
    if float(training.get("learning_rate", math.nan)) != source.learning_rate:
        raise ValueError(f"A1 learning-rate mismatch: {run_dir}")
    if training.get("max_steps") != EXPECTED_STEPS:
        raise ValueError(f"A1 optimizer-step budget mismatch: {run_dir}")
    block_size = preprocessing.get("block_size")
    micro_batch = training.get("micro_batch_size")
    accumulation = training.get("gradient_accumulation_steps")
    if (
        isinstance(block_size, bool)
        or isinstance(micro_batch, bool)
        or isinstance(accumulation, bool)
        or not all(isinstance(value, int) for value in (block_size, micro_batch, accumulation))
        or block_size * micro_batch * accumulation != EXPECTED_TOKENS_PER_STEP
    ):
        raise ValueError(f"A1 global token batch mismatch: {run_dir}")
    if EXPECTED_STEPS * EXPECTED_TOKENS_PER_STEP != EXPECTED_TRAINING_TOKENS:
        raise AssertionError("Internal A1 budget constants disagree.")
    if (
        validation.get("partition") != "selection"
        or validation.get("partition_hash") != EXPECTED_SELECTION_HASH
    ):
        raise ValueError(f"A1 selection-validation partition mismatch: {run_dir}")


def _validate_manifest(
    source: A1Source,
    manifest: dict[str, Any],
    run_dir: Path,
) -> None:
    expected_config_path = f"experiments/{TRANCHE_ID}/run/{source.config_id}.yaml"
    expected_result_path = (
        f"experiments/{TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
    )
    expected = {
        "status": source.terminal_status,
        "mode": "pretrain",
        "tranche_id": TRANCHE_ID,
        "config_id": source.config_id,
        "run_id": source.run_id,
        "git_commit": source.git_commit,
        "git_dirty": False,
        "config_sha256": source.config_sha256,
        "condition_fingerprint": source.condition_fingerprint,
        "case_group_id": GROUP_ID,
        "training_implementation_id": EXPECTED_IMPLEMENTATION_ID,
        "config_path": expected_config_path,
        "result_path": expected_result_path,
        "seed": 0,
        "model_initialization_seed": 0,
        "data_order_seed": 0,
        "training_schedule_hash": EXPECTED_SCHEDULE_HASH,
        "validation_partition": "selection",
        "validation_partition_hash": EXPECTED_SELECTION_HASH,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if manifest.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"A1 manifest provenance mismatch ({', '.join(mismatches)}): {run_dir}"
        )
    training = _mapping(manifest.get("training"), "manifest.training", run_dir)
    if (
        training.get("completed_steps") != EXPECTED_STEPS
        or training.get("max_steps") != EXPECTED_STEPS
        or training.get("tokens_per_step") != EXPECTED_TOKENS_PER_STEP
        or training.get("stopped_by_operational_wall_time_limit") is not False
    ):
        raise ValueError(f"A1 terminal training envelope mismatch: {run_dir}")


def _validate_metrics(metrics: dict[str, Any], run_dir: Path) -> float:
    checks = {
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
    mismatches = [
        field
        for field, expected_value in checks.items()
        if metrics.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(f"A1 metric budget mismatch ({', '.join(mismatches)}): {run_dir}")
    if metrics.get("training/validation_complete_block_coverage") is not True:
        raise ValueError(f"A1 validation coverage is incomplete: {run_dir}")
    if metrics.get("training/wall_time_limit_reached") is not False:
        raise ValueError(f"A1 run stopped at an operational wall-time limit: {run_dir}")
    loss_value = metrics.get("training/validation_loss_final")
    if isinstance(loss_value, bool) or not isinstance(loss_value, int | float):
        raise ValueError(f"A1 final validation loss is not numeric: {run_dir}")
    loss = float(loss_value)
    if not math.isfinite(loss):
        raise ValueError(f"A1 final validation loss is not finite: {run_dir}")
    return loss


def _build_provenance(
    *,
    root: Path,
    points: Sequence[A1Point],
    inputs: Sequence[dict[str, Any]],
    selected: A1Point,
    staged_outputs: Sequence[Path],
    final_outputs: Sequence[Path],
) -> dict[str, Any]:
    selected_is_upper_boundary = selected.learning_rate == max(
        point.learning_rate for point in points
    )
    module_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "suite_id": "a1-learning-rate-screen",
        "evidence_level": "exploratory",
        "metric": "training/validation_loss_final",
        "metric_step": EXPECTED_STEPS,
        "selection_rule": (
            "lowest finite final validation loss; exact tie favors lower peak LR"
        ),
        "claim_boundary": (
            "Best tested peak learning rate at the fixed 400M-token A1 horizon; "
            "one seed per learning rate; the selected point is the upper tested "
            "boundary, not a global or horizon-independent optimum."
            if selected_is_upper_boundary
            else "Best tested peak learning rate at the fixed 400M-token A1 horizon; "
            "one seed per learning rate; not a global or horizon-independent optimum."
        ),
        "cohort": {
            "tranche_id": TRANCHE_ID,
            "case_group_id": GROUP_ID,
            "cell_count": len(points),
            "seed_count_per_cell": 1,
            "seed": 0,
            "optimizer_steps": EXPECTED_STEPS,
            "tokens_per_step": EXPECTED_TOKENS_PER_STEP,
            "training_tokens": EXPECTED_TRAINING_TOKENS,
            "validation_tokens": EXPECTED_VALIDATION_TOKENS,
        },
        "selected": {
            "config_id": selected.config_id,
            "run_id": selected.run_id,
            "learning_rate": selected.learning_rate,
            "final_validation_loss": selected.final_validation_loss,
            "upper_tested_boundary": selected_is_upper_boundary,
        },
        "points": [asdict(point) for point in points],
        "inputs": list(inputs),
        "generator": {
            "path": GENERATOR_PATH,
            "sha256": _sha256(module_path),
        },
        "outputs": [
            {
                "path": _relative(final_path, root),
                "sha256": _sha256(staged_path),
                "size_bytes": staged_path.stat().st_size,
            }
            for staged_path, final_path in zip(
                staged_outputs, final_outputs, strict=True
            )
        ],
    }


def _build_progress_provenance(
    *,
    root: Path,
    traces: Sequence[A1ProgressTrace],
    inputs: Sequence[dict[str, Any]],
    staged_outputs: Sequence[Path],
    final_outputs: Sequence[Path],
) -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "figure_id": "a1-training-progress",
        "evidence_level": "exploratory",
        "claim_boundary": (
            "Descriptive logged training progress at the fixed 400M-token A1 "
            "horizon; one seed per learning rate; no uncertainty estimate or "
            "horizon-independent convergence claim."
        ),
        "cohort": {
            "tranche_id": TRANCHE_ID,
            "case_group_id": GROUP_ID,
            "cell_count": len(traces),
            "seed_count_per_cell": 1,
            "seed": 0,
            "optimizer_steps": EXPECTED_STEPS,
            "tokens_per_step": EXPECTED_TOKENS_PER_STEP,
            "training_tokens": EXPECTED_TRAINING_TOKENS,
            "validation_tokens_per_evaluation": EXPECTED_VALIDATION_TOKENS,
        },
        "panels": {
            "learning_trajectories": {
                "event": "validation",
                "x": "tokens_seen",
                "y": "validation_loss",
                "point_count_per_run": len(EXPECTED_VALIDATION_STEPS),
            },
            "learning_rate_schedules": {
                "event": "train",
                "x": "tokens_seen",
                "y": "learning_rate",
                "y_scale": "log_base_2",
                "point_count_per_run": len(EXPECTED_TRAIN_LOG_STEPS),
            },
        },
        "traces": [
            {
                "config_id": trace.config_id,
                "run_id": trace.run_id,
                "peak_learning_rate": trace.peak_learning_rate,
                "validation_point_count": len(trace.validation_steps),
                "learning_rate_point_count": len(trace.learning_rate_steps),
                "final_validation_loss": trace.validation_losses[-1],
                "final_effective_learning_rate": trace.effective_learning_rates[-1],
            }
            for trace in traces
        ],
        "inputs": list(inputs),
        "generator": {
            "path": GENERATOR_PATH,
            "sha256": _sha256(module_path),
        },
        "outputs": [
            {
                "path": _relative(final_path, root),
                "sha256": _sha256(staged_path),
                "size_bytes": staged_path.stat().st_size,
            }
            for staged_path, final_path in zip(
                staged_outputs, final_outputs, strict=True
            )
        ],
    }


def _repository_root(repository: str | Path | None) -> Path:
    return (
        Path(repository).resolve()
        if repository is not None
        else Path(__file__).resolve().parents[3]
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _mapping(value: Any, field: str, run_dir: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {field} to be a mapping: {run_dir}")
    return value


def _plain_lr_label(value: float) -> str:
    exponent = math.floor(math.log10(value))
    coefficient = value / (10**exponent)
    coefficient_text = f"{coefficient:g}"
    return f"{coefficient_text}e{exponent}"


def _math_lr_label(value: float) -> str:
    exponent = math.floor(math.log10(value))
    coefficient = value / (10**exponent)
    return rf"${coefficient:g}\times10^{{{exponent}}}$"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"A1 provenance path is outside the repository: {path}") from error


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
    parser = argparse.ArgumentParser(
        description="Generate a pinned eleven-cell A1 figure suite."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help="Repository root (defaults to the checkout containing this module).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Generate the two-panel training-progress figure instead of figure 01.",
    )
    args = parser.parse_args(argv)
    root = _repository_root(args.repository)
    outputs = (
        generate_a1_training_progress(root)
        if args.progress
        else generate_a1_lr_screen(root)
    )
    for output in outputs:
        print(_relative(output, root))


if __name__ == "__main__":
    main()
