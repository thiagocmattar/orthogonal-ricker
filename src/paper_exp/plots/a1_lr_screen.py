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
from .style import PAPER_STYLE, series_style


TRANCHE_ID = "01-a1-lr-screen"
GROUP_ID = "A1-lr-screen"
OUTPUT_STEM = "01-a1-learning-rate-screen"
GENERATOR_PATH = "src/paper_exp/plots/a1_lr_screen.py"
EXPECTED_STEPS = 1_526
EXPECTED_TOKENS_PER_STEP = 262_144
EXPECTED_TRAINING_TOKENS = 400_031_744
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
    selected = select_a1_point(ordered)
    learning_rates = [point.learning_rate for point in ordered]
    losses = [point.final_validation_loss for point in ordered]

    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.4))
    sweep_style = series_style(0)
    selected_style = series_style(1)
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
    axis.scatter(
        [selected.learning_rate],
        [selected.final_validation_loss],
        color=selected_style.color,
        marker="D",
        s=50,
        zorder=3,
    )

    axis.set_title("Pythia-14M A1 learning-rate screen (400M training tokens)")
    axis.set_xlabel(r"Peak learning rate (log$_2$ scale)")
    axis.set_ylabel("Final selection validation loss (zoomed y-axis)")
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
        label = f"{point.final_validation_loss:.3f}"
        if point == selected:
            label += "  selected"
        offset = (10, -5) if point == selected else (0, 9)
        axis.annotate(
            label,
            (point.learning_rate, point.final_validation_loss),
            xytext=offset,
            textcoords="offset points",
            ha="left" if point == selected else "center",
            va="top" if point == selected else "bottom",
            fontsize=8,
        )

    selected_is_upper_boundary = selected == ordered[-1]
    boundary_note = (
        "Selected point is the upper tested boundary; best tested, not a global optimum."
        if selected_is_upper_boundary
        else "Selected point lies inside the tested range."
    )
    figure.text(
        0.5,
        0.055,
        "Seed 0; n = 1 per learning rate; fixed 400,031,744-token horizon; "
        "lower is better.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.text(
        0.5,
        0.018,
        boundary_note,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.28)
    return figure


def build_a1_lr_table(points: Sequence[A1Point]) -> str:
    """Return the complete human-readable A1 result table."""

    ordered = tuple(sorted(points, key=lambda point: point.learning_rate))
    selected = select_a1_point(ordered)
    selected_is_upper_boundary = selected == ordered[-1]
    rows = [
        "# A1 learning-rate screen",
        "",
        (
            "Pythia-14M, seed 0, one run per learning rate, 1,526 optimizer "
            "updates and 400,031,744 training tokens per run. Lower final "
            "selection validation loss is better."
        ),
        "",
        (
            "| Config | Pinned run | Peak LR | Seed | Updates | Training tokens | "
            "Final selection loss | Terminal status | Eligibility | Evidence | Selection |"
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
                "input hashes are retained in the provenance sidecar. With n = 1 per "
                "learning rate, no uncertainty estimate is available."
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
        description="Generate the pinned eleven-cell A1 learning-rate table and curve."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help="Repository root (defaults to the checkout containing this module).",
    )
    args = parser.parse_args(argv)
    root = _repository_root(args.repository)
    for output in generate_a1_lr_screen(root):
        print(_relative(output, root))


if __name__ == "__main__":
    main()
