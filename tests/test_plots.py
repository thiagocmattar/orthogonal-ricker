from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper_exp.plots.a1_lr_screen import (
    A1Point,
    A1_SOURCES,
    build_a1_lr_figure,
    generate_a1_lr_screen,
    load_a1_lr_points,
    select_a1_point,
)
from paper_exp.plots.export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    publication_figure_issues,
)
from paper_exp.plots import (
    build_activation_histograms,
    build_activation_propagation,
    build_clipping_frontier,
    build_weight_histograms,
    plot_artifact,
)


def test_clipping_plot_uses_one_explicit_run_and_exports_once(tmp_path: Path) -> None:
    scaffold = _scaffold(tmp_path, "01-clipping-tests")
    run_dir = scaffold / "raw" / "001-clipping" / "001-test"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("experiment_name: clipping\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tranche_id": "01-clipping-tests",
                "config_id": "001-clipping",
                "run_id": "001-test",
                "status": "completed",
                "git_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "threshold": 0.0,
            "achieved_sparsity": 0.0,
            "validation_loss": 7.60,
            "validation_tokens": 8192,
        },
        {
            "threshold": 0.01,
            "achieved_sparsity": 0.1,
            "validation_loss": 7.61,
            "validation_tokens": 8192,
        },
    ]
    _write_jsonl(run_dir / "clipping_frontier.jsonl", rows)

    outputs = plot_artifact(
        kind="clipping",
        run_dir=run_dir,
        output=scaffold / "figs" / "01-clipping.pdf",
        save_png=True,
        repository=tmp_path,
    )

    assert outputs == [
        scaffold / "figs" / "01-clipping.pdf",
        scaffold / "figs" / "01-clipping.png",
        scaffold / "figs" / "01-clipping.provenance.json",
    ]
    assert all(path.is_file() for path in outputs)
    provenance = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert provenance["plot_kind"] == "clipping"
    assert provenance["source"]["tranche_id"] == "01-clipping-tests"
    assert provenance["source"]["config_id"] == "001-clipping"
    assert {row["path"] for row in provenance["outputs"]} == {
        outputs[0].name,
        outputs[1].name,
    }
    assert {row["size_bytes"] for row in provenance["outputs"]} == {
        outputs[0].stat().st_size,
        outputs[1].stat().st_size,
    }


def test_activation_histogram_pools_counts_and_separates_zero_atom() -> None:
    payload = {
        "bin_edges": [-1.0, 0.0, 1.0],
        "methods": [
            {
                "label": "method",
                "layers": [
                    {
                        "counts": [2, 8],
                        "total": 10,
                        "threshold_hits": {"0": 3},
                    },
                    {
                        "counts": [1, 9],
                        "total": 10,
                        "threshold_hits": {"0": 1},
                    },
                ],
            }
        ],
    }

    figure = build_activation_histograms(payload)
    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].patches[0].get_height() == 20.0
        assert "given nonzero" in figure.axes[1].get_title().lower()
    finally:
        plt.close(figure)


def test_activation_propagation_pools_integer_counts_across_layers() -> None:
    payload = {
        "activation_stage_order": ["hidden"],
        "activation_stage_labels": {"hidden": "Hidden"},
        "matmul_stage_order": ["w2"],
        "matmul_stage_labels": {"w2": "W2"},
        "methods": [
            {
                "label": "method",
                "activations": [
                    {"name": "hidden", "layer": 0, "zero_count": 1, "total": 4},
                    {"name": "hidden", "layer": 1, "zero_count": 3, "total": 6},
                ],
                "matmuls": [
                    {"name": "w2", "layer": 0, "zero_count": 10, "total": 20},
                    {"name": "w2", "layer": 1, "zero_count": 20, "total": 30},
                ],
            }
        ],
    }

    figure = build_activation_propagation(payload)
    try:
        image_axes = [axis for axis in figure.axes if axis.images]
        assert image_axes[0].images[0].get_array().item() == 40.0
        assert image_axes[1].images[0].get_array().item() == 60.0
    finally:
        plt.close(figure)


def test_activation_propagation_masks_explicitly_unavailable_stages() -> None:
    payload = {
        "activation_stage_order": ["optional_gate"],
        "activation_stage_labels": {"optional_gate": "Optional gate"},
        "matmul_stage_order": ["w2"],
        "matmul_stage_labels": {"w2": "W2"},
        "methods": [
            {
                "label": "baseline",
                "activations": [
                    {
                        "name": "optional_gate",
                        "layer": 0,
                        "zero_count": None,
                        "total": None,
                    }
                ],
                "matmuls": [
                    {"name": "w2", "layer": 0, "zero_count": 0, "total": 4}
                ],
            }
        ],
    }

    figure = build_activation_propagation(payload)
    try:
        image_axes = [axis for axis in figure.axes if axis.images]
        assert image_axes[0].images[0].get_array().mask.item() is True
        assert any(text.get_text() == "N/A" for text in image_axes[0].texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("sparsity", [-0.01, 1.01, float("nan")])
def test_clipping_frontier_rejects_invalid_sparsity(sparsity: float) -> None:
    with pytest.raises(ValueError, match="achieved_sparsity"):
        build_clipping_frontier(
            [{"achieved_sparsity": sparsity, "validation_loss": 1.0}]
        )


def test_weight_histogram_handles_no_in_range_mass_without_log_axis() -> None:
    payload = {
        "bin_edges": [-1.0, 0.0, 1.0],
        "methods": [
            {
                "label": "outside",
                "layers": [{"counts": [0, 0], "total": 5}],
            }
        ],
    }

    figure = build_weight_histograms(payload)
    try:
        assert figure.axes[0].get_yscale() == "linear"
        assert any("No in-range" in text.get_text() for text in figure.axes[0].texts)
    finally:
        plt.close(figure)


def test_versioned_plot_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    scaffold = _scaffold(tmp_path, "01-histogram-tests")
    run_dir = scaffold / "raw" / "001-histogram" / "001-test"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("experiment_name: histogram\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tranche_id": "01-histogram-tests",
                "config_id": "001-histogram",
                "run_id": "001-test",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "activation_histograms.json").write_text(
        json.dumps({"schema_version": 999}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        plot_artifact(
            kind="activation-histograms",
            run_dir=run_dir,
            output=scaffold / "figs" / "01-histogram.pdf",
            repository=tmp_path,
        )


def test_plot_output_must_stay_in_the_source_scaffold(tmp_path: Path) -> None:
    source = _scaffold(tmp_path, "01-source-tests")
    other = _scaffold(tmp_path, "02-other-tests")
    run_dir = source / "raw" / "001-source" / "001-run"
    run_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="source run's scaffold"):
        plot_artifact(
            kind="run",
            run_dir=run_dir,
            output=other / "figs" / "01-wrong.pdf",
            repository=tmp_path,
        )


def test_full_propagation_diagnostic_fits_the_publication_canvas() -> None:
    activation_order = [f"activation_{index}" for index in range(21)]
    matmul_order = [f"matmul_{index}" for index in range(6)]
    payload = {
        "activation_stage_order": activation_order,
        "activation_stage_labels": {
            name: f"Recorded activation stage {index}"
            for index, name in enumerate(activation_order)
        },
        "matmul_stage_order": matmul_order,
        "matmul_stage_labels": {
            name: f"Logical operation {index}"
            for index, name in enumerate(matmul_order)
        },
        "methods": [
            {
                "label": f"method {method_index}",
                "activations": [
                    {
                        "name": name,
                        "layer": 0,
                        "zero_count": method_index + index,
                        "total": 100,
                    }
                    for index, name in enumerate(activation_order)
                ],
                "matmuls": [
                    {
                        "name": name,
                        "layer": 0,
                        "zero_count": method_index + index,
                        "total": 100,
                    }
                    for index, name in enumerate(matmul_order)
                ],
            }
            for method_index in range(5)
        ],
    }

    figure = build_activation_propagation(payload)
    try:
        assert publication_figure_issues(
            figure,
            DOUBLE_COLUMN_PUBLICATION_PROFILE,
        ) == ()
    finally:
        plt.close(figure)


def test_a1_lr_selection_breaks_an_exact_loss_tie_toward_lower_lr() -> None:
    common = {
        "run_id": "001-test",
        "seed": 0,
        "optimizer_steps": 1526,
        "training_tokens": 400_031_744,
        "validation_tokens": 311_296,
        "terminal_status": "completed",
        "case_class": "eligible",
        "evidence_status": "valid",
    }
    lower = A1Point(
        config_id="001-lower",
        learning_rate=1e-3,
        final_validation_loss=4.0,
        **common,
    )
    higher = A1Point(
        config_id="002-higher",
        learning_rate=2e-3,
        final_validation_loss=4.0,
        **common,
    )

    assert select_a1_point((higher, lower)) == lower


def test_a1_lr_screen_publishes_complete_deterministic_suite(tmp_path: Path) -> None:
    _write_a1_lr_fixture(tmp_path)

    outputs = generate_a1_lr_screen(tmp_path)
    first_bytes = tuple(path.read_bytes() for path in outputs)
    points, inputs = load_a1_lr_points(tmp_path)
    figure = build_a1_lr_figure(points)
    try:
        assert figure.axes[0].get_xscale() == "log"
        assert len(figure.axes[0].lines[0].get_xdata()) == 8
        assert "zoomed" in figure.axes[0].get_ylabel()
        assert any(
            "upper tested boundary" in text.get_text() for text in figure.texts
        )
        assert publication_figure_issues(
            figure,
            DOUBLE_COLUMN_PUBLICATION_PROFILE,
        ) == ()
    finally:
        plt.close(figure)

    assert [path.name for path in outputs] == [
        "01-a1-learning-rate-screen.pdf",
        "01-a1-learning-rate-screen.png",
        "01-a1-learning-rate-screen.md",
        "01-a1-learning-rate-screen.provenance.json",
    ]
    assert len(points) == 8
    assert len(inputs) == 32
    table = outputs[2].read_text(encoding="utf-8")
    assert sum(line.startswith("| `") for line in table.splitlines()) == 8
    assert all(source.config_id in table for source in A1_SOURCES)
    assert all(source.run_id in table for source in A1_SOURCES)
    assert "selected (upper tested boundary)" in table

    provenance = json.loads(outputs[3].read_text(encoding="utf-8"))
    assert provenance["cohort"]["cell_count"] == 8
    assert provenance["cohort"]["seed_count_per_cell"] == 1
    assert provenance["selected"] == {
        "config_id": "008-a1-lr-6p4e-2",
        "final_validation_loss": 4.0587728086270785,
        "learning_rate": 0.064,
        "run_id": "001-20260826-190546-4df1c441",
        "upper_tested_boundary": True,
    }
    assert len(provenance["inputs"]) == 32
    assert len(provenance["outputs"]) == 3
    assert all(point["terminal_status"] == "completed" for point in provenance["points"])
    assert all(point["case_class"] == "eligible" for point in provenance["points"])
    assert all(point["evidence_status"] == "valid" for point in provenance["points"])

    assert generate_a1_lr_screen(tmp_path) == outputs
    assert tuple(path.read_bytes() for path in outputs) == first_bytes


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    [
        ("manifest.json", "status", "failed", "manifest provenance mismatch"),
        ("metrics.json", "training/tokens_seen", 1, "metric budget mismatch"),
        (
            "metrics.json",
            "training/validation_complete_block_coverage",
            False,
            "validation coverage is incomplete",
        ),
        (
            "metrics.json",
            "training/validation_loss_final",
            None,
            "final validation loss is not numeric",
        ),
    ],
)
def test_a1_lr_loader_rejects_ineligible_terminal_evidence(
    tmp_path: Path,
    artifact: str,
    field: str,
    value: object,
    message: str,
) -> None:
    _write_a1_lr_fixture(tmp_path)
    source = A1_SOURCES[0]
    path = (
        tmp_path
        / "experiments"
        / "01-a1-lr-screen"
        / "raw"
        / source.config_id
        / source.run_id
        / artifact
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_a1_lr_points(tmp_path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_a1_lr_fixture(repository: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    losses = (
        5.191815257072449,
        4.840334835805391,
        4.292332950391267,
        4.224078962677403,
        4.1769665165951375,
        4.112285005418878,
        4.082745991255107,
        4.0587728086270785,
    )
    for source, loss in zip(A1_SOURCES, losses, strict=True):
        recipe_source = (
            source_root
            / "experiments"
            / "01-a1-lr-screen"
            / "run"
            / f"{source.config_id}.yaml"
        )
        recipe_target = (
            repository
            / "experiments"
            / "01-a1-lr-screen"
            / "run"
            / recipe_source.name
        )
        run_dir = (
            repository
            / "experiments"
            / "01-a1-lr-screen"
            / "raw"
            / source.config_id
            / source.run_id
        )
        recipe_target.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_bytes = recipe_source.read_bytes()
        recipe_target.write_bytes(config_bytes)
        (run_dir / "config.yaml").write_bytes(config_bytes)
        manifest = {
            "status": "completed",
            "mode": "pretrain",
            "tranche_id": "01-a1-lr-screen",
            "config_id": source.config_id,
            "run_id": source.run_id,
            "git_commit": source.git_commit,
            "git_dirty": False,
            "config_sha256": source.config_sha256,
            "condition_fingerprint": source.condition_fingerprint,
            "case_group_id": "A1-lr-screen",
            "training_implementation_id": "a1_pretraining_v1",
            "config_path": (
                f"experiments/01-a1-lr-screen/run/{source.config_id}.yaml"
            ),
            "result_path": (
                "experiments/01-a1-lr-screen/raw/"
                f"{source.config_id}/{source.run_id}"
            ),
            "seed": 0,
            "model_initialization_seed": 0,
            "data_order_seed": 0,
            "training_schedule_hash": (
                "5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c"
            ),
            "validation_partition": "selection",
            "validation_partition_hash": (
                "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
            ),
            "training": {
                "completed_steps": 1526,
                "max_steps": 1526,
                "tokens_per_step": 262_144,
                "stopped_by_operational_wall_time_limit": False,
            },
        }
        metrics = {
            "training/optimizer_steps": 1526,
            "training/planned_optimizer_steps": 1526,
            "training/tokens_per_step": 262_144,
            "training/tokens_seen": 400_031_744,
            "training/validation_loss_final": loss,
            "training/validation_loss_final_step": 1526,
            "training/validation_tokens_final": 311_296,
            "training/validation_sequences_final": 152,
            "training/validation_available_complete_blocks": 152,
            "training/validation_batches_final": 38,
            "training/validation_partition": "selection",
            "training/validation_partition_hash": (
                "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
            ),
            "training/training_schedule_hash": (
                "5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c"
            ),
            "training/validation_complete_block_coverage": True,
            "training/wall_time_limit_reached": False,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )


def _scaffold(tmp_path: Path, scaffold_id: str) -> Path:
    scaffold = tmp_path / "experiments" / scaffold_id
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    return scaffold
