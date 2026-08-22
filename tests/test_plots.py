from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _scaffold(tmp_path: Path, scaffold_id: str) -> Path:
    scaffold = tmp_path / "experiments" / scaffold_id
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    return scaffold
