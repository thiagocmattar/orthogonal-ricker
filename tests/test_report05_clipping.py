from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pytest

from paper_exp import plot_report05_clipping
from paper_exp.plot_api import REPORT04_PUBLICATION_PROFILE, publication_figure_issues
from paper_exp.plot_style import REPORT04_PLOT_STYLE


def test_site_reduction_uses_each_sweeps_own_threshold_zero_loss() -> None:
    inputs = _synthetic_site_sweeps()
    adamw_rows = inputs["one_relu"]["mlp_hiddens"][0]["rows"]
    or_rows = inputs["one_relu"]["mlp_hiddens"][1]["rows"]
    adamw_rows[0]["validation_loss"] = 5.0
    adamw_rows[1]["validation_loss"] = 5.2
    or_rows[0]["validation_loss"] = 8.0
    or_rows[1]["validation_loss"] = 8.1

    reduced = plot_report05_clipping._reduce_report05_site_clipping_frontiers(inputs)
    one_relu = reduced[0]
    series = one_relu["sites"]["mlp_hiddens"]["series"]

    assert [item["method"] for item in series] == ["AdamW", "OR", "OL1"]
    assert series[0]["baseline_loss"] == pytest.approx(5.0)
    assert [point["loss_change"] for point in series[0]["points"]] == pytest.approx(
        [0.0, 0.2]
    )
    assert series[1]["baseline_loss"] == pytest.approx(8.0)
    assert [point["loss_change"] for point in series[1]["points"]] == pytest.approx(
        [0.0, 0.1]
    )


def test_site_reduction_reads_selected_site_fraction_and_sorts_thresholds() -> None:
    inputs = _synthetic_site_sweeps(reverse_rows=True)

    reduced = plot_report05_clipping._reduce_report05_site_clipping_frontiers(inputs)
    points = reduced[2]["sites"]["query_gate_outputs"]["series"][0]["points"]

    assert [point["threshold"] for point in points] == [0.0, 0.1]
    assert [point["x_percent"] for point in points] == pytest.approx([43.0, 68.0])
    assert [point["is_threshold_zero"] for point in points] == [True, False]


def test_site_reduction_rejects_missing_threshold_zero() -> None:
    inputs = _synthetic_site_sweeps()
    inputs["one_relu"]["mlp_hiddens"][0]["rows"] = [
        inputs["one_relu"]["mlp_hiddens"][0]["rows"][1]
    ]

    with pytest.raises(ValueError, match="threshold-0"):
        plot_report05_clipping._reduce_report05_site_clipping_frontiers(inputs)


def test_model_matmul_reduction_uses_direct_counts_not_projection_proxy() -> None:
    inputs = _synthetic_joint_sweeps()
    first_row = inputs["one_relu"][0]["rows"][0]
    first_row["eligible_projection_skip_fraction"] = 0.999

    reduced = plot_report05_clipping._reduce_report05_model_matmul_frontiers(inputs)
    points = reduced[0]["series"][0]["points"]

    assert [point["x_percent"] for point in points] == pytest.approx([10.0, 25.0])
    assert [point["loss_change"] for point in points] == pytest.approx([0.0, 0.2])


def test_model_matmul_reduction_rejects_inconsistent_saved_fraction() -> None:
    inputs = _synthetic_joint_sweeps()
    inputs["one_relu"][0]["rows"][0][
        "potentially_avoidable_model_matmul_fraction"
    ] = 0.7

    with pytest.raises(ValueError, match="disagrees with direct counters"):
        plot_report05_clipping._reduce_report05_model_matmul_frontiers(inputs)


def test_model_matmul_reduction_requires_exact_product_diagnostics() -> None:
    inputs = _synthetic_joint_sweeps()
    inputs["one_relu"][0]["rows"][0].pop("block_zero_product_count")

    with pytest.raises(ValueError, match="--measure-zero-products"):
        plot_report05_clipping._reduce_report05_model_matmul_frontiers(inputs)


def test_site_frontier_figure_has_four_rows_six_columns_and_shared_axes() -> None:
    reduced = plot_report05_clipping._reduce_report05_site_clipping_frontiers(
        _synthetic_site_sweeps()
    )
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report05_clipping._plot_report05_site_clipping_frontiers(reduced)
    try:
        axes = figure.axes
        assert len(axes) == 24
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert figure.get_figheight() <= 8.8
        assert all(axes[0].get_shared_x_axes().joined(axes[0], axis) for axis in axes[1:])
        assert all(axes[0].get_shared_y_axes().joined(axes[0], axis) for axis in axes[1:])
        assert len([text for axis in axes for text in axis.texts if text.get_text() == "N/A"]) == 8
        assert [axis.get_title() for axis in axes[:6]] == [
            "Attention input",
            "MLP input",
            "MLP hidden",
            "Q gate",
            "K gate",
            "V gate",
        ]
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "AdamW",
            "OR",
            "OL1",
        ]
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def test_model_matmul_figure_has_one_shared_row_per_architecture() -> None:
    reduced = plot_report05_clipping._reduce_report05_model_matmul_frontiers(
        _synthetic_joint_sweeps()
    )
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report05_clipping._plot_report05_model_matmul_frontiers(reduced)
    try:
        axes = figure.axes
        assert len(axes) == 4
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert figure.get_figheight() <= 8.8
        assert all(axes[0].get_shared_x_axes().joined(axes[0], axis) for axis in axes[1:])
        assert all(axes[0].get_shared_y_axes().joined(axes[0], axis) for axis in axes[1:])
        assert [axis.get_title(loc="left") for axis in axes] == [
            "One-ReLU",
            "Three-ReLU",
            "Six-ReLU PRE",
            "Six-ReLU POST",
        ]
        assert all(len(axis.lines) == 4 for axis in axes)
        assert all(len(axis.collections) == 3 for axis in axes)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "AdamW",
            "OR",
            "OL1",
            "Threshold 0",
        ]
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def _synthetic_site_sweeps(*, reverse_rows: bool = False) -> dict[str, Any]:
    active_sites = plot_report05_clipping.REPORT05_ACTIVE_CLIPPING_SITES
    result: dict[str, Any] = {}
    for architecture_index, (architecture_id, _label) in enumerate(
        plot_report05_clipping.REPORT05_CLIPPING_ARCHITECTURES
    ):
        result[architecture_id] = {}
        for site_index, site in enumerate(active_sites[architecture_id]):
            method_series = []
            for method_index, method in enumerate(
                plot_report05_clipping.REPORT05_CLIPPING_METHODS
            ):
                baseline_fraction = 0.40 + 0.01 * site_index + 0.02 * method_index
                rows = [
                    {
                        "event": "clipping_sweep",
                        "mode": "threshold",
                        "threshold": 0.0,
                        "validation_loss": 5.0 + architecture_index + method_index,
                        "validation_tokens": 692_224,
                        "sites": [site],
                        "achieved_sparsity": baseline_fraction,
                        "site_achieved_sparsity": {site: baseline_fraction},
                    },
                    {
                        "event": "clipping_sweep",
                        "mode": "threshold",
                        "threshold": 0.1,
                        "validation_loss": 5.2 + architecture_index + method_index,
                        "validation_tokens": 692_224,
                        "sites": [site],
                        "achieved_sparsity": baseline_fraction + 0.25,
                        "site_achieved_sparsity": {site: baseline_fraction + 0.25},
                    },
                ]
                if reverse_rows:
                    rows.reverse()
                method_series.append({"method": method, "rows": rows})
            result[architecture_id][site] = method_series
    return result


def _synthetic_joint_sweeps() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for architecture_index, (architecture_id, _label) in enumerate(
        plot_report05_clipping.REPORT05_CLIPPING_ARCHITECTURES
    ):
        result[architecture_id] = []
        for method_index, method in enumerate(
            plot_report05_clipping.REPORT05_CLIPPING_METHODS
        ):
            denominator = 1_000
            base_zero_count = 100 + 10 * architecture_index + 5 * method_index
            rows = []
            for threshold, extra_zero_count, loss_change in (
                (0.0, 0, 0.0),
                (0.1, 150, 0.2),
            ):
                zero_count = base_zero_count + extra_zero_count
                rows.append(
                    {
                        "event": "clipping_sweep",
                        "mode": "threshold",
                        "threshold": threshold,
                        "validation_loss": 5.0
                        + architecture_index
                        + method_index
                        + loss_change,
                        "validation_tokens": 692_224,
                        "block_zero_product_count": zero_count,
                        "model_matmul_product_count": denominator,
                        "potentially_avoidable_model_matmul_fraction": (
                            zero_count / denominator
                        ),
                    }
                )
            result[architecture_id].append({"method": method, "rows": rows})
    return result
