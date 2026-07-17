from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pytest

from paper_exp import plot_report05
from paper_exp.plot_api import REPORT04_PUBLICATION_PROFILE, publication_figure_issues
from paper_exp.plot_style import REPORT04_PLOT_STYLE


def test_report05_cohort_constants_pin_the_four_architecture_families() -> None:
    assert plot_report05.REPORT05_STOCK_RUN == (
        "GELU AdamW",
        "50-pythia-14m-minipile-adamw-full-pass",
    )
    assert tuple(
        experiment_id.split("-", 1)[0]
        for _label, experiment_id in plot_report05.REPORT05_ONE_RELU_RUNS
    ) == ("77", "79", "81")
    assert tuple(
        experiment_id.split("-", 1)[0]
        for _label, experiment_id in plot_report05.REPORT05_THREE_RELU_RUNS
    ) == ("98", "103", "99")
    assert tuple(
        experiment_id.split("-", 1)[0]
        for _label, experiment_id in plot_report05.REPORT05_PRE_RUNS
    ) == ("107", "108", "109")
    assert tuple(
        experiment_id.split("-", 1)[0]
        for _label, experiment_id in plot_report05.REPORT05_POST_RUNS
    ) == ("110", "111", "112")
    assert len(plot_report05.REPORT05_TRAINING_RUNS) == 13
    assert len({label for label, _experiment_id in plot_report05.REPORT05_TRAINING_RUNS}) == 13
    assert len(
        {experiment_id for _label, experiment_id in plot_report05.REPORT05_TRAINING_RUNS}
    ) == 13


def test_report05_endpoint_rows_are_ordered_numeric_and_relative_to_stock() -> None:
    series = _synthetic_report05_series()

    rows = plot_report05._report05_endpoint_rows(series)
    table = plot_report05._report05_endpoint_table(series)

    assert len(rows) == 13
    assert rows[0]["architecture"] == "Stock (GELU)"
    assert rows[0]["method"] == "AdamW"
    assert rows[0]["step"] == 22_762
    assert rows[0]["tokens_seen"] == 1_491_730_432
    assert rows[0]["validation_loss"] == pytest.approx(5.0)
    assert rows[0]["delta_vs_stock"] == pytest.approx(0.0)
    assert [row["architecture_id"] for row in rows[1:4]] == ["one_relu"] * 3
    assert [row["method"] for row in rows[1:4]] == ["AdamW", "OR", "OL1"]
    assert rows[-1]["label"] == "Six-ReLU POST OL1"
    assert rows[-1]["delta_vs_stock"] == pytest.approx(0.6)
    assert table[0] == ("Stock (GELU)", "AdamW", 5.0, 0.0)
    assert table[-1] == pytest.approx(
        ("Six-ReLU POST", "OL1", 5.6, 0.6),
    )


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_report05_endpoint_rows_reject_incomplete_or_ambiguous_cohorts(
    failure: str,
) -> None:
    series = _synthetic_report05_series()
    if failure == "missing":
        series.pop()
    else:
        series.append(dict(series[-1]))

    with pytest.raises(ValueError, match=failure):
        plot_report05._report05_endpoint_rows(series)


def test_report05_validation_point_reduction_sorts_and_drops_nonfinite_rows() -> None:
    item = {
        "label": "Example",
        "validation_events": [
            {"step": 2, "tokens_seen": 200, "validation_loss": 4.0},
            {"step": 3, "tokens_seen": 300, "validation_loss": float("nan")},
            {"step": 1, "tokens_seen": 100, "validation_loss": 5.0},
            {"step": 4, "tokens_seen": -1, "validation_loss": 3.0},
            {"event": "validation"},
        ],
    }

    points = plot_report05._report05_validation_points(item)

    assert [point["tokens_seen"] for point in points] == [100, 200]
    assert [point["validation_loss"] for point in points] == [5.0, 4.0]


def test_report05_architecture_schematic_is_publication_sized_and_explicit() -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report05._plot_report05_architecture_schematic()
    try:
        panel_text = [
            "\n".join(text.get_text() for text in axis.texts)
            for axis in figure.axes
        ]

        assert len(figure.axes) == 4
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert figure.get_figheight() <= 8.8
        assert [axis.get_title(loc="left").split(")", 1)[-1].strip() for axis in figure.axes] == [
            "One-ReLU (MLP hidden)",
            "Three-ReLU",
            "Six-ReLU PRE",
            "Six-ReLU POST",
        ]
        assert "split -> ReLU(Q,K,V)\nRoPE(Q,K)" in panel_text[2]
        assert "split -> RoPE(Q,K)\nReLU(Q,K); ReLU(V)" in panel_text[3]
        assert all("QKV\n128 -> 384" in text for text in panel_text)
        assert all("$W_1$\n128 -> 512" in text for text in panel_text)
        assert all("$W_2$\n512 -> 128" in text for text in panel_text)
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def test_report05_learning_curves_use_four_shared_rows_and_common_reference() -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report05._plot_report05_validation_learning_curves(
            _synthetic_report05_series()
        )
    try:
        axes = figure.axes

        assert len(axes) == 4
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert figure.get_figheight() <= 8.8
        assert all(axes[0].get_shared_x_axes().joined(axes[0], axis) for axis in axes[1:])
        assert all(axes[0].get_shared_y_axes().joined(axes[0], axis) for axis in axes[1:])
        assert all(len(axis.lines) == 4 for axis in axes)
        assert all(
            [line.get_label() for line in axis.lines]
            == ["Stock GELU AdamW", "AdamW", "OR", "OL1"]
            for axis in axes
        )
        assert len(figure.legends) == 1
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Stock GELU AdamW",
            "AdamW",
            "OR",
            "OL1",
        ]
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def _synthetic_report05_series() -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, (label, _experiment_id) in enumerate(
        plot_report05.REPORT05_TRAINING_RUNS
    ):
        final_loss = 5.0 + 0.05 * index
        series.append(
            {
                "label": label,
                "validation_events": [
                    {
                        "event": "validation",
                        "step": 22_762,
                        "tokens_seen": 1_491_730_432,
                        "validation_loss": final_loss,
                        "validation_tokens": 692_224,
                    },
                    {
                        "event": "validation",
                        "step": 1,
                        "tokens_seen": 65_536,
                        "validation_loss": 10.8 + 0.01 * index,
                        "validation_tokens": 692_224,
                    },
                    {
                        "event": "validation",
                        "step": 11_000,
                        "tokens_seen": 720_896_000,
                        "validation_loss": 6.4 + 0.03 * index,
                        "validation_tokens": 692_224,
                    },
                    {
                        "event": "validation",
                        "step": 22_763,
                        "tokens_seen": 1_491_730_433,
                        "validation_loss": float("nan"),
                        "validation_tokens": 692_224,
                    },
                ],
            }
        )
    return series
