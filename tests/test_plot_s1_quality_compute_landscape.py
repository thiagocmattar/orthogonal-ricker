from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper_exp.plot_api import publication_figure_issues
from paper_exp.plot_report07 import load_s1_rows
from paper_exp.plot_s1_quality_compute_landscape import (
    QUALITY_COMPUTE_PROFILE,
    _central_screen_rows,
    _pareto_frontier,
    build_quality_compute_landscape_figure,
    generate_quality_compute_landscape_figure,
    write_frontier_table,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_s1_rows(
        ROOT / "docs/experimental-design/config-registry.yaml",
        ROOT / "docs/experimental-design/run-registry.yaml",
    )


def test_primary_landscape_uses_seed0_and_common_lr(cohort) -> None:
    central = _central_screen_rows(cohort)
    assert len(central) == 112
    assert {row.number for row in cohort}.difference(
        row.number for row in central
    ) == {*range(135, 145), *range(293, 303)}


def test_quality_compute_pareto_frontiers_use_observed_endpoints(cohort) -> None:
    assert [row.number for row in _pareto_frontier(cohort)] == [
        141,
        142,
        143,
        144,
        231,
        195,
        159,
        171,
        286,
        272,
    ]
    assert [
        row.number for row in _pareto_frontier(_central_screen_rows(cohort))
    ] == [
        124,
        204,
        238,
        205,
        237,
        206,
        236,
        231,
        195,
        159,
        171,
        286,
        272,
    ]


def test_quality_compute_landscape_contains_complete_and_zoom_clouds(
    cohort,
) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_quality_compute_landscape_figure(cohort)
        try:
            figure.canvas.draw()
            assert len(figure.axes) == 2
            assert sum(
                len(collection.get_offsets())
                for collection in figure.axes[0].collections
            ) == 132
            assert sum(
                len(collection.get_offsets())
                for collection in figure.axes[1].collections
            ) == 125
            assert figure.axes[1].get_ylim() == pytest.approx((6.975, 7.215))
            figure_text = " ".join(text.get_text() for text in figure.texts)
            assert "Potentially avoidable logical products" in figure_text
            assert "not a promotion rule" not in figure_text
            assert "tokens/endpoint" not in figure_text
            assert figure.get_size_inches() == pytest.approx((7.16, 3.35))
            assert figure._suptitle is None
            assert figure.axes[0].get_title(loc="left").startswith(
                r"$\mathbf{(a)}$"
            )
            assert figure.axes[1].get_title(loc="left").startswith(
                r"$\mathbf{(b)}$"
            )
            assert all(
                axis._left_title.get_fontweight() == "normal"
                for axis in figure.axes
            )
            assert (
                figure.axes[1].get_position().width
                / figure.axes[0].get_position().width
            ) == pytest.approx(1.25 / 0.75)
            assert len(figure.axes[0].lines) == 0
            assert len(figure.axes[1].lines) == 1
            assert figure.axes[1].lines[0].get_linewidth() == pytest.approx(0.9)
            assert [
                text.get_text()
                for text in figure.legends[0].get_texts()
            ] == [
                "B0 arch./LR",
                "B1 fixed",
                "B2 learned",
                "B3 pressure",
                "B4 seeds",
                "Descriptive nondominated envelope",
            ]
            assert figure.axes[1].get_legend() is None
            renderer = figure.canvas.get_renderer()
            assert (
                figure.legends[0].get_window_extent(renderer).y1
                < figure.axes[0].get_window_extent(renderer).y0
            )
            assert all(not axis.get_xlabel() for axis in figure.axes)
            assert all(not axis.get_ylabel() for axis in figure.axes)
            assert {text.get_text() for text in figure.axes[1].texts} == {
                f"F{index}" for index in range(1, 14)
            }
            assert publication_figure_issues(
                figure,
                QUALITY_COMPUTE_PROFILE,
            ) == ()
        finally:
            plt.close(figure)


def test_quality_compute_landscape_exports_pdf_and_png(tmp_path) -> None:
    output = tmp_path / "quality-compute-landscape.pdf"
    table_output = tmp_path / "frontier.tex"
    generated = generate_quality_compute_landscape_figure(
        output=output,
        table_output=table_output,
        save_png=True,
    )
    assert generated == (output, output.with_suffix(".png"))
    assert all(path.is_file() for path in generated)
    assert output.stat().st_size > 10_000
    assert table_output.is_file()
    table_text = table_output.read_text(encoding="utf-8")
    assert r"\begin{table}[H]" in table_text
    assert "F1 & 124" in table_text
    assert "F13 & 272" in table_text
    assert "n=112" in table_text
    assert "$R_b$" not in table_text
    assert "$U$" not in table_text
    assert "$R_m$" in table_text


def test_frontier_table_and_figure_share_exact_members(cohort, tmp_path) -> None:
    output = write_frontier_table(cohort, tmp_path / "frontier.tex")
    text = output.read_text(encoding="utf-8")
    expected = [
        124,
        204,
        238,
        205,
        237,
        206,
        236,
        231,
        195,
        159,
        171,
        286,
        272,
    ]
    assert [
        row.number for row in _pareto_frontier(_central_screen_rows(cohort))
    ] == expected
    assert all(f"F{index} & {number}" in text for index, number in enumerate(expected, 1))
