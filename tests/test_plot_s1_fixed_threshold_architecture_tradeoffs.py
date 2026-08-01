from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper_exp.plot_api import publication_figure_issues
from paper_exp.plot_report07 import load_s1_rows
from paper_exp.plot_s1_fixed_threshold_architecture_tradeoffs import (
    FIXED_THRESHOLD_FRONTIER_PROFILE,
    FIXED_THRESHOLD_TRADEOFF_PROFILE,
    build_fixed_threshold_architecture_tradeoff_figure,
    build_fixed_threshold_quality_opportunity_frontier_figure,
    generate_fixed_threshold_architecture_tradeoff_figure,
    generate_fixed_threshold_quality_opportunity_frontier_figure,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_s1_rows(
        ROOT / "docs/experimental-design/config-registry.yaml",
        ROOT / "docs/experimental-design/run-registry.yaml",
    )


def test_fixed_threshold_tradeoff_uses_four_shared_scale_panels(cohort) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_fixed_threshold_architecture_tradeoff_figure(cohort)
        try:
            figure.canvas.draw()
            assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 4.3))
            assert len(figure.axes) == 8
            loss_axes = figure.axes[:4]
            opportunity_axes = figure.axes[4:]
            assert [axis.get_title(loc="left") for axis in loss_axes] == [
                "(a) A5-QK-PRE",
                "(b) A5-QK-POST",
                "(c) A6-PRE-QKV",
                "(d) A6-POST-QKV",
            ]
            assert all(len(axis.lines) == 2 for axis in loss_axes)
            assert all(len(axis.lines) == 2 for axis in opportunity_axes)
            assert all(axis.get_ylim() == loss_axes[0].get_ylim() for axis in loss_axes)
            assert all(
                axis.get_ylim() == opportunity_axes[0].get_ylim()
                for axis in opportunity_axes
            )
            assert all(
                [tick.get_text() for tick in axis.get_xticklabels()]
                == ["0", "0.03", "0.10", "0.30"]
                for axis in loss_axes
            )
            assert [text.get_text() for text in figure.legends[0].get_texts()] == [
                "Validation loss",
                r"$R_{\mathrm{model}}$",
                r"$G^+$",
                r"$G^\pm$",
            ]
            assert figure.texts == []
            assert publication_figure_issues(
                figure,
                FIXED_THRESHOLD_TRADEOFF_PROFILE,
            ) == ()
        finally:
            plt.close(figure)


def test_fixed_threshold_tradeoff_exports_pdf_and_png(tmp_path) -> None:
    output = tmp_path / "fixed-threshold-architecture-tradeoffs.pdf"
    generated = generate_fixed_threshold_architecture_tradeoff_figure(
        output=output,
        save_png=True,
    )
    assert generated == (output, output.with_suffix(".png"))
    assert all(path.is_file() for path in generated)
    assert output.stat().st_size > 10_000


def test_fixed_threshold_frontier_encodes_architecture_gate_and_kappa(
    cohort,
) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_fixed_threshold_quality_opportunity_frontier_figure(cohort)
        try:
            figure.canvas.draw()
            assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 4.2))
            assert len(figure.axes) == 1
            axis = figure.axes[0]
            assert len(axis.lines) == 8
            assert axis.get_xlabel() == r"$R_{\mathrm{model}}$ (%)"
            assert axis.get_ylabel() == "Validation loss"
            assert len(axis.texts) == 8
            assert {text.get_text() for text in axis.texts} == {
                r"$\kappa=0$",
                r"$\kappa=0.03$",
                r"$\kappa=0.1$",
                r"$\kappa=0.3$",
            }
            assert all(text.get_color() == "black" for text in axis.texts)
            assert [text.get_text() for text in figure.legends[0].get_texts()] == [
                "A5-QK-PRE",
                "A5-QK-POST",
                "A6-PRE-QKV",
                "A6-POST-QKV",
                r"$G^+$",
                r"$G^\pm$",
            ]
            assert publication_figure_issues(
                figure,
                FIXED_THRESHOLD_FRONTIER_PROFILE,
            ) == ()
        finally:
            plt.close(figure)


def test_fixed_threshold_frontier_exports_pdf_and_png(tmp_path) -> None:
    output = tmp_path / "fixed-threshold-quality-opportunity-frontiers.pdf"
    generated = generate_fixed_threshold_quality_opportunity_frontier_figure(
        output=output,
        save_png=True,
    )
    assert generated == (output, output.with_suffix(".png"))
    assert all(path.is_file() for path in generated)
    assert output.stat().st_size > 10_000
