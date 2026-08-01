from __future__ import annotations

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pytest

from paper_exp.plot_api import publication_figure_issues
from paper_exp.plot_report07 import load_s1_rows
from paper_exp.plot_s1_pressure_frontiers import (
    METHOD_STYLES,
    PRESSURE_FRONTIER_PROFILE,
    build_pressure_quality_opportunity_frontier_figure,
    generate_pressure_quality_opportunity_frontier_figure,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_s1_rows(
        ROOT / "docs/experimental-design/config-registry.yaml",
        ROOT / "docs/experimental-design/run-registry.yaml",
    )


def test_pressure_frontier_encodes_b3_weight_ladders(cohort) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_pressure_quality_opportunity_frontier_figure(cohort)
        try:
            figure.canvas.draw()
            assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 4.3))
            assert len(figure.axes) == 1
            axis = figure.axes[0]
            assert axis.get_title(loc="left") == (
                "B3 pressure-weight quality-opportunity paths"
            )
            assert len(axis.lines) == 8
            assert len(axis.collections) == 5
            assert sorted(
                len(collection.get_offsets()) for collection in axis.collections
            ) == [2, 6, 6, 6, 6]
            assert METHOD_STYLES["l1_naive"][2] == "o"
            assert METHOD_STYLES["ricker_naive"][2] == "o"
            assert METHOD_STYLES["orthogonal_l1"][2] == "s"
            assert METHOD_STYLES["orthogonal_ricker"][2] == "s"
            assert axis.collections[-1].get_facecolors()[0, :3] == pytest.approx(
                (0.0, 0.0, 0.0)
            )
            assert all(line.get_marker() == "None" for line in axis.lines)
            assert Counter(line.get_linestyle() for line in axis.lines) == {
                "-": 2,
                "--": 2,
                ":": 2,
                "-.": 2,
            }
            assert axis.get_xlabel() == r"$R_{\mathrm{model}}$ (%)"
            assert axis.get_ylabel() == "Validation loss"
            assert Counter(text.get_text() for text in axis.texts) == {
                r"$w=0.15$": 1,
                r"$w=0.10$": 1,
                r"$w=0.30$": 1,
                r"$w=1$": 2,
                r"$w=5$": 1,
            }
            assert all(text.get_color() == "black" for text in axis.texts)
            assert [text.get_text() for text in figure.legends[0].get_texts()] == [
                "A3",
                "A6-POST",
                "L1N",
                "OL1",
                "RN",
                "OR",
                "AdamW",
            ]
            assert publication_figure_issues(
                figure,
                PRESSURE_FRONTIER_PROFILE,
            ) == ()
        finally:
            plt.close(figure)


def test_pressure_frontier_exports_pdf_and_png(tmp_path) -> None:
    output = tmp_path / "pressure-quality-opportunity-frontiers.pdf"
    generated = generate_pressure_quality_opportunity_frontier_figure(
        output=output,
        save_png=True,
    )
    assert generated == (output, output.with_suffix(".png"))
    assert all(path.is_file() for path in generated)
    assert output.stat().st_size > 10_000
