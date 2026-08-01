from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper_exp.plot_api import publication_figure_issues
from paper_exp.plot_report07 import load_s1_rows
from paper_exp.plot_s1_learned_threshold_frontiers import (
    LEARNED_THRESHOLD_FRONTIER_PROFILE,
    build_learned_threshold_quality_opportunity_frontier_figure,
    generate_learned_threshold_quality_opportunity_frontier_figure,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_s1_rows(
        ROOT / "docs/experimental-design/config-registry.yaml",
        ROOT / "docs/experimental-design/run-registry.yaml",
    )


def test_learned_threshold_frontier_encodes_matched_b2_paths(cohort) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_learned_threshold_quality_opportunity_frontier_figure(cohort)
        try:
            figure.canvas.draw()
            assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 4.5))
            assert len(figure.axes) == 1
            axis = figure.axes[0]
            assert len(axis.lines) == 11
            assert len(axis.collections) == 3
            assert all(len(collection.get_offsets()) == 11 for collection in axis.collections)
            assert axis.get_xlabel() == r"$R_{\mathrm{model}}$ (%)"
            assert axis.get_ylabel() == "Validation loss"
            assert len(axis.texts) == 0
            assert [text.get_text() for text in figure.legends[0].get_texts()] == [
                "A1-H",
                "A3",
                "A5-QK-PRE",
                "A5-QK-POST",
                "A6-PRE-QKV",
                "A6-POST-QKV",
                "A6-POST-ALL",
            ]
            assert [text.get_text() for text in figure.legends[1].get_texts()] == [
                r"Gate: $G^+$",
                r"Gate: $G^\pm$",
                r"Fixed $\kappa=0.10$",
                "Learned absolute",
                "Learned RMS-relative",
            ]
            assert publication_figure_issues(
                figure,
                LEARNED_THRESHOLD_FRONTIER_PROFILE,
            ) == ()
        finally:
            plt.close(figure)


def test_learned_threshold_frontier_exports_pdf_and_png(tmp_path) -> None:
    output = tmp_path / "learned-threshold-quality-opportunity-frontiers.pdf"
    generated = generate_learned_threshold_quality_opportunity_frontier_figure(
        output=output,
        save_png=True,
    )
    assert generated == (output, output.with_suffix(".png"))
    assert all(path.is_file() for path in generated)
    assert output.stat().st_size > 10_000
