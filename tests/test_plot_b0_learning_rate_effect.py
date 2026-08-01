from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper_exp.plot_api import publication_figure_issues
from paper_exp.plot_b0_learning_rate_effect import (
    LEARNING_RATE_EFFECT_PROFILE,
    _lr_1e4_endpoints,
    build_learning_rate_effect_figure,
    generate_learning_rate_effect_figure,
)
from paper_exp.plot_report07 import load_s1_rows
from paper_exp.plot_style import REPORT04_PLOT_STYLE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_s1_rows(
        ROOT / "docs/experimental-design/config-registry.yaml",
        ROOT / "docs/experimental-design/run-registry.yaml",
    )


def test_lr_1e4_zoom_uses_exact_registered_endpoints(cohort) -> None:
    endpoints = _lr_1e4_endpoints(cohort)
    assert [architecture for architecture, _row in endpoints] == [
        "A0",
        "A1-H",
        "A3",
        "A6-PRE",
        "A6-POST",
    ]
    assert [row.loss for _architecture, row in endpoints] == pytest.approx(
        [
            5.938874294883327,
            5.874738103465030,
            5.917684304086786,
            5.935388401934975,
            6.063203761452122,
        ],
        abs=1e-12,
    )


def test_lr_effect_figure_is_three_panel_and_publication_sized(cohort) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_learning_rate_effect_figure(cohort)
        try:
            assert len(figure.axes) == 3
            assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 3.35))
            assert all(not axis.child_axes for axis in figure.axes)
            assert figure.texts == []
            endpoint_axis = figure.axes[2]
            assert endpoint_axis.get_ylim() == pytest.approx((5.84, 6.09))
            assert endpoint_axis.get_title(
                loc="left"
            ) == r"(c) Loss at LR $=10^{-4}$"
            assert [
                tick.get_text()
                for tick in endpoint_axis.get_xticklabels()
            ] == [
                "A0",
                "A1-H",
                "A3",
                "A6\nPRE",
                "A6\nPOST",
            ]
            assert figure.axes[1].get_title(loc="left") == (
                "(b) Logical product opportunity"
            )
            assert publication_figure_issues(
                figure,
                LEARNING_RATE_EFFECT_PROFILE,
            ) == ()
            assert "(c)" not in " ".join(
                text.get_text()
                for axis in figure.axes
                for text in axis.texts
            )
        finally:
            plt.close(figure)


def test_lr_effect_figure_exports_pdf_and_png(tmp_path) -> None:
    output = tmp_path / "learning-rate-effect.pdf"
    generated = generate_learning_rate_effect_figure(
        output=output,
        save_png=True,
    )
    assert generated == (output, output.with_suffix(".png"))
    assert all(path.is_file() for path in generated)
    assert output.stat().st_size > 10_000
