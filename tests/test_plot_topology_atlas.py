from __future__ import annotations

import matplotlib.pyplot as plt

from paper_exp.plot_api import publication_figure_issues
from paper_exp.plot_style import REPORT04_PLOT_STYLE
from paper_exp.plot_topology_atlas import (
    SITE_ORDER,
    TOPOLOGIES,
    TOPOLOGY_ATLAS_PROFILE,
    build_topology_atlas_figure,
    generate_topology_atlas,
)


def test_topology_atlas_covers_every_report07_topology() -> None:
    assert [row.name for row in TOPOLOGIES] == [
        "A0",
        "A1-H",
        "A3",
        "A4-Q",
        "A4-K",
        "A4-V",
        "A5-QK-PRE",
        "A5-QK-POST",
        "A6-PRE",
        "A6-POST",
    ]
    assert SITE_ORDER == (
        "a",
        "m",
        "h",
        "q_pre",
        "k_pre",
        "q_post",
        "k_post",
        "v",
    )
    by_name = {row.name: row for row in TOPOLOGIES}
    assert by_name["A4-Q"].active_sites - by_name["A3"].active_sites == {"q_post"}
    assert by_name["A4-K"].active_sites - by_name["A3"].active_sites == {"k_post"}
    assert by_name["A4-V"].active_sites - by_name["A3"].active_sites == {"v"}
    assert not hasattr(by_name["A6-PRE"], "r_block_max")
    assert by_name["A6-POST"].r_model_max == "29.9524"


def test_topology_atlas_meets_half_page_profile() -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = build_topology_atlas_figure()
        try:
            assert publication_figure_issues(figure, TOPOLOGY_ATLAS_PROFILE) == ()
        finally:
            plt.close(figure)


def test_topology_atlas_exports_pdf(tmp_path) -> None:
    output = tmp_path / "topology-atlas.pdf"
    generated = generate_topology_atlas(output)
    assert generated == (output,)
    assert output.is_file()
    assert output.stat().st_size > 10_000
