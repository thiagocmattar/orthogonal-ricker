"""Render the compact Pythia-14M sparsification-topology atlas.

The upper panel draws the computation shared by every registered topology and
marks every available exact-zero gate port. The lower panel selects those ports
for each executed A0--A6 architecture and reports the exact reach ceilings used
by Report 07.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch

from paper_exp.plot_api import (
    DOUBLE_COLUMN_WIDTH_INCHES,
    PublicationProfile,
    export_figure,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


DEFAULT_OUTPUT = Path(
    "report/07-2026-07-27-s1-ablation-study/"
    "110-pythia-14m-s1-topology-atlas.pdf"
)
TOPOLOGY_ATLAS_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=3.75,
    min_text_points=8.0,
)

GATE_FACE = "#F5C6DF"
GATE_EDGE = "#A23B72"
OPERATION_FACE = "#F2F2F2"
OPERATION_EDGE = "#666666"
PROJECTION_FACE = "#D9EEF8"
PROJECTION_EDGE = "#0072B2"
FLOW_COLOR = "#333333"
GUIDE_COLOR = "#C7C7C7"


@dataclass(frozen=True)
class TopologySpec:
    """One topology row in the compact occupancy atlas."""

    name: str
    active_sites: frozenset[str]
    r_model_max: str
    stock_gelu: bool = False


SITE_ORDER = (
    "a",
    "m",
    "h",
    "q_pre",
    "k_pre",
    "q_post",
    "k_post",
    "v",
)

TOPOLOGIES = (
    TopologySpec("A0", frozenset(), "0", stock_gelu=True),
    TopologySpec("A1-H", frozenset({"h"}), "4.2777"),
    TopologySpec("A3", frozenset({"a", "m", "h"}), "11.7637"),
    TopologySpec(
        "A4-Q",
        frozenset({"a", "m", "h", "q_post"}),
        "20.3233",
    ),
    TopologySpec(
        "A4-K",
        frozenset({"a", "m", "h", "k_post"}),
        "20.3233",
    ),
    TopologySpec(
        "A4-V",
        frozenset({"a", "m", "h", "v"}),
        "21.3928",
    ),
    TopologySpec(
        "A5-QK-PRE",
        frozenset({"a", "m", "h", "q_pre", "k_pre"}),
        "20.3233",
    ),
    TopologySpec(
        "A5-QK-POST",
        frozenset({"a", "m", "h", "q_post", "k_post"}),
        "20.3233",
    ),
    TopologySpec(
        "A6-PRE",
        frozenset({"a", "m", "h", "q_pre", "k_pre", "v"}),
        "29.9524",
    ),
    TopologySpec(
        "A6-POST",
        frozenset({"a", "m", "h", "q_post", "k_post", "v"}),
        "29.9524",
    ),
)


def _box(
    axis: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    projection: bool = False,
) -> None:
    face = PROJECTION_FACE if projection else OPERATION_FACE
    edge = PROJECTION_EDGE if projection else OPERATION_EDGE
    axis.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    axis.text(
        x + width / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=8.0,
        zorder=3,
    )


def _gate_port(axis: Axes, x: float, y: float, label: str) -> None:
    axis.add_patch(
        Circle(
            (x, y),
            radius=0.0125,
            linewidth=1.0,
            edgecolor=OPERATION_EDGE,
            facecolor="white",
            zorder=4,
        )
    )
    axis.text(
        x,
        y,
        rf"${label}$",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=FLOW_COLOR,
        zorder=5,
    )


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    arrow: bool = True,
    color: str = FLOW_COLOR,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>" if arrow else "-",
            "color": color,
            "linewidth": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 6,
        },
        zorder=1,
    )


def _draw_shared_block(axis: Axes) -> None:
    axis.text(
        0.01,
        0.975,
        "(a) Shared Pythia block",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )

    _box(axis, x=0.012, y=0.745, width=0.052, height=0.070, label=r"$H_\ell$")
    _box(axis, x=0.090, y=0.870, width=0.054, height=0.048, label=r"$\mathrm{LN}_a$")
    _gate_port(axis, 0.163, 0.870, "a")
    _box(
        axis,
        x=0.187,
        y=0.870,
        width=0.068,
        height=0.048,
        label=r"$W_{\mathrm{QKV}}$",
        projection=True,
    )

    _arrow(axis, (0.064, 0.755), (0.090, 0.870))
    _arrow(axis, (0.144, 0.870), (0.1505, 0.870))
    _arrow(axis, (0.1755, 0.870), (0.187, 0.870))
    _arrow(axis, (0.255, 0.870), (0.274, 0.870))
    axis.plot(
        [0.274, 0.274],
        [0.745, 0.915],
        color=FLOW_COLOR,
        linewidth=0.9,
        zorder=1,
    )

    axis.text(0.322, 0.936, "PRE", ha="center", va="bottom", fontsize=8.0)
    axis.text(0.482, 0.936, "POST", ha="center", va="bottom", fontsize=8.0)

    for lane_y, lane_label in ((0.915, "Q"), (0.825, "K")):
        axis.text(
            0.284,
            lane_y,
            lane_label,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )
        _arrow(axis, (0.274, lane_y), (0.289, lane_y))
    _gate_port(axis, 0.322, 0.915, "q")
    _gate_port(axis, 0.322, 0.825, "k")
    _arrow(axis, (0.289, 0.915), (0.3095, 0.915))
    _arrow(axis, (0.3345, 0.915), (0.374, 0.915))
    _arrow(axis, (0.289, 0.825), (0.3095, 0.825))
    _arrow(axis, (0.3345, 0.825), (0.374, 0.825))

    _box(
        axis,
        x=0.374,
        y=0.870,
        width=0.060,
        height=0.116,
        label="RoPE",
    )
    _arrow(axis, (0.434, 0.915), (0.4695, 0.915))
    _arrow(axis, (0.434, 0.825), (0.4695, 0.825))
    _gate_port(axis, 0.482, 0.915, "q")
    _gate_port(axis, 0.482, 0.825, "k")

    _box(axis, x=0.538, y=0.870, width=0.055, height=0.052, label=r"$QK^\top$")
    _arrow(axis, (0.4945, 0.915), (0.538, 0.884))
    _arrow(axis, (0.4945, 0.825), (0.538, 0.856))
    _box(
        axis,
        x=0.614,
        y=0.870,
        width=0.079,
        height=0.052,
        label=r"$\mathrm{softmax}\ P$",
    )
    _arrow(axis, (0.593, 0.870), (0.614, 0.870))

    axis.text(
        0.284,
        0.745,
        "V",
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    _arrow(axis, (0.274, 0.745), (0.3095, 0.745))
    _gate_port(axis, 0.322, 0.745, "v")
    _arrow(axis, (0.3345, 0.745), (0.748, 0.745))

    _box(axis, x=0.748, y=0.785, width=0.052, height=0.052, label=r"$PV$")
    _arrow(axis, (0.693, 0.870), (0.748, 0.798))
    _box(
        axis,
        x=0.827,
        y=0.785,
        width=0.055,
        height=0.052,
        label=r"$W_o$",
        projection=True,
    )
    _arrow(axis, (0.800, 0.785), (0.827, 0.785))

    _box(axis, x=0.090, y=0.630, width=0.054, height=0.048, label=r"$\mathrm{LN}_m$")
    _gate_port(axis, 0.163, 0.630, "m")
    _box(
        axis,
        x=0.187,
        y=0.630,
        width=0.055,
        height=0.048,
        label=r"$W_1$",
        projection=True,
    )
    _gate_port(axis, 0.262, 0.630, "h")
    _box(
        axis,
        x=0.285,
        y=0.630,
        width=0.055,
        height=0.048,
        label=r"$W_2$",
        projection=True,
    )
    _arrow(axis, (0.064, 0.735), (0.090, 0.630))
    _arrow(axis, (0.144, 0.630), (0.1505, 0.630))
    _arrow(axis, (0.1755, 0.630), (0.187, 0.630))
    _arrow(axis, (0.242, 0.630), (0.2495, 0.630))
    _arrow(axis, (0.2745, 0.630), (0.285, 0.630))

    sum_x, sum_y = 0.935, 0.675
    axis.add_patch(
        Circle(
            (sum_x, sum_y),
            radius=0.018,
            linewidth=1.0,
            edgecolor=FLOW_COLOR,
            facecolor="white",
            zorder=3,
        )
    )
    axis.text(
        sum_x,
        sum_y,
        "+",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        zorder=4,
    )
    _arrow(axis, (0.882, 0.785), (sum_x - 0.013, sum_y + 0.013))
    _arrow(axis, (0.340, 0.630), (sum_x - 0.018, 0.630))
    _arrow(axis, (0.064, 0.720), (0.078, 0.565), arrow=False)
    _arrow(axis, (0.078, 0.565), (0.900, 0.565), arrow=False)
    _arrow(axis, (0.900, 0.565), (sum_x - 0.010, sum_y - 0.014))
    _arrow(axis, (sum_x + 0.018, sum_y), (0.956, sum_y))
    axis.text(
        0.962,
        sum_y,
        r"$H_{\ell+1}$",
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )


def _draw_topology_matrix(axis: Axes) -> None:
    axis.text(
        0.01,
        0.525,
        "(b) Executed topologies; sites repeat in all six blocks",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
    )

    site_x = {
        "a": 0.244,
        "m": 0.284,
        "h": 0.324,
        "q_pre": 0.402,
        "k_pre": 0.442,
        "q_post": 0.512,
        "k_post": 0.552,
        "v": 0.624,
    }
    model_x = 0.946

    axis.text(0.284, 0.486, "branch", ha="center", va="center", fontsize=8.0)
    axis.text(0.422, 0.486, "before RoPE", ha="center", va="center", fontsize=8.0)
    axis.text(0.532, 0.486, "after RoPE", ha="center", va="center", fontsize=8.0)
    axis.text(0.624, 0.486, r"$V{\to}PV$", ha="center", va="center", fontsize=8.0)
    axis.text(
        0.900,
        0.486,
        "logical reach ceiling (%)",
        ha="center",
        va="center",
        fontsize=8.0,
    )

    header_labels = {
        "a": r"$a$",
        "m": r"$m$",
        "h": r"$h$",
        "q_pre": r"$Q$",
        "k_pre": r"$K$",
        "q_post": r"$Q$",
        "k_post": r"$K$",
        "v": r"$V$",
    }
    for site, x_value in site_x.items():
        axis.text(
            x_value,
            0.454,
            header_labels[site],
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )
    axis.text(
        model_x,
        0.454,
        r"$R_{\mathrm{model}}^{\max}$",
        ha="right",
        va="center",
        fontsize=8.0,
    )
    axis.plot([0.01, 0.985], [0.435, 0.435], color=FLOW_COLOR, linewidth=0.8)

    row_start = 0.409
    row_step = 0.0355
    group_breaks = {3, 6, 8}
    for row_index, topology in enumerate(TOPOLOGIES):
        row_y = row_start - row_index * row_step
        if row_index in group_breaks:
            separator_y = row_y + row_step / 2
            axis.plot(
                [0.01, 0.985],
                [separator_y, separator_y],
                color=GUIDE_COLOR,
                linewidth=0.6,
            )

        axis.text(
            0.012,
            row_y,
            topology.name,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )
        for site in SITE_ORDER:
            active = site in topology.active_sites
            axis.scatter(
                [site_x[site]],
                [row_y],
                s=25,
                marker="o",
                facecolors=GATE_FACE if active else "white",
                edgecolors=GATE_EDGE if active else GUIDE_COLOR,
                linewidths=1.0 if active else 0.7,
                zorder=3,
            )
        if topology.stock_gelu:
            axis.scatter(
                [site_x["h"]],
                [row_y],
                s=35,
                marker="s",
                facecolors="#D9D9D9",
                edgecolors=OPERATION_EDGE,
                linewidths=0.8,
                zorder=4,
            )
            axis.text(
                site_x["h"],
                row_y,
                "G",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                zorder=5,
            )

        axis.text(
            model_x,
            row_y,
            topology.r_model_max,
            ha="right",
            va="center",
            fontsize=8.0,
            family="monospace",
        )

    footer_y_top = 0.054
    footer_y_bottom = 0.020
    axis.scatter(
        [0.018],
        [footer_y_top],
        s=25,
        facecolors=GATE_FACE,
        edgecolors=GATE_EDGE,
        linewidths=1.0,
    )
    axis.scatter(
        [0.128],
        [footer_y_top],
        s=25,
        facecolors="white",
        edgecolors=GUIDE_COLOR,
        linewidths=0.7,
    )
    axis.text(
        0.030,
        footer_y_top,
        "active gate",
        ha="left",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        0.140,
        footer_y_top,
        "available port",
        ha="left",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        0.250,
        footer_y_top,
        r"$\mathsf{G}$ = stock GELU at $h$",
        ha="left",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        0.425,
        footer_y_top,
        r"operator, $\kappa$, optimizer, and pressure are separate; PRE/POST is Q/K only.",
        ha="left",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        0.010,
        footer_y_bottom,
        r"Ceilings: all active outputs $\equiv0$; logical product reach, not runtime.",
        ha="left",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        0.545,
        footer_y_bottom,
        r"A6 closure: $V\equiv0\Rightarrow PV\equiv0$; partial masks may not persist.",
        ha="left",
        va="center",
        fontsize=8.0,
    )


def build_topology_atlas_figure() -> Figure:
    """Build the vector topology atlas at its intended half-page size."""

    figure = plt.figure(
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, TOPOLOGY_ATLAS_PROFILE.max_height_inches)
    )
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    _draw_shared_block(axis)
    _draw_topology_matrix(axis)
    return figure


def generate_topology_atlas(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    save_png: bool = False,
) -> tuple[Path, ...]:
    """Export the topology atlas as deterministic vector PDF."""

    return tuple(
        export_figure(
            build_topology_atlas_figure,
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=TOPOLOGY_ATLAS_PROFILE,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--png", action="store_true", help="Also write a PNG preview.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for output in generate_topology_atlas(args.output, save_png=args.png):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
