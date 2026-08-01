"""Render the architecture-first S1 fixed-threshold trade-off figure.

This supplemental view preserves Figure 104 and reorganizes the same complete
B1 A5/A6 ladders into one panel per architecture. Validation loss and logical
model-product opportunity share the threshold axis but retain explicit,
globally shared y-scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from paper_exp.plot_api import (
    DOUBLE_COLUMN_WIDTH_INCHES,
    PublicationProfile,
    export_figure,
)
from paper_exp.plot_report07 import (
    B1_ARCHITECTURE_COLORS,
    B1_FACTORIAL_ARCHITECTURES,
    B1_GATE_STYLES,
    S1Row,
    _b1_factorial_series,
    _b1_metric_limits,
    _finish_axis,
    _panel_title,
    load_s1_rows,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


DEFAULT_OUTPUT = Path(
    "figures/114-pythia-14m-s1-fixed-threshold-architecture-tradeoffs.pdf"
)
FRONTIER_OUTPUT = Path(
    "figures/116-pythia-14m-s1-fixed-threshold-quality-opportunity-frontiers.pdf"
)
FIXED_THRESHOLD_TRADEOFF_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=4.3,
    min_text_points=8.0,
)
FIXED_THRESHOLD_FRONTIER_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=4.2,
    min_text_points=8.0,
)
LOSS_COLOR = "#D55E00"
OPPORTUNITY_COLOR = "#0072B2"
FRONTIER_ANNOTATION_OFFSETS = {
    ("A6-PRE-QKV", "gplus"): (
        (-8.0, -10.0), (0.0, 6.0), (-2.0, 6.0), (-8.0, -10.0),
    ),
    ("A6-PRE-QKV", "gpm"): (
        (-6.0, -10.0), (-8.0, 6.0), (-8.0, 6.0), (-8.0, 6.0),
    ),
}


def build_fixed_threshold_architecture_tradeoff_figure(
    cohort: Sequence[S1Row],
) -> Figure:
    """Build four shared-scale architecture panels for the complete B1 ladders."""

    series = _b1_factorial_series(cohort)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            FIXED_THRESHOLD_TRADEOFF_PROFILE.width_inches,
            FIXED_THRESHOLD_TRADEOFF_PROFILE.max_height_inches,
        ),
        sharex=True,
        sharey=True,
    )
    loss_limits = _b1_metric_limits(series, "loss")
    opportunity_limits = _b1_metric_limits(series, "r_model_pct")

    for index, (architecture, _placement, _scope, _parent) in enumerate(
        B1_FACTORIAL_ARCHITECTURES
    ):
        loss_axis = axes.flat[index]
        opportunity_axis = loss_axis.twinx()
        for family in ("gplus", "gpm"):
            items = series[(architecture, family)]
            style = B1_GATE_STYLES[family]
            x_values = [kappa for kappa, _row in items]
            filled = family == "gplus"
            loss_axis.plot(
                x_values,
                [row.loss for _kappa, row in items],
                color=LOSS_COLOR,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=LOSS_COLOR if filled else "white",
                markeredgecolor=LOSS_COLOR,
                markeredgewidth=0.9,
                markersize=4.5,
                linewidth=1.15,
                zorder=3,
            )
            opportunity_axis.plot(
                x_values,
                [row.r_model_pct for _kappa, row in items],
                color=OPPORTUNITY_COLOR,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=OPPORTUNITY_COLOR if filled else "white",
                markeredgecolor=OPPORTUNITY_COLOR,
                markeredgewidth=0.9,
                markersize=4.5,
                linewidth=1.15,
                zorder=4,
            )

        _panel_title(loss_axis, chr(ord("a") + index), architecture)
        loss_axis.set_xlim(-0.012, 0.315)
        loss_axis.set_ylim(*loss_limits)
        opportunity_axis.set_ylim(*opportunity_limits)
        loss_axis.set_xticks(
            (0.0, 0.03, 0.10, 0.30),
            ("0", "0.03", "0.10", "0.30"),
        )
        loss_axis.set_xlabel(r"Fixed threshold $\kappa$")
        loss_axis.set_ylabel("Validation loss", color=LOSS_COLOR)
        opportunity_axis.set_ylabel(
            r"$R_{\mathrm{model}}$ (%)",
            color=OPPORTUNITY_COLOR,
        )
        loss_axis.tick_params(axis="x", labelbottom=True)
        loss_axis.tick_params(axis="y", colors=LOSS_COLOR)
        opportunity_axis.tick_params(axis="y", colors=OPPORTUNITY_COLOR)
        _finish_axis(loss_axis)
        loss_axis.spines["left"].set_color(LOSS_COLOR)
        opportunity_axis.grid(False)
        opportunity_axis.spines["top"].set_visible(False)
        opportunity_axis.spines["left"].set_visible(False)
        opportunity_axis.spines["right"].set_color(OPPORTUNITY_COLOR)

    metric_handles = (
        Line2D([0], [0], color=LOSS_COLOR, label="Validation loss"),
        Line2D(
            [0],
            [0],
            color=OPPORTUNITY_COLOR,
            label=r"$R_{\mathrm{model}}$",
        ),
    )
    gate_handles = tuple(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=B1_GATE_STYLES[family]["linestyle"],
            marker=B1_GATE_STYLES[family]["marker"],
            markerfacecolor="#444444" if family == "gplus" else "white",
            label=B1_GATE_STYLES[family]["label"],
        )
        for family in ("gplus", "gpm")
    )
    figure.legend(
        handles=metric_handles + gate_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.13,
        top=0.86,
        hspace=0.50,
        wspace=0.50,
    )
    return figure


def generate_fixed_threshold_architecture_tradeoff_figure(
    *,
    config_registry: str | Path = (
        "docs/experimental-design/config-registry.yaml"
    ),
    run_registry: str | Path = (
        "docs/experimental-design/run-registry.yaml"
    ),
    output: str | Path = DEFAULT_OUTPUT,
    save_png: bool = True,
) -> tuple[Path, ...]:
    """Load the canonical S1 cohort and export the supplemental figure."""

    cohort = load_s1_rows(config_registry, run_registry)
    return tuple(
        export_figure(
            lambda: build_fixed_threshold_architecture_tradeoff_figure(cohort),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=FIXED_THRESHOLD_TRADEOFF_PROFILE,
        )
    )


def build_fixed_threshold_quality_opportunity_frontier_figure(
    cohort: Sequence[S1Row],
) -> Figure:
    """Build the B1 threshold paths in validation-loss/opportunity space."""

    series = _b1_factorial_series(cohort)
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(
            FIXED_THRESHOLD_FRONTIER_PROFILE.width_inches,
            FIXED_THRESHOLD_FRONTIER_PROFILE.max_height_inches,
        ),
    )
    for architecture, _placement, _scope, _parent in B1_FACTORIAL_ARCHITECTURES:
        color = B1_ARCHITECTURE_COLORS[architecture]
        for family in ("gplus", "gpm"):
            items = series[(architecture, family)]
            style = B1_GATE_STYLES[family]
            axis.plot(
                [row.r_model_pct for _kappa, row in items],
                [row.loss for _kappa, row in items],
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=color if family == "gplus" else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=5.0,
                linewidth=1.25,
                zorder=3,
            )
            offsets = FRONTIER_ANNOTATION_OFFSETS.get((architecture, family))
            if offsets is None:
                continue
            for (kappa, row), offset in zip(items, offsets, strict=True):
                axis.annotate(
                    rf"$\kappa={kappa:g}$",
                    (row.r_model_pct, row.loss),
                    xytext=offset,
                    textcoords="offset points",
                    color="black",
                    fontsize=8.0,
                    ha="right" if offset[0] < 0.0 else "left",
                    va="bottom" if offset[1] >= 0.0 else "top",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 0.18,
                    },
                    zorder=5,
                )

    axis.set_xlim(4.7, 23.1)
    axis.set_ylim(7.010, 7.0385)
    axis.set_xlabel(r"$R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Validation loss")
    axis.set_title(
        "Fixed-threshold quality-opportunity paths",
        loc="left",
        fontweight="bold",
    )
    _finish_axis(axis)

    architecture_handles = tuple(
        Line2D(
            [0],
            [0],
            color=B1_ARCHITECTURE_COLORS[architecture],
            label=architecture,
        )
        for architecture, _placement, _scope, _parent in B1_FACTORIAL_ARCHITECTURES
    )
    gate_handles = tuple(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=B1_GATE_STYLES[family]["linestyle"],
            marker=B1_GATE_STYLES[family]["marker"],
            markerfacecolor="#444444" if family == "gplus" else "white",
            label=B1_GATE_STYLES[family]["label"],
        )
        for family in ("gplus", "gpm")
    )
    figure.legend(
        handles=architecture_handles + gate_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.15,
        top=0.82,
    )
    return figure


def generate_fixed_threshold_quality_opportunity_frontier_figure(
    *,
    config_registry: str | Path = (
        "docs/experimental-design/config-registry.yaml"
    ),
    run_registry: str | Path = (
        "docs/experimental-design/run-registry.yaml"
    ),
    output: str | Path = FRONTIER_OUTPUT,
    save_png: bool = True,
) -> tuple[Path, ...]:
    """Load the canonical S1 cohort and export the threshold frontier figure."""

    cohort = load_s1_rows(config_registry, run_registry)
    return tuple(
        export_figure(
            lambda: build_fixed_threshold_quality_opportunity_frontier_figure(
                cohort
            ),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=FIXED_THRESHOLD_FRONTIER_PROFILE,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-registry",
        default="docs/experimental-design/config-registry.yaml",
    )
    parser.add_argument(
        "--run-registry",
        default="docs/experimental-design/run-registry.yaml",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frontier-output", default=str(FRONTIER_OUTPUT))
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Write only the vector PDF.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = list(
        generate_fixed_threshold_architecture_tradeoff_figure(
            config_registry=args.config_registry,
            run_registry=args.run_registry,
            output=args.output,
            save_png=not args.no_png,
        )
    )
    outputs.extend(
        generate_fixed_threshold_quality_opportunity_frontier_figure(
            config_registry=args.config_registry,
            run_registry=args.run_registry,
            output=args.frontier_output,
            save_png=not args.no_png,
        )
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
