"""Render the S1-B2 learned-threshold quality-opportunity paths."""

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
    B1_GATE_STYLES,
    B2_ARCHITECTURE_COLORS,
    B2_ARCHITECTURE_ORDER,
    S1Row,
    _b2_threshold_triplets,
    _finish_axis,
    load_s1_rows,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


DEFAULT_OUTPUT = Path(
    "figures/117-pythia-14m-s1-learned-threshold-quality-opportunity-frontiers.pdf"
)
LEARNED_THRESHOLD_FRONTIER_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=4.5,
    min_text_points=8.0,
)
SETTING_STYLES = {
    "fixed": (r"Fixed $\kappa=0.10$", "o"),
    "absolute": ("Learned absolute", "s"),
    "rms": ("Learned RMS-relative", "D"),
}


def build_learned_threshold_quality_opportunity_frontier_figure(
    cohort: Sequence[S1Row],
) -> Figure:
    """Plot the 11 matched B2 fixed/absolute/RMS triplets in outcome space."""

    triplets = _b2_threshold_triplets(cohort)
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(
            LEARNED_THRESHOLD_FRONTIER_PROFILE.width_inches,
            LEARNED_THRESHOLD_FRONTIER_PROFILE.max_height_inches,
        ),
    )

    plotted_rows = []
    setting_points = {setting: [] for setting in SETTING_STYLES}
    for fixed, absolute, rms, architecture in triplets:
        family = str(absolute.config["gate_family"])
        style = B1_GATE_STYLES[family]
        color = B2_ARCHITECTURE_COLORS[architecture]
        path = (("rms", rms), ("fixed", fixed), ("absolute", absolute))
        plotted_rows.extend(row for _setting, row in path)
        axis.plot(
            [row.r_model_pct for _setting, row in path],
            [row.loss for _setting, row in path],
            color=color,
            linestyle=style["linestyle"],
            linewidth=1.25,
            alpha=0.92,
            zorder=3,
        )
        for setting, row in path:
            setting_points[setting].append((row, color))

    for setting, points in setting_points.items():
        _label, marker = SETTING_STYLES[setting]
        axis.scatter(
            [row.r_model_pct for row, _color in points],
            [row.loss for row, _color in points],
            c=[color for _row, color in points],
            edgecolors=[color for _row, color in points],
            marker=marker,
            s=29.0,
            linewidths=0.9,
            zorder=4,
        )

    r_values = [row.r_model_pct for row in plotted_rows]
    loss_values = [row.loss for row in plotted_rows]
    r_padding = max(0.05 * (max(r_values) - min(r_values)), 0.5)
    loss_padding = max(0.06 * (max(loss_values) - min(loss_values)), 0.003)
    axis.set_xlim(min(r_values) - r_padding, max(r_values) + r_padding)
    axis.set_ylim(min(loss_values) - loss_padding, max(loss_values) + loss_padding)
    axis.set_xlabel(r"$R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Validation loss")
    axis.set_title(
        "B2 learned-threshold quality-opportunity paths",
        loc="left",
        fontweight="bold",
    )
    _finish_axis(axis)

    architecture_handles = tuple(
        Line2D(
            [0],
            [0],
            color=B2_ARCHITECTURE_COLORS[architecture],
            label=architecture,
        )
        for architecture in B2_ARCHITECTURE_ORDER
    )
    gate_handles = tuple(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=B1_GATE_STYLES[family]["linestyle"],
            label=f"Gate: {B1_GATE_STYLES[family]['label']}",
        )
        for family in ("gplus", "gpm")
    )
    setting_handles = tuple(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="none",
            marker=marker,
            markerfacecolor="#777777",
            markeredgecolor="#444444",
            label=label,
        )
        for label, marker in SETTING_STYLES.values()
    )
    figure.legend(
        handles=architecture_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.legend(
        handles=gate_handles + setting_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.945),
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.14,
        top=0.835,
    )
    return figure


def generate_learned_threshold_quality_opportunity_frontier_figure(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    config_registry: str | Path = "docs/experimental-design/config-registry.yaml",
    run_registry: str | Path = "docs/experimental-design/run-registry.yaml",
    save_png: bool = True,
) -> tuple[Path, ...]:
    """Generate the supplemental learned-threshold frontier PDF and PNG."""

    cohort = load_s1_rows(config_registry, run_registry)
    return tuple(
        export_figure(
            lambda: build_learned_threshold_quality_opportunity_frontier_figure(
                cohort
            ),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=LEARNED_THRESHOLD_FRONTIER_PROFILE,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the S1-B2 learned-threshold frontier figure."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--config-registry",
        default="docs/experimental-design/config-registry.yaml",
    )
    parser.add_argument(
        "--run-registry",
        default="docs/experimental-design/run-registry.yaml",
    )
    parser.add_argument("--no-png", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    for path in generate_learned_threshold_quality_opportunity_frontier_figure(
        output=args.output,
        config_registry=args.config_registry,
        run_registry=args.run_registry,
        save_png=not args.no_png,
    ):
        print(path)


if __name__ == "__main__":
    main()
