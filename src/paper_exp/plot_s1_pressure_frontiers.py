"""Render S1-B3 L1 and Ricker pressure-weight outcome paths."""

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
    ARCHITECTURE_COLORS,
    S1Row,
    _b3_weight_endpoints,
    _finish_axis,
    load_s1_rows,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


DEFAULT_OUTPUT = Path(
    "figures/118-pythia-14m-s1-pressure-quality-opportunity-frontiers.pdf"
)
PRESSURE_FRONTIER_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=4.3,
    min_text_points=8.0,
)
ARCHITECTURES = ("A3", "A6-POST")
METHOD_STYLES = {
    "l1_naive": ("L1N", "-", "o"),
    "orthogonal_l1": ("OL1", "--", "s"),
    "ricker_naive": ("RN", ":", "o"),
    "orthogonal_ricker": ("OR", "-.", "s"),
}
PRESSURE_SPECS = {
    "L1": (
        (0.15, 1.0, 5.0),
        ("AdamW", r"$w=0.15$", r"$w=1$", r"$w=5$"),
        ("l1_naive", "orthogonal_l1"),
    ),
    "Ricker": (
        (0.10, 0.30, 1.0),
        ("AdamW", r"$w=0.10$", r"$w=0.30$", r"$w=1$"),
        ("ricker_naive", "orthogonal_ricker"),
    ),
}
ANNOTATED_FRONTIERS = {
    ("A6-POST", "l1_naive"): (
        (-7.0, -10.0),
        (5.0, -9.0),
        (5.0, 5.0),
        (5.0, 5.0),
    ),
    ("A3", "ricker_naive"): (
        (-8.0, 6.0),
        (5.0, -9.0),
        (5.0, 5.0),
        (-5.0, -9.0),
    ),
}


def build_pressure_quality_opportunity_frontier_figure(
    cohort: Sequence[S1Row],
) -> Figure:
    """Plot matched AdamW-to-pressure paths for the B3 weight ladders."""

    endpoints = _b3_weight_endpoints(cohort)
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(
            PRESSURE_FRONTIER_PROFILE.width_inches,
            PRESSURE_FRONTIER_PROFILE.max_height_inches,
        ),
    )
    plotted_rows = []
    method_points = {method_id: [] for method_id in METHOD_STYLES}
    adamw_rows = {}
    for pressure, (weights, setting_labels, method_ids) in PRESSURE_SPECS.items():
        selected = [endpoint for endpoint in endpoints if endpoint.pressure == pressure]
        for architecture in ARCHITECTURES:
            architecture_rows = sorted(
                (
                    endpoint
                    for endpoint in selected
                    if endpoint.architecture == architecture
                ),
                key=lambda endpoint: endpoint.weight,
            )
            if tuple(endpoint.weight for endpoint in architecture_rows) != weights:
                raise ValueError(
                    f"Unexpected {pressure} weight ladder for {architecture}."
                )
            color = ARCHITECTURE_COLORS[architecture]
            for field, method_id in zip(
                ("naive", "orthogonal"),
                method_ids,
                strict=True,
            ):
                _method_label, linestyle, _marker = METHOD_STYLES[method_id]
                path = [architecture_rows[0].adamw] + [
                    getattr(endpoint, field) for endpoint in architecture_rows
                ]
                plotted_rows.extend(path)
                adamw_rows[architecture] = path[0]
                method_points[method_id].extend(
                    (row, color) for row in path[1:]
                )
                axis.plot(
                    [row.r_model_pct for row in path],
                    [row.loss for row in path],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    alpha=0.92,
                    zorder=3,
                )
                offsets = ANNOTATED_FRONTIERS.get((architecture, method_id))
                if offsets is None:
                    continue
                for row, label, offset in zip(
                    path[1:],
                    setting_labels[1:],
                    offsets[1:],
                    strict=True,
                ):
                    axis.annotate(
                        label,
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

    for method_id, points in method_points.items():
        _method_label, _linestyle, marker = METHOD_STYLES[method_id]
        axis.scatter(
            [row.r_model_pct for row, _color in points],
            [row.loss for row, _color in points],
            c=[color for _row, color in points],
            edgecolors=[color for _row, color in points],
            marker=marker,
            s=30,
            linewidths=0.9,
            zorder=4,
        )
    controls = tuple(adamw_rows.values())
    axis.scatter(
        [row.r_model_pct for row in controls],
        [row.loss for row in controls],
        color="black",
        edgecolors="black",
        marker="o",
        s=30,
        linewidths=0.9,
        zorder=5,
    )

    r_values = [row.r_model_pct for row in plotted_rows]
    loss_values = [row.loss for row in plotted_rows]
    r_padding = max(0.05 * (max(r_values) - min(r_values)), 0.6)
    loss_padding = max(0.12 * (max(loss_values) - min(loss_values)), 0.022)
    axis.set_xlim(min(r_values) - r_padding, max(r_values) + r_padding)
    axis.set_ylim(
        min(loss_values) - 0.45 * loss_padding,
        max(loss_values) + loss_padding,
    )
    axis.set_xlabel(r"$R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Validation loss")
    axis.set_title(
        "B3 pressure-weight quality-opportunity paths",
        loc="left",
        fontweight="bold",
    )
    _finish_axis(axis)

    architecture_handles = tuple(
        Line2D(
            [0],
            [0],
            color=ARCHITECTURE_COLORS[architecture],
            label=architecture,
        )
        for architecture in ARCHITECTURES
    )
    method_handles = tuple(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="#444444",
            markeredgecolor="#444444",
            label=label,
        )
        for label, linestyle, marker in METHOD_STYLES.values()
    )
    adamw_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="none",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        label="AdamW",
    )
    figure.legend(
        handles=architecture_handles + method_handles + (adamw_handle,),
        loc="upper center",
        ncol=7,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.14,
        top=0.82,
    )
    return figure


def generate_pressure_quality_opportunity_frontier_figure(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    config_registry: str | Path = "docs/experimental-design/config-registry.yaml",
    run_registry: str | Path = "docs/experimental-design/run-registry.yaml",
    save_png: bool = True,
) -> tuple[Path, ...]:
    """Generate the supplemental B3 pressure frontier PDF and PNG."""

    cohort = load_s1_rows(config_registry, run_registry)
    return tuple(
        export_figure(
            lambda: build_pressure_quality_opportunity_frontier_figure(cohort),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=PRESSURE_FRONTIER_PROFILE,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the S1-B3 pressure-weight frontier figure."
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
    for path in generate_pressure_quality_opportunity_frontier_figure(
        output=args.output,
        config_registry=args.config_registry,
        run_registry=args.run_registry,
        save_png=not args.no_png,
    ):
        print(path)


if __name__ == "__main__":
    main()
