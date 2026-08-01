"""Render the enlarged S1-B0 learning-rate-effect figure.

This supplemental figure preserves the registered Figure 103 cohort. It shows
the within-architecture learning-rate responses and a separate architecture
comparison of validation-loss endpoints at ``model_learning_rate = 1e-4``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from paper_exp.plot_api import (
    DOUBLE_COLUMN_WIDTH_INCHES,
    PublicationProfile,
    export_figure,
)
from paper_exp.plot_report07 import (
    ARCHITECTURE_COLORS,
    ARCHITECTURE_MARKERS,
    B0_LR_ARCHITECTURES,
    S1Row,
    _b0_lr_triplets,
    _finish_axis,
    load_s1_rows,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


DEFAULT_OUTPUT = Path(
    "figures/112-pythia-14m-s1-learning-rate-effect.pdf"
)
LEARNING_RATE_EFFECT_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=3.35,
    min_text_points=8.0,
)
LR_TICKS = (1e-5, 3e-5, 1e-4)
LR_TICK_LABELS = (
    r"$10^{-5}$",
    r"$3{\times}10^{-5}$",
    r"$10^{-4}$",
)
ZOOM_LR = 1e-4


def _lr_1e4_endpoints(
    cohort: Sequence[S1Row],
) -> tuple[tuple[str, S1Row], ...]:
    """Return the five matched ``1e-4`` endpoints in legend order."""

    endpoints: list[tuple[str, S1Row]] = []
    for architecture, rows in _b0_lr_triplets(cohort):
        matches = tuple(
            row
            for row in rows
            if np.isclose(
                float(row.config["model_learning_rate"]),
                ZOOM_LR,
                rtol=0.0,
                atol=1e-15,
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"{architecture} must have exactly one LR={ZOOM_LR:g} endpoint; "
                f"found {len(matches)}."
            )
        endpoints.append((architecture, matches[0]))
    return tuple(endpoints)


def _draw_endpoint_panel(axis: object, cohort: Sequence[S1Row]) -> None:
    """Draw validation loss by architecture at ``model LR = 1e-4``."""

    endpoints = _lr_1e4_endpoints(cohort)
    x_positions = np.arange(len(endpoints), dtype=float)

    for x_position, (architecture, row) in zip(
        x_positions,
        endpoints,
        strict=True,
    ):
        color = ARCHITECTURE_COLORS[architecture]
        marker = ARCHITECTURE_MARKERS[architecture]
        axis.scatter(
            [x_position],
            [row.loss],
            color=color,
            marker=marker,
            s=28,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axis.text(
            x_position,
            row.loss + 0.008,
            f"{row.loss:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    axis.set_xlim(-0.55, len(endpoints) - 0.45)
    axis.set_ylim(5.84, 6.09)
    axis.set_xticks(
        x_positions,
        ("A0", "A1-H", "A3", "A6\nPRE", "A6\nPOST"),
    )
    axis.set_yticks((5.85, 5.90, 5.95, 6.00, 6.05))
    axis.set_xlabel("Architecture")
    axis.set_ylabel("Validation loss")
    axis.set_title(
        r"(c) Loss at LR $=10^{-4}$",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    axis.tick_params(axis="x", labelsize=8.0, length=0, pad=2.0)
    axis.tick_params(axis="y", labelsize=8.0, pad=1.5)
    _finish_axis(axis)
    axis.grid(axis="x", visible=False)


def build_learning_rate_effect_figure(cohort: Sequence[S1Row]) -> Figure:
    """Build the three-panel B0 learning-rate response at report size."""

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            LEARNING_RATE_EFFECT_PROFILE.width_inches,
            LEARNING_RATE_EFFECT_PROFILE.max_height_inches,
        ),
        gridspec_kw={"width_ratios": (1.08, 1.08, 1.00)},
    )

    for architecture, rows in _b0_lr_triplets(cohort):
        x_values = [
            float(row.config["model_learning_rate"])
            for row in rows
        ]
        color = ARCHITECTURE_COLORS[architecture]
        marker = ARCHITECTURE_MARKERS[architecture]
        axes[0].plot(
            x_values,
            [row.loss for row in rows],
            color=color,
            marker=marker,
            markersize=5.2,
            linewidth=1.45,
            label=architecture,
            zorder=2,
        )
        axes[1].plot(
            x_values,
            [row.r_model_pct for row in rows],
            color=color,
            marker=marker,
            markersize=5.2,
            linewidth=1.45,
            zorder=2,
        )

    for axis in axes[:2]:
        axis.set_xscale("log")
        axis.set_xticks(LR_TICKS, LR_TICK_LABELS)
        axis.set_xlabel("Model learning rate")
        axis.axvline(
            ZOOM_LR,
            color="#777777",
            linestyle=":",
            linewidth=0.8,
            zorder=0,
        )
        _finish_axis(axis)

    axes[0].set_ylabel("Validation loss")
    axes[0].set_ylim(5.75, 8.53)
    axes[0].set_title(
        "(a) Validation loss",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    axes[1].set_ylabel(r"$R_{\mathrm{model}}$ (%)")
    axes[1].set_ylim(-0.55, 15.65)
    axes[1].set_yticks(np.arange(0.0, 16.0, 2.5))
    axes[1].set_title(
        "(b) Logical product opportunity",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )

    _draw_endpoint_panel(axes[2], cohort)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=1.45,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.190,
        top=0.815,
        wspace=0.310,
    )
    return figure


def generate_learning_rate_effect_figure(
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
            lambda: build_learning_rate_effect_figure(cohort),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=LEARNING_RATE_EFFECT_PROFILE,
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
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Write only the vector PDF.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = generate_learning_rate_effect_figure(
        config_registry=args.config_registry,
        run_registry=args.run_registry,
        output=args.output,
        save_png=not args.no_png,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
