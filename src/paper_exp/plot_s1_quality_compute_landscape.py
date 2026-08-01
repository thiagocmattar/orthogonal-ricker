"""Render the S1 quality--compute landscape and descriptive envelope table.

The first panel shows every canonical S1 scientific endpoint.  The second uses
the common primary-seed, common-learning-rate cohort and draws its exact
nondominated envelope.  The envelope is descriptive and is not a promotion
rule.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from paper_exp.plot_api import (
    PublicationProfile,
    export_figure,
)
from paper_exp.plot_report07 import S1Row, _finish_axis, load_s1_rows
from paper_exp.plot_style import REPORT04_PLOT_STYLE


DEFAULT_OUTPUT = Path(
    "figures/113-pythia-14m-s1-quality-compute-endpoint-landscape.pdf"
)
DEFAULT_TABLE_OUTPUT = Path(
    "report/07-2026-07-27-s1-ablation-study/"
    "07-2026-07-27-s1-frontier-table.tex"
)
PAPER_FIGURE_WIDTH_INCHES = 7.16
QUALITY_COMPUTE_PROFILE = PublicationProfile(
    width_inches=PAPER_FIGURE_WIDTH_INCHES,
    max_height_inches=3.35,
    min_text_points=7.0,
)
BLOCK_ORDER = ("S1-B0", "S1-B1", "S1-B2", "S1-B3", "S1-B4")
BLOCK_STYLES = {
    "S1-B0": ("B0 arch./LR", "#666666", "o"),
    "S1-B1": ("B1 fixed", "#D55E00", "s"),
    "S1-B2": ("B2 learned", "#0072B2", "^"),
    "S1-B3": ("B3 pressure", "#009E73", "D"),
    "S1-B4": ("B4 seeds", "#CC79A7", "P"),
}
ZOOM_LOSS_LIMITS = (6.975, 7.215)


def _central_screen_rows(cohort: Sequence[S1Row]) -> tuple[S1Row, ...]:
    """Return the primary-seed, common-LR, common-budget S1 screen."""

    rows = tuple(
        row
        for row in cohort
        if int(row.config["model_initialization_seed"]) == 0
        and int(row.config["data_order_seed"]) == 0
        and np.isclose(
            float(row.config["model_learning_rate"]),
            3e-5,
            rtol=0.0,
            atol=1e-15,
        )
    )
    if len(rows) != 112:
        raise ValueError(
            "The primary-seed common-LR S1 landscape must contain 112 "
            f"endpoints; found {len(rows)}."
        )
    return rows


def _scatter_blocks(axis: object, rows: Sequence[S1Row]) -> None:
    """Plot endpoint blocks with redundant color and marker encodings."""

    for block in BLOCK_ORDER:
        items = [row for row in rows if row.block == block]
        _label, color, marker = BLOCK_STYLES[block]
        axis.scatter(
            [row.r_model_pct for row in items],
            [row.loss for row in items],
            s=31,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            zorder=3,
        )


def _pareto_frontier(rows: Sequence[S1Row]) -> tuple[S1Row, ...]:
    """Return the descriptive loss-minimizing, opportunity-maximizing envelope."""

    frontier = [
        row
        for row in rows
        if not any(
            other.loss <= row.loss
            and other.r_model_pct >= row.r_model_pct
            and (
                other.loss < row.loss
                or other.r_model_pct > row.r_model_pct
            )
            for other in rows
        )
    ]
    return tuple(sorted(frontier, key=lambda row: row.r_model_pct))


def _draw_pareto_frontier(axis: object, rows: Sequence[S1Row]) -> None:
    """Draw and label the exact descriptive nondominated envelope."""

    frontier = _pareto_frontier(rows)
    x_values = [row.r_model_pct for row in frontier]
    y_values = [row.loss for row in frontier]
    axis.plot(
        x_values,
        y_values,
        color="#4D4D4D",
        linestyle=(0, (3.0, 2.2)),
        linewidth=0.9,
        alpha=0.78,
        zorder=3.5,
    )
    label_offsets = (
        (-8, -10),
        (-7, 7),
        (5, -9),
        (5, 7),
        (5, -9),
        (5, -9),
        (5, -9),
        (5, -9),
        (5, -9),
        (-16, -11),
        (5, 7),
        (-16, -10),
        (-17, 7),
    )
    if len(frontier) != len(label_offsets):
        raise ValueError(
            f"Expected 13 descriptive frontier points; found {len(frontier)}."
        )
    for index, (row, offset) in enumerate(
        zip(frontier, label_offsets, strict=True),
        start=1,
    ):
        _label, color, marker = BLOCK_STYLES[row.block]
        axis.scatter(
            [row.r_model_pct],
            [row.loss],
            s=49,
            marker=marker,
            facecolor=color,
            edgecolor="#202020",
            linewidth=1.05,
            zorder=4.0,
        )
        axis.annotate(
            f"F{index}",
            (row.r_model_pct, row.loss),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold",
            color="#202020",
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.78},
            zorder=5.0,
        )


def build_quality_compute_landscape_figure(
    cohort: Sequence[S1Row],
) -> Figure:
    """Build the complete-cloud and controlled-LR S1 endpoint panels."""

    if len(cohort) != 132:
        raise ValueError(
            f"The complete S1 landscape requires 132 endpoints; found {len(cohort)}."
        )
    central = _central_screen_rows(cohort)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            PAPER_FIGURE_WIDTH_INCHES,
            QUALITY_COMPUTE_PROFILE.max_height_inches,
        ),
        sharex=True,
        gridspec_kw={"width_ratios": (0.75, 1.25)},
    )

    _draw_pareto_frontier(axes[1], central)
    _scatter_blocks(axes[0], cohort)
    _scatter_blocks(axes[1], central)

    for axis in axes:
        axis.set_xlim(-0.8, 26.1)
        axis.set_xticks(np.arange(0.0, 26.0, 5.0))
        _finish_axis(axis)

    axes[0].set_ylim(5.72, 8.54)
    axes[0].set_title(
        r"$\mathbf{(a)}$ All S1 endpoints ($n=132$)",
        loc="left",
        color="#333333",
        fontweight="normal",
        fontsize=8.5,
        pad=6.0,
    )
    axes[0].add_patch(
        Rectangle(
            (-0.8, ZOOM_LOSS_LIMITS[0]),
            26.9,
            ZOOM_LOSS_LIMITS[1] - ZOOM_LOSS_LIMITS[0],
            fill=False,
            edgecolor="#555555",
            linewidth=0.8,
            linestyle=":",
            zorder=2,
        )
    )
    axes[0].text(
        25.4,
        ZOOM_LOSS_LIMITS[1] + 0.035,
        "panel (b)",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )

    axes[1].set_ylim(*ZOOM_LOSS_LIMITS)
    axes[1].set_yticks(np.arange(6.98, 7.211, 0.04))
    axes[1].set_title(
        r"$\mathbf{(b)}$ Primary-seed common-LR screen ($n=112$)",
        loc="left",
        color="#333333",
        fontweight="normal",
        fontsize=8.5,
        pad=6.0,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=5.0,
            markerfacecolor=color,
            markeredgecolor="white",
            label=label,
        )
        for label, color, marker in (
            BLOCK_STYLES[block] for block in BLOCK_ORDER
        )
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#4D4D4D",
            linestyle=(0, (3.0, 2.2)),
            linewidth=0.9,
            alpha=0.78,
            label="Descriptive nondominated envelope",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.3,
        fontsize=7.0,
        handlelength=1.7,
    )
    figure.supxlabel(
        (
            r"Potentially avoidable logical products, "
            r"$R_{\mathrm{model}}$ (%)"
        ),
        x=0.54,
        y=0.105,
        fontsize=8.0,
    )
    figure.supylabel(
        "Validation loss",
        x=0.015,
        y=0.51,
        fontsize=8.0,
    )
    figure.subplots_adjust(
        left=0.090,
        right=0.995,
        bottom=0.245,
        top=0.90,
        wspace=0.235,
    )
    return figure


def _frontier_design(row: S1Row) -> str:
    """Return a compact, TeX-ready design label from the registered config."""

    if row.block == "S1-B0":
        return f"{row.architecture} AdamW"
    family = r"$G^+$" if row.config.get("gate_family") == "gplus" else r"$G^\pm$"
    if row.block == "S1-B1":
        return (
            f"{row.architecture} {family}, "
            rf"$\kappa={float(row.config['kappa']):.2f}$"
        )
    if row.block == "S1-B2":
        scale = (
            "RMS"
            if row.config.get("threshold_scale") == "rms_relative"
            else "ABS"
        )
        sharing = (
            "PLS"
            if row.config.get("kappa_scope") == "per_layer_site"
            else str(row.config.get("kappa_scope"))
        )
        return f"{row.architecture} ATG {family}, {scale}, {sharing}"
    if row.block == "S1-B3":
        method = {
            "ricker_naive": "RN",
            "orthogonal_ricker": "OR",
            "l1_naive": "L1N",
            "orthogonal_l1": "OL1",
        }[str(row.config["pressure_method"])]
        if method in {"RN", "OR"}:
            return (
                f"{row.architecture} {method}, "
                rf"$(w,c,\sigma)=({float(row.config['pressure_weight']):g},"
                rf"{float(row.config['ricker_c']):g},"
                rf"{float(row.config['ricker_sigma']):g})$"
            )
        return (
            f"{row.architecture} {method}, "
            rf"$w={float(row.config['pressure_weight']):g}$"
        )
    raise ValueError(f"Unsupported frontier block {row.block!r}.")


def write_frontier_table(
    cohort: Sequence[S1Row],
    output: str | Path = DEFAULT_TABLE_OUTPUT,
) -> Path:
    """Write the exact Figure 113 envelope as a report-ready TeX table."""

    frontier = _pareto_frontier(_central_screen_rows(cohort))
    if len(frontier) != 13:
        raise ValueError(f"Expected 13 frontier endpoints; found {len(frontier)}.")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{Exact members of the descriptive nondominated envelope in Figure~\ref{fig:landscape}. The eligible cohort fixes model/data-order seeds at 0/0, model LR at $3\times10^{-5}$, the 2,048-step budget, and the frozen selection partition ($n=112$). F12--F13 exceed the predeclared $+0.05$ pressure-loss guardrail and are descriptive, not promotion choices.}",
        r"\label{tab:frontier}",
        r"\begin{tabularx}{\textwidth}{c r c X r r}",
        r"\toprule",
        r"ID & Cfg & Block & Design & $L$ & $R_m$ (\%) \\",
        r"\midrule",
    ]
    for index, row in enumerate(frontier, start=1):
        lines.append(
            f"F{index} & {row.number} & {row.block.removeprefix('S1-')} & "
            f"{_frontier_design(row)} & {row.loss:.5f} & "
            f"{row.r_model_pct:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_quality_compute_landscape_figure(
    *,
    config_registry: str | Path = (
        "docs/experimental-design/config-registry.yaml"
    ),
    run_registry: str | Path = (
        "docs/experimental-design/run-registry.yaml"
    ),
    output: str | Path = DEFAULT_OUTPUT,
    table_output: str | Path | None = DEFAULT_TABLE_OUTPUT,
    save_png: bool = True,
) -> tuple[Path, ...]:
    """Load the canonical S1 cohort and export the standalone landscape."""

    cohort = load_s1_rows(config_registry, run_registry)
    generated = tuple(
        export_figure(
            lambda: build_quality_compute_landscape_figure(cohort),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=QUALITY_COMPUTE_PROFILE,
        )
    )
    if table_output is not None:
        write_frontier_table(cohort, table_output)
    return generated


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
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE_OUTPUT))
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Write only the vector PDF.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = generate_quality_compute_landscape_figure(
        config_registry=args.config_registry,
        run_registry=args.run_registry,
        output=args.output,
        table_output=args.table_output,
        save_png=not args.no_png,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
