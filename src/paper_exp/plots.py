from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from paper_exp.utils import read_json


COLORBLIND_SAFE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

PLOT_STYLE = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE_COLORS),
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def generate_plots(
    *,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    rows = collect_numeric_metrics(Path(results_dir))
    output_pdf = figures_path / "01-results-summary.pdf"

    if rows:
        _plot_metric_summary(rows, output_pdf)
    else:
        _plot_empty_placeholder(output_pdf)

    outputs = [output_pdf]
    if save_png:
        output_png = output_pdf.with_suffix(".png")
        if rows:
            _plot_metric_summary(rows, output_png)
        else:
            _plot_empty_placeholder(output_png)
        outputs.append(output_png)

    return outputs


def collect_numeric_metrics(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*/*/metrics.json")):
        metrics = read_json(metrics_path)
        manifest_path = metrics_path.parent / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}

        for metric_name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            rows.append(
                {
                    "experiment_name": manifest.get("experiment_name", metrics_path.parent.parent.name),
                    "config_id": manifest.get("config_id", metrics_path.parent.parent.name),
                    "run_id": manifest.get("run_id", metrics_path.parent.name),
                    "metric_name": metric_name,
                    "value": float(value),
                }
            )
    return rows


def _plot_metric_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    metric_name = sorted({row["metric_name"] for row in rows})[0]
    selected = [row for row in rows if row["metric_name"] == metric_name]
    labels = [f"{row['config_id']}\n{row['run_id'][:8]}" for row in selected]
    values = [row["value"] for row in selected]

    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_title("Result Summary")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Run")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if values and min(values) >= 0:
        ax.set_ylim(bottom=0)
    ax.text(0.99, 0.98, f"n={len(selected)} runs", transform=ax.transAxes, ha="right", va="top")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_empty_placeholder(output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "No numeric results found.\nRun make smoke or make baseline first.",
        ha="center",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
