from __future__ import annotations

import json
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


def generate_run_diagnostics(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = _read_jsonl(run_path / "events.jsonl")
    metrics = read_json(run_path / "metrics.json")
    train_events = [event for event in events if event.get("event") == "train"]
    validation_events = [event for event in events if event.get("event") == "validation"]
    if not train_events:
        raise ValueError(f"No train events found in {run_path / 'events.jsonl'}")

    _plot_run_diagnostics(train_events, validation_events, metrics, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_run_diagnostics(train_events, validation_events, metrics, png_path)
        outputs.append(png_path)
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


def _plot_run_diagnostics(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 8.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.42, wspace=0.48)

    ax_loss = fig.add_subplot(grid[0, :])
    ax_grad = fig.add_subplot(grid[1, 0])
    ax_weight = fig.add_subplot(grid[1, 1])
    ax_stats = fig.add_subplot(grid[2, :])

    train_tokens = [event["tokens_seen"] for event in train_events]
    train_loss = [event["train_loss"] for event in train_events]
    ax_loss.plot(train_tokens, train_loss, marker="o", markersize=2.5, linewidth=1.3, label="train")
    if validation_events:
        val_tokens = [event["tokens_seen"] for event in validation_events]
        val_loss = [event["validation_loss"] for event in validation_events]
        ax_loss.plot(val_tokens, val_loss, marker="s", markersize=3.0, linewidth=1.3, label="validation")
    ax_loss.set_title("Loss vs Tokens")
    ax_loss.set_xlabel("Tokens seen")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)

    grad_events = [event for event in train_events if event.get("grad_norm") is not None]
    ax_grad.plot(
        [event["tokens_seen"] for event in grad_events],
        [event["grad_norm"] for event in grad_events],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    ax_grad.set_title("Gradient Norm")
    ax_grad.set_xlabel("Tokens seen")
    ax_grad.set_ylabel("L2 norm")

    weight_events = [event for event in train_events if event.get("weight_norm") is not None]
    ax_weight.plot(
        [event["tokens_seen"] for event in weight_events],
        [event["weight_norm"] for event in weight_events],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    ax_weight.set_title("Weight Norm")
    ax_weight.set_xlabel("Tokens seen")
    ax_weight.set_ylabel("L2 norm")
    ax_weight.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_weight.tick_params(axis="y", labelsize=7)

    tokens_per_step = metrics.get("calibration/tokens_per_step")
    logged_step_rates = [
        tokens_per_step / event["step_wall_seconds"]
        for event in train_events
        if tokens_per_step and event.get("step_wall_seconds")
    ]
    peak_logged_tokens_per_second = max(logged_step_rates) if logged_step_rates else None
    avg_tokens_per_second = metrics.get("calibration/tokens_per_second")
    if peak_logged_tokens_per_second is not None and avg_tokens_per_second is not None:
        peak_tokens_per_second = max(peak_logged_tokens_per_second, avg_tokens_per_second)
    else:
        peak_tokens_per_second = peak_logged_tokens_per_second or avg_tokens_per_second
    stats = [
        ("Peak tokens/s", peak_tokens_per_second, "tok/s"),
        ("Average tokens/s", avg_tokens_per_second, "tok/s"),
        ("Peak GPU allocated", metrics.get("calibration/peak_gpu_memory_mb"), "MB"),
        ("Peak GPU reserved", metrics.get("calibration/peak_gpu_reserved_mb"), "MB"),
        ("Train wall time", metrics.get("calibration/wall_seconds_train"), "sec"),
        ("Total wall time", metrics.get("calibration/wall_seconds_total"), "sec"),
        ("Final model size", metrics.get("checkpoint/final_size_mb"), "MB"),
    ]
    ax_stats.axis("off")
    ax_stats.set_title("Run Statistics")
    _draw_stats_panel(ax_stats, stats)

    fig.suptitle("Pythia-14M MiniPile Run Diagnostics", y=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _format_stat_value(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_stats_panel(ax: Any, stats: list[tuple[str, Any, str]]) -> None:
    visible_stats = [(label, value, unit) for label, value, unit in stats if value is not None]
    columns = 3
    rows = (len(visible_stats) + columns - 1) // columns
    for index, (label, value, unit) in enumerate(visible_stats):
        row = index // columns
        column = index % columns
        x = (column + 0.5) / columns
        y = 0.74 - row * (0.58 / max(rows - 1, 1))
        ax.text(x, y + 0.08, label, transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.text(
            x,
            y - 0.07,
            f"{_format_stat_value(float(value))} {unit}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )


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
