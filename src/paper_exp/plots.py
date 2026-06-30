from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
ADAMW_COLOR = "#000000"
ADAMW_LINEWIDTH = 2.6
ADAMW_MARKER_SCALE = 1.35
DEFAULT_SERIES_LINEWIDTH = 1.2

PRESSURE_SHORT_RUNS = [
    ("AdamW baseline", "03-pythia-14m-minipile-random-full-10min"),
    ("Ricker naive", "08-pythia-14m-minipile-ricker-naive-short"),
    ("L1 naive", "09-pythia-14m-minipile-l1-naive-short"),
    ("Orthogonal Ricker", "10-pythia-14m-minipile-orthogonal-ricker-short"),
    ("Orthogonal L1", "11-pythia-14m-minipile-orthogonal-l1-short"),
]

FIXED_STEP_SWEEP_NAME = "pressure_fixed_step_v1"
FIXED_STEP_ROLE_LABELS = {
    "adamw": "AdamW",
    "ricker_naive": "Ricker naive",
    "orthogonal_ricker": "Orthogonal Ricker",
    "l1_naive": "L1 naive",
    "orthogonal_l1": "Orthogonal L1",
}
FIXED_STEP_ROLE_MARKERS = {
    "adamw": "D",
    "ricker_naive": "o",
    "orthogonal_ricker": "s",
    "l1_naive": "^",
    "orthogonal_l1": "P",
}
FIXED_STEP_METHOD_FIGURES = (
    (8, "ricker_naive", "naive-ricker", "Naive Ricker"),
    (9, "l1_naive", "naive-l1", "Naive L1"),
    (10, "orthogonal_ricker", "orthogonal-ricker", "Orthogonal Ricker"),
    (11, "orthogonal_l1", "orthogonal-l1", "Orthogonal L1"),
)
SELECTED_CLIPPING_FRONTIER_CONFIGS = (12, 29, 33, 18, 25)
HIGH_PRESSURE_EXPANSION_CONFIGS = tuple([12, *range(35, 49)])
HIGH_PRESSURE_LEARNING_FIGURES = (
    (17, (12, *range(35, 40)), "rn", "High-pressure RN Learning Curves", "AdamW plus RN configs 35-39"),
    (18, (12, *range(40, 45)), "or", "High-pressure OR Learning Curves", "AdamW plus OR configs 40-44"),
    (19, (12, 45, 46, 47, 48), "l1", "High-pressure L1N/OL1 Learning Curves", "AdamW plus L1N/OL1 configs 45-48"),
)
HIGH_PRESSURE_OR_L1_NORM_CONFIGS = tuple([12, *range(40, 49)])
SELECTED_ACTIVATION_HISTOGRAM_EXPERIMENT = "49-pythia-14m-pressure-fixed-2048-selected-activation-histograms"
FULL_PASS_SELECTED_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("L1N w0.5", "51-pythia-14m-minipile-l1-naive-full-pass-w0p5"),
    ("OL1 w0.5", "52-pythia-14m-minipile-orthogonal-l1-full-pass-w0p5"),
    ("RN w0.1 c0.05 s0.025", "53-pythia-14m-minipile-ricker-naive-full-pass-w0p1-c0p05-s0p025"),
    ("OR w0.1 c0.05 s0.025", "54-pythia-14m-minipile-orthogonal-ricker-full-pass-w0p1-c0p05-s0p025"),
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

    results_path = Path(results_dir)
    outputs = _generate_known_paper_figures(results_path, figures_path, save_png=save_png)
    if outputs:
        return outputs

    rows = collect_numeric_metrics(results_path)
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


def _generate_known_paper_figures(results_path: Path, figures_path: Path, *, save_png: bool) -> list[Path]:
    outputs: list[Path] = []

    diagnostic_run = _latest_run_with(
        results_path / "03-pythia-14m-minipile-random-full-10min",
        "events.jsonl",
    )
    if diagnostic_run is not None:
        output_pdf = figures_path / "01-pythia-14m-minipile-random-full-10min-diagnostics.pdf"
        outputs.extend(generate_run_diagnostics(run_dir=diagnostic_run, output=output_pdf, save_png=save_png))

    clipping_run = _latest_run_with(
        results_path / "03-pythia-14m-minipile-random-full-10min-clipping-sweep",
        "clipping_frontier.jsonl",
    )
    if clipping_run is not None:
        output_pdf = figures_path / "02-pythia-14m-minipile-clipping-frontier-smoke.pdf"
        outputs.extend(generate_clipping_frontier(run_dir=clipping_run, output=output_pdf, save_png=save_png))

    pressure_runs = _latest_labeled_runs(results_path, PRESSURE_SHORT_RUNS, "events.jsonl")
    if len(pressure_runs) >= 2:
        output_pdf = figures_path / "03-pythia-14m-pressure-short-learning-curves.pdf"
        outputs.extend(generate_pressure_comparison(runs=pressure_runs, output=output_pdf, save_png=save_png))

    clipping_runs = _latest_labeled_runs(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in PRESSURE_SHORT_RUNS[1:]],
        "clipping_frontier.jsonl",
    )
    if len(clipping_runs) >= 2:
        output_pdf = figures_path / "04-pythia-14m-pressure-short-clipping-frontiers.pdf"
        outputs.extend(generate_clipping_comparison(runs=clipping_runs, output=output_pdf, save_png=save_png))

    fixed_step_rows = _load_fixed_step_sweep_rows(results_path)
    if len(fixed_step_rows) >= 2:
        output_pdf = figures_path / "05-pythia-14m-pressure-fixed-2048-summary.pdf"
        outputs.extend(
            generate_fixed_step_sweep_summary(rows=fixed_step_rows, output=output_pdf, save_png=save_png)
        )

        output_pdf = figures_path / "06-pythia-14m-pressure-fixed-2048-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_learning_curves(rows=fixed_step_rows, output=output_pdf, save_png=save_png)
        )

    fixed_step_clipping = _load_fixed_step_clipping_series(results_path, fixed_step_rows)
    if len(fixed_step_clipping) >= 2:
        output_pdf = figures_path / "07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_clipping_frontiers(
                series=fixed_step_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for index, role, slug, role_label in FIXED_STEP_METHOD_FIGURES:
        role_rows = _select_fixed_step_rows_for_role(fixed_step_rows, role)
        if len(role_rows) < 2:
            continue
        output_pdf = figures_path / f"{index:02d}-pythia-14m-pressure-fixed-2048-{slug}-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_role_learning_curves(
                rows=role_rows,
                role_label=role_label,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for offset, role, slug, role_label in FIXED_STEP_METHOD_FIGURES:
        role_series = _select_fixed_step_clipping_for_role(fixed_step_clipping, role)
        if len(role_series) < 2:
            continue
        output_pdf = figures_path / f"{offset + 4:02d}-pythia-14m-pressure-fixed-2048-{slug}-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_role_clipping_frontiers(
                series=role_series,
                role_label=role_label,
                output=output_pdf,
                save_png=save_png,
            )
        )

    selected_clipping = _select_fixed_step_clipping_by_config_indices(
        fixed_step_clipping,
        SELECTED_CLIPPING_FRONTIER_CONFIGS,
    )
    if len(selected_clipping) == len(SELECTED_CLIPPING_FRONTIER_CONFIGS):
        output_pdf = figures_path / "16-pythia-14m-pressure-fixed-2048-selected-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_selected_clipping_frontiers(
                series=selected_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for index, config_indices, slug, title, scope_note in HIGH_PRESSURE_LEARNING_FIGURES:
        high_pressure_rows = _select_fixed_step_rows_by_config_indices(
            fixed_step_rows,
            config_indices,
        )
        if len(high_pressure_rows) < 2:
            continue
        output_pdf = figures_path / f"{index:02d}-pythia-14m-pressure-fixed-2048-high-pressure-{slug}-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_learning_curves(
                rows=high_pressure_rows,
                output=output_pdf,
                title=title,
                scope_note=scope_note,
                save_png=save_png,
            )
        )

    high_pressure_clipping = _select_fixed_step_clipping_by_config_indices(
        fixed_step_clipping,
        HIGH_PRESSURE_EXPANSION_CONFIGS,
    )
    if len(high_pressure_clipping) >= 2:
        output_pdf = figures_path / "20-pythia-14m-pressure-fixed-2048-high-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_clipping_frontiers(
                series=high_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    high_pressure_or_l1_rows = _select_fixed_step_rows_by_config_indices(
        fixed_step_rows,
        HIGH_PRESSURE_OR_L1_NORM_CONFIGS,
    )
    if len(high_pressure_or_l1_rows) >= 2:
        output_pdf = figures_path / "21-pythia-14m-pressure-fixed-2048-or-l1-weight-norms.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_weight_norms(
                rows=high_pressure_or_l1_rows,
                output=output_pdf,
                save_png=save_png,
            )
        )

    activation_histogram_run = _latest_run_with(
        results_path / SELECTED_ACTIVATION_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if activation_histogram_run is not None:
        output_pdf = figures_path / "22-pythia-14m-pressure-fixed-2048-selected-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=activation_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_runs = _latest_labeled_runs(results_path, FULL_PASS_SELECTED_RUNS, "events.jsonl")
    if len(full_pass_runs) >= 2:
        output_pdf = figures_path / "28-pythia-14m-minipile-full-pass-selected-gradient-diagnostics.pdf"
        outputs.extend(
            generate_full_pass_gradient_diagnostics(
                runs=full_pass_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )
        output_pdf = figures_path / "30-pythia-14m-minipile-full-pass-pressure-dominance.pdf"
        outputs.extend(
            generate_full_pass_pressure_dominance(
                runs=full_pass_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_rms_clipping = _latest_labeled_runs(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in FULL_PASS_SELECTED_RUNS],
        "clipping_frontier.jsonl",
    )
    full_pass_rms_clipping = [
        (label, run)
        for label, run in full_pass_rms_clipping
        if _is_rms_clipping_run(run)
    ]
    if len(full_pass_rms_clipping) >= 2:
        output_pdf = figures_path / "29-pythia-14m-minipile-full-pass-selected-rms-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_rms_clipping_frontiers(
                runs=full_pass_rms_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    return outputs


def _latest_run_with(experiment_dir: Path, artifact_name: str) -> Path | None:
    if not experiment_dir.exists():
        return None
    candidates = [path for path in sorted(experiment_dir.iterdir()) if (path / artifact_name).exists()]
    return candidates[-1] if candidates else None


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


def generate_clipping_frontier(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frontier_path = run_path / "clipping_frontier.jsonl"
    rows = _read_jsonl(frontier_path)
    if not rows:
        raise ValueError(f"No clipping frontier rows found in {frontier_path}")

    _plot_clipping_frontier(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_clipping_frontier(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_pressure_comparison(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_pressure_series(runs)
    if not series:
        raise ValueError("No pressure comparison runs with train events were found.")

    _plot_pressure_comparison(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_pressure_comparison(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_clipping_comparison(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No clipping comparison runs were found.")

    _plot_clipping_comparison(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_clipping_comparison(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_gradient_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No full-pass train events were found.")

    _plot_full_pass_gradient_diagnostics(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_gradient_diagnostics(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_rms_clipping_frontiers(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No RMS clipping comparison runs were found.")

    _plot_full_pass_rms_clipping_frontiers(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_rms_clipping_frontiers(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_pressure_dominance(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No full-pass train events were found.")

    _plot_full_pass_pressure_dominance(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_pressure_dominance(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_sweep_summary(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_fixed_step_sweep_summary(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_sweep_summary(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_learning_curves(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = _select_representative_fixed_step_rows(rows)
    _plot_fixed_step_learning_curves(selected_rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(selected_rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_series = _select_representative_clipping_series(series)
    _plot_fixed_step_clipping_frontiers(selected_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(selected_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_role_learning_curves(
    *,
    rows: list[dict[str, Any]],
    role_label: str,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{role_label} Fixed-step Learning Curves"
    subtitle = f"n={len(rows)} runs: AdamW plus all {role_label} settings"
    _plot_fixed_step_learning_curves(
        rows,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(
            rows,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_role_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    role_label: str,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{role_label} Post-hoc Clipping Frontiers"
    total_points = sum(len(item.get("rows", [])) for item in series)
    subtitle = (
        f"n={len(series)} runs, {total_points} sweep points: "
        f"AdamW plus all {role_label} settings; validation-loss axis zoomed"
    )
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_selected_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(len(item.get("rows", [])) for item in series)
    title = "Selected Post-hoc Clipping Frontiers"
    subtitle = f"n={len(series)} runs, {total_points} sweep points; validation-loss axis zoomed"
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_learning_curves(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    title: str,
    scope_note: str,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = sorted({int(row["tokens_seen"]) for row in rows if row.get("tokens_seen")})
    token_note = f"{tokens[0]:,} tokens/run" if len(tokens) == 1 else "fixed token budget"
    subtitle = f"n={len(rows)} runs: {scope_note}; {token_note}"
    _plot_fixed_step_learning_curves(
        rows,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
        legend_ncol=2,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(
            rows,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
            legend_ncol=2,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(len(item.get("rows", [])) for item in series)
    title = "High-pressure Post-hoc Clipping Frontiers"
    subtitle = f"n={len(series)} runs, {total_points} sweep points: AdamW plus configs 35-48"
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
        legend_ncol=3,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
            legend_ncol=3,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_weight_norms(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_fixed_step_high_pressure_weight_norms(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_high_pressure_weight_norms(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_activation_histogram_grid(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "activation_histograms.json")
    _plot_activation_histogram_grid(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_activation_histogram_grid(payload, png_path)
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
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            rows.append(
                {
                    "experiment_name": manifest.get("experiment_name", metrics_path.parent.parent.name),
                    "config_id": manifest.get("config_id", metrics_path.parent.parent.name),
                    "run_id": manifest.get("run_id", metrics_path.parent.name),
                    "metric_name": metric_name,
                    "value": numeric_value,
                }
            )
    return rows


def _latest_labeled_runs(
    results_path: Path,
    experiments: list[tuple[str, str]],
    artifact_name: str,
) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for label, experiment_id in experiments:
        run = _latest_run_with(results_path / experiment_id, artifact_name)
        if run is not None:
            runs.append((label, run))
    return runs


def _load_pressure_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        events_path = run_path / "events.jsonl"
        metrics_path = run_path / "metrics.json"
        if not events_path.exists() or not metrics_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if not train_events:
            continue
        series.append(
            {
                "label": label,
                "run_dir": run_path,
                "train_events": train_events,
                "validation_events": validation_events,
                "metrics": read_json(metrics_path),
            }
        )
    return series


def _load_clipping_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        frontier_path = run_path / "clipping_frontier.jsonl"
        if not frontier_path.exists():
            continue
        rows = [
            row
            for row in _read_jsonl(frontier_path)
            if row.get("achieved_sparsity") is not None
            and row.get("validation_loss") is not None
            and math.isfinite(float(row["achieved_sparsity"]))
            and math.isfinite(float(row["validation_loss"]))
        ]
        if rows:
            series.append({"label": label, "run_dir": run_path, "rows": rows})
    return series


def _load_event_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        events_path = run_path / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if train_events:
            series.append(
                {
                    "label": label,
                    "run_dir": run_path,
                    "train_events": train_events,
                    "validation_events": validation_events,
                }
            )
    return series


def _is_rms_clipping_run(run_dir: str | Path) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    return bool(manifest.get("rms_multipliers"))


def _load_fixed_step_sweep_rows(results_path: Path) -> list[dict[str, Any]]:
    latest_by_config: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(results_path.glob("*/*/metrics.json")):
        run_path = metrics_path.parent
        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        sweep = manifest.get("sweep") or {}
        if sweep.get("name") != FIXED_STEP_SWEEP_NAME:
            continue
        metrics = read_json(metrics_path)
        if int(metrics.get("calibration/optimizer_steps", -1)) != int(sweep.get("fixed_steps", -1)):
            continue

        config_id = str(manifest.get("config_id", run_path.parent.name))
        pressure = manifest.get("activation_pressure") or {}
        row = {
            "config_index": _config_index(config_id),
            "config_id": config_id,
            "run_id": manifest.get("run_id", run_path.name),
            "run_sequence": int(manifest.get("run_sequence", 0)),
            "run_dir": run_path,
            "role": sweep.get("role", "unknown"),
            "label": _fixed_step_label(sweep.get("role", "unknown"), pressure),
            "short_label": _fixed_step_short_label(sweep.get("role", "unknown"), pressure),
            "pressure": pressure,
            "train_loss": metrics.get("calibration/train_loss_final"),
            "train_loss_mean": metrics.get("calibration/train_loss_mean"),
            "validation_loss": metrics.get("calibration/validation_loss_final"),
            "validation_loss_best": metrics.get("calibration/validation_loss_best"),
            "tokens_seen": metrics.get("calibration/tokens_seen"),
            "tokens_per_second": metrics.get("calibration/tokens_per_second"),
            "wall_seconds": metrics.get("calibration/wall_seconds_total"),
            "peak_gpu_memory_mb": metrics.get("calibration/peak_gpu_memory_mb"),
            "final_model_size_mb": metrics.get("checkpoint/final_size_mb"),
            "mlp_weight_norm_final": metrics.get("calibration/mlp_weight_norm_final"),
            "near_zero_k01": metrics.get("final/activation/near_zero_mass/k1em02"),
            "near_zero_k03": metrics.get("final/activation/near_zero_mass/k3em02"),
            "pressure_loss": metrics.get("final/pressure_loss"),
        }
        previous = latest_by_config.get(config_id)
        if previous is None or row["run_sequence"] >= previous["run_sequence"]:
            latest_by_config[config_id] = row
    return sorted(latest_by_config.values(), key=lambda row: row["config_index"])


def _load_fixed_step_clipping_series(results_path: Path, training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_by_config = {row["config_id"]: row for row in training_rows}
    latest_by_config: dict[str, dict[str, Any]] = {}
    for frontier_path in sorted(results_path.glob("*-clipping-sweep/*/clipping_frontier.jsonl")):
        run_path = frontier_path.parent
        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        source_sweep = manifest.get("source_sweep") or {}
        if source_sweep.get("name") != FIXED_STEP_SWEEP_NAME:
            continue
        source_run = manifest.get("source_run")
        if not source_run:
            continue
        source_config_id = Path(source_run).parent.name
        training_row = training_by_config.get(source_config_id)
        if training_row is None:
            continue
        rows = [
            row
            for row in _read_jsonl(frontier_path)
            if row.get("achieved_sparsity") is not None
            and row.get("validation_loss") is not None
            and math.isfinite(float(row["achieved_sparsity"]))
            and math.isfinite(float(row["validation_loss"]))
        ]
        if not rows:
            continue
        series = {
            "config_id": source_config_id,
            "run_sequence": int(manifest.get("run_sequence", 0)),
            "run_dir": run_path,
            "training": training_row,
            "label": training_row["label"],
            "short_label": training_row["short_label"],
            "role": training_row["role"],
            "rows": rows,
        }
        previous = latest_by_config.get(source_config_id)
        if previous is None or series["run_sequence"] >= previous["run_sequence"]:
            latest_by_config[source_config_id] = series
    return sorted(latest_by_config.values(), key=lambda item: item["training"]["config_index"])


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


def _final_mlp_weight_norm_from_checkpoint(run_path: Path) -> float | None:
    checkpoint_path = run_path / "checkpoints" / "final" / "model.safetensors"
    if not checkpoint_path.exists():
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    total = 0.0
    tensor_count = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".mlp." not in key or not key.endswith(".weight"):
                continue
            tensor = handle.get_tensor(key).float()
            param_norm = tensor.norm(2).item()
            total += param_norm * param_norm
            tensor_count += 1
    if tensor_count == 0:
        return None
    return total**0.5


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


def _plot_pressure_comparison(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(7.4, 8.4))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    token_limit = _short_run_token_limit(series)
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        label = item["label"]
        train_events = _events_up_to(item["train_events"], token_limit)
        validation_events = _events_up_to(item["validation_events"], token_limit)

        ax_train.plot(
            [event["tokens_seen"] for event in train_events],
            [event["train_loss"] for event in train_events],
            marker="o",
            markersize=_marker_size(label, 2.2),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

        if validation_events:
            ax_val.plot(
                [event["tokens_seen"] for event in validation_events],
                [event["validation_loss"] for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.5),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                [event["tokens_seen"] for event in near_zero_events],
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                [event["tokens_seen"] for event in pressure_events],
                [event["pressure_loss"] for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen")
    ax_train.set_ylabel("Task loss")

    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen")
    ax_val.set_ylabel("Loss")

    ax_near_zero.set_title("MLP Hidden Near-zero Mass")
    ax_near_zero.set_xlabel("Tokens seen")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    if not ax_near_zero.lines:
        ax_near_zero.text(0.5, 0.5, "No activation-pressure metrics found.", ha="center", va="center")

    ax_pressure.set_title("Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure loss metrics found.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 3),
            frameon=False,
        )
    fig.suptitle("Pythia-14M Short Pressure Pretraining Checks", y=0.995)
    fig.text(
        0.5,
        0.945,
        f"n={len(series)} runs; AdamW baseline is shown only through the shared short-run token horizon",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_clipping_comparison(series: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    all_losses: list[float] = []
    total_points = 0
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        loss = [float(row["validation_loss"]) for row in rows]
        total_points += len(rows)
        all_losses.extend(loss)
        ax.plot(
            sparsity,
            loss,
            marker="o",
            markersize=_marker_size(item["label"], 3.5),
            linewidth=_line_width(item["label"]),
            color=colors[item["label"]],
            label=item["label"],
        )

    ax.set_title("Post-hoc MLP Activation Clipping Frontiers")
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False, fontsize=8)

    if all_losses and min(all_losses) > 0.0:
        loss_span = max(all_losses) - min(all_losses)
        margin = max(loss_span * 0.15, 1e-4)
        ax.set_ylim(min(all_losses) - margin, max(all_losses) + margin)
        axis_note = "validation-loss axis zoomed"
    else:
        axis_note = "validation-loss axis starts at zero"

    ax.text(
        0.99,
        0.02,
        f"n={len(series)} runs, {total_points} sweep points; {axis_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_gradient_diagnostics(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.5))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_task = fig.add_subplot(grid[0, 0])
    ax_pressure = fig.add_subplot(grid[0, 1])
    ax_weighted = fig.add_subplot(grid[1, 0])
    ax_ratio = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    total_events = 0

    for item in series:
        label = item["label"]
        train_events = item["train_events"]
        total_events += len(train_events)
        tokens = _tokens_millions(train_events)
        task_norms = [
            float(event.get("pressure/task_gradient_norm", event.get("grad_norm")))
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        task_tokens = [
            float(event["tokens_seen"]) / 1_000_000.0
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        if task_norms:
            ax_task.plot(
                task_tokens,
                task_norms,
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        pressure_events = [
            event
            for event in train_events
            if event.get("pressure/pressure_gradient_norm") is not None
            and event.get("pressure/task_gradient_norm") is not None
        ]
        if not pressure_events:
            continue
        pressure_tokens = [float(event["tokens_seen"]) / 1_000_000.0 for event in pressure_events]
        pressure_norms = [float(event["pressure/pressure_gradient_norm"]) for event in pressure_events]
        weighted_pressure_norms = [
            float(event.get("pressure_weight", 1.0)) * float(event["pressure/pressure_gradient_norm"])
            for event in pressure_events
        ]
        weighted_ratios = [
            (
                float(event.get("pressure_weight", 1.0))
                * float(event["pressure/pressure_gradient_norm"])
                / (float(event["pressure/task_gradient_norm"]) + 1e-12)
            )
            for event in pressure_events
        ]

        ax_pressure.plot(
            pressure_tokens,
            pressure_norms,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )
        ax_weighted.plot(
            pressure_tokens,
            weighted_pressure_norms,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )
        ax_ratio.plot(
            pressure_tokens,
            weighted_ratios,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax_task.set_title("Task Gradient Norm")
    ax_task.set_xlabel("Tokens seen (millions)")
    ax_task.set_ylabel("L2 norm")
    ax_task.set_ylim(bottom=0.0)

    ax_pressure.set_title("Pressure Gradient Norm")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Raw pressure L2 norm")
    ax_pressure.set_ylim(bottom=0.0)
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_weighted.set_title("Weighted Pressure Gradient Norm")
    ax_weighted.set_xlabel("Tokens seen (millions)")
    ax_weighted.set_ylabel("weight * pressure grad norm")
    ax_weighted.set_ylim(bottom=0.0)
    if not ax_weighted.lines:
        ax_weighted.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_ratio.set_title("Weighted Pressure / Task Gradient")
    ax_ratio.set_xlabel("Tokens seen (millions)")
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(bottom=0.0)
    if not ax_ratio.lines:
        ax_ratio.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    handles, labels = ax_task.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass Selected Methods Gradient Diagnostics", y=0.992)
    fig.text(
        0.5,
        0.948,
        f"n={len(series)} runs; {total_events} train log events; pressure panels omit AdamW where no pressure is applied",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.89, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_rms_clipping_frontiers(series: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    total_points = sum(len(item.get("rows", [])) for item in series)
    all_losses: list[float] = []
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        label = str(item["label"])
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(label, 3.4),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    _zoom_loss_axis(ax, all_losses)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass RMS-normalized Post-hoc Clipping Frontiers", y=0.975)
    fig.text(
        0.5,
        0.935,
        (
            f"n={len(series)} runs, {total_points} sweep points; "
            "threshold = multiplier * RMS(A) per captured tensor/pass; validation-loss axis zoomed"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.83, bottom=0.25, left=0.11, right=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_pressure_dominance(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_task = fig.add_subplot(grid[0, 1])
    ax_ratio = fig.add_subplot(grid[1, 0])
    ax_phase = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    pressure_method_count = 0

    for item in series:
        label = str(item["label"])
        color = colors[label]
        train_events = item["train_events"]
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        task_events = [
            event
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        pressure_events = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None
            and event.get("pressure/task_gradient_norm") is not None
            and event.get("pressure/pressure_gradient_norm") is not None
            and event.get("pressure_weight") is not None
        ]

        if near_zero_events:
            near_zero_tokens = _tokens_millions(near_zero_events)
            near_zero_values = [
                100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events
            ]
            ax_near_zero.plot(
                near_zero_tokens,
                near_zero_values,
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
            peak_index = max(range(len(near_zero_events)), key=lambda index: near_zero_values[index])
            ax_near_zero.scatter(
                [near_zero_tokens[peak_index]],
                [near_zero_values[peak_index]],
                marker="*",
                s=_scatter_size(label, 70.0),
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )

        if task_events:
            ax_task.plot(
                _tokens_millions(task_events),
                [
                    float(event.get("pressure/task_gradient_norm", event.get("grad_norm")))
                    for event in task_events
                ],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        if not pressure_events:
            continue
        pressure_method_count += 1
        ratio_tokens = _tokens_millions(pressure_events)
        ratio_values = [
            100.0
            * float(event["pressure_weight"])
            * float(event["pressure/pressure_gradient_norm"])
            / (float(event["pressure/task_gradient_norm"]) + 1e-12)
            for event in pressure_events
        ]
        near_zero_values = [
            100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in pressure_events
        ]
        ax_ratio.plot(
            ratio_tokens,
            ratio_values,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        ax_phase.plot(
            ratio_values,
            near_zero_values,
            marker="o",
            markersize=_marker_size(label, 1.7),
            linewidth=_line_width(label),
            color=color,
            alpha=0.92,
            label=label,
        )
        peak_index = max(range(len(near_zero_values)), key=lambda index: near_zero_values[index])
        ax_phase.scatter(
            [ratio_values[peak_index]],
            [near_zero_values[peak_index]],
            marker="*",
            s=_scatter_size(label, 78.0),
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        ax_phase.scatter(
            [ratio_values[-1]],
            [near_zero_values[-1]],
            marker="X",
            s=_scatter_size(label, 48.0),
            color=color,
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_task.set_title("Task Gradient Norm")
    ax_task.set_xlabel("Tokens seen (millions)")
    ax_task.set_ylabel("L2 norm")
    ax_task.set_ylim(bottom=0.0)

    ax_ratio.set_title("Weighted Pressure / Task Gradient")
    ax_ratio.set_xlabel("Tokens seen (millions)")
    ax_ratio.set_ylabel("ratio (%)")
    ax_ratio.set_ylim(bottom=0.0)
    if not ax_ratio.lines:
        ax_ratio.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_phase.set_title("Near-zero Mass vs Pressure Ratio")
    ax_phase.set_xlabel("weighted pressure/task gradient (%)")
    ax_phase.set_ylabel("|activation| <= 0.01 (%)")
    ax_phase.set_ylim(bottom=0.0)
    ax_phase.set_xlim(left=0.0)
    if not ax_phase.lines:
        ax_phase.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass Pressure Dominance Timing Diagnostic", y=0.992)
    fig.text(
        0.5,
        0.948,
        (
            f"n={len(series)} runs; {pressure_method_count} pressure methods; "
            "stars mark peak near-zero mass, X marks final pressure log event"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.89, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_sweep_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    plottable = [
        row
        for row in rows
        if _finite(row.get("validation_loss"))
        and _finite(row.get("near_zero_k01"))
        and _finite(row.get("near_zero_k03"))
    ]
    if not plottable:
        raise ValueError("No fixed-step sweep rows with validation loss and activation metrics were found.")

    fig = plt.figure(figsize=(7.4, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)
    ax_k01 = fig.add_subplot(grid[0, 0])
    ax_k03 = fig.add_subplot(grid[1, 0])

    roles = _ordered_roles(plottable)
    colors = _series_colors([FIXED_STEP_ROLE_LABELS.get(role, role) for role in roles])
    baseline = next((row for row in plottable if row["role"] == "adamw"), None)

    for role in roles:
        role_rows = [row for row in plottable if row["role"] == role]
        label = FIXED_STEP_ROLE_LABELS.get(role, role)
        color = colors[label]
        marker = FIXED_STEP_ROLE_MARKERS.get(role, "o")
        ax_k01.scatter(
            [100.0 * float(row["near_zero_k01"]) for row in role_rows],
            [float(row["validation_loss"]) for row in role_rows],
            marker=marker,
            s=_scatter_size(label, 42),
            color=color,
            label=label,
            alpha=0.9,
        )
        ax_k03.scatter(
            [100.0 * float(row["near_zero_k03"]) for row in role_rows],
            [float(row["validation_loss"]) for row in role_rows],
            marker=marker,
            s=_scatter_size(label, 42),
            color=color,
            label=label,
            alpha=0.9,
        )

    for row in _fixed_step_annotation_rows(plottable):
        ax_k01.annotate(
            row["short_label"],
            (100.0 * float(row["near_zero_k01"]), float(row["validation_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
        ax_k03.annotate(
            row["short_label"],
            (100.0 * float(row["near_zero_k03"]), float(row["validation_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    for ax, threshold in [(ax_k01, 0.01), (ax_k03, 0.03)]:
        ax.set_xlabel(f"MLP hidden mass with |activation| <= {threshold:g} (%)")
        ax.set_ylabel("Final validation loss")
        if baseline is not None:
            ax.axhline(
                float(baseline["validation_loss"]),
                color=ADAMW_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
            )
        _zoom_loss_axis(ax, [float(row["validation_loss"]) for row in plottable])

    ax_k01.set_title("Fixed-step Tradeoff at Threshold 0.01")
    ax_k03.set_title("Fixed-step Tradeoff at Threshold 0.03")
    handles, labels = ax_k01.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    tokens = sorted({int(row["tokens_seen"]) for row in plottable if row.get("tokens_seen")})
    token_note = f"{tokens[0]:,} tokens/run" if len(tokens) == 1 else "fixed token budget"
    fig.suptitle("Pythia-14M MiniPile Fixed-step Pressure Screen", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(plottable)} runs; {token_note}; validation-loss axes zoomed",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.17)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_learning_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Representative Fixed-step Learning Curves",
    subtitle: str | None = None,
    use_short_labels: bool = False,
    legend_ncol: int = 2,
) -> None:
    series = []
    for row in rows:
        events_path = Path(row["run_dir"]) / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if train_events:
            series.append({**row, "train_events": train_events, "validation_events": validation_events})
    if not series:
        raise ValueError("No fixed-step sweep learning curves were found.")

    fig = plt.figure(figsize=(7.6, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    labels_for_colors = [_plot_label(item, use_short_labels=use_short_labels) for item in series]
    colors = _series_colors(labels_for_colors)

    for item in series:
        label = _plot_label(item, use_short_labels=use_short_labels)
        color = colors[label]
        train_events = item["train_events"]
        validation_events = item["validation_events"]
        ax_train.plot(
            _tokens_millions(train_events),
            [event["train_loss"] for event in train_events],
            marker="o",
            markersize=_marker_size(label, 2.0),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        if validation_events:
            ax_val.plot(
                _tokens_millions(validation_events),
                [event["validation_loss"] for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.4),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                _tokens_millions(pressure_events),
                [event["pressure_loss"] for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen (millions)")
    ax_train.set_ylabel("Task loss")
    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen (millions)")
    ax_val.set_ylabel("Loss")
    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_pressure.set_title("Auxiliary Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-loss series selected.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_ncol,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.99)
    subtitle = subtitle or f"n={len(series)} selected runs: AdamW plus best validation-loss run per pressure family"
    fig.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_high_pressure_weight_norms(rows: list[dict[str, Any]], output_path: Path) -> None:
    series = []
    for row in rows:
        events_path = Path(row["run_dir"]) / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        if not train_events:
            continue
        final_mlp_weight_norm = row.get("mlp_weight_norm_final")
        if final_mlp_weight_norm is None:
            final_mlp_weight_norm = _final_mlp_weight_norm_from_checkpoint(Path(row["run_dir"]))
        series.append({**row, "train_events": train_events, "final_mlp_weight_norm": final_mlp_weight_norm})
    if not series:
        raise ValueError("No high-pressure OR/L1 weight-norm series were found.")

    fig = plt.figure(figsize=(8.2, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_global_tokens = fig.add_subplot(grid[0, 1])
    ax_global_near_zero = fig.add_subplot(grid[1, 0])
    ax_mlp_near_zero = fig.add_subplot(grid[1, 1])

    labels_for_colors = [_plot_label(item, use_short_labels=True) for item in series]
    colors = _series_colors(labels_for_colors)
    mlp_time_series_count = 0
    mlp_final_count = 0
    global_fit_x: list[float] = []
    global_fit_y: list[float] = []
    mlp_fit_x: list[float] = []
    mlp_fit_y: list[float] = []

    for item in series:
        label = _plot_label(item, use_short_labels=True)
        color = colors[label]
        marker = FIXED_STEP_ROLE_MARKERS.get(str(item.get("role")), "o")
        train_events = item["train_events"]
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        weight_events = [event for event in train_events if event.get("weight_norm") is not None]
        paired_global = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("weight_norm") is not None
        ]
        paired_mlp = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("mlp_weight_norm") is not None
        ]

        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if weight_events:
            ax_global_tokens.plot(
                _tokens_millions(weight_events),
                [float(event["weight_norm"]) for event in weight_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if paired_global:
            final_event = paired_global[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["weight_norm"])
            global_fit_x.append(final_x)
            global_fit_y.append(final_y)
            ax_global_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 28.0),
                color=color,
                alpha=0.9,
                linewidths=0.0,
                zorder=2,
            )
        if paired_mlp:
            mlp_time_series_count += 1
            xs = [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in paired_mlp]
            ys = [float(event["mlp_weight_norm"]) for event in paired_mlp]
            mlp_fit_x.extend(xs)
            mlp_fit_y.extend(ys)
            ax_mlp_near_zero.scatter(
                xs,
                ys,
                marker=marker,
                s=_scatter_size(label, 12.0),
                color=color,
                alpha=0.72,
                linewidths=0.0,
                zorder=2,
            )
        elif item.get("final_mlp_weight_norm") is not None:
            final_event = paired_global[-1] if paired_global else None
            if final_event is not None:
                final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
                final_y = float(item["final_mlp_weight_norm"])
                mlp_fit_x.append(final_x)
                mlp_fit_y.append(final_y)
                mlp_final_count += 1
                ax_mlp_near_zero.scatter(
                    [final_x],
                    [final_y],
                    marker=marker,
                    s=_scatter_size(label, 28.0),
                    color=color,
                    alpha=0.9,
                    linewidths=0.0,
                    zorder=2,
                )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")

    ax_global_tokens.set_title("Global Weight Norm")
    ax_global_tokens.set_xlabel("Tokens seen (millions)")
    ax_global_tokens.set_ylabel("L2 norm")
    ax_global_tokens.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax_global_near_zero.set_title("Final Global Norm vs Near-zero Mass")
    ax_global_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_global_near_zero.set_ylabel("Global L2 norm")
    ax_global_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_global_near_zero.collections:
        ax_global_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_global_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_global_near_zero, global_fit_x, global_fit_y, loc="lower right")

    mlp_title_prefix = "MLP Weight Norm"
    if mlp_time_series_count == 0 and mlp_final_count > 0:
        mlp_title_prefix = "Final MLP Weight Norm"
    if mlp_title_prefix == "Final MLP Weight Norm":
        ax_mlp_near_zero.set_title("Final MLP Norm vs Near-zero Mass")
    else:
        ax_mlp_near_zero.set_title(f"{mlp_title_prefix} vs Near-zero Mass")
    ax_mlp_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_mlp_near_zero.set_ylabel("MLP weight L2 norm")
    ax_mlp_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if not ax_mlp_near_zero.lines and not ax_mlp_near_zero.collections:
        ax_mlp_near_zero.text(0.5, 0.5, "No MLP weight norms found.", ha="center", va="center")
    elif mlp_time_series_count == 0 and mlp_final_count > 0:
        ax_mlp_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.\nFuture runs log MLP norm per step.",
            transform=ax_mlp_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_mlp_near_zero, mlp_fit_x, mlp_fit_y, loc="upper right")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=7,
        )

    tokens = sorted(
        {
            int(event["tokens_seen"])
            for item in series
            for event in item["train_events"]
            if event.get("tokens_seen")
        }
    )
    token_note = f"up to {tokens[-1]:,} tokens/run" if tokens else "fixed token budget"
    fig.suptitle("High-pressure OR and L1 Weight Norm Diagnostics", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(series)} runs: AdamW, OR configs 40-44, and L1N/OL1 configs 45-48; {token_note}",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_activation_histogram_grid(payload: dict[str, Any], output_path: Path) -> None:
    methods = payload.get("methods", [])
    edges = [float(value) for value in payload.get("bin_edges", [])]
    if not methods or len(edges) < 2:
        raise ValueError("Activation histogram payload has no plottable methods or bin edges.")

    layer_names = [layer["name"] for layer in methods[0].get("layers", [])]
    if not layer_names:
        raise ValueError("Activation histogram payload has no layer histograms.")
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    labels = [str(method["label"]) for method in methods]
    colors = _series_colors(labels)

    positive_values: list[float] = []
    probability_by_method_layer: list[list[list[float]]] = []
    overflow_notes: list[str] = []
    for method in methods:
        method_layers: list[list[float]] = []
        method_overflow = 0.0
        method_total = 0.0
        layers_by_name = {layer["name"]: layer for layer in method.get("layers", [])}
        for layer_name in layer_names:
            layer = layers_by_name[layer_name]
            total = float(layer.get("total") or sum(layer.get("counts", [])) or 1.0)
            counts = [float(value) for value in layer["counts"]]
            probabilities = [count / total for count in counts]
            positive_values.extend(value for value in probabilities if value > 0.0)
            method_layers.append(probabilities)
            method_overflow += float(layer.get("underflow", 0)) + float(layer.get("overflow", 0))
            method_total += total
        probability_by_method_layer.append(method_layers)
        overflow_fraction = method_overflow / method_total if method_total else 0.0
        if overflow_fraction > 0.001:
            overflow_notes.append(f"{method['label']}: {100.0 * overflow_fraction:.2f}% outside range")

    rows = len(methods)
    cols = len(layer_names)
    fig_width = max(12.0, 2.05 * cols)
    fig_height = max(14.0, 1.34 * rows + 2.6)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    y_min = max(min(positive_values) * 0.6, 1e-9) if positive_values else 1e-9
    y_max = max(positive_values) * 2.0 if positive_values else 1.0
    for row_index, method in enumerate(methods):
        label = str(method["label"])
        color = colors[label]
        for col_index, layer_name in enumerate(layer_names):
            ax = axes[row_index][col_index]
            probabilities = probability_by_method_layer[row_index][col_index]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in probabilities]
            ax.fill_between(
                centers,
                y_min,
                y_values,
                step="mid",
                color=color,
                alpha=0.26,
                linewidth=0,
            )
            ax.step(centers, y_values, where="mid", color=color, linewidth=0.75, alpha=0.95)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(min(edges), 1.0)
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.07, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.45, alpha=0.5)
            if row_index == 0:
                ax.set_title(layer_name.replace("mlp_hiddens.layer_", "MLP "), fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{label}\nprobability", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    eval_tokens = int(payload.get("validation_tokens") or 0)
    eval_sequences = int(payload.get("validation_sequences") or 0)
    overflow_note = ""
    if overflow_notes:
        overflow_note = "; outside plotted range: " + ", ".join(overflow_notes[:3])
        if len(overflow_notes) > 3:
            overflow_note += ", ..."
    fig.suptitle("Selected Method MLP Activation Distributions", y=0.997, fontsize=14)
    fig.text(
        0.5,
        0.982,
        (
            f"n={len(methods)} checkpoints; {eval_sequences:,} validation blocks; "
            f"{eval_tokens:,} validation tokens; y-axis is log probability per bin"
            f"{overflow_note}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Activation value", y=0.035, fontsize=10)
    fig.supylabel("Probability per bin (log scale)", x=0.006, fontsize=10)
    fig.text(
        0.5,
        0.02,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.955, bottom=0.065, hspace=0.18, wspace=0.08)
    fig.savefig(output_path)
    plt.close(fig)


def _trimmed_decimal_tick(value: float, _position: int) -> str:
    label = f"{value:.2f}".rstrip("0")
    if label.endswith("."):
        label += "0"
    return label


def _plot_fixed_step_clipping_frontiers(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Post-hoc Clipping Frontiers After Fixed-step Pretraining",
    subtitle: str | None = None,
    use_short_labels: bool = False,
    legend_ncol: int | None = None,
) -> None:
    if not series:
        raise ValueError("No fixed-step clipping frontier series were found.")

    figsize = (7.0, 5.4) if legend_ncol is not None else (7.0, 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    labels_for_colors = [_plot_label(item, use_short_labels=use_short_labels) for item in series]
    colors = _series_colors(labels_for_colors)
    all_losses: list[float] = []
    total_points = 0
    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        total_points += len(rows)
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(_plot_label(item, use_short_labels=use_short_labels), 3.2),
            linewidth=_line_width(_plot_label(item, use_short_labels=use_short_labels)),
            color=colors[_plot_label(item, use_short_labels=use_short_labels)],
            label=_plot_label(item, use_short_labels=use_short_labels),
        )

    ax.set_title(title)
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    if legend_ncol is None:
        ax.legend(frameon=False, fontsize=8)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=legend_ncol,
                frameon=False,
                fontsize=7,
            )
    _zoom_loss_axis(ax, all_losses)
    ax.text(
        0.99,
        0.02,
        subtitle or f"n={len(series)} selected runs, {total_points} sweep points; validation-loss axis zoomed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    if legend_ncol is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(bottom=0.34)
    fig.savefig(output_path)
    plt.close(fig)


def _short_run_token_limit(series: list[dict[str, Any]]) -> int | None:
    pressure_maxima = [
        max((int(event["tokens_seen"]) for event in item["train_events"]), default=0)
        for item in series
        if item["label"] != "AdamW baseline"
    ]
    pressure_maxima = [value for value in pressure_maxima if value > 0]
    if pressure_maxima:
        return max(pressure_maxima)
    all_maxima = [max((int(event["tokens_seen"]) for event in item["train_events"]), default=0) for item in series]
    all_maxima = [value for value in all_maxima if value > 0]
    return max(all_maxima) if all_maxima else None


def _events_up_to(events: list[dict[str, Any]], token_limit: int | None) -> list[dict[str, Any]]:
    if token_limit is None:
        return events
    return [event for event in events if int(event.get("tokens_seen", 0)) <= token_limit]


def _series_colors(labels: list[str]) -> dict[str, str]:
    unique_labels = list(dict.fromkeys(labels))
    colors: dict[str, str] = {}
    color_index = 0
    for label in unique_labels:
        if _is_adamw_label(label):
            colors[label] = ADAMW_COLOR
            continue
        colors[label] = COLORBLIND_SAFE_COLORS[color_index % len(COLORBLIND_SAFE_COLORS)]
        color_index += 1
    return colors


def _is_adamw_label(label: str) -> bool:
    return label.lower().startswith("adamw")


def _line_width(label: str) -> float:
    return ADAMW_LINEWIDTH if _is_adamw_label(label) else DEFAULT_SERIES_LINEWIDTH


def _marker_size(label: str, default: float) -> float:
    return default * ADAMW_MARKER_SCALE if _is_adamw_label(label) else default


def _scatter_size(label: str, default: float) -> float:
    return default * ADAMW_MARKER_SCALE if _is_adamw_label(label) else default


def _ordered_roles(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["adamw", "ricker_naive", "orthogonal_ricker", "l1_naive", "orthogonal_l1"]
    present = {str(row["role"]) for row in rows}
    ordered = [role for role in preferred if role in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _fixed_step_label(role: str, pressure: dict[str, Any]) -> str:
    role_label = FIXED_STEP_ROLE_LABELS.get(role, role)
    if role == "adamw":
        return role_label
    weight = pressure.get("weight")
    if "ricker" in role:
        return (
            f"{role_label} "
            f"w={_compact_number(weight)}, c={_compact_number(pressure.get('ricker_c'))}, "
            f"s={_compact_number(pressure.get('ricker_sigma'))}"
        )
    if "l1" in role:
        return f"{role_label} w={_compact_number(weight)}"
    return role_label


def _fixed_step_short_label(role: str, pressure: dict[str, Any]) -> str:
    if role == "adamw":
        return "AdamW"
    prefix = {
        "ricker_naive": "RN",
        "orthogonal_ricker": "OR",
        "l1_naive": "L1N",
        "orthogonal_l1": "OL1",
    }.get(role, role)
    weight = _compact_number(pressure.get("weight"))
    if "ricker" in role:
        return f"{prefix} w{weight} c{_compact_number(pressure.get('ricker_c'))} s{_compact_number(pressure.get('ricker_sigma'))}"
    return f"{prefix} w{weight}"


def _compact_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):g}"


def _config_index(config_id: str) -> int:
    first = config_id.split("-", 1)[0]
    try:
        return int(first)
    except ValueError:
        return 9999


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))


def _fixed_step_annotation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected[baseline["config_id"]] = baseline
    for role in _ordered_roles(rows):
        role_rows = [row for row in rows if row["role"] == role and _finite(row.get("validation_loss"))]
        if role == "adamw" or not role_rows:
            continue
        sparse = max(role_rows, key=lambda row: float(row.get("near_zero_k03") or 0.0))
        selected[sparse["config_id"]] = sparse
    return sorted(selected.values(), key=lambda row: row["config_index"])


def _select_representative_fixed_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    for role in ("ricker_naive", "orthogonal_ricker", "l1_naive", "orthogonal_l1"):
        role_rows = [row for row in rows if row["role"] == role and _finite(row.get("validation_loss"))]
        if role_rows:
            selected.append(min(role_rows, key=lambda row: float(row["validation_loss"])))
    return selected


def _select_fixed_step_rows_for_role(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    selected.extend(sorted((row for row in rows if row["role"] == role), key=lambda row: row["config_index"]))
    return selected


def _select_fixed_step_rows_by_config_indices(
    rows: list[dict[str, Any]],
    config_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_index = {int(row["config_index"]): row for row in rows}
    return [by_index[index] for index in config_indices if index in by_index]


def _select_representative_clipping_series(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_rows = [item["training"] for item in series]
    selected_configs = {row["config_id"] for row in _select_representative_fixed_step_rows(training_rows)}
    return [item for item in series if item["config_id"] in selected_configs]


def _select_fixed_step_clipping_for_role(series: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((item for item in series if item["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    selected.extend(sorted((item for item in series if item["role"] == role), key=lambda item: item["training"]["config_index"]))
    return selected


def _select_fixed_step_clipping_by_config_indices(
    series: list[dict[str, Any]],
    config_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_index = {int(item["training"]["config_index"]): item for item in series}
    return [by_index[index] for index in config_indices if index in by_index]


def _tokens_millions(events: list[dict[str, Any]]) -> list[float]:
    return [float(event["tokens_seen"]) / 1_000_000.0 for event in events]


def _plot_label(item: dict[str, Any], *, use_short_labels: bool) -> str:
    return str(item.get("short_label") if use_short_labels else item.get("label"))


def _add_linear_fit_annotation(ax: Any, x_values: list[float], y_values: list[float], *, loc: str) -> None:
    points = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=False)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(points) < 2:
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    if x_var <= 0.0:
        return
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / x_var
    intercept = y_mean - slope * x_mean
    residual_sum = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    total_sum = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0

    x_line = [min(xs), max(xs)]
    y_line = [slope * x + intercept for x in x_line]
    ax.plot(
        x_line,
        y_line,
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="_nolegend_",
        zorder=1,
    )

    positions = {
        "upper right": (0.98, 0.97, "right", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
    }
    x_pos, y_pos, ha, va = positions.get(loc, positions["upper right"])
    ax.text(
        x_pos,
        y_pos,
        f"slope={slope:.3g}/pp\nR2={r2:.2f}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def _zoom_loss_axis(ax: Any, values: list[float]) -> None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return
    low = min(finite_values)
    high = max(finite_values)
    span = high - low
    margin = max(span * 0.15, 1e-4)
    ax.set_ylim(low - margin, high + margin)


def _plot_clipping_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    finite_rows = [
        row
        for row in rows
        if row.get("achieved_sparsity") is not None
        and row.get("validation_loss") is not None
        and math.isfinite(float(row["achieved_sparsity"]))
        and math.isfinite(float(row["validation_loss"]))
    ]
    if not finite_rows:
        raise ValueError("No finite clipping frontier points to plot.")

    finite_rows = sorted(finite_rows, key=lambda row: float(row["achieved_sparsity"]))
    sparsity = [100.0 * float(row["achieved_sparsity"]) for row in finite_rows]
    loss = [float(row["validation_loss"]) for row in finite_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(sparsity, loss, marker="o", linewidth=1.5, markersize=4.5)
    ax.set_title("Post-hoc Activation Clipping Frontier")
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")

    for index, (point_x, point_y, row) in enumerate(zip(sparsity, loss, finite_rows, strict=True)):
        label = _clipping_label(row)
        ax.annotate(
            label,
            (point_x, point_y),
            textcoords="offset points",
            xytext=_clipping_label_offset(index, len(finite_rows)),
            fontsize=7,
        )

    loss_span = max(loss) - min(loss)
    if min(loss) > 0.0:
        margin = max(loss_span * 0.15, 1e-4)
        ax.set_ylim(min(loss) - margin, max(loss) + margin)
        axis_note = "validation-loss axis zoomed"
    else:
        axis_note = "validation-loss axis starts at zero"

    eval_tokens = sorted({int(row["validation_tokens"]) for row in finite_rows if row.get("validation_tokens")})
    token_note = f"; {eval_tokens[0]:,} validation tokens/point" if len(eval_tokens) == 1 else ""
    ax.text(
        0.99,
        0.02,
        f"n={len(finite_rows)} sweep points{token_note}; {axis_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _clipping_label(row: dict[str, Any]) -> str:
    if row.get("threshold") is not None:
        return f"t={float(row['threshold']):g}"
    if row.get("quantile") is not None:
        return f"q={float(row['quantile']):g}"
    if row.get("rms_multiplier") is not None:
        return f"r={float(row['rms_multiplier']):g}"
    return str(row.get("mode", "clip"))


def _clipping_label_offset(index: int, total: int) -> tuple[int, int]:
    if index == 0:
        return (6, -16)
    if index == total - 1:
        return (-36, 6)
    return (6, 8 + 5 * (index % 2))


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
