from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_exp.activation_histograms import run_activation_histograms
from paper_exp.activation_propagation import run_activation_propagation
from paper_exp.calibration import run_calibration
from paper_exp.clipping import run_clipping_sweep
from paper_exp.config import ConfigError, load_config
from paper_exp.data import prepare_tokenized_data
from paper_exp.integrity import check_repository
from paper_exp.plots import generate_clipping_frontier, generate_plots, generate_run_diagnostics
from paper_exp.run import run_baseline, run_smoke
from paper_exp.sweeps import run_pressure_fixed_step_clipping_sweeps
from paper_exp.sweeps import run_pressure_fixed_step_sweep
from paper_exp.sweeps import write_pressure_fixed_step_configs
from paper_exp.weight_histograms import run_weight_histograms

DEFAULT_CLIPPING_THRESHOLDS = "0,0.001,0.003,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.3"
DEFAULT_RMS_CLIPPING_MULTIPLIERS = "0,0.001,0.003,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lean paper experiment harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a tiny local sanity check.")
    smoke.add_argument("--config", default="configs/01-pythia-14m-minipile-smoke.yaml")

    baseline = subparsers.add_parser("baseline", help="Run the configured baseline.")
    baseline.add_argument("--config", default="configs/02-pythia-14m-minipile-baseline.yaml")

    prepare_data = subparsers.add_parser("prepare-data", help="Download and tokenize the configured dataset.")
    prepare_data.add_argument("--config", default="configs/01-pythia-14m-minipile-smoke.yaml")

    calibrate = subparsers.add_parser("calibrate", help="Run a short model throughput calibration.")
    calibrate.add_argument("--config", default="configs/01-pythia-14m-minipile-smoke.yaml")

    pretrain = subparsers.add_parser("pretrain", help="Run a random-initialized pretraining job.")
    pretrain.add_argument("--config", default="configs/02-pythia-14m-minipile-baseline.yaml")

    plots = subparsers.add_parser("plots", help="Regenerate paper-style plots from saved results.")
    plots.add_argument("--results", default="results")
    plots.add_argument("--figures", default="figures")
    plots.add_argument("--png", action="store_true", help="Also save PNG copies.")

    check = subparsers.add_parser(
        "check",
        help="Inspect repository conventions and artifact references without writing files.",
    )
    check.add_argument("--root", default=".", help="Repository root to inspect.")
    check.add_argument(
        "--verbose",
        action="store_true",
        help="Also print informational findings for completed runs.",
    )
    check.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero status for warnings as well as errors.",
    )

    plot_run = subparsers.add_parser("plot-run", help="Generate diagnostics for one run directory.")
    plot_run.add_argument("--run-dir", required=True)
    plot_run.add_argument("--output", required=True)
    plot_run.add_argument("--png", action="store_true", help="Also save a PNG copy.")

    plot_clipping = subparsers.add_parser(
        "plot-clipping-frontier",
        help="Generate a clipping frontier plot from a clipping sweep run.",
    )
    plot_clipping.add_argument("--run-dir", required=True)
    plot_clipping.add_argument("--output", required=True)
    plot_clipping.add_argument("--png", action="store_true", help="Also save a PNG copy.")

    clip_sweep = subparsers.add_parser("clip-sweep", help="Run a post-hoc activation clipping frontier.")
    clip_sweep.add_argument("--run-dir", required=True)
    clip_sweep.add_argument("--thresholds", default=DEFAULT_CLIPPING_THRESHOLDS)
    clip_sweep.add_argument("--quantiles", default="")
    clip_sweep.add_argument("--rms-multipliers", default="")
    clip_sweep.add_argument("--sites", default="", help="Comma-separated activation sites to clip for this sweep.")
    clip_sweep.add_argument("--experiment-suffix", default="", help="Optional suffix for the clipping sweep result folder.")
    clip_sweep.add_argument("--eval-batches", type=int, default=None)
    clip_sweep.add_argument("--seed", type=int, default=0)

    write_sweep = subparsers.add_parser(
        "write-pressure-sweep-configs",
        help="Write the fixed-step pressure screening configs.",
    )
    write_sweep.add_argument("--configs-dir", default="configs")

    run_sweep = subparsers.add_parser(
        "run-pressure-sweep",
        help="Run the fixed-step pressure screening matrix.",
    )
    run_sweep.add_argument("--configs-dir", default="configs")
    run_sweep.add_argument("--start-at", default="")
    run_sweep.add_argument("--stop-after", type=int, default=None)

    run_sweep_clipping = subparsers.add_parser(
        "run-pressure-sweep-clipping",
        help="Run post-hoc clipping sweeps for completed fixed-step pressure screening runs.",
    )
    run_sweep_clipping.add_argument("--configs-dir", default="configs")
    run_sweep_clipping.add_argument("--thresholds", default=DEFAULT_CLIPPING_THRESHOLDS)
    run_sweep_clipping.add_argument("--quantiles", default="")
    run_sweep_clipping.add_argument("--rms-multipliers", default="")
    run_sweep_clipping.add_argument("--eval-batches", type=int, default=8)
    run_sweep_clipping.add_argument("--seed", type=int, default=0)
    run_sweep_clipping.add_argument("--start-at", default="")
    run_sweep_clipping.add_argument("--stop-after", type=int, default=None)

    activation_histograms = subparsers.add_parser(
        "activation-histograms",
        help="Measure validation activation histograms for selected checkpoints.",
    )
    activation_histograms.add_argument(
        "--config",
        default="configs/49-pythia-14m-pressure-fixed-2048-selected-activation-histograms.yaml",
    )

    activation_propagation = subparsers.add_parser(
        "activation-propagation",
        help="Measure exact-zero propagation through selected GPT-NeoX checkpoints.",
    )
    activation_propagation.add_argument(
        "--config",
        default="configs/102-pythia-14m-minipile-post-layernorm-relu-activation-propagation.yaml",
    )

    weight_histograms = subparsers.add_parser(
        "weight-histograms",
        help="Measure checkpoint weight histograms for selected runs.",
    )
    weight_histograms.add_argument(
        "--config",
        default="configs/61-pythia-14m-minipile-full-pass-high-pressure-weight-histograms.yaml",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = _command_string(argv)

    try:
        if args.command == "smoke":
            config = load_config(args.config, allow_todos=True)
            run_dir = run_smoke(config, config_path=args.config, command=command)
            print(f"Smoke run written to {run_dir}")
            return 0

        if args.command == "baseline":
            config = load_config(args.config, allow_todos=False)
            run_dir = run_baseline(config, config_path=args.config, command=command)
            print(f"Baseline run written to {run_dir}")
            return 0

        if args.command == "prepare-data":
            config = load_config(args.config, allow_todos=False)
            run_dir = prepare_tokenized_data(config, config_path=args.config, command=command)
            print(f"Prepared tokenized data; run written to {run_dir}")
            return 0

        if args.command == "calibrate":
            config = load_config(args.config, allow_todos=False)
            run_dir = run_calibration(config, config_path=args.config, command=command)
            print(f"Calibration run written to {run_dir}")
            return 0

        if args.command == "pretrain":
            config = load_config(args.config, allow_todos=False)
            run_dir = run_calibration(config, config_path=args.config, command=command, mode="pretrain")
            print(f"Pretraining run written to {run_dir}")
            return 0

        if args.command == "plots":
            outputs = generate_plots(results_dir=args.results, figures_dir=args.figures, save_png=args.png)
            for output in outputs:
                print(f"Wrote {output}")
            return 0

        if args.command == "check":
            findings = check_repository(args.root)
            visible_findings = (
                findings
                if args.verbose
                else [finding for finding in findings if finding.severity != "info"]
            )
            for finding in visible_findings:
                print(
                    f"{finding.severity.upper()} [{finding.code}] "
                    f"{finding.path}: {finding.message}"
                )

            counts = {
                severity: sum(finding.severity == severity for finding in findings)
                for severity in ("error", "warning", "info")
            }
            print(
                "Integrity summary: "
                f"{counts['error']} error(s), {counts['warning']} warning(s), "
                f"{counts['info']} informational finding(s)."
            )
            if counts["error"] or (args.strict and counts["warning"]):
                return 1
            return 0

        if args.command == "plot-run":
            outputs = generate_run_diagnostics(run_dir=args.run_dir, output=args.output, save_png=args.png)
            for output in outputs:
                print(f"Wrote {output}")
            return 0

        if args.command == "plot-clipping-frontier":
            outputs = generate_clipping_frontier(run_dir=args.run_dir, output=args.output, save_png=args.png)
            for output in outputs:
                print(f"Wrote {output}")
            return 0

        if args.command == "clip-sweep":
            run_dir = run_clipping_sweep(
                checkpoint_run_dir=args.run_dir,
                command=command,
                thresholds=_parse_float_list(args.thresholds),
                quantiles=_parse_float_list(args.quantiles),
                rms_multipliers=_parse_float_list(args.rms_multipliers),
                sites=_parse_str_list(args.sites) or None,
                experiment_suffix=args.experiment_suffix or None,
                eval_batches=args.eval_batches,
                seed=args.seed,
            )
            print(f"Clipping sweep written to {run_dir}")
            return 0

        if args.command == "write-pressure-sweep-configs":
            outputs = write_pressure_fixed_step_configs(args.configs_dir)
            for output in outputs:
                print(f"Wrote {output}")
            return 0

        if args.command == "run-pressure-sweep":
            outputs = run_pressure_fixed_step_sweep(
                configs_dir=args.configs_dir,
                command=command,
                start_at=args.start_at or None,
                stop_after=args.stop_after,
            )
            for output in outputs:
                print(f"Completed {output}")
            return 0

        if args.command == "run-pressure-sweep-clipping":
            outputs = run_pressure_fixed_step_clipping_sweeps(
                configs_dir=args.configs_dir,
                command=command,
                thresholds=_parse_float_list(args.thresholds),
                quantiles=_parse_float_list(args.quantiles),
                rms_multipliers=_parse_float_list(args.rms_multipliers),
                eval_batches=args.eval_batches,
                seed=args.seed,
                start_at=args.start_at or None,
                stop_after=args.stop_after,
            )
            for output in outputs:
                print(f"Completed {output}")
            return 0

        if args.command == "activation-histograms":
            config = load_config(args.config, allow_todos=False)
            run_dir = run_activation_histograms(config, config_path=args.config, command=command)
            print(f"Activation histograms written to {run_dir}")
            return 0

        if args.command == "activation-propagation":
            config = load_config(args.config, allow_todos=False)
            run_dir = run_activation_propagation(config, config_path=args.config, command=command)
            print(f"Activation propagation written to {run_dir}")
            return 0

        if args.command == "weight-histograms":
            config = load_config(args.config, allow_todos=False)
            run_dir = run_weight_histograms(config, config_path=args.config, command=command)
            print(f"Weight histograms written to {run_dir}")
            return 0
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


def _command_string(argv: list[str] | None) -> str:
    if argv is None:
        return " ".join([Path(sys.executable).name, *sys.argv])
    return " ".join([Path(sys.executable).name, "-m", "paper_exp.cli", *argv])


def _parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
