from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_exp.calibration import run_calibration
from paper_exp.config import ConfigError, load_config
from paper_exp.data import prepare_tokenized_data
from paper_exp.plots import generate_plots, generate_run_diagnostics
from paper_exp.run import run_baseline, run_smoke


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

    plot_run = subparsers.add_parser("plot-run", help="Generate diagnostics for one run directory.")
    plot_run.add_argument("--run-dir", required=True)
    plot_run.add_argument("--output", required=True)
    plot_run.add_argument("--png", action="store_true", help="Also save a PNG copy.")

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

        if args.command == "plot-run":
            outputs = generate_run_diagnostics(run_dir=args.run_dir, output=args.output, save_png=args.png)
            for output in outputs:
                print(f"Wrote {output}")
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


if __name__ == "__main__":
    raise SystemExit(main())
