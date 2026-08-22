from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

from paper_exp.config import ConfigError, load_config
from paper_exp.launch import (
    LaunchError,
    direct_launch_guard,
    require_results_output,
    require_token_cache_output,
    resolve_launch_config,
    resolve_launch_run_dir,
)


PLOT_KINDS = (
    "run",
    "clipping",
    "activation-histograms",
    "weight-histograms",
    "activation-propagation",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lean paper experiment harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a tiny local harness check.")
    smoke.add_argument("--config", required=True)

    prepare_data = subparsers.add_parser(
        "prepare-data",
        help="Download and tokenize the dataset declared by a config.",
    )
    prepare_data.add_argument("--config", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Run a configured throughput calibration.",
    )
    calibrate.add_argument("--config", required=True)

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
        help="Return nonzero for warnings as well as errors.",
    )

    clip_sweep = subparsers.add_parser(
        "clip-sweep",
        help="Run a post-hoc activation clipping frontier.",
    )
    clip_sweep.add_argument("--run-dir", required=True)
    clip_sweep.add_argument("--thresholds", default="")
    clip_sweep.add_argument("--quantiles", default="")
    clip_sweep.add_argument("--rms-multipliers", default="")
    clip_sweep.add_argument(
        "--sites",
        default="",
        help="Comma-separated activation sites to clip.",
    )
    clip_sweep.add_argument(
        "--experiment-suffix",
        default="",
        help="Optional suffix for the clipping result folder.",
    )
    clip_sweep.add_argument("--eval-batches", type=int, default=None)
    clip_sweep.add_argument(
        "--measure-zero-products",
        action="store_true",
        help=(
            "Count exact logical zero products in QKV, QK, PV, attention output, "
            "W1, and W2; include the LM head only in the model denominator."
        ),
    )
    clip_sweep.add_argument("--seed", type=int, required=True)

    activation_histograms = subparsers.add_parser(
        "activation-histograms",
        help="Measure validation activation histograms for configured checkpoints.",
    )
    activation_histograms.add_argument("--config", required=True)

    activation_propagation = subparsers.add_parser(
        "activation-propagation",
        help="Measure exact-zero propagation for configured checkpoints.",
    )
    activation_propagation.add_argument("--config", required=True)

    weight_histograms = subparsers.add_parser(
        "weight-histograms",
        help="Measure weight histograms for configured checkpoints.",
    )
    weight_histograms.add_argument("--config", required=True)

    plot = subparsers.add_parser(
        "plot",
        help="Render one explicit saved diagnostic artifact.",
    )
    plot.add_argument("--kind", choices=PLOT_KINDS, required=True)
    plot.add_argument("--run-dir", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--png", action="store_true", help="Also save a PNG copy.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = _command_string(argv)

    try:
        if args.command == "smoke":
            from paper_exp.run import run_smoke

            config = load_config(args.config, allow_todos=True)
            run_dir = run_smoke(config, config_path=args.config, command=command)
            print(f"Smoke run written to {run_dir}")
            return 0

        if args.command == "prepare-data":
            from paper_exp.data import prepare_tokenized_data

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_results_output(config, repository=repository, source=config_path)
            require_token_cache_output(config, repository=repository, source=config_path)
            with direct_launch_guard(repository=repository):
                run_dir = prepare_tokenized_data(
                    config,
                    config_path=config_path,
                    command=command,
                )
            print(f"Prepared tokenized data; run written to {run_dir}")
            return 0

        if args.command == "calibrate":
            from paper_exp.training import run_training

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_results_output(config, repository=repository, source=config_path)
            require_token_cache_output(config, repository=repository, source=config_path)
            with direct_launch_guard(repository=repository):
                run_dir = run_training(
                    config,
                    config_path=config_path,
                    command=command,
                    mode="calibrate",
                )
            print(f"Calibration run written to {run_dir}")
            return 0

        if args.command == "check":
            from paper_exp.integrity import check_repository

            findings = check_repository(args.root)
            visible = (
                findings
                if args.verbose
                else [finding for finding in findings if finding.severity != "info"]
            )
            for finding in visible:
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
            return int(bool(counts["error"] or (args.strict and counts["warning"])))

        if args.command == "clip-sweep":
            from paper_exp.diagnostics.clipping import run_clipping_sweep

            repository, source_run = resolve_launch_run_dir(args.run_dir)
            with direct_launch_guard(repository=repository):
                run_dir = run_clipping_sweep(
                    checkpoint_run_dir=source_run,
                    command=command,
                    thresholds=_parse_float_list(args.thresholds),
                    quantiles=_parse_float_list(args.quantiles),
                    rms_multipliers=_parse_float_list(args.rms_multipliers),
                    sites=_parse_str_list(args.sites) or None,
                    experiment_suffix=args.experiment_suffix or None,
                    eval_batches=args.eval_batches,
                    measure_zero_products=args.measure_zero_products,
                    seed=args.seed,
                )
            print(f"Clipping sweep written to {run_dir}")
            return 0

        if args.command == "activation-histograms":
            from paper_exp.diagnostics.activation_histograms import run_activation_histograms

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_results_output(config, repository=repository, source=config_path)
            with direct_launch_guard(repository=repository):
                run_dir = run_activation_histograms(
                    config,
                    config_path=config_path,
                    command=command,
                )
            print(f"Activation histograms written to {run_dir}")
            return 0

        if args.command == "activation-propagation":
            from paper_exp.diagnostics.propagation import run_activation_propagation

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_results_output(config, repository=repository, source=config_path)
            with direct_launch_guard(repository=repository):
                run_dir = run_activation_propagation(
                    config,
                    config_path=config_path,
                    command=command,
                )
            print(f"Activation propagation written to {run_dir}")
            return 0

        if args.command == "weight-histograms":
            from paper_exp.diagnostics.weight_histograms import run_weight_histograms

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_results_output(config, repository=repository, source=config_path)
            with direct_launch_guard(repository=repository):
                run_dir = run_weight_histograms(
                    config,
                    config_path=config_path,
                    command=command,
                )
            print(f"Weight histograms written to {run_dir}")
            return 0

        if args.command == "plot":
            from paper_exp.plots import plot_artifact

            outputs = plot_artifact(
                kind=args.kind,
                run_dir=args.run_dir,
                output=args.output,
                save_png=args.png,
            )
            for output in outputs:
                print(f"Wrote {output}")
            return 0
    except (ConfigError, LaunchError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


def _command_string(argv: list[str] | None) -> str:
    parts = (
        [Path(sys.executable).name, *sys.argv]
        if argv is None
        else [Path(sys.executable).name, "-m", "paper_exp.cli", *argv]
    )
    return shlex.join(parts)


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
