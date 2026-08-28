from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys

from paper_exp.config import ConfigError, load_config
from paper_exp.launch import (
    LaunchError,
    direct_launch_guard,
    repository_path,
    require_raw_output,
    require_token_cache_output,
    resolve_launch_config,
    resolve_launch_run_dir,
)
from paper_exp.topology import SITE_ALIAS_ORDER


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
    smoke.add_argument(
        "--worker-slot",
        action="append",
        type=_smoke_worker_slot,
        default=[],
        metavar="SLOT=CUDA_DEVICE",
        help="Explicit worker mapping; repeat exactly twice for concurrent smoke.",
    )
    smoke.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require one BF16 CUDA GPU per concurrent smoke worker.",
    )
    smoke.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Infrastructure-smoke-only opt-in to map both workers to one GPU.",
    )

    prepare_data = subparsers.add_parser(
        "prepare-data",
        help="Download and tokenize the dataset declared by a config.",
    )
    prepare_data.add_argument("--config", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Run a configured throughput calibration.",
    )
    calibrate.add_argument(
        "--config",
        action="append",
        required=True,
        help="Exact immutable config; repeat for bounded concurrent calibration.",
    )
    calibrate.add_argument(
        "--worker-slot",
        action="append",
        type=_gpu_worker_slot,
        default=[],
        metavar="SLOT=CUDA_DEVICE",
        help=(
            "Explicit one-GPU worker mapping; repeat at least twice only with "
            "multiple calibration configs."
        ),
    )

    profile_hardware = subparsers.add_parser(
        "profile-hardware",
        help="Run a non-scientific physical-microbatch profile on one GPU.",
    )
    profile_hardware.add_argument("--architecture", required=True)
    profile_hardware.add_argument("--revision", required=True)
    profile_hardware.add_argument("--gpu-class", required=True)
    profile_hardware.add_argument(
        "--candidate-microbatches",
        required=True,
        type=_positive_int_list,
        metavar="N[,N...]",
    )
    profile_hardware.add_argument("--repeats", type=_positive_int, default=2)
    profile_hardware.add_argument("--cuda-device", type=_nonnegative_int, required=True)
    profile_hardware.add_argument(
        "--worker-timeout-seconds",
        type=_positive_float,
        required=True,
        help="Hard timeout for each fresh profiling worker.",
    )
    profile_hardware.add_argument(
        "--container-image",
        required=True,
        help="Immutable container reference ending in @sha256:<64 hex>.",
    )
    profile_hardware.add_argument("--work-root", required=True)
    profile_hardware.add_argument(
        "--retry-failed",
        action="store_true",
        help="Resume only after reviewing a preserved infrastructure failure.",
    )

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
        help=f"Comma-separated transformer sites to clip: {', '.join(SITE_ALIAS_ORDER)}.",
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

    clipping_frontier = subparsers.add_parser(
        "clipping-frontier",
        help="Measure one configured multi-checkpoint clipping frontier.",
    )
    clipping_frontier.add_argument("--config", required=True)

    calibrate_clipping_frontier = subparsers.add_parser(
        "calibrate-clipping-frontier",
        help=(
            "Time the first configured zero-threshold clipping point without "
            "creating scientific artifacts."
        ),
    )
    calibrate_clipping_frontier.add_argument("--config", required=True)

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

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=True)
            require_raw_output(
                config, repository=repository, config_path=config_path
            )
            run_dir = run_smoke(
                config,
                config_path=config_path,
                command=command,
                worker_slots=args.worker_slot,
                require_cuda=args.require_cuda,
                allow_shared_gpu=args.allow_shared_gpu,
            )
            print(f"Smoke run written to {run_dir}")
            return 0

        if args.command == "prepare-data":
            from paper_exp.data import prepare_tokenized_data

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_raw_output(config, repository=repository, config_path=config_path)
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
            from paper_exp.runner import run_calibrations

            run_dirs = run_calibrations(
                args.config,
                command=command,
                worker_slots=args.worker_slot,
            )
            for run_dir in run_dirs:
                print(f"Calibration run written to {run_dir}")
            return 0

        if args.command == "profile-hardware":
            from paper_exp.hardware_profile import HardwareProfileRequest
            from paper_exp.hardware_profile_run import run_hardware_profile

            request = HardwareProfileRequest(
                architecture=args.architecture,
                revision=args.revision,
                gpu_class=args.gpu_class,
                candidate_microbatches=tuple(args.candidate_microbatches),
                repeats=args.repeats,
            )
            work_root = Path(args.work_root)
            if not work_root.is_absolute():
                work_root = repository_path() / work_root
            result = run_hardware_profile(
                request,
                cuda_device=args.cuda_device,
                work_root=work_root,
                checkpoint_scratch=work_root / "checkpoint-scratch",
                worker_timeout_seconds=args.worker_timeout_seconds,
                container_image=args.container_image,
                retry_failed=args.retry_failed,
            )
            print(
                "Hardware profile written to "
                f"{result.artifact_path} (sha256 {result.artifact_sha256})"
            )
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

        if args.command == "clipping-frontier":
            from paper_exp.diagnostics.clipping_frontier import run_clipping_frontier

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_raw_output(config, repository=repository, config_path=config_path)
            with direct_launch_guard(repository=repository):
                run_dir = run_clipping_frontier(
                    config,
                    config_path=config_path,
                    command=command,
                    repository=repository,
                )
            print(f"Clipping frontier written to {run_dir}")
            return 0

        if args.command == "calibrate-clipping-frontier":
            from paper_exp.diagnostics.clipping_frontier import (
                calibrate_clipping_frontier,
            )

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_raw_output(config, repository=repository, config_path=config_path)
            with direct_launch_guard(repository=repository):
                report = calibrate_clipping_frontier(
                    config,
                    repository=repository,
                )
            print(json.dumps(report, sort_keys=True))
            return 0

        if args.command == "activation-histograms":
            from paper_exp.diagnostics.activation_histograms import run_activation_histograms

            repository, config_path = resolve_launch_config(args.config)
            config = load_config(config_path, allow_todos=False)
            require_raw_output(config, repository=repository, config_path=config_path)
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
            require_raw_output(config, repository=repository, config_path=config_path)
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
            require_raw_output(config, repository=repository, config_path=config_path)
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


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_int_list(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of positive integers"
        )
    return tuple(_positive_int(part.strip()) for part in parts)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _gpu_worker_slot(value: str):
    from paper_exp.runner import RunnerError, parse_worker_slot

    try:
        return parse_worker_slot(value)
    except RunnerError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


_smoke_worker_slot = _gpu_worker_slot


if __name__ == "__main__":
    raise SystemExit(main())
