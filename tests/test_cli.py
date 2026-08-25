from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_exp.cli import PLOT_KINDS, build_parser, main


EXPECTED_COMMANDS = {
    "smoke",
    "prepare-data",
    "calibrate",
    "profile-hardware",
    "check",
    "clip-sweep",
    "activation-histograms",
    "activation-propagation",
    "weight-histograms",
    "plot",
}


def test_cli_exposes_only_the_reset_command_surface() -> None:
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparser_action.choices) == EXPECTED_COMMANDS


def test_smoke_requires_the_explicit_clean_config() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["smoke"])

    args = build_parser().parse_args(
        [
            "smoke",
            "--config",
            "experiments/00-infrastructure-smoke/run/00-smoke.yaml",
        ]
    )

    assert args.config == "experiments/00-infrastructure-smoke/run/00-smoke.yaml"
    assert args.worker_slot == []
    assert args.require_cuda is False
    assert args.allow_shared_gpu is False


def test_smoke_parses_two_explicit_cuda_workers() -> None:
    args = build_parser().parse_args(
        [
            "smoke",
            "--config",
            "experiments/00-infrastructure-smoke/run/00-smoke.yaml",
            "--worker-slot",
            "gpu-0=0",
            "--worker-slot",
            "gpu-1=1",
            "--require-cuda",
        ]
    )

    assert [(slot.slot_id, slot.payload) for slot in args.worker_slot] == [
        ("gpu-0", "0"),
        ("gpu-1", "1"),
    ]
    assert args.require_cuda is True

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "smoke",
                "--config",
                "experiments/00-infrastructure-smoke/run/00-smoke.yaml",
                "--worker-slot",
                "gpu-0=-1",
            ]
        )


def test_smoke_main_forwards_concurrent_worker_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    config_path = repository / "experiments/00-infrastructure-smoke/run/00-smoke.yaml"
    config = {"kind": "smoke"}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "paper_exp.cli.resolve_launch_config",
        lambda value: (repository, config_path),
    )
    monkeypatch.setattr(
        "paper_exp.cli.load_config",
        lambda path, *, allow_todos: config,
    )
    monkeypatch.setattr("paper_exp.cli.require_raw_output", lambda *args, **kwargs: None)

    def fake_run_smoke(passed_config, **kwargs):
        captured["config"] = passed_config
        captured.update(kwargs)
        return tmp_path / "smoke-run"

    monkeypatch.setattr("paper_exp.run.run_smoke", fake_run_smoke)

    assert main(
        [
            "smoke",
            "--config",
            str(config_path),
            "--worker-slot",
            "gpu-0=0",
            "--worker-slot",
            "gpu-1=0",
            "--require-cuda",
            "--allow-shared-gpu",
        ]
    ) == 0
    assert captured["config"] is config
    assert captured["config_path"] == config_path
    assert [
        (slot.slot_id, slot.payload) for slot in captured["worker_slots"]
    ] == [("gpu-0", "0"), ("gpu-1", "0")]
    assert captured["require_cuda"] is True
    assert captured["allow_shared_gpu"] is True


def test_calibration_duration_has_no_cli_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "calibrate",
            "--config",
            "experiments/01-a1-grid/run/001-example.yaml",
        ]
    )
    assert args.config == "experiments/01-a1-grid/run/001-example.yaml"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "calibrate",
                "--config",
                "experiments/01-a1-grid/run/001-example.yaml",
                "--max-wall-seconds",
                "1",
            ]
        )


def test_hardware_profile_requires_explicit_pinned_operational_inputs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "profile-hardware",
            "--architecture",
            "EleutherAI/pythia-14m-deduped",
            "--revision",
            "a" * 40,
            "--gpu-class",
            "NVIDIA A40 48GB",
            "--candidate-microbatches",
            "1,2,4",
            "--cuda-device",
            "0",
            "--worker-timeout-seconds",
            "1200",
            "--container-image",
            "runpod/pytorch@sha256:" + "d" * 64,
            "--work-root",
            "experiments/00-infrastructure-smoke/raw/profile-14m-a40",
        ]
    )

    assert args.candidate_microbatches == (1, 2, 4)
    assert args.repeats == 2
    assert args.cuda_device == 0
    assert args.worker_timeout_seconds == 1200.0
    assert args.container_image.endswith("d" * 64)
    assert args.retry_failed is False

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "profile-hardware",
                "--architecture",
                "model",
                "--revision",
                "a" * 40,
                "--gpu-class",
                "gpu",
                "--candidate-microbatches",
                "1,0",
                "--cuda-device",
                "0",
                "--worker-timeout-seconds",
                "1200",
                "--container-image",
                "runpod/pytorch@sha256:" + "d" * 64,
                "--work-root",
                "profile",
            ]
        )


@pytest.mark.parametrize("absolute_root", (False, True))
def test_profile_main_forwards_pinned_inputs_and_resolves_relative_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absolute_root: bool,
) -> None:
    repository = tmp_path / "repository"
    root_argument = (
        tmp_path / "absolute-profile"
        if absolute_root
        else Path("experiments/00-infrastructure-smoke/raw/profile-14m-a40")
    )
    expected_root = root_argument if absolute_root else repository / root_argument
    captured: dict[str, object] = {}

    monkeypatch.setattr("paper_exp.cli.repository_path", lambda: repository)

    def fake_profile(request, **kwargs):
        captured["request"] = request
        captured.update(kwargs)
        return SimpleNamespace(
            artifact_path=expected_root / "hardware_profile.json",
            artifact_sha256="f" * 64,
        )

    monkeypatch.setattr(
        "paper_exp.hardware_profile_run.run_hardware_profile",
        fake_profile,
    )
    image = "runpod/pytorch@sha256:" + "d" * 64
    assert main(
        [
            "profile-hardware",
            "--architecture",
            "EleutherAI/pythia-14m-deduped",
            "--revision",
            "a" * 40,
            "--gpu-class",
            "NVIDIA A40 48GB",
            "--candidate-microbatches",
            "1,2,4",
            "--repeats",
            "3",
            "--cuda-device",
            "1",
            "--worker-timeout-seconds",
            "1200",
            "--container-image",
            image,
            "--work-root",
            str(root_argument),
            "--retry-failed",
        ]
    ) == 0

    request = captured["request"]
    assert request.architecture == "EleutherAI/pythia-14m-deduped"
    assert request.revision == "a" * 40
    assert request.gpu_class == "NVIDIA A40 48GB"
    assert request.candidate_microbatches == (1, 2, 4)
    assert request.repeats == 3
    assert captured["cuda_device"] == 1
    assert captured["work_root"] == expected_root
    assert captured["checkpoint_scratch"] == expected_root / "checkpoint-scratch"
    assert captured["worker_timeout_seconds"] == 1200.0
    assert captured["container_image"] == image
    assert captured["retry_failed"] is True


@pytest.mark.parametrize(
    "command",
    [
        "prepare-data",
        "calibrate",
        "activation-histograms",
        "activation-propagation",
        "weight-histograms",
    ],
)
def test_config_driven_commands_require_an_explicit_config(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_plot_requires_one_explicit_supported_artifact_kind() -> None:
    for kind in PLOT_KINDS:
        args = build_parser().parse_args(
            [
                "plot",
                "--kind",
                kind,
                "--run-dir",
                "experiments/01-a1-grid/raw/001-example/001-run",
                "--output",
                "experiments/01-a1-grid/figs/01-example.pdf",
            ]
        )
        assert args.kind == kind


def test_clipping_has_no_implicit_cutoff_and_requires_an_explicit_seed() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "clip-sweep",
                "--run-dir",
                "experiments/01-a1-grid/raw/001-source/001-run",
            ]
        )

    args = parser.parse_args(
        [
            "clip-sweep",
            "--run-dir",
            "experiments/01-a1-grid/raw/001-source/001-run",
            "--seed",
            "7",
        ]
    )
    assert args.thresholds == ""
    assert args.quantiles == ""
    assert args.rms_multipliers == ""
    assert args.seed == 7
