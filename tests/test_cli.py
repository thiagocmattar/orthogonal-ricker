from __future__ import annotations

import argparse

import pytest

from paper_exp.cli import PLOT_KINDS, build_parser


EXPECTED_COMMANDS = {
    "smoke",
    "prepare-data",
    "calibrate",
    "pretrain",
    "run-configs",
    "run-status",
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
        ["smoke", "--config", "configs/00-smoke.yaml"]
    )

    assert args.config == "configs/00-smoke.yaml"


@pytest.mark.parametrize(
    "command",
    [
        "prepare-data",
        "calibrate",
        "pretrain",
        "activation-histograms",
        "activation-propagation",
        "weight-histograms",
    ],
)
def test_config_driven_commands_require_an_explicit_config(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_run_configs_preserves_repeated_config_order_and_state_alias() -> None:
    args = build_parser().parse_args(
        [
            "run-configs",
            "--config",
            "configs/01-a.yaml",
            "--config",
            "configs/02-b.yaml",
            "--state-path",
            "runtime/state.json",
        ]
    )

    assert args.config == ["configs/01-a.yaml", "configs/02-b.yaml"]
    assert args.state == "runtime/state.json"


def test_plot_requires_one_explicit_supported_artifact_kind() -> None:
    for kind in PLOT_KINDS:
        args = build_parser().parse_args(
            [
                "plot",
                "--kind",
                kind,
                "--run-dir",
                "results/example/001-run",
                "--output",
                "figures/01-example.pdf",
            ]
        )
        assert args.kind == kind


def test_clipping_has_no_implicit_cutoff_and_requires_an_explicit_seed() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["clip-sweep", "--run-dir", "results/source/001-run"])

    args = parser.parse_args(
        [
            "clip-sweep",
            "--run-dir",
            "results/source/001-run",
            "--seed",
            "7",
        ]
    )
    assert args.thresholds == ""
    assert args.quantiles == ""
    assert args.rms_multipliers == ""
    assert args.seed == 7


def test_clipping_sweep_requires_an_explicit_evaluation_seed() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["clip-sweep", "--run-dir", "results/example/001-run"]
        )

    args = build_parser().parse_args(
        [
            "clip-sweep",
            "--run-dir",
            "results/example/001-run",
            "--seed",
            "17",
        ]
    )
    assert args.seed == 17
