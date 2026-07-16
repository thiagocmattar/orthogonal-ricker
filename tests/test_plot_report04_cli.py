from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import paper_exp.cli as cli
from paper_exp.plots import Report04InputError


def test_plot_report04_parser_is_strict_by_default() -> None:
    args = cli.build_parser().parse_args(["plot-report04"])

    assert args.command == "plot-report04"
    assert args.results == "results"
    assert args.figures == "figures"
    assert args.png is False
    assert args.allow_partial is False
    assert args.include_rn is False


@pytest.mark.parametrize(
    ("extra_args", "expected_strict", "expected_png", "expected_include_rn"),
    [
        ([], True, False, False),
        (["--allow-partial", "--png"], False, True, False),
        (["--include-rn"], True, False, True),
    ],
)
def test_plot_report04_cli_routes_suite_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    expected_strict: bool,
    expected_png: bool,
    expected_include_rn: bool,
) -> None:
    calls: list[dict[str, Any]] = []
    output = Path("selected-figures") / "79-example.pdf"

    def record_suite(**kwargs: Any) -> list[Path]:
        calls.append(kwargs)
        return [output]

    monkeypatch.setattr(cli, "generate_report04_figures", record_suite)

    exit_code = cli.main(
        [
            "plot-report04",
            "--results",
            "selected-results",
            "--figures",
            "selected-figures",
            *extra_args,
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "results_dir": "selected-results",
            "figures_dir": "selected-figures",
            "save_png": expected_png,
            "strict": expected_strict,
            "write_provenance": expected_strict,
            "include_rn": expected_include_rn,
        }
    ]
    assert capsys.readouterr().out == f"Wrote {output}\n"


def test_plot_report04_cli_reports_strict_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_preflight(**_kwargs: Any) -> list[Path]:
        raise Report04InputError("Report 04 input preflight failed")

    monkeypatch.setattr(cli, "generate_report04_figures", fail_preflight)

    assert cli.main(["plot-report04"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Report 04 input preflight failed\n"
