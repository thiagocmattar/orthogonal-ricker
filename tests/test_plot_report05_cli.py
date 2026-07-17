from __future__ import annotations

from pathlib import Path

from paper_exp import cli
from paper_exp.plots import Report05InputError


def test_plot_report05_parser_is_strict_by_default() -> None:
    args = cli.build_parser().parse_args(["plot-report05"])

    assert args.command == "plot-report05"
    assert args.results == "results"
    assert args.figures == "figures"
    assert not args.png
    assert not args.allow_partial


def test_plot_report05_cli_routes_publication_options(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def record_suite(**kwargs: object) -> list[Path]:
        calls.append(kwargs)
        return [Path("paper/91.pdf"), Path("paper/91.png")]

    monkeypatch.setattr(cli, "generate_report05_figures", record_suite)

    assert cli.main(
        [
            "plot-report05",
            "--results",
            "saved",
            "--figures",
            "paper",
            "--png",
            "--allow-partial",
        ]
    ) == 0
    assert calls == [
        {
            "results_dir": "saved",
            "figures_dir": "paper",
            "save_png": True,
            "strict": False,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"Wrote {Path('paper/91.pdf')}",
        f"Wrote {Path('paper/91.png')}",
    ]


def test_plot_report05_cli_reports_strict_preflight_failure(monkeypatch, capsys) -> None:
    def fail_preflight(**_kwargs: object) -> list[Path]:
        raise Report05InputError("missing exact joint clipping")

    monkeypatch.setattr(cli, "generate_report05_figures", fail_preflight)

    assert cli.main(["plot-report05"]) == 2
    assert "missing exact joint clipping" in capsys.readouterr().err


def test_plot_catalog_report05_is_explicit_opt_in(capsys) -> None:
    assert cli.main(["plot-catalog", "--report", "05"]) == 0
    rows = capsys.readouterr().out.splitlines()
    assert rows[0].startswith("91 | architecture_diagram |")
    assert rows[-1].startswith("102 | logical_compute_frontier |")
