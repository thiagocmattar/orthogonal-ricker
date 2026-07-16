from __future__ import annotations

from paper_exp.cli import build_parser, main
from paper_exp.plot_catalog import report04_catalog_rows


def test_plot_catalog_parser_defaults_to_the_full_suite() -> None:
    args = build_parser().parse_args(["plot-catalog"])

    assert args.command == "plot-catalog"
    assert args.embedded_only is False


def test_plot_catalog_parser_accepts_embedded_only() -> None:
    args = build_parser().parse_args(["plot-catalog", "--embedded-only"])

    assert args.command == "plot-catalog"
    assert args.embedded_only is True


def test_plot_catalog_command_prints_the_deterministic_full_catalog(capsys) -> None:
    assert main(["plot-catalog"]) == 0

    assert capsys.readouterr().out.splitlines() == list(report04_catalog_rows())


def test_plot_catalog_command_filters_to_embedded_figures(capsys) -> None:
    assert main(["plot-catalog", "--embedded-only"]) == 0

    output_rows = capsys.readouterr().out.splitlines()
    assert output_rows == list(report04_catalog_rows(embedded_only=True))
    assert len(output_rows) == 10
    assert not any(row.startswith(("81 |", "84 |")) for row in output_rows)
