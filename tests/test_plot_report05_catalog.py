from __future__ import annotations

import pytest

from paper_exp.plot_catalog import (
    REPORT05_FIGURES,
    get_report05_figure,
    list_report05_figures,
    report05_catalog_rows,
)


def test_report05_catalog_is_complete_ordered_and_unique() -> None:
    assert [entry.number for entry in REPORT05_FIGURES] == list(range(91, 103))
    assert len({entry.filename for entry in REPORT05_FIGURES}) == 12
    assert len({entry.plot_type for entry in REPORT05_FIGURES}) == 12
    assert len({entry.public_wrapper for entry in REPORT05_FIGURES}) == 12
    assert list_report05_figures(embedded_only=True) == REPORT05_FIGURES


def test_report05_catalog_lookup_and_rows_are_deterministic() -> None:
    entry = get_report05_figure(102)
    assert get_report05_figure(entry.filename) is entry
    assert get_report05_figure(entry.public_wrapper) is entry
    assert report05_catalog_rows()[-1].startswith("102 | logical_compute_frontier |")


def test_report05_catalog_rejects_unknown_identifier() -> None:
    with pytest.raises(KeyError, match="Unknown Report 05 figure"):
        get_report05_figure("missing")
