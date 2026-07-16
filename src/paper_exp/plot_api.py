"""Small shared API for paper-figure layout and export.

Figure families keep their scientific inputs, reductions, labels, and panel
contents explicit.  This module only standardizes two mechanical concerns that
otherwise repeat across every family: predictable panel grids and scoped
PDF/PNG export.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from paper_exp.plot_style import PLOT_STYLE


# Practical starting widths for venue-ready figures.  Figure families may
# override them when a venue supplies a different final typeset width.
SINGLE_COLUMN_WIDTH_INCHES = 3.5
DOUBLE_COLUMN_WIDTH_INCHES = 7.16

ShareMode = bool | Literal["none", "all", "row", "col"]


def _normalize_share_mode(mode: ShareMode) -> Literal["none", "all", "row", "col"] | str:
    if mode is True:
        return "all"
    if mode is False:
        return "none"
    return mode


@dataclass(frozen=True)
class GridLayout:
    """Physical and structural defaults for a regular panel grid.

    ``panel_aspect`` is panel width divided by panel height.  When
    ``height_inches`` is omitted, the figure height is derived from the final
    figure width, effective column count, row count, and this aspect ratio.
    """

    columns: int
    width_inches: float = DOUBLE_COLUMN_WIDTH_INCHES
    panel_aspect: float = 4.0 / 3.0
    height_inches: float | None = None
    sharex: ShareMode = False
    sharey: ShareMode = False
    hspace: float | None = None
    wspace: float | None = None
    colorbar_width_ratio: float = 0.04

    def __post_init__(self) -> None:
        if (
            isinstance(self.columns, bool)
            or not isinstance(self.columns, Integral)
            or self.columns <= 0
        ):
            raise ValueError("GridLayout.columns must be a positive integer.")
        if not math.isfinite(self.width_inches) or self.width_inches <= 0.0:
            raise ValueError("GridLayout.width_inches must be finite and positive.")
        if not math.isfinite(self.panel_aspect) or self.panel_aspect <= 0.0:
            raise ValueError("GridLayout.panel_aspect must be finite and positive.")
        if self.height_inches is not None and (
            not math.isfinite(self.height_inches) or self.height_inches <= 0.0
        ):
            raise ValueError("GridLayout.height_inches must be finite and positive when provided.")
        if _normalize_share_mode(self.sharex) not in {"none", "all", "row", "col"}:
            raise ValueError(f"Unsupported sharex mode: {self.sharex!r}.")
        if _normalize_share_mode(self.sharey) not in {"none", "all", "row", "col"}:
            raise ValueError(f"Unsupported sharey mode: {self.sharey!r}.")
        for name, value in (("hspace", self.hspace), ("wspace", self.wspace)):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"GridLayout.{name} must be finite and nonnegative when provided.")
        if not math.isfinite(self.colorbar_width_ratio) or self.colorbar_width_ratio <= 0.0:
            raise ValueError("GridLayout.colorbar_width_ratio must be finite and positive.")


SINGLE_COLUMN_LAYOUT = GridLayout(
    columns=1,
    width_inches=SINGLE_COLUMN_WIDTH_INCHES,
)
DOUBLE_COLUMN_LAYOUT = GridLayout(
    columns=2,
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
)


@dataclass(frozen=True)
class PublicationProfile:
    """Mechanical constraints checked on a figure at its authored size."""

    width_inches: float
    max_height_inches: float
    min_text_points: float = 8.0
    size_tolerance_inches: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("width_inches", self.width_inches),
            ("max_height_inches", self.max_height_inches),
            ("min_text_points", self.min_text_points),
            ("size_tolerance_inches", self.size_tolerance_inches),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"PublicationProfile.{name} must be finite and positive.")


REPORT04_PUBLICATION_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=8.8,
    min_text_points=8.0,
)


class PlotQualityError(ValueError):
    """Raised when a figure violates its explicit publication profile."""


def publication_figure_issues(
    figure: Figure,
    profile: PublicationProfile,
) -> tuple[str, ...]:
    """Return deterministic size, typography, and text-containment issues."""

    issues: list[str] = []
    width, height = (float(value) for value in figure.get_size_inches())
    if abs(width - profile.width_inches) > profile.size_tolerance_inches:
        issues.append(
            f"figure width is {width:.3f} in; expected {profile.width_inches:.3f} in"
        )
    if height > profile.max_height_inches + profile.size_tolerance_inches:
        issues.append(
            f"figure height is {height:.3f} in; maximum is {profile.max_height_inches:.3f} in"
        )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox
    tolerance_pixels = figure.dpi / 72.0
    nonrendered_tick_labels = _nonrendered_tick_labels(figure)
    for text_artist in figure.findobj(match=Text):
        if (
            not text_artist.get_visible()
            or not text_artist.get_text().strip()
            or text_artist in nonrendered_tick_labels
        ):
            continue
        font_size = float(text_artist.get_fontsize())
        if font_size + 1e-9 < profile.min_text_points:
            excerpt = " ".join(text_artist.get_text().split())[:60]
            issues.append(
                f"text {excerpt!r} uses {font_size:g} pt; minimum is {profile.min_text_points:g} pt"
            )
        box = text_artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < figure_box.x0 - tolerance_pixels
            or box.y0 < figure_box.y0 - tolerance_pixels
            or box.x1 > figure_box.x1 + tolerance_pixels
            or box.y1 > figure_box.y1 + tolerance_pixels
        ):
            excerpt = " ".join(text_artist.get_text().split())[:60]
            issues.append(f"text {excerpt!r} extends outside the figure canvas")
    return tuple(issues)


def validate_publication_figure(
    figure: Figure,
    profile: PublicationProfile,
) -> None:
    """Raise one actionable error containing every publication-profile issue."""

    issues = publication_figure_issues(figure, profile)
    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise PlotQualityError(f"Figure failed publication checks:\n{detail}")


@dataclass(frozen=True)
class PanelGrid:
    """A regular grid with stable two-dimensional and used-flat axis views."""

    figure: Figure
    axes: np.ndarray[Any, np.dtype[np.object_]]
    flat_axes: tuple[Axes, ...]
    hidden_axes: tuple[Axes, ...]
    colorbar_axis: Axes | None = None


def make_panel_grid(
    panel_count: int,
    layout: GridLayout,
    *,
    shared_colorbar: bool = False,
    subplot_kwargs: Mapping[str, Any] | None = None,
) -> PanelGrid:
    """Create a count-derived grid whose ``axes`` array is always two-dimensional.

    The requested column count is capped at ``panel_count``.  Any trailing cell
    in the final row is retained in ``axes`` but hidden, while ``flat_axes``
    contains only the visible panel axes.  A shared colorbar axis, when
    requested, spans every panel row and is not part of either panel-axis view.
    """

    if isinstance(panel_count, bool) or not isinstance(panel_count, int) or panel_count <= 0:
        raise ValueError("panel_count must be a positive integer.")

    columns = min(layout.columns, panel_count)
    rows = math.ceil(panel_count / columns)
    colorbar_ratio = layout.colorbar_width_ratio if shared_colorbar else 0.0
    if layout.height_inches is None:
        panel_width = layout.width_inches / (columns + colorbar_ratio)
        height_inches = rows * panel_width / layout.panel_aspect
    else:
        height_inches = layout.height_inches

    width_ratios = [1.0] * columns
    if shared_colorbar:
        width_ratios.append(layout.colorbar_width_ratio)
    gridspec_kwargs: dict[str, Any] = {"width_ratios": width_ratios}
    if layout.hspace is not None:
        gridspec_kwargs["hspace"] = layout.hspace
    if layout.wspace is not None:
        gridspec_kwargs["wspace"] = layout.wspace
    axis_kwargs = dict(subplot_kwargs or {})
    reserved_subplot_keys = {"sharex", "sharey"}.intersection(axis_kwargs)
    if reserved_subplot_keys:
        reserved = ", ".join(sorted(reserved_subplot_keys))
        raise ValueError(f"Configure {reserved} through GridLayout, not subplot_kwargs.")

    figure = plt.figure(figsize=(layout.width_inches, height_inches))
    try:
        grid = figure.add_gridspec(rows, len(width_ratios), **gridspec_kwargs)
        axes = np.empty((rows, columns), dtype=object)
        sharex_mode = _normalize_share_mode(layout.sharex)
        sharey_mode = _normalize_share_mode(layout.sharey)
        for row in range(rows):
            for column in range(columns):
                sharex = _share_anchor(axes, row, column, sharex_mode)
                sharey = _share_anchor(axes, row, column, sharey_mode)
                axes[row, column] = figure.add_subplot(
                    grid[row, column],
                    sharex=sharex,
                    sharey=sharey,
                    **axis_kwargs,
                )

        all_axes = tuple(axes.flat)
        flat_axes = all_axes[:panel_count]
        hidden_axes = all_axes[panel_count:]
        for axis in hidden_axes:
            axis.set_visible(False)

        colorbar_axis = figure.add_subplot(grid[:, -1]) if shared_colorbar else None
        return PanelGrid(
            figure=figure,
            axes=axes,
            flat_axes=flat_axes,
            hidden_axes=hidden_axes,
            colorbar_axis=colorbar_axis,
        )
    except BaseException:
        plt.close(figure)
        raise


def export_figure(
    build_figure: Callable[[], Figure],
    output: str | Path,
    *,
    save_png: bool = False,
    style: Mapping[str, Any] | None = None,
    profile: PublicationProfile | None = None,
) -> list[Path]:
    """Build once, export PDF and optional PNG, and close all new figures.

    The Matplotlib rc state is scoped to the build and export operation.  The
    primary output must be a PDF; the optional PNG uses the same fully-built
    ``Figure`` object, so both formats contain the same artists and limits.
    """

    output_path = Path(output)
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("Primary plot output must use a .pdf suffix.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [output_path]
    if save_png:
        output_paths.append(output_path.with_suffix(".png"))

    existing_figures = set(plt.get_fignums())
    figure: Figure | None = None
    staged_paths: dict[Path, Path] = {}
    rc = dict(PLOT_STYLE if style is None else style)
    if profile is not None:
        # Publication profiles constrain the authored canvas. Tight bounding
        # boxes would crop that canvas during save and silently change the PDF
        # MediaBox or PNG dimensions after validation.
        rc["savefig.bbox"] = None
    try:
        with plt.rc_context(rc=rc):
            built_figure = build_figure()
            if not isinstance(built_figure, Figure):
                raise TypeError("build_figure must return a matplotlib.figure.Figure.")
            figure = built_figure
            if profile is not None:
                validate_publication_figure(figure, profile)
            for path in output_paths:
                staged_path = _temporary_sibling(path, "stage")
                staged_paths[path] = staged_path
                save_kwargs: dict[str, Any] = {}
                if path.suffix.lower() == ".pdf":
                    # Matplotlib otherwise injects the wall-clock creation
                    # time, making scientifically identical paper artifacts
                    # differ byte-for-byte and invalidating output hashes.
                    save_kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
                figure.savefig(staged_path, **save_kwargs)
        publish_staged_outputs(staged_paths)
    finally:
        for staged_path in staged_paths.values():
            staged_path.unlink(missing_ok=True)
        if figure is not None:
            plt.close(figure)
        for figure_number in set(plt.get_fignums()).difference(existing_figures):
            plt.close(figure_number)

    return output_paths


def _nonrendered_tick_labels(figure: Figure) -> set[Text]:
    """Return tick-label artists that Matplotlib will not draw for this view."""

    nonrendered: set[Text] = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            all_ticks = (*axis.get_major_ticks(), *axis.get_minor_ticks())
            axis_is_drawn = axes.get_visible() and axes.axison and axis.get_visible()
            active_ticks = set(axis._update_ticks()) if axis_is_drawn else set()
            for tick in all_ticks:
                if tick not in active_ticks:
                    nonrendered.update((tick.label1, tick.label2))
    return nonrendered


def _temporary_sibling(path: Path, purpose: str) -> Path:
    """Reserve a same-directory temporary path that preserves output format."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=f".{purpose}{path.suffix}",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def publish_staged_outputs(staged_paths: Mapping[Path, Path]) -> None:
    """Atomically promote ``final_path -> staged_path`` entries with rollback.

    Every staged file must be on the same filesystem as its final path. Existing
    final files are backed up before promotion and restored if any promotion
    fails, so callers never retain a partially updated output set.
    """

    backup_paths: dict[Path, Path] = {}
    published_paths: list[Path] = []
    publishing_started = False
    try:
        for output_path in staged_paths:
            if output_path.exists():
                backup_path = _temporary_sibling(output_path, "backup")
                backup_paths[output_path] = backup_path
                shutil.copy2(output_path, backup_path)

        publishing_started = True
        for output_path, staged_path in staged_paths.items():
            staged_path.replace(output_path)
            published_paths.append(output_path)
    except BaseException:
        if publishing_started:
            for output_path in published_paths:
                output_path.unlink(missing_ok=True)
            for output_path, backup_path in backup_paths.items():
                backup_path.replace(output_path)
        raise
    finally:
        for staged_path in staged_paths.values():
            staged_path.unlink(missing_ok=True)
        for backup_path in backup_paths.values():
            backup_path.unlink(missing_ok=True)


def _share_anchor(
    axes: np.ndarray[Any, np.dtype[np.object_]],
    row: int,
    column: int,
    mode: str,
) -> Axes | None:
    if mode == "all" and (row > 0 or column > 0):
        return axes[0, 0]
    if mode == "row" and column > 0:
        return axes[row, 0]
    if mode == "col" and row > 0:
        return axes[0, column]
    return None
