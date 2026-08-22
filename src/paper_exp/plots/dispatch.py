"""Load, validate, dispatch, and record plots from one saved run artifact.

This module deliberately has no experiment registry, run discovery, or paper
cohort selection. Callers name the exact run directory and artifact kind.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from matplotlib.figure import Figure

from paper_exp.launch import resolve_figure_output, resolve_launch_run_dir
from paper_exp.utils import read_json

from .activation_histograms import build_activation_histograms
from .clipping import build_clipping_frontier
from .export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    export_figure,
)
from .histograms import finite_number
from .propagation import build_activation_propagation
from .run_diagnostics import build_run_diagnostics
from .style import PAPER_STYLE
from .weight_histograms import build_weight_histograms


PLOT_KINDS = (
    "run",
    "clipping",
    "activation-histograms",
    "weight-histograms",
    "activation-propagation",
)
_PLOT_INPUT_FILES = {
    "run": ("config.yaml", "manifest.json", "events.jsonl", "metrics.json"),
    "clipping": ("config.yaml", "manifest.json", "clipping_frontier.jsonl"),
    "activation-histograms": (
        "config.yaml",
        "manifest.json",
        "activation_histograms.json",
    ),
    "weight-histograms": (
        "config.yaml",
        "manifest.json",
        "weight_histograms.json",
    ),
    "activation-propagation": (
        "config.yaml",
        "manifest.json",
        "activation_propagation.json",
    ),
}


def plot_artifact(
    *,
    kind: str,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
    repository: str | Path | None = None,
) -> list[Path]:
    """Render one explicit saved artifact to PDF and, optionally, PNG."""

    handlers: dict[str, tuple[Callable[[Path], Any], Callable[[Any], Figure]]] = {
        "run": (_load_run_diagnostics, build_run_diagnostics),
        "clipping": (_load_clipping_frontier, build_clipping_frontier),
        "activation-histograms": (
            lambda path: _load_versioned_mapping(
                path / "activation_histograms.json",
                expected_version=3,
            ),
            build_activation_histograms,
        ),
        "weight-histograms": (
            lambda path: _load_versioned_mapping(
                path / "weight_histograms.json",
                expected_version=1,
            ),
            build_weight_histograms,
        ),
        "activation-propagation": (
            lambda path: _load_versioned_mapping(
                path / "activation_propagation.json",
                expected_version=5,
            ),
            build_activation_propagation,
        ),
    }
    if kind not in handlers:
        choices = ", ".join(PLOT_KINDS)
        raise ValueError(f"Unknown plot kind {kind!r}; expected one of: {choices}.")
    root, run_path = resolve_launch_run_dir(run_dir, repository=repository)
    output_path = resolve_figure_output(
        output,
        source_run=run_path,
        repository=root,
    )
    input_paths = _plot_input_paths(kind, run_path)

    loader, builder = handlers[kind]
    payload = loader(run_path)
    outputs = export_figure(
        lambda: builder(payload),
        output_path,
        save_png=save_png,
        style=PAPER_STYLE,
        profile=DOUBLE_COLUMN_PUBLICATION_PROFILE,
    )
    provenance_path = _write_plot_provenance(
        kind=kind,
        run_path=run_path,
        input_paths=input_paths,
        outputs=outputs,
        artifact_schema_version=(
            payload.get("schema_version") if isinstance(payload, dict) else None
        ),
    )
    return [*outputs, provenance_path]


def _load_run_diagnostics(run_path: Path) -> dict[str, Any]:
    events = _read_jsonl(run_path / "events.jsonl")
    metrics = _load_mapping(run_path / "metrics.json")
    manifest_path = run_path / "manifest.json"
    manifest = _load_mapping(manifest_path) if manifest_path.is_file() else {}

    train_events = [
        row
        for row in events
        if row.get("event") == "train"
        and finite_number(row.get("tokens_seen"))
        and finite_number(row.get("train_loss"))
    ]
    if not train_events:
        raise ValueError(f"No finite train events found in {run_path / 'events.jsonl'}.")
    validation_events = [
        row
        for row in events
        if row.get("event") == "validation"
        and finite_number(row.get("tokens_seen"))
        and finite_number(row.get("validation_loss"))
    ]
    return {
        "train_events": train_events,
        "validation_events": validation_events,
        "metrics": metrics,
        "manifest": manifest,
    }


def _load_clipping_frontier(run_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(run_path / "clipping_frontier.jsonl")
    if not rows:
        raise ValueError(f"No clipping rows found in {run_path / 'clipping_frontier.jsonl'}.")
    return rows


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {path}")
    return payload


def _load_versioned_mapping(
    path: Path,
    *,
    expected_version: int,
) -> dict[str, Any]:
    payload = _load_mapping(path)
    actual_version = payload.get("schema_version")
    if actual_version != expected_version:
        raise ValueError(
            f"Unsupported schema_version in {path}: expected {expected_version}, "
            f"found {actual_version!r}."
        )
    return payload


def _plot_input_paths(kind: str, run_path: Path) -> list[Path]:
    paths = [run_path / name for name in _PLOT_INPUT_FILES[kind]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Plot inputs are incomplete; missing: {rendered}")
    manifest = _load_mapping(run_path / "manifest.json")
    if (
        manifest.get("config_id") != run_path.parent.name
        or manifest.get("run_id") != run_path.name
    ):
        raise ValueError(f"Run manifest identity does not match its directory: {run_path}")
    tranche_id = run_path.parents[2].name
    if manifest.get("tranche_id") not in {None, tranche_id}:
        raise ValueError(
            f"Run manifest tranche does not match its directory: {run_path}"
        )
    if manifest.get("status") != "completed":
        raise ValueError(f"Plot input run is not completed: {run_path}")
    return paths


def _write_plot_provenance(
    *,
    kind: str,
    run_path: Path,
    input_paths: list[Path],
    outputs: list[Path],
    artifact_schema_version: Any,
) -> Path:
    manifest = _load_mapping(run_path / "manifest.json")
    sidecar = Path(outputs[0]).with_suffix(".provenance.json")
    payload = {
        "schema_version": 1,
        "plot_kind": kind,
        "artifact_schema_version": artifact_schema_version,
        "source": {
            "tranche_id": manifest.get("tranche_id"),
            "config_id": manifest.get("config_id"),
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status"),
            "git_commit": manifest.get("git_commit"),
            "source_run": manifest.get("source_run"),
            "source_runs": manifest.get("source_runs"),
            "source_checkpoint": manifest.get("source_checkpoint"),
            "source_checkpoints": manifest.get("source_checkpoints"),
        },
        "inputs": [
            {
                "path": _relative_path(path, anchor=sidecar.parent),
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": _relative_path(path, anchor=sidecar.parent),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    temporary = sidecar.with_name(f".{sidecar.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, *, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), start=anchor.resolve())).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Plot provenance cannot express a portable path across volumes: {path}"
        ) from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}.")
            rows.append(row)
    return rows
