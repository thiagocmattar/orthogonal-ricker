"""Shared source-run identity and checkpoint helpers for saved-run diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_exp.run import CORE_RUN_ARTIFACTS
from paper_exp.utils import portable_path, read_json


STANDARD_CHECKPOINT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
)


def find_source_run(
    config: dict[str, Any],
    selected: dict[str, Any],
    *,
    section: str,
    checkpoint_files: tuple[str, ...] = STANDARD_CHECKPOINT_FILES,
) -> Path:
    """Resolve one exact completed selected run with a usable saved checkpoint."""

    config_id = str(selected.get("config_id") or "").strip()
    run_id = str(selected.get("run_id") or "").strip()
    if not config_id or not run_id:
        raise ValueError(
            f"{section}.selected_runs entries require exact config_id and run_id."
        )
    run_dir = Path(config["output"]["dir"]) / config_id / run_id
    verify_completed_checkpoint_run(
        run_dir,
        config_id=config_id,
        run_id=run_id,
        checkpoint_files=checkpoint_files,
    )
    return run_dir


def verify_completed_checkpoint_run(
    run_dir: Path,
    *,
    config_id: str,
    run_id: str,
    checkpoint_files: tuple[str, ...] = STANDARD_CHECKPOINT_FILES,
) -> None:
    """Require the completed source envelope and an accepted checkpoint file."""

    missing = [name for name in CORE_RUN_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selected source run is missing required artifacts ({', '.join(missing)}): {run_dir}"
        )
    manifest = read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError(f"Selected source manifest is not an object: {run_dir}")
    if manifest.get("config_id") != config_id or manifest.get("run_id") != run_id:
        raise ValueError(f"Selected source run identity is inconsistent: {run_dir}")
    if manifest.get("status") != "completed":
        raise ValueError(f"Selected source run is not completed: {run_dir}")
    checkpoint_path = source_checkpoint_path(run_dir, manifest)
    model_files = tuple(checkpoint_path / name for name in checkpoint_files)
    if not checkpoint_path.is_dir() or not any(path.is_file() for path in model_files):
        raise FileNotFoundError(f"Selected source checkpoint is incomplete: {checkpoint_path}")


def source_checkpoint_path(source_run: Path, manifest: dict[str, Any]) -> Path:
    """Resolve the exact checkpoint path recorded by a completed source manifest."""

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("saved") is not True:
        raise ValueError(f"Selected source run has no saved checkpoint: {source_run}")
    return resolve_source_path(checkpoint.get("path"), source_run=source_run)


def resolve_source_path(value: Any, *, source_run: Path) -> Path:
    """Resolve a recorded absolute, repository-relative, or run-relative path."""

    path_text = str(value or "").strip()
    if not path_text:
        raise ValueError(f"Source run has no checkpoint path: {source_run}")
    recorded = Path(path_text)
    if recorded.is_absolute():
        return recorded.resolve()
    repository_path = (Path.cwd() / recorded).resolve()
    run_relative_path = (source_run / recorded).resolve()
    try:
        repository_path.relative_to(source_run.resolve())
        return repository_path
    except ValueError:
        pass
    if run_relative_path.exists() or not repository_path.exists():
        return run_relative_path
    return repository_path


def validate_shared_validation_cache(
    source_manifests: list[dict[str, Any]],
    reference: dict[str, Any],
) -> None:
    """Require every selected run to name the same validation-cache identity."""

    identity_fields = [
        "tokens_path",
        "dtype",
        "block_size",
        "tokens",
        "tokens_bytes",
        "tokens_sha256",
    ]
    if reference.get("partition") in {"selection", "confirmation"}:
        identity_fields.extend(
            [
                "partition",
                "partition_scheme",
                "partition_seed",
                "source_document_indices_sha256",
            ]
        )
    missing_reference = [field for field in identity_fields if reference.get(field) is None]
    if missing_reference:
        raise ValueError(
            "Reference validation cache is missing required identity fields: "
            + ", ".join(missing_reference)
            + "."
        )
    for manifest in source_manifests:
        candidate = (manifest.get("tokenized_data") or {}).get("validation")
        if candidate is None:
            raise ValueError(f"Source run {manifest['config_id']} has no validation token cache.")
        if candidate.get("partition") != reference.get("partition"):
            raise ValueError(
                "Selected runs do not share the same validation token cache "
                "identity field partition."
            )
        for field in identity_fields:
            if candidate.get(field) != reference[field]:
                raise ValueError(
                    "Selected runs do not share the same validation token cache "
                    f"identity field {field}."
                )
