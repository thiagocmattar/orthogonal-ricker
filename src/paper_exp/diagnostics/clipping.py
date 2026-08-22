from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from paper_exp.activations import SUPPORTED_SITE_ALIASES, resolve_site_aliases
from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import CORE_RUN_ARTIFACTS, RunHandle, complete_run, run_lifecycle
from paper_exp.utils import read_json, write_jsonl

from . import clipping_evaluation as _evaluation
from .evaluation import eval_starts, select_device, select_dtype


def run_clipping_sweep(
    *,
    checkpoint_run_dir: str | Path,
    command: str,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float] | None = None,
    sites: list[str] | None = None,
    experiment_suffix: str | None = None,
    eval_batches: int | None,
    measure_zero_products: bool = False,
    seed: int = 0,
    run_id: str | None = None,
) -> Path:
    run_path = Path(checkpoint_run_dir)
    source_config_path = run_path / "config.yaml"
    source_manifest_path = run_path / "manifest.json"
    if not source_config_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint run must contain config.yaml and manifest.json: {run_path}")

    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
    source_manifest = read_json(source_manifest_path)
    checkpoint_path = _validate_clipping_source(run_path, source_manifest)
    checkpoint_identity = _checkpoint_content_identity(checkpoint_path)
    validation_cache, validation_tokens_path = _clipping_validation_cache_identity(
        source_manifest,
        source_run=run_path,
        validation_config=config.get("validation", {}),
    )
    config["experiment_name"] = f"{config['experiment_name']}_clipping_sweep"
    rms_values = list(rms_multipliers or [])
    _validate_clipping_arguments(
        thresholds=thresholds,
        quantiles=quantiles,
        rms_multipliers=rms_values,
        eval_batches=eval_batches,
        experiment_suffix=experiment_suffix,
        seed=seed,
    )
    threshold_values = [float(value) for value in thresholds]
    quantile_values = [float(value) for value in quantiles]
    rms_values = [float(value) for value in rms_values]
    if sites is not None:
        config.setdefault("activation_clipping", {})["sites"] = list(sites)
    clipping_sites = _clipping_sites(config)
    suffix = experiment_suffix or _site_suffix(clipping_sites)
    sweep_spec = {
        "source": {
            "config_id": str(source_manifest["config_id"]),
            "run_id": str(source_manifest["run_id"]),
            "run_path": _portable_path(run_path),
            "checkpoint_path": _portable_path(checkpoint_path),
            "checkpoint_content": checkpoint_identity,
            "validation_cache": validation_cache,
        },
        "thresholds": threshold_values,
        "quantiles": quantile_values,
        "rms_multipliers": rms_values,
        "sites": clipping_sites,
        "eval_batches": eval_batches,
        "measure_zero_products": bool(measure_zero_products),
        "evaluation_seed": seed,
        "experiment_suffix": experiment_suffix,
        "effective_suffix": suffix,
    }
    config.setdefault("activation_clipping", {})["sweep"] = sweep_spec
    digest_spec = {
        **sweep_spec,
        "source": {
            "config_id": sweep_spec["source"]["config_id"],
            "run_id": sweep_spec["source"]["run_id"],
            "checkpoint_content": checkpoint_identity,
            "validation_cache": {
                key: value
                for key, value in validation_cache.items()
                if key != "tokens_path"
            },
        },
    }
    sweep_digest = hashlib.sha256(
        json.dumps(
            digest_spec,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:10]
    sweep_config_path = f"{source_manifest['config_id']}-clip-{sweep_digest}.yaml"
    with run_lifecycle(
        config,
        config_path=sweep_config_path,
        command=command,
        mode="clip-sweep",
        run_id=run_id,
    ) as run:
        return _run_clipping_sweep(
            run,
            source_run=run_path,
            source_manifest=source_manifest,
            checkpoint_path=checkpoint_path,
            checkpoint_identity=checkpoint_identity,
            validation_cache=validation_cache,
            validation_tokens_path=validation_tokens_path,
            thresholds=threshold_values,
            quantiles=quantile_values,
            rms_multipliers=rms_values,
            suffix=suffix,
            eval_batches=eval_batches,
            measure_zero_products=measure_zero_products,
            seed=seed,
        )


def _run_clipping_sweep(
    run: RunHandle,
    *,
    source_run: Path,
    source_manifest: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_identity: dict[str, Any],
    validation_cache: dict[str, Any],
    validation_tokens_path: Path,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float],
    suffix: str | None,
    eval_batches: int | None,
    measure_zero_products: bool,
    seed: int,
) -> Path:
    config = run.config

    torch, np, auto_model, modeling_gpt_neox = _evaluation._load_clipping_dependencies()
    np.random.seed(seed)
    training = config["training"]
    device = select_device(torch, training.get("device", "auto"))
    dtype = select_dtype(
        torch, device, training.get("precision", "auto")
    )

    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    validation_tokens = np.memmap(validation_tokens_path, dtype=np.int32, mode="r")
    block_size = int(validation_cache["block_size"])
    batch_size = int(config["validation"]["batch_size"])
    starts = eval_starts(
        validation_tokens,
        block_size,
        eval_batches=eval_batches,
        batch_size=batch_size,
        np=np,
    )

    output_dir = run.run_dir
    rows: list[dict[str, Any]] = []
    for clipping_cfg in _clipping_configs(
        config,
        thresholds=thresholds,
        quantiles=quantiles,
        rms_multipliers=rms_multipliers,
    ):
        result = _evaluation._evaluate_clipped_loss(
            model=model,
            torch=torch,
            np=np,
            tokens=validation_tokens,
            block_size=block_size,
            batch_size=batch_size,
            eval_batches=eval_batches,
            starts=starts,
            device=device,
            dtype=dtype,
            clipping_cfg=clipping_cfg,
            measure_zero_products=measure_zero_products,
            modeling_gpt_neox=modeling_gpt_neox,
        )
        rows.append(result)

    best_loss = min(rows, key=lambda row: row["validation_loss"]) if rows else None
    if _checkpoint_content_identity(checkpoint_path) != checkpoint_identity:
        raise RuntimeError("Clipping source checkpoint changed during evaluation.")
    if (
        validation_tokens_path.stat().st_size != validation_cache["tokens_bytes"]
        or _file_sha256(validation_tokens_path) != validation_cache["tokens_sha256"]
    ):
        raise RuntimeError("Clipping source validation cache changed during evaluation.")
    metrics = {
        "clipping/num_points": len(rows),
        "clipping/best_validation_loss": best_loss["validation_loss"] if best_loss else None,
        "clipping/best_achieved_sparsity": best_loss["achieved_sparsity"] if best_loss else None,
        "clipping/max_potentially_avoidable_model_matmul_fraction": (
            max(
                (
                    float(row["potentially_avoidable_model_matmul_fraction"])
                    for row in rows
                    if row.get("potentially_avoidable_model_matmul_fraction") is not None
                ),
                default=None,
            )
        ),
    }
    manifest = {
        "source_run": _portable_path(source_run),
        "source_checkpoint": _portable_path(checkpoint_path),
        "source_checkpoint_content": checkpoint_identity,
        "source_validation_cache": validation_cache,
        "thresholds": thresholds,
        "quantiles": quantiles,
        "rms_multipliers": rms_multipliers,
        "clipping_sites": _clipping_sites(config),
    }
    if suffix:
        manifest["clipping_sweep_suffix"] = suffix
    if rms_multipliers:
        manifest["rms_threshold_semantics"] = (
            "For each captured activation tensor and forward pass, clip entries with "
            "|a| <= rms_multiplier * RMS(A), where RMS(A) is computed over that tensor."
        )
    manifest["eval_batches"] = eval_batches
    manifest["clipping_seed"] = seed
    manifest["measure_zero_products"] = bool(measure_zero_products)
    if measure_zero_products:
        manifest["logical_zero_products"] = {
            "exact_zero_definition": "A tensor coordinate is zero iff value == 0 with no tolerance.",
            "block_denominator": (
                "QKV, valid-causal QK, valid-causal PV, attention output, MLP up, and MLP down "
                "logical scalar products across every transformer block."
            ),
            "model_denominator": (
                "The block denominator plus the final hidden-to-vocabulary LM-head products."
            ),
            "interpretation": (
                "Potentially avoidable logical products, not measured dense-kernel speedup. "
                "PRE-RoPE Q/K clipping is credited only after standard RoPE, using the actual QK operands."
            ),
        }
    if "sweep" in source_manifest:
        manifest["source_sweep"] = source_manifest["sweep"]

    write_jsonl(output_dir / "clipping_frontier.jsonl", rows)
    return complete_run(
        run,
        metrics=metrics,
        predictions=rows,
        manifest_updates=manifest,
    )


def _clipping_configs(
    config: dict[str, Any],
    *,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float],
) -> list[dict[str, Any]]:
    sites = _clipping_sites(config)
    configs = []
    for threshold in thresholds:
        configs.append({"enabled": True, "mode": "threshold", "sites": sites, "threshold": threshold})
    for quantile in quantiles:
        configs.append({"enabled": True, "mode": "quantile", "sites": sites, "quantile": quantile})
    for multiplier in rms_multipliers:
        configs.append(
            {
                "enabled": True,
                "mode": "rms_threshold",
                "sites": sites,
                "rms_multiplier": multiplier,
            }
        )
    return configs


def _clipping_sites(config: dict[str, Any]) -> list[str]:
    sites = config.get("activation_clipping", {}).get("sites")
    if not isinstance(sites, list) or not sites or any(
        not isinstance(site, str) or not site.strip() for site in sites
    ):
        raise ValueError(
            "Clipping sites must be an explicit non-empty activation_clipping.sites list "
            "or --sites argument."
        )
    return list(resolve_site_aliases(sites))


def _validate_clipping_arguments(
    *,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float],
    eval_batches: int | None,
    experiment_suffix: str | None,
    seed: int,
) -> None:
    if not thresholds and not quantiles and not rms_multipliers:
        raise ValueError(
            "Clipping sweep requires at least one explicit threshold, quantile, or RMS multiplier."
        )
    for field, values, lower, upper in (
        ("thresholds", thresholds, 0.0, None),
        ("quantiles", quantiles, 0.0, 1.0),
        ("rms_multipliers", rms_multipliers, 0.0, None),
    ):
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"Clipping {field} must contain only finite numbers.")
            numeric = float(value)
            if (
                not math.isfinite(numeric)
                or numeric < lower
                or (upper is not None and numeric > upper)
            ):
                bounds = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
                raise ValueError(f"Clipping {field} values must be finite and {bounds}.")
        if len({float(value) for value in values}) != len(values):
            raise ValueError(f"Clipping {field} must not contain duplicates.")
    if eval_batches is not None and (
        isinstance(eval_batches, bool) or not isinstance(eval_batches, int) or eval_batches <= 0
    ):
        raise ValueError("Clipping eval_batches must be a positive integer when provided.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Clipping seed must be an integer.")
    if experiment_suffix is not None and re.fullmatch(
        r"[a-z0-9][a-z0-9-]*", experiment_suffix
    ) is None:
        raise ValueError(
            "Clipping experiment_suffix must contain only lowercase letters, digits, and hyphens."
        )


def _site_suffix(sites: list[str] | None) -> str | None:
    if not sites:
        return None
    if set(sites) == SUPPORTED_SITE_ALIASES:
        return "all-sites"
    return "sites-" + "-".join(site.replace("_", "-") for site in sites)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validate_clipping_source(run_path: Path, manifest: Any) -> Path:
    missing = [name for name in CORE_RUN_ARTIFACTS if not (run_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Clipping source run is missing required artifacts ({', '.join(missing)}): "
            f"{run_path}"
        )
    if not isinstance(manifest, dict):
        raise ValueError(f"Clipping source manifest is not an object: {run_path}")
    config_id = str(manifest.get("config_id") or "").strip()
    run_id = str(manifest.get("run_id") or "").strip()
    if not config_id or not run_id:
        raise ValueError(f"Clipping source manifest has no complete identity: {run_path}")
    if run_path.name != run_id or run_path.parent.name != config_id:
        raise ValueError(f"Clipping source manifest identity does not match its path: {run_path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"Clipping source run is not completed: {run_path}")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("saved") is not True:
        raise ValueError(f"Clipping source run has no saved checkpoint: {run_path}")
    checkpoint_path = _resolve_source_path(checkpoint.get("path"), source_run=run_path)
    model_files = (
        checkpoint_path / "model.safetensors",
        checkpoint_path / "model.safetensors.index.json",
    )
    if not checkpoint_path.is_dir() or not any(path.is_file() for path in model_files):
        raise FileNotFoundError(f"Clipping source checkpoint is incomplete: {checkpoint_path}")
    return checkpoint_path


def _checkpoint_content_identity(checkpoint_path: Path) -> dict[str, Any]:
    files = sorted(
        (path for path in checkpoint_path.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(checkpoint_path).as_posix(),
    )
    if not files:
        raise FileNotFoundError(f"Clipping source checkpoint is empty: {checkpoint_path}")
    return {
        "files": [
            {
                "path": path.relative_to(checkpoint_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in files
        ]
    }


def _clipping_validation_cache_identity(
    manifest: dict[str, Any],
    *,
    source_run: Path,
    validation_config: Any,
) -> tuple[dict[str, Any], Path]:
    metadata = (manifest.get("tokenized_data") or {}).get("validation")
    if not isinstance(metadata, dict):
        raise ValueError("Clipping source run has no validation token cache in its manifest.")
    recorded_path = str(metadata.get("tokens_path") or "").strip()
    if not recorded_path:
        raise ValueError("Clipping source validation cache has no tokens_path.")
    tokens_path = _resolve_validation_path(recorded_path, source_run=source_run)

    tokens = metadata.get("tokens")
    tokens_bytes = metadata.get("tokens_bytes")
    tokens_sha256 = metadata.get("tokens_sha256")
    block_size = metadata.get("block_size")
    if (
        metadata.get("dtype") != "int32"
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens < 0
        or isinstance(tokens_bytes, bool)
        or not isinstance(tokens_bytes, int)
        or tokens_bytes != tokens * 4
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
        or not isinstance(tokens_sha256, str)
        or len(tokens_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tokens_sha256)
    ):
        raise ValueError("Clipping source validation cache identity is incomplete or invalid.")
    if not tokens_path.is_file() or tokens_path.stat().st_size != tokens_bytes:
        raise ValueError(
            "Clipping source validation cache size does not match its manifest."
        )
    if _file_sha256(tokens_path) != tokens_sha256:
        raise ValueError(
            "Clipping source validation cache hash does not match its manifest."
        )

    requested = validation_config if isinstance(validation_config, dict) else {}
    requested_partition = requested.get("partition")
    if requested_partition is not None and metadata.get("partition") != requested_partition:
        raise ValueError(
            "Clipping source validation partition does not match its saved config."
        )
    requested_hash = requested.get("partition_hash")
    if (
        requested_hash is not None
        and metadata.get("source_document_indices_sha256") != requested_hash
    ):
        raise ValueError(
            "Clipping source validation partition hash does not match its saved config."
        )

    identity_fields = (
        "dtype",
        "tokens",
        "tokens_bytes",
        "tokens_sha256",
        "block_size",
        "split",
        "max_documents",
        "partition",
        "partition_scheme",
        "partition_seed",
        "source_document_indices_sha256",
    )
    identity = {
        "tokens_path": _portable_path(tokens_path),
        **{field: metadata.get(field) for field in identity_fields},
    }
    return identity, tokens_path


def _resolve_validation_path(value: str, *, source_run: Path) -> Path:
    recorded = Path(value)
    if recorded.is_absolute():
        return recorded.resolve()
    repository_path = (Path.cwd() / recorded).resolve()
    run_relative_path = (source_run / recorded).resolve()
    if repository_path.exists() or not run_relative_path.exists():
        return repository_path
    return run_relative_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_path(value: Any, *, source_run: Path) -> Path:
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
