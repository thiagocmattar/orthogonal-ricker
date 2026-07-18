from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from paper_exp.config import validate_config
from paper_exp.reproducibility import VALIDATION_PARTITION_SCHEME
from paper_exp.reproducibility import validation_document_indices
from paper_exp.reproducibility import validation_document_indices_sha256
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, write_json


def prepare_tokenized_data(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    validate_config(config, allow_todos=False)

    np, load_dataset, auto_tokenizer = _load_data_dependencies()
    experiment_id, numbered_run_id, run_dir = create_run_dir(config, config_path, run_id=run_id)
    cache_dir = tokenized_cache_dir(config, experiment_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_or_write_cache(
        config=config,
        cache_dir=cache_dir,
        split=config["data"]["split"],
        max_documents=config["data"].get("max_documents"),
        np=np,
        load_dataset=load_dataset,
        auto_tokenizer=auto_tokenizer,
    )

    validation_metadata = None
    validation_partitions: dict[str, dict[str, Any]] = {}
    validation_config = config.get("validation", {})
    if validation_config.get("enabled", False):
        partition = validation_config.get("partition")
        partition_seed = validation_config.get("partition_seed")
        partitions = (
            ("selection", "confirmation")
            if partition in {"selection", "confirmation"}
            else (None,)
        )
        for current_partition in partitions:
            document_indices = None
            expected_partition_hash = None
            validation_cache_dir = cache_dir / "validation"
            if current_partition is not None:
                document_indices, generated_partition_hash = validation_document_indices(
                    np,
                    source_documents=int(validation_config["max_documents"]),
                    partition=current_partition,
                    seed=int(partition_seed),
                )
                frozen_partition_hash = (
                    validation_config.get("partition_hash")
                    if current_partition == partition
                    else None
                )
                if (
                    frozen_partition_hash is not None
                    and generated_partition_hash != frozen_partition_hash
                ):
                    raise ValueError(
                        "Generated validation partition does not match the frozen config hash: "
                        f"expected {frozen_partition_hash}, got {generated_partition_hash}."
                    )
                expected_partition_hash = frozen_partition_hash or generated_partition_hash
                validation_cache_dir /= current_partition
            current_metadata = _load_or_write_cache(
                config=config,
                cache_dir=validation_cache_dir,
                split=validation_config["split"],
                max_documents=validation_config.get("max_documents"),
                document_indices=document_indices,
                partition=current_partition,
                partition_seed=partition_seed,
                expected_partition_hash=expected_partition_hash,
                np=np,
                load_dataset=load_dataset,
                auto_tokenizer=auto_tokenizer,
            )
            if current_partition is None or current_partition == partition:
                validation_metadata = current_metadata
            if current_partition is not None:
                validation_partitions[current_partition] = current_metadata

    metrics = {
        "data/documents": metadata["documents"],
        "data/tokens": metadata["tokens"],
        "data/block_size": metadata["block_size"],
        "data/blocks": metadata["tokens"] // metadata["block_size"],
    }
    if validation_metadata is not None:
        metrics.update(
            {
                "validation/documents": validation_metadata["documents"],
                "validation/tokens": validation_metadata["tokens"],
                "validation/blocks": validation_metadata["tokens"] // validation_metadata["block_size"],
            }
        )
    for partition_name, partition_metadata in validation_partitions.items():
        metrics.update(
            {
                f"validation/{partition_name}/documents": partition_metadata["documents"],
                f"validation/{partition_name}/tokens": partition_metadata["tokens"],
                f"validation/{partition_name}/blocks": (
                    partition_metadata["tokens"] // partition_metadata["block_size"]
                ),
            }
        )
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode="prepare-data",
        config_id=experiment_id,
        result_path=run_dir,
    )
    manifest["tokenized_data"] = {
        "train": metadata,
        "validation": validation_metadata,
        "validation_partitions": validation_partitions or None,
    }
    predictions = [
        {
            "event": "tokenized_cache",
            "split": metadata["split"],
            "path": metadata["tokens_path"],
            "documents": metadata["documents"],
            "tokens": metadata["tokens"],
        }
    ]
    if validation_metadata is not None:
        rows = validation_partitions or {"full": validation_metadata}
        for partition_name, partition_metadata in rows.items():
            predictions.append(
                {
                    "event": "tokenized_cache",
                    "split": partition_metadata["split"],
                    "partition": partition_name,
                    "path": partition_metadata["tokens_path"],
                    "documents": partition_metadata["documents"],
                    "tokens": partition_metadata["tokens"],
                    "partition_hash": partition_metadata.get("source_document_indices_sha256"),
                }
            )

    write_run_artifacts(run_dir, config=config, metrics=metrics, manifest=manifest, predictions=predictions)
    return run_dir


def tokenized_cache_dir(config: dict[str, Any], config_id: str) -> Path:
    cache_id = config["preprocessing"].get("cache_id", config_id)
    return Path(config["preprocessing"]["output_dir"]) / cache_id


def validation_metadata_path(config: dict[str, Any], config_id: str) -> Path:
    path = tokenized_cache_dir(config, config_id) / "validation"
    partition = config.get("validation", {}).get("partition")
    if partition in {"selection", "confirmation"}:
        path /= partition
    return path / "metadata.json"


def metadata_matches_config(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    split: str,
    max_documents: int | None,
    partition: str | None = None,
    partition_seed: int | None = None,
) -> bool:
    return _metadata_matches_config(
        metadata,
        config,
        split=split,
        max_documents=max_documents,
        partition=partition,
        partition_seed=partition_seed,
    )


def _load_or_write_cache(
    *,
    config: dict[str, Any],
    cache_dir: Path,
    split: str,
    max_documents: int | None,
    document_indices: Any = None,
    partition: str | None = None,
    partition_seed: int | None = None,
    expected_partition_hash: str | None = None,
    np: Any,
    load_dataset: Any,
    auto_tokenizer: Any,
) -> dict[str, Any]:
    metadata_path = cache_dir / "metadata.json"
    tokens_path = cache_dir / "tokens.int32.bin"
    if tokens_path.exists() and metadata_path.exists() and not config["preprocessing"].get("overwrite", False):
        metadata = _read_metadata(metadata_path)
        if _metadata_matches_config(
            metadata,
            config,
            split=split,
            max_documents=max_documents,
            partition=partition,
            partition_seed=partition_seed,
            document_indices=document_indices,
            expected_partition_hash=expected_partition_hash,
        ):
            return metadata

    metadata = _write_token_cache(
        config=config,
        cache_dir=cache_dir,
        tokens_path=tokens_path,
        split=split,
        max_documents=max_documents,
        document_indices=document_indices,
        partition=partition,
        partition_seed=partition_seed,
        expected_partition_hash=expected_partition_hash,
        np=np,
        load_dataset=load_dataset,
        auto_tokenizer=auto_tokenizer,
    )
    write_json(metadata_path, metadata)
    return metadata


def _write_token_cache(
    *,
    config: dict[str, Any],
    cache_dir: Path,
    tokens_path: Path,
    split: str,
    max_documents: int | None,
    document_indices: Any,
    partition: str | None,
    partition_seed: int | None,
    expected_partition_hash: str | None,
    np: Any,
    load_dataset: Any,
    auto_tokenizer: Any,
) -> dict[str, Any]:
    data_config = config["data"]
    preprocessing = config["preprocessing"]
    tokenizer_config = config["tokenizer"]

    tokenizer = auto_tokenizer.from_pretrained(
        tokenizer_config["name"],
        revision=tokenizer_config.get("revision"),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for MiniPile preprocessing.")

    requested_split = split
    dataset_split = f"{split}[:{max_documents}]" if max_documents else split

    dataset = load_dataset(
        data_config["name"],
        split=dataset_split,
        revision=data_config.get("revision"),
    )
    source_documents = len(dataset)
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    selected_indices: list[int] | None = None
    selected_indices_hash: str | None = None
    if document_indices is not None:
        if partition not in {"selection", "confirmation"} or partition_seed is None:
            raise ValueError("Validation document indices require a named partition and seed.")
        if max_documents is None or source_documents != int(max_documents):
            raise ValueError(
                "Validation source document count does not match the partition contract: "
                f"expected {max_documents}, got {source_documents}."
            )
        selected_indices = [int(index) for index in document_indices]
        selected_indices_hash = validation_document_indices_sha256(
            selected_indices,
            source_documents=source_documents,
            partition=str(partition),
            seed=int(partition_seed),
        )
        frozen_partition_hash = expected_partition_hash or _configured_partition_hash(
            config, partition
        )
        if frozen_partition_hash is not None and selected_indices_hash != frozen_partition_hash:
            raise ValueError(
                "Selected validation document indices do not match the expected partition hash: "
                f"expected {frozen_partition_hash}, got {selected_indices_hash}."
            )
        dataset = dataset.select(selected_indices)
    text_column = data_config.get("text_column", "text")
    append_eos = bool(preprocessing.get("append_eos", True))

    tokens = 0
    documents = 0
    buffer: list[int] = []
    flush_tokens = 1_000_000
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    if tokens_path.exists():
        tokens_path.unlink()

    with tokens_path.open("ab") as handle:
        for row in dataset:
            text = row.get(text_column)
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if append_eos:
                ids.append(tokenizer.eos_token_id)
            buffer.extend(ids)
            tokens += len(ids)
            documents += 1
            if len(buffer) >= flush_tokens:
                np.asarray(buffer, dtype=np.int32).tofile(handle)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=np.int32).tofile(handle)

    metadata = {
        "cache_dir": str(cache_dir),
        "tokens_path": str(tokens_path),
        "dtype": "int32",
        "dataset_name": data_config["name"],
        "dataset_revision": data_config.get("revision"),
        "split": requested_split,
        "text_column": text_column,
        "max_documents": max_documents,
        "documents": documents,
        "tokens": tokens,
        "tokens_bytes": tokens_path.stat().st_size,
        "tokens_sha256": _file_sha256(tokens_path),
        "tokenizer_name": tokenizer_config["name"],
        "tokenizer_revision": tokenizer_config.get("revision"),
        "block_size": preprocessing["block_size"],
        "append_eos": append_eos,
        "dataset_fingerprint": dataset_fingerprint,
    }
    if selected_indices is not None:
        metadata.update(
            {
                "partition": partition,
                "partition_scheme": VALIDATION_PARTITION_SCHEME,
                "partition_seed": int(partition_seed),
                "source_documents": source_documents,
                "source_document_indices": selected_indices,
                "source_document_indices_sha256": selected_indices_hash,
            }
        )
    return metadata


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    import json

    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_matches_config(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    split: str,
    max_documents: int | None,
    partition: str | None = None,
    partition_seed: int | None = None,
    document_indices: Any = None,
    expected_partition_hash: str | None = None,
) -> bool:
    data_config = config["data"]
    preprocessing = config["preprocessing"]
    tokenizer_config = config["tokenizer"]
    expected = {
        "dataset_name": data_config["name"],
        "dataset_revision": data_config.get("revision"),
        "split": split,
        "text_column": data_config.get("text_column", "text"),
        "max_documents": max_documents,
        "tokenizer_name": tokenizer_config["name"],
        "tokenizer_revision": tokenizer_config.get("revision"),
        "block_size": preprocessing["block_size"],
        "append_eos": bool(preprocessing.get("append_eos", True)),
    }
    matches = all(_metadata_value_matches(metadata, key, value) for key, value in expected.items())
    if not matches:
        return False
    if partition in {"selection", "confirmation"}:
        expected_scheme = config.get("validation", {}).get(
            "partition_scheme", VALIDATION_PARTITION_SCHEME
        )
        if partition_seed is None or max_documents is None:
            return False
        if not (
            metadata.get("partition") == partition
            and metadata.get("partition_scheme") == expected_scheme
            and expected_scheme == VALIDATION_PARTITION_SCHEME
            and metadata.get("partition_seed") == int(partition_seed)
            and metadata.get("source_documents") == int(max_documents)
        ):
            return False

        actual_indices = metadata.get("source_document_indices")
        if not isinstance(actual_indices, list):
            return False
        if any(isinstance(index, bool) or not isinstance(index, int) for index in actual_indices):
            return False
        expected_length = (
            int(max_documents) // 2
            if partition == "selection"
            else int(max_documents) - int(max_documents) // 2
        )
        if (
            len(actual_indices) != expected_length
            or len(set(actual_indices)) != len(actual_indices)
            or any(index < 0 or index >= int(max_documents) for index in actual_indices)
        ):
            return False
        if document_indices is not None and actual_indices != [int(index) for index in document_indices]:
            return False

        actual_indices_hash = validation_document_indices_sha256(
            actual_indices,
            source_documents=int(max_documents),
            partition=partition,
            seed=int(partition_seed),
        )
        if metadata.get("source_document_indices_sha256") != actual_indices_hash:
            return False
        frozen_partition_hash = expected_partition_hash or _configured_partition_hash(
            config, partition
        )
        return frozen_partition_hash is None or actual_indices_hash == frozen_partition_hash
    return metadata.get("partition") in {None, "full"}


def _configured_partition_hash(config: dict[str, Any], partition: str | None) -> str | None:
    validation_config = config.get("validation", {})
    if validation_config.get("partition") != partition:
        return None
    value = validation_config.get("partition_hash")
    return str(value) if value is not None else None


def _metadata_value_matches(metadata: dict[str, Any], key: str, value: Any) -> bool:
    if metadata.get(key) == value:
        return True
    requested_key = f"{key}_requested"
    return metadata.get(requested_key) == value


def _load_data_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "MiniPile preparation requires numpy, datasets, and transformers. Run `make install` first."
        ) from exc
    return np, load_dataset, AutoTokenizer
