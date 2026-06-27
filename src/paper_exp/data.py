from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_exp.config import validate_config
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
    validation_config = config.get("validation", {})
    if validation_config.get("enabled", False):
        validation_metadata = _load_or_write_cache(
            config=config,
            cache_dir=cache_dir / "validation",
            split=validation_config["split"],
            max_documents=validation_config.get("max_documents"),
            np=np,
            load_dataset=load_dataset,
            auto_tokenizer=auto_tokenizer,
        )

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
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode="prepare-data",
        config_id=experiment_id,
        result_path=run_dir,
    )
    manifest["tokenized_data"] = {"train": metadata, "validation": validation_metadata}
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
        predictions.append(
            {
                "event": "tokenized_cache",
                "split": validation_metadata["split"],
                "path": validation_metadata["tokens_path"],
                "documents": validation_metadata["documents"],
                "tokens": validation_metadata["tokens"],
            }
        )

    write_run_artifacts(run_dir, config=config, metrics=metrics, manifest=manifest, predictions=predictions)
    return run_dir


def tokenized_cache_dir(config: dict[str, Any], config_id: str) -> Path:
    return Path(config["preprocessing"]["output_dir"]) / config_id


def validation_metadata_path(config: dict[str, Any], config_id: str) -> Path:
    return tokenized_cache_dir(config, config_id) / "validation" / "metadata.json"


def _load_or_write_cache(
    *,
    config: dict[str, Any],
    cache_dir: Path,
    split: str,
    max_documents: int | None,
    np: Any,
    load_dataset: Any,
    auto_tokenizer: Any,
) -> dict[str, Any]:
    metadata_path = cache_dir / "metadata.json"
    tokens_path = cache_dir / "tokens.int32.bin"
    if tokens_path.exists() and metadata_path.exists() and not config["preprocessing"].get("overwrite", False):
        metadata = _read_metadata(metadata_path)
        if _metadata_matches_config(metadata, config, split=split, max_documents=max_documents):
            return metadata

    metadata = _write_token_cache(
        config=config,
        cache_dir=cache_dir,
        tokens_path=tokens_path,
        split=split,
        max_documents=max_documents,
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

    return {
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
        "tokenizer_name": tokenizer_config["name"],
        "tokenizer_revision": tokenizer_config.get("revision"),
        "block_size": preprocessing["block_size"],
        "append_eos": append_eos,
    }


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    import json

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata_matches_config(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    split: str,
    max_documents: int | None,
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
    return all(metadata.get(key) == value for key, value in expected.items())


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
