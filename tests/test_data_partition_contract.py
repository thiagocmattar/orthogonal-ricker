from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import paper_exp.data as data_module
from paper_exp.data import _load_or_write_cache
from paper_exp.data import metadata_matches_config
from paper_exp.reproducibility import VALIDATION_PARTITION_SCHEME
from paper_exp.reproducibility import validation_document_indices
from paper_exp.reproducibility import validation_document_indices_sha256


PARTITION_SEED = 11
SOURCE_DOCUMENTS = 6


class _Dataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows
        self._fingerprint = "offline-test-dataset"

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices: list[int]) -> _Dataset:
        return _Dataset([self._rows[index] for index in indices])


class _DatasetLoader:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, name: str, *, split: str, revision: str) -> _Dataset:
        self.calls += 1
        assert name == "offline/dataset"
        assert split == "validation[:6]"
        assert revision == "dataset-revision"
        return _Dataset([{"text": f"document-{index}"} for index in range(SOURCE_DOCUMENTS)])


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [len(text)]


class _AutoTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def from_pretrained(self, name: str, *, revision: str) -> _Tokenizer:
        self.calls += 1
        assert name == "offline/tokenizer"
        assert revision == "tokenizer-revision"
        return _Tokenizer()


def _config(partition_hash: str) -> dict[str, Any]:
    return {
        "data": {
            "name": "offline/dataset",
            "revision": "dataset-revision",
            "text_column": "text",
        },
        "tokenizer": {
            "name": "offline/tokenizer",
            "revision": "tokenizer-revision",
        },
        "preprocessing": {
            "block_size": 4,
            "append_eos": True,
            "overwrite": False,
        },
        "validation": {
            "partition": "selection",
            "partition_scheme": VALIDATION_PARTITION_SCHEME,
            "partition_seed": PARTITION_SEED,
            "partition_hash": partition_hash,
        },
    }


def _load_partition_cache(
    tmp_path: Path,
    *,
    config: dict[str, Any],
    indices: Any,
    expected_hash: str,
    loader: _DatasetLoader,
    auto_tokenizer: _AutoTokenizer,
) -> dict[str, Any]:
    return _load_or_write_cache(
        config=config,
        cache_dir=tmp_path / "selection",
        split="validation",
        max_documents=SOURCE_DOCUMENTS,
        document_indices=indices,
        partition="selection",
        partition_seed=PARTITION_SEED,
        expected_partition_hash=expected_hash,
        np=np,
        load_dataset=loader,
        auto_tokenizer=auto_tokenizer,
    )


def test_partition_cache_hashes_the_indices_actually_passed_to_dataset_select(
    tmp_path: Path,
) -> None:
    indices, frozen_hash = validation_document_indices(
        np,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )
    config = _config(frozen_hash)
    metadata = _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=_DatasetLoader(),
        auto_tokenizer=_AutoTokenizer(),
    )

    assert metadata["source_document_indices"] == indices.tolist()
    assert metadata["source_document_indices_sha256"] == frozen_hash
    assert metadata_matches_config(
        metadata,
        config,
        split="validation",
        max_documents=SOURCE_DOCUMENTS,
        partition="selection",
        partition_seed=PARTITION_SEED,
    )


def test_partition_cache_rejects_selected_indices_that_do_not_match_frozen_hash(
    tmp_path: Path,
) -> None:
    indices, frozen_hash = validation_document_indices(
        np,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )

    with pytest.raises(ValueError, match="do not match the expected partition hash"):
        _load_partition_cache(
            tmp_path,
            config=_config(frozen_hash),
            indices=indices[::-1],
            expected_hash=frozen_hash,
            loader=_DatasetLoader(),
            auto_tokenizer=_AutoTokenizer(),
        )


def _replace_indices_and_hash(metadata: dict[str, Any]) -> None:
    indices = list(reversed(metadata["source_document_indices"]))
    metadata["source_document_indices"] = indices
    metadata["source_document_indices_sha256"] = validation_document_indices_sha256(
        indices,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(_replace_indices_and_hash, id="actual-indices-and-hash"),
        pytest.param(
            lambda metadata: metadata.__setitem__(
                "source_document_indices_sha256", "0" * 64
            ),
            id="stored-index-hash",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("source_documents", SOURCE_DOCUMENTS + 1),
            id="source-document-count",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("partition_scheme", "wrong-scheme"),
            id="partition-scheme",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("partition_seed", PARTITION_SEED + 1),
            id="partition-seed",
        ),
        pytest.param(
            lambda metadata: metadata.__setitem__("dtype", "float32"),
            id="token-dtype",
        ),
    ],
)
def test_partition_cache_reuse_rejects_corrupt_contract_metadata(
    tmp_path: Path,
    corrupt: Callable[[dict[str, Any]], None],
) -> None:
    indices, frozen_hash = validation_document_indices(
        np,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )
    loader = _DatasetLoader()
    auto_tokenizer = _AutoTokenizer()
    config = _config(frozen_hash)
    _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )

    metadata_path = tmp_path / "selection" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    corrupt(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    repaired = _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )

    assert loader.calls == 2
    assert auto_tokenizer.calls == 2
    assert repaired["source_document_indices"] == indices.tolist()
    assert repaired["source_document_indices_sha256"] == frozen_hash


def test_partition_cache_rebuilds_non_object_metadata(tmp_path: Path) -> None:
    indices, frozen_hash = validation_document_indices(
        np,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )
    loader = _DatasetLoader()
    auto_tokenizer = _AutoTokenizer()
    config = _config(frozen_hash)
    _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )
    (tmp_path / "selection" / "metadata.json").write_text("null\n", encoding="utf-8")

    repaired = _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )

    assert loader.calls == 2
    assert auto_tokenizer.calls == 2
    assert repaired["dtype"] == "int32"


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda payload: payload[:-4], id="truncated"),
        pytest.param(
            lambda payload: bytes([payload[0] ^ 1]) + payload[1:],
            id="same-size-content-change",
        ),
    ],
)
def test_partition_cache_reuse_rejects_corrupt_token_file(
    tmp_path: Path,
    corrupt: Callable[[bytes], bytes],
) -> None:
    indices, frozen_hash = validation_document_indices(
        np,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )
    loader = _DatasetLoader()
    auto_tokenizer = _AutoTokenizer()
    config = _config(frozen_hash)
    _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )
    tokens_path = tmp_path / "selection" / "tokens.int32.bin"
    tokens_path.write_bytes(corrupt(tokens_path.read_bytes()))

    repaired = _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )

    assert loader.calls == 2
    assert auto_tokenizer.calls == 2
    assert repaired["tokens_bytes"] == tokens_path.stat().st_size
    assert repaired["tokens_sha256"] == hashlib.sha256(tokens_path.read_bytes()).hexdigest()


def test_interrupted_cache_rebuild_preserves_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indices, frozen_hash = validation_document_indices(
        np,
        source_documents=SOURCE_DOCUMENTS,
        partition="selection",
        seed=PARTITION_SEED,
    )
    loader = _DatasetLoader()
    auto_tokenizer = _AutoTokenizer()
    config = _config(frozen_hash)
    _load_partition_cache(
        tmp_path,
        config=config,
        indices=indices,
        expected_hash=frozen_hash,
        loader=loader,
        auto_tokenizer=auto_tokenizer,
    )
    cache_dir = tmp_path / "selection"
    tokens_path = cache_dir / "tokens.int32.bin"
    metadata_path = cache_dir / "metadata.json"
    published_tokens = tokens_path.read_bytes()
    published_metadata = metadata_path.read_bytes()
    config["preprocessing"]["overwrite"] = True

    def interrupt_write(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["tokens_path"]).write_bytes(b"partial-token-cache")
        raise RuntimeError("tokenization interrupted")

    monkeypatch.setattr(data_module, "_write_token_cache", interrupt_write)

    with pytest.raises(RuntimeError, match="tokenization interrupted"):
        _load_partition_cache(
            tmp_path,
            config=config,
            indices=indices,
            expected_hash=frozen_hash,
            loader=loader,
            auto_tokenizer=auto_tokenizer,
        )

    assert tokens_path.read_bytes() == published_tokens
    assert metadata_path.read_bytes() == published_metadata
    assert not list(cache_dir.glob(".*.tmp"))
