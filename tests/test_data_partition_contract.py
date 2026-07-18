from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

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
        assert name == "offline/minipile"
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
            "name": "offline/minipile",
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
