from __future__ import annotations

import hashlib
import json
import struct
from typing import Any


TRAINING_SCHEDULE_SCHEME = "random_contiguous_blocks_with_replacement_v1"
VALIDATION_PARTITION_SCHEME = "shuffled_source_documents_half_v1"


def build_training_schedule(
    np: Any,
    *,
    token_count: int,
    block_size: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    micro_batch_size: int,
    seed: int,
) -> tuple[Any, str]:
    max_start = int(token_count) - int(block_size) - 1
    if max_start <= 0:
        raise ValueError("Token cache is too small for the configured block size.")
    shape = (
        int(max_steps),
        int(gradient_accumulation_steps),
        int(micro_batch_size),
    )
    if any(value <= 0 for value in shape):
        raise ValueError("Training schedule dimensions must be positive.")

    schedule = np.random.default_rng(int(seed)).integers(
        0,
        max_start,
        size=shape,
        dtype=np.int64,
    )
    metadata = {
        "scheme": TRAINING_SCHEDULE_SCHEME,
        "seed": int(seed),
        "token_count": int(token_count),
        "block_size": int(block_size),
        "max_steps": shape[0],
        "gradient_accumulation_steps": shape[1],
        "micro_batch_size": shape[2],
    }
    return schedule, _hash_integer_array(np, schedule, metadata)


def validation_document_indices(
    np: Any,
    *,
    source_documents: int,
    partition: str,
    seed: int,
) -> tuple[Any, str]:
    source_documents = int(source_documents)
    if source_documents < 2:
        raise ValueError("Document-disjoint validation requires at least two source documents.")
    if partition not in {"selection", "confirmation"}:
        raise ValueError("Validation partition must be 'selection' or 'confirmation'.")

    order = np.random.default_rng(int(seed)).permutation(source_documents).astype(np.int64, copy=False)
    split = source_documents // 2
    indices = order[:split] if partition == "selection" else order[split:]
    return indices, validation_document_indices_sha256(
        indices,
        source_documents=source_documents,
        partition=partition,
        seed=seed,
    )


def validation_document_indices_sha256(
    indices: Any,
    *,
    source_documents: int,
    partition: str,
    seed: int,
) -> str:
    """Hash the exact ordered validation-document indices and their partition contract."""

    metadata = {
        "scheme": VALIDATION_PARTITION_SCHEME,
        "partition": str(partition),
        "seed": int(seed),
        "source_documents": int(source_documents),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    for index in indices:
        digest.update(struct.pack("<q", int(index)))
    return digest.hexdigest()


def _hash_integer_array(np: Any, values: Any, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()
