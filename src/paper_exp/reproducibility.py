from __future__ import annotations

import hashlib
import json
import struct
from typing import Any


TRAINING_SCHEDULE_SCHEME = "seeded_complete_block_permutation_wrap_v1"
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
    metadata = training_schedule_metadata(
        token_count=token_count,
        block_size=block_size,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        micro_batch_size=micro_batch_size,
        seed=seed,
    )
    complete_blocks = metadata["complete_blocks"]
    scheduled_blocks = metadata["scheduled_blocks"]
    permutation = np.random.default_rng(int(seed)).permutation(complete_blocks).astype(
        np.int64,
        copy=False,
    )
    order = np.resize(permutation, scheduled_blocks)
    starts = order * int(block_size)
    shape = (
        metadata["max_steps"],
        metadata["gradient_accumulation_steps"],
        metadata["micro_batch_size"],
    )
    schedule = starts.reshape(shape)
    return schedule, _hash_integer_array(np, schedule, metadata)


def training_schedule_metadata(
    *,
    token_count: int,
    block_size: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, int | str]:
    """Describe the exact complete-block schedule inputs and wrap coverage."""

    token_count = int(token_count)
    block_size = int(block_size)
    max_steps = int(max_steps)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    micro_batch_size = int(micro_batch_size)
    if token_count <= 0 or block_size <= 0:
        raise ValueError("Training token count and block size must be positive.")
    if max_steps <= 0 or gradient_accumulation_steps <= 0 or micro_batch_size <= 0:
        raise ValueError("Training schedule dimensions must be positive.")

    complete_blocks, excluded_tail_tokens = divmod(token_count, block_size)
    if complete_blocks <= 0:
        raise ValueError("Token cache has no complete block for the configured block size.")
    sequences_per_update = gradient_accumulation_steps * micro_batch_size
    scheduled_blocks = max_steps * sequences_per_update
    return {
        "scheme": TRAINING_SCHEDULE_SCHEME,
        "seed": int(seed),
        "token_count": token_count,
        "block_size": block_size,
        "complete_blocks": complete_blocks,
        "excluded_tail_tokens": excluded_tail_tokens,
        "max_steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "micro_batch_size": micro_batch_size,
        "sequences_per_update": sequences_per_update,
        "scheduled_blocks": scheduled_blocks,
        "wrapped_blocks": max(0, scheduled_blocks - complete_blocks),
    }


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
