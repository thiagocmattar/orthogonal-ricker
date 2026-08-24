"""Infrastructure-only physical-batch profiling contracts.

This module deliberately contains no CUDA, Torch, model, config, or launch
code.  A later subprocess worker can consume :class:`HardwareProfileWorkItem`
values and return :class:`ProfileRepeatResult` values.  The pure validation,
selection, and artifact functions here keep hardware tuning separate from
scientific evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from statistics import median
from typing import Any, Protocol


SEQUENCE_LENGTH = 2_048
GLOBAL_SEQUENCES = 128
MIN_SUCCESSFUL_REPEATS = 2
MAX_RESERVED_VRAM_FRACTION = 0.90
THROUGHPUT_TIE_FRACTION = 0.02

PROHIBITED_USE = (
    "Infrastructure-only hardware profiling. This artifact must not be used "
    "to select or support claims about loss, quality, sparsity, optimization, "
    "or model behavior."
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOATING_REVISIONS = frozenset({"main", "master", "head", "latest"})
_SCIENTIFIC_KEY_TOKENS = frozenset(
    {
        "accuracy",
        "activation",
        "auc",
        "bleu",
        "f1",
        "logit",
        "logits",
        "loss",
        "metric",
        "metrics",
        "perplexity",
        "ppl",
        "prediction",
        "predictions",
        "quality",
        "reward",
        "rouge",
        "sparsity",
        "zero",
    }
)
_SCIENTIFIC_KEY_EXEMPTIONS = frozenset({"scientific_evidence"})


@dataclass(frozen=True)
class MicrobatchCandidate:
    """One physical batch decomposition preserving the global batch."""

    microbatch_sequences: int
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        _require_positive_int(self.microbatch_sequences, "microbatch_sequences")
        _require_positive_int(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        if self.microbatch_sequences * self.gradient_accumulation_steps != GLOBAL_SEQUENCES:
            raise ValueError(
                "microbatch_sequences * gradient_accumulation_steps must equal "
                f"{GLOBAL_SEQUENCES}."
            )

    @classmethod
    def from_microbatch(cls, microbatch_sequences: int) -> MicrobatchCandidate:
        """Derive accumulation for one valid physical microbatch."""

        _require_positive_int(microbatch_sequences, "microbatch_sequences")
        if GLOBAL_SEQUENCES % microbatch_sequences:
            raise ValueError(
                f"microbatch_sequences must divide {GLOBAL_SEQUENCES} exactly."
            )
        return cls(
            microbatch_sequences=microbatch_sequences,
            gradient_accumulation_steps=GLOBAL_SEQUENCES // microbatch_sequences,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "microbatch_sequences": self.microbatch_sequences,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }


@dataclass(frozen=True)
class HardwareProfileRequest:
    """A pinned, infrastructure-only sweep for one GPU class."""

    architecture: str
    revision: str
    gpu_class: str
    candidate_microbatches: tuple[int, ...]
    repeats: int = MIN_SUCCESSFUL_REPEATS
    sequence_length: int = SEQUENCE_LENGTH
    global_sequences: int = GLOBAL_SEQUENCES

    def __post_init__(self) -> None:
        _require_nonempty_string(self.architecture, "architecture")
        _require_nonempty_string(self.revision, "revision")
        _require_nonempty_string(self.gpu_class, "gpu_class")
        if self.revision.strip().lower() in _FLOATING_REVISIONS:
            raise ValueError("revision must be pinned, not a floating revision name.")
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length must be {SEQUENCE_LENGTH}.")
        if self.global_sequences != GLOBAL_SEQUENCES:
            raise ValueError(f"global_sequences must be {GLOBAL_SEQUENCES}.")
        _require_positive_int(self.repeats, "repeats")
        if self.repeats < MIN_SUCCESSFUL_REPEATS:
            raise ValueError(
                f"repeats must be at least {MIN_SUCCESSFUL_REPEATS}."
            )
        if not isinstance(self.candidate_microbatches, Sequence) or isinstance(
            self.candidate_microbatches,
            (str, bytes),
        ):
            raise TypeError("candidate_microbatches must be an ordered integer sequence.")
        candidates = tuple(self.candidate_microbatches)
        if not candidates:
            raise ValueError("candidate_microbatches must not be empty.")
        for value in candidates:
            MicrobatchCandidate.from_microbatch(value)
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_microbatches must be distinct.")
        object.__setattr__(self, "candidate_microbatches", candidates)

    @property
    def candidates(self) -> tuple[MicrobatchCandidate, ...]:
        return tuple(
            MicrobatchCandidate.from_microbatch(value)
            for value in self.candidate_microbatches
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "revision": self.revision,
            "gpu_class": self.gpu_class,
            "sequence_length": self.sequence_length,
            "global_sequences": self.global_sequences,
            "candidate_microbatches": list(self.candidate_microbatches),
            "repeats": self.repeats,
        }


@dataclass(frozen=True)
class HardwareProfileWorkItem:
    """Serializable input boundary for a later isolated CUDA worker."""

    microbatch_sequences: int
    gradient_accumulation_steps: int
    repeat_index: int
    synthetic_grouping_hash: str

    def __post_init__(self) -> None:
        MicrobatchCandidate(
            self.microbatch_sequences,
            self.gradient_accumulation_steps,
        )
        _require_positive_int(self.repeat_index, "repeat_index")
        if _HASH_RE.fullmatch(self.synthetic_grouping_hash) is None:
            raise ValueError("synthetic_grouping_hash must be a lowercase SHA-256 digest.")

    def as_dict(self) -> dict[str, object]:
        return {
            "microbatch_sequences": self.microbatch_sequences,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "repeat_index": self.repeat_index,
            "synthetic_grouping_hash": self.synthetic_grouping_hash,
        }


class HardwareProfileWorker(Protocol):
    """Callable boundary implemented later by an isolated CUDA subprocess."""

    def __call__(
        self,
        request: HardwareProfileRequest,
        work_item: HardwareProfileWorkItem,
    ) -> ProfileRepeatResult: ...


@dataclass(frozen=True)
class ProfileRepeatResult:
    """Operational measurements from one synchronized worker repeat."""

    microbatch_sequences: int
    repeat_index: int
    fit: bool
    error: str | None
    synchronized_seconds: float | None
    tokens_per_second: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    total_vram_bytes: int | None

    def __post_init__(self) -> None:
        _require_positive_int(self.microbatch_sequences, "microbatch_sequences")
        if GLOBAL_SEQUENCES % self.microbatch_sequences:
            raise ValueError(
                f"microbatch_sequences must divide {GLOBAL_SEQUENCES} exactly."
            )
        _require_positive_int(self.repeat_index, "repeat_index")
        if not isinstance(self.fit, bool):
            raise TypeError("fit must be a bool.")

        timing_values = (self.synchronized_seconds, self.tokens_per_second)
        memory_values = (
            self.peak_allocated_bytes,
            self.peak_reserved_bytes,
            self.total_vram_bytes,
        )
        if self.fit:
            if self.error is not None:
                raise ValueError("A fitting repeat cannot contain an error.")
            if any(value is None for value in timing_values + memory_values):
                raise ValueError("A fitting repeat requires all operational measurements.")
        else:
            _require_nonempty_string(self.error, "error")
            if any(value is not None for value in timing_values):
                raise ValueError(
                    "A failed repeat cannot publish timing or throughput measurements."
                )
            if any(value is not None for value in memory_values) and any(
                value is None for value in memory_values
            ):
                raise ValueError(
                    "A failed repeat must publish either all or no memory measurements."
                )

        if self.synchronized_seconds is not None:
            _require_positive_finite(
                self.synchronized_seconds,
                "synchronized_seconds",
            )
        if self.tokens_per_second is not None:
            _require_positive_finite(self.tokens_per_second, "tokens_per_second")
        if all(value is not None for value in memory_values):
            allocated = _require_nonnegative_int(
                self.peak_allocated_bytes,
                "peak_allocated_bytes",
            )
            reserved = _require_nonnegative_int(
                self.peak_reserved_bytes,
                "peak_reserved_bytes",
            )
            total = _require_positive_int(self.total_vram_bytes, "total_vram_bytes")
            if allocated > reserved:
                raise ValueError("peak_allocated_bytes cannot exceed peak_reserved_bytes.")
            if reserved > total:
                raise ValueError("peak_reserved_bytes cannot exceed total_vram_bytes.")

    def as_dict(self) -> dict[str, object]:
        return {
            "microbatch_sequences": self.microbatch_sequences,
            "repeat_index": self.repeat_index,
            "fit": self.fit,
            "error": self.error,
            "synchronized_seconds": self.synchronized_seconds,
            "tokens_per_second": self.tokens_per_second,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "total_vram_bytes": self.total_vram_bytes,
        }


@dataclass(frozen=True)
class CandidateProfileResult:
    """All operational repeats for one physical batch candidate."""

    microbatch_sequences: int
    gradient_accumulation_steps: int
    repeats: tuple[ProfileRepeatResult, ...]

    def __post_init__(self) -> None:
        MicrobatchCandidate(
            self.microbatch_sequences,
            self.gradient_accumulation_steps,
        )
        if not isinstance(self.repeats, Sequence) or isinstance(
            self.repeats,
            (str, bytes),
        ):
            raise TypeError("repeats must be an ordered result sequence.")
        repeats = tuple(self.repeats)
        if not repeats:
            raise ValueError("A candidate result must contain at least one repeat.")
        indexes: list[int] = []
        for result in repeats:
            if not isinstance(result, ProfileRepeatResult):
                raise TypeError("repeats must contain ProfileRepeatResult values.")
            if result.microbatch_sequences != self.microbatch_sequences:
                raise ValueError("Repeat microbatch does not match its candidate.")
            indexes.append(result.repeat_index)
        if len(set(indexes)) != len(indexes):
            raise ValueError("Repeat indexes must be distinct within a candidate.")
        object.__setattr__(
            self,
            "repeats",
            tuple(sorted(repeats, key=lambda result: result.repeat_index)),
        )

    @property
    def successful_repeats(self) -> tuple[ProfileRepeatResult, ...]:
        return tuple(result for result in self.repeats if result.fit)

    def as_dict(self) -> dict[str, object]:
        return {
            "microbatch_sequences": self.microbatch_sequences,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "repeats": [result.as_dict() for result in self.repeats],
        }


@dataclass(frozen=True)
class ProfileSelection:
    """Deterministic operational winner of a complete profile sweep."""

    microbatch_sequences: int
    gradient_accumulation_steps: int
    successful_repeats: int
    median_tokens_per_second: float
    peak_reserved_bytes: int
    total_vram_bytes: int

    def __post_init__(self) -> None:
        MicrobatchCandidate(
            self.microbatch_sequences,
            self.gradient_accumulation_steps,
        )
        _require_positive_int(self.successful_repeats, "successful_repeats")
        if self.successful_repeats < MIN_SUCCESSFUL_REPEATS:
            raise ValueError(
                f"A selection requires at least {MIN_SUCCESSFUL_REPEATS} successful repeats."
            )
        _require_positive_finite(
            self.median_tokens_per_second,
            "median_tokens_per_second",
        )
        reserved = _require_nonnegative_int(
            self.peak_reserved_bytes,
            "peak_reserved_bytes",
        )
        total = _require_positive_int(self.total_vram_bytes, "total_vram_bytes")
        if reserved * 10 > total * 9:
            raise ValueError("A selection cannot exceed 90% reserved VRAM.")

    def as_dict(self) -> dict[str, object]:
        return {
            "microbatch_sequences": self.microbatch_sequences,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "successful_repeats": self.successful_repeats,
            "median_tokens_per_second": self.median_tokens_per_second,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "total_vram_bytes": self.total_vram_bytes,
        }


class NoEligibleProfileCandidate(ValueError):
    """Raised when no candidate satisfies repetition and memory limits."""


def build_profile_work_items(
    request: HardwareProfileRequest,
) -> tuple[HardwareProfileWorkItem, ...]:
    """Build deterministic inputs for later isolated CUDA worker calls."""

    if not isinstance(request, HardwareProfileRequest):
        raise TypeError("request must be a HardwareProfileRequest.")
    items: list[HardwareProfileWorkItem] = []
    for candidate in request.candidates:
        grouping_hash = flat_synthetic_grouping_hash(
            request,
            microbatch_sequences=candidate.microbatch_sequences,
        )
        for repeat_index in range(1, request.repeats + 1):
            items.append(
                HardwareProfileWorkItem(
                    microbatch_sequences=candidate.microbatch_sequences,
                    gradient_accumulation_steps=candidate.gradient_accumulation_steps,
                    repeat_index=repeat_index,
                    synthetic_grouping_hash=grouping_hash,
                )
            )
    return tuple(items)


def flat_synthetic_grouping_hash(
    request: HardwareProfileRequest,
    *,
    microbatch_sequences: int,
) -> str:
    """Hash canonical synthetic tokens while ignoring reshape boundaries.

    Candidate microbatches reshape the same ordered 128-by-2048 synthetic
    token matrix.  Group delimiters are intentionally absent from the digest,
    so any valid physical decomposition produces the same hash while a change
    to flat sequence/token order would change it.
    """

    if not isinstance(request, HardwareProfileRequest):
        raise TypeError("request must be a HardwareProfileRequest.")
    candidate = MicrobatchCandidate.from_microbatch(microbatch_sequences)
    if candidate.microbatch_sequences not in request.candidate_microbatches:
        raise ValueError("microbatch_sequences is not a candidate in this request.")

    digest = sha256()
    digest.update(b"paper-exp-flat-synthetic-grouping-v1\0")
    digest.update(request.sequence_length.to_bytes(8, "big"))
    digest.update(request.global_sequences.to_bytes(8, "big"))
    for group_start in range(0, request.global_sequences, microbatch_sequences):
        group_stop = group_start + microbatch_sequences
        for sequence_index in range(group_start, group_stop):
            token_start = sequence_index * request.sequence_length
            token_stop = token_start + request.sequence_length
            for flat_token_index in range(token_start, token_stop):
                digest.update(flat_token_index.to_bytes(8, "big"))
    return digest.hexdigest()


def select_profile_candidate(
    request: HardwareProfileRequest,
    candidate_results: Sequence[CandidateProfileResult],
) -> ProfileSelection:
    """Select by throughput, treating values within 2% as a memory tie.

    Eligibility requires the complete requested repeat record, at least two
    successful synchronized repeats, and no successful repeat whose reserved
    VRAM exceeds 90% of total VRAM.  Among candidates within 2% of the best
    median throughput, lower worst-repeat reserved memory wins.  Remaining
    ties prefer higher throughput and then the lower microbatch.
    """

    ordered = _validate_result_set(request, candidate_results)
    eligible: list[ProfileSelection] = []
    for candidate in ordered:
        successful = candidate.successful_repeats
        if len(successful) < MIN_SUCCESSFUL_REPEATS:
            continue
        if any(
            result.peak_reserved_bytes * 10 > result.total_vram_bytes * 9
            for result in successful
        ):
            continue
        throughputs = [result.tokens_per_second for result in successful]
        reserved_values = [result.peak_reserved_bytes for result in successful]
        total_values = {result.total_vram_bytes for result in successful}
        # Successful-repeat validation guarantees these values are not None.
        eligible.append(
            ProfileSelection(
                microbatch_sequences=candidate.microbatch_sequences,
                gradient_accumulation_steps=candidate.gradient_accumulation_steps,
                successful_repeats=len(successful),
                median_tokens_per_second=float(median(throughputs)),  # type: ignore[arg-type]
                peak_reserved_bytes=max(reserved_values),  # type: ignore[arg-type]
                total_vram_bytes=next(iter(total_values)),  # type: ignore[arg-type]
            )
        )

    if not eligible:
        raise NoEligibleProfileCandidate(
            "No candidate has two successful repeats within 90% reserved VRAM."
        )
    fastest = max(value.median_tokens_per_second for value in eligible)
    tied = [
        value
        for value in eligible
        if value.median_tokens_per_second
        >= fastest * (1.0 - THROUGHPUT_TIE_FRACTION)
    ]
    return min(
        tied,
        key=lambda value: (
            value.peak_reserved_bytes,
            -value.median_tokens_per_second,
            value.microbatch_sequences,
        ),
    )


def build_hardware_profile_artifact(
    request: HardwareProfileRequest,
    candidate_results: Sequence[CandidateProfileResult],
    *,
    setup_seconds: float,
    validation_seconds: float,
    checkpoint_seconds: float,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Build a JSON-safe, explicitly non-scientific profiling artifact."""

    setup_seconds = _require_nonnegative_finite(setup_seconds, "setup_seconds")
    validation_seconds = _require_nonnegative_finite(
        validation_seconds,
        "validation_seconds",
    )
    checkpoint_seconds = _require_nonnegative_finite(
        checkpoint_seconds,
        "checkpoint_seconds",
    )
    ordered = _validate_result_set(request, candidate_results)
    selection = select_profile_candidate(request, ordered)
    grouping_hashes = {
        flat_synthetic_grouping_hash(
            request,
            microbatch_sequences=candidate.microbatch_sequences,
        )
        for candidate in request.candidates
    }
    if len(grouping_hashes) != 1:
        raise RuntimeError("Synthetic grouping changed across microbatch reshapes.")

    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping.")
    safe_provenance: dict[str, Any] = deepcopy(dict(provenance or {}))
    reject_scientific_keys(safe_provenance)
    artifact: dict[str, object] = {
        "artifact_type": "hardware_profile",
        "schema_version": 1,
        "scientific_evidence": False,
        "prohibited_use": PROHIBITED_USE,
        "request": request.as_dict(),
        "synthetic_grouping_hash": next(iter(grouping_hashes)),
        "setup_seconds": setup_seconds,
        "validation_seconds": validation_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "candidates": [candidate.as_dict() for candidate in ordered],
        "selection": selection.as_dict(),
        "provenance": safe_provenance,
    }
    reject_scientific_keys(artifact)
    try:
        json.dumps(artifact, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Hardware profile artifact must be finite and JSON-safe.") from error
    return artifact


def reject_scientific_keys(value: object, *, path: str = "artifact") -> None:
    """Reject scientific result keys at any depth in an artifact payload."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            normalized = _normalize_key(key)
            if normalized == "scientific_evidence" and nested is not False:
                raise ValueError(
                    f"scientific_evidence must be false at {path}.{key}."
                )
            if normalized not in _SCIENTIFIC_KEY_EXEMPTIONS:
                tokens = frozenset(part for part in normalized.split("_") if part)
                forbidden = sorted(tokens & _SCIENTIFIC_KEY_TOKENS)
                if forbidden:
                    raise ValueError(
                        f"Scientific key {key!r} is prohibited at {path}; "
                        f"matched {forbidden[0]!r}."
                    )
            reject_scientific_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            reject_scientific_keys(nested, path=f"{path}[{index}]")


def _validate_result_set(
    request: HardwareProfileRequest,
    candidate_results: Sequence[CandidateProfileResult],
) -> tuple[CandidateProfileResult, ...]:
    if not isinstance(request, HardwareProfileRequest):
        raise TypeError("request must be a HardwareProfileRequest.")
    if not isinstance(candidate_results, Sequence) or isinstance(
        candidate_results,
        (str, bytes),
    ):
        raise TypeError("candidate_results must be an ordered result sequence.")
    results = tuple(candidate_results)
    if any(not isinstance(result, CandidateProfileResult) for result in results):
        raise TypeError("candidate_results must contain CandidateProfileResult values.")
    by_microbatch = {result.microbatch_sequences: result for result in results}
    if len(by_microbatch) != len(results):
        raise ValueError("candidate_results contains duplicate microbatches.")
    expected = set(request.candidate_microbatches)
    actual = set(by_microbatch)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "candidate_results must cover the request exactly; "
            f"missing={missing}, unexpected={unexpected}."
        )

    ordered: list[CandidateProfileResult] = []
    all_total_vram: set[int] = set()
    expected_indexes = set(range(1, request.repeats + 1))
    for microbatch in request.candidate_microbatches:
        result = by_microbatch[microbatch]
        expected_candidate = MicrobatchCandidate.from_microbatch(microbatch)
        if result.gradient_accumulation_steps != expected_candidate.gradient_accumulation_steps:
            raise ValueError("Candidate accumulation does not match the request.")
        indexes = {repeat.repeat_index for repeat in result.repeats}
        if indexes != expected_indexes or len(result.repeats) != request.repeats:
            raise ValueError(
                "Each candidate must contain exactly the requested repeat indexes."
            )
        for repeat in result.successful_repeats:
            # Successful-repeat validation guarantees total_vram_bytes is an int.
            all_total_vram.add(repeat.total_vram_bytes)  # type: ignore[arg-type]
        ordered.append(result)
    if len(all_total_vram) > 1:
        raise ValueError("Successful repeats must report one consistent total VRAM value.")
    return tuple(ordered)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a nonempty string.")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer.")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a nonnegative integer.")
    return value


def _require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return numeric


def _require_nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return numeric
