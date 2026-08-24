from __future__ import annotations

import json

import pytest

from paper_exp.hardware_profile import (
    CandidateProfileResult,
    GLOBAL_SEQUENCES,
    HardwareProfileRequest,
    NoEligibleProfileCandidate,
    ProfileRepeatResult,
    build_hardware_profile_artifact,
    build_profile_work_items,
    flat_synthetic_grouping_hash,
    reject_scientific_keys,
    select_profile_candidate,
)


TOTAL_VRAM = 48_000_000_000


def test_request_derives_only_exact_global_batch_decompositions() -> None:
    request = _request(candidates=(1, 4, 16, 128))

    assert [
        (candidate.microbatch_sequences, candidate.gradient_accumulation_steps)
        for candidate in request.candidates
    ] == [(1, 128), (4, 32), (16, 8), (128, 1)]
    assert request.sequence_length == 2_048
    assert request.global_sequences == GLOBAL_SEQUENCES

    with pytest.raises(ValueError, match="distinct"):
        _request(candidates=(2, 2))
    with pytest.raises(TypeError, match="ordered integer sequence"):
        HardwareProfileRequest(
            architecture="EleutherAI/pythia-14m-deduped",
            revision="a" * 40,
            gpu_class="NVIDIA RTX A6000 48GB",
            candidate_microbatches={2, 4},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="divide 128"):
        _request(candidates=(3,))
    with pytest.raises(ValueError, match="sequence_length must be 2048"):
        _request(candidates=(2,), sequence_length=1_024)
    with pytest.raises(ValueError, match="global_sequences must be 128"):
        _request(candidates=(2,), global_sequences=64)
    with pytest.raises(ValueError, match="immutable"):
        _request(candidates=(2,), revision="main")
    with pytest.raises(ValueError, match="lowercase hex"):
        _request(candidates=(2,), revision="A" * 40)
    with pytest.raises(ValueError, match="at least 2"):
        _request(candidates=(2,), repeats=1)


def test_flat_synthetic_hash_is_stable_across_microbatch_reshapes() -> None:
    request = _request(candidates=(1, 2, 8, 32, 128))

    hashes = {
        flat_synthetic_grouping_hash(request, microbatch_sequences=value)
        for value in request.candidate_microbatches
    }

    assert len(hashes) == 1
    assert len(next(iter(hashes))) == 64
    assert (
        flat_synthetic_grouping_hash(request, microbatch_sequences=8)
        == flat_synthetic_grouping_hash(request, microbatch_sequences=8)
    )
    with pytest.raises(ValueError, match="not a candidate"):
        flat_synthetic_grouping_hash(request, microbatch_sequences=4)


def test_work_items_form_a_deterministic_cuda_subprocess_boundary() -> None:
    request = _request(candidates=(2, 8), repeats=2)

    items = build_profile_work_items(request)

    assert [
        (
            item.microbatch_sequences,
            item.gradient_accumulation_steps,
            item.repeat_index,
        )
        for item in items
    ] == [(2, 64, 1), (2, 64, 2), (8, 16, 1), (8, 16, 2)]
    assert len({item.synthetic_grouping_hash for item in items}) == 1
    assert build_profile_work_items(request) == items


def test_repeat_schema_rejects_partial_or_scientific_shaped_results() -> None:
    valid = _success(2, 1, throughput=1_000.0, reserved=20_000_000_000)
    assert valid.fit is True
    assert valid.error is None

    with pytest.raises(ValueError, match="all operational measurements"):
        ProfileRepeatResult(
            microbatch_sequences=2,
            repeat_index=1,
            fit=True,
            error=None,
            synchronized_seconds=1.0,
            tokens_per_second=1_000.0,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            total_vram_bytes=None,
        )
    with pytest.raises(ValueError, match="cannot publish timing"):
        ProfileRepeatResult(
            microbatch_sequences=2,
            repeat_index=1,
            fit=False,
            error="out of memory",
            synchronized_seconds=1.0,
            tokens_per_second=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            total_vram_bytes=None,
        )
    with pytest.raises(ValueError, match="allocated_bytes cannot exceed"):
        _success(
            2,
            1,
            throughput=1_000.0,
            allocated=21_000_000_000,
            reserved=20_000_000_000,
        )


def test_selector_uses_two_repeats_memory_limit_and_two_percent_tie_band() -> None:
    request = _request(candidates=(2, 4, 8))
    results = (
        _candidate(2, throughputs=(1_000.0, 1_000.0), reserved=20_000_000_000),
        # Within 2% of the fastest and lower-memory, so this candidate wins.
        _candidate(4, throughputs=(990.0, 990.0), reserved=18_000_000_000),
        # Faster, but above the strict 90% reserved-memory ceiling.
        _candidate(8, throughputs=(1_100.0, 1_100.0), reserved=43_200_000_001),
    )

    selected = select_profile_candidate(request, results)

    assert selected.microbatch_sequences == 4
    assert selected.gradient_accumulation_steps == 32
    assert selected.successful_repeats == 2
    assert selected.median_tokens_per_second == 990.0
    assert selected.peak_reserved_bytes == 18_000_000_000


def test_selector_does_not_use_one_success_or_incomplete_repeat_records() -> None:
    request = _request(candidates=(2, 4))
    failed = ProfileRepeatResult(
        microbatch_sequences=2,
        repeat_index=2,
        fit=False,
        error="out of memory",
        synchronized_seconds=None,
        tokens_per_second=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        total_vram_bytes=None,
    )
    one_success = CandidateProfileResult(
        microbatch_sequences=2,
        gradient_accumulation_steps=64,
        repeats=(_success(2, 1, throughput=9_999.0), failed),
    )
    valid = _candidate(4, throughputs=(900.0, 910.0), reserved=19_000_000_000)

    assert select_profile_candidate(request, (one_success, valid)).microbatch_sequences == 4

    incomplete = CandidateProfileResult(
        microbatch_sequences=4,
        gradient_accumulation_steps=32,
        repeats=(_success(4, 1, throughput=900.0),),
    )
    with pytest.raises(ValueError, match="exactly the requested repeat indexes"):
        select_profile_candidate(request, (one_success, incomplete))


def test_selector_fails_when_no_candidate_is_eligible() -> None:
    request = _request(candidates=(2,))
    result = _candidate(
        2,
        throughputs=(1_000.0, 1_001.0),
        reserved=43_200_000_001,
    )

    with pytest.raises(NoEligibleProfileCandidate, match="two successful repeats"):
        select_profile_candidate(request, (result,))


def test_artifact_is_explicitly_non_evidence_and_json_safe() -> None:
    request = _request(candidates=(2, 4))
    candidates = (
        _candidate(2, throughputs=(1_000.0, 1_020.0), reserved=20_000_000_000),
        _candidate(4, throughputs=(900.0, 910.0), reserved=18_000_000_000),
    )

    artifact = build_hardware_profile_artifact(
        request,
        candidates,
        setup_seconds=12.5,
        validation_seconds=4.0,
        checkpoint_seconds=6.5,
        provenance={"git_commit": "a" * 40, "image": "pinned-image@sha256:abc"},
    )

    assert artifact["artifact_type"] == "hardware_profile"
    assert artifact["scientific_evidence"] is False
    assert "must not be used" in str(artifact["prohibited_use"])
    assert artifact["setup_seconds"] == 12.5
    assert artifact["validation_seconds"] == 4.0
    assert artifact["checkpoint_seconds"] == 6.5
    assert artifact["selection"]["microbatch_sequences"] == 2
    json.dumps(artifact, allow_nan=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"loss": 1.0},
        {"outer": [{"near-zero-mass": 0.5}]},
        {"outer": {"activation_sparsity": 0.5}},
        {"reported_metrics": {"tokens_per_second": 1.0}},
        {"prediction": "text"},
    ],
)
def test_scientific_keys_are_rejected_recursively(payload: object) -> None:
    with pytest.raises(ValueError, match="Scientific key"):
        reject_scientific_keys(payload)


def test_artifact_builder_rejects_nested_scientific_provenance() -> None:
    request = _request(candidates=(2,))
    candidate = _candidate(
        2,
        throughputs=(1_000.0, 1_001.0),
        reserved=20_000_000_000,
    )

    with pytest.raises(ValueError, match="Scientific key 'loss'"):
        build_hardware_profile_artifact(
            request,
            (candidate,),
            setup_seconds=1.0,
            validation_seconds=1.0,
            checkpoint_seconds=1.0,
            provenance={"nested": [{"loss": 5.0}]},
        )

    with pytest.raises(ValueError, match="scientific_evidence must be false"):
        build_hardware_profile_artifact(
            request,
            (candidate,),
            setup_seconds=1.0,
            validation_seconds=1.0,
            checkpoint_seconds=1.0,
            provenance={"scientific_evidence": True},
        )


def _request(
    *,
    candidates: tuple[int, ...],
    revision: str = "a" * 40,
    repeats: int = 2,
    sequence_length: int = 2_048,
    global_sequences: int = 128,
) -> HardwareProfileRequest:
    return HardwareProfileRequest(
        architecture="EleutherAI/pythia-14m-deduped",
        revision=revision,
        gpu_class="NVIDIA RTX A6000 48GB",
        candidate_microbatches=candidates,
        repeats=repeats,
        sequence_length=sequence_length,
        global_sequences=global_sequences,
    )


def _candidate(
    microbatch: int,
    *,
    throughputs: tuple[float, ...],
    reserved: int,
) -> CandidateProfileResult:
    return CandidateProfileResult(
        microbatch_sequences=microbatch,
        gradient_accumulation_steps=GLOBAL_SEQUENCES // microbatch,
        repeats=tuple(
            _success(
                microbatch,
                repeat_index,
                throughput=throughput,
                reserved=reserved,
            )
            for repeat_index, throughput in enumerate(throughputs, start=1)
        ),
    )


def _success(
    microbatch: int,
    repeat_index: int,
    *,
    throughput: float,
    allocated: int = 16_000_000_000,
    reserved: int = 20_000_000_000,
) -> ProfileRepeatResult:
    return ProfileRepeatResult(
        microbatch_sequences=microbatch,
        repeat_index=repeat_index,
        fit=True,
        error=None,
        synchronized_seconds=2.0,
        tokens_per_second=throughput,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        total_vram_bytes=TOTAL_VRAM,
    )
