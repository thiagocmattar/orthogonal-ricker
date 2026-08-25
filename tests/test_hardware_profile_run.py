from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

import paper_exp.hardware_profile_run as hardware_profile_run
from paper_exp.hardware_profile import HardwareProfileRequest
from paper_exp.hardware_profile_run import ARTIFACT_FILE
from paper_exp.hardware_profile_run import ARTIFACT_SHA_FILE
from paper_exp.hardware_profile_run import HardwareProfileStateError
from paper_exp.hardware_profile_run import HardwareProfileWorkerError
from paper_exp.hardware_profile_run import _publish
from paper_exp.hardware_profile_run import run_hardware_profile


CONTAINER_IMAGE = "runpod/pytorch@sha256:" + "d" * 64
WORKER_TIMEOUT_SECONDS = 30.0


def test_real_spawn_profile_publishes_and_reuses_completed_work(tmp_path: Path) -> None:
    request = _request(candidates=(1, 2))
    root = tmp_path / "profile"
    scratch = root / "checkpoint-scratch"

    first = run_hardware_profile(
        request,
        cuda_device=3,
        work_root=root,
        checkpoint_scratch=scratch,
        worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
        container_image=CONTAINER_IMAGE,
        _test_worker_mode=True,
    )

    assert first.reused_repeats == 0
    owner = _json(root / "profile-owner.json")
    assert owner["worker_timeout_seconds"] == WORKER_TIMEOUT_SECONDS
    assert owner["container_image"] == CONTAINER_IMAGE
    assert first.artifact["scientific_evidence"] is False
    assert first.artifact["selection"]["microbatch_sequences"] == 2
    provenance = first.artifact["provenance"]
    assert provenance["worker_timeout_seconds"] == WORKER_TIMEOUT_SECONDS
    assert provenance["container_image"] == CONTAINER_IMAGE
    assert provenance["worker_identity"]["nvidia_driver_version"] == "999.0"
    assert provenance["worker_identity"]["gpu"] == {
        "uuid": "TEST-GPU-3",
        "name": "NVIDIA RTX A6000 48GB",
        "total_vram_bytes": 48_000_000_000,
        "bf16_supported": True,
        "compute_capability": [9, 9],
    }
    assert provenance["selected_checkpoint"] == {
        "sha256": "b" * 64,
        "bytes_written": 1234,
        "disposable": True,
    }
    assert list(scratch.iterdir()) == []
    assert sha256(first.artifact_path.read_bytes()).hexdigest() == first.artifact_sha256
    assert first.artifact_sha256_path.read_text(encoding="ascii") == (
        f"{first.artifact_sha256}  {ARTIFACT_FILE}\n"
    )

    before = _attempt_counts(root)
    second = run_hardware_profile(
        request,
        cuda_device=3,
        work_root=root,
        checkpoint_scratch=scratch,
        worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
        container_image=CONTAINER_IMAGE,
        _test_worker_mode=True,
    )

    assert second.reused_repeats == 4
    assert second.artifact_sha256 == first.artifact_sha256
    assert _attempt_counts(root) == before


def test_cuda_oom_is_completed_nonfit_and_other_candidate_can_win(tmp_path: Path) -> None:
    request = _request(candidates=(1, 2))
    root = tmp_path / "profile"

    result = run_hardware_profile(
        request,
        cuda_device=0,
        work_root=root,
        checkpoint_scratch=root / "scratch",
        worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
        container_image=CONTAINER_IMAGE,
        _test_worker_mode=True,
        _test_oom_work_keys=("mb-001-repeat-001", "mb-001-repeat-002"),
    )

    first_candidate = result.artifact["candidates"][0]
    assert [repeat["fit"] for repeat in first_candidate["repeats"]] == [False, False]
    assert [repeat["error"] for repeat in first_candidate["repeats"]] == [
        "cuda_out_of_memory",
        "cuda_out_of_memory",
    ]
    assert result.artifact["selection"]["microbatch_sequences"] == 2


def test_worker_failure_is_preserved_and_requires_explicit_retry(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }

    with pytest.raises(HardwareProfileWorkerError, match="injected test worker failure"):
        run_hardware_profile(
            request,
            **kwargs,
            _test_fail_work_keys=("mb-001-repeat-001",),
        )
    failed = root / "repeats" / "mb-001-repeat-001" / "001"
    assert _json(failed / "state.json")["status"] == "failed"

    with pytest.raises(HardwareProfileStateError, match="explicit retry_failed=True"):
        run_hardware_profile(request, **kwargs)
    assert not (failed.parent / "002").exists()

    completed = run_hardware_profile(request, **kwargs, retry_failed=True)
    assert _json(failed.parent / "002" / "state.json")["status"] == "completed"
    assert completed.reused_repeats == 0


def test_timeout_is_terminal_and_requires_explicit_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    original_run = subprocess.run
    seen_timeouts: list[float] = []

    def timeout_once(*args: Any, **call_kwargs: Any) -> Any:
        seen_timeouts.append(call_kwargs["timeout"])
        if len(seen_timeouts) == 1:
            raise subprocess.TimeoutExpired(args[0], call_kwargs["timeout"])
        return original_run(*args, **call_kwargs)

    monkeypatch.setattr("paper_exp.hardware_profile_run.subprocess.run", timeout_once)

    with pytest.raises(HardwareProfileWorkerError, match="timed out after 30 seconds"):
        run_hardware_profile(request, **kwargs)
    failed = root / "repeats" / "mb-001-repeat-001" / "001"
    assert _json(failed / "state.json")["status"] == "failed"
    response = _json(failed / "response.json")
    assert response["failure"] == {
        "type": "WorkerTimeoutError",
        "message": "repeat worker exceeded 30 seconds.",
    }

    with pytest.raises(HardwareProfileStateError, match="explicit retry_failed=True"):
        run_hardware_profile(request, **kwargs)
    completed = run_hardware_profile(request, **kwargs, retry_failed=True)

    assert _json(failed.parent / "002" / "state.json")["status"] == "completed"
    assert completed.reused_repeats == 0
    assert seen_timeouts == [WORKER_TIMEOUT_SECONDS] * 4


def test_selected_timeout_cleans_owned_scratch_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    scratch = root / "scratch"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": scratch,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    original_run = subprocess.run
    selected_timeouts = 0
    selected_scratch_paths: list[Path] = []

    def timeout_selected_once(*args: Any, **call_kwargs: Any) -> Any:
        nonlocal selected_timeouts
        command = args[0]
        if command[4] == "selected" and selected_timeouts == 0:
            selected_timeouts += 1
            selected_scratch = Path(command[-1])
            selected_scratch_paths.append(selected_scratch)
            assert selected_scratch.parent == scratch
            (selected_scratch / "partial-model.safetensors").write_bytes(b"partial")
            raise subprocess.TimeoutExpired(command, call_kwargs["timeout"])
        return original_run(*args, **call_kwargs)

    monkeypatch.setattr(
        "paper_exp.hardware_profile_run.subprocess.run",
        timeout_selected_once,
    )

    with pytest.raises(HardwareProfileWorkerError, match="selected worker timed out"):
        run_hardware_profile(request, **kwargs)

    failed = root / "selected" / "001"
    assert _json(failed / "state.json")["status"] == "failed"
    assert selected_timeouts == 1
    assert len(selected_scratch_paths) == 1
    assert not selected_scratch_paths[0].exists()
    assert list(scratch.iterdir()) == []

    with pytest.raises(HardwareProfileStateError, match="explicit retry_failed=True"):
        run_hardware_profile(request, **kwargs)
    assert list(scratch.iterdir()) == []

    completed = run_hardware_profile(request, **kwargs, retry_failed=True)

    assert _json(root / "selected" / "002" / "state.json")["status"] == "completed"
    assert completed.reused_repeats == 2
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(
    ("interrupt_after", "artifact_exists"),
    [(ARTIFACT_SHA_FILE, False), (ARTIFACT_FILE, True)],
)
def test_profile_publication_recovers_after_each_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after: str,
    artifact_exists: bool,
) -> None:
    artifact_path = tmp_path / ARTIFACT_FILE
    checksum_path = tmp_path / ARTIFACT_SHA_FILE
    data = b'{"artifact_type":"hardware_profile"}\n'
    digest = sha256(data).hexdigest()
    sidecar = f"{digest}  {ARTIFACT_FILE}\n".encode("ascii")
    original_write = hardware_profile_run._write_bytes
    interrupted = False

    def write_then_interrupt(path: Path, payload: bytes) -> None:
        nonlocal interrupted
        original_write(path, payload)
        if path.name == interrupt_after and not interrupted:
            interrupted = True
            raise RuntimeError("injected publication interruption")

    with monkeypatch.context() as patch:
        patch.setattr(hardware_profile_run, "_write_bytes", write_then_interrupt)
        with pytest.raises(RuntimeError, match="publication interruption"):
            _publish(artifact_path, checksum_path, data, digest)

    assert interrupted is True
    assert checksum_path.read_bytes() == sidecar
    assert artifact_path.exists() is artifact_exists
    _publish(artifact_path, checksum_path, data, digest)
    assert artifact_path.read_bytes() == data
    assert checksum_path.read_bytes() == sidecar


def test_profile_publication_reconciles_exact_legacy_report_only(tmp_path: Path) -> None:
    artifact_path = tmp_path / ARTIFACT_FILE
    checksum_path = tmp_path / ARTIFACT_SHA_FILE
    data = b'{"artifact_type":"hardware_profile"}\n'
    digest = sha256(data).hexdigest()
    artifact_path.write_bytes(data)

    _publish(artifact_path, checksum_path, data, digest)

    assert artifact_path.read_bytes() == data
    assert checksum_path.read_text(encoding="ascii") == (
        f"{digest}  {ARTIFACT_FILE}\n"
    )


@pytest.mark.parametrize("mismatch", ["artifact", "checksum"])
def test_profile_publication_rejects_one_sided_mismatch_without_mutation(
    tmp_path: Path,
    mismatch: str,
) -> None:
    artifact_path = tmp_path / ARTIFACT_FILE
    checksum_path = tmp_path / ARTIFACT_SHA_FILE
    data = b'{"artifact_type":"hardware_profile"}\n'
    digest = sha256(data).hexdigest()
    mismatched = b"mismatched\n"
    target = artifact_path if mismatch == "artifact" else checksum_path
    counterpart = checksum_path if mismatch == "artifact" else artifact_path
    target.write_bytes(mismatched)

    with pytest.raises(HardwareProfileStateError, match="differs"):
        _publish(artifact_path, checksum_path, data, digest)

    assert target.read_bytes() == mismatched
    assert not counterpart.exists()


def test_running_or_malformed_attempt_is_rejected_without_new_work(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    run_hardware_profile(request, **kwargs)
    attempt = root / "repeats" / "mb-001-repeat-001" / "001"
    state = _json(attempt / "state.json")
    state = {
        key: value
        for key, value in state.items()
        if key not in {"finished_at", "response_sha256"}
    }
    state["status"] = "running"
    _write_json(attempt / "state.json", state)

    with pytest.raises(HardwareProfileStateError, match="marked running"):
        run_hardware_profile(request, **kwargs, retry_failed=True)
    assert not (attempt.parent / "002").exists()


def test_identity_mismatch_and_request_change_fail_closed(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    run_hardware_profile(request, **kwargs)
    attempt = root / "repeats" / "mb-001-repeat-002" / "001"
    response = _json(attempt / "response.json")
    response["identity"]["gpu"]["uuid"] = "DIFFERENT-GPU"
    _write_json(attempt / "response.json", response)
    state = _json(attempt / "state.json")
    state["response_sha256"] = sha256(_canonical_bytes(response)).hexdigest()
    _write_json(attempt / "state.json", state)

    with pytest.raises(HardwareProfileStateError, match="inconsistent Git/environment"):
        run_hardware_profile(request, **kwargs)

    with pytest.raises(HardwareProfileStateError, match="ownership/request marker"):
        run_hardware_profile(
            _request(candidates=(2,)),
            **kwargs,
        )


def test_dirty_worker_identity_is_rejected(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    run_hardware_profile(request, **kwargs)
    attempt = root / "repeats" / "mb-001-repeat-001" / "001"
    response = _json(attempt / "response.json")
    response["identity"]["repo_git_dirty"] = True
    _write_json(attempt / "response.json", response)
    state = _json(attempt / "state.json")
    state["response_sha256"] = sha256(_canonical_bytes(response)).hexdigest()
    _write_json(attempt / "state.json", state)

    with pytest.raises(HardwareProfileStateError, match="immutable clean checkout"):
        run_hardware_profile(request, **kwargs)


def test_malformed_driver_identity_is_rejected(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    run_hardware_profile(request, **kwargs)
    attempt = root / "repeats" / "mb-001-repeat-001" / "001"
    response = _json(attempt / "response.json")
    response["identity"]["nvidia_driver_version"] = "unknown"
    _write_json(attempt / "response.json", response)
    state = _json(attempt / "state.json")
    state["response_sha256"] = sha256(_canonical_bytes(response)).hexdigest()
    _write_json(attempt / "state.json", state)

    with pytest.raises(HardwareProfileStateError, match="NVIDIA driver identity"):
        run_hardware_profile(request, **kwargs)


def test_changed_execution_options_reject_reuse(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    kwargs = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "container_image": CONTAINER_IMAGE,
        "_test_worker_mode": True,
    }
    run_hardware_profile(request, **kwargs)
    before = _attempt_counts(root)

    with pytest.raises(HardwareProfileStateError, match="ownership/request marker"):
        run_hardware_profile(
            request,
            **{**kwargs, "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS + 1},
        )
    with pytest.raises(HardwareProfileStateError, match="ownership/request marker"):
        run_hardware_profile(
            request,
            **{
                **kwargs,
                "container_image": "runpod/pytorch@sha256:" + "e" * 64,
            },
        )
    assert _attempt_counts(root) == before


def test_execution_options_are_validated_before_root_mutation(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    root = tmp_path / "profile"
    common = {
        "cuda_device": 0,
        "work_root": root,
        "checkpoint_scratch": root / "scratch",
        "_test_worker_mode": True,
    }

    with pytest.raises(TypeError, match="worker_timeout_seconds"):
        run_hardware_profile(request, **common)  # type: ignore[call-arg]
    for timeout in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            run_hardware_profile(
                request,
                **common,
                worker_timeout_seconds=timeout,
                container_image=CONTAINER_IMAGE,
            )
    with pytest.raises(TypeError, match="must be numeric"):
        run_hardware_profile(
            request,
            **common,
            worker_timeout_seconds=True,
            container_image=CONTAINER_IMAGE,
        )
    for image in (
        "",
        "runpod/pytorch:latest",
        "@sha256:" + "a" * 64,
        "runpod/pytorch@sha256:" + "A" * 64,
    ):
        with pytest.raises(ValueError, match="immutable image reference"):
            run_hardware_profile(
                request,
                **common,
                worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
                container_image=image,
            )
    assert not root.exists()


def test_work_root_and_scratch_must_be_empty_owned_safe_paths(tmp_path: Path) -> None:
    request = _request(candidates=(1,))
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "foreign.txt").write_text("not owned", encoding="utf-8")

    with pytest.raises(HardwareProfileStateError, match="ownership/request marker"):
        run_hardware_profile(
            request,
            cuda_device=0,
            work_root=occupied,
            checkpoint_scratch=occupied / "scratch",
            worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
            container_image=CONTAINER_IMAGE,
            _test_worker_mode=True,
        )
    with pytest.raises(ValueError, match="direct child"):
        run_hardware_profile(
            request,
            cuda_device=0,
            work_root=tmp_path / "profile",
            checkpoint_scratch=tmp_path / "outside",
            worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
            container_image=CONTAINER_IMAGE,
            _test_worker_mode=True,
        )
    with pytest.raises(ValueError, match="reserved"):
        run_hardware_profile(
            request,
            cuda_device=0,
            work_root=tmp_path / "profile",
            checkpoint_scratch=tmp_path / "profile" / "repeats",
            worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
            container_image=CONTAINER_IMAGE,
            _test_worker_mode=True,
        )
    with pytest.raises(TypeError, match="nonnegative integer"):
        run_hardware_profile(
            request,
            cuda_device=True,
            work_root=tmp_path / "profile",
            checkpoint_scratch=tmp_path / "profile" / "scratch",
            worker_timeout_seconds=WORKER_TIMEOUT_SECONDS,
            container_image=CONTAINER_IMAGE,
            _test_worker_mode=True,
        )


def _request(*, candidates: tuple[int, ...]) -> HardwareProfileRequest:
    return HardwareProfileRequest(
        architecture="EleutherAI/pythia-14m-deduped",
        revision="a" * 40,
        gpu_class="NVIDIA RTX A6000 48GB",
        candidate_microbatches=candidates,
        repeats=2,
    )


def _attempt_counts(root: Path) -> dict[str, int]:
    return {
        path.parent.name + "/" + path.name: len(list(path.iterdir()))
        for path in sorted((root / "repeats").glob("*/"))
    } | {"selected": len(list((root / "selected").iterdir()))}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))
