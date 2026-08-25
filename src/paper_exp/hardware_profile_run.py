"""Restart-safe profiling coordinator that imports Torch only in fresh workers."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from paper_exp.hardware_profile import CandidateProfileResult, HardwareProfileRequest
from paper_exp.hardware_profile import HardwareProfileWorkItem, MicrobatchCandidate
from paper_exp.hardware_profile import ProfileRepeatResult, build_hardware_profile_artifact
from paper_exp.hardware_profile import build_profile_work_items, select_profile_candidate


SCHEMA_VERSION = 1
MAX_JSON_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_FAILURE_CHARS = 2_048
OWNER_FILE = "profile-owner.json"
ARTIFACT_FILE = "hardware_profile.json"
ARTIFACT_SHA_FILE = "hardware_profile.sha256"
_STATE_COMMON = {
    "schema_version", "kind", "request_sha256", "work_key", "attempt_index", "status", "started_at",
}
_IDENTITY_FIELDS = {
    "repo_git_commit", "repo_git_dirty", "hostname", "platform", "python_executable",
    "python_version", "package_versions", "cuda_runtime", "nvidia_driver_version",
    "cuda_visible_devices", "gpu",
}
_GPU_FIELDS = {"uuid", "name", "total_vram_bytes", "bf16_supported", "compute_capability"}
_CONTAINER_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_NVIDIA_DRIVER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


@dataclass(frozen=True)
class HardwareProfileRun:
    work_root: Path
    artifact_path: Path
    artifact_sha256_path: Path
    artifact_sha256: str
    artifact: dict[str, object]
    reused_repeats: int


class HardwareProfileStateError(RuntimeError):
    """Durable profile state is unsafe to consume or resume."""


class HardwareProfileWorkerError(RuntimeError):
    """A terminal failed worker attempt was preserved."""


def run_hardware_profile(
    request: HardwareProfileRequest,
    *,
    cuda_device: int,
    work_root: str | Path,
    checkpoint_scratch: str | Path,
    worker_timeout_seconds: float,
    container_image: str,
    retry_failed: bool = False,
    _test_worker_mode: bool = False,
    _test_fail_work_keys: Sequence[str] = (),
    _test_oom_work_keys: Sequence[str] = (),
) -> HardwareProfileRun:
    """Run or safely resume one pinned profile on one numeric CUDA device."""

    if not isinstance(request, HardwareProfileRequest):
        raise TypeError("request must be a HardwareProfileRequest.")
    if isinstance(cuda_device, bool) or not isinstance(cuda_device, int) or cuda_device < 0:
        raise TypeError("cuda_device must be a nonnegative integer.")
    timeout_seconds = _positive_finite_timeout(worker_timeout_seconds)
    image = _immutable_container_image(container_image)
    if not isinstance(retry_failed, bool):
        raise TypeError("retry_failed must be a bool.")
    request_sha = _request_sha(request)
    root, scratch = _owned_root(
        work_root,
        checkpoint_scratch,
        request=request,
        request_sha=request_sha,
        worker_timeout_seconds=timeout_seconds,
        container_image=image,
    )
    failures, ooms = frozenset(_test_fail_work_keys), frozenset(_test_oom_work_keys)
    envelopes: list[dict[str, Any]] = []
    reused = 0
    for item in build_profile_work_items(request):
        key = _repeat_key(item)
        envelope, was_reused = _attempt(
            kind="repeat", request=request, request_sha=request_sha,
            payload=item.as_dict(), work_key=key, attempts_root=root / "repeats" / key,
            cuda_device=cuda_device, scratch=scratch, retry_failed=retry_failed,
            worker_timeout_seconds=timeout_seconds,
            test_mode=_test_worker_mode, test_failure=key in failures, test_oom=key in ooms,
        )
        _repeat_result(envelope, request_sha=request_sha, expected=item)
        _require_device(envelope, cuda_device)
        envelopes.append(envelope)
        reused += int(was_reused)

    identity = _consistent_identity(envelopes)
    candidates = _candidate_results(request, envelopes, request_sha=request_sha)
    selection = select_profile_candidate(request, candidates)
    selected = MicrobatchCandidate(
        selection.microbatch_sequences, selection.gradient_accumulation_steps
    )
    final, _ = _attempt(
        kind="selected", request=request, request_sha=request_sha,
        payload=selected.as_dict(), work_key="selected", attempts_root=root / "selected",
        cuda_device=cuda_device, scratch=scratch, retry_failed=retry_failed,
        worker_timeout_seconds=timeout_seconds,
        test_mode=_test_worker_mode, test_failure="selected" in failures, test_oom=False,
    )
    timing = _selected_timing(final, request_sha=request_sha, expected=selected)
    if final["identity"] != identity:
        raise HardwareProfileStateError("Selected worker identity differs from repeat workers.")
    _require_device(final, cuda_device)
    if any(scratch.iterdir()):
        raise HardwareProfileStateError("Checkpoint scratch is not empty after timing.")

    from paper_exp.hardware_profile_worker import profile_only_workload_metadata

    artifact = build_hardware_profile_artifact(
        request, candidates,
        setup_seconds=timing["setup_seconds"],
        validation_seconds=timing["validation_seconds"],
        checkpoint_seconds=timing["checkpoint_seconds"],
        provenance={
            "container_image": image,
            "worker_timeout_seconds": timeout_seconds,
            "worker_identity": identity,
            "workload": profile_only_workload_metadata(),
            "selected_checkpoint": {
                "sha256": timing["checkpoint_sha256"],
                "bytes_written": timing["checkpoint_bytes"],
                "disposable": True,
            },
        },
    )
    artifact_bytes = _json_bytes(artifact, MAX_ARTIFACT_BYTES)
    artifact_sha = sha256(artifact_bytes).hexdigest()
    artifact_path, sha_path = root / ARTIFACT_FILE, root / ARTIFACT_SHA_FILE
    _publish(artifact_path, sha_path, artifact_bytes, artifact_sha)
    return HardwareProfileRun(
        root, artifact_path, sha_path, artifact_sha, artifact, reused
    )


def _attempt(
    *, kind: str, request: HardwareProfileRequest, request_sha: str,
    payload: dict[str, object], work_key: str, attempts_root: Path,
    cuda_device: int, scratch: Path, retry_failed: bool, test_mode: bool,
    worker_timeout_seconds: float, test_failure: bool, test_oom: bool,
) -> tuple[dict[str, Any], bool]:
    attempts_root.mkdir(parents=True, exist_ok=True)
    _not_symlink(attempts_root)
    attempts = _attempt_dirs(attempts_root)
    completed: dict[str, Any] | None = None
    failed = False
    for index, directory in enumerate(attempts, 1):
        if completed is not None:
            raise HardwareProfileStateError(f"Attempt exists after completion: {directory}")
        if directory.name != f"{index:03d}":
            raise HardwareProfileStateError(f"Non-contiguous attempts: {attempts_root}")
        if _read_json(directory / "request.json") != request.as_dict():
            raise HardwareProfileStateError(f"Attempt request mismatch: {directory}")
        if _read_json(directory / "payload.json") != payload:
            raise HardwareProfileStateError(f"Attempt payload mismatch: {directory}")
        state = _read_json(directory / "state.json")
        _state(state, kind, request_sha, work_key, index)
        if state["status"] == "running":
            raise HardwareProfileStateError(f"Attempt is still marked running: {directory}")
        response = _read_json(directory / "response.json")
        if sha256(_json_bytes(response, MAX_JSON_BYTES)).hexdigest() != state["response_sha256"]:
            raise HardwareProfileStateError(f"Response SHA mismatch: {directory}")
        _header(response, kind, request_sha, payload)
        if response["status"] != state["status"]:
            raise HardwareProfileStateError(f"State/response status mismatch: {directory}")
        if state["status"] == "completed":
            completed = response
        else:
            _failed_response(response)
            failed = True
    if completed is not None:
        return completed, True
    if failed and not retry_failed:
        raise HardwareProfileStateError(
            f"A failed {kind} attempt requires explicit retry_failed=True: {attempts_root}"
        )

    index = len(attempts) + 1
    directory = attempts_root / f"{index:03d}"
    directory.mkdir(exist_ok=False)
    _write_json(directory / "request.json", request.as_dict(), MAX_JSON_BYTES)
    _write_json(directory / "payload.json", payload, MAX_JSON_BYTES)
    running = {
        "schema_version": SCHEMA_VERSION, "kind": kind,
        "request_sha256": request_sha, "work_key": work_key,
        "attempt_index": index, "status": "running", "started_at": _now(),
    }
    _write_json(directory / "state.json", running, MAX_JSON_BYTES)
    response_path = directory / "response.json"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(cuda_device),
        "PAPER_EXP_PROFILE_TEST_WORKER": "1" if test_mode else "0",
        "PAPER_EXP_PROFILE_TEST_FAIL": "1" if test_failure else "0",
        "PAPER_EXP_PROFILE_TEST_OOM": "1" if test_oom else "0",
    })
    with ExitStack() as cleanup:
        worker_scratch = scratch
        if kind == "selected":
            _not_symlink(scratch)
            worker_scratch = Path(
                cleanup.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=f"selected-attempt-{index:03d}-",
                        dir=scratch,
                    )
                )
            )
            _not_symlink(worker_scratch)
            if worker_scratch.parent != scratch or not worker_scratch.is_dir():
                raise HardwareProfileStateError(
                    "Selected-worker scratch escaped its owned parent."
                )
        command = [
            sys.executable, "-m", "paper_exp.hardware_profile_run", "_worker", kind,
            str(directory / "request.json"), str(directory / "payload.json"),
            str(response_path), request_sha, str(worker_scratch),
        ]
        try:
            process = subprocess.run(
                command, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                timeout=worker_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            timeout_error = TimeoutError(
                f"{kind} worker exceeded {worker_timeout_seconds:g} seconds."
            )
            response = _failure(
                kind,
                request_sha,
                payload,
                timeout_error,
                error_type="WorkerTimeoutError",
            )
            _write_json(response_path, response, MAX_JSON_BYTES)
            _terminal(directory, running, "failed", response)
            raise HardwareProfileWorkerError(
                f"{kind} worker timed out after {worker_timeout_seconds:g} seconds."
            ) from error
        except OSError as error:
            response = _failure(kind, request_sha, payload, error)
            _write_json(response_path, response, MAX_JSON_BYTES)
            _terminal(directory, running, "failed", response)
            raise HardwareProfileWorkerError(
                f"{kind} worker could not start: {type(error).__qualname__}: {_message(error)}"
            ) from error
        try:
            response = _read_json(response_path)
            _header(response, kind, request_sha, payload)
            if response["status"] == "failed":
                _failed_response(response)
            elif kind == "repeat":
                _repeat_result(
                    response, request_sha=request_sha, expected=_parse_item(payload)
                )
            else:
                _selected_timing(
                    response, request_sha=request_sha, expected=_parse_candidate(payload)
                )
        except BaseException as error:
            response = _failure(kind, request_sha, payload, error)
            _write_json(response_path, response, MAX_JSON_BYTES)
        if process.returncode != 0 and response["status"] == "completed":
            response = _failure(
                kind, request_sha, payload,
                RuntimeError(f"Worker exited with code {process.returncode}."),
                error_type="WorkerExitError",
            )
            _write_json(response_path, response, MAX_JSON_BYTES)
        status = (
            "completed"
            if process.returncode == 0 and response["status"] == "completed"
            else "failed"
        )
        _terminal(directory, running, status, response)
        if status == "failed":
            failure = response.get("failure", {})
            raise HardwareProfileWorkerError(
                f"{kind} worker failed: {failure.get('type')}: {failure.get('message')}"
            )
        return response, False


def _terminal(
    directory: Path, running: dict[str, object], status: str, response: dict[str, Any]
) -> None:
    terminal = {
        **running, "status": status, "finished_at": _now(),
        "response_sha256": sha256(_json_bytes(response, MAX_JSON_BYTES)).hexdigest(),
    }
    _write_json(directory / "state.json", terminal, MAX_JSON_BYTES)


def _candidate_results(
    request: HardwareProfileRequest, envelopes: Sequence[dict[str, Any]], *, request_sha: str
) -> tuple[CandidateProfileResult, ...]:
    grouped: dict[int, list[ProfileRepeatResult]] = {
        value: [] for value in request.candidate_microbatches
    }
    for envelope, item in zip(envelopes, build_profile_work_items(request), strict=True):
        grouped[item.microbatch_sequences].append(
            _repeat_result(envelope, request_sha=request_sha, expected=item)
        )
    return tuple(
        CandidateProfileResult(
            candidate.microbatch_sequences, candidate.gradient_accumulation_steps,
            tuple(grouped[candidate.microbatch_sequences]),
        )
        for candidate in request.candidates
    )


def _repeat_result(
    envelope: Mapping[str, Any], *, request_sha: str, expected: HardwareProfileWorkItem
) -> ProfileRepeatResult:
    _header(envelope, "repeat", request_sha, expected.as_dict())
    fields = {
        "schema_version", "kind", "request_sha256", "payload", "status", "identity", "result"
    }
    result_fields = {
        "microbatch_sequences", "repeat_index", "fit", "error",
        "synchronized_seconds", "tokens_per_second", "peak_allocated_bytes",
        "peak_reserved_bytes", "total_vram_bytes",
    }
    result = envelope.get("result")
    if set(envelope) != fields or not isinstance(result, Mapping) or set(result) != result_fields:
        raise HardwareProfileStateError("Repeat response fields are not exact.")
    parsed = ProfileRepeatResult(**dict(result))
    if (parsed.microbatch_sequences, parsed.repeat_index) != (
        expected.microbatch_sequences, expected.repeat_index
    ):
        raise HardwareProfileStateError("Repeat result identity mismatch.")
    if not parsed.fit and parsed.error != "cuda_out_of_memory":
        raise HardwareProfileStateError("Only CUDA OOM may be recorded as fit=false.")
    _identity(envelope["identity"])
    return parsed


def _selected_timing(
    envelope: Mapping[str, Any], *, request_sha: str, expected: MicrobatchCandidate
) -> dict[str, Any]:
    _header(envelope, "selected", request_sha, expected.as_dict())
    fields = {
        "schema_version", "kind", "request_sha256", "payload", "status", "identity", "timing"
    }
    timing_fields = {
        "setup_seconds", "validation_seconds", "checkpoint_seconds",
        "checkpoint_sha256", "checkpoint_bytes",
    }
    timing = envelope.get("timing")
    if set(envelope) != fields or not isinstance(timing, Mapping) or set(timing) != timing_fields:
        raise HardwareProfileStateError("Selected timing fields are not exact.")
    for field in ("setup_seconds", "validation_seconds", "checkpoint_seconds"):
        value = timing[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise HardwareProfileStateError(f"{field} must be positive.")
    if not _sha(timing["checkpoint_sha256"]):
        raise HardwareProfileStateError("checkpoint_sha256 is invalid.")
    size = timing["checkpoint_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HardwareProfileStateError("checkpoint_bytes must be positive.")
    _identity(envelope["identity"])
    return dict(timing)


def _consistent_identity(envelopes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not envelopes:
        raise HardwareProfileStateError("A profile must contain repeat workers.")
    identity = envelopes[0]["identity"]
    _identity(identity)
    expected = _json_bytes(identity, MAX_JSON_BYTES)
    if any(_json_bytes(value["identity"], MAX_JSON_BYTES) != expected for value in envelopes[1:]):
        raise HardwareProfileStateError("Repeat workers have inconsistent Git/environment identity.")
    return dict(identity)


def _run_child(args: Sequence[str]) -> int:
    if len(args) != 6:
        return 2
    kind, request_path, payload_path, response_path, request_sha, scratch = args
    payload: dict[str, object] = {}
    try:
        if kind not in {"repeat", "selected"}:
            raise ValueError("Unknown worker kind.")
        request = _parse_request(_read_json(Path(request_path)))
        if _request_sha(request) != request_sha:
            raise ValueError("Worker request SHA mismatch.")
        loaded = _read_json(Path(payload_path))
        if not isinstance(loaded, dict):
            raise TypeError("Worker payload must be an object.")
        payload = loaded
        if os.environ.get("PAPER_EXP_PROFILE_TEST_FAIL") == "1":
            raise RuntimeError("injected test worker failure")
        if os.environ.get("PAPER_EXP_PROFILE_TEST_WORKER") == "1":
            response = _test_response(kind, request, request_sha, payload)
        else:
            from paper_exp.hardware_profile_worker import collect_hardware_profile_identity
            from paper_exp.hardware_profile_worker import run_hardware_profile_repeat
            from paper_exp.hardware_profile_worker import run_selected_candidate_timing

            identity = collect_hardware_profile_identity(request)
            _identity(identity)
            if kind == "repeat":
                result = run_hardware_profile_repeat(request, _parse_item(payload))
                response = _success(kind, request_sha, payload, identity, "result", result.as_dict())
            else:
                timing = run_selected_candidate_timing(
                    request, _parse_candidate(payload), checkpoint_scratch=scratch
                )
                response = _success(kind, request_sha, payload, identity, "timing", timing.as_dict())
        _write_json(Path(response_path), response, MAX_JSON_BYTES)
        return 0
    except BaseException as error:
        safe_kind = kind if kind in {"repeat", "selected"} else "invalid"
        try:
            _write_json(
                Path(response_path), _failure(safe_kind, request_sha, payload, error),
                MAX_JSON_BYTES,
            )
        except BaseException:
            return 2
        return 1


def _test_response(
    kind: str, request: HardwareProfileRequest, request_sha: str,
    payload: dict[str, object],
) -> dict[str, object]:
    identity = _test_identity(request)
    if kind == "selected":
        candidate = _parse_candidate(payload)
        if candidate.microbatch_sequences not in request.candidate_microbatches:
            raise ValueError("Selected test candidate is absent from request.")
        timing = {
            "setup_seconds": 1.25, "validation_seconds": 2.5,
            "checkpoint_seconds": 0.75, "checkpoint_sha256": "b" * 64,
            "checkpoint_bytes": 1234,
        }
        return _success(kind, request_sha, payload, identity, "timing", timing)
    item = _parse_item(payload)
    if os.environ.get("PAPER_EXP_PROFILE_TEST_OOM") == "1":
        result = ProfileRepeatResult(
            item.microbatch_sequences, item.repeat_index, False, "cuda_out_of_memory",
            None, None, 40_000_000_000, 48_000_000_000, 48_000_000_000,
        )
    else:
        throughput = float(1_000 * item.microbatch_sequences)
        measured = 3 * request.global_sequences * request.sequence_length
        result = ProfileRepeatResult(
            item.microbatch_sequences, item.repeat_index, True, None,
            measured / throughput, throughput,
            8_000_000_000 + item.microbatch_sequences,
            12_000_000_000 + item.microbatch_sequences, 48_000_000_000,
        )
    return _success(kind, request_sha, payload, identity, "result", result.as_dict())


def _owned_root(
    work_root: str | Path, checkpoint_scratch: str | Path, *,
    request: HardwareProfileRequest, request_sha: str,
    worker_timeout_seconds: float, container_image: str,
) -> tuple[Path, Path]:
    root, scratch = _safe_path(work_root), _safe_path(checkpoint_scratch)
    if scratch.parent != root:
        raise ValueError("checkpoint_scratch must be a direct child of work_root.")
    if scratch.name in {OWNER_FILE, "repeats", "selected", ARTIFACT_FILE, ARTIFACT_SHA_FILE}:
        raise ValueError("checkpoint_scratch uses a reserved work-root name.")
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    if not root.exists():
        root.mkdir(exist_ok=False)
    marker = {
        "schema_version": SCHEMA_VERSION, "artifact_type": "hardware_profile_work_root",
        "request_sha256": request_sha, "request": request.as_dict(),
        "worker_timeout_seconds": worker_timeout_seconds,
        "container_image": container_image,
    }
    owner = root / OWNER_FILE
    if not any(root.iterdir()):
        _write_json(owner, marker, MAX_JSON_BYTES)
    else:
        try:
            existing = _read_json(owner)
        except HardwareProfileStateError as error:
            raise HardwareProfileStateError("Work root ownership/request marker is invalid.") from error
        if existing != marker:
            raise HardwareProfileStateError("Work root ownership/request marker is invalid.")
    allowed = {OWNER_FILE, "repeats", "selected", scratch.name, ARTIFACT_FILE, ARTIFACT_SHA_FILE}
    entries = list(root.iterdir())
    for path in entries:
        _not_symlink(path)
    unexpected = sorted(path.name for path in entries if path.name not in allowed)
    if unexpected:
        raise HardwareProfileStateError(f"Unexpected work-root entries: {unexpected}")
    for name in ("repeats", "selected"):
        if (root / name).exists() and not (root / name).is_dir():
            raise HardwareProfileStateError(f"Owned work-root directory is invalid: {name}")
    for name in (ARTIFACT_FILE, ARTIFACT_SHA_FILE):
        if (root / name).exists() and not (root / name).is_file():
            raise HardwareProfileStateError(f"Owned work-root artifact is invalid: {name}")
    if scratch.exists() and not scratch.is_dir():
        raise NotADirectoryError(scratch)
    scratch.mkdir(exist_ok=True)
    _not_symlink(scratch)
    if any(scratch.iterdir()):
        raise HardwareProfileStateError("Checkpoint scratch must be empty.")
    (root / "repeats").mkdir(exist_ok=True)
    _not_symlink(root / "repeats")
    return root, scratch


def _attempt_dirs(path: Path) -> list[Path]:
    result: list[Path] = []
    for entry in sorted(path.iterdir(), key=lambda value: value.name):
        _not_symlink(entry)
        if not entry.is_dir() or len(entry.name) != 3 or not entry.name.isdigit():
            raise HardwareProfileStateError(f"Malformed attempt entry: {entry}")
        allowed = {"request.json", "payload.json", "state.json", "response.json"}
        if any(child.name not in allowed for child in entry.iterdir()):
            raise HardwareProfileStateError(f"Unexpected attempt entry: {entry}")
        result.append(entry)
    return result


def _state(
    value: Any, kind: str, request_sha: str, work_key: str, index: int
) -> None:
    if not isinstance(value, Mapping) or value.get("status") not in {"running", "failed", "completed"}:
        raise HardwareProfileStateError("Attempt state/status is invalid.")
    fields = _STATE_COMMON if value["status"] == "running" else _STATE_COMMON | {
        "finished_at", "response_sha256"
    }
    identity = (
        value.get("schema_version"), value.get("kind"), value.get("request_sha256"),
        value.get("work_key"), value.get("attempt_index"),
    )
    if set(value) != fields or identity != (SCHEMA_VERSION, kind, request_sha, work_key, index):
        raise HardwareProfileStateError("Attempt state identity/fields are invalid.")
    if value["status"] != "running" and not _sha(value["response_sha256"]):
        raise HardwareProfileStateError("Attempt response SHA is invalid.")


def _header(
    value: Mapping[str, Any], kind: str, request_sha: str, payload: dict[str, object]
) -> None:
    if not isinstance(value, Mapping) or (
        value.get("schema_version"), value.get("kind"), value.get("request_sha256"),
        value.get("payload"), value.get("status"),
    ) != (SCHEMA_VERSION, kind, request_sha, payload, value.get("status")):
        raise HardwareProfileStateError("Worker response identity is invalid.")
    if value.get("status") not in {"completed", "failed"}:
        raise HardwareProfileStateError("Worker response status is invalid.")


def _failed_response(value: Mapping[str, Any]) -> None:
    failure = value.get("failure")
    fields = {"schema_version", "kind", "request_sha256", "payload", "status", "failure"}
    if set(value) != fields or not isinstance(failure, Mapping) or set(failure) != {"type", "message"}:
        raise HardwareProfileStateError("Failed response fields are not exact.")
    if any(not isinstance(failure.get(key), str) for key in ("type", "message")):
        raise HardwareProfileStateError("Worker failure values must be strings.")


def _identity(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise HardwareProfileStateError("Worker identity fields are not exact.")
    if not _immutable_sha(value["repo_git_commit"]) or value["repo_git_dirty"] is not False:
        raise HardwareProfileStateError(
            "Worker Git identity must be an immutable clean checkout."
        )
    strings = (
        "hostname", "platform", "python_executable", "python_version",
        "cuda_runtime", "nvidia_driver_version", "cuda_visible_devices",
    )
    if any(not isinstance(value[key], str) or not value[key].strip() for key in strings):
        raise HardwareProfileStateError("Worker environment identity is invalid.")
    if _NVIDIA_DRIVER_RE.fullmatch(value["nvidia_driver_version"]) is None:
        raise HardwareProfileStateError("Worker NVIDIA driver identity is invalid.")
    if not value["cuda_visible_devices"].isdigit() or not isinstance(value["package_versions"], Mapping):
        raise HardwareProfileStateError("Worker CUDA/package identity is invalid.")
    gpu = value["gpu"]
    if not isinstance(gpu, Mapping) or set(gpu) != _GPU_FIELDS:
        raise HardwareProfileStateError("GPU identity fields are not exact.")
    if any(not isinstance(gpu[key], str) or not gpu[key].strip() for key in ("uuid", "name")):
        raise HardwareProfileStateError("GPU UUID/name is missing.")
    memory = gpu["total_vram_bytes"]
    capability = gpu["compute_capability"]
    if (
        gpu["bf16_supported"] is not True
        or isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0
        or not isinstance(capability, list) or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability)
    ):
        raise HardwareProfileStateError("GPU VRAM/BF16/capability identity is invalid.")


def _require_device(envelope: Mapping[str, Any], cuda_device: int) -> None:
    if envelope["identity"]["cuda_visible_devices"] != str(cuda_device):
        raise HardwareProfileStateError("Worker used an unexpected CUDA device.")


def _publish(path: Path, sha_path: Path, data: bytes, digest: str) -> None:
    _not_symlink(path)
    _not_symlink(sha_path)
    if not _sha(digest) or sha256(data).hexdigest() != digest:
        raise HardwareProfileStateError("Profile artifact digest does not match its bytes.")
    sidecar = f"{digest}  {ARTIFACT_FILE}\n".encode("ascii")
    artifact_exists = path.exists()
    checksum_exists = sha_path.exists()
    if artifact_exists and not path.is_file():
        raise HardwareProfileStateError("Existing profile artifact is not a file.")
    if checksum_exists and not sha_path.is_file():
        raise HardwareProfileStateError("Existing profile checksum is not a file.")
    if artifact_exists and path.read_bytes() != data:
        raise HardwareProfileStateError("Existing profile artifact differs.")
    if checksum_exists and sha_path.read_bytes() != sidecar:
        raise HardwareProfileStateError("Existing profile checksum differs.")
    if artifact_exists and checksum_exists:
        return

    # Publish the checksum first and the report last.  An exact checksum-only
    # state is therefore a recoverable interrupted publication.  An exact
    # report-only state from the former report-first implementation is also
    # reconciled, then atomically republished after its checksum.
    if not checksum_exists:
        _write_bytes(sha_path, sidecar)
    _write_bytes(path, data)


def _parse_request(value: Any) -> HardwareProfileRequest:
    fields = {
        "architecture", "revision", "gpu_class", "sequence_length",
        "global_sequences", "candidate_microbatches", "repeats",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Hardware profile request fields are not exact.")
    return HardwareProfileRequest(
        architecture=value["architecture"], revision=value["revision"],
        gpu_class=value["gpu_class"], sequence_length=value["sequence_length"],
        global_sequences=value["global_sequences"],
        candidate_microbatches=tuple(value["candidate_microbatches"]),
        repeats=value["repeats"],
    )


def _parse_item(value: Mapping[str, Any]) -> HardwareProfileWorkItem:
    fields = {
        "microbatch_sequences", "gradient_accumulation_steps",
        "repeat_index", "synthetic_grouping_hash",
    }
    if set(value) != fields:
        raise ValueError("Hardware profile work-item fields are not exact.")
    return HardwareProfileWorkItem(**dict(value))


def _parse_candidate(value: Mapping[str, Any]) -> MicrobatchCandidate:
    if set(value) != {"microbatch_sequences", "gradient_accumulation_steps"}:
        raise ValueError("Selected candidate fields are not exact.")
    return MicrobatchCandidate(**dict(value))


def _success(
    kind: str, request_sha: str, payload: dict[str, object], identity: dict[str, object],
    result_key: str, result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION, "kind": kind,
        "request_sha256": request_sha, "payload": payload, "status": "completed",
        "identity": identity, result_key: result,
    }


def _failure(
    kind: str, request_sha: str, payload: dict[str, object], error: BaseException,
    *, error_type: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION, "kind": kind,
        "request_sha256": request_sha, "payload": payload, "status": "failed",
        "failure": {"type": error_type or type(error).__qualname__, "message": _message(error)},
    }


def _test_identity(request: HardwareProfileRequest) -> dict[str, object]:
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    return {
        "repo_git_commit": "a" * 40, "repo_git_dirty": False,
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python_executable": sys.executable, "python_version": sys.version,
        "package_versions": {"paper-exp": "test"}, "cuda_runtime": "test",
        "nvidia_driver_version": "999.0",
        "cuda_visible_devices": visible,
        "gpu": {
            "uuid": f"TEST-GPU-{visible}", "name": request.gpu_class,
            "total_vram_bytes": 48_000_000_000, "bf16_supported": True,
            "compute_capability": [9, 9],
        },
    }


def _request_sha(request: HardwareProfileRequest) -> str:
    return sha256(_json_bytes(request.as_dict(), MAX_JSON_BYTES)).hexdigest()


def _repeat_key(item: HardwareProfileWorkItem) -> str:
    return f"mb-{item.microbatch_sequences:03d}-repeat-{item.repeat_index:03d}"


def _safe_path(value: str | Path) -> Path:
    path = Path(value)
    path = Path(os.path.abspath(path if path.is_absolute() else Path.cwd() / path))
    for candidate in reversed((path, *path.parents)):
        _not_symlink(candidate)
    return path


def _not_symlink(path: Path) -> None:
    is_junction = getattr(path, "is_junction", None)
    if path.is_symlink() or (callable(is_junction) and is_junction()):
        raise HardwareProfileStateError(
            f"Symlink and junction paths are prohibited: {path}"
        )


def _read_json(path: Path) -> Any:
    _not_symlink(path)
    if not path.is_file():
        raise HardwareProfileStateError(f"Required JSON file is missing: {path}")
    data = path.read_bytes()
    if not data or len(data) > MAX_JSON_BYTES:
        raise HardwareProfileStateError(f"JSON IPC size is invalid: {path}")
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_distinct,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HardwareProfileStateError(f"Malformed JSON IPC: {path}") from error


def _distinct(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(value: object, limit: int) -> bytes:
    try:
        data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise HardwareProfileStateError("Value is not finite JSON.") from error
    if len(data) > limit:
        raise HardwareProfileStateError(f"JSON exceeds the {limit}-byte limit.")
    return data


def _write_json(path: Path, value: object, limit: int) -> None:
    _write_bytes(path, _json_bytes(value, limit))


def _write_bytes(path: Path, data: bytes) -> None:
    _not_symlink(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _immutable_sha(value: object) -> bool:
    return isinstance(value, str) and 40 <= len(value) <= 64 and all(c in "0123456789abcdef" for c in value)


def _positive_finite_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("worker_timeout_seconds must be numeric.")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker_timeout_seconds must be finite and positive.")
    return timeout


def _immutable_container_image(value: object) -> str:
    if not isinstance(value, str) or _CONTAINER_IMAGE_RE.fullmatch(value) is None:
        raise ValueError(
            "container_image must be a nonempty immutable image reference ending "
            "in @sha256:<64 lowercase hex>."
        )
    return value


def _message(error: BaseException) -> str:
    return str(error)[:MAX_FAILURE_CHARS]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        raise SystemExit(_run_child(sys.argv[2:]))
    raise SystemExit(2)
