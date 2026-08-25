"""Infrastructure-only proof of bounded concurrent subprocess execution.

Recovery and completed-work reuse are exercised as separate passes within one
coordinator invocation.  This smoke does not claim recovery after coordinator
interruption.
"""
from __future__ import annotations

from dataclasses import dataclass
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
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

from paper_exp.parallel import ParallelReport, ParallelRunError, WorkItem, WorkerSlot, run_bounded
from paper_exp.utils import collect_git_commit, collect_git_dirty
from paper_exp.utils import collect_package_versions

SCHEMA_VERSION = 1
SCENARIO_ID = "failure-recovery-restart"
SMOKE_TASK_IDS = ("infra-task-01", "infra-task-02", "infra-task-03")
REPORT_NAME = "concurrent-smoke-report.json"
REPORT_HASH_NAME = "concurrent-smoke-report.sha256"
PASS_IDS = ("injected-failure", "explicit-recovery", "completed-restart")
WORKER_PASS_IDS = PASS_IDS[:2]
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_ID_RE = re.compile(r"^[0-9]{3}$")
REQUEST_KEYS = {
    "schema_version", "scientific_evidence", "scenario_id", "pass_id",
    "task_id", "task_index", "task_input_sha256", "attempt_id", "slot_id",
    "device_mapping", "require_cuda", "sleep_seconds", "barrier_timeout",
    "barrier_tasks", "inject_failure",
}
MANIFEST_KEYS = {
    "schema_version", "scientific_evidence", "artifact_type", "request_sha256",
    "task_id", "attempt_id", "slot_id", "device_mapping", "status",
    "process_id", "parent_pid", "hostname", "git_commit", "git_dirty",
    "git_identity_sha256", "environment", "environment_sha256", "module_sha256",
    "slot_mapping_sha256", "gpu", "payload_started_ns", "finished_ns",
}
GPU_KEYS = {
    "visible_name", "visible_vram_bytes", "visible_device_count",
    "bf16_supported", "bf16_operation_verified", "physical_uuid",
    "physical_name", "physical_vram_mib",
}

class InfrastructureSmokeError(RuntimeError):
    """The infrastructure smoke did not prove its operational contract."""

@dataclass(frozen=True)
class _Task:
    task_id: str
    index: int
    input_sha256: str

@dataclass(frozen=True)
class _State:
    status: str
    attempt_dir: Path | None = None
    manifest: Mapping[str, Any] | None = None

def run_concurrent_infrastructure_smoke(
    slots: Sequence[WorkerSlot[str]],
    work_root: str | Path,
    *,
    require_cuda: bool,
    task_sleep_seconds: float = 0.10,
    barrier_timeout_seconds: float = 60.0,
    allow_shared_gpu: bool = False,
) -> dict[str, Any]:
    """Prove drain, recovery, reuse, and isolation with fresh subprocesses.

    ``work_root`` should be a new child of an active infrastructure smoke run.
    This function loads no model, data, training, or scientific config.
    Its recovery/restart proof is scoped to this single invocation.
    """
    if not isinstance(require_cuda, bool):
        raise TypeError("require_cuda must be a bool")
    if not isinstance(allow_shared_gpu, bool):
        raise TypeError("allow_shared_gpu must be a bool")
    worker_slots = _validate_slots(slots)
    for label, value in (
        ("task_sleep_seconds", task_sleep_seconds),
        ("barrier_timeout_seconds", barrier_timeout_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be positive and finite")
    if task_sleep_seconds < 0.02:
        raise ValueError("task_sleep_seconds must be at least 0.02 for overlap proof.")
    root = _empty_owned_root_candidate(work_root)
    tasks = _tasks(require_cuda)
    coordinator = _provenance()
    if require_cuda:
        _require_clean_git(coordinator, label="coordinator")
    root.mkdir(exist_ok=True)
    _require_real_directory(root, label="infrastructure smoke root")
    failed = _run_pass(
        root, "injected-failure", tasks, worker_slots, require_cuda, task_sleep_seconds,
        barrier_timeout_seconds, tasks[0].task_id, False,
    )
    first = _state(root, tasks[0]).manifest
    sibling = _state(root, tasks[1]).manifest
    assert first is not None and sibling is not None
    overlap_ns = max(
        0,
        min(first["finished_ns"], sibling["finished_ns"])
        - max(first["payload_started_ns"], sibling["payload_started_ns"]),
    )
    _require(overlap_ns > 0, "the initially admitted subprocesses did not overlap")
    _require(
        failed["unadmitted_task_ids"] == [tasks[2].task_id],
        "the coordinator admitted work after observing failure",
    )
    _require(
        first["status"] == "failed"
        and sibling["status"] == "completed"
        and sibling["finished_ns"] > first["finished_ns"]
        and failed["finished_ns"] >= sibling["finished_ns"],
        "the admitted sibling was not drained after failure",
    )
    recovered = _run_pass(
        root, "explicit-recovery", tasks, worker_slots, require_cuda, task_sleep_seconds,
        barrier_timeout_seconds, None, True,
    )
    _require(
        recovered["skipped_task_ids"] == [tasks[1].task_id]
        and sorted(result["task_id"] for result in recovered["results"])
        == sorted((tasks[0].task_id, tasks[2].task_id))
        and all(result["status"] == "completed" for result in recovered["results"]),
        "explicit recovery did not reuse the sibling and complete remaining work",
    )
    restarted = _run_pass(
        root, "completed-restart", tasks, worker_slots, require_cuda, task_sleep_seconds,
        barrier_timeout_seconds, None, False,
    )
    _require(
        not restarted["results"]
        and restarted["skipped_task_ids"] == list(SMOKE_TASK_IDS),
        "completed restart did not skip every coherent completed task",
    )
    manifests = [
        _read_json(attempt / "manifest.json")
        for task in tasks
        for attempt in sorted((root / "tasks" / task.task_id).iterdir())
    ]
    _require(len(manifests) == 4, "the script did not preserve exactly four attempts")
    _require(
        all(manifest["process_id"] != coordinator["process_id"] for manifest in manifests),
        "a smoke item did not run in a subprocess",
    )
    _require(
        all(manifest["parent_pid"] == coordinator["process_id"] for manifest in manifests),
        "a smoke worker was not a direct coordinator subprocess",
    )
    _require(
        all(
            manifest["git_identity_sha256"] == coordinator["git_identity_sha256"]
            and manifest["environment_sha256"] == coordinator["environment_sha256"]
            and manifest["module_sha256"] == coordinator["module_sha256"]
            for manifest in manifests
        ),
        "worker Git or environment identity differs from the coordinator",
    )
    gpu_slots = _validate_gpu_history(
        manifests,
        worker_slots,
        require_cuda=require_cuda,
        allow_shared_gpu=allow_shared_gpu,
    )
    gpu_uuids = (
        [gpu_slots[slot.slot_id]["physical_uuid"] for slot in worker_slots]
        if require_cuda
        else []
    )
    references = []
    for manifest in manifests:
        attempt = root / "tasks" / manifest["task_id"] / manifest["attempt_id"]
        references.append(
            {
                "task_id": manifest["task_id"],
                "status": manifest["status"],
                "attempt_dir": attempt.relative_to(root).as_posix(),
                "manifest_sha256": _sha256((attempt / "manifest.json").read_bytes()),
                "request_sha256": manifest["request_sha256"],
            }
        )
    _require(
        len({reference["attempt_dir"] for reference in references}) == len(references),
        "worker artifact roots are not disjoint",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "concurrent_infrastructure_smoke",
        "scientific_evidence": False,
        "mode": "remote_cuda" if require_cuda else "local",
        "recovery_scope": "single_coordinator_invocation",
        "coordinator_interruption_recovery_proved": False,
        "allow_shared_gpu": allow_shared_gpu,
        "coordinator": coordinator,
        "slots": [
            {
                "slot_id": slot.slot_id,
                "cuda_visible_devices": slot.payload,
                "mapping_sha256": _sha256((slot.payload or "").encode()),
            }
            for slot in worker_slots
        ],
        "passes": {
            "injected_failure": failed,
            "explicit_recovery": recovered,
            "completed_restart": restarted,
        },
        "evidence": {
            "initial_overlap_nanoseconds": overlap_ns,
            "initial_physical_gpu_uuids": gpu_uuids if require_cuda else [],
            "stable_gpu_identity_by_slot": gpu_slots if require_cuda else {},
        },
        "attempts": references,
        "passed": True,
        "finished_ns": time.time_ns(),
    }
    _validate_state_tree(root, tasks)
    _publish_report(root, report)
    return report

def _validate_slots(slots: Sequence[WorkerSlot[str]]) -> tuple[WorkerSlot[str], ...]:
    values = tuple(slots)
    if len(values) != 2:
        raise ValueError("The deterministic smoke requires exactly two worker slots.")
    for slot in values:
        if not isinstance(slot, WorkerSlot):
            raise TypeError("slots must contain WorkerSlot values.")
        if (
            not isinstance(slot.slot_id, str)
            or slot.slot_id != slot.slot_id.strip()
            or not slot.slot_id
        ):
            raise ValueError("Each slot needs a nonempty slot ID.")
        if (
            not isinstance(slot.payload, str)
            or slot.payload != slot.payload.strip()
            or not slot.payload
            or "," in slot.payload
        ):
            raise ValueError("Each slot needs exactly one explicit GPU mapping.")
    if values[0].slot_id == values[1].slot_id:
        raise ValueError("Worker slot IDs must be distinct.")
    return values

def _tasks(require_cuda: bool) -> tuple[_Task, ...]:
    return tuple(
        _Task(
            task_id,
            index,
            _json_sha256(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": SCENARIO_ID,
                    "task_id": task_id,
                    "task_index": index,
                    "require_cuda": require_cuda,
                    "scientific_evidence": False,
                }
            ),
        )
        for index, task_id in enumerate(SMOKE_TASK_IDS, 1)
    )

def _run_pass(
    root: Path, pass_id: str, tasks: tuple[_Task, ...],
    slots: tuple[WorkerSlot[str], ...], require_cuda: bool, sleep_seconds: float,
    barrier_timeout: float, fail_task_id: str | None, retry_failed: bool,
) -> dict[str, Any]:
    _validate_state_tree(root, tasks)
    skipped: list[str] = []
    pending: list[_Task] = []
    for task in tasks:
        state = _state(root, task)
        if state.status == "completed":
            skipped.append(task.task_id)
        elif state.status == "failed":
            if not retry_failed:
                raise InfrastructureSmokeError(f"{task.task_id} requires explicit recovery")
            pending.append(task)
        else:
            pending.append(task)
    barrier_tasks = [task.task_id for task in pending[:2]]
    def worker(item: WorkItem[_Task], slot: WorkerSlot[str]) -> None:
        return _spawn_worker(
            root, pass_id, item.payload, slot, require_cuda, sleep_seconds,
            barrier_timeout, barrier_tasks, item.config_id == fail_task_id,
        )
    parallel_error: ParallelRunError | None = None
    try:
        parallel_report = run_bounded(
            tuple(WorkItem(task.task_id, task) for task in pending), slots, worker
        )
    except ParallelRunError as error:
        parallel_error = error
        parallel_report = error.report
    if (parallel_error is not None) != (fail_task_id is not None):
        raise InfrastructureSmokeError(f"Pass {pass_id!r} had an unexpected outcome")
    return _pass_record(pass_id, time.time_ns(), skipped, parallel_report)

def _spawn_worker(
    root: Path, pass_id: str, task: _Task, slot: WorkerSlot[str],
    require_cuda: bool, sleep_seconds: float, barrier_timeout: float,
    barrier_tasks: list[str], inject_failure: bool,
) -> None:
    attempt = _new_attempt(root / "tasks" / task.task_id)
    assert isinstance(slot.payload, str)
    request = {
        "schema_version": SCHEMA_VERSION,
        "scientific_evidence": False,
        "scenario_id": SCENARIO_ID,
        "pass_id": pass_id,
        "task_id": task.task_id,
        "task_index": task.index,
        "task_input_sha256": task.input_sha256,
        "attempt_id": attempt.name,
        "slot_id": slot.slot_id,
        "device_mapping": slot.payload,
        "require_cuda": require_cuda,
        "sleep_seconds": sleep_seconds,
        "barrier_timeout": barrier_timeout,
        "barrier_tasks": barrier_tasks,
        "inject_failure": inject_failure,
    }
    request_path = attempt / "request.json"
    _atomic_write_json(request_path, request)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = slot.payload
    completed = subprocess.run(
        [sys.executable, "-m", "paper_exp.infrastructure_smoke", "--worker", str(request_path)],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    state = _state(root, task)
    if state.attempt_dir != attempt or state.manifest is None:
        raise InfrastructureSmokeError(f"{task.task_id} did not publish its attempt")
    expected_status = "completed" if completed.returncode == 0 else "failed"
    if state.manifest["status"] != expected_status:
        raise InfrastructureSmokeError(f"{task.task_id} exit and manifest disagree")
    if completed.returncode:
        failure = state.manifest.get("failure", {})
        raise InfrastructureSmokeError(
            f"{task.task_id} failed on {slot.slot_id}: {failure.get('message', 'worker exit')}"
        )
    return None

def _pass_record(
    pass_id: str, finished_ns: int, skipped: list[str], report: ParallelReport[object]
) -> dict[str, Any]:
    return {
        "pass_id": pass_id,
        "finished_ns": finished_ns,
        "skipped_task_ids": skipped,
        "results": [
            {
                "task_id": result.assignment.config_id,
                "slot_id": result.assignment.slot_id,
                "status": result.status,
            }
            for result in report.results
        ],
        "unadmitted_task_ids": list(report.unadmitted_config_ids),
    }

def _state(root: Path, task: _Task) -> _State:
    root = _lexical_absolute(root)
    if not _path_exists(root):
        return _State("absent")
    _require_real_directory(root, label="infrastructure smoke root")
    tasks_root = root / "tasks"
    task_root = tasks_root / task.task_id
    if not _path_exists(tasks_root):
        return _State("absent")
    _require_real_directory(tasks_root, label="tasks root")
    if not _path_exists(task_root):
        return _State("absent")
    _require_real_directory(task_root, label="task root")
    entries = sorted(task_root.iterdir())
    if not entries:
        return _State("absent")
    statuses: list[str] = []
    latest: tuple[Path, Mapping[str, Any]] | None = None
    for sequence, attempt in enumerate(entries, 1):
        if attempt.name != f"{sequence:03d}":
            raise InfrastructureSmokeError(f"Invalid attempt entry: {attempt}")
        _require_real_directory(attempt, label="attempt directory")
        _require_exact_entries(attempt, {"request.json", "manifest.json"})
        request_path = attempt / "request.json"
        request, request_bytes = _load_json_object(request_path)
        _validate_worker_request(request, request_path)
        if (
            request["task_id"] != task.task_id
            or request["task_index"] != task.index
            or request["task_input_sha256"] != task.input_sha256
            or request["attempt_id"] != attempt.name
        ):
            raise InfrastructureSmokeError(f"Request identity mismatch: {attempt}")
        manifest, _ = _load_json_object(attempt / "manifest.json")
        _validate_terminal_manifest(manifest, request, request_bytes, attempt)
        status = manifest["status"]
        statuses.append(status)
        latest = (attempt, manifest)
    if statuses.count("completed") > 1 or "completed" in statuses[:-1]:
        raise InfrastructureSmokeError(f"Incoherent attempt history for {task.task_id}")
    assert latest is not None
    return _State(statuses[-1], latest[0], latest[1])

def _worker(request_path: Path) -> int:
    request_path, attempt, _ = _validate_worker_request_path(request_path)
    _require_exact_entries(attempt, {"request.json"})
    request, request_bytes = _load_json_object(request_path)
    _validate_worker_request(request, request_path)
    manifest_path = attempt / "manifest.json"
    if _path_exists(manifest_path):
        raise InfrastructureSmokeError(f"Worker manifest already exists: {manifest_path}")
    provenance = _provenance()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scientific_evidence": False,
        "artifact_type": "infrastructure_smoke_worker",
        "request_sha256": _sha256(request_bytes),
        "task_id": request["task_id"],
        "attempt_id": request["attempt_id"],
        "slot_id": request["slot_id"],
        "device_mapping": request["device_mapping"],
        "status": "running",
        **provenance,
        "slot_mapping_sha256": _sha256(request["device_mapping"].encode()),
        "gpu": None,
        "payload_started_ns": None,
        "finished_ns": None,
    }
    _atomic_write_json(manifest_path, manifest)
    exit_code = 0
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != request["device_mapping"]:
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the assigned slot")
        if request["require_cuda"]:
            _require_clean_git(provenance, label="worker")
        manifest["gpu"] = (
            _collect_remote_gpu_identity(request["device_mapping"])
            if request["require_cuda"]
            else None
        )
        _barrier(attempt.parents[2], request)
        if request["require_cuda"]:
            _run_bf16_cuda_operation()
            assert isinstance(manifest["gpu"], dict)
            manifest["gpu"]["bf16_operation_verified"] = True
        manifest["payload_started_ns"] = time.time_ns()
        if request["inject_failure"]:
            time.sleep(min(request["sleep_seconds"] * 0.2, 0.02))
            raise RuntimeError("injected infrastructure smoke failure")
        time.sleep(request["sleep_seconds"])
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__qualname__, "message": str(error)}
        exit_code = 17
    finally:
        if manifest["payload_started_ns"] is None:
            manifest["payload_started_ns"] = time.time_ns()
        manifest["finished_ns"] = time.time_ns()
        _atomic_write_json(manifest_path, manifest)
    return exit_code

def _barrier(root: Path, request: Mapping[str, Any]) -> None:
    participants = request["barrier_tasks"]
    if request["task_id"] not in participants or len(participants) < 2:
        return
    _require_real_directory(root, label="infrastructure smoke root")
    barriers = root / "barriers"
    barriers.mkdir(exist_ok=True)
    _require_real_directory(barriers, label="barriers root")
    barrier = barriers / request["pass_id"]
    barrier.mkdir(parents=True, exist_ok=True)
    _require_real_directory(barrier, label="barrier directory")
    ready_path = barrier / f"{request['task_id']}.ready"
    if _path_exists(ready_path):
        raise InfrastructureSmokeError(f"Barrier marker already exists: {ready_path}")
    with ready_path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    deadline = time.monotonic() + request["barrier_timeout"]
    while True:
        markers = [barrier / f"{task_id}.ready" for task_id in participants]
        invalid = [path for path in markers if _path_exists(path) and not _is_real_file(path)]
        if invalid:
            raise InfrastructureSmokeError(f"Invalid barrier marker: {invalid[0]}")
        if all(_is_real_file(path) for path in markers):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("subprocess rendezvous timed out")
        time.sleep(0.005)

def _collect_remote_gpu_identity(
    device_mapping: str, *, torch_module: Any | None = None,
) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as error:
            raise RuntimeError("remote smoke requires Torch") from error
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("remote smoke requires CUDA availability")
    if cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {cuda.device_count()}")
    if not cuda.is_bf16_supported():
        raise RuntimeError("remote smoke requires BF16 support")
    properties = cuda.get_device_properties(0)
    gpu_uuid = getattr(properties, "uuid", None)
    if isinstance(gpu_uuid, bytes):
        gpu_uuid = gpu_uuid.decode("ascii")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_mapping}",
                    "--query-gpu=uuid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("could not resolve the physical GPU UUID") from error
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError("GPU mapping did not resolve to one physical UUID")
        gpu_uuid = rows[0]
    name = str(properties.name)
    vram_bytes = int(properties.total_memory)
    if not name or vram_bytes <= 0:
        raise RuntimeError("physical GPU identity is incomplete")
    return {
        "visible_name": name,
        "visible_vram_bytes": vram_bytes,
        "visible_device_count": 1,
        "bf16_supported": True,
        "bf16_operation_verified": False,
        "physical_uuid": gpu_uuid.strip(),
        "physical_name": name,
        "physical_vram_mib": vram_bytes // (1024 * 1024),
    }


def _run_bf16_cuda_operation(*, torch_module: Any | None = None) -> None:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as error:
            raise RuntimeError("remote smoke requires Torch") from error
    probe = torch_module.ones((16, 16), device="cuda:0", dtype=torch_module.bfloat16)
    result = torch_module.matmul(probe, probe)
    torch_module.cuda.synchronize()
    observed = float(result[0, 0].float().item())
    if observed != 16.0:
        raise RuntimeError(f"BF16 CUDA operation returned {observed!r}, expected 16.0")

def _provenance() -> dict[str, Any]:
    environment = {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": collect_package_versions(),
    }
    git_identity = {
        "commit": collect_git_commit(Path.cwd()),
        "dirty": collect_git_dirty(Path.cwd()),
    }
    return {
        "process_id": os.getpid(),
        "parent_pid": os.getppid(),
        "hostname": socket.gethostname(),
        "git_commit": git_identity["commit"],
        "git_dirty": git_identity["dirty"],
        "git_identity_sha256": _json_sha256(git_identity),
        "environment": environment,
        "environment_sha256": _json_sha256(environment),
        "module_sha256": _sha256(Path(__file__).read_bytes()),
    }


def _require_clean_git(provenance: Mapping[str, Any], *, label: str) -> None:
    commit = provenance.get("git_commit")
    if not isinstance(commit, str) or not commit or provenance.get("git_dirty") is not False:
        raise InfrastructureSmokeError(
            f"Remote CUDA smoke requires a clean non-null Git SHA for the {label}"
        )


def _empty_owned_root_candidate(work_root: str | Path) -> Path:
    root = _lexical_absolute(work_root)
    _require_no_link_ancestors(root.parent)
    _require_real_directory(root.parent, label="smoke root parent")
    if _path_exists(root):
        _require_real_directory(root, label="infrastructure smoke root")
        if any(root.iterdir()):
            raise FileExistsError(f"Infrastructure smoke root must be empty: {root}")
    return root


def _validate_state_tree(root: Path, tasks: Sequence[_Task]) -> None:
    _require_real_directory(root, label="infrastructure smoke root")
    allowed_root = {"tasks", "barriers"}
    unexpected_root = {entry.name for entry in root.iterdir()} - allowed_root
    if unexpected_root:
        raise InfrastructureSmokeError(
            f"Unexpected smoke-root entries: {', '.join(sorted(unexpected_root))}"
        )
    tasks_root = root / "tasks"
    if _path_exists(tasks_root):
        _require_real_directory(tasks_root, label="tasks root")
        allowed_tasks = {task.task_id for task in tasks}
        unexpected_tasks = {entry.name for entry in tasks_root.iterdir()} - allowed_tasks
        if unexpected_tasks:
            raise InfrastructureSmokeError(
                f"Unexpected task entries: {', '.join(sorted(unexpected_tasks))}"
            )
        for task_root in tasks_root.iterdir():
            _require_real_directory(task_root, label="task root")
    barriers = root / "barriers"
    if not _path_exists(barriers):
        return
    _require_real_directory(barriers, label="barriers root")
    expected_markers = {
        "injected-failure": {f"{SMOKE_TASK_IDS[0]}.ready", f"{SMOKE_TASK_IDS[1]}.ready"},
        "explicit-recovery": {f"{SMOKE_TASK_IDS[0]}.ready", f"{SMOKE_TASK_IDS[2]}.ready"},
    }
    unexpected_passes = {entry.name for entry in barriers.iterdir()} - set(expected_markers)
    if unexpected_passes:
        raise InfrastructureSmokeError(
            f"Unexpected barrier passes: {', '.join(sorted(unexpected_passes))}"
        )
    for pass_root in barriers.iterdir():
        _require_real_directory(pass_root, label="barrier directory")
        _require_exact_entries(pass_root, expected_markers[pass_root.name])
        for marker in pass_root.iterdir():
            _require_real_file(marker, label="barrier marker")
            try:
                value = marker.read_text(encoding="ascii").strip()
            except (OSError, UnicodeError) as error:
                raise InfrastructureSmokeError(f"Cannot read barrier marker: {marker}") from error
            if not value.isdecimal() or int(value) <= 0:
                raise InfrastructureSmokeError(f"Invalid barrier marker: {marker}")


def _new_attempt(task_root: Path) -> Path:
    tasks_root = task_root.parent
    tasks_root.mkdir(exist_ok=True)
    _require_real_directory(tasks_root, label="tasks root")
    task_root.mkdir(exist_ok=True)
    _require_real_directory(task_root, label="task root")
    sequence = len(list(task_root.iterdir())) + 1
    attempt = task_root / f"{sequence:03d}"
    attempt.mkdir(exist_ok=False)
    _require_real_directory(attempt, label="attempt directory")
    return attempt


def _validate_worker_request_path(request_path: Path) -> tuple[Path, Path, Path]:
    path = _lexical_absolute(request_path)
    _require_no_link_ancestors(path)
    _require_real_file(path, label="worker request")
    attempt = path.parent
    task_root = attempt.parent
    tasks_root = task_root.parent
    root = tasks_root.parent
    if (
        path.name != "request.json"
        or tasks_root.name != "tasks"
        or not ATTEMPT_ID_RE.fullmatch(attempt.name)
    ):
        raise InfrastructureSmokeError(f"Invalid worker request path: {path}")
    for value, label in (
        (root, "infrastructure smoke root"),
        (tasks_root, "tasks root"),
        (task_root, "task root"),
        (attempt, "attempt directory"),
    ):
        _require_real_directory(value, label=label)
    return path, attempt, root


def _validate_worker_request(request: Mapping[str, Any], request_path: Path) -> None:
    path, attempt, _ = _validate_worker_request_path(request_path)
    if set(request) != REQUEST_KEYS:
        raise InfrastructureSmokeError(f"Invalid request schema: {path}")
    if (
        request["schema_version"] != SCHEMA_VERSION
        or request["scientific_evidence"] is not False
        or request["scenario_id"] != SCENARIO_ID
    ):
        raise InfrastructureSmokeError(f"Invalid request constants: {path}")
    for key in ("require_cuda", "inject_failure"):
        if not isinstance(request[key], bool):
            raise InfrastructureSmokeError(f"Request {key} must be a bool: {path}")
    for key in ("sleep_seconds", "barrier_timeout"):
        value = request[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise InfrastructureSmokeError(f"Request {key} must be positive and finite: {path}")
    if request["sleep_seconds"] < 0.02:
        raise InfrastructureSmokeError(f"Request sleep is too short: {path}")
    task_id = request["task_id"]
    pass_id = request["pass_id"]
    if not isinstance(task_id, str) or task_id not in SMOKE_TASK_IDS:
        raise InfrastructureSmokeError(f"Invalid request task: {path}")
    if not isinstance(pass_id, str) or pass_id not in WORKER_PASS_IDS:
        raise InfrastructureSmokeError(f"Invalid request pass: {path}")
    task_index = SMOKE_TASK_IDS.index(task_id) + 1
    expected_task = _tasks(request["require_cuda"])[task_index - 1]
    expected_by_pass = {
        "injected-failure": {
            SMOKE_TASK_IDS[0]: ("001", True),
            SMOKE_TASK_IDS[1]: ("001", False),
        },
        "explicit-recovery": {
            SMOKE_TASK_IDS[0]: ("002", False),
            SMOKE_TASK_IDS[2]: ("001", False),
        },
    }
    pass_tasks = expected_by_pass[pass_id]
    if task_id not in pass_tasks:
        raise InfrastructureSmokeError(f"Task is not valid for request pass: {path}")
    expected_attempt, expected_failure = pass_tasks[task_id]
    expected_barrier = list(pass_tasks)
    if (
        request["task_index"] != task_index
        or request["task_input_sha256"] != expected_task.input_sha256
        or request["attempt_id"] != expected_attempt
        or request["inject_failure"] is not expected_failure
        or request["barrier_tasks"] != expected_barrier
        or attempt.name != expected_attempt
        or attempt.parent.name != task_id
    ):
        raise InfrastructureSmokeError(f"Request scenario identity mismatch: {path}")
    for key in ("slot_id", "device_mapping"):
        value = request[key]
        if not isinstance(value, str) or value != value.strip() or not value:
            raise InfrastructureSmokeError(f"Invalid request {key}: {path}")
    if "," in request["device_mapping"]:
        raise InfrastructureSmokeError(f"Request maps more than one GPU: {path}")
    if not HEX_SHA256_RE.fullmatch(request["task_input_sha256"]):
        raise InfrastructureSmokeError(f"Invalid task hash: {path}")


def _validate_terminal_manifest(
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    request_bytes: bytes,
    attempt: Path,
) -> None:
    status = manifest.get("status")
    if status not in {"completed", "failed"}:
        raise InfrastructureSmokeError(f"Nonterminal or invalid attempt: {attempt}")
    expected_keys = MANIFEST_KEYS | ({"failure"} if status == "failed" else set())
    if set(manifest) != expected_keys:
        raise InfrastructureSmokeError(f"Invalid manifest schema: {attempt}")
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "scientific_evidence": False,
        "artifact_type": "infrastructure_smoke_worker",
        "request_sha256": _sha256(request_bytes),
        "task_id": request["task_id"],
        "attempt_id": request["attempt_id"],
        "slot_id": request["slot_id"],
        "device_mapping": request["device_mapping"],
        "slot_mapping_sha256": _sha256(request["device_mapping"].encode()),
    }
    if any(manifest[key] != value for key, value in expected_identity.items()):
        raise InfrastructureSmokeError(f"Manifest identity mismatch: {attempt}")
    for key in ("process_id", "parent_pid", "payload_started_ns", "finished_ns"):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InfrastructureSmokeError(f"Invalid manifest {key}: {attempt}")
    if manifest["finished_ns"] < manifest["payload_started_ns"]:
        raise InfrastructureSmokeError(f"Invalid manifest timestamps: {attempt}")
    if not isinstance(manifest["hostname"], str) or not manifest["hostname"]:
        raise InfrastructureSmokeError(f"Invalid manifest hostname: {attempt}")
    commit = manifest["git_commit"]
    dirty = manifest["git_dirty"]
    if commit is not None and (not isinstance(commit, str) or not commit):
        raise InfrastructureSmokeError(f"Invalid Git commit: {attempt}")
    if dirty is not None and not isinstance(dirty, bool):
        raise InfrastructureSmokeError(f"Invalid Git dirty state: {attempt}")
    git_identity = {"commit": commit, "dirty": dirty}
    if manifest["git_identity_sha256"] != _json_sha256(git_identity):
        raise InfrastructureSmokeError(f"Invalid Git identity hash: {attempt}")
    environment = manifest["environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "python", "executable", "platform", "packages"
    }:
        raise InfrastructureSmokeError(f"Invalid environment schema: {attempt}")
    if any(not isinstance(environment[key], str) or not environment[key] for key in (
        "python", "executable", "platform"
    )):
        raise InfrastructureSmokeError(f"Invalid environment identity: {attempt}")
    packages = environment["packages"]
    if not isinstance(packages, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in packages.items()
    ):
        raise InfrastructureSmokeError(f"Invalid package identity: {attempt}")
    if manifest["environment_sha256"] != _json_sha256(environment):
        raise InfrastructureSmokeError(f"Invalid environment hash: {attempt}")
    for key in ("git_identity_sha256", "environment_sha256", "module_sha256"):
        if not isinstance(manifest[key], str) or not HEX_SHA256_RE.fullmatch(manifest[key]):
            raise InfrastructureSmokeError(f"Invalid manifest hash {key}: {attempt}")
    _validate_gpu_record(manifest["gpu"], require_cuda=request["require_cuda"], status=status)
    if status == "failed":
        failure = manifest["failure"]
        if (
            not isinstance(failure, dict)
            or set(failure) != {"type", "message"}
            or any(not isinstance(failure[key], str) or not failure[key] for key in failure)
        ):
            raise InfrastructureSmokeError(f"Invalid failure record: {attempt}")


def _validate_gpu_record(value: object, *, require_cuda: bool, status: str) -> None:
    if not require_cuda:
        if value is not None:
            raise InfrastructureSmokeError("Local smoke manifest unexpectedly records a GPU")
        return
    if value is None and status == "failed":
        return
    if not isinstance(value, dict) or set(value) != GPU_KEYS:
        raise InfrastructureSmokeError("Remote smoke manifest has an invalid GPU schema")
    for key in ("visible_name", "physical_uuid", "physical_name"):
        if not isinstance(value[key], str) or not value[key]:
            raise InfrastructureSmokeError(f"Remote GPU {key} is invalid")
    for key in ("visible_vram_bytes", "physical_vram_mib"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise InfrastructureSmokeError(f"Remote GPU {key} is invalid")
    if (
        value["visible_device_count"] != 1
        or value["bf16_supported"] is not True
        or not isinstance(value["bf16_operation_verified"], bool)
    ):
        raise InfrastructureSmokeError("Remote GPU visibility or BF16 identity is invalid")


def _validate_gpu_history(
    manifests: Sequence[Mapping[str, Any]],
    slots: Sequence[WorkerSlot[str]],
    *,
    require_cuda: bool,
    allow_shared_gpu: bool,
) -> dict[str, dict[str, Any]]:
    slot_mappings = {slot.slot_id: slot.payload for slot in slots}
    identities: dict[str, set[tuple[str, str, int]]] = {
        slot.slot_id: set() for slot in slots
    }
    for manifest in manifests:
        slot_id = manifest["slot_id"]
        if slot_id not in slot_mappings or manifest["device_mapping"] != slot_mappings[slot_id]:
            raise InfrastructureSmokeError("Worker manifest does not match a declared slot")
        gpu = manifest["gpu"]
        if not require_cuda:
            if gpu is not None:
                raise InfrastructureSmokeError("Local worker unexpectedly records a GPU")
            continue
        if not isinstance(gpu, dict) or gpu.get("bf16_operation_verified") is not True:
            raise InfrastructureSmokeError("Every remote worker must verify a BF16 CUDA operation")
        identities[slot_id].add(
            (gpu["physical_uuid"], gpu["physical_name"], gpu["physical_vram_mib"])
        )
    if not require_cuda:
        return {}
    if any(len(values) != 1 for values in identities.values()):
        raise InfrastructureSmokeError("Physical GPU identity changed within a worker slot")
    result = {
        slot.slot_id: {
            "cuda_visible_devices": slot.payload,
            "physical_uuid": next(iter(identities[slot.slot_id]))[0],
            "physical_name": next(iter(identities[slot.slot_id]))[1],
            "physical_vram_mib": next(iter(identities[slot.slot_id]))[2],
            "bf16_operation_verified": True,
        }
        for slot in slots
    }
    _require_gpu_isolation(
        [result[slot.slot_id]["physical_uuid"] for slot in slots],
        allow_shared_gpu=allow_shared_gpu,
    )
    return result


def _require_gpu_isolation(uuids: Sequence[object], *, allow_shared_gpu: bool) -> None:
    if len(uuids) != 2 or any(not isinstance(value, str) or not value for value in uuids):
        raise InfrastructureSmokeError("remote workers did not record physical GPU UUIDs")
    if not allow_shared_gpu and uuids[0] == uuids[1]:
        raise InfrastructureSmokeError(
            "worker slots resolved to one physical GPU; pass allow_shared_gpu=True explicitly"
        )


def _publish_report(root: Path, report: Mapping[str, Any]) -> str:
    _require_real_directory(root, label="infrastructure smoke root")
    report_path = root / REPORT_NAME
    hash_path = root / REPORT_HASH_NAME
    if _path_exists(report_path) or _path_exists(hash_path):
        raise InfrastructureSmokeError("Smoke report or checksum already exists")
    payload = _json_bytes(report)
    report_hash = _sha256(payload)
    _atomic_write_bytes(hash_path, (report_hash + "\n").encode("ascii"))
    _atomic_write_bytes(report_path, payload)
    return report_hash


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(value))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    _require_real_directory(path.parent, label="artifact parent")
    if _is_link(path):
        raise InfrastructureSmokeError(f"Refusing to replace a linked artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    value, _ = _load_json_object(path)
    return value


def _load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    _require_real_file(path, label="JSON artifact")
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except InfrastructureSmokeError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise InfrastructureSmokeError(f"Cannot parse infrastructure JSON: {path}") from error
    if not isinstance(value, dict):
        raise InfrastructureSmokeError(f"Infrastructure artifact is not an object: {path}")
    return value, payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InfrastructureSmokeError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise InfrastructureSmokeError(f"Invalid JSON numeric constant: {value}")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _require_no_link_ancestors(path: Path) -> None:
    for candidate in (path, *path.parents):
        if _path_exists(candidate) and _is_link(candidate):
            raise InfrastructureSmokeError(f"Linked path component is not allowed: {candidate}")


def _require_real_directory(path: Path, *, label: str) -> None:
    if _is_link(path) or not path.is_dir():
        raise InfrastructureSmokeError(f"Invalid {label}: {path}")


def _is_real_file(path: Path) -> bool:
    return not _is_link(path) and path.is_file()


def _require_real_file(path: Path, *, label: str) -> None:
    if not _is_real_file(path):
        raise InfrastructureSmokeError(f"Invalid {label}: {path}")


def _require_exact_entries(root: Path, expected: set[str]) -> None:
    _require_real_directory(root, label="artifact directory")
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise InfrastructureSmokeError(
            f"Unexpected artifact entries in {root}: expected {sorted(expected)}, got {sorted(actual)}"
        )

def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()

def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    return _sha256(payload)

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InfrastructureSmokeError(message)

def _main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("internal worker usage requires one request path")
    return _worker(Path(sys.argv[2]))

if __name__ == "__main__":
    raise SystemExit(_main())
