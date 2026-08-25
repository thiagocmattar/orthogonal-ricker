from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

import paper_exp.infrastructure_smoke as smoke
from paper_exp.infrastructure_smoke import (
    REPORT_HASH_NAME,
    REPORT_NAME,
    SMOKE_TASK_IDS,
    InfrastructureSmokeError,
    _collect_remote_gpu_identity,
    _publish_report,
    _require_gpu_isolation,
    _run_bf16_cuda_operation,
    _state,
    _tasks,
    _validate_gpu_history,
    run_concurrent_infrastructure_smoke,
)
from paper_exp.parallel import WorkerSlot


def test_local_script_proves_overlap_drain_recovery_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")
    root = tmp_path / "concurrent"
    report = run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        root,
        require_cuda=False,
        task_sleep_seconds=0.06,
        barrier_timeout_seconds=10,
    )

    assert report["scientific_evidence"] is False
    assert report["passed"] is True
    assert report["recovery_scope"] == "single_coordinator_invocation"
    assert report["coordinator_interruption_recovery_proved"] is False
    assert report["evidence"]["initial_overlap_nanoseconds"] > 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent"
    failed = report["passes"]["injected_failure"]
    assert [item["status"] for item in failed["results"][:2]] == [
        "failed",
        "completed",
    ]
    assert failed["unadmitted_task_ids"] in ([], [SMOKE_TASK_IDS[2]])
    recovered = report["passes"]["explicit_recovery"]
    completed_first = [
        item["task_id"] for item in failed["results"] if item["status"] == "completed"
    ]
    assert recovered["skipped_task_ids"] == completed_first
    assert [item["task_id"] for item in recovered["results"]] == [
        task_id for task_id in SMOKE_TASK_IDS if task_id not in completed_first
    ]
    assert report["passes"]["completed_restart"]["results"] == []

    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("tasks/*/*/manifest.json"))
    ]
    assert len(manifests) == 4
    assert [item["status"] for item in manifests if item["task_id"] == SMOKE_TASK_IDS[0]] == [
        "failed",
        "completed",
    ]
    assert all(item["scientific_evidence"] is False for item in manifests)
    assert all(item["process_id"] != os.getpid() for item in manifests)
    assert all(item["parent_pid"] == os.getpid() for item in manifests)
    assert {item["device_mapping"] for item in manifests} == {"0", "1"}
    report_path = root / REPORT_NAME
    assert (root / REPORT_HASH_NAME).read_text(encoding="ascii").strip() == sha256(
        report_path.read_bytes()
    ).hexdigest()


def test_smoke_accepts_work_admitted_before_failure_becomes_observable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_spawn = smoke._spawn_worker
    third_completed = Event()

    def reordered_spawn(*args: object, **kwargs: object) -> None:
        pass_id = args[1]
        task = args[2]
        assert isinstance(task, smoke._Task)
        if pass_id == "injected-failure" and task.task_id == SMOKE_TASK_IDS[0]:
            try:
                original_spawn(*args, **kwargs)  # type: ignore[arg-type]
            except InfrastructureSmokeError:
                assert third_completed.wait(timeout=10)
                raise
            pytest.fail("the injected failure unexpectedly completed")
        original_spawn(*args, **kwargs)  # type: ignore[arg-type]
        if pass_id == "injected-failure" and task.task_id == SMOKE_TASK_IDS[2]:
            third_completed.set()

    monkeypatch.setattr(smoke, "_spawn_worker", reordered_spawn)
    report = run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        tmp_path / "concurrent",
        require_cuda=False,
        task_sleep_seconds=0.03,
        barrier_timeout_seconds=10,
    )

    failed = report["passes"]["injected_failure"]
    assert [item["status"] for item in failed["results"]] == [
        "failed",
        "completed",
        "completed",
    ]
    assert failed["unadmitted_task_ids"] == []
    recovered = report["passes"]["explicit_recovery"]
    assert recovered["skipped_task_ids"] == list(SMOKE_TASK_IDS[1:])
    assert [item["task_id"] for item in recovered["results"]] == [SMOKE_TASK_IDS[0]]
    assert report["passes"]["completed_restart"]["skipped_task_ids"] == list(
        SMOKE_TASK_IDS
    )
    assert len(report["attempts"]) == 4


def test_tampered_request_cannot_satisfy_completed_restart(tmp_path: Path) -> None:
    root = tmp_path / "concurrent"
    run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        root,
        require_cuda=False,
        task_sleep_seconds=0.03,
        barrier_timeout_seconds=10,
    )
    request_path = root / "tasks" / SMOKE_TASK_IDS[1] / "001" / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["task_id"] = "copied-from-another-task"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(InfrastructureSmokeError, match="Invalid request task"):
        _state(root, _tasks(False)[1])


def test_duplicate_malformed_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "concurrent"
    run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        root,
        require_cuda=False,
        task_sleep_seconds=0.03,
        barrier_timeout_seconds=10,
    )
    request_path = root / "tasks" / SMOKE_TASK_IDS[1] / "001" / "request.json"
    original = request_path.read_text(encoding="utf-8")
    payloads = (
        ('{"task_id":"duplicate",' + original.lstrip()[1:], "Duplicate JSON key"),
        ('{"unexpected":true,' + original.lstrip()[1:], "Invalid request schema"),
        ("{", "Cannot parse"),
        (original.replace('"sleep_seconds": 0.03', '"sleep_seconds": NaN'), "numeric constant"),
    )
    for payload, message in payloads:
        request_path.write_text(payload, encoding="utf-8")
        with pytest.raises(InfrastructureSmokeError, match=message):
            _state(root, _tasks(False)[1])
        request_path.write_text(original, encoding="utf-8")


def test_unexpected_attempt_artifact_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "concurrent"
    run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        root,
        require_cuda=False,
        task_sleep_seconds=0.03,
        barrier_timeout_seconds=10,
    )
    attempt = root / "tasks" / SMOKE_TASK_IDS[1] / "001"
    (attempt / "unexpected.txt").write_text("unexpected", encoding="ascii")
    with pytest.raises(InfrastructureSmokeError, match="Unexpected artifact entries"):
        _state(root, _tasks(False)[1])


def test_root_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "concurrent"
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(InfrastructureSmokeError, match="infrastructure smoke root"):
        run_concurrent_infrastructure_smoke(
            (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
            root,
            require_cuda=False,
        )
    assert list(target.iterdir()) == []


def test_linked_state_artifact_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "concurrent"
    run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        root,
        require_cuda=False,
        task_sleep_seconds=0.03,
        barrier_timeout_seconds=10,
    )
    request_path = root / "tasks" / SMOKE_TASK_IDS[1] / "001" / "request.json"
    target = tmp_path / "request-copy.json"
    target.write_bytes(request_path.read_bytes())
    request_path.unlink()
    try:
        request_path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    with pytest.raises(InfrastructureSmokeError, match="Invalid JSON artifact"):
        _state(root, _tasks(False)[1])


def test_worker_rejects_invalid_vocabulary_before_manifest_mutation(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    run_concurrent_infrastructure_smoke(
        (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
        source_root,
        require_cuda=False,
        task_sleep_seconds=0.03,
        barrier_timeout_seconds=10,
    )
    source = source_root / "tasks" / SMOKE_TASK_IDS[2] / "001" / "request.json"
    request = json.loads(source.read_text(encoding="utf-8"))
    request["pass_id"] = "../../escape"
    attempt = tmp_path / "target" / "tasks" / SMOKE_TASK_IDS[2] / "001"
    attempt.mkdir(parents=True)
    request_path = attempt / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(InfrastructureSmokeError, match="Invalid request pass"):
        smoke._worker(request_path)
    assert not (attempt / "manifest.json").exists()


@pytest.mark.parametrize(
    "slots",
    (
        (WorkerSlot("only", "0"),),
        (WorkerSlot("same", "0"), WorkerSlot("same", "1")),
        (WorkerSlot("a", "0,1"), WorkerSlot("b", "2")),
        (WorkerSlot("a", None), WorkerSlot("b", "1")),
    ),
)
def test_invalid_slots_fail_before_mutation(
    tmp_path: Path, slots: tuple[WorkerSlot[object], ...]
) -> None:
    root = tmp_path / "absent"
    with pytest.raises((TypeError, ValueError)):
        run_concurrent_infrastructure_smoke(
            slots, root, require_cuda=False  # type: ignore[arg-type]
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"require_cuda": 1},
        {"require_cuda": False, "allow_shared_gpu": 1},
    ),
)
def test_non_boolean_modes_fail_before_mutation(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    root = tmp_path / "absent"
    with pytest.raises(TypeError, match="bool"):
        run_concurrent_infrastructure_smoke(
            (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1")),
            root,
            **kwargs,  # type: ignore[arg-type]
        )
    assert not root.exists()


def test_remote_smoke_requires_clean_nonnull_git_identity() -> None:
    with pytest.raises(InfrastructureSmokeError, match="clean non-null"):
        smoke._require_clean_git({"git_commit": None, "git_dirty": False}, label="test")
    with pytest.raises(InfrastructureSmokeError, match="clean non-null"):
        smoke._require_clean_git({"git_commit": "abc", "git_dirty": True}, label="test")
    smoke._require_clean_git({"git_commit": "abc", "git_dirty": False}, label="test")


def test_remote_probe_and_shared_gpu_policy() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        is_bf16_supported=lambda: True,
        get_device_properties=lambda _index: SimpleNamespace(
            name="NVIDIA Test", total_memory=48 * 1024**3, uuid="GPU-1"
        ),
    )
    identity = _collect_remote_gpu_identity("0", torch_module=SimpleNamespace(cuda=cuda))
    assert identity["physical_uuid"] == "GPU-1"
    assert identity["visible_device_count"] == 1
    with pytest.raises(InfrastructureSmokeError, match="allow_shared_gpu"):
        _require_gpu_isolation(["GPU-1", "GPU-1"], allow_shared_gpu=False)
    _require_gpu_isolation(["GPU-1", "GPU-1"], allow_shared_gpu=True)


def test_bf16_probe_runs_and_synchronizes_cuda_operation() -> None:
    calls: list[str] = []

    class Result:
        def __getitem__(self, _index: object) -> Result:
            return self

        def float(self) -> Result:
            return self

        def item(self) -> float:
            return 16.0

    torch_module = SimpleNamespace(
        bfloat16=object(),
        cuda=SimpleNamespace(synchronize=lambda: calls.append("synchronize")),
        ones=lambda *args, **kwargs: calls.append("ones") or object(),
        matmul=lambda left, right: calls.append("matmul") or Result(),
    )
    _run_bf16_cuda_operation(torch_module=torch_module)
    assert calls == ["ones", "matmul", "synchronize"]


def test_gpu_identity_must_remain_stable_for_every_slot_attempt() -> None:
    slots = (WorkerSlot("slot-0", "0"), WorkerSlot("slot-1", "1"))

    def manifest(slot: int, uuid: str) -> dict[str, object]:
        return {
            "slot_id": f"slot-{slot}",
            "device_mapping": str(slot),
            "gpu": {
                "physical_uuid": uuid,
                "physical_name": "NVIDIA Test",
                "physical_vram_mib": 48 * 1024,
                "bf16_operation_verified": True,
            },
        }

    manifests = [
        manifest(0, "GPU-0"),
        manifest(1, "GPU-1"),
        manifest(0, "GPU-0"),
        manifest(1, "GPU-1"),
    ]
    identities = _validate_gpu_history(
        manifests, slots, require_cuda=True, allow_shared_gpu=False
    )
    assert identities["slot-0"]["physical_uuid"] == "GPU-0"
    manifests[-1] = manifest(1, "GPU-0")
    with pytest.raises(InfrastructureSmokeError, match="changed within"):
        _validate_gpu_history(
            manifests, slots, require_cuda=True, allow_shared_gpu=False
        )


def test_checksum_is_atomic_and_published_before_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    original = smoke._atomic_write_bytes

    def recording_write(path: Path, payload: bytes) -> None:
        order.append(path.name)
        original(path, payload)

    monkeypatch.setattr(smoke, "_atomic_write_bytes", recording_write)
    report_hash = _publish_report(tmp_path, {"scientific_evidence": False})
    assert order == [REPORT_HASH_NAME, REPORT_NAME]
    assert (tmp_path / REPORT_HASH_NAME).read_text(encoding="ascii").strip() == report_hash
    assert report_hash == sha256((tmp_path / REPORT_NAME).read_bytes()).hexdigest()


def test_checksum_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("unchanged", encoding="ascii")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / REPORT_HASH_NAME).symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    with pytest.raises(InfrastructureSmokeError, match="already exists"):
        _publish_report(root, {"scientific_evidence": False})
    assert target.read_text(encoding="ascii") == "unchanged"
    assert not (root / REPORT_NAME).exists()


@pytest.mark.parametrize(
    ("available", "count", "bf16", "message"),
    ((False, 0, False, "CUDA"), (True, 2, True, "exactly one"), (True, 1, False, "BF16")),
)
def test_remote_probe_rejects_invalid_cuda(
    available: bool, count: int, bf16: bool, message: str
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        is_bf16_supported=lambda: bf16,
    )
    with pytest.raises(RuntimeError, match=message):
        _collect_remote_gpu_identity("0", torch_module=SimpleNamespace(cuda=cuda))
