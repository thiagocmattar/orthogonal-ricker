from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Iterator

import pytest

import paper_exp.integrity as integrity
import paper_exp.launch as launch
import paper_exp.runner as runner


def test_parent_runner_executes_one_config_at_a_time_in_numeric_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    active = False
    calls: list[str] = []

    @contextmanager
    def guard(**_kwargs: object) -> Iterator[None]:
        nonlocal active
        assert not active
        active = True
        try:
            yield
        finally:
            active = False

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        assert active
        assert command.endswith("experiments/01-first-set/run/runner.py")
        calls.append(config_path.name)
        return (
            tmp_path
            / "experiments"
            / "01-first-set"
            / "raw"
            / config_path.stem
            / "001-test"
        )

    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "direct_launch_guard", guard)
    monkeypatch.setattr(runner, "_run_one", run_one)

    completed = runner.run_launch(runner_path, configs, repository=tmp_path)

    assert calls == ["001-case.yaml", "002-case.yaml", "003-case.yaml"]
    assert len(completed) == 3
    assert active is False


def test_parent_runner_dispatches_configs_once_to_explicit_worker_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path, **_kwargs: {
            "name": Path(path).stem,
            "training": {"device": "cuda", "precision": "bfloat16"},
        },
    )
    initial_workers = Barrier(2)
    state_lock = Lock()
    active = 0
    maximum_active = 0
    calls: list[tuple[str, str, str]] = []

    def run_isolated(
        _config: dict[str, object],
        *,
        config_path: Path,
        slot: runner.WorkerSlot[str],
        **_kwargs: object,
    ) -> Path:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            calls.append((config_path.name, slot.slot_id, str(slot.payload)))
        try:
            if config_path in configs[:2]:
                initial_workers.wait(timeout=5)
            return config_path.parent.parent / "raw" / config_path.stem / "001-test"
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(runner, "_run_one_isolated", run_isolated)
    completed = runner.run_launch(
        runner_path,
        configs,
        repository=tmp_path,
        worker_slots=(
            runner.WorkerSlot("gpu-0", "0"),
            runner.WorkerSlot("gpu-1", "1"),
        ),
    )

    assert sorted(name for name, _slot, _device in calls) == [
        "001-case.yaml",
        "002-case.yaml",
        "003-case.yaml",
    ]
    assert maximum_active == 2
    assert len(completed) == 3
    assert [path.parent.name for path in completed] == [
        "001-case",
        "002-case",
        "003-case",
    ]
    assert {slot for _name, slot, _device in calls} == {"gpu-0", "gpu-1"}


def test_parent_runner_skips_completed_parallel_configs_without_gpu_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    _stub_preflight(monkeypatch)

    def load(path: str | Path, **_kwargs: object) -> dict[str, object]:
        return {
            "name": Path(path).stem,
            "training": {"device": "cuda", "precision": "bfloat16"},
        }

    monkeypatch.setattr(runner, "load_config", load)
    existing = [
        _write_attempt(path, sequence=1, status="complete", config=load(path))
        for path in configs
    ]
    monkeypatch.setattr(
        runner,
        "_probe_worker_slots",
        lambda _slots: pytest.fail("completed configs must not require live GPUs"),
    )
    monkeypatch.setattr(
        runner,
        "_run_one_isolated",
        lambda *_args, **_kwargs: pytest.fail("completed configs must not run"),
    )

    assert runner.run_launch(
        runner_path,
        configs,
        repository=tmp_path,
        worker_slots=(runner.WorkerSlot("gpu-0", "0"),),
    ) == existing


def test_parent_runner_probes_once_and_dispatches_only_pending_parallel_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _stub_preflight(monkeypatch)

    def load(path: str | Path, **_kwargs: object) -> dict[str, object]:
        return {
            "name": Path(path).stem,
            "training": {"device": "cuda", "precision": "bfloat16"},
        }

    monkeypatch.setattr(runner, "load_config", load)
    first = _write_attempt(
        configs[0], sequence=1, status="complete", config=load(configs[0])
    )
    third = _write_attempt(
        configs[2], sequence=1, status="complete", config=load(configs[2])
    )
    probe_calls: list[tuple[runner.WorkerSlot[str], ...]] = []

    def probe(
        slots: tuple[runner.WorkerSlot[str], ...],
    ) -> dict[str, dict[str, object]]:
        probe_calls.append(slots)
        return {
            slot.slot_id: {
                "uuid": f"GPU-{slot.payload}",
                "name": "Test GPU",
                "total_memory_bytes": 48 * 1024**3,
                "compute_capability": "8.9",
            }
            for slot in slots
        }

    isolated_calls: list[str] = []

    def run_isolated(
        _config: dict[str, object],
        *,
        config_path: Path,
        **_kwargs: object,
    ) -> Path:
        isolated_calls.append(config_path.name)
        return config_path.parent.parent / "raw" / config_path.stem / "new-attempt"

    slots = (
        runner.WorkerSlot("gpu-0", "0"),
        runner.WorkerSlot("gpu-1", "1"),
    )
    monkeypatch.setattr(runner, "_probe_worker_slots", probe)
    monkeypatch.setattr(runner, "_run_one_isolated", run_isolated)

    completed = runner.run_launch(
        runner_path,
        configs,
        repository=tmp_path,
        worker_slots=slots,
    )

    assert probe_calls == [slots]
    assert isolated_calls == [configs[1].name]
    assert completed == [
        first,
        configs[1].parent.parent / "raw" / configs[1].stem / "new-attempt",
        third,
    ]


def test_parent_runner_parallel_failure_stops_later_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path, **_kwargs: {
            "name": Path(path).stem,
            "training": {"device": "cuda", "precision": "bfloat16"},
        },
    )
    calls: list[str] = []

    def fail_first(
        _config: dict[str, object],
        *,
        config_path: Path,
        **_kwargs: object,
    ) -> Path:
        calls.append(config_path.name)
        raise RuntimeError("injected isolated failure")

    monkeypatch.setattr(runner, "_run_one_isolated", fail_first)
    with pytest.raises(runner.RunnerError, match="001-case"):
        runner.run_launch(
            runner_path,
            configs,
            repository=tmp_path,
            worker_slots=(runner.WorkerSlot("gpu-0", "0"),),
        )

    assert calls == ["001-case.yaml"]


def test_parallel_worker_maps_slot_environment_and_reports_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[dict[str, object]] = []

    class Sender:
        def send(self, message: dict[str, object]) -> None:
            messages.append(message)

        def close(self) -> None:
            pass

    environment: dict[str, str] = {}
    monkeypatch.setattr(runner.os, "environ", environment)
    monkeypatch.setattr(runner.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        runner,
        "_require_isolated_cuda_runtime",
        lambda _config: {
            "uuid": "GPU-test",
            "name": "Test GPU",
            "total_memory_bytes": 48 * 1024**3,
            "compute_capability": "8.9",
        },
    )

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        assert config_path.name == "001-case.yaml"
        assert command == "case runner command"
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert environment["PAPER_EXP_WORKER_SLOT_ID"] == "gpu-3"
        assert environment["PAPER_EXP_WORKER_GPU_UUID"] == "GPU-test"
        return tmp_path / "completed"

    monkeypatch.setattr(runner, "_run_one", run_one)
    runner._worker_process_entry(
        {
            "name": "001-case",
            "training": {"device": "cuda", "precision": "bfloat16"},
        },
        str(tmp_path / "001-case.yaml"),
        "case runner command",
        str(tmp_path),
        "launch-id",
        1,
        3,
        "gpu-3",
        "3",
        "GPU-test",
        123,
        Sender(),
    )

    assert messages == [
        {"status": "completed", "run_dir": str(tmp_path / "completed")}
    ]
    assert environment["PAPER_EXP_COORDINATOR_PID"] == "123"


def test_isolated_worker_ipc_error_still_drains_and_closes_child() -> None:
    events: list[str] = []

    class Receiver:
        def recv(self) -> object:
            events.append("recv")
            raise OSError("injected receive failure")

        def close(self) -> None:
            events.append("receiver-close")

    class Process:
        exitcode = 1

        def join(self) -> None:
            events.append("join")

        def is_alive(self) -> bool:
            events.append("is-alive")
            return False

        def close(self) -> None:
            events.append("process-close")

    with pytest.raises(
        runner.WorkerProcessError, match="receiving or reaping"
    ) as caught:
        runner._receive_and_reap_worker(Process(), Receiver())

    assert isinstance(caught.value.__cause__, OSError)
    assert events == [
        "recv",
        "receiver-close",
        "join",
        "is-alive",
        "process-close",
    ]


def test_isolated_worker_does_not_close_a_live_child_after_join_failure() -> None:
    events: list[str] = []

    class Receiver:
        def recv(self) -> object:
            events.append("recv")
            return {"status": "completed"}

        def close(self) -> None:
            events.append("receiver-close")

    class Process:
        exitcode = None

        def join(self) -> None:
            events.append("join")
            raise OSError("injected join failure")

        def is_alive(self) -> bool:
            events.append("is-alive")
            return True

        def close(self) -> None:
            pytest.fail("a live process handle must not be closed")

    with pytest.raises(
        runner.WorkerProcessError, match="receiving or reaping"
    ) as caught:
        runner._receive_and_reap_worker(Process(), Receiver())

    assert isinstance(caught.value.__cause__, OSError)
    assert events == ["recv", "receiver-close", "join", "is-alive"]


def test_gpu_uuid_mismatch_prevents_training_attempt_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[dict[str, object]] = []

    class Sender:
        def send(self, message: dict[str, object]) -> None:
            messages.append(message)

        def close(self) -> None:
            pass

    monkeypatch.setattr(runner.os, "environ", {})
    monkeypatch.setattr(runner.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        runner,
        "_require_isolated_cuda_runtime",
        lambda _config: {
            "uuid": "GPU-observed",
            "name": "Test GPU",
            "total_memory_bytes": 48 * 1024**3,
            "compute_capability": "8.9",
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail(
            "UUID mismatch must fail before creating a training attempt"
        ),
    )

    with pytest.raises(runner.WorkerProcessError, match="identity changed"):
        runner._worker_process_entry(
            {
                "name": "001-case",
                "training": {"device": "cuda", "precision": "bfloat16"},
            },
            str(tmp_path / "001-case.yaml"),
            "case runner command",
            str(tmp_path),
            "launch-id",
            1,
            1,
            "gpu-0",
            "0",
            "GPU-expected",
            123,
            Sender(),
        )

    assert messages == [
        {
            "status": "failed",
            "error_type": "WorkerProcessError",
            "error_message": (
                "Assigned CUDA device identity changed after coordinator preflight: "
                "expected GPU-expected, observed GPU-observed."
            ),
        }
    ]


@pytest.mark.parametrize("prefixes", [(2, 1), (1, 1)])
def test_parent_runner_rejects_non_increasing_config_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefixes: tuple[int, int],
) -> None:
    runner_path, configs = _layout(tmp_path, prefixes)
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="strictly increasing"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def test_parent_runner_validates_the_whole_set_before_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    _stub_preflight(monkeypatch)
    validated: list[str] = []

    def validate(config: dict[str, object]) -> None:
        validated.append(str(config["name"]))
        if config["name"] == "002-case":
            raise ValueError("invalid second config")

    monkeypatch.setattr(runner, "validate_training_config", validate)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("no run may start after failed preflight"),
    )

    with pytest.raises(ValueError, match="invalid second config"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert validated == ["001-case", "002-case"]


def test_parent_runner_stops_after_first_failed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _stub_preflight(monkeypatch)
    calls: list[str] = []

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        del command
        calls.append(config_path.name)
        if config_path.name.startswith("002-"):
            raise RuntimeError("experiment failed")
        return tmp_path / "result"

    monkeypatch.setattr(runner, "_run_one", run_one)

    with pytest.raises(RuntimeError, match="experiment failed"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert calls == ["001-case.yaml", "002-case.yaml"]


def test_parent_runner_skips_one_completion_and_retries_only_failed_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _write_attempt(configs[0], sequence=1, status="failed")
    existing = _write_attempt(configs[0], sequence=2, status="complete")
    _write_attempt(configs[1], sequence=1, status="failed")
    _write_attempt(configs[1], sequence=2, status="failed")
    calls: list[str] = []

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        assert command.endswith(
            "experiments/01-first-set/run/runner.py --retry-failed"
        )
        calls.append(config_path.name)
        return config_path.parent.parent / "raw" / config_path.stem / "new-attempt"

    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "_run_one", run_one)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [str(runner_path), "--retry-failed"],
    )

    completed = runner.run_launch(runner_path, configs, repository=tmp_path)

    assert calls == ["002-case.yaml", "003-case.yaml"]
    assert completed == [
        existing,
        configs[1].parent.parent / "raw" / configs[1].stem / "new-attempt",
        configs[2].parent.parent / "raw" / configs[2].stem / "new-attempt",
    ]


def test_parent_runner_requires_explicit_failed_retry_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    failed = _write_attempt(configs[0], sequence=1, status="failed")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: pytest.fail("unauthorized retry must fail before locking"),
    )

    with pytest.raises(runner.RunnerError, match="--retry-failed"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert sorted(path.name for path in failed.parent.iterdir()) == [failed.name]


@pytest.mark.parametrize(
    "arguments",
    (["--unknown"], ["--retry-failed", "--retry-failed"]),
)
def test_parent_runner_rejects_unsupported_script_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner.sys, "argv", [str(runner_path), *arguments])

    with pytest.raises(runner.RunnerError, match="Unsupported case-runner"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def test_case_runner_parses_retry_and_explicit_worker_slots() -> None:
    retry, slots = runner._parse_runner_arguments(
        [
            "--worker-slot",
            "gpu-0=0",
            "--retry-failed",
            "--worker-slot=gpu-1=1",
        ]
    )

    assert retry is True
    assert slots == (
        runner.WorkerSlot("gpu-0", "0"),
        runner.WorkerSlot("gpu-1", "1"),
    )


def test_case_runner_rejects_same_device_packing() -> None:
    arguments = ["--worker-slot", "worker-a=0", "--worker-slot", "worker-b=0"]
    with pytest.raises(runner.RunnerError, match="distinct CUDA device"):
        runner._parse_runner_arguments(arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["--worker-slot"], "requires SLOT=CUDA_DEVICE"),
        (["--worker-slot", "GPU_0=0"], "Worker slot must be"),
        (["--worker-slot", "gpu-0=-1"], "exactly one CUDA device"),
        (["--worker-slot", "gpu-0=0,1"], "exactly one CUDA device"),
        (
            ["--worker-slot", "gpu-0=0", "--worker-slot", "gpu-0=1"],
            "slot IDs must be unique",
        ),
    ),
)
def test_case_runner_rejects_invalid_worker_slots(
    arguments: list[str],
    message: str,
) -> None:
    with pytest.raises(runner.RunnerError, match=message):
        runner._parse_runner_arguments(arguments)


def test_parallel_preflight_rejects_an_unmappable_config_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path, **_kwargs: {
            "name": Path(path).stem,
            "training": {"device": "cuda:1", "precision": "bfloat16"},
        },
    )
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: pytest.fail("device mapping must fail before locking"),
    )

    with pytest.raises(runner.RunnerError, match="CUDA_VISIBLE_DEVICES"):
        runner.run_launch(
            runner_path,
            configs,
            repository=tmp_path,
            worker_slots=(runner.WorkerSlot("gpu-0", "0"),),
        )


def test_concurrent_scientific_launch_requires_resolved_readiness_items(
    tmp_path: Path,
) -> None:
    workboard = tmp_path / "docs" / "experimental-design" / "workboard.md"
    workboard.parent.mkdir(parents=True)
    workboard.write_text(
        "\n".join(
            (
                "| ID | State | Blocks |",
                "| --- | --- | --- |",
                "| `CLOUD-01` | resolved | launch |",
                "| `OPS-04` | resolved | launch |",
                "| `OPS-05` | open | launch |",
                "| `OPS-06` | resolved | launch |",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.RunnerError, match="OPS-05"):
        runner._require_parallel_launch_ready(tmp_path)

    workboard.write_text(
        workboard.read_text(encoding="utf-8").replace(
            "| `OPS-05` | open |", "| `OPS-05` | resolved |"
        ),
        encoding="utf-8",
    )
    runner._require_parallel_launch_ready(tmp_path)


@pytest.mark.parametrize(
    ("device", "precision", "message"),
    (
        ("auto", "bfloat16", "must select cuda"),
        ("cuda", "auto", "must select bfloat16"),
    ),
)
def test_parallel_preflight_requires_explicit_cuda_bfloat16(
    device: str,
    precision: str,
    message: str,
) -> None:
    with pytest.raises(runner.RunnerError, match=message):
        runner._require_worker_mappable_device(
            {"training": {"device": device, "precision": precision}},
            config_path=Path("001-case.yaml"),
        )


@pytest.mark.parametrize(
    ("available", "count", "bf16", "message"),
    (
        (False, 0, False, "unavailable"),
        (True, 2, True, "exactly one"),
        (True, 1, False, "does not support BF16"),
    ),
)
def test_isolated_worker_rejects_invalid_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    count: int,
    bf16: bool,
    message: str,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            is_bf16_supported=lambda: bf16,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(runner.WorkerProcessError, match=message):
        runner._require_isolated_cuda_runtime(
            {"training": {"device": "cuda", "precision": "bfloat16"}}
        )


def test_isolated_worker_accepts_one_bfloat16_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            is_bf16_supported=lambda: True,
            get_device_properties=lambda _index: SimpleNamespace(
                uuid="GPU-test",
                name="Test GPU",
                total_memory=48 * 1024**3,
                major=8,
                minor=9,
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert runner._require_isolated_cuda_runtime(
        {"training": {"device": "cuda", "precision": "bfloat16"}}
    ) == {
        "uuid": "GPU-test",
        "name": "Test GPU",
        "total_memory_bytes": 48 * 1024**3,
        "compute_capability": "8.9",
    }


def test_worker_slot_probe_requires_distinct_homogeneous_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = (
        runner.WorkerSlot("gpu-0", "0"),
        runner.WorkerSlot("gpu-1", "1"),
    )
    identities = {
        "0": {
            "uuid": "GPU-a",
            "name": "Test GPU",
            "total_memory_bytes": 48 * 1024**3,
            "compute_capability": "8.9",
        },
        "1": {
            "uuid": "GPU-b",
            "name": "Test GPU",
            "total_memory_bytes": 48 * 1024**3,
            "compute_capability": "8.9",
        },
    }
    monkeypatch.setattr(
        runner,
        "_probe_cuda_slot",
        lambda slot: identities[str(slot.payload)],
    )

    assert runner._probe_worker_slots(slots) == {
        "gpu-0": identities["0"],
        "gpu-1": identities["1"],
    }

    identities["1"] = {**identities["1"], "uuid": "GPU-a"}
    with pytest.raises(runner.RunnerError, match="same physical GPU"):
        runner._probe_worker_slots(slots)

    identities["1"] = {
        **identities["1"],
        "uuid": "GPU-b",
        "name": "Different GPU",
    }
    with pytest.raises(runner.RunnerError, match="homogeneous GPU class"):
        runner._probe_worker_slots(slots)


@pytest.mark.parametrize(
    ("mode", "status"),
    (("calibrate", "complete"), ("prepare-data", "failed")),
)
def test_parent_runner_ignores_non_pretrain_attempt_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    status: str,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _write_attempt(configs[0], sequence=1, status=status, mode=mode)
    _stub_preflight(monkeypatch)
    expected = configs[0].parent.parent / "raw" / configs[0].stem / "new-pretrain"
    monkeypatch.setattr(runner, "_run_one", lambda *_args, **_kwargs: expected)

    assert runner.run_launch(runner_path, configs, repository=tmp_path) == [expected]


@pytest.mark.parametrize("unsafe_status", ("running", "inconsistent"))
def test_parent_runner_rejects_unsafe_attempt_state_before_launch_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_status: str,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    first = _write_attempt(configs[0], sequence=1, status="failed")
    _write_attempt(configs[1], sequence=1, status=unsafe_status)
    guard_entered = False

    @contextmanager
    def guard(**_kwargs: object) -> Iterator[None]:
        nonlocal guard_entered
        guard_entered = True
        yield

    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "direct_launch_guard", guard)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("unsafe state must abort the tranche"),
    )

    with pytest.raises(runner.RunnerError, match="unsafe attempt state"):
        runner.run_launch(
            runner_path,
            configs,
            repository=tmp_path,
            retry_failed=True,
        )

    assert guard_entered is False
    assert sorted(path.name for path in first.parent.iterdir()) == [first.name]


def test_parent_runner_rejects_unreviewed_statusless_pretrain_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    attempt = _write_attempt(configs[0], sequence=1, status="statusless")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "classify_run_directory", lambda _path: "complete")

    with pytest.raises(runner.RunnerError, match="statusless"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert attempt.is_dir()


def test_parent_runner_rejects_multiple_coherent_completions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _write_attempt(configs[0], sequence=1, status="complete")
    _write_attempt(configs[0], sequence=2, status="complete")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: pytest.fail("ambiguous state must fail before locking"),
    )

    with pytest.raises(runner.RunnerError, match="multiple coherent completed"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def test_parent_runner_rechecks_attempt_state_after_taking_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _stub_preflight(monkeypatch)
    existing: Path | None = None

    @contextmanager
    def guard(**_kwargs: object) -> Iterator[None]:
        nonlocal existing
        existing = _write_attempt(configs[0], sequence=1, status="complete")
        yield

    monkeypatch.setattr(runner, "direct_launch_guard", guard)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("locked recheck must reuse completion"),
    )

    completed = runner.run_launch(runner_path, configs, repository=tmp_path)

    assert existing is not None
    assert completed == [existing]


def test_parent_runner_rejects_attempt_from_mutated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _write_attempt(
        configs[0],
        sequence=1,
        status="failed",
        config={"name": "changed-after-attempt"},
    )
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="inconsistent"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


@pytest.mark.parametrize("invalid_name", ("launch.py", "01-first-set.py"))
def test_case_runner_requires_exact_scaffold_location_and_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_name: str,
) -> None:
    scaffold = _scaffold(tmp_path, "01-first-set")
    invalid = scaffold / "run" / invalid_name
    invalid.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)

    with pytest.raises(runner.RunnerError, match="run/runner.py"):
        runner._resolve_runner(invalid, tmp_path)

    misplaced = scaffold / "runner.py"
    misplaced.write_text("", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="run/runner.py"):
        runner._resolve_runner(misplaced, tmp_path)


def test_smoke_scaffold_cannot_be_a_scientific_case_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path = (
        _scaffold(tmp_path, "00-infrastructure-smoke") / "run" / "runner.py"
    )
    runner_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)

    with pytest.raises(runner.RunnerError, match="nonzero scaffold"):
        runner._resolve_runner(runner_path, tmp_path)


@pytest.mark.parametrize(
    "invalid_name",
    ("000-case.yaml", "01-case.yaml", "001-case.yml"),
)
def test_parent_runner_requires_exact_scientific_config_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_name: str,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    invalid = configs[0].with_name(invalid_name)
    configs[0].rename(invalid)
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="CCC-<case>.yaml"):
        runner.run_launch(runner_path, [invalid], repository=tmp_path)


def test_case_runner_owns_one_matching_config_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    wrong_root = _scaffold(tmp_path, "02-other-set") / "run"
    wrong_config = wrong_root / configs[0].name
    wrong_config.write_text("", encoding="utf-8")
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="may only use configs"):
        runner.run_launch(runner_path, [wrong_config], repository=tmp_path)


def test_case_runner_requires_every_yaml_in_its_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    omitted = configs[0].parent / "003-omitted.yaml"
    omitted.write_text("", encoding="utf-8")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("an incomplete tranche must not start"),
    )

    with pytest.raises(runner.RunnerError, match="exactly all YAML files"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def _layout(tmp_path: Path, prefixes: tuple[int, ...]) -> tuple[Path, list[Path]]:
    scaffold = _scaffold(tmp_path, "01-first-set")
    runner_path = scaffold / "run" / "runner.py"
    runner_path.write_text("", encoding="utf-8")
    config_root = scaffold / "run"
    configs: list[Path] = []
    for index, prefix in enumerate(prefixes):
        suffix = "case" if prefixes.count(prefix) == 1 else f"case-{index}"
        path = config_root / f"{prefix:03d}-{suffix}.yaml"
        path.write_text("", encoding="utf-8")
        configs.append(path)
    return runner_path, configs


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)
    monkeypatch.setattr(launch, "require_tracked_file", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path, **_kwargs: {"name": Path(path).stem},
    )
    monkeypatch.setattr(runner, "validate_training_config", lambda _config: None)
    monkeypatch.setattr(integrity, "validate_training_config", lambda _config: None)
    monkeypatch.setattr(runner, "require_raw_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "require_token_cache_output",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: _null_guard(),
    )
    monkeypatch.setattr(runner, "_require_parallel_launch_ready", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_probe_worker_slots",
        lambda slots: {
            slot.slot_id: {
                "uuid": f"GPU-{slot.payload}",
                "name": "Test GPU",
                "total_memory_bytes": 48 * 1024**3,
                "compute_capability": "8.9",
            }
            for slot in slots
        },
    )


@contextmanager
def _null_guard() -> Iterator[None]:
    yield


def _write_attempt(
    config_path: Path,
    *,
    sequence: int,
    status: str,
    config: dict[str, object] | None = None,
    mode: str = "pretrain",
) -> Path:
    run_dir = (
        config_path.parent.parent
        / "raw"
        / config_path.stem
        / f"{sequence:03d}-{status}"
    )
    run_dir.mkdir(parents=True)
    snapshot = config or {"name": config_path.stem}
    (run_dir / "config.yaml").write_text(
        json.dumps(snapshot) + "\n", encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "config_id": config_path.stem,
        "run_id": run_dir.name,
        "tranche_id": config_path.parent.parent.name,
        "mode": mode,
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    if status in {"complete", "statusless"}:
        (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "events.jsonl").write_text(
            '{"event": "train"}\n', encoding="utf-8"
        )
    if status in {"complete", "running", "failed"}:
        lifecycle_status = "completed" if status == "complete" else status
        manifest.update(
            {
                "status": lifecycle_status,
                "started_at": "2026-01-01T00:00:00Z",
            }
        )
        if status in {"complete", "failed"}:
            manifest["finished_at"] = "2026-01-01T00:01:00Z"
        if lifecycle_status == "failed":
            manifest.update(
                {
                    "failure": {"type": "RuntimeError", "message": "test"},
                }
            )
    elif status not in {"inconsistent", "statusless"}:
        raise ValueError(f"Unsupported test attempt status: {status}")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return run_dir


def _scaffold(tmp_path: Path, scaffold_id: str) -> Path:
    scaffold = tmp_path / "experiments" / scaffold_id
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    return scaffold
