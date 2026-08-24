from __future__ import annotations

from collections import Counter
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from typing import Any

import pytest

import paper_exp.parallel as parallel
from paper_exp.parallel import WorkItem, WorkerSlot, run_bounded


def test_bounded_coordinator_admits_distinct_configs_once_in_order() -> None:
    items = tuple(
        WorkItem(f"{index:03d}-case", Path(f"{index:03d}-case.yaml"))
        for index in range(1, 5)
    )
    slots = (
        WorkerSlot("gpu-0", {"device": "0"}),
        WorkerSlot("gpu-1", {"device": "1"}),
    )
    initial_workers = Barrier(2)
    state_lock = Lock()
    active = 0
    maximum_active = 0
    calls: list[tuple[str, str, str]] = []

    def worker(
        item: WorkItem[Path], slot: WorkerSlot[dict[str, str]]
    ) -> str:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            calls.append((item.config_id, slot.slot_id, slot.payload["device"]))
        try:
            if item.config_id in {"001-case", "002-case"}:
                initial_workers.wait(timeout=5)
            return f"{item.config_id}@{slot.slot_id}"
        finally:
            with state_lock:
                active -= 1

    report = run_bounded(items, slots, worker)

    assert [assignment.admission_index for assignment in report.assignments] == [
        0,
        1,
        2,
        3,
    ]
    assert [assignment.config_id for assignment in report.assignments] == [
        item.config_id for item in items
    ]
    assert [assignment.slot_id for assignment in report.assignments[:2]] == [
        "gpu-0",
        "gpu-1",
    ]
    assert Counter(config_id for config_id, _slot, _device in calls) == Counter(
        item.config_id for item in items
    )
    assert all(slot == f"gpu-{device}" for _config, slot, device in calls)
    assert maximum_active == 2
    assert [result.status for result in report.results] == ["completed"] * 4
    assert [result.value for result in report.results] == [
        f"{assignment.config_id}@{assignment.slot_id}"
        for assignment in report.assignments
    ]
    assert report.completed == report.results
    assert report.failed == ()
    assert report.unadmitted_config_ids == ()
    assert report.triggering_failure is None


def test_first_failure_stops_admission_and_drains_running_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = (
        WorkItem("001-fails", None),
        WorkItem("002-drains", None),
        WorkItem("003-must-not-start", None),
    )
    slots = (WorkerSlot("gpu-0"), WorkerSlot("gpu-1"))
    sibling_started = Event()
    failure_returned_to_coordinator = Event()
    release_sibling = Event()
    sibling_finished = Event()
    calls: list[str] = []
    calls_lock = Lock()
    original_wait = parallel.wait

    def observing_wait(*args: Any, **kwargs: Any) -> Any:
        done, pending = original_wait(*args, **kwargs)
        if any(future.result().status == "failed" for future in done):
            failure_returned_to_coordinator.set()
        return done, pending

    def worker(item: WorkItem[None], _slot: WorkerSlot[None]) -> str:
        with calls_lock:
            calls.append(item.config_id)
        if item.config_id == "001-fails":
            assert sibling_started.wait(timeout=5)
            raise RuntimeError("injected worker failure")
        if item.config_id == "002-drains":
            sibling_started.set()
            assert release_sibling.wait(timeout=5)
            sibling_finished.set()
            return "durable sibling result"
        pytest.fail("The coordinator admitted work after observing a failure.")

    captured: list[BaseException] = []

    def coordinate() -> None:
        try:
            run_bounded(items, slots, worker)
        except BaseException as error:
            captured.append(error)

    monkeypatch.setattr(parallel, "wait", observing_wait)
    coordinator = Thread(target=coordinate)
    coordinator.start()
    assert failure_returned_to_coordinator.wait(timeout=5)
    assert coordinator.is_alive()
    release_sibling.set()
    coordinator.join(timeout=5)

    assert not coordinator.is_alive()
    assert sibling_finished.is_set()
    assert len(captured) == 1
    error = captured[0]
    assert isinstance(error, parallel.ParallelRunError)
    assert isinstance(error.__cause__, RuntimeError)
    assert str(error.__cause__) == "injected worker failure"
    report = error.report
    assert [assignment.config_id for assignment in report.assignments] == [
        "001-fails",
        "002-drains",
    ]
    assert Counter(calls) == Counter({"001-fails": 1, "002-drains": 1})
    assert [result.status for result in report.results] == ["failed", "completed"]
    assert report.results[1].value == "durable sibling result"
    assert report.unadmitted_config_ids == ("003-must-not-start",)
    assert report.triggering_failure is report.results[0]


def test_single_slot_failure_never_invokes_later_configs() -> None:
    items = (
        WorkItem("001-fails", None),
        WorkItem("002-pending", None),
        WorkItem("003-pending", None),
    )
    calls: list[str] = []

    def worker(item: WorkItem[None], _slot: WorkerSlot[None]) -> None:
        calls.append(item.config_id)
        raise ValueError("stop")

    with pytest.raises(parallel.ParallelRunError) as caught:
        run_bounded(items, (WorkerSlot("only-slot"),), worker)

    assert calls == ["001-fails"]
    assert caught.value.report.unadmitted_config_ids == (
        "002-pending",
        "003-pending",
    )


@pytest.mark.parametrize(
    ("items", "slots", "message"),
    (
        (
            (WorkItem("same", 1), WorkItem("same", 2)),
            (WorkerSlot("slot-0"),),
            "Duplicate config ID",
        ),
        (
            (WorkItem("one", 1),),
            (WorkerSlot("same"), WorkerSlot("same")),
            "Duplicate worker slot ID",
        ),
        (
            (WorkItem("one", 1),),
            (),
            "At least one worker slot",
        ),
    ),
)
def test_invalid_claim_sets_fail_before_worker_invocation(
    items: tuple[WorkItem[int], ...],
    slots: tuple[WorkerSlot[None], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_bounded(
            items,
            slots,
            lambda _item, _slot: pytest.fail("invalid work must not start"),
        )


def test_empty_work_is_a_successful_noop() -> None:
    report = run_bounded((), (), lambda _item, _slot: None)

    assert report == parallel.ParallelReport((), (), ())
