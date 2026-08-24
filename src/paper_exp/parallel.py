"""Bounded config scheduling for one authoritative launch coordinator.

This module owns only in-process admission and result accounting.  A caller
provides the worker function that creates the required subprocess or remote
isolation for each explicit slot.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generic, Literal, Sequence, TypeVar


ItemT = TypeVar("ItemT")
SlotT = TypeVar("SlotT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class WorkItem(Generic[ItemT]):
    """One distinct immutable config and the caller-owned execution payload."""

    config_id: str
    payload: ItemT


@dataclass(frozen=True)
class WorkerSlot(Generic[SlotT]):
    """One explicit worker slot, such as a subprocess, Pod, or GPU identity."""

    slot_id: str
    payload: SlotT | None = None


@dataclass(frozen=True)
class Assignment:
    """Coordinator record written when one config is admitted to one slot."""

    admission_index: int
    config_id: str
    slot_id: str
    admitted_at: str


@dataclass(frozen=True)
class WorkResult(Generic[ResultT]):
    """Terminal result for an admitted assignment."""

    assignment: Assignment
    status: Literal["completed", "failed"]
    started_at: str
    finished_at: str
    value: ResultT | None = None
    error_type: str | None = None
    error_message: str | None = None
    _error: BaseException | None = field(default=None, repr=False, compare=False)

    @property
    def error(self) -> BaseException | None:
        """Return the original worker exception for local recovery or chaining."""

        return self._error


@dataclass(frozen=True)
class ParallelReport(Generic[ResultT]):
    """Complete coordinator record after every admitted worker has drained."""

    assignments: tuple[Assignment, ...]
    results: tuple[WorkResult[ResultT], ...]
    unadmitted_config_ids: tuple[str, ...]
    triggering_failure: WorkResult[ResultT] | None = None

    @property
    def completed(self) -> tuple[WorkResult[ResultT], ...]:
        return tuple(result for result in self.results if result.status == "completed")

    @property
    def failed(self) -> tuple[WorkResult[ResultT], ...]:
        return tuple(result for result in self.results if result.status == "failed")


class ParallelRunError(RuntimeError):
    """Raised after a worker failure and after all admitted siblings drain."""

    def __init__(self, report: ParallelReport[object]) -> None:
        failure = report.triggering_failure
        if failure is None:
            raise ValueError("A parallel run error requires a triggering failure.")
        assignment = failure.assignment
        message = (
            f"Config {assignment.config_id} failed on slot {assignment.slot_id}: "
            f"{failure.error_type}: {failure.error_message}"
        )
        if report.unadmitted_config_ids:
            message += (
                f"; {len(report.unadmitted_config_ids)} config(s) were not admitted"
            )
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class _Running(Generic[ItemT, SlotT]):
    item: WorkItem[ItemT]
    slot: WorkerSlot[SlotT]
    assignment: Assignment


def run_bounded(
    items: Sequence[WorkItem[ItemT]],
    slots: Sequence[WorkerSlot[SlotT]],
    worker: Callable[[WorkItem[ItemT], WorkerSlot[SlotT]], ResultT],
) -> ParallelReport[ResultT]:
    """Execute distinct configs with bounded, fail-stop admission.

    Admission follows ``items`` order.  Initially free slots follow ``slots``
    order; after completions, simultaneously freed slots are reused in that
    same order.  The coordinator admits each config at most once.  Once it
    observes any worker failure, it admits nothing else, waits for every
    already-admitted worker to finish, and raises :class:`ParallelRunError`
    carrying the complete report.
    """

    queued = tuple(items)
    worker_slots = tuple(slots)
    _validate_inputs(queued, worker_slots)
    if not queued:
        return ParallelReport((), (), ())

    slot_order = {slot.slot_id: index for index, slot in enumerate(worker_slots)}
    available = list(worker_slots)
    running: dict[Future[WorkResult[ResultT]], _Running[ItemT, SlotT]] = {}
    assignments: list[Assignment] = []
    results: dict[int, WorkResult[ResultT]] = {}
    next_item = 0
    triggering_failure: WorkResult[ResultT] | None = None

    with ThreadPoolExecutor(
        max_workers=len(worker_slots),
        thread_name_prefix="paper-exp-worker",
    ) as executor:
        next_item = _admit_available(
            queued,
            available,
            next_item,
            worker,
            executor,
            assignments,
            running,
        )

        while running:
            done, _ = wait(tuple(running), return_when="FIRST_COMPLETED")
            finished = sorted(
                done,
                key=lambda future: running[future].assignment.admission_index,
            )
            freed: list[WorkerSlot[SlotT]] = []
            batch_results: list[WorkResult[ResultT]] = []
            for future in finished:
                active = running.pop(future)
                result = future.result()
                results[active.assignment.admission_index] = result
                batch_results.append(result)
                freed.append(active.slot)

            failures = [result for result in batch_results if result.status == "failed"]
            if failures and triggering_failure is None:
                triggering_failure = failures[0]

            available.extend(freed)
            available.sort(key=lambda slot: slot_order[slot.slot_id])
            if triggering_failure is None:
                next_item = _admit_available(
                    queued,
                    available,
                    next_item,
                    worker,
                    executor,
                    assignments,
                    running,
                )

    ordered_results = tuple(results[index] for index in sorted(results))
    report = ParallelReport(
        assignments=tuple(assignments),
        results=ordered_results,
        unadmitted_config_ids=tuple(item.config_id for item in queued[next_item:]),
        triggering_failure=triggering_failure,
    )
    if triggering_failure is not None:
        error = ParallelRunError(report)
        raise error from triggering_failure.error
    return report


def _admit_available(
    items: tuple[WorkItem[ItemT], ...],
    available: list[WorkerSlot[SlotT]],
    next_item: int,
    worker: Callable[[WorkItem[ItemT], WorkerSlot[SlotT]], ResultT],
    executor: ThreadPoolExecutor,
    assignments: list[Assignment],
    running: dict[Future[WorkResult[ResultT]], _Running[ItemT, SlotT]],
) -> int:
    while available and next_item < len(items):
        item = items[next_item]
        slot = available.pop(0)
        assignment = Assignment(
            admission_index=next_item,
            config_id=item.config_id,
            slot_id=slot.slot_id,
            admitted_at=_utc_now(),
        )
        future = executor.submit(_invoke_worker, item, slot, assignment, worker)
        assignments.append(assignment)
        running[future] = _Running(item, slot, assignment)
        next_item += 1
    return next_item


def _invoke_worker(
    item: WorkItem[ItemT],
    slot: WorkerSlot[SlotT],
    assignment: Assignment,
    worker: Callable[[WorkItem[ItemT], WorkerSlot[SlotT]], ResultT],
) -> WorkResult[ResultT]:
    started_at = _utc_now()
    try:
        value = worker(item, slot)
    except BaseException as error:
        return WorkResult(
            assignment=assignment,
            status="failed",
            started_at=started_at,
            finished_at=_utc_now(),
            error_type=type(error).__qualname__,
            error_message=str(error),
            _error=error,
        )
    return WorkResult(
        assignment=assignment,
        status="completed",
        started_at=started_at,
        finished_at=_utc_now(),
        value=value,
    )


def _validate_inputs(
    items: tuple[WorkItem[object], ...],
    slots: tuple[WorkerSlot[object], ...],
) -> None:
    if items and not slots:
        raise ValueError("At least one worker slot is required for nonempty work.")
    _require_distinct_nonempty(
        [item.config_id for item in items],
        label="config ID",
    )
    _require_distinct_nonempty(
        [slot.slot_id for slot in slots],
        label="worker slot ID",
    )


def _require_distinct_nonempty(values: list[str], *, label: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"Every {label} must be a nonempty string.")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}(s): {', '.join(duplicates)}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
