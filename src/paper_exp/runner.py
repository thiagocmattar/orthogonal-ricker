"""Parent runner for one ordered experiment-set launch."""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Literal, Sequence
from uuid import uuid4

from yaml import YAMLError, safe_load

from paper_exp.config import load_config, validate_training_config
from paper_exp.integrity import classify_run_directory
from paper_exp.launch import (
    LaunchError,
    direct_launch_guard,
    repository_path,
    require_raw_output,
    require_token_cache_output,
    require_tracked_file,
    resolve_experiment_scaffold,
    resolve_launch_config,
)
from paper_exp.parallel import (
    ParallelRunError,
    WorkItem,
    WorkerSlot,
    run_bounded,
)


class RunnerError(LaunchError):
    """Raised when an experiment-set runner is malformed or a run fails."""


_NON_PRETRAIN_MODES = frozenset(
    {
        "activation-histograms",
        "activation-propagation",
        "calibrate",
        "clip-sweep",
        "prepare-data",
        "smoke",
        "weight-histograms",
    }
)
_WORKER_SLOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CUDA_DEVICE_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_CALIBRATION_PARALLEL_READY_ITEMS = ("CLOUD-01", "OPS-04", "OPS-05", "OPS-06")
_DEFINITIVE_PARALLEL_READY_ITEMS = (
    *_CALIBRATION_PARALLEL_READY_ITEMS,
    "OPS-09",
)


class WorkerProcessError(RuntimeError):
    """Raised when one isolated training worker does not complete."""


@dataclass(frozen=True)
class ParallelLaunchAuthorization:
    """Tracked case-runner authorization for one exact parallel GPU shape."""

    worker_count: int
    required_gpu_name: str
    config_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.worker_count, int) or self.worker_count < 2:
            raise ValueError("Parallel authorization worker_count must be at least two.")
        if (
            not isinstance(self.required_gpu_name, str)
            or not self.required_gpu_name.strip()
        ):
            raise ValueError(
                "Parallel authorization required_gpu_name must be a nonempty string."
            )
        if (
            not isinstance(self.config_ids, tuple)
            or not self.config_ids
            or any(
                not isinstance(config_id, str) or not config_id.strip()
                for config_id in self.config_ids
            )
            or len(set(self.config_ids)) != len(self.config_ids)
        ):
            raise ValueError(
                "Parallel authorization config_ids must be a nonempty tuple of "
                "distinct nonempty config IDs."
            )


def run_launch(
    runner_path: str | Path,
    config_paths: Sequence[str | Path],
    *,
    repository: str | Path | None = None,
    retry_failed: bool | None = None,
    worker_slots: Sequence[WorkerSlot[str]] | None = None,
    parallel_authorization: ParallelLaunchAuthorization | None = None,
    required_completed_config_ids: Sequence[str] = (),
) -> list[Path]:
    """Run one plan-defined config list under a single launch lock.

    A case-runner invocation containing ``--retry-failed`` explicitly opts in
    to retrying coherent failed attempts. Direct callers may pass the keyword
    instead. Without worker slots, configs run serially. With at least two
    explicit worker slots, pending configs run under one bounded coordinator
    with one isolated process per distinct homogeneous GPU. Config IDs listed
    in ``required_completed_config_ids`` must already have one coherent
    completed attempt; the runner fails closed instead of rerunning them.
    """

    root = repository_path(repository)
    runner = _resolve_runner(runner_path, root)
    retry_failed, resolved_slots = _resolve_launch_options(
        runner,
        requested_retry=retry_failed,
        requested_slots=worker_slots,
    )
    if not config_paths:
        raise RunnerError(f"Runner has no configs: {runner}")

    try:
        configs = [
            resolve_launch_config(path, repository=root)[1]
            for path in config_paths
        ]
    except LaunchError as error:
        raise RunnerError(str(error)) from error
    if resolved_slots and parallel_authorization is None:
        raise RunnerError(
            "Concurrent definitive pretraining requires an explicit tracked "
            "parallel launch authorization from the case runner."
        )
    if (
        resolved_slots
        and parallel_authorization is not None
        and len(resolved_slots) != parallel_authorization.worker_count
    ):
        raise RunnerError(
            "Concurrent definitive pretraining requires exactly "
            f"{parallel_authorization.worker_count} authorized worker slots; "
            f"received {len(resolved_slots)}."
        )
    config_ids = tuple(path.stem for path in configs)
    required_completed = tuple(required_completed_config_ids)
    if (
        any(
            not isinstance(config_id, str) or not config_id.strip()
            for config_id in required_completed
        )
        or len(set(required_completed)) != len(required_completed)
    ):
        raise RunnerError(
            "Required completed config IDs must be distinct nonempty strings."
        )
    unknown_required = [
        config_id for config_id in required_completed if config_id not in config_ids
    ]
    if unknown_required:
        raise RunnerError(
            "Required completed config IDs are not present in this runner: "
            + ", ".join(unknown_required)
        )
    required_set = set(required_completed)
    runner_order = tuple(
        config_id for config_id in config_ids if config_id in required_set
    )
    if required_completed != runner_order:
        raise RunnerError(
            "Required completed config IDs must follow the runner config order."
        )
    if (
        resolved_slots
        and parallel_authorization is not None
        and config_ids != parallel_authorization.config_ids
    ):
        raise RunnerError(
            "Concurrent definitive pretraining config IDs do not match the "
            "tracked parallel launch authorization: expected "
            f"{parallel_authorization.config_ids!r}, received {config_ids!r}."
        )
    if len(resolved_slots) > len(configs):
        raise RunnerError(
            "Concurrent definitive pretraining cannot allocate more worker "
            "slots than tranche configs."
        )
    expected_config_root = runner.parent
    misplaced = [path for path in configs if path.parent != expected_config_root]
    if misplaced:
        raise RunnerError(
            f"Runner {runner} may only use configs beside it in its scaffold "
            "run directory."
        )
    if len(set(configs)) != len(configs):
        raise RunnerError(f"Runner contains a duplicate config: {runner}")
    prefixes = [int(path.name.split("-", 1)[0]) for path in configs]
    if prefixes != sorted(prefixes) or len(prefixes) != len(set(prefixes)):
        raise RunnerError(f"Runner configs must be in strictly increasing numeric order: {runner}")
    folder_configs = {
        path.resolve()
        for path in expected_config_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    }
    listed_configs = set(configs)
    if listed_configs != folder_configs:
        missing = sorted(path.name for path in folder_configs - listed_configs)
        unexpected = sorted(path.name for path in listed_configs - folder_configs)
        details: list[str] = []
        if missing:
            details.append(f"missing from CONFIGS: {', '.join(missing)}")
        if unexpected:
            details.append(f"not present in the folder: {', '.join(unexpected)}")
        raise RunnerError(
            f"Runner {runner.name} must list exactly all YAML files in "
            f"{expected_config_root.relative_to(root).as_posix()}/ "
            f"({'; '.join(details)})."
        )

    loaded: list[tuple[Path, dict[str, Any]]] = []
    if resolved_slots:
        _require_parallel_launch_ready(root, mode="pretrain")
    for path in configs:
        config = load_config(path, allow_todos=False)
        validate_training_config(config)
        if resolved_slots:
            _require_worker_mappable_device(config, config_path=path)
        require_raw_output(config, repository=root, config_path=path)
        require_token_cache_output(config, repository=root, source=path)
        loaded.append((path, config))

    command_parts = [
        Path(sys.executable).name,
        runner.relative_to(root).as_posix(),
    ]
    if retry_failed:
        command_parts.append("--retry-failed")
    for slot in resolved_slots:
        command_parts.extend(
            ("--worker-slot", f"{slot.slot_id}={slot.payload}")
        )
    command = shlex.join(command_parts)
    prior_completed = [
        _completed_attempt_for_config(path, config, allow_failed_retry=retry_failed)
        for path, config in loaded
    ]
    _require_completed_reuse(config_ids, prior_completed, required_completed)
    with direct_launch_guard(repository=root):
        # Close the preflight-to-lock race before creating any new attempt.
        prior_completed = [
            _completed_attempt_for_config(path, config, allow_failed_retry=retry_failed)
            for path, config in loaded
        ]
        _require_completed_reuse(config_ids, prior_completed, required_completed)
        pending: list[WorkItem[tuple[int, Path, dict[str, Any]]]] = []
        completed_by_config: dict[str, Path] = {}
        for index, ((path, config), existing) in enumerate(
            zip(loaded, prior_completed, strict=True), start=1
        ):
            display = path.relative_to(root).as_posix()
            if existing is not None:
                run_display = existing.relative_to(root).as_posix()
                print(
                    f"[{index}/{len(loaded)}] skip completed {display} -> {run_display}",
                    flush=True,
                )
                completed_by_config[path.stem] = existing
                continue
            pending.append(
                WorkItem(
                    config_id=path.stem,
                    payload=(index, path, config),
                )
            )

        if not resolved_slots:
            for item in pending:
                index, path, config = item.payload
                display = path.relative_to(root).as_posix()
                print(f"[{index}/{len(loaded)}] pretrain {display}", flush=True)
                completed_by_config[item.config_id] = _run_one(
                    config,
                    config_path=path,
                    command=command,
                )
        elif pending:
            assert parallel_authorization is not None
            launch_id = uuid4().hex
            slot_identities = _probe_worker_slots(resolved_slots)
            _require_authorized_worker_identities(
                slot_identities,
                authorization=parallel_authorization,
            )

            def worker(
                item: WorkItem[tuple[int, Path, dict[str, Any]]],
                slot: WorkerSlot[str],
            ) -> Path:
                index, path, config = item.payload
                display = path.relative_to(root).as_posix()
                print(
                    f"[{index}/{len(loaded)}] pretrain {display} "
                    f"on {slot.slot_id} (CUDA_VISIBLE_DEVICES={slot.payload})",
                    flush=True,
                )
                return _run_one_isolated(
                    config,
                    config_path=path,
                    command=command,
                    repository=root,
                    launch_id=launch_id,
                    launch_position=index,
                    launch_size=len(loaded),
                    slot=slot,
                    expected_identity=slot_identities[slot.slot_id],
                    mode="pretrain",
                )

            try:
                report = run_bounded(pending, resolved_slots, worker)
            except ParallelRunError as error:
                raise RunnerError(str(error)) from error
            completed_by_config.update(
                {
                    result.assignment.config_id: result.value
                    for result in report.completed
                    if result.value is not None
                }
            )

        completed = [completed_by_config[path.stem] for path, _config in loaded]
    return completed


def _require_completed_reuse(
    config_ids: Sequence[str],
    completed_attempts: Sequence[Path | None],
    required_completed_config_ids: Sequence[str],
) -> None:
    required = set(required_completed_config_ids)
    missing = [
        config_id
        for config_id, completed in zip(config_ids, completed_attempts, strict=True)
        if config_id in required and completed is None
    ]
    if missing:
        raise RunnerError(
            "Required completed reuse is unavailable for: "
            + ", ".join(missing)
            + ". Refusing to rerun these configs."
        )


def run_calibrations(
    config_paths: Sequence[str | Path],
    *,
    command: str,
    repository: str | Path | None = None,
    worker_slots: Sequence[WorkerSlot[str]] = (),
) -> list[Path]:
    """Run exact calibration configs under one coordinator and launch lock.

    The no-slot form accepts exactly one config and is the solo calibration.
    The bounded form requires at least two distinct configs and two distinct
    homogeneous physical GPUs. More configs than slots are admitted in stable
    order as slots become free. Calibration attempts are always fresh and use
    lifecycle mode ``calibrate``; they are never reused as pretraining.
    """

    root = repository_path(repository)
    slots = tuple(worker_slots)
    _validate_worker_slots(slots)
    if not config_paths:
        raise RunnerError("Calibration requires at least one exact config.")
    try:
        configs = [
            resolve_launch_config(path, repository=root)[1]
            for path in config_paths
        ]
    except LaunchError as error:
        raise RunnerError(str(error)) from error
    if len(set(configs)) != len(configs):
        raise RunnerError("Concurrent calibration configs must be distinct.")
    config_roots = {path.parent for path in configs}
    if len(config_roots) != 1:
        raise RunnerError(
            "One calibration coordinator may use configs from only one scaffold."
        )
    if slots:
        if len(slots) < 2 or len(configs) < 2:
            raise RunnerError(
                "Concurrent calibration requires at least two distinct configs "
                "and two distinct worker slots."
            )
        if len(slots) > len(configs):
            raise RunnerError(
                "Concurrent calibration cannot allocate more worker slots than configs."
            )
        _require_parallel_launch_ready(root, mode="calibrate")
    elif len(configs) != 1:
        raise RunnerError(
            "A calibration without worker slots accepts exactly one config."
        )

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in configs:
        config = load_config(path, allow_todos=False)
        validate_training_config(config)
        if slots:
            _require_worker_mappable_device(config, config_path=path)
        require_raw_output(config, repository=root, config_path=path)
        require_token_cache_output(config, repository=root, source=path)
        loaded.append((path, config))

    with direct_launch_guard(repository=root):
        if not slots:
            path, config = loaded[0]
            display = path.relative_to(root).as_posix()
            print(f"[1/1] calibrate {display}", flush=True)
            return [
                _run_one_calibration(
                    config,
                    config_path=path,
                    command=command,
                )
            ]

        launch_id = uuid4().hex
        slot_identities = _probe_worker_slots(slots)
        pending = tuple(
            WorkItem(
                config_id=path.stem,
                payload=(index, path, config),
            )
            for index, (path, config) in enumerate(loaded, start=1)
        )

        def worker(
            item: WorkItem[tuple[int, Path, dict[str, Any]]],
            slot: WorkerSlot[str],
        ) -> Path:
            index, path, config = item.payload
            display = path.relative_to(root).as_posix()
            print(
                f"[{index}/{len(loaded)}] calibrate {display} "
                f"on {slot.slot_id} (CUDA_VISIBLE_DEVICES={slot.payload})",
                flush=True,
            )
            return _run_one_isolated(
                config,
                config_path=path,
                command=command,
                repository=root,
                launch_id=launch_id,
                launch_position=index,
                launch_size=len(loaded),
                slot=slot,
                expected_identity=slot_identities[slot.slot_id],
                mode="calibrate",
            )

        try:
            report = run_bounded(pending, slots, worker)
        except ParallelRunError as error:
            raise RunnerError(str(error)) from error
        completed_by_config = {
            result.assignment.config_id: result.value
            for result in report.completed
            if result.value is not None
        }
        return [completed_by_config[path.stem] for path, _config in loaded]


def _completed_attempt_for_config(
    config_path: Path,
    config: dict[str, Any],
    *,
    allow_failed_retry: bool,
) -> Path | None:
    """Return one reusable completion, or require a safe retry state."""

    result_group = config_path.parent.parent / "raw" / config_path.stem
    if result_group.is_symlink():
        raise RunnerError(
            f"Cannot resume {config_path.name}: result group is not a regular "
            f"directory: {result_group}"
        )
    if not result_group.exists():
        return None
    if not result_group.is_dir():
        raise RunnerError(
            f"Cannot resume {config_path.name}: result group is not a regular "
            f"directory: {result_group}"
        )

    attempts: list[tuple[Path, str]] = []
    for attempt in sorted(result_group.iterdir(), key=lambda path: path.name):
        if attempt.is_symlink() or not attempt.is_dir():
            attempts.append((attempt, "inconsistent"))
            continue
        manifest = _read_manifest(attempt / "manifest.json")
        mode = manifest.get("mode") if manifest is not None else None
        if mode in _NON_PRETRAIN_MODES:
            continue
        if manifest is None or mode != "pretrain":
            attempts.append((attempt, "inconsistent"))
            continue
        status = classify_run_directory(attempt)
        if status == "complete" and manifest.get("status") != "completed":
            status = "statusless"
        if status != "inconsistent" and not _config_snapshot_matches(
            attempt / "config.yaml", config
        ):
            status = "inconsistent"
        attempts.append((attempt, status))

    unsafe = [
        (attempt, status)
        for attempt, status in attempts
        if status in {"running", "inconsistent", "statusless"}
    ]
    if unsafe:
        details = ", ".join(
            f"{attempt.name}={status}" for attempt, status in unsafe
        )
        raise RunnerError(
            f"Cannot resume {config_path.name} safely; unsafe attempt state: "
            f"{details}."
        )

    coherent_completed = [
        attempt for attempt, status in attempts if status == "complete"
    ]
    if len(coherent_completed) > 1:
        names = ", ".join(path.name for path in coherent_completed)
        raise RunnerError(
            f"Cannot resume {config_path.name}: multiple coherent completed "
            f"attempts are ambiguous: {names}."
        )
    if coherent_completed:
        return coherent_completed[0]

    failed = [attempt for attempt, status in attempts if status == "failed"]
    if failed and not allow_failed_retry:
        names = ", ".join(path.name for path in failed)
        raise RunnerError(
            f"Cannot retry {config_path.name} without explicit recovery "
            f"authorization; coherent failed attempts: {names}. Record the "
            "reviewed infrastructure failure, then invoke the case runner "
            "with --retry-failed."
        )
    return None


def _resolve_launch_options(
    runner: Path,
    *,
    requested_retry: bool | None,
    requested_slots: Sequence[WorkerSlot[str]] | None,
) -> tuple[bool, tuple[WorkerSlot[str], ...]]:
    try:
        invoked_runner = Path(sys.argv[0]).resolve() == runner
    except (OSError, RuntimeError):
        invoked_runner = False
    parsed_retry, parsed_slots = (
        _parse_runner_arguments(sys.argv[1:])
        if invoked_runner
        else (False, ())
    )
    retry = parsed_retry if requested_retry is None else requested_retry
    slots = parsed_slots if requested_slots is None else tuple(requested_slots)
    _validate_worker_slots(slots)
    return retry, slots


def _parse_runner_arguments(
    arguments: Sequence[str],
) -> tuple[bool, tuple[WorkerSlot[str], ...]]:
    retry_failed = False
    slots: list[WorkerSlot[str]] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--retry-failed":
            if retry_failed:
                raise RunnerError(
                    "Unsupported case-runner arguments: --retry-failed may appear "
                    "only once."
                )
            retry_failed = True
            index += 1
            continue
        if argument == "--worker-slot":
            if index + 1 >= len(arguments):
                raise RunnerError("Case runner --worker-slot requires SLOT=CUDA_DEVICE.")
            slots.append(parse_worker_slot(arguments[index + 1]))
            index += 2
            continue
        if argument.startswith("--worker-slot="):
            slots.append(parse_worker_slot(argument.split("=", 1)[1]))
            index += 1
            continue
        rendered = " ".join(arguments)
        raise RunnerError(
            f"Unsupported case-runner arguments: {rendered}. Supported arguments "
            "are --retry-failed and repeated --worker-slot SLOT=CUDA_DEVICE."
        )
    result = tuple(slots)
    _validate_worker_slots(result)
    return retry_failed, result


def parse_worker_slot(value: str) -> WorkerSlot[str]:
    """Parse one explicit CUDA worker mapping shared by runners and smoke."""

    slot_id, separator, cuda_device = value.partition("=")
    if not separator or not _WORKER_SLOT_ID_RE.fullmatch(slot_id):
        raise RunnerError(
            "Worker slot must be SLOT=CUDA_DEVICE with a lowercase alphanumeric "
            "or hyphenated slot ID."
        )
    if (
        not cuda_device
        or "," in cuda_device
        or any(character.isspace() for character in cuda_device)
        or _CUDA_DEVICE_RE.fullmatch(cuda_device) is None
    ):
        raise RunnerError("Each worker slot must expose exactly one CUDA device.")
    return WorkerSlot(slot_id=slot_id, payload=cuda_device)


def _validate_worker_slots(slots: Sequence[WorkerSlot[str]]) -> None:
    slot_ids = [slot.slot_id for slot in slots]
    if len(set(slot_ids)) != len(slot_ids):
        raise RunnerError("Case runner worker slot IDs must be unique.")
    for slot in slots:
        if slot.payload is None:
            raise RunnerError("Every case runner worker slot requires a CUDA device.")
        parse_worker_slot(f"{slot.slot_id}={slot.payload}")
    devices = [str(slot.payload) for slot in slots]
    duplicate_devices = sorted(
        {device for device in devices if devices.count(device) > 1}
    )
    if duplicate_devices:
        raise RunnerError(
            "GPU worker slots must map to distinct CUDA device ordinals; "
            "same-device packing requires a future launch contract bound to an "
            f"exact reviewed hardware profile: {', '.join(duplicate_devices)}."
        )


def _require_worker_mappable_device(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> None:
    requested = config["training"]["device"]
    if requested not in {"cuda", "cuda:0"}:
        raise RunnerError(
            f"Parallel config {config_path.name} must select cuda or cuda:0; "
            "the coordinator maps its assigned physical GPU through "
            "CUDA_VISIBLE_DEVICES."
        )
    if config["training"]["precision"] != "bfloat16":
        raise RunnerError(
            f"Parallel config {config_path.name} must select bfloat16 precision; "
            "the isolated worker verifies native BF16 support before training."
        )


def _require_parallel_launch_ready(
    repository: Path,
    *,
    mode: Literal["pretrain", "calibrate"] = "calibrate",
) -> None:
    workboard = repository / "docs" / "experimental-design" / "workboard.md"
    try:
        text = workboard.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise RunnerError(f"Cannot read parallel-launch workboard: {workboard}") from error
    items = (
        _DEFINITIVE_PARALLEL_READY_ITEMS
        if mode == "pretrain"
        else _CALIBRATION_PARALLEL_READY_ITEMS
    )
    unresolved = [
        item
        for item in items
        if re.search(
            rf"^\|\s*`{re.escape(item)}`\s*\|\s*resolved\s*\|",
            text,
            flags=re.MULTILINE,
        )
        is None
    ]
    if unresolved:
        workflow = "Definitive pretraining" if mode == "pretrain" else "Calibration"
        raise RunnerError(
            f"Concurrent {workflow.lower()} remains blocked until these workboard "
            f"items are resolved: {', '.join(unresolved)}."
        )


def _require_authorized_worker_identities(
    identities: dict[str, dict[str, Any]],
    *,
    authorization: ParallelLaunchAuthorization,
) -> None:
    mismatches = [
        f"{slot_id}={identity.get('name')!r}"
        for slot_id, identity in identities.items()
        if identity.get("name") != authorization.required_gpu_name
    ]
    if mismatches:
        raise RunnerError(
            "Definitive worker GPU identity does not match the tracked parallel "
            f"authorization ({authorization.required_gpu_name!r} required): "
            + ", ".join(mismatches)
            + "."
        )


def _run_one_isolated(
    config: dict[str, Any],
    *,
    config_path: Path,
    command: str,
    repository: Path,
    launch_id: str,
    launch_position: int,
    launch_size: int,
    slot: WorkerSlot[str],
    expected_identity: dict[str, Any],
    mode: Literal["pretrain", "calibrate"] = "calibrate",
) -> Path:
    """Run one training mode in a fresh process on one explicit visible GPU."""

    if mode not in {"pretrain", "calibrate"}:
        raise RunnerError(f"Unsupported isolated training mode: {mode}")

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_process_entry,
        args=(
            config,
            str(config_path),
            command,
            str(repository),
            launch_id,
            launch_position,
            launch_size,
            slot.slot_id,
            slot.payload,
            expected_identity,
            os.getpid(),
            sender,
            mode,
        ),
        name=f"paper-exp-{slot.slot_id}-{config_path.stem}",
    )
    try:
        process.start()
    except BaseException:
        sender.close()
        receiver.close()
        raise
    sender.close()
    message, exit_code = _receive_and_reap_worker(process, receiver)

    if (
        isinstance(message, dict)
        and message.get("status") == "completed"
        and exit_code == 0
    ):
        return Path(str(message["run_dir"]))

    if isinstance(message, dict) and message.get("status") == "failed":
        detail = f"{message.get('error_type')}: {message.get('error_message')}"
    else:
        detail = "worker exited without a terminal result"
    raise WorkerProcessError(
        f"Isolated {mode} worker for {config_path.name} on {slot.slot_id} failed "
        f"(exit code {exit_code}): {detail}."
    )


def _receive_and_reap_worker(process: Any, receiver: Any) -> tuple[Any, int | None]:
    """Receive one bounded result, drain the child, and close a dead process."""

    lifecycle_error: BaseException | None = None
    message: Any = None
    exit_code: int | None = None
    try:
        try:
            message = receiver.recv()
        except EOFError:
            message = None
        except BaseException as error:
            lifecycle_error = error
    finally:
        try:
            receiver.close()
        except BaseException as error:
            if lifecycle_error is None:
                lifecycle_error = error
        try:
            process.join()
        except BaseException as error:
            if lifecycle_error is None:
                lifecycle_error = error
        try:
            process_alive = bool(process.is_alive())
        except BaseException as error:
            process_alive = True
            if lifecycle_error is None:
                lifecycle_error = error
        if process_alive and lifecycle_error is None:
            lifecycle_error = WorkerProcessError(
                "Isolated worker remained live after a blocking join."
            )
        if not process_alive:
            try:
                exit_code = process.exitcode
            except BaseException as error:
                if lifecycle_error is None:
                    lifecycle_error = error
            try:
                process.close()
            except BaseException as error:
                if lifecycle_error is None:
                    lifecycle_error = error
    if lifecycle_error is not None:
        raise WorkerProcessError(
            "Failed while receiving or reaping an isolated worker; its process "
            "handle was closed only after confirming that the child had exited."
        ) from lifecycle_error
    return message, exit_code


def _worker_process_entry(
    config: dict[str, Any],
    config_path: str,
    command: str,
    repository: str,
    launch_id: str,
    launch_position: int,
    launch_size: int,
    slot_id: str,
    cuda_visible_device: str,
    expected_identity: dict[str, Any],
    coordinator_pid: int,
    sender: Any,
    mode: Literal["pretrain", "calibrate"] = "calibrate",
) -> None:
    """Training child entrypoint; the parent remains the sole coordinator."""

    os.chdir(repository)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
    os.environ["PAPER_EXP_PARALLEL_LAUNCH_ID"] = launch_id
    os.environ["PAPER_EXP_WORKER_SLOT_ID"] = slot_id
    os.environ["PAPER_EXP_WORKER_CONFIG_ID"] = Path(config_path).stem
    os.environ["PAPER_EXP_WORKER_LAUNCH_POSITION"] = str(launch_position)
    os.environ["PAPER_EXP_WORKER_LAUNCH_SIZE"] = str(launch_size)
    os.environ["PAPER_EXP_COORDINATOR_PID"] = str(coordinator_pid)
    try:
        gpu = _require_isolated_cuda_runtime(config)
        if gpu != expected_identity:
            raise WorkerProcessError(
                "Assigned CUDA device or runtime identity changed after coordinator "
                f"preflight: expected {expected_identity!r}, observed {gpu!r}."
            )
        os.environ["PAPER_EXP_WORKER_GPU_UUID"] = gpu["uuid"]
        os.environ["PAPER_EXP_WORKER_GPU_NAME"] = gpu["name"]
        os.environ["PAPER_EXP_WORKER_GPU_TOTAL_MEMORY_BYTES"] = str(
            gpu["total_memory_bytes"]
        )
        os.environ["PAPER_EXP_WORKER_GPU_COMPUTE_CAPABILITY"] = gpu[
            "compute_capability"
        ]
        os.environ["PAPER_EXP_WORKER_TORCH_VERSION"] = gpu["torch_version"]
        os.environ["PAPER_EXP_WORKER_CUDA_RUNTIME_VERSION"] = gpu[
            "cuda_runtime_version"
        ]
        run_one = _run_one if mode == "pretrain" else _run_one_calibration
        run_dir = run_one(
            config,
            config_path=Path(config_path),
            command=command,
        )
    except BaseException as error:
        sender.send(
            {
                "status": "failed",
                "error_type": type(error).__qualname__,
                "error_message": _bounded_error_message(error),
            }
        )
        raise
    else:
        sender.send({"status": "completed", "run_dir": str(run_dir)})
    finally:
        sender.close()


def _bounded_error_message(error: BaseException, *, limit: int = 4096) -> str:
    message = str(error)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _probe_worker_slots(
    slots: Sequence[WorkerSlot[str]],
) -> dict[str, dict[str, Any]]:
    """Resolve every ordinal to one stable, distinct physical GPU identity."""

    identities = {slot.slot_id: _probe_cuda_slot(slot) for slot in slots}
    uuids = [str(identity["uuid"]) for identity in identities.values()]
    if len(set(uuids)) != len(uuids):
        raise RunnerError(
            "Distinct CUDA device ordinals resolved to the same physical GPU UUID."
        )
    hardware = {
        (
            str(identity["name"]),
            int(identity["total_memory_bytes"]),
            str(identity["compute_capability"]),
            str(identity["torch_version"]),
            str(identity["cuda_runtime_version"]),
        )
        for identity in identities.values()
    }
    if len(hardware) != 1:
        raise RunnerError(
            "All calibration worker slots must resolve to one homogeneous GPU and "
            "Torch/CUDA runtime."
        )
    return identities


def _probe_cuda_slot(slot: WorkerSlot[str]) -> dict[str, Any]:
    """Probe one slot in a fresh process before any scientific attempt starts."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_cuda_probe_process_entry,
        args=(str(slot.payload), sender),
        name=f"paper-exp-probe-{slot.slot_id}",
    )
    try:
        process.start()
    except BaseException:
        sender.close()
        receiver.close()
        raise
    sender.close()
    message, exit_code = _receive_and_reap_worker(process, receiver)
    if (
        isinstance(message, dict)
        and message.get("status") == "completed"
        and isinstance(message.get("gpu"), dict)
        and exit_code == 0
    ):
        return dict(message["gpu"])
    if isinstance(message, dict) and message.get("status") == "failed":
        detail = f"{message.get('error_type')}: {message.get('error_message')}"
    else:
        detail = "probe exited without a terminal result"
    raise RunnerError(
        f"CUDA probe for slot {slot.slot_id} failed (exit code {exit_code}): {detail}."
    )


def _cuda_probe_process_entry(cuda_visible_device: str, sender: Any) -> None:
    """Fresh-process CUDA probe entrypoint."""

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
    try:
        gpu = _probe_visible_cuda_device()
    except BaseException as error:
        sender.send(
            {
                "status": "failed",
                "error_type": type(error).__qualname__,
                "error_message": _bounded_error_message(error),
            }
        )
        raise
    else:
        sender.send({"status": "completed", "gpu": gpu})
    finally:
        sender.close()


def _require_isolated_cuda_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the child sees one BF16-capable CUDA device."""

    if config["training"]["device"] not in {"cuda", "cuda:0"}:
        raise WorkerProcessError(
            "Isolated GPU worker requires training.device cuda or cuda:0."
        )
    if config["training"]["precision"] != "bfloat16":
        raise WorkerProcessError(
            "Isolated GPU worker requires training.precision bfloat16."
        )
    return _probe_visible_cuda_device()


def _probe_visible_cuda_device() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise WorkerProcessError(
            "Isolated GPU worker cannot import torch after device isolation."
        ) from error
    if not torch.cuda.is_available():
        raise WorkerProcessError(
            "Assigned CUDA device is unavailable after applying CUDA_VISIBLE_DEVICES."
        )
    visible_devices = int(torch.cuda.device_count())
    if visible_devices != 1:
        raise WorkerProcessError(
            "Isolated GPU worker must see exactly one CUDA device; "
            f"observed {visible_devices}."
        )
    if not torch.cuda.is_bf16_supported():
        raise WorkerProcessError("Assigned CUDA device does not support BF16.")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = str(getattr(properties, "uuid", "")).strip()
    gpu_name = str(getattr(properties, "name", "")).strip()
    total_memory = int(getattr(properties, "total_memory", 0))
    major = int(getattr(properties, "major", -1))
    minor = int(getattr(properties, "minor", -1))
    torch_version = str(getattr(torch, "__version__", "")).strip()
    torch_runtime = getattr(torch, "version", None)
    cuda_runtime_version = str(
        getattr(torch_runtime, "cuda", "") or ""
    ).strip()
    if not gpu_uuid or not gpu_name or total_memory <= 0 or major < 0 or minor < 0:
        raise WorkerProcessError(
            "Assigned CUDA device did not expose complete UUID, name, memory, and "
            "compute-capability identity."
        )
    if not torch_version or not cuda_runtime_version:
        raise WorkerProcessError(
            "Assigned CUDA device did not expose exact Torch and CUDA runtime versions."
        )
    return {
        "uuid": gpu_uuid,
        "name": gpu_name,
        "total_memory_bytes": total_memory,
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch_version,
        "cuda_runtime_version": cuda_runtime_version,
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _config_snapshot_matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            snapshot = safe_load(handle)
    except (OSError, UnicodeError, YAMLError):
        return False
    return isinstance(snapshot, dict) and snapshot == expected


def _resolve_runner(path: str | Path, repository: Path) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository / candidate).resolve()
    )
    experiments_root = (repository / "experiments").resolve()
    try:
        relative = resolved.relative_to(experiments_root)
    except ValueError as error:
        raise RunnerError(f"Runner must be inside {experiments_root}: {resolved}") from error
    if (
        len(relative.parts) != 3
        or relative.parts[1:] != ("run", "runner.py")
    ):
        raise RunnerError(
            "Runner must be experiments/NN-<phase>-<tranche>/run/runner.py."
        )
    scaffold = resolve_experiment_scaffold(relative.parts[0], repository=repository)
    if (
        scaffold.is_smoke
        or resolved != scaffold.runner_path
        or not resolved.is_file()
    ):
        raise RunnerError(
            "Scientific runner must be the tracked runner.py in a nonzero scaffold."
        )
    require_tracked_file(repository, resolved)
    return resolved


def _run_one(
    config: dict[str, Any],
    *,
    config_path: Path,
    command: str,
) -> Path:
    from paper_exp.training import run_training

    return run_training(
        config,
        config_path=config_path,
        command=command,
        mode="pretrain",
    )


def _run_one_calibration(
    config: dict[str, Any],
    *,
    config_path: Path,
    command: str,
) -> Path:
    from paper_exp.training import run_training

    return run_training(
        config,
        config_path=config_path,
        command=command,
        mode="calibrate",
    )
