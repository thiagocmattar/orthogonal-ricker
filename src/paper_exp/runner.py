"""Parent runner for one ordered experiment-set launch."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Sequence
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
_PARALLEL_READY_ITEMS = ("CLOUD-01", "OPS-04", "OPS-05", "OPS-06")


class WorkerProcessError(RuntimeError):
    """Raised when one isolated training worker does not complete."""


def run_launch(
    runner_path: str | Path,
    config_paths: Sequence[str | Path],
    *,
    repository: str | Path | None = None,
    retry_failed: bool | None = None,
    worker_slots: Sequence[WorkerSlot[str]] | None = None,
) -> list[Path]:
    """Run one plan-defined config list under a single launch lock.

    A case-runner invocation containing ``--retry-failed`` explicitly opts in
    to retrying coherent failed attempts. Direct callers may pass the keyword
    instead. Repeated ``--worker-slot <slot-id>=<cuda-device>`` arguments opt
    into bounded subprocess execution while retaining this process as the only
    coordinator and launch-lock owner.
    """

    root = repository_path(repository)
    runner = _resolve_runner(runner_path, root)
    retry_failed, resolved_slots = _resolve_launch_options(
        runner,
        requested_retry=retry_failed,
        requested_slots=worker_slots,
    )
    if resolved_slots:
        _require_parallel_launch_ready(root)
    if not config_paths:
        raise RunnerError(f"Runner has no configs: {runner}")

    try:
        configs = [
            resolve_launch_config(path, repository=root)[1]
            for path in config_paths
        ]
    except LaunchError as error:
        raise RunnerError(str(error)) from error
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
        command_parts.extend(("--worker-slot", f"{slot.slot_id}={slot.payload}"))
    command = shlex.join(command_parts)
    prior_completed = [
        _completed_attempt_for_config(path, config, allow_failed_retry=retry_failed)
        for path, config in loaded
    ]
    with direct_launch_guard(repository=root):
        # Close the preflight-to-lock race before creating any new attempt.
        prior_completed = [
            _completed_attempt_for_config(path, config, allow_failed_retry=retry_failed)
            for path, config in loaded
        ]
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
            launch_id = uuid4().hex

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
            slots.append(_parse_worker_slot(arguments[index + 1]))
            index += 2
            continue
        if argument.startswith("--worker-slot="):
            slots.append(_parse_worker_slot(argument.split("=", 1)[1]))
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


def _parse_worker_slot(value: str) -> WorkerSlot[str]:
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
        _parse_worker_slot(f"{slot.slot_id}={slot.payload}")
    devices = [str(slot.payload) for slot in slots]
    duplicate_devices = sorted(
        {device for device in devices if devices.count(device) > 1}
    )
    if duplicate_devices:
        raise RunnerError(
            "Scientific worker slots must map to distinct CUDA device ordinals; "
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


def _require_parallel_launch_ready(repository: Path) -> None:
    workboard = repository / "docs" / "experimental-design" / "workboard.md"
    try:
        text = workboard.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise RunnerError(f"Cannot read parallel-launch workboard: {workboard}") from error
    unresolved = [
        item
        for item in _PARALLEL_READY_ITEMS
        if re.search(
            rf"^\|\s*`{re.escape(item)}`\s*\|\s*resolved\s*\|",
            text,
            flags=re.MULTILINE,
        )
        is None
    ]
    if unresolved:
        raise RunnerError(
            "Concurrent scientific launch remains blocked until these workboard "
            f"items are resolved: {', '.join(unresolved)}."
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
) -> Path:
    """Run one config in a fresh process with one explicit visible GPU."""

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
            os.getpid(),
            sender,
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
        f"Isolated worker for {config_path.name} on {slot.slot_id} failed "
        f"(exit code {exit_code}): {detail}."
    )


def _receive_and_reap_worker(process: Any, receiver: Any) -> tuple[Any, int | None]:
    """Receive one bounded result and always drain a successfully started child."""

    receive_error: BaseException | None = None
    message: Any = None
    try:
        try:
            message = receiver.recv()
        except EOFError:
            message = None
        except BaseException as error:
            receive_error = error
    finally:
        try:
            receiver.close()
        finally:
            process.join()
    exit_code = process.exitcode
    process.close()
    if receive_error is not None:
        raise WorkerProcessError(
            "Failed while receiving an isolated worker result after draining the child."
        ) from receive_error
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
    coordinator_pid: int,
    sender: Any,
) -> None:
    """Child-process entrypoint; the parent remains the sole coordinator."""

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
        os.environ["PAPER_EXP_WORKER_GPU_UUID"] = gpu["uuid"]
        os.environ["PAPER_EXP_WORKER_GPU_NAME"] = gpu["name"]
        os.environ["PAPER_EXP_WORKER_GPU_TOTAL_MEMORY_BYTES"] = str(
            gpu["total_memory_bytes"]
        )
        os.environ["PAPER_EXP_WORKER_GPU_COMPUTE_CAPABILITY"] = gpu[
            "compute_capability"
        ]
        run_dir = _run_one(
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


def _require_isolated_cuda_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the child sees one BF16-capable CUDA device."""

    try:
        import torch
    except ImportError as error:
        raise WorkerProcessError(
            "Isolated GPU worker cannot import torch after device isolation."
        ) from error
    if config["training"]["device"] not in {"cuda", "cuda:0"}:
        raise WorkerProcessError(
            "Isolated GPU worker requires training.device cuda or cuda:0."
        )
    if config["training"]["precision"] != "bfloat16":
        raise WorkerProcessError(
            "Isolated GPU worker requires training.precision bfloat16."
        )
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
    if not gpu_uuid or not gpu_name or total_memory <= 0 or major < 0 or minor < 0:
        raise WorkerProcessError(
            "Assigned CUDA device did not expose complete UUID, name, memory, and "
            "compute-capability identity."
        )
    return {
        "uuid": gpu_uuid,
        "name": gpu_name,
        "total_memory_bytes": total_memory,
        "compute_capability": f"{major}.{minor}",
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
