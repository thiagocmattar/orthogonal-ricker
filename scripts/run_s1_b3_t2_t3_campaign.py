"""Run the gated S1-B3 T2 -> diagnostic 265 -> T3 campaign.

This is one fail-closed outer controller around two immutable pretraining
queues.  T3 is deliberately registered only after T2 and pooled diagnostic
265 have passed their recorded gates.  The controller never retries a failed
training or diagnostic attempt.

Use ``--preflight-only`` before launching the detached controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
T2_RUNNER = Path(r"C:\tmp\osp-s1-b3-t2-runner")
T3_RUNNER = Path(r"C:\tmp\osp-s1-b3-t3-runner")
T2_LAUNCH_COMMIT = "23247f860d474718bf888a2a11ad2f9132059912"

CONTROLLER_STATE = ROOT / "run-logs/s1-b3-t2-t3-sequential-controller.json"
T2_STATE = ROOT / "run-logs/s1-b3-t2-l1-flanks-257-264-queue.json"
T2_LOGS = ROOT / "run-logs/s1-b3-t2-l1-flanks-257-264"
T3_STATE = ROOT / "run-logs/s1-b3-t3-rk-weight-266-273-queue.json"
T3_LOGS = ROOT / "run-logs/s1-b3-t3-rk-weight-266-273"

DRAFTS = ROOT / "tmp/s1-b3-config-drafts"
CONFIG_REGISTRY = ROOT / "docs/experimental-design/config-registry.yaml"
RUN_REGISTRY = ROOT / "docs/experimental-design/run-registry.yaml"

T2_IDS = (
    "257-s1-b3-p14m-a3-l1n-w0p15-s0",
    "258-s1-b3-p14m-a3-ol1-w0p15-s0",
    "259-s1-b3-p14m-a6post-l1n-w0p15-s0",
    "260-s1-b3-p14m-a6post-ol1-w0p15-s0",
    "261-s1-b3-p14m-a3-l1n-w5-s0",
    "262-s1-b3-p14m-a3-ol1-w5-s0",
    "263-s1-b3-p14m-a6post-l1n-w5-s0",
    "264-s1-b3-p14m-a6post-ol1-w5-s0",
)
T3_IDS = (
    "266-s1-b3-p14m-a3-rn-w0p1-c0p1-sg0p1-s0",
    "267-s1-b3-p14m-a3-or-w0p1-c0p1-sg0p1-s0",
    "268-s1-b3-p14m-a6post-rn-w0p1-c0p1-sg0p1-s0",
    "269-s1-b3-p14m-a6post-or-w0p1-c0p1-sg0p1-s0",
    "270-s1-b3-p14m-a3-rn-w1-c0p1-sg0p1-s0",
    "271-s1-b3-p14m-a3-or-w1-c0p1-sg0p1-s0",
    "272-s1-b3-p14m-a6post-rn-w1-c0p1-sg0p1-s0",
    "273-s1-b3-p14m-a6post-or-w1-c0p1-sg0p1-s0",
)

HARD_SCREEN_FLAGS = (
    "nonfinite",
    "orthogonal_step_budget_violation",
)


class CampaignError(RuntimeError):
    """A fail-closed campaign gate did not pass."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(
    args: Sequence[str | Path],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    expected: Iterable[int] = (0,),
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [str(arg) for arg in args]
    print(f"[{now_iso()}] RUN cwd={cwd}: {subprocess.list2cmdline(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if capture:
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if completed.returncode not in set(expected):
        raise CampaignError(
            f"Command returned {completed.returncode}, expected {sorted(set(expected))}: "
            f"{subprocess.list2cmdline(command)}"
        )
    return completed


def git(*args: str, cwd: Path = ROOT, capture: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    if cwd != ROOT:
        command.extend(["-c", f"safe.directory={cwd.as_posix()}"])
    command.extend(["-C", str(cwd), *args])
    return run(command, cwd=ROOT, capture=capture)


def git_text(*args: str, cwd: Path = ROOT) -> str:
    return git(*args, cwd=cwd).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def require_clean(cwd: Path = ROOT) -> None:
    status = git_text("status", "--short", cwd=cwd)
    require(not status, f"Worktree is not clean: {cwd}\n{status}")


def result_root(config_id: str) -> Path:
    return ROOT / "results" / config_id


def config_path(config_id: str, base: Path = ROOT) -> Path:
    return base / "configs" / f"{config_id}.yaml"


def ensure_absent(path: Path, label: str) -> None:
    require(not path.exists(), f"{label} already exists: {path}")


def ensure_empty_directory(path: Path, label: str) -> None:
    require(path.is_dir(), f"{label} is missing: {path}")
    require(not any(path.iterdir()), f"{label} is not empty: {path}")


def verify_t2_runner() -> None:
    require(T2_RUNNER.is_dir(), f"Prepared T2 runner is missing: {T2_RUNNER}")
    require(git_text("rev-parse", "HEAD", cwd=T2_RUNNER) == T2_LAUNCH_COMMIT, "T2 runner commit changed")
    require_clean(T2_RUNNER)
    token_link = T2_RUNNER / "data/tokenized"
    require(token_link.exists(), "T2 token-cache junction is missing")
    require(os.path.samefile(token_link, ROOT / "data/tokenized"), "T2 token-cache junction target changed")
    for config_id in T2_IDS:
        main_config = config_path(config_id)
        runner_config = config_path(config_id, T2_RUNNER)
        require(main_config.is_file() and runner_config.is_file(), f"Missing T2 config: {config_id}")
        require(sha256(main_config) == sha256(runner_config), f"T2 config differs in runner: {config_id}")
        main_result = result_root(config_id)
        runner_result = T2_RUNNER / "results" / config_id
        ensure_empty_directory(main_result, f"T2 result root {config_id}")
        require(runner_result.exists(), f"T2 result junction is missing: {config_id}")
        require(os.path.samefile(main_result, runner_result), f"T2 result junction target changed: {config_id}")


def verify_initial_preflight() -> dict[str, Any]:
    require_clean(ROOT)
    script_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    git("ls-files", "--error-unmatch", script_rel)
    require(not git_text("diff", "HEAD", "--", script_rel), "Controller differs from committed bytes")
    require(
        run(["git", "merge-base", "--is-ancestor", T2_LAUNCH_COMMIT, "HEAD"], capture=True).returncode == 0,
        "Current main history does not contain the reviewed T2 launch commit",
    )
    ensure_absent(CONTROLLER_STATE, "Controller state")
    ensure_absent(T2_STATE, "T2 queue state")
    ensure_absent(T2_LOGS, "T2 child-log directory")
    ensure_absent(T3_STATE, "T3 queue state")
    ensure_absent(T3_LOGS, "T3 child-log directory")
    ensure_absent(T3_RUNNER, "T3 runner")
    ensure_absent(config_path("265-s1-b3-t2-l1-flanks-selection-propagation"), "Diagnostic 265 config")
    ensure_absent(config_path(T3_IDS[0]), "First T3 config")
    ensure_absent(config_path("274-s1-b3-t3-rk-weight-selection-propagation"), "Diagnostic 274 config")
    verify_t2_runner()
    scratch = [path.name for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("pytest_tmp")]
    require(not scratch, f"Pytest scratch directories remain in the worktree: {scratch}")
    return {
        "main_commit": git_text("rev-parse", "HEAD"),
        "t2_launch_commit": T2_LAUNCH_COMMIT,
        "controller_sha256": sha256(Path(__file__).resolve()),
        "python": str(PYTHON),
        "t2_configs": list(T2_IDS),
        "t3_configs": list(T3_IDS),
        "estimated_hours": 10.9167,
    }


class State:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": "s1-b3-t2-t3-sequential",
            "status": "running",
            "phase": "preflight",
            "controller_pid": os.getpid(),
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "history": [],
            **initial,
        }
        self.write()

    def write(self) -> None:
        CONTROLLER_STATE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CONTROLLER_STATE.with_name(f".{CONTROLLER_STATE.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, CONTROLLER_STATE)

    def phase(self, name: str, **details: Any) -> None:
        timestamp = now_iso()
        self.payload["phase"] = name
        self.payload["updated_at"] = timestamp
        self.payload["history"].append({"phase": name, "started_at": timestamp, **details})
        self.write()
        print(f"[{timestamp}] PHASE {name}", flush=True)

    def record(self, **details: Any) -> None:
        self.payload.update(details)
        self.payload["updated_at"] = now_iso()
        self.write()

    def complete(self) -> None:
        self.payload.update(status="completed", phase="completed", finished_at=now_iso(), updated_at=now_iso())
        self.write()

    def fail(self, error: BaseException) -> None:
        self.payload.update(
            status="failed",
            phase="failed",
            finished_at=now_iso(),
            updated_at=now_iso(),
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self.write()


def verify_hash_manifest(bundle: Path, manifest_name: str = "bundle-output-sha256.yaml") -> None:
    manifest_path = bundle / manifest_name
    require(manifest_path.is_file(), f"Bundle hash manifest is missing: {manifest_path}")
    if manifest_path.suffix == ".json":
        payload = load_json(manifest_path)
        files = payload.get("output_sha256")
    else:
        payload = load_yaml(manifest_path)
        files = payload.get("files")
    require(isinstance(files, dict) and files, f"Invalid bundle hash map: {manifest_path}")
    for relative, expected_hash in files.items():
        path = bundle / relative
        require(path.is_file(), f"Bundle file is missing: {path}")
        require(sha256(path) == expected_hash, f"Bundle hash mismatch: {path}")


def atomic_install(source: Path, destination: Path, *, must_be_new: bool = False) -> None:
    require(source.is_file(), f"Candidate is missing: {source}")
    if must_be_new:
        require(not destination.exists(), f"Destination unexpectedly exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.campaign-tmp")
    require(not temporary.exists(), f"Temporary install path already exists: {temporary}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        require(sha256(source) == sha256(temporary), f"Candidate copy hash mismatch: {source}")
        os.replace(temporary, destination)
        require(sha256(source) == sha256(destination), f"Installed candidate hash mismatch: {destination}")
    finally:
        if temporary.exists():
            temporary.unlink()


def porcelain_paths() -> set[str]:
    paths: set[str] = set()
    # Preserve the leading two-character status field. git_text() strips the
    # first line and would turn an unstaged " M path" entry into "M path".
    for line in git("status", "--porcelain=v1").stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path.replace("\\", "/"))
    return paths


def validate_clean_tooling() -> None:
    require_clean(ROOT)
    run([PYTHON, DRAFTS / "audit_b3_drafts.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "tmp/s1-b3-config-drafts/test_reconcile_b3_tranche.py",
            "tmp/s1-b3-config-drafts/test_reconcile_s1_pooled_diagnostic.py",
        ]
    )
    run([PYTHON, "-m", "paper_exp.cli", "check"])
    git("diff", "--check")


def validate_candidate() -> None:
    # The live dry-run and emitted bundle are the registration/reconciliation
    # gates. Run state-coupled unit suites while the worktree is still clean;
    # after candidate installation, validate only the resulting repository.
    run([PYTHON, DRAFTS / "audit_b3_drafts.py"])
    run([PYTHON, "-m", "paper_exp.cli", "check"])
    git("diff", "--check")


def commit_exact(paths: Sequence[Path], message: str) -> str:
    relative = [path.resolve().relative_to(ROOT).as_posix() for path in paths]
    require(porcelain_paths() == set(relative), f"Unexpected tracked changes before commit: {sorted(porcelain_paths())}")
    validate_candidate()
    git("add", "--", *relative)
    staged = set(git_text("diff", "--cached", "--name-only").splitlines())
    require(staged == set(relative), f"Unexpected staged files: {sorted(staged)}")
    git("commit", "--no-gpg-sign", "-m", message)
    require_clean(ROOT)
    return git_text("rev-parse", "HEAD")


def audit_s1(
    expected_completed: int,
    expected_diagnostic: tuple[str, str] | None = None,
    expected_pending_ids: Sequence[str] = (),
) -> dict[str, Any]:
    completed = run(
        [PYTHON, ROOT / "tmp/s1-final-audit/audit_s1.py", "--json"],
        expected=(1, 2),
        capture=True,
    )
    payload = json.loads(completed.stdout)
    errors = payload.get("errors")
    if expected_pending_ids:
        expected_details = {
            f"{config_id}: canonical run has no propagation_result_path" for config_id in expected_pending_ids
        }
        observed_details = {
            error.get("detail")
            for error in (errors if isinstance(errors, list) else [])
            if isinstance(error, dict) and error.get("code") == "science.evidence_invalid"
        }
        require(
            isinstance(errors, list)
            and len(errors) == len(expected_pending_ids)
            and observed_details == expected_details,
            f"S1 audit errors differ from the exact pending-endpoint envelope: {errors}",
        )
    else:
        require(errors == [], f"S1 audit has structural errors: {errors}")
    require(payload.get("completed_scientific_cells") == expected_completed, "Unexpected S1 completed-cell count")
    integrity = payload.get("evidence", {}).get("integrity", {})
    require(integrity.get("returncode") == 0, f"S1 integrity failed: {integrity}")
    if expected_diagnostic:
        prefix, status = expected_diagnostic
        observed = payload.get("evidence", {}).get("required_diagnostics", {}).get(prefix)
        require(observed == status, f"Diagnostic {prefix} status is {observed}, expected {status}")
    return payload


def queue_command(config_ids: Sequence[str], state_path: Path, logs_dir: Path) -> list[str | Path]:
    command: list[str | Path] = [PYTHON, "-m", "paper_exp.cli", "run-pretrain-queue"]
    for config_id in config_ids:
        command.extend(["--config", f"configs/{config_id}.yaml"])
    command.extend(["--state-path", state_path, "--logs-dir", logs_dir])
    return command


def verify_completed_queue(state_path: Path, config_ids: Sequence[str]) -> dict[str, Any]:
    payload = load_json(state_path)
    require(payload.get("schema_version") == 1 and payload.get("status") == "completed", "Queue did not complete")
    items = payload.get("items")
    require(isinstance(items, list) and len(items) == len(config_ids), "Queue item count mismatch")
    observed = [Path(item["config_path"]).stem for item in items]
    require(observed == list(config_ids), f"Queue order mismatch: {observed}")
    for item in items:
        require(item.get("status") == "completed" and item.get("returncode") == 0, f"Queue child failed: {item}")
    return payload


def run_queue(runner: Path, config_ids: Sequence[str], state_path: Path, logs_dir: Path) -> dict[str, Any]:
    ensure_absent(state_path, "Queue state")
    ensure_absent(logs_dir, "Queue child-log directory")
    require_clean(runner)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runner / "src")
    run(queue_command(config_ids, state_path, logs_dir), cwd=runner, env=environment)
    return verify_completed_queue(state_path, config_ids)


def reject_hard_screen_flags(summary: dict[str, Any]) -> None:
    flags = summary.get("screening_flags")
    require(isinstance(flags, dict) and flags, "Reconciliation bundle has no screening flags")
    failures: list[str] = []
    for config_id, values in flags.items():
        for flag in HARD_SCREEN_FLAGS:
            if values.get(flag) is True:
                failures.append(f"{config_id}:{flag}")
    require(not failures, f"Scientific hard-screen flags fired: {failures}")


def reconcile_tranche(
    *,
    tranche: str,
    queue_state: Path,
    diagnostic_id: str,
    expected_completed: int,
    commit_message: str,
) -> str:
    validate_clean_tooling()
    output = DRAFTS / "reconciliation-review" / tranche
    ensure_absent(output, f"{tranche} reconciliation bundle")
    base = [PYTHON, DRAFTS / "reconcile_b3_tranche.py", "--tranche", tranche, "--queue-state", queue_state]
    run(base)
    run([*base, "--emit", "--reviewed-at", now_iso()])
    verify_hash_manifest(output, "bundle-manifest.json")
    manifest = load_json(output / "bundle-manifest.json")
    summary = load_yaml(output / "review-summary.yaml")
    require(manifest.get("status") == "review_only_not_applied" and manifest.get("tranche") == tranche, "Wrong tranche bundle")
    require(summary.get("hard_artifact_checks") == "passed", "Tranche artifact checks did not pass")
    reject_hard_screen_flags(summary)
    installed = [
        (output / "config-registry.updated.yaml", CONFIG_REGISTRY, False),
        (output / "run-registry.updated.yaml", RUN_REGISTRY, False),
        (output / f"{diagnostic_id}.yaml", config_path(diagnostic_id), True),
    ]
    require_clean(ROOT)
    for source, destination, is_new in installed:
        atomic_install(source, destination, must_be_new=is_new)
    commit = commit_exact([destination for _, destination, _ in installed], commit_message)
    pending_ids = T2_IDS if tranche == "t2-l1-flanks" else T3_IDS
    audit_s1(expected_completed, (diagnostic_id.split("-", 1)[0], "ready"), pending_ids)
    return commit


def run_diagnostic(diagnostic_id: str) -> None:
    root = result_root(diagnostic_id)
    ensure_absent(root, f"Diagnostic result root {diagnostic_id}")
    require_clean(ROOT)
    run([PYTHON, "-m", "paper_exp.cli", "activation-propagation", "--config", config_path(diagnostic_id)])
    attempts = [path for path in root.iterdir() if path.is_dir()]
    require(len(attempts) == 1, f"Expected one diagnostic attempt, found {len(attempts)}")
    for name in ("config.yaml", "manifest.json", "metrics.json", "predictions.jsonl", "activation_propagation.json"):
        require((attempts[0] / name).is_file(), f"Diagnostic artifact is missing: {attempts[0] / name}")


def close_diagnostic(prefix: int, expected_completed: int, commit_message: str) -> tuple[str, Path]:
    validate_clean_tooling()
    output = DRAFTS / "diagnostic-reconciliation-review" / f"diagnostic-{prefix}"
    ensure_absent(output, f"Diagnostic {prefix} closure bundle")
    base = [PYTHON, DRAFTS / "reconcile_s1_pooled_diagnostic.py", "--diagnostic", str(prefix)]
    run(base)
    run([*base, "--emit", "--reviewed-at", now_iso()])
    verify_hash_manifest(output)
    summary = load_yaml(output / "review-summary.yaml")
    require(summary.get("status") == "review_only_not_applied", f"Diagnostic {prefix} closure status is invalid")
    require(summary.get("source_count") == 8 and summary.get("validation_tokens") == 311_296, "Diagnostic coverage mismatch")
    require(summary.get("completed_scientific_cells_after_apply") == expected_completed, "Diagnostic completion count mismatch")
    installed = [
        (output / "config-registry.updated.yaml", CONFIG_REGISTRY),
        (output / "run-registry.updated.yaml", RUN_REGISTRY),
    ]
    require_clean(ROOT)
    for source, destination in installed:
        atomic_install(source, destination)
    commit = commit_exact([destination for _, destination in installed], commit_message)
    audit_s1(expected_completed, (str(prefix), "closed_valid"))
    return commit, output / "results-handoff.md"


def register_t3() -> str:
    validate_clean_tooling()
    tranche = "t3-rk-weight"
    output = DRAFTS / "registration-review" / tranche
    ensure_absent(output, "T3 registration bundle")
    base = [PYTHON, DRAFTS / "prepare_b3_registration.py", "--tranche", tranche]
    run(base)
    run([*base, "--emit"])
    verify_hash_manifest(output)
    summary = load_yaml(output / "review-summary.yaml")
    require(summary.get("status") == "review_only_not_applied" and summary.get("tranche") == tranche, "Wrong T3 registration bundle")
    require(summary.get("scientific_config_ids") == list(T3_IDS), "T3 registration order mismatch")
    installed: list[tuple[Path, Path, bool]] = [
        (output / "config-registry.updated.yaml", CONFIG_REGISTRY, False)
    ]
    installed.extend((output / "configs" / f"{config_id}.yaml", config_path(config_id), True) for config_id in T3_IDS)
    require_clean(ROOT)
    for source, destination, is_new in installed:
        atomic_install(source, destination, must_be_new=is_new)
    commit = commit_exact([destination for _, destination, _ in installed], "Register S1-B3 T3 Ricker weights")
    audit_s1(98, ("265", "closed_valid"))
    ensure_absent(config_path("274-s1-b3-t3-rk-weight-selection-propagation"), "Diagnostic 274 config")
    return commit


def ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def make_junction(link: Path, target: Path) -> None:
    command = f"New-Item -ItemType Junction -Path {ps_quote(link)} -Target {ps_quote(target)} | Out-Null"
    run(["powershell.exe", "-NoProfile", "-Command", command])
    require(link.exists() and os.path.samefile(link, target), f"Junction verification failed: {link}")


def create_t3_runner(launch_commit: str) -> None:
    ensure_absent(T3_RUNNER, "T3 runner")
    require_clean(ROOT)
    run(["git", "-c", "core.autocrlf=false", "worktree", "add", "--detach", T3_RUNNER, launch_commit])
    require(git_text("rev-parse", "HEAD", cwd=T3_RUNNER) == launch_commit, "T3 runner commit mismatch")
    token_link = T3_RUNNER / "data/tokenized"
    ensure_absent(token_link, "T3 token-cache junction")
    make_junction(token_link, ROOT / "data/tokenized")
    for config_id in T3_IDS:
        main_result = result_root(config_id)
        runner_result = T3_RUNNER / "results" / config_id
        ensure_absent(main_result, f"T3 result root {config_id}")
        ensure_absent(runner_result, f"T3 result junction {config_id}")
        main_result.mkdir(parents=True)
        make_junction(runner_result, main_result)
        require(sha256(config_path(config_id)) == sha256(config_path(config_id, T3_RUNNER)), f"T3 runner config differs: {config_id}")
    require_clean(T3_RUNNER)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(T3_RUNNER / "src")
    probe = run(
        [
            PYTHON,
            "-c",
            "import pathlib,paper_exp,sys; p=pathlib.Path(paper_exp.__file__).resolve(); "
            "r=pathlib.Path(sys.argv[1]).resolve(); assert p.is_relative_to(r/'src'), (p,r)",
            T3_RUNNER,
        ],
        cwd=T3_RUNNER,
        env=environment,
        capture=True,
    )
    require(probe.returncode == 0, "T3 runner import probe failed")


def campaign() -> None:
    initial = verify_initial_preflight()
    state = State(initial)
    try:
        state.phase("t2_queue", config_count=len(T2_IDS), queue_state=str(T2_STATE))
        t2_queue = run_queue(T2_RUNNER, T2_IDS, T2_STATE, T2_LOGS)
        state.record(t2_queue_id=t2_queue.get("queue_id"), t2_queue_finished_at=t2_queue.get("finished_at"))

        state.phase("t2_reconciliation")
        t2_reconcile_commit = reconcile_tranche(
            tranche="t2-l1-flanks",
            queue_state=T2_STATE,
            diagnostic_id="265-s1-b3-t2-l1-flanks-selection-propagation",
            expected_completed=90,
            commit_message="Reconcile S1-B3 T2 L1 flanks",
        )
        state.record(t2_reconciliation_commit=t2_reconcile_commit)

        state.phase("diagnostic_265")
        run_diagnostic("265-s1-b3-t2-l1-flanks-selection-propagation")
        diagnostic_265_commit, handoff_265 = close_diagnostic(265, 98, "Close S1-B3 T2 pooled diagnostic")
        state.record(diagnostic_265_commit=diagnostic_265_commit, results_handoff_265=str(handoff_265))

        state.phase("t3_registration")
        t3_registration_commit = register_t3()
        state.record(t3_registration_commit=t3_registration_commit)

        state.phase("t3_runner_setup")
        create_t3_runner(t3_registration_commit)

        state.phase("t3_queue", config_count=len(T3_IDS), queue_state=str(T3_STATE))
        t3_queue = run_queue(T3_RUNNER, T3_IDS, T3_STATE, T3_LOGS)
        state.record(t3_queue_id=t3_queue.get("queue_id"), t3_queue_finished_at=t3_queue.get("finished_at"))

        state.phase("t3_reconciliation")
        t3_reconcile_commit = reconcile_tranche(
            tranche="t3-rk-weight",
            queue_state=T3_STATE,
            diagnostic_id="274-s1-b3-t3-rk-weight-selection-propagation",
            expected_completed=98,
            commit_message="Reconcile S1-B3 T3 Ricker weights",
        )
        state.record(t3_reconciliation_commit=t3_reconcile_commit)

        state.phase("diagnostic_274")
        run_diagnostic("274-s1-b3-t3-rk-weight-selection-propagation")
        diagnostic_274_commit, handoff_274 = close_diagnostic(274, 106, "Close S1-B3 T3 pooled diagnostic")
        state.record(diagnostic_274_commit=diagnostic_274_commit, results_handoff_274=str(handoff_274))

        state.complete()
        print(f"[{now_iso()}] Campaign completed.", flush=True)
    except BaseException as error:
        state.fail(error)
        traceback.print_exc()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="Validate the initial launch envelope without writing state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight_only:
        print(json.dumps(verify_initial_preflight(), indent=2, sort_keys=True))
        return 0
    campaign()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
