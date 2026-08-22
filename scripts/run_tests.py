"""Run the test suite and leave no repository-local pytest scratch state."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".pytest_tmp_run"


def _clean_scratch() -> None:
    if SCRATCH.parent != ROOT or SCRATCH.name != ".pytest_tmp_run":
        raise RuntimeError(f"Refusing to clean unexpected path: {SCRATCH}")
    if SCRATCH.is_symlink() or SCRATCH.is_file():
        SCRATCH.unlink()
    elif SCRATCH.exists():
        shutil.rmtree(SCRATCH)


def main() -> int:
    _clean_scratch()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--basetemp",
        str(SCRATCH),
        *sys.argv[1:],
    ]
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    finally:
        _clean_scratch()


if __name__ == "__main__":
    raise SystemExit(main())
