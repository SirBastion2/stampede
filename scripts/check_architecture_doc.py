#!/usr/bin/env python3
"""CI gate §10.3 #9: engine/ or contracts/ changes must update docs/ARCHITECTURE.md.

The repository-root ARCHITECTURE.md file is only a pointer at docs/ARCHITECTURE.md.
Editing that pointer does not satisfy the gate (ADR-013).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Satisfies §10.3 gate 9. The root pointer ARCHITECTURE.md does not.
DOC_REPO_PATH = "docs/ARCHITECTURE.md"
DOC = ROOT / DOC_REPO_PATH
WATCH_PREFIXES = ("engine/", "contracts/")


def changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    files = changed_files(base)
    touched_watch = [f for f in files if f.startswith(WATCH_PREFIXES)]
    if not touched_watch:
        return 0
    if not DOC.is_file():
        print(
            f"Architecture doc check failed: {DOC_REPO_PATH} is missing.",
            file=sys.stderr,
        )
        return 1
    if DOC_REPO_PATH in files:
        return 0
    print(
        "Architecture doc check failed: changes under engine/ or contracts/ "
        "require an update to docs/ARCHITECTURE.md. "
        "The root ARCHITECTURE.md pointer does not satisfy this gate.",
        file=sys.stderr,
    )
    for f in touched_watch:
        print(f"  - {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
