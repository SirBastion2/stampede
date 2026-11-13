#!/usr/bin/env python3
"""Fast determinism smoke: fake-engine stepping is identical across two runs."""

from __future__ import annotations

import sys

from engine import ModuleMode, build_engine


def _trace() -> tuple[tuple[str, float, tuple[object, ...]], ...]:
    modules = build_engine(default_mode=ModuleMode.FAKE)
    rows: list[tuple[str, float, tuple[object, ...]]] = []
    for step in range(3):
        t = float(step)
        for module_id in sorted(modules):
            out = modules[module_id].step(t)
            rows.append((module_id, t, tuple(out)))
    return tuple(rows)


def main() -> int:
    first, second = _trace(), _trace()
    if first != second:
        print("determinism smoke FAILED: two back-to-back runs differ", file=sys.stderr)
        return 1
    print("determinism smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
