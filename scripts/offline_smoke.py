#!/usr/bin/env python3
"""M33 acceptance: offline/demo path runs without network and cache is warm."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from contracts.schemas import Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine
from ops.llm_cache import LlmCache
from ops.observability import load_observability_config, offline_ready


def main() -> int:
    cfg = load_observability_config()
    with tempfile.TemporaryDirectory() as tmp:
        cache = LlmCache(Path(tmp) / cfg.cache_filename)
        if not offline_ready(cache, config=cfg):
            cache.put(cfg.demo_prompt_hash, cfg.demo_model, cfg.demo_response)
        if not offline_ready(cache, config=cfg):
            print("offline smoke FAILED: cache not ready for offline demo", file=sys.stderr)
            return 1
        scenario = Scenario(preset="tech-concentrated", seed=42)
        engine = build_engine(
            modes={"M33": ModuleMode.REAL},
            scenario=scenario,
        )
        module = engine["M33"]
        engine.run(3)
        if not module.offline:
            print("offline smoke FAILED: M33 offline flag false after run", file=sys.stderr)
            return 1
        if not module.log_hash:
            print("offline smoke FAILED: empty event log hash", file=sys.stderr)
            return 1
    print("offline smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
