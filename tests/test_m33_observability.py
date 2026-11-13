"""M33 observability: Philox, event log, offline cache, real module parity."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from contracts.schemas import Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.l8_calibration import ObservabilityFake
from ops.event_log import event_log_hash, read_event_log_parquet, write_event_log_parquet
from ops.llm_cache import LlmCache
from ops.observability import (
    ObservabilityConfig,
    ObservabilityModule,
    load_observability_config,
    offline_ready,
)
from ops.rng import philox_uniform, philox_uniform_cpu


def test_philox_cpu_is_deterministic() -> None:
    first = philox_uniform_cpu(99, 8)
    second = philox_uniform_cpu(99, 8)
    assert first == second
    assert len(first) == 8
    assert all(0.0 <= value < 1.0 for value in first)


def test_philox_cpu_and_prefer_gpu_match_without_cupy() -> None:
    cpu = philox_uniform_cpu(7, 4, offset=2)
    blended = philox_uniform(7, 4, offset=2, prefer_gpu=True)
    assert cpu == blended


def test_event_log_hash_matches_fake() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=3)
    log = run_scenario(scenario, steps=2)
    fake = ObservabilityFake("M33")
    fake.bus = type("B", (), {"log": log})()  # minimal bus stand-in
    fake.on_tick(0.0, [])
    assert fake.log_hash == event_log_hash(log)


def test_parquet_round_trip() -> None:
    scenario = Scenario(preset="insured-community", seed=1)
    events = run_scenario(scenario, steps=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.parquet"
        write_event_log_parquet(events, path)
        rows = read_event_log_parquet(path)
        assert len(rows) == len(events)
        assert json.loads(rows[0]["payload_json"])["t"] == events[0].t


def test_offline_ready_with_demo_seed() -> None:
    cfg = load_observability_config()
    with tempfile.TemporaryDirectory() as tmp:
        cache = LlmCache(Path(tmp) / "cache.sqlite")
        cache.put(cfg.demo_prompt_hash, cfg.demo_model, cfg.demo_response)
        assert offline_ready(cache, config=cfg)


def test_real_module_log_hash_matches_fake_engine() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=11)
    engine = build_engine(modes={"M33": ModuleMode.REAL}, scenario=scenario)
    real = engine["M33"]
    assert isinstance(real, ObservabilityModule)
    engine.run(3)
    fake = ObservabilityFake("M33")
    fake.bus = engine.bus
    fake.on_tick(0.0, [])
    assert real.log_hash == fake.log_hash
    assert real.event_count == fake.event_count
    assert real.offline is True
    assert real.metrics is not None
    assert real.metrics.philox_first >= 0.0


def test_offline_smoke_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/offline_smoke.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_custom_config_demo_mode() -> None:
    cfg = ObservabilityConfig(
        demo_mode_default=False,
        offline_default=True,
        philox_samples_per_tick=2,
        cache_filename="x.sqlite",
        demo_prompt_hash="h",
        demo_model="m",
        demo_response="r",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cache = LlmCache(Path(tmp) / "x.sqlite")
        assert not offline_ready(cache, config=cfg)
        cache.put("h", "m", "r")
        assert offline_ready(cache, config=cfg)
