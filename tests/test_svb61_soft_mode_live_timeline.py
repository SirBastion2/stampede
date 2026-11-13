"""SVB-61: CPU soft decide-once requests match the gpu-sim kernel; live runs the calibrated
model over SVB's timeline (ADR-106). CPU, offline. The two parity tests run 390 ticks each."""

from __future__ import annotations

import os

import numpy as np
import pytest

from contracts.events import WithdrawalRequested
from params.loader import get_parameter

FIT_SOFTNESS_ROW = "Calibration behavioral noise temperature"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def _fit_engine(softness: float, seed: int = 3060):
    from calibration.fit import _default_svb_scenario, real_calibration_modes
    from engine.agents.decide import apply_calibration_vector
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    engine = build_engine(modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(seed))
    apply_calibration_vector(engine["M06"], np.full(17, 0.5))
    engine["M06"].softness = softness
    return engine


def _requests(log) -> list[WithdrawalRequested]:
    return [e for e in log if isinstance(e, WithdrawalRequested)]


def test_fit_softness_row_is_positive() -> None:
    assert float(get_parameter(FIT_SOFTNESS_ROW).value) > 0.0


@pytest.mark.parametrize("seed", [3060, 11])
def test_cpu_soft_mode_matches_gpu_sim_on_e1_day_one(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    """Before SVB-61 the CPU engine requested 0 here (diag-08); now it matches gpu-sim."""
    from calibration.fit import _default_svb_scenario, measure_outflow

    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    softness = float(get_parameter(FIT_SOFTNESS_ROW).value)
    vector = np.full(17, 0.5)
    cpu_day1, _, _ = measure_outflow(_default_svb_scenario(seed), steps=390, param_vector=vector,
                                     requested=True, disable_caps=True, softness=softness)
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    gpu_day1, _, _ = measure_outflow(_default_svb_scenario(seed), steps=390, param_vector=vector,
                                     requested=True, disable_caps=True, softness=softness)
    assert cpu_day1 > 0.05
    assert cpu_day1 == pytest.approx(gpu_day1, rel=0.02)


def test_soft_mode_publishes_a_few_aggregated_requests_per_tick() -> None:
    from ui.live.session import step_engine

    engine = _fit_engine(0.06)
    m06 = engine["M06"]
    n_segments = len(set(m06.segments)) if len(m06.segments) else 64
    for t in range(300):
        step_engine(engine, float(t))
    requests = _requests(engine.bus.log)
    assert requests, "soft mode must publish requests"
    assert all(r.depositor_id.startswith("agg:") for r in requests)
    per_tick: dict[float, int] = {}
    for r in requests:
        per_tick[r.t] = per_tick.get(r.t, 0) + 1
    assert max(per_tick.values()) <= max(n_segments, 1) * 4


def test_hard_mode_still_publishes_per_depositor() -> None:
    from ui.live.session import step_engine

    engine = _fit_engine(0.0)
    for t in range(60):
        step_engine(engine, float(t))
    requests = _requests(engine.bus.log)
    assert all(not r.depositor_id.startswith("agg:") for r in requests)


def test_live_real_mode_runs_the_fit_softness(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    assert session.engine["M06"].softness == pytest.approx(float(get_parameter(FIT_SOFTNESS_ROW).value))


def test_calibration_engine_keeps_its_own_softness() -> None:
    from calibration.fit import _default_svb_scenario, real_calibration_modes
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    engine = build_engine(modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(11))
    assert engine["M06"].softness == 0.0


def test_live_window_follows_svb_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.pacing import crisis_sim_minutes

    assert crisis_sim_minutes() == 120.0
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    assert crisis_sim_minutes() == 1441.0  # P-A: the closure at minute 1440 is the 1441st step
    row = get_parameter("M29 live real crisis sim minutes")
    assert row.status == "ASSUMED" and "Barr" in row.source
