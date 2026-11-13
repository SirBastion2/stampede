"""SVB-65: the CPU engine runs the fitted feedback the GPU calibration kernels run (ADR-110)."""

from __future__ import annotations

import math

import numpy as np

from calibration.fit import (
    _default_svb_scenario,
    _disable_caps_and_closure,
    real_calibration_modes,
    resolve_tick_schedule,
)
from contracts.events import AttentionTick, PriceTick, WithdrawalRequested
from engine.agents.decide import apply_calibration_vector
from engine.agents.population import clear_sample_population_cache
from engine.fitted_feedback import fitted_feedback_enabled
from engine.gpu_full_io import apply_lane_feedback
from engine.gpu_sim import clear_static_pack_cache, simulate_logs
from engine.modes import ModuleMode
from engine.registry import build_engine
from params.loader import get_parameter

_SEED = 3060
_STEPS = 120
_VECTOR = np.full(17, 0.5)


def _calibration_env(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "400")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    for name in ("STAMPEDE_BACKEND", "STAMPEDE_CALIB_GPU_SIM", "STAMPEDE_CALIB_GPU_FULL"):
        monkeypatch.delenv(name, raising=False)


def _cpu_engine(*, gain: float | None = None, fitted: bool | None = None):
    clear_sample_population_cache()
    engine = build_engine(
        modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(_SEED)
    )
    module = engine["M06"]
    apply_calibration_vector(module, _VECTOR, clamp_inputs=False)
    apply_lane_feedback(module, _VECTOR)
    module.softness = 0.06
    if gain is not None:
        module.feedback_gain = gain
    if fitted is not None:
        engine.fitted_feedback = fitted
    _disable_caps_and_closure(engine)
    return engine


def _ticks_by_cause(log, t: float) -> tuple[list[float], list[float]]:
    raw = [e.level for e in log if isinstance(e, AttentionTick) and e.t == t and e.cause_id != "svb06-news"]
    composed = [e.level for e in log if isinstance(e, AttentionTick) and e.t == t and e.cause_id == "svb06-news"]
    return raw, composed


def _per_tick(log) -> tuple[dict[float, float], dict[float, float]]:
    requested: dict[float, float] = {}
    prices: dict[float, float] = {}
    for event in log:
        if isinstance(event, WithdrawalRequested):
            requested[float(event.t)] = requested.get(float(event.t), 0.0) + float(event.amount)
        elif isinstance(event, PriceTick):
            prices[float(event.t)] = float(event.price)
    return requested, prices


def test_cpu_calibration_engine_publishes_the_composed_attention(monkeypatch) -> None:
    _calibration_env(monkeypatch)
    engine = _cpu_engine()
    assert fitted_feedback_enabled(engine)
    log = engine.run(_STEPS)
    for t in range(_STEPS):
        raw, composed = _ticks_by_cause(log, float(t))
        assert len(raw) == 1 and len(composed) == 1
    # No requests before t=0, so news is 0 and the composed level is tanh of M03's.
    raw0, composed0 = _ticks_by_cause(log, 0.0)
    assert math.isclose(composed0[0], math.tanh(raw0[0]), rel_tol=0.0, abs_tol=1e-12)
    # The composed tick comes right after M03's own tick at the same t.
    kinds = [e.cause_id == "svb06-news" for e in log if isinstance(e, AttentionTick) and e.t == 5.0]
    assert kinds == [False, True]


def test_news_follows_requests_and_gain(monkeypatch) -> None:
    _calibration_env(monkeypatch)
    engine = _cpu_engine()
    log = engine.run(_STEPS)
    requested = sum(float(e.amount) for e in log if isinstance(e, WithdrawalRequested))
    assert requested > 0.0
    assert float(getattr(engine["M03"], "_gpu_news", 0.0)) > 0.0
    quiet = _cpu_engine(gain=0.0)
    quiet_log = quiet.run(_STEPS)
    assert float(getattr(quiet["M03"], "_gpu_news", 0.0)) == 0.0
    for t in range(_STEPS):
        raw, composed = _ticks_by_cause(quiet_log, float(t))
        assert math.isclose(composed[0], math.tanh(raw[0]), rel_tol=0.0, abs_tol=1e-12)


def test_engine_without_fitted_feedback_keeps_the_old_path(monkeypatch) -> None:
    _calibration_env(monkeypatch)
    engine = _cpu_engine(fitted=False)
    log = engine.run(30)
    assert not any(isinstance(e, AttentionTick) and e.cause_id == "svb06-news" for e in log)
    monkeypatch.delenv("STAMPEDE_CALIBRATION_ENGINE")
    assert fitted_feedback_enabled(build_engine(scenario=_default_svb_scenario(_SEED))) is False


def test_cpu_gpu_sim_and_gpu_full_run_one_model(monkeypatch) -> None:
    _calibration_env(monkeypatch)
    logs = {"cpu": _cpu_engine().run(_STEPS)}
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    for name, full in (("gpu-sim", False), ("gpu-full", True)):
        if full:
            monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
        clear_sample_population_cache()
        clear_static_pack_cache()
        logs[name] = simulate_logs(
            _default_svb_scenario(_SEED),
            _STEPS,
            [_VECTOR],
            array_module="numpy",
            disable_caps=True,
            softness=0.06,
            clamp_inputs=False,
            schedule=resolve_tick_schedule(_STEPS, None),
        )[0]
    reference_req, reference_px = _per_tick(logs["gpu-full"])
    total = sum(reference_req.values())
    assert total > 0.0
    for name in ("cpu", "gpu-sim"):
        requested, prices = _per_tick(logs[name])
        cumulative = cumulative_ref = 0.0
        for t in sorted(reference_px):
            cumulative += requested.get(t, 0.0)
            cumulative_ref += reference_req.get(t, 0.0)
            # float64 CPU decisions against the float32 kernels: within 1e-5 of the run's total.
            assert abs(cumulative - cumulative_ref) <= 1e-5 * total, (name, t)
            assert abs(prices[t] - reference_px[t]) <= 1e-6 * reference_px[t], (name, t)


def test_live_real_mode_runs_the_fitted_feedback(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIBRATION_ENGINE", raising=False)
    from ui.live.session import LiveSession

    session = LiveSession(seed=42)
    assert session.engine.fitted_feedback is True
    prior = float(get_parameter("M03 feedback gain (ASSUMED)").value)
    assert math.isclose(session.engine["M06"].feedback_gain, prior, rel_tol=1e-12)
    session.start()
    for _ in range(3):
        session.step_once()
    assert any(
        isinstance(e, AttentionTick) and e.cause_id == "svb06-news" for e in session.engine.bus.log
    )
