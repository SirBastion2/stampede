"""SVB-21: stage 2 batched capacity-multiplier grid."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from calibration.fit import (
    _DAY1_CLOSE,
    CapsOnValidation,
    FitResult,
    HoldoutResult,
    _default_svb_scenario,
    _score_outflow_log,
    fit_payment_capacity,
    fixed_softness,
)
from calibration.run_fit import _run
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, simulate_logs


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def test_stage2_grid_hits_settled_on_stub() -> None:
    calls: list[list[float]] = []

    def stub(multipliers):
        calls.append([float(m) for m in multipliers])
        return [0.24 + 0.05 * math.log(float(m) / 2.0) for m in multipliers]

    multiplier = fit_payment_capacity(measure=stub, target=0.24)
    assert 1 <= len(calls) <= 2
    assert all(len(batch) == 16 for batch in calls)
    observed = 0.24 + 0.05 * math.log(float(multiplier) / 2.0)
    assert abs(observed - 0.24) <= 0.002


def test_registry_multiplier_changes_settled_day_one(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "24")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    clear_static_pack_cache()

    from params.loader import list_by_module

    scenario = _default_svb_scenario(21)
    vector = np.full(12, 0.5)
    schedule = [(float(t), 1.0) for t in range(60)]

    def _run_with_multiplier(value: str) -> float:
        def _rows(module: str, path=None):
            rows = list_by_module(module, path)
            out = []
            for row in rows:
                if row.parameter == "M07 capacity multiplier (FIT)":
                    out.append(replace(row, value=value))
                else:
                    out.append(row)
            return out

        monkeypatch.setattr("engine.bank.payments.list_by_module", _rows)
        clear_static_pack_cache()
        logs = simulate_logs(
            scenario,
            60,
            [vector],
            array_module="numpy",
            disable_caps=False,
            softness=fixed_softness(),
            clamp_inputs=False,
            schedule=schedule,
        )
        day1, _next_open, _closed = _score_outflow_log(logs[0], requested=False)
        assert all(event.t < _DAY1_CLOSE for event in logs[0] if hasattr(event, "t"))
        return day1

    low = _run_with_multiplier("1e-6")
    high = _run_with_multiplier("1.0")
    assert low != high

    clear_static_pack_cache()
    clear_sample_population_cache()


def test_run_fit_calls_stage2_once(monkeypatch, tmp_path: Path) -> None:
    _silence_worker_threads(monkeypatch)
    stage2_calls: list[dict[str, object]] = []

    def _fake_fit(seed, **_kwargs) -> FitResult:
        return FitResult(seed=int(seed), vector=(0.2,) * 12, objective=1.0, generations=1)

    def _fake_stage2(**kwargs):
        stage2_calls.append(dict(kwargs))
        return 1.0

    monkeypatch.setattr("calibration.run_fit.fit_behavioral_parameters", _fake_fit)
    monkeypatch.setattr("calibration.run_fit.fit_payment_capacity", _fake_stage2)
    monkeypatch.setattr(
        "calibration.run_fit.run_holdout",
        lambda *_a, **_k: HoldoutResult(signature_day1_fraction=0.2, placebo_survives=True),
    )
    monkeypatch.setattr(
        "calibration.run_fit.validate_fitted_parameters",
        lambda *_a, **_k: CapsOnValidation(
            day1_settled=0.2,
            two_day_settled=0.8,
            day1_requested=0.2,
            two_day_requested=0.8,
            closed=False,
        ),
    )
    monkeypatch.setattr("calibration.run_fit.build_validation_board", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "calibration.run_fit.format_validation_board_markdown",
        lambda *_a, **_k: "",
    )
    monkeypatch.setattr("calibration.run_fit.write_fitted_registry", lambda *_a, **_k: None)
    monkeypatch.setattr("calibration.run_fit.write_run_diagnostics", lambda *_a, **_k: None)
    monkeypatch.setattr("calibration.run_fit._REPO_ROOT", tmp_path)
    (tmp_path / "docs").mkdir()

    args = argparse.Namespace(
        gpu_full=False,
        gpu_sim=False,
        gpu=False,
        workers=1,
        seed=3060,
        extra_seeds="",
        eval_minutes=7,
        holdout_minutes=42,
        resume=False,
        generations=1,
        population=4,
        checkpoint_every=1,
        patience=0,
        improvement_epsilon=1e-3,
        runs_dir=tmp_path / "runs",
    )
    logger = logging.getLogger("test_svb21")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert _run(args, "svb21-test", run_dir, logger) == 0
    assert len(stage2_calls) == 1
    assert stage2_calls[0]["vector"] == (0.2,) * 12
    assert stage2_calls[0]["write"] is True
