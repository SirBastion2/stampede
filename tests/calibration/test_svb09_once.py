"""SVB-09: pack static state once per seed; run_fit defaults to one pass."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from calibration.fit import (
    CapsOnValidation,
    FitResult,
    HoldoutResult,
    _default_svb_scenario,
    _score_outflow_log,
)
from calibration.run_fit import _parse_args, _run
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, simulate_logs


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def test_three_generations_pack_static_once_per_seed(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "24")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    clear_static_pack_cache()

    import engine.gpu_sim as gpu_sim

    seeds_seen: list[int] = []
    original = gpu_sim._pack_static

    def _counting(modules, xp):
        seeds_seen.append(int(modules[0].seed))
        return original(modules, xp)

    monkeypatch.setattr(gpu_sim, "_pack_static", _counting)

    scenario = _default_svb_scenario(31)
    vectors = [np.full(12, 0.3), np.full(12, 0.4)]
    seeds = (31, 37)
    logs: list[list[list[object]]] = []
    for _ in range(3):
        logs.append(simulate_logs(scenario, 2, vectors, array_module="numpy", seeds=seeds))

    assert seeds_seen == [31, 37]
    first = [_score_outflow_log(log) for log in logs[0]]
    for later in logs[1:]:
        assert [_score_outflow_log(log) for log in later] == first

    clear_static_pack_cache()
    clear_sample_population_cache()


def test_run_fit_default_flags_one_pass(monkeypatch, tmp_path: Path) -> None:
    _silence_worker_threads(monkeypatch)
    passes: list[int] = []

    def _fake_fit(seed, **_kwargs) -> FitResult:
        passes.append(int(seed))
        return FitResult(seed=int(seed), vector=(0.2,) * 12, objective=1.0, generations=1)

    monkeypatch.setattr("calibration.run_fit.fit_behavioral_parameters", _fake_fit)
    monkeypatch.setattr("calibration.run_fit.fit_payment_capacity", lambda **_k: 1.0)
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

    monkeypatch.setattr(sys, "argv", ["run_fit"])
    parsed = _parse_args()
    assert parsed.extra_seeds == ""

    args = argparse.Namespace(
        gpu_full=False,
        gpu_sim=False,
        gpu=False,
        workers=1,
        seed=3060,
        extra_seeds=parsed.extra_seeds,
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
    logger = logging.getLogger("test_svb09")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert _run(args, "svb09-test", run_dir, logger) == 0
    assert passes == [3060]
