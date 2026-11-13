"""SVB-04: log-error floor, unclamped holdout, holdout steps, registry from global best."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from calibration.fit import FitResult, _log_sq_error
from calibration.run_fit import _run


def test_log_sq_error_zero_is_worse_than_small_positive() -> None:
    assert _log_sq_error(0.0, 0.24) > _log_sq_error(0.001, 0.24) > _log_sq_error(0.2, 0.24)


@pytest.mark.parametrize("write_registry", [True, False])
def test_run_fit_holdout_unclamped_and_writes_global_best_once(
    monkeypatch, tmp_path: Path, write_registry: bool
) -> None:
    recorded: list[dict[str, object]] = []
    registry_writes: list[tuple[tuple[float, ...], int]] = []
    worse = (0.11, 0.12, 0.13, 0.14)
    better = (0.91, 0.92, 0.93, 0.94)
    holdout_minutes = 42

    def _record_measure_outflow(
        scenario,
        *,
        steps,
        param_vector=None,
        requested=False,
        disable_caps=False,
        softness=0.0,
        clamp_inputs=True,
    ):
        del param_vector, requested, disable_caps, softness
        recorded.append(
            {
                "steps": steps,
                "clamp_inputs": clamp_inputs,
                "preset": scenario.preset,
            }
        )
        return 0.24, 0.81, False

    def _fake_fit(seed, **_kwargs) -> FitResult:
        if seed == 11:
            return FitResult(seed=11, vector=worse, objective=5.0, generations=1)
        return FitResult(seed=22, vector=better, objective=0.5, generations=1)

    def _record_registry(vector, *, run_id, seed, path=None) -> None:
        del run_id, path
        registry_writes.append((tuple(vector), int(seed)))

    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)
    monkeypatch.setattr("calibration.fit.measure_outflow", _record_measure_outflow)
    monkeypatch.setattr("calibration.run_fit.fit_behavioral_parameters", _fake_fit)
    monkeypatch.setattr("calibration.run_fit.fit_payment_capacity", lambda **_k: 1.0)
    monkeypatch.setattr("calibration.run_fit.write_fitted_registry", _record_registry)
    monkeypatch.setattr("calibration.run_fit.write_run_diagnostics", lambda *_a, **_k: None)
    monkeypatch.setattr("calibration.run_fit._REPO_ROOT", tmp_path)
    (tmp_path / "docs").mkdir()

    args = argparse.Namespace(
        gpu_full=False,
        gpu_sim=False,
        gpu=False,
        workers=1,
        seed=11,
        extra_seeds="22",
        eval_minutes=7,
        holdout_minutes=holdout_minutes,
        resume=False,
        generations=1,
        population=4,
        checkpoint_every=1,
        patience=0,
        improvement_epsilon=1e-3,
        runs_dir=tmp_path / "runs",
        write_registry=write_registry,
    )
    logger = logging.getLogger("test_svb04")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert _run(args, "svb04-test", run_dir, logger) == 0

    holdout_calls = [row for row in recorded if row["preset"] in {"crypto-heavy", "insured-community"}]
    assert holdout_calls
    assert all(row["steps"] == holdout_minutes for row in holdout_calls)
    assert recorded
    assert all(row["clamp_inputs"] is False for row in recorded)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert (run_dir / "validation-board-svb04-test.md").exists()
    assert not list((tmp_path / "docs").glob("validation-board-*.md"))
    if write_registry:
        # SVB-49: only with --write-registry, and then once, from the global best.
        assert registry_writes == [(better, 22)]
        assert summary["persist_command"] is None
    else:
        assert registry_writes == []
        assert summary["fitted_rows"] == {}
        assert summary["fitted_rows_written"] is False
        assert summary["persist_command"] == f"python -m calibration.persist_fit --run-dir {run_dir}"


def test_run_fit_default_does_not_write_the_registry(monkeypatch) -> None:
    """SVB-49: --write-registry is opt-in."""
    import sys

    from calibration.run_fit import _parse_args

    monkeypatch.setattr(sys, "argv", ["run_fit"])
    assert _parse_args().write_registry is False
    monkeypatch.setattr(sys, "argv", ["run_fit", "--write-registry"])
    assert _parse_args().write_registry is True


def test_run_fit_writes_the_board_into_the_run_dir_only() -> None:
    """SVB-55 (E2): the validation board path is under the run dir, not docs/."""
    import inspect

    import calibration.run_fit as run_fit

    source = inspect.getsource(run_fit._run)
    assert 'run_dir / f"validation-board-{run_id}.md"' in source
    assert '_REPO_ROOT / "docs" / f"validation-board' not in source
