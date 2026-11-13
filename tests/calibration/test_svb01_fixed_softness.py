"""SVB-01: one fixed behavioral-noise temperature in fit, validation, and holdouts."""

from __future__ import annotations

from calibration.fit import (
    fit_behavioral_parameters,
    fixed_softness,
    run_holdout,
    validate_fitted_parameters,
)


def test_fit_validate_holdout_use_fixed_softness(monkeypatch) -> None:
    recorded: list[float] = []

    def _record_measure_outflow(
        _scenario,
        *,
        steps,
        param_vector=None,
        requested=False,
        disable_caps=False,
        softness=0.0,
        clamp_inputs=True,
    ):
        del steps, param_vector, requested, disable_caps, clamp_inputs
        recorded.append(float(softness))
        return 0.24, 0.81, False

    def _record_simulate_logs(*args, **kwargs):
        softness = kwargs.get("softness")
        if softness is None and len(args) > 4:
            softness = args[4]
        recorded.append(float(0.0 if softness is None else softness))
        vectors = args[2] if len(args) >= 3 else kwargs.get("vectors", [])
        seeds = kwargs.get("seeds")
        n_vectors = len(vectors)
        n_seeds = len(seeds) if seeds else 1
        return [[] for _ in range(max(1, n_vectors) * n_seeds)]

    def _schedule_must_not_run(*_args, **_kwargs) -> float:
        raise AssertionError("softness_schedule should not be called")

    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)
    monkeypatch.setattr("calibration.fit.measure_outflow", _record_measure_outflow)
    monkeypatch.setattr("engine.gpu_sim.simulate_logs", _record_simulate_logs)
    monkeypatch.setattr("calibration.fit.softness_schedule", _schedule_must_not_run)

    expected = fixed_softness()
    fit = fit_behavioral_parameters(0, generations=2)
    validate_fitted_parameters(fit.vector, 0)
    run_holdout(fit.vector, 0)

    assert recorded
    assert all(value == expected for value in recorded)
