"""SVB-03: real CMA-ES, bracket only for the start, pickle resume."""

from __future__ import annotations

import inspect

import numpy as np

from calibration.fit import (
    _project_theta,
    fit_behavioral_parameters,
    identifiable_indices,
    identifiable_names,
    load_cma_es,
    log_params_from_vector,
)


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _sphere_on_theta(samples, **_kwargs) -> list[float]:
    indices = identifiable_indices()
    losses: list[float] = []
    for sample in samples:
        theta = log_params_from_vector(np.asarray(sample, dtype=np.float64), indices)
        losses.append(float(np.dot(theta, theta)))
    return losses


def _stub_bracket(monkeypatch) -> None:
    monkeypatch.setattr(
        "calibration.fit.bracket_tipping_point",
        lambda **_kwargs: ((1.0, 1.4), []),
    )


def _isolate_cma_state() -> None:
    """Drop leftover CMA state so a combined run matches isolation."""
    if hasattr(fit_behavioral_parameters, "last_es"):
        delattr(fit_behavioral_parameters, "last_es")
    np.random.seed(17)


def _cma_fit(monkeypatch, **kwargs):
    _silence_worker_threads(monkeypatch)
    _isolate_cma_state()
    _stub_bracket(monkeypatch)
    monkeypatch.setattr("calibration.fit.evaluate_population_objectives", _sphere_on_theta)
    defaults = {
        "generations": 60,
        "population": 8,
        "restarts": 0,
        "use_fake_objective": True,
        "workers": 1,
    }
    defaults.update(kwargs)
    return fit_behavioral_parameters(17, **defaults)


def test_cma_reaches_sphere_below_1e6(monkeypatch) -> None:
    indices = identifiable_indices()
    result = _cma_fit(monkeypatch, generations=60)
    assert len(result.log_params) == len(indices)
    assert result.identifiable == identifiable_names()
    assert result.objective < 1e-6


def test_resume_from_es_pkl_matches_uninterrupted_ask(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    _cma_fit(
        monkeypatch,
        generations=3,
        checkpoint_dir=run_dir,
        checkpoint_every=1,
    )
    live = fit_behavioral_parameters.last_es
    resumed = load_cma_es(run_dir)
    assert resumed is not None
    assert np.allclose(np.asarray(resumed.ask()), np.asarray(live.ask()))


def test_project_theta_never_receives_box(monkeypatch) -> None:
    original = _project_theta
    calls: list[dict] = []

    def _wrapped(*args, **kwargs):
        calls.append(kwargs)
        assert "box" not in kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr("calibration.fit._project_theta", _wrapped)
    source = inspect.getsource(original)
    assert "box=" not in source
    assert "box" not in inspect.signature(original).parameters
    _cma_fit(monkeypatch, generations=2)
    assert calls
