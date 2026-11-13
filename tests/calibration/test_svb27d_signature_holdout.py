"""SVB-27d: the Signature (E5) holdout compares requested day one, which moves with the vector."""
from __future__ import annotations

import numpy as np

import calibration.fit as fit


def _fake_measure(scenario, *, steps, param_vector=None, requested=False, disable_caps=False, **kwargs):
    level = float(np.mean(param_vector)) if param_vector is not None else 0.5
    if requested:
        return 0.30 * level, 0.35 * level, False      # behavior: moves with the vector
    return 0.239979, 0.239979, False                 # settled: pinned at the shared capacity


def test_signature_requested_moves_with_the_vector(monkeypatch):
    monkeypatch.setattr(fit, "measure_outflow", _fake_measure)
    low = fit.run_holdout(tuple(np.full(16, 0.4)), 3060, steps=780)
    high = fit.run_holdout(tuple(np.full(16, 0.6)), 3060, steps=780)
    assert low.signature_day1_fraction == high.signature_day1_fraction == 0.239979
    assert low.signature_day1_requested != high.signature_day1_requested


def test_board_e5_row_uses_requested_when_the_target_row_exists(monkeypatch):
    monkeypatch.setattr(fit, "measure_outflow", _fake_measure)
    targets = fit.load_calibration_targets()
    assert targets.signature_day1_requested_fraction is not None, (
        "apply calibration/registry_updates/svb27d_e5_requested_target.json"
    )
    vector = tuple(np.full(16, 0.5))
    holdout = fit.run_holdout(vector, 3060, steps=780)
    board = fit.build_validation_board(fit.FitResult(seed=3060, vector=vector, objective=0.0, generations=1), holdout)
    e5 = next(row for row in board if row.target_id == "E5-day1")
    assert e5.observed == holdout.signature_day1_requested
    assert e5.expected == targets.signature_day1_requested_fraction
    assert "requested" in e5.detail
