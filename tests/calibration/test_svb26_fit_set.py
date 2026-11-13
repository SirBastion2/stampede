"""SVB-26: two-coordinate fit set, Signature holdout, ±10% board, global best file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.fit import (
    FitResult,
    _default_svb_scenario,
    _parse_tolerance,
    _signature_scenario,
    _within_band,
    build_validation_board,
    fit_behavioral_parameters,
    identifiable_indices,
    identifiable_names,
    load_calibration_targets,
    log_params_from_vector,
    measure_outflow,
    run_holdout,
    vector_from_log_params,
    write_fitted_registry,
)
from params.loader import get_parameter


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def test_identifiable_indices_are_price_impact_and_mobile_share() -> None:
    row = get_parameter("Calibration identifiable parameter indices")
    assert row.value == "7,15,16"
    assert identifiable_indices() == (7, 15, 16)
    assert identifiable_names() == (
        "worry_bias",
        "mobile_share",
        "decision_to_request_rate",
    )
    assert "day-1" in row.source
    assert "mobile share" in row.source


def test_cma_dimension_and_fitted_writeback_are_two_coords(tmp_path: Path) -> None:
    indices = identifiable_indices()
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    assert len(indices) == 3
    theta = np.linspace(0.1, -0.2, num=len(indices), dtype=np.float64)
    vector = vector_from_log_params(theta, indices)
    assert vector.shape == (dim,)
    assert np.allclose(vector[[i for i in range(dim) if i not in indices]], 0.5)
    mapped = log_params_from_vector(vector, indices)
    assert mapped.shape == (len(indices),)

    src = Path("params/registry.csv")
    dest = tmp_path / "registry.csv"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    write_fitted_registry(tuple(float(x) for x in vector), run_id="svb26", seed=1, path=dest)
    text = dest.read_text(encoding="utf-8")
    assert "M14 price impact (ASSUMED)" in text
    assert "Depositor mobile share of uninsured deposits (FIT)" in text
    fitted_names = [
        line.split(",")[0]
        for line in text.splitlines()
        if ",FITTED," in line
    ]
    assert "M06 worry bias (ASSUMED)" in fitted_names
    assert "M14 price impact (ASSUMED)" not in fitted_names  # SVB-36: stock is reported, not fitted
    assert "Depositor mobile share of uninsured deposits (FIT)" in fitted_names
    assert "M07 decision-to-request rate (FIT)" in fitted_names
    assert "M06 awareness media rate (FIT)" not in fitted_names
    assert "M03 feedback gain (ASSUMED)" not in fitted_names
    assert "M06 breaking-point dispersion (ASSUMED)" not in fitted_names


def test_svb_and_signature_presets_differ_on_fake_path() -> None:
    vector = np.full(16, 0.5, dtype=np.float64)
    kwargs = {
        "steps": 10,
        "param_vector": vector,
        "requested": True,
        "softness": 0.06,
        "clamp_inputs": False,
    }
    svb = measure_outflow(_default_svb_scenario(7), **kwargs)
    signature = measure_outflow(_signature_scenario(7), **kwargs)
    assert _signature_scenario(7).preset == "crypto-heavy"
    assert _default_svb_scenario(7).preset == "tech-concentrated"
    assert svb[0] != signature[0]
    assert svb[1] != signature[1]


def test_holdout_uses_signature_preset_and_does_not_refit_capacity(monkeypatch) -> None:
    recorded: list[str] = []

    def _record(scenario, **_kwargs):
        recorded.append(scenario.preset)
        if "community" in scenario.preset:
            return 0.0, 0.0, False
        return 0.11, 0.11, False

    def _must_not_solve(**_kwargs):
        raise AssertionError("holdout must not re-solve capacity")

    monkeypatch.setattr("calibration.fit.measure_outflow", _record)
    monkeypatch.setattr("calibration.fit.fit_payment_capacity", _must_not_solve)
    result = run_holdout(tuple(0.5 for _ in range(16)), seed=3, steps=8)
    assert "crypto-heavy" in recorded
    assert "insured-community" in recorded
    assert "tech-concentrated" not in recorded
    assert result.signature_day1_fraction == 0.11
    assert result.placebo_survives is True


def test_next_open_tolerance_is_ten_percent_relative() -> None:
    row = get_parameter("Validation target: requests by next open")
    assert row.value == "±10%"
    band = _parse_tolerance(row.value)
    assert band.relative == 0.10
    assert band.absolute is None
    expected = 0.81
    assert _within_band(0.9237, expected, band) is False
    assert _within_band(0.80, expected, band) is True
    assert abs(0.9237 - expected) > expected * 0.10
    assert abs(0.80 - expected) <= expected * 0.10


def test_board_compares_next_open_requests_to_e1_two_day(monkeypatch) -> None:
    targets = load_calibration_targets()
    assert targets.svb_two_day_fraction == 0.81

    def _fake_measure(_scenario, **kwargs):
        if kwargs.get("requested"):
            return 0.24, 0.81, False
        return 0.24, 0.24, False

    monkeypatch.setattr("calibration.fit.measure_outflow", _fake_measure)
    from calibration.fit import HoldoutResult

    board = build_validation_board(
        FitResult(seed=0, vector=(0.5,) * 16, objective=0.0, generations=1),
        HoldoutResult(signature_day1_fraction=0.20, placebo_survives=True),
    )
    next_open = next(row for row in board if row.target_id == "E1-next-open")
    assert next_open.expected == targets.svb_two_day_fraction
    assert next_open.observed == 0.81
    assert next_open.tolerance.relative == 0.10


def test_best_vector_json_matches_fit_result_after_nonmonotone_run(
    monkeypatch, tmp_path: Path
) -> None:
    _silence_worker_threads(monkeypatch)
    monkeypatch.setattr(
        "calibration.fit.bracket_tipping_point",
        lambda **_kwargs: ((0.4, 0.8), []),
    )
    waves = {"n": 0}

    def _nonmonotone(samples, **_kwargs) -> list[float]:
        waves["n"] += 1
        # Generation 1 is best; later generations get worse so a last-gen
        # write would disagree with FitResult.
        bump = 0.0 if waves["n"] == 1 else 2.0 * waves["n"]
        losses = []
        for sample in samples:
            theta = log_params_from_vector(np.asarray(sample, dtype=np.float64), identifiable_indices())
            losses.append(float(np.dot(theta, theta) + bump))
        return losses

    monkeypatch.setattr("calibration.fit.evaluate_population_objectives", _nonmonotone)
    if hasattr(fit_behavioral_parameters, "last_es"):
        delattr(fit_behavioral_parameters, "last_es")
    run_dir = tmp_path / "run"
    result = fit_behavioral_parameters(
        17,
        generations=4,
        population=6,
        restarts=0,
        use_fake_objective=True,
        workers=1,
        checkpoint_dir=run_dir,
        checkpoint_every=1,
    )
    payload = json.loads((run_dir / "best_vector.json").read_text(encoding="utf-8"))
    assert payload["objective"] == result.objective
    assert payload["vector"] == list(result.vector)
    assert len(result.log_params) == len(identifiable_indices())
