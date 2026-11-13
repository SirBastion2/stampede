"""SVB-40 S1: stage 2 must measure the same model the board measures.

The fit vector is normalized against the registry at fit time. ``persist_fit --stage2`` used to
write the fitted rows first and then solve stage 2 with that vector, which applied the fit a
second time. These tests are CPU-only: the solver and the engine setup are monkeypatched.
"""
from __future__ import annotations

import csv
import json
import math

import pytest

_SVB_REG = (
    "Parameter,Value,Status,Source,Module,Unit\n"
    "M14 price impact (ASSUMED),1.13,ASSUMED,SVB-25 prior,M14,1/share\n"
    "Depositor mobile share of uninsured deposits (FIT),0.88,ASSUMED,SVB-24,M06,fraction\n"
)

_FIT_VECTOR = [0.5] * 14 + [0.559, 0.4985]


def _run_dir(tmp_path):
    run = tmp_path / "m27-x"
    run.mkdir()
    (run / "best_vector.json").write_text(
        json.dumps({"vector": _FIT_VECTOR, "seed": 3060, "objective": 1e-4, "run_id": "m27-x"}),
        encoding="utf-8",
    )
    return run


def _fake_updates(registry):
    """Like the real mapping: physical value = current registry value x a vector scale."""

    def compute(vector, *, path=None):
        rows = {r["Parameter"]: r for r in csv.DictReader(registry.open(encoding="utf-8"))}
        impact = float(rows["M14 price impact (ASSUMED)"]["Value"]) * 1.0484
        share = float(rows["Depositor mobile share of uninsured deposits (FIT)"]["Value"]) * 0.9988
        return {
            "M14 price impact (ASSUMED)": f"{impact:.6g}",
            "Depositor mobile share of uninsured deposits (FIT)": f"{share:.6g}",
        }

    return compute


def _patch_persist(monkeypatch, registry):
    import calibration.fit as fit
    import calibration.persist_fit as persist

    monkeypatch.setattr(fit, "fitted_registry_updates", _fake_updates(registry))
    monkeypatch.setattr(fit, "fitted_m06_values", lambda vector: {})
    monkeypatch.setattr(persist, "apply_stage2_engine", lambda logger=None: {"stub": True})
    return fit, persist


def test_stage2_solves_before_fitted_rows_are_written(tmp_path, monkeypatch):
    registry = tmp_path / "registry.csv"
    registry.write_text(_SVB_REG, encoding="utf-8")
    fit, persist = _patch_persist(monkeypatch, registry)
    seen: dict[str, object] = {}

    def fake_capacity(**kwargs):
        seen["registry_at_solve"] = registry.read_text(encoding="utf-8")
        seen["vector"] = kwargs["vector"]
        return 13.4

    monkeypatch.setattr(fit, "fit_payment_capacity", fake_capacity)
    before = registry.read_text(encoding="utf-8")
    run = _run_dir(tmp_path)

    assert persist.main(["--run-dir", str(run), "--registry", str(registry), "--stage2"]) == 0
    # The solve saw the untouched (fit-time) registry, paired with the fit vector.
    assert seen["registry_at_solve"] == before
    assert seen["vector"] == tuple(_FIT_VECTOR)
    # The fitted rows are written afterwards.
    written = {r["Parameter"]: r for r in csv.DictReader(registry.open(encoding="utf-8"))}
    assert written["M14 price impact (ASSUMED)"]["Status"] == "FITTED"
    assert written["Depositor mobile share of uninsured deposits (FIT)"]["Status"] == "FITTED"


def test_failed_stage2_solve_writes_nothing(tmp_path, monkeypatch):
    registry = tmp_path / "registry.csv"
    registry.write_text(_SVB_REG, encoding="utf-8")
    fit, persist = _patch_persist(monkeypatch, registry)

    def failing_capacity(**kwargs):
        raise RuntimeError("stage 2 failed")

    monkeypatch.setattr(fit, "fit_payment_capacity", failing_capacity)
    before = registry.read_bytes()
    run = _run_dir(tmp_path)
    with pytest.raises(RuntimeError, match="stage 2 failed"):
        persist.main(["--run-dir", str(run), "--registry", str(registry), "--stage2"])
    assert registry.read_bytes() == before


def test_without_stage2_rows_are_still_written(tmp_path, monkeypatch):
    registry = tmp_path / "registry.csv"
    registry.write_text(_SVB_REG, encoding="utf-8")
    fit, persist = _patch_persist(monkeypatch, registry)

    def must_not_run(**kwargs):
        raise AssertionError("stage 2 must not run without --stage2")

    monkeypatch.setattr(fit, "fit_payment_capacity", must_not_run)
    run = _run_dir(tmp_path)
    assert persist.main(["--run-dir", str(run), "--registry", str(registry)]) == 0
    assert "FITTED" in registry.read_text(encoding="utf-8")


# --- the pairing guard inside fit_payment_capacity ---------------------------------------------

_SAVED_REG = (
    "Parameter,Value,Status,Source,Module,Unit\n"
    'Calibration identifiable parameter indices,"7,15,16",ASSUMED,test,M27,index\n'
    "M06 worry bias (ASSUMED),-1.73736,{status},run-a seed=3060,M06,logit\n"
    "Depositor mobile share of uninsured deposits (FIT),0.963547,{status},run-a seed=3060,"
    "M06,fraction\n"
    "M07 decision-to-request rate (FIT),0.00210008,{status},run-a seed=3060,M07,1/min\n"
)


def _fit_vector_17() -> list[float]:
    vector = [0.5] * 17
    vector[7], vector[15], vector[16] = 0.6658, 0.6, 0.78
    return vector


def _monotone_curve(xs, **_kwargs):
    return [0.25 * (1.0 - math.exp(-float(x) / 5.0)) for x in xs]


def _registry(tmp_path, status: str):
    path = tmp_path / f"registry-{status}.csv"
    path.write_text(_SAVED_REG.format(status=status), encoding="utf-8")
    return path


def test_guard_refuses_fit_vector_on_its_own_saved_rows(tmp_path, monkeypatch):
    import calibration.fit as fit

    def must_not_measure(*args, **kwargs):
        raise AssertionError("the guard must stop before any simulation")

    monkeypatch.setattr(fit, "_settled_day1_for_multipliers", must_not_measure)
    registry = _registry(tmp_path, "FITTED")
    with pytest.raises(RuntimeError, match="on top of its own saved rows"):
        fit.fit_payment_capacity(
            vector=_fit_vector_17(), target=0.2, seed=3060, run_id="run-a", path=registry
        )


def test_guard_allows_the_half_vector_on_saved_rows(tmp_path, monkeypatch):
    import calibration.fit as fit

    monkeypatch.setattr(fit, "_settled_day1_for_multipliers", _monotone_curve)
    registry = _registry(tmp_path, "FITTED")
    capacity = fit.fit_payment_capacity(
        vector=[0.5] * 17, target=0.2, seed=3060, run_id="run-a", path=registry
    )
    assert 0.05 <= capacity <= 20.0


def test_guard_allows_fit_vector_on_rows_from_another_run_or_unsaved(tmp_path, monkeypatch):
    import calibration.fit as fit

    monkeypatch.setattr(fit, "_settled_day1_for_multipliers", _monotone_curve)
    saved = _registry(tmp_path, "FITTED")
    other_run = fit.fit_payment_capacity(
        vector=_fit_vector_17(), target=0.2, seed=3060, run_id="run-b", path=saved
    )
    unsaved = _registry(tmp_path, "ASSUMED")
    same_run_unsaved = fit.fit_payment_capacity(
        vector=_fit_vector_17(), target=0.2, seed=3060, run_id="run-a", path=unsaved
    )
    assert other_run == pytest.approx(same_run_unsaved)
