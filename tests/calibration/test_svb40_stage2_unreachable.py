"""SVB-40 S2: an unreachable settled day-one target is reported, not guessed. CPU-only."""
from __future__ import annotations

import csv
import math

import pytest

# 8-seed sweep of stage 2's settled measure (Nov 2 2026), log-linear between points.
_SWEEP = (
    (0.05, 0.0044),
    (1.22, 0.043),
    (4.05, 0.114),
    (4.5, 0.124),
    (6.03, 0.157),
    (8.997, 0.215),
    (13.41, 0.2202),
    (20.0, 0.2202),
)


def _sweep_curve(xs):
    import numpy as np

    log_x = np.log([m for m, _ in _SWEEP])
    ys = [s for _, s in _SWEEP]
    return [float(np.interp(math.log(float(x)), log_x, ys)) for x in xs]


def test_unreachable_target_returns_plateau_start():
    from calibration.fit import _solve_capacity_grid

    report: dict[str, object] = {}
    capacity = _solve_capacity_grid(_sweep_curve, 0.24, report=report)
    assert capacity == pytest.approx(13.414, rel=1e-3)
    assert report["reachable"] is False
    assert report["plateau"] == pytest.approx(0.2202, abs=1e-9)
    assert report["multiplier"] == pytest.approx(capacity)
    assert report["target"] == pytest.approx(0.24)
    # The settled value at the returned multiplier is on the plateau.
    assert _sweep_curve([capacity])[0] >= 0.2202 - 0.002


def test_reachable_target_is_unchanged():
    from calibration.fit import _interpolate_capacity, _log_spaced_multipliers, _solve_capacity_grid

    report: dict[str, object] = {}
    capacity = _solve_capacity_grid(_sweep_curve, 0.15, report=report)
    assert capacity == pytest.approx(5.5709, rel=1e-3)
    assert report["reachable"] is True
    grid = _log_spaced_multipliers(0.05, 20.0, 16)
    assert capacity == _interpolate_capacity(_sweep_curve, grid, _sweep_curve(grid), 0.15)


def test_target_below_the_curve_still_returns_the_low_edge():
    from calibration.fit import _solve_capacity_grid

    report: dict[str, object] = {}
    assert _solve_capacity_grid(_sweep_curve, 0.001, report=report) == pytest.approx(0.05)
    assert report["reachable"] is True


def test_fit_payment_capacity_fills_report_and_marks_the_row(tmp_path):
    from calibration.fit import fit_payment_capacity

    registry = tmp_path / "registry.csv"
    registry.write_text("Parameter,Value,Status,Source,Module,Unit\n", encoding="utf-8")
    report: dict[str, object] = {}
    capacity = fit_payment_capacity(
        measure=_sweep_curve,
        target=0.24,
        path=registry,
        write=True,
        run_id="run-a",
        seed=3060,
        report=report,
    )
    assert report["reachable"] is False
    rows = {r["Parameter"]: r for r in csv.DictReader(registry.open(encoding="utf-8"))}
    row = rows["M07 capacity multiplier (FIT)"]
    assert float(row["Value"]) == pytest.approx(capacity, rel=1e-5)
    assert row["Status"] == "FITTED"
    assert row["Source"].startswith("run-a seed=3060; target 0.24 unreachable; plateau 0.2202")


def test_reachable_write_keeps_the_plain_source(tmp_path):
    from calibration.fit import fit_payment_capacity

    registry = tmp_path / "registry.csv"
    registry.write_text("Parameter,Value,Status,Source,Module,Unit\n", encoding="utf-8")
    fit_payment_capacity(
        measure=_sweep_curve, target=0.15, path=registry, write=True, run_id="run-a", seed=3060
    )
    rows = {r["Parameter"]: r for r in csv.DictReader(registry.open(encoding="utf-8"))}
    assert rows["M07 capacity multiplier (FIT)"]["Source"] == "run-a seed=3060"
