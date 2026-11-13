"""SVB-36: the E1 stock ratio is reported, not fitted; price impact leaves the fit set."""

from __future__ import annotations

import math

from calibration.fit import (
    _objective_from_fractions,
    e1_stock_ratio_summary,
    identifiable_indices,
    identifiable_names,
    load_calibration_targets,
)
from params.loader import get_parameter


def test_stock_weight_is_zero_and_target_is_still_printed() -> None:
    row = get_parameter("Calibration E1 day-1 stock objective weight")
    assert float(row.value) == 0.0
    assert "SVB-36" in row.source
    summary = e1_stock_ratio_summary(0.73)
    assert summary == {"observed": 0.73, "expected": 0.40}


def test_objective_ignores_stock_at_zero_weight() -> None:
    targets = load_calibration_targets()
    kwargs = {"targets": targets, "w_day1": 1.0, "w_day2": 1.0}
    base = _objective_from_fractions(0.249, 0.81, **kwargs)
    for stock in (0.0, 0.40, 0.73, 1.0, float("nan")):
        loss = _objective_from_fractions(0.249, 0.81, stock_ratio=stock, **kwargs)
        assert loss == base
        assert math.isfinite(loss)


def test_objective_keeps_the_stock_term_when_a_weight_is_set(tmp_path) -> None:
    src = get_parameter("Calibration E1 day-1 stock objective weight")
    text = open("params/registry.csv", encoding="utf-8").read()
    line = next(row for row in text.splitlines() if row.startswith(src.parameter + ",0,"))
    weighted = tmp_path / "registry.csv"
    weighted.write_text(text.replace(line, line.replace(",0,", ",1,", 1)), encoding="utf-8")
    targets = load_calibration_targets()
    kwargs = {"targets": targets, "w_day1": 1.0, "w_day2": 1.0, "path": weighted}
    hit = _objective_from_fractions(0.249, 0.81, stock_ratio=0.40, **kwargs)
    miss = _objective_from_fractions(0.249, 0.81, stock_ratio=0.80, **kwargs)
    assert miss > hit


def test_fit_set_drops_price_impact() -> None:
    assert identifiable_indices() == (7, 15, 16)
    assert "price_impact" not in identifiable_names()
    impact = get_parameter("M14 price impact (ASSUMED)")
    assert impact.status == "ASSUMED"
    assert impact.value == "1.18464"
