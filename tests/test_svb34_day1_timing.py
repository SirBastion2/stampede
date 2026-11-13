"""SVB-34: stage-1 objective fits requested-by-day-one-close; E5 stays a holdout."""

from __future__ import annotations

from dataclasses import replace

from calibration.fit import (
    FitResult,
    HoldoutResult,
    _objective_from_fractions,
    _score_outflow_log,
    build_validation_board,
    day1_close_minute,
    e1_stock_ratio_summary,
    format_validation_board_markdown,
    identifiable_indices,
    load_calibration_targets,
)
from contracts.events import WithdrawalRequested
from params.loader import get_parameter


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def test_objective_is_higher_when_day1_requests_miss_target() -> None:
    """Next-open and settled on target is not enough: day-one close must hit 0.249."""
    targets = load_calibration_targets()
    assert targets.svb_requested_day1_fraction == 0.249
    assert targets.svb_two_day_fraction == 0.81
    kwargs = {"targets": targets, "w_day1": 1.0, "w_day2": 1.0, "stock_ratio": 0.40}
    on_target = _objective_from_fractions(0.249, 0.81, **kwargs)
    # Same next-open (0.81) and stock (0.40); settled 0.24 is stage 2, not this loss.
    saturated_day1 = _objective_from_fractions(0.8093, 0.81, **kwargs)
    assert saturated_day1 > on_target


def test_board_gains_requested_day1_row_and_stock_ratio_field(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)

    def _fake_measure(_scenario, **kwargs):
        if kwargs.get("requested"):
            return 0.249, 0.81, False
        return 0.24, 0.24, False

    monkeypatch.setattr("calibration.fit.measure_outflow", _fake_measure)
    board = build_validation_board(
        FitResult(seed=0, vector=(0.5,) * 17, objective=0.0, generations=1),
        HoldoutResult(
            signature_day1_fraction=0.20,
            placebo_survives=True,
            signature_day1_requested=0.249,
        ),
    )
    ids = [row.target_id for row in board]
    assert "E1-requested-day1" in ids
    day1 = next(row for row in board if row.target_id == "E1-requested-day1")
    assert day1.expected == 0.249
    assert day1.observed == 0.249
    assert day1.tolerance.relative == 0.10
    assert day1.passed is True

    stock = e1_stock_ratio_summary(0.40)
    assert stock["observed"] == 0.40
    assert stock["expected"] == 0.40
    md = format_validation_board_markdown(
        board,
        run_id="svb34",
        seed=0,
        fit_objective=0.0,
        stock_ratio=stock["observed"],
        stock_target=stock["expected"],
    )
    assert "E1-requested-day1" in md
    assert "E1 stock ratio (observed vs target)" in md
    assert "`0.4` vs `0.4`" in md


def test_e5_holdout_is_not_part_of_the_fit_objective() -> None:
    targets = load_calibration_targets()
    base = _objective_from_fractions(
        0.249, 0.81, targets=targets, w_day1=1.0, w_day2=1.0
    )
    tweaked = replace(
        targets,
        signature_day1_fraction=0.99,
        signature_day1_requested_fraction=0.01,
    )
    same = _objective_from_fractions(
        0.249, 0.81, targets=tweaked, w_day1=1.0, w_day2=1.0
    )
    assert same == base
    assert identifiable_indices() == (7, 15, 16)  # SVB-36: price impact left the fit set
    assert int(float(get_parameter("Calibration behavioral parameter dimension").value)) == 17


def test_day1_close_reads_the_trading_day_clock(monkeypatch) -> None:
    assert day1_close_minute() == float(
        get_parameter("Non-bank sim minutes per day").value
    )
    assert day1_close_minute() == 390.0
    monkeypatch.setattr("calibration.fit.day1_close_minute", lambda path=None: 10.0)
    inside = WithdrawalRequested(
        t=9.0, seed=0, depositor_id="a", amount=42.0, channel="wire"
    )
    at_close = WithdrawalRequested(
        t=10.0, seed=0, depositor_id="b", amount=100.0, channel="wire"
    )
    day1, _next_open, _ = _score_outflow_log([inside, at_close], requested=True)
    deposits = 175_000_000_000.0
    assert abs(day1 - 42.0 / deposits) < 1e-18
