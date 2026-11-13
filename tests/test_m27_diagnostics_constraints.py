"""M27 diagnostics, exact-zero invariance, and non-bank request bound."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.rumor_input import IRRELEVANT_READ_AS, classify_rumor_market
from calibration.diagnostics import (
    attach_on_generation,
    bind_rates_from_log,
    record_generation,
    withdrawn_share_from_log,
    write_run_diagnostics,
)
from calibration.diagnostics.summary import format_summary
from calibration.diagnostics.summary import main as summary_main
from calibration.fit import _default_svb_scenario
from calibration.invariance import (
    IRRELEVANT_PROBES,
    gated_rumors_for_text,
    invariance_passed,
    is_irrelevant_input,
    validate_exact_zero_invariance,
)
from calibration.nonbank import (
    build_nonbank_scenario,
    daily_deposit_request_fraction,
    fit_nonbank_parameters,
    measure_nonbank_observables,
    nonbank_request_bound,
    nonbank_request_bound_penalty,
    validate_nonbank_bound,
)
from calibration.rumor_labels import calibration_rumor_classification
from contracts.events import BankClosed, PaymentSettled, PriceTick, WithdrawalRequested
from engine.registry import run_scenario
from ops.event_log import event_log_hash
from params.loader import get_parameter

_SEED = 11
_STEPS = 8


def test_record_generation_writes_diagnostics_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "m27-diag"
    seen: list[int] = []
    hook = attach_on_generation(run_dir, lambda gen, _obj, _vec: seen.append(gen), seed=7)
    hook(1, 1.5, (0.0, 0.5, 1.0))
    record_generation(
        run_dir,
        2,
        1.2,
        (0.1, 0.2, 0.3),
        sigma=0.18,
        mean=(0.4, 0.4, 0.4),
        samples=((0.1, 0.2, 0.3), (0.9, 0.1, 0.2)),
        withdrawn_share=0.02,
        payment_cap_bind_rate=0.25,
        closure_bind_rate=0.0,
        seed=8,
    )
    write_run_diagnostics(run_dir, run_id="m27-diag", seed_objectives=((7, 1.5), (8, 1.2)))
    for name in (
        "parameter_sweeps.json",
        "withdrawn_share.json",
        "bind_rates.json",
        "seed_variance.json",
        "parameters_at_bounds.json",
        "sigma_history.json",
        "summary.json",
    ):
        assert (run_dir / name).is_file()
    text = format_summary(run_dir)
    assert "run_id=m27-diag" in text
    assert "generations=2" in text
    assert "parameters_at_bounds=1" in text
    assert seen == [1]
    assert summary_main([str(run_dir)]) == 0


def test_log_metrics_use_withdrawal_and_bind_events() -> None:
    log = [
        WithdrawalRequested(t=0.0, seed=1, depositor_id="d1", amount=10.0, channel="app"),
        WithdrawalRequested(t=1.0, seed=1, depositor_id="d1", amount=5.0, channel="app"),
        WithdrawalRequested(t=2.0, seed=1, depositor_id="d2", amount=1.0, channel="wire"),
        PaymentSettled(t=3.0, seed=1, amount=10.0, channel="app", queued_remaining=2),
        PaymentSettled(t=4.0, seed=1, amount=1.0, channel="wire", queued_remaining=0),
        BankClosed(t=5.0, seed=1, reason="cash"),
    ]
    assert withdrawn_share_from_log(log, depositor_count=4) == 0.5
    binds = bind_rates_from_log(log)
    assert binds.payment_cap_bind_rate == 0.5
    assert binds.closure_bind_rate == 1.0
    assert binds.closed is True


def test_unrelated_inputs_are_irrelevant_and_leave_log_identical() -> None:
    for text in IRRELEVANT_PROBES:
        label = classify_rumor_market(text)
        assert is_irrelevant_input(text)
        assert label.category == "irrelevant"
        assert not label.affects_simulation
        assert label.read_as == IRRELEVANT_READ_AS
        assert gated_rumors_for_text(text) == []
    scenario = _default_svb_scenario(_SEED)
    checks = validate_exact_zero_invariance(scenario, steps=_STEPS)
    assert invariance_passed(checks)
    baseline = event_log_hash(run_scenario(scenario, _STEPS))
    for check in checks:
        assert check.baseline_hash == baseline
        assert check.probe_hash == baseline


def test_same_seed_svb_claim_still_uses_shared_classifier() -> None:
    claim = _default_svb_scenario(_SEED).rumors[0].claim
    label = calibration_rumor_classification(claim)
    assert label.category == "market"
    assert label.tone == "fear"
    first = event_log_hash(run_scenario(_default_svb_scenario(_SEED), _STEPS))
    second = event_log_hash(run_scenario(_default_svb_scenario(_SEED), _STEPS))
    assert first == second


def test_nonbank_bound_penalty_and_validation() -> None:
    bound = nonbank_request_bound()
    assert bound == 0.001
    row = get_parameter("Non-bank daily deposit-request bound")
    assert row.status == "ASSUMED"
    assert nonbank_request_bound_penalty(0.0005) == 0.0
    assert nonbank_request_bound_penalty(bound) == 0.0
    assert nonbank_request_bound_penalty(0.002) == 1.0
    assert validate_nonbank_bound(bound)
    assert not validate_nonbank_bound(0.002)
    over = [
        WithdrawalRequested(
            t=10.0,
            seed=1,
            depositor_id="d1",
            amount=2.0,
            channel="app",
        )
    ]
    fraction = daily_deposit_request_fraction(over)
    assert fraction < bound
    assert validate_nonbank_bound(fraction)


def test_daily_request_fraction_groups_by_day() -> None:
    log = [
        WithdrawalRequested(t=10.0, seed=1, depositor_id="a", amount=100.0, channel="app"),
        WithdrawalRequested(t=1450.0, seed=1, depositor_id="b", amount=50.0, channel="app"),
    ]
    # Monkeypatch by computing the formula against the registry deposits.
    deposits = 175_000_000_000.0
    observed = daily_deposit_request_fraction(log)
    assert observed == 100.0 / deposits
    assert validate_nonbank_bound(observed)


def test_gamestop_market_fear_stays_inside_nonbank_bound() -> None:
    text = "GameStop stock fell"
    label = classify_rumor_market(text)
    assert label.category == "market"
    assert label.tone == "fear"
    assert label.affects_simulation
    quiet = build_nonbank_scenario("equity", _SEED)
    quiet = quiet.model_copy(update={"rumors": []})
    market = quiet.model_copy(update={"rumors": gated_rumors_for_text(text, rumor_id="gme")})
    quiet_log = run_scenario(quiet, _STEPS)
    market_log = run_scenario(market, _STEPS)
    quiet_prices = [event.price for event in quiet_log if isinstance(event, PriceTick)]
    market_prices = [event.price for event in market_log if isinstance(event, PriceTick)]
    assert quiet_prices
    assert market_prices
    assert market_prices != quiet_prices
    fraction = daily_deposit_request_fraction(market_log)
    assert validate_nonbank_bound(fraction)
    assert nonbank_request_bound_penalty(fraction) == 0.0


def test_fake_nonbank_fit_stays_on_pinned_seed() -> None:
    first = fit_nonbank_parameters(11, generations=1, population=4, steps=4)
    second = fit_nonbank_parameters(11, generations=1, population=4, steps=4)
    assert first.vector == second.vector
    assert first.objective == second.objective
    assert first.vector == (
        0.5068385534506369,
        0.7719495080619924,
        0.7449442157171865,
    )
    # SVB-04 (c04ff18) floors each log error at 1e-4, so the 4-tick fake path's zero
    # peak-to-trough now scores at the floor (diag-01, SVB-67):
    #   (ln(0.2639009 / 1e-4))**2 = 62.06538696011822   peak-to-trough
    #   (ln(3.218019 / 1.360721))**2 = 0.7408926623026201   volume multiple
    #   days-to-bottom excluded at 4 ticks; request penalty 0.
    assert first.objective == 62.80627962242084
    stats, fraction = measure_nonbank_observables(
        build_nonbank_scenario("equity", 11),
        steps=4,
    )
    assert stats.n_prints == 4
    assert validate_nonbank_bound(fraction)
    assert np.isfinite(first.objective)
