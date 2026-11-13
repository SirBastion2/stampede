"""SVB-06: withdrawal price impact and news-attention feedback."""

from __future__ import annotations

import math

import numpy as np

from calibration.fit import (
    _DAY1_CLOSE,
    _log_sq_error,
    _objective_from_fractions,
    _score_day1_stock_ratio,
    identifiable_names,
    load_calibration_targets,
)
from params.loader import get_parameter
from contracts.events import PriceTick, WithdrawalRequested
from engine.gpu_full_io import (
    apply_price_impact,
    compose_attention,
    feedback_params_from_vector,
    price_impact_move,
    step_news,
)
from engine.gpu_sim import relative_error
from mocks.support import OPENING_PRICE


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch, n_depositors: int = 48) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(n_depositors))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _registry_float(name: str) -> float:
    return float(get_parameter(name).value)


def _vector(*, feedback: float = 0.5, dispersion: float = 0.5, impact: float = 0.5) -> np.ndarray:
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    values = np.full(dim, 0.5, dtype=np.float64)
    values[12] = float(feedback)
    values[13] = float(dispersion)
    values[14] = float(impact)
    return values


def test_identifiable_stage1_slots() -> None:
    assert identifiable_names() == (
        "worry_bias",
        "mobile_share",
        "decision_to_request_rate",
    )


def test_stage1_objective_includes_stock_term() -> None:
    targets = load_calibration_targets()
    base = _objective_from_fractions(
        0.24,
        0.81,
        targets=targets,
        w_day1=1.0,
        w_day2=1.0,
    )
    hit = _objective_from_fractions(
        0.24,
        0.81,
        targets=targets,
        w_day1=1.0,
        w_day2=1.0,
        stock_ratio=0.40,
    )
    miss = _objective_from_fractions(
        0.24,
        0.81,
        targets=targets,
        w_day1=1.0,
        w_day2=1.0,
        stock_ratio=1.0,
    )
    # SVB-36: the stock term carries its registry weight (0 by default: reported, not fitted).
    weight = _registry_float("Calibration E1 day-1 stock objective weight")
    assert hit == base
    assert miss == base + weight * _log_sq_error(1.0, 0.40)
    log = [
        PriceTick(t=0.0, seed=0, price=OPENING_PRICE, volume=1.0),
        PriceTick(t=_DAY1_CLOSE - 1.0, seed=0, price=0.40 * OPENING_PRICE, volume=1.0),
        PriceTick(t=_DAY1_CLOSE, seed=0, price=0.10 * OPENING_PRICE, volume=1.0),
    ]
    assert _score_day1_stock_ratio(log) == 0.40


def test_zero_feedback_price_only_moves_from_impact() -> None:
    share = np.array([0.04, 0.00], dtype=np.float64)
    quoted = np.array([OPENING_PRICE, OPENING_PRICE], dtype=np.float64)
    no_impact = apply_price_impact(np, quoted, share, 0.0, dt=1.0, noise=0.0)
    with_impact = apply_price_impact(np, quoted, share, 0.8, dt=1.0, noise=0.0)
    assert no_impact[0] == OPENING_PRICE
    assert no_impact[1] == OPENING_PRICE
    assert with_impact[1] == OPENING_PRICE
    assert with_impact[0] < OPENING_PRICE
    news0 = step_news(np, np.array([0.2, 0.0]), share, 0.0, dt=1.0)
    assert news0[0] == 0.2 * 0.5 ** (1.0 / 60.0)
    assert news0[1] == 0.0


def test_stock_never_falls_more_than_one_percent_per_minute() -> None:
    cap = _registry_float("M14 price impact cap per minute")
    assert cap == 0.01
    shares = np.array([0.0, 0.02, 0.5, 4.0], dtype=np.float64)
    quoted = np.full(shares.shape, 100.0)
    for dt in (1.0, 5.0, 15.0):
        move = price_impact_move(np, quoted, shares, 10.0, dt, noise=0.0)
        assert np.all(move <= cap * dt + 1e-15)
        assert np.all(move >= -cap * dt - 1e-15)
        price = apply_price_impact(np, quoted, shares, 10.0, dt, noise=0.0)
        drop = 1.0 - price / 100.0
        assert np.all(drop <= 1.0 - math.exp(-cap * dt) + 1e-12)


def test_feedback_terms_dt_invariant() -> None:
    # Level targeting: the same cumulative share is the same target. Five
    # one-minute steps and one five-minute step both close the same gap
    # when the remaining drop is inside cap * 5.
    cum_share = 0.03
    impact = 0.7
    one = np.array([100.0], dtype=np.float64)
    for _ in range(5):
        one = apply_price_impact(np, one, cum_share, impact, dt=1.0, noise=0.0)
    five = apply_price_impact(np, np.array([100.0]), cum_share, impact, dt=5.0, noise=0.0)
    assert relative_error(float(one[0]), float(five[0])) <= 0.02

    news0 = 0.8
    decay_one = news0
    for _ in range(5):
        decay_one = float(step_news(np, np.array([decay_one]), 0.0, 1.0, dt=1.0)[0])
    decay_five = float(step_news(np, np.array([news0]), 0.0, 1.0, dt=5.0)[0])
    assert relative_error(decay_one, decay_five) <= 0.02

    start = 0.1
    news_one = start
    for _ in range(5):
        news_one = float(step_news(np, np.array([news_one]), cum_share / 5.0, 0.4, dt=1.0)[0])
    news_five = float(step_news(np, np.array([start]), cum_share, 0.4, dt=5.0)[0])
    assert relative_error(news_one, news_five) <= 0.02
    attn_one = float(compose_attention(np, 0.2, news_one))
    attn_five = float(compose_attention(np, 0.2, news_five))
    assert relative_error(attn_one, attn_five) <= 0.02


def _requested_share(log: list[object], deposits: float) -> float:
    total = sum(float(event.amount) for event in log if isinstance(event, WithdrawalRequested))
    return total / deposits if deposits else 0.0


def _prices(log: list[object]) -> list[tuple[float, float]]:
    return [
        (float(event.t), float(event.price))
        for event in log
        if isinstance(event, PriceTick)
    ]


def test_higher_feedback_raises_requests_monotone(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch, n_depositors=48)
    from calibration.fit import _default_svb_scenario
    from engine.agents.population import clear_sample_population_cache
    from engine.gpu_sim import clear_static_pack_cache, simulate_logs
    from mocks.params_access import get_param

    clear_sample_population_cache()
    clear_static_pack_cache()
    scenario = _default_svb_scenario(17)
    slots = (0.25, 0.50, 0.80)
    vectors = [_vector(feedback=slot, impact=0.5) for slot in slots]
    logs = simulate_logs(
        scenario,
        390,
        vectors,
        array_module="numpy",
        disable_caps=True,
        softness=0.06,
        clamp_inputs=False,
    )
    deposits = float(get_param("SVB-like preset: deposits"))
    shares = [_requested_share(log, deposits) for log in logs]
    assert shares[0] <= shares[1] + 1e-8
    assert shares[1] <= shares[2] + 1e-8
    assert shares[2] > shares[0]
    clear_static_pack_cache()
    clear_sample_population_cache()


def test_placebo_95_insured_stays_below_2pct_at_3x_feedback(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch, n_depositors=64)
    import engine.agents.population as population
    from calibration.fit import _default_svb_scenario
    from contracts.schemas import DepositorArrays
    from engine.agents.population import clear_sample_population_cache
    from engine.gpu_sim import clear_static_pack_cache, simulate_logs
    from mocks.params_access import get_param

    original = population.sample_population

    def _ninety_five_insured(scenario, **kwargs):
        arrays, summary = original(scenario, **kwargs)
        balance = list(arrays.balance)
        insured = [0.95 * amount for amount in balance]
        forced = DepositorArrays(
            bank_idx=list(arrays.bank_idx),
            segment=list(arrays.segment),
            balance=balance,
            insured=insured,
            belief=list(arrays.belief),
            threshold=list(arrays.threshold),
            attention=list(arrays.attention),
            needs_flag=list(arrays.needs_flag),
            has_second_bank=list(arrays.has_second_bank),
            state=list(arrays.state),
        )
        return forced, summary

    monkeypatch.setattr(population, "sample_population", _ninety_five_insured)
    clear_sample_population_cache()
    clear_static_pack_cache()
    scenario = _default_svb_scenario(5).model_copy(update={"rumors": []})
    triple = 0.5 + math.log(3.0) / 0.8
    vector = _vector(feedback=triple, impact=0.5)
    gain, _disp, _impact = feedback_params_from_vector(vector)
    prior = float(get_param("M03 feedback gain (ASSUMED)"))
    assert abs(gain / prior - 3.0) < 1e-12
    log = simulate_logs(
        scenario,
        30,
        [vector],
        array_module="numpy",
        disable_caps=True,
        softness=0.06,
        clamp_inputs=False,
    )[0]
    deposits = float(get_param("SVB-like preset: deposits"))
    assert _requested_share(log, deposits) < 0.02
    prices = _prices(log)
    for index in range(1, len(prices)):
        prev_t, prev_px = prices[index - 1]
        t, price = prices[index]
        dt = max(t - prev_t, 1.0)
        if prev_px <= 0.0:
            continue
        assert (prev_px - price) / prev_px <= 0.01 * dt + 1e-9
    clear_static_pack_cache()
    clear_sample_population_cache()
