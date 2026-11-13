"""M14 order book: price-time match, spreads, halt, determinism, contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from contracts import parse_event
from contracts.events import (
    AnalystNote,
    AttentionTick,
    CapitalChanged,
    HeadlinePublished,
    PriceTick,
    RatesChanged,
    ThesisPublished,
    TradingHalt,
)
from contracts.schemas import Scenario
from engine.markets.book import (
    KIND_SHORT,
    OrderBook,
    _match_price_time_py,
    halt_triggered,
    load_book_params,
    market_maker_half_spread,
    match_price_time,
    public_fair,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.bus import EventBus
from mocks.l3_markets import MarketFake
from params.loader import get_parameter
from tests.test_fakes import panic


def _scenario(seed: int = 3) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed)


def _book(seed: int = 3) -> OrderBook:
    book = OrderBook("M14")
    book.attach(EventBus(), _scenario(seed))
    return book


def test_halt_params_are_marked_assumed() -> None:
    for name in (
        "M14 halt price move threshold (X%)",
        "M14 halt window (Y minutes)",
    ):
        row = get_parameter(name)
        assert row.status == "ASSUMED"
        assert row.module == "M14"
    # The unspecified TBD rows stay in the registry and are not the source.
    assert get_parameter("Trading halt: price move threshold (X%)").value == "TBD"
    assert get_parameter("Trading halt: window (Y minutes)").value == "TBD"


def test_market_makers_widen_spreads_as_volatility_rises() -> None:
    params = load_book_params()
    calm = market_maker_half_spread(0.0, params.mm_base_half_spread, params.mm_vol_multiplier)
    stressed = market_maker_half_spread(
        0.05, params.mm_base_half_spread, params.mm_vol_multiplier
    )
    assert stressed > calm

    book = _book()
    book.step(0.0)
    calm_spread = book.half_spread
    book.inbox.append(
        AttentionTick(
            t=1.0,
            seed=book.seed,
            bank_id=book.bank_id,
            level=1.0,
            delta_4h=0.4,
        )
    )
    book.step(1.0)
    book.step(2.0)
    assert book.half_spread > calm_spread


def test_halt_rule_is_two_sided_and_windowed() -> None:
    params = load_book_params()
    window = params.halt_window_min
    move = params.halt_move
    times = np.array([0.0, window], dtype=np.float64)
    inside = np.array([100.0, 100.0 * (1.0 - move / 2.0)], dtype=np.float64)
    down = np.array([100.0, 100.0 * (1.0 - move * 1.5)], dtype=np.float64)
    up = np.array([100.0, 100.0 * (1.0 + move * 1.5)], dtype=np.float64)
    assert not halt_triggered(times, inside, window, move, window)
    assert halt_triggered(times, down, window, move, window)
    assert halt_triggered(times, up, window, move, window)

    # A crash older than Y minutes is outside the window.
    aged_times = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    aged_prices = np.array([100.0, 50.0, 50.0], dtype=np.float64)
    assert not halt_triggered(aged_times, aged_prices, 20.0, move, window)
    assert halt_triggered(aged_times, aged_prices, 10.0, move, window)


def test_price_time_priority_and_numba_matches_python() -> None:
    # Better price fills before an earlier worse bid. Trade price is the earlier order.
    side = np.array([1, 1, -1], dtype=np.int64)
    price = np.array([10.0, 12.0, 11.0], dtype=np.float64)
    qty = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    time_key = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    filled, notional, vwap, volume = match_price_time(side, price, qty, time_key)
    py_filled, py_notional, py_vwap, py_volume = _match_price_time_py(
        side, price, qty, time_key
    )
    assert volume == pytest.approx(5.0)
    assert vwap == pytest.approx(12.0)
    assert filled[0] == pytest.approx(0.0)
    assert filled[1] == pytest.approx(5.0)
    assert filled[2] == pytest.approx(5.0)
    assert notional[1] == pytest.approx(60.0)
    assert np.allclose(filled, py_filled)
    assert np.allclose(notional, py_notional)
    assert vwap == pytest.approx(py_vwap)
    assert volume == pytest.approx(py_volume)

    # Same price: the earlier bid is filled first.
    tie_side = np.array([1, 1, -1], dtype=np.int64)
    tie_price = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    tie_qty = np.array([5.0, 5.0, 7.0], dtype=np.float64)
    tie_time = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tie_filled, _, _, tie_volume = match_price_time(tie_side, tie_price, tie_qty, tie_time)
    assert tie_volume == pytest.approx(7.0)
    assert tie_filled[0] == pytest.approx(5.0)
    assert tie_filled[1] == pytest.approx(2.0)

    # The resting (earlier) ask sets the print.
    ask_side = np.array([-1, 1], dtype=np.int64)
    ask_price = np.array([9.0, 11.0], dtype=np.float64)
    ask_qty = np.array([4.0, 4.0], dtype=np.float64)
    ask_time = np.array([1.0, 2.0], dtype=np.float64)
    _, _, ask_vwap, ask_volume = match_price_time(ask_side, ask_price, ask_qty, ask_time)
    assert ask_volume == pytest.approx(4.0)
    assert ask_vwap == pytest.approx(9.0)

    again, _, again_vwap, again_volume = match_price_time(side, price, qty, time_key)
    assert np.allclose(filled, again)
    assert again_vwap == pytest.approx(vwap)
    assert again_volume == pytest.approx(volume)


def test_attention_and_rates_lower_the_print() -> None:
    params = load_book_params()
    calm_fair = public_fair(
        params.opening_price,
        params.opening_cet1,
        params.opening_cet1,
        params.opening_policy_rate,
        params.opening_policy_rate,
        params.bond_duration,
        0.0,
        0.0,
        0.0,
        params.attention_premium,
    )
    hot_fair = public_fair(
        params.opening_price,
        params.opening_cet1,
        params.opening_cet1,
        params.opening_policy_rate,
        params.opening_policy_rate,
        params.bond_duration,
        1.0,
        0.4,
        0.0,
        params.attention_premium,
    )
    assert hot_fair < calm_fair

    calm = _book()
    calm.step(0.0)
    hot = _book()
    hot.inbox.append(
        AttentionTick(t=0.0, seed=hot.seed, bank_id=hot.bank_id, level=1.0, delta_4h=0.4)
    )
    hot.step(0.0)
    assert hot.price < calm.price

    rated = _book()
    rated.inbox.append(
        RatesChanged(
            t=0.0,
            seed=rated.seed,
            curve=[0.07, 0.075],
            policy_rate=params.opening_policy_rate + 0.02,
        )
    )
    rated.step(0.0)
    assert rated.curve == (0.07, 0.075)
    assert rated.price < params.opening_price


def test_halt_fires_on_a_capital_shock_and_then_holds() -> None:
    params = load_book_params()
    book = _book()
    book.inbox.append(
        CapitalChanged(t=0.0, seed=book.seed, bank_id=book.bank_id, cet1_ratio=0.09)
    )
    # A different bank does not move this book.
    book.inbox.append(
        CapitalChanged(t=0.0, seed=book.seed, bank_id="bank-other", cet1_ratio=0.01)
    )
    events = book.step(0.0)
    assert book.cet1 == pytest.approx(0.09)
    ticks = [event for event in events if isinstance(event, PriceTick)]
    halts = [event for event in events if isinstance(event, TradingHalt)]
    assert len(ticks) == 1
    assert ticks[0].price < params.opening_price * (1.0 - params.halt_move)
    assert ticks[0].volume > 0.0
    assert len(halts) == 1
    assert halts[0].reason == "volatility"
    assert halts[0].duration == pytest.approx(params.halt_duration_min)
    assert halts[0].bank_id == book.bank_id
    parsed = parse_event(halts[0].model_dump(mode="json"))
    assert isinstance(parsed, TradingHalt)

    held = book.step(1.0)
    assert len(held) == 1
    assert isinstance(held[0], PriceTick)
    assert held[0].volume == pytest.approx(0.0)
    assert held[0].price == pytest.approx(ticks[0].price)


def test_short_sellers_stop_at_the_borrow_limit() -> None:
    book = _book()
    book.inbox.append(
        ThesisPublished(
            t=0.0,
            seed=book.seed,
            bank_id=book.bank_id,
            author_id="short-1",
            reach=50_000,
        )
    )
    for minute in range(8):
        book.step(float(minute))
    shorts = book.inventory[book.kinds == KIND_SHORT]
    limit = book.params.short_borrow_limit
    assert shorts.size > 0
    assert np.all(shorts < 0.0)
    # Borrowing availability caps each short. Earlier orders fill first, so
    # not every trader is full, but nobody is past the limit and the cap binds.
    assert np.all(shorts >= -limit - 1e-6)
    assert float(np.min(shorts)) == pytest.approx(-limit)
    quiet = _book(seed=4)
    quiet.step(0.0)
    assert np.allclose(quiet.inventory[quiet.kinds == KIND_SHORT], 0.0)


def test_headlines_and_notes_raise_news_pressure() -> None:
    book = _book()
    book.inbox.append(
        HeadlinePublished(
            t=0.0,
            seed=book.seed,
            bank_id=book.bank_id,
            text="Depositors are asking for cash",
            reach=25_000,
        )
    )
    book.inbox.append(
        AnalystNote(
            t=0.0,
            seed=book.seed,
            bank_id=book.bank_id,
            text="Capital looks thin",
            cited_event_ids=["loss-1"],
        )
    )
    book.step(0.0)
    assert book.news == pytest.approx(0.6)


def test_order_book_steps_are_deterministic() -> None:
    def run_once() -> list[dict[str, Any]]:
        book = _book(seed=11)
        book.inbox.append(
            AttentionTick(
                t=0.0, seed=11, bank_id="bank-fvb", level=0.8, delta_4h=0.2
            )
        )
        book.inbox.append(
            CapitalChanged(t=1.0, seed=11, bank_id="bank-fvb", cet1_ratio=0.10)
        )
        dumped: list[dict[str, Any]] = []
        for minute in range(4):
            for event in book.step(float(minute)):
                dumped.append(event.model_dump(mode="json"))
        return dumped

    assert run_once() == run_once()


def test_real_m14_engine_is_deterministic() -> None:
    scenario = panic(9)
    modes = {"M14": ModuleMode.REAL}
    first = run_scenario(scenario, 8, modes=modes)
    second = run_scenario(scenario, 8, modes=modes)
    assert _dump(first) == _dump(second)


def test_real_vs_fake_price_events_share_contract() -> None:
    scenario = panic(7)
    steps = 12
    fake_log = run_scenario(scenario, steps, modes={"M14": ModuleMode.FAKE})
    real_log = run_scenario(scenario, steps, modes={"M14": ModuleMode.REAL})
    for log in (fake_log, real_log):
        prices = [event for event in log if isinstance(event, PriceTick)]
        assert prices
        for event in prices:
            parsed = parse_event(event.model_dump(mode="json"))
            assert isinstance(parsed, PriceTick)
            assert parsed.price > 0.0
            assert parsed.volume >= 0.0
            assert parsed.bank_id
        for event in log:
            if isinstance(event, TradingHalt):
                parsed_halt = parse_event(event.model_dump(mode="json"))
                assert isinstance(parsed_halt, TradingHalt)
                assert parsed_halt.reason
                assert parsed_halt.duration > 0.0


def test_build_engine_real_m14_replaces_the_fake() -> None:
    engine = build_engine(modes={"M14": ModuleMode.REAL}, scenario=_scenario())
    assert isinstance(engine["M14"], OrderBook)
    assert not isinstance(engine["M14"], MarketFake)
    assert engine["M14"].publishes == ("PriceTick", "TradingHalt")
    assert "PolicyEnacted" not in engine["M14"].consumes
    assert "AnalystNote" in engine["M14"].consumes


def test_default_mode_stays_fake() -> None:
    engine = build_engine(scenario=_scenario())
    assert isinstance(engine["M14"], MarketFake)


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json") for event in events]
