"""SVB-69: the fakes read only the events since their last tick, with results identical to a full replay."""

from __future__ import annotations

import hashlib
import json

import pytest

from contracts.events import LiquidityDrawn, PaymentSettled, PolicyEnacted
from engine.fitted_feedback import step_module
from engine.registry import build_engine
from mocks import support
from mocks.l7_market_ui import MarketScreenFake
from tests.test_fakes import panic


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _full_hash(log) -> str:  # the pre-SVB-69 M33 implementation
    rows = [event.model_dump(mode="json", by_alias=True) for event in log]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _full_fold(events, deposits: float, cet1: float) -> support.Book:  # the pre-SVB-69 fold_book
    return support.fold_book(tuple(events), deposits=deposits, opening_cet1=cet1)  # tuples always start over


def _tick(engine, t: float) -> None:
    for module_id in engine.order:  # as Engine.run steps each minute
        step_module(engine, module_id, t)


def test_m33_hash_equals_a_full_replay_every_tick() -> None:
    engine = build_engine(scenario=panic(7))
    m33 = engine["M33"]
    for t in range(60):
        _tick(engine, float(t))
        assert m33.event_count == len(engine.bus.log)
        assert m33.log_hash == _full_hash(engine.bus.log)
    assert len(engine.bus.log) > 100


def test_m33_dumps_only_new_events(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = build_engine(scenario=panic(7))
    m33 = engine["M33"]
    for t in range(20):
        _tick(engine, float(t))
    import mocks.l8_calibration as l8

    dumped = []
    before = len(engine.bus.log)

    class _CountingJson:
        @staticmethod
        def dumps(*args, **kwargs):
            dumped.append(1)
            return json.dumps(*args, **kwargs)

    monkeypatch.setattr(l8, "json", _CountingJson)
    _tick(engine, 20.0)
    new = len(engine.bus.log) - before
    assert new > 1 and len(dumped) == new  # one dump per new event (the old code dumped the whole log in one call)
    assert m33.log_hash == _full_hash(engine.bus.log)


def test_m33_starts_over_when_the_log_is_replaced() -> None:
    engine = build_engine(scenario=panic(7))
    m33 = engine["M33"]
    _tick(engine, 0.0)
    engine.bus.log[:] = engine.bus.log[:3]  # shorter than what M33 has hashed
    m33.on_tick(1.0, [])
    assert m33.log_hash == _full_hash(engine.bus.log) and m33.event_count == 3
    m33.bus.log = list(engine.bus.log)  # a different list object with the same events
    m33.on_tick(2.0, [])
    assert m33.log_hash == _full_hash(engine.bus.log)


def test_fold_book_resumes_exactly_and_starts_over_when_it_must() -> None:
    log: list = []
    for i in range(50):
        log.append(PaymentSettled(t=float(i), seed=1, bank_id="b", amount=1000.0 + i, channel="wire", queued_remaining=0)
                   if i % 3 else LiquidityDrawn(t=float(i), seed=1, bank_id="b", source="repo", amount=500.0 * i, eta=1.0))
        assert support.fold_book(log, deposits=1e6, opening_cet1=0.1) == _full_fold(log, 1e6, 0.1)
    assert support.fold_book(log, deposits=2e6, opening_cet1=0.1) == _full_fold(log, 2e6, 0.1)  # other inputs
    del log[10:]
    assert support.fold_book(log, deposits=1e6, opening_cet1=0.1) == _full_fold(log, 1e6, 0.1)  # shrank
    log[9] = PaymentSettled(t=9.0, seed=1, bank_id="b", amount=7.0, channel="wire", queued_remaining=0)
    assert support.fold_book(log, deposits=1e6, opening_cet1=0.1) == _full_fold(log, 1e6, 0.1)  # replaced
    log.append(PolicyEnacted(t=11.0, seed=1, bank_id="b", lever="emergency_lending", scope="b", cost_model={}))
    log.append(PolicyEnacted(t=12.0, seed=1, bank_id="b", lever="emergency_lending", scope="b", cost_model={}))
    assert support.levers_in(log) == ("emergency_lending",) == support.levers_in(tuple(log))


def test_m30_series_match_a_rebuild() -> None:
    engine = build_engine(scenario=panic(7))
    for t in range(40):
        _tick(engine, float(t))
    live = engine["M30"]
    fresh = MarketScreenFake("M30-check")
    fresh.bus = engine.bus
    fresh.on_tick(40.0, [])
    for name in ("prices", "volumes", "depth", "markers", "halt", "statements", "futures"):
        assert getattr(live, name) == getattr(fresh, name), name
    assert live.prices
