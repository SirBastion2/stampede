"""SVB-56: SVB-like M08 opening book (SOURCED rows), off for calibration, on for the live demo."""

from __future__ import annotations

import pytest

from engine.bank.ledger import (
    LedgerModule,
    initial_ledger_state,
    invariant_holds,
    opening_book_fractions,
    registry_deposits_total,
    svb_like_opening_book,
    svb_opening_book_active,
)
from params.loader import get_parameter

CASH, SECURITIES = 0.0797359, 0.678128


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def test_rows_are_sourced_with_the_10k_citation_and_the_flag_is_off() -> None:
    for name, value in (
        ("M08 SVB-like opening cash fraction of deposits", CASH),
        ("M08 SVB-like opening securities fraction of deposits", SECURITIES),
    ):
        row = get_parameter(name)
        assert row.status == "SOURCED"
        assert float(row.value) == pytest.approx(value)
        assert "0000719739-23-000021" in row.source
    assert get_parameter("M08 SVB-like opening book enabled").value == "false"
    assert get_parameter("M29 live SVB-like opening book").value == "true"
    assert svb_opening_book_active() is False


def test_cited_figures_give_the_row_values() -> None:
    assert 13_803 / 173_109 == pytest.approx(CASH, abs=5e-8)
    assert (26_069 + 91_321) / 173_109 == pytest.approx(SECURITIES, abs=5e-7)


def test_default_opening_book_is_unchanged() -> None:
    deposits = registry_deposits_total()
    assert opening_book_fractions() == (0.001, 0.004)
    state = initial_ledger_state(deposits=deposits)
    assert state.cash == pytest.approx(0.001 * deposits)
    assert state.securities == pytest.approx(0.004 * deposits)


def test_context_builds_the_svb_like_book() -> None:
    deposits = registry_deposits_total()
    with svb_like_opening_book():
        assert svb_opening_book_active()
        cash_f, sec_f = opening_book_fractions()
        state = initial_ledger_state(deposits=deposits, cash_fraction=cash_f, securities_fraction=sec_f)
    assert state.cash == pytest.approx(CASH * deposits)
    assert state.securities == pytest.approx(SECURITIES * deposits)
    assert state.loans > 0.0
    assert invariant_holds(state)
    assert not svb_opening_book_active()


def test_calibration_engine_keeps_the_old_book() -> None:
    from calibration.fit import _default_svb_scenario, real_calibration_modes
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    engine = build_engine(
        modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(11)
    )
    ledger = engine["M08"]
    assert isinstance(ledger, LedgerModule)
    assert ledger.state.cash == pytest.approx(0.001 * registry_deposits_total())


def test_live_real_mode_bank_opens_with_the_svb_like_book(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    state = session.engine["M08"].state
    deposits = registry_deposits_total()
    assert state.cash == pytest.approx(CASH * deposits)
    assert state.securities == pytest.approx(SECURITIES * deposits)
    assert not svb_opening_book_active()


def test_live_fake_mode_is_unaffected() -> None:
    from ui.live.session import LiveSession

    session = LiveSession(seed=42)
    assert session.follows_engine is False
    assert not svb_opening_book_active()
