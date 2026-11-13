"""SVB-58 (item 6, option B): live depositors ignore the order-book price (ADR-104)."""

from __future__ import annotations

import pytest

import engine.live_overrides as overrides
from engine.agents.decide import DecisionModule
from engine.live_overrides import flag_enabled, live_flags
from params.loader import RegistryError, get_parameter

ROW = "M06 depositors ignore stock price"
LIVE_ROW = "M29 live depositors ignore stock price"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def test_rows_are_assumed_engine_off_live_on() -> None:
    assert get_parameter(ROW).value == "false" and get_parameter(ROW).status == "ASSUMED"
    assert get_parameter(LIVE_ROW).value == "true"
    assert flag_enabled(ROW) is False
    assert DecisionModule("M06").ignore_price is False


def test_live_context_turns_it_on_only_inside() -> None:
    with live_flags({ROW}):
        assert DecisionModule("M06").ignore_price is True
    assert DecisionModule("M06").ignore_price is False


def test_live_real_session_ignores_the_order_book_price(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    session.start()
    session.submit_rumor("Kestrel Bank is out of cash and may delay uninsured wires", "vc-partner", "bank-fvb", now=0.0)
    for _ in range(8):
        session.step_once()
    m06, m14 = session.engine["M06"], session.engine["M14"]
    assert m06.ignore_price is True
    assert m14.price < m14.params.opening_price
    assert m06.stock_return == 0.0


def test_calibration_engine_keeps_the_price_feature() -> None:
    from calibration.fit import _default_svb_scenario, real_calibration_modes
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    engine = build_engine(modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(11))
    assert engine["M06"].ignore_price is False


def test_registry_on_is_refused_under_gpu_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setattr(overrides, "registry_flag", lambda row: row == ROW)
    with pytest.raises(RegistryError):
        DecisionModule("M06")
