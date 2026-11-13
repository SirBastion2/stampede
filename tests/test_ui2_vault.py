"""UI-2: the vault's depositor sample, the bank's borrowing, the response cards and the preset lines."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
import pytest

from agents.guardrails import moderate_rumor
from ui.live.cards import response_cards
from ui.live.presets import LIVE_PRESETS, preset_table

BANNED = re.compile(r"predict|\bSVB\b|Silicon Valley", re.IGNORECASE)


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _real(monkeypatch: pytest.MonkeyPatch, seed: int = 1):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    return LiveSession(seed=seed)


def _spread(lo: float, hi: float) -> dict:
    return {"min": lo, "mean": (lo + hi) / 2, "max": hi}


BATCH = {"summary": [  # diag-17's measured ranges (10 seeds); the debt is illustrative
    dict(preset="lost-bonds", response=None, runs=10, closed_runs=10, requested_1440=_spread(0.8192, 0.8211), cash_out_runs=10, cash_out_minute=_spread(960, 991)),
    dict(preset="lost-bonds", response="guarantee", runs=10, closed_runs=0, requested_1440=_spread(0.0337, 0.0503), cash_out_runs=0, cash_out_minute=None),
    dict(preset="lost-bonds", response="credit-line", runs=10, closed_runs=10, requested_1440=_spread(0.8192, 0.8211), cash_out_runs=0, cash_out_minute=None, emergency_usd=_spread(8.0e9, 8.75e9)),
    dict(preset="lost-bonds", response="withdrawal-limit", runs=10, closed_runs=10, requested_1440=_spread(0.272, 0.274), cash_out_runs=0, cash_out_minute=None),
]}


def test_sample_is_real_depositors_fixed_for_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    assert session.vault_layout() == {"ready": False}  # M06 fills its arrays on its first step
    session.load_preset("lost-bonds")
    session.step_once()
    layout = session.vault_layout()
    assert layout["ready"] and layout["n"] == 10_000 and layout["n_unins"] == 7_000 and layout["top"] == 400
    m06 = session.engine["M06"]
    deposits = np.asarray(m06.insured) + np.asarray(m06.initial_uninsured)
    idx = session._ui_sample["idx"]
    assert np.all(np.asarray(m06.initial_uninsured)[idx[:7000]] > 0) and np.all(np.asarray(m06.initial_uninsured)[idx[7000:]] <= 0)
    biggest = np.sort(deposits[np.asarray(m06.initial_uninsured) > 0])[-400:]
    assert np.allclose(np.sort(deposits[idx[:400]]), biggest)  # the 400 biggest with uninsured money
    again = _real(monkeypatch); again.load_preset("lost-bonds"); again.step_once(); again.vault_layout()
    assert np.array_equal(again._ui_sample["idx"], idx)  # same seed, same dots


def test_fracs_are_each_depositors_own_share_and_add_up(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    for _ in range(120):
        session.step_once()
    u = session.ui_state()
    frac = np.frombuffer(base64.b64decode(u["fracs"]), dtype=np.uint8)
    assert frac.size == 10_000 and frac[7000:].max() == 0  # fully insured depositors never ask
    m06 = session.engine["M06"]
    asked = np.asarray(getattr(m06, "mobile_share", 1.0)) * np.asarray(m06.initial_uninsured) * np.asarray(m06.decided)
    # the dots' quantity is the engine's total (request events carry whole-dollar amounts: a $1 gap)
    assert abs(asked.sum() - u["money"]["requested_usd"]) <= 10.0


def test_borrowing_is_scanned(monkeypatch: pytest.MonkeyPatch) -> None:
    from contracts.events import LiquidityDrawn

    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    session.step_once()
    for source, amount in (("emergency", 2e9), ("repo", 1e9)):
        session.engine.bus.publish(LiquidityDrawn(t=1.0, seed=1, bank_id="bank-fvb", source=source, amount=amount, eta=0.25))
    session.step_once()
    money = session.ui_state()["money"]
    assert money["emergency_usd"] >= 2e9 and money["borrowed_usd"] >= 3e9


def test_cards_from_the_batch_and_the_cost_model() -> None:
    cards = response_cards(BATCH)
    assert list(cards) == ["guarantee", "credit-line", "withdrawal-limit"] and all(c["measured"] for c in cards.values())
    bars = {n: {s["key"]: s["bar"] for s in c["stats"]} for n, c in cards.items()}
    assert bars["guarantee"]["stops"] == 5 and bars["credit-line"]["stops"] == 0 and 3 <= bars["withdrawal-limit"]["stops"] <= 4
    assert bars["guarantee"]["survives"] == 5 and bars["credit-line"]["survives"] == 0
    assert bars["guarantee"]["public"] == 5 and bars["credit-line"]["public"] == 0 and bars["withdrawal-limit"]["public"] == 0
    assert bars["withdrawal-limit"]["hazard"] < bars["credit-line"]["hazard"] < bars["guarantee"]["hazard"] == 5
    assert any("emergency debt" in c for c in cards["credit-line"]["cons"]) and any("Fed" in c for c in cards["credit-line"]["cons"])
    for card in cards.values():
        for text in [card["note"], card["footer"], *card["pros"], *card["cons"], *[s["text"] for s in card["stats"]]]:
            assert not BANNED.search(text) and moderate_rumor(text).allowed, text
    bare = response_cards(None)
    assert not bare["guarantee"]["measured"] and [s["key"] for s in bare["guarantee"]["stats"]] == ["public", "hazard"]


def test_presets_say_what_to_expect() -> None:
    table = {row["name"]: row for row in preset_table()}
    assert all(table[n]["expect"] for n in LIVE_PRESETS)
    for row in table.values():
        for text in (row["expect"], row["note"], row["label"]):
            assert not BANNED.search(text), text


def test_batch_rows_record_the_debt() -> None:
    from ui.live import batch

    row = batch.run_one("lost-bonds", "credit-line", 2, minutes=3)
    assert row["emergency_usd"] >= 0.0 and row["borrowed_usd"] >= 0.0
    summary, = [s for s in batch.summarize([row]) if s["response"] == "credit-line"]
    assert summary["emergency_usd"]["min"] == row["emergency_usd"]


def test_the_page_draws_the_vault() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    for needle in ("/api/ui/depositors", "function drawVault", "ORBIT_W0", "function showCard", "Expected result for this rumor"):
        assert needle in html, needle
    assert "function drawDayChart" not in html   # UI-5: the Stocks screen is removed
    assert "Chase is insolvent" not in html


def test_server_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    client = TestClient(server.app)
    assert "ready" in client.get("/api/ui/depositors").json()
    responses = client.get("/api/presets").json()["responses"]
    assert all(r["card"]["label"] == r["label"] for r in responses)
