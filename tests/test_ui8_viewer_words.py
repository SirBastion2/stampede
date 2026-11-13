"""UI-8: viewer-facing words. One vocabulary ("withdrawal requests"), plans as sentences, the answer revealed by
beat 2's replay, the finale explaining each plan, margins in dollars, one set of play controls per beat."""

from __future__ import annotations

from pathlib import Path

PAGE = Path("ui/live/stampede_ui.html")


def _html() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_plans_read_as_sentences_with_one_word_for_withdrawals() -> None:
    html = _html()
    assert "`once withdrawal requests reach ${pc(w[1])} of deposits`" in html
    assert ".join(tight ? ' + ' : ', then ')" in html
    assert "' requested'}" not in html and "% requested'" not in html            # the engine's word stays off screen
    assert "(withdrawal requests at ${Math.round(T.level * 100)}% of deposits)" in html


def test_the_finale_explains_why_the_winner_still_shows_82_percent() -> None:
    html = _html()
    assert "function howItWorks(p)" in html
    assert "Customers still ask for ${share}; the bank has the money to pay" in html
    assert "The guarantee stops the run at ${share}, but needs the government" in html


def test_margins_are_in_dollars_from_the_servers_deposit_total() -> None:
    html = _html()
    assert "function spare(D, m)" in html and "fmtUSD(m * D.deposits_usd)" in html
    assert "percentage points of deposits" not in html
    server = Path("ui/server.py").read_text(encoding="utf-8")
    assert '"deposits_usd": float(registry_deposits_total())' in server


def test_the_panel_ends_on_the_plans_that_kept_the_bank_open() -> None:
    html = _html()
    assert "`Kept the bank open, of ${N} days`" in html
    assert "other plans dropped out:" in html and "Useful on how many of the" not in html
    assert "more plan${moreIn === 1 ? '' : 's'} still in" in html


def test_one_set_of_play_controls_and_withheld_plans_in_the_evidence_fold() -> None:
    html = _html()
    assert "body[data-screen=search] .clockbox .lead" in html and "body[data-screen=why] .clockbox .lead" in html
    assert "plans await more simulation and are excluded from ranking" not in html
    assert "plans need their own simulations before they can be ranked" in html


def test_beat_one_money_words() -> None:
    html = _html()
    assert "sim.cash < 0 ? 'Out of cash' : fmtUSD(sim.cash)" in html
    assert "'Cash and bonds on hand'" in html and "Cash + securities (book value)" not in html
    assert "Customers asked to withdraw ${pct(share)} of deposits." in html


def test_the_badges_say_calibrated_and_what_if_without_a_real_bank_name() -> None:
    from ui.live.labels import FITTED_BADGE, SCENARIO_BADGE

    assert FITTED_BADGE == "Calibrated to a real 2023 bank run"
    assert SCENARIO_BADGE == "What-if on the same calibrated model"
    assert "SVB" not in FITTED_BADGE + SCENARIO_BADGE


def test_search_margin_total_comes_from_registry_without_starting_a_session(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from engine.bank import ledger
    from ui.live import search
    from ui.server import app

    monkeypatch.setattr(search, "available_run_options", lambda: [10, 25, 50])
    metadata = {"runs_per_plan": 25, "series": {"hidden": []}, "pick": "saved-pick"}
    monkeypatch.setattr(search, "load_search", lambda runs=None: metadata)
    monkeypatch.setattr(ledger, "registry_deposits_total", lambda: 234_000_000_000)
    # No TestClient lifespan: serving saved metadata must not require an engine.
    response = TestClient(app).get("/api/search", params={"runs": 25})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "runs_per_plan": 25, "pick": "saved-pick",
                               "run_options": [10, 25, 50], "deposits_usd": 234_000_000_000.0}
    assert metadata["series"] == {"hidden": []}
