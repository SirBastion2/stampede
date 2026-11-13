"""CM-1: sourced values (rows 261, 303, 306) and the cards' public stat as money at risk with the expected loss."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from params.loader import get_parameter

BANNED = re.compile(r"predict|\bSVB\b|Silicon Valley|BTFP|First Republic|FDIC", re.IGNORECASE)


def _spread(lo: float, hi: float) -> dict:
    return {"min": lo, "mean": (lo + hi) / 2, "max": hi}


BATCH = {"summary": [
    dict(preset="lost-bonds", response=None, runs=10, closed_runs=10, requested_1440=_spread(0.8192, 0.8211),
         cash_out_runs=10, cash_out_minute=_spread(960, 991)),
    dict(preset="lost-bonds", response="guarantee", runs=10, closed_runs=0, requested_1440=_spread(0.04, 0.04),
         cash_out_runs=0, cash_out_minute=None),
    dict(preset="lost-bonds", response="credit-line", runs=10, closed_runs=0, requested_1440=_spread(0.8192, 0.8211),
         cash_out_runs=0, cash_out_minute=None, emergency_usd=_spread(1.0e11, 1.05e11)),
    dict(preset="lost-bonds", response="withdrawal-limit", runs=10, closed_runs=10, requested_1440=_spread(0.27, 0.27),
         cash_out_runs=10, cash_out_minute=_spread(1200, 1300)),
]}


def test_rows() -> None:
    dif = get_parameter("M22 DIF loss uninsured fraction (ASSUMED)")
    assert float(dif.value) == 0.104 and dif.status == "SOURCED" and "$15.7B" in dif.source
    g = get_parameter("M06 guarantee uninsured-gap scale (ASSUMED)")
    assert g.value == "0" and g.status == "ASSUMED" and "Martin, Puri & Ufier" in g.source  # M06 loads ASSUMED only
    mh = get_parameter("M22 moral hazard base by lever (ASSUMED)")
    assert mh.status == "ASSUMED" and "Detragiache" in mh.source and mh.value.startswith("guarantee_deposits:0.55")
    spec = json.loads(Path("calibration/registry_updates/cm1_sourced_values.json").read_text(encoding="utf-8"))
    assert set(spec["updates"]) == {dif.parameter, g.parameter, mh.parameter} and not spec.get("append")


def test_the_guarantee_states_money_at_risk_and_the_expected_loss() -> None:
    from ui.live.cards import response_cards

    cards = response_cards(BATCH)
    stat = {n: {s["key"]: s for s in c["stats"]} for n, c in cards.items()}
    pub = stat["guarantee"]["public"]
    assert pub["label"] == "Public money at risk" and pub["bar"] == 5
    # uninsured 0.93 x $175B = $162.75B, less the 4% asked before it landed ($7B) = $155.75B; 10.4% of that = $16.2B
    assert "about $156B guaranteed" in pub["text"] and "no loss expected (10 of 10 runs open)" in pub["text"]
    assert "about $16.2B if it fails" in pub["text"]
    assert any("about $16.2B lost if the bank still fails" in c for c in cards["guarantee"]["cons"])
    cl = stat["credit-line"]["public"]
    assert 0 < cl["bar"] < 5 and "the Fed lends about $102B against bonds" in cl["text"]
    assert stat["withdrawal-limit"]["public"]["bar"] == 0 and stat["withdrawal-limit"]["public"]["text"] == "none"
    lim = cards["withdrawal-limit"]["cons"]
    assert "until the next open" in lim[0] and any("rush before it lands" in c for c in lim)
    assert any("Still closed at the next open in 10 of 10 runs" == c for c in lim)
    assert not any("capital is gone" in c for c in lim)
    for card in cards.values():
        for t in [card["note"], *card["pros"], *card["cons"], *[s["text"] for s in card["stats"]]]:
            assert not BANNED.search(t), t


def test_unmeasured_cards_give_the_ceiling() -> None:
    from ui.live.cards import response_cards

    pub = {s["key"]: s for s in response_cards(None)["guarantee"]["stats"]}["public"]
    assert pub["text"] == "up to about $163B guaranteed; about $16.9B lost if it fails"


def test_notes_and_limits_say_what_the_engine_does() -> None:
    from ui.live.labels import limits_screen
    from ui.live.presets import RESPONSES

    assert "before the next open; the rest waits" in RESPONSES["withdrawal-limit"].note and "30%" in RESPONSES["withdrawal-limit"].note
    text = " ".join(s["body"] for s in limits_screen()["sections"])
    assert "the engine tries the responses" in text and "insured depositors never run" in text
    assert "follow 2023's terms" in text and "Response sizes and timing are assumed" not in text
    assert not re.search(r"predict", text, re.IGNORECASE)
