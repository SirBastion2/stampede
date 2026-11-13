"""VF-3: the live screen names every lever in plain words and says who acts; the early warning says its
comparison is with no response."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("lever,label,by", [
    ("pledge_collateral", "Bonds pledged in advance", "bank"),
    ("big_bank_deposits", "Deposits from big banks", "bank"),
    ("statement", "Reassuring statement", "bank"),
    ("deposit_rate", "Deposit rate raised to 8%", "bank"),
    ("guarantee_deposits", "Deposit guarantee", "regulator"),
    ("emergency_lending", "Emergency credit line", "regulator"),
    ("cap_withdrawals", "Withdrawal limit", "regulator"),
])
def test_every_lever_has_plain_words_and_who_acts(lever: str, label: str, by: str) -> None:
    from ui.live.session import LiveSession

    got, who, note = LiveSession._lever_words(lever)
    assert (got, who) == (label, by)
    assert "_" not in got
    if by == "bank":
        assert note


def test_an_unknown_lever_still_reads_as_words() -> None:
    from ui.live.session import LiveSession

    assert LiveSession._lever_words("some_new_lever")[0] == "Some new lever"


def test_a_live_run_with_pledged_bonds_sends_its_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    from ui.live.search import PRESET
    from ui.live.session import LiveSession

    s = LiveSession(seed=1)
    s.load_preset(PRESET, plan=[("prepared", 0)])
    for _ in range(3):
        s.step_once()
    responses = [e for e in s._timeline if e["type"] == "response"]
    assert responses and responses[0]["lever"] == "pledge_collateral"
    assert responses[0]["text"] == "Bonds pledged in advance" and responses[0]["by"] == "bank"


def test_the_page_uses_who_acts_and_qualifies_the_warning() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    assert "const own = e.by === 'bank';" in html
    assert "headline(own ? `Copperleaf: ${e.text}` : `${e.text} announced`" in html
    assert "Without help: <b" in html and "similar simulated days ended in closure</b>" in html
