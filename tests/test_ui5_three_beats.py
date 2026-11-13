"""UI-5: three beats (a rumor hits; the engine tries the plans; what helps), no Bank or Stocks screens, one headline
made from the search's numbers, plain words on screen, and the knock-out counts the What helps page shows."""

from __future__ import annotations

import re
from pathlib import Path

PAGE = Path("ui/live/stampede_ui.html")


def _html() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_nav_is_three_beats_with_recap_and_peers_for_the_operator_only() -> None:
    nav = re.search(r'<nav class="tabs"[^>]*>(.*?)</nav>', _html(), re.S).group(1)
    tabs = [(t, 'class="opx"' in attrs, re.sub(r"<[^>]+>", "", label))
            for t, attrs, label in re.findall(r'<button role="tab" data-tab="(\w+)"([^>]*)>(.*?)</button>', nav)]
    assert [(t, label) for t, opx, label in tabs if not opx] == [
        ("crowd", "1A rumor hits"), ("search", "2The engine tries the plans"), ("why", "3What helps")]
    assert sorted(t for t, opx, _ in tabs if opx) == ["banks", "recap"]


def test_the_bank_and_stocks_screens_are_gone() -> None:
    html = _html()
    for gone in ('id="tab-bank"', 'id="tab-market"', 'data-tab="bank"', 'data-tab="market"', 'id="openScreens"',
                 "function drawChart", "function updateMarketDom", "function updateBankScreen", "function drawFlow",
                 "function tapeTick", "function drawWatch", "function drawDayChart", "'#bkCrisis'", "$('#news')"):
        assert gone not in html, gone
    assert "const TABMAP = { main: 'crowd', recap: 'recap', search: 'search', why: 'why' };" in html
    assert "const MARK = {" in html   # the recap chart's event markers stay


def test_one_headline_from_the_search_numbers_with_its_label() -> None:
    html = _html()
    assert html.count('id="headline"') == 1
    for needle in ("function headlineText", "function renderHeadline", "Best tested plan: ${shortLabel(p, false)}",
                   "≈ calculated from saved runs", "if (ENGINE) wallLoad();"):
        assert needle in html, needle


def test_the_answer_waits_for_beat_two() -> None:
    """Option B, sharpened by UI-8: beat 1, its recap and beat 2's replay show the neutral line; the answer appears
    when the replay finishes, and on beat 3."""
    html = _html()
    assert "const REVEAL_TABS = ['search', 'why'];" in html
    assert "const reveal = active === 'why' || (active === 'search' && wall.phase === 'final');" in html
    assert "const h = reveal ? headlineText(wall.data) : { text: HEADLINE0, tag: '' };" in html
    assert "if ((wall.phase === 'final') !== wall._revealed)" in html   # revealed by the replay, hidden again on Reset
    assert "queueMicrotask(renderHeadline);" in html   # every tab change, after the search state exists
    neutral = re.search(r"const HEADLINE0 = '([^']+)';", html).group(1)
    assert "What helps" not in neutral and "versions of the day" in neutral
    assert neutral in re.search(r'<div class="headline" id="headline"[^>]*>(.*?)</div>', html).group(1)


def test_the_beats_lead_to_each_other() -> None:
    html = _html()
    assert 'data-goto="search">Next: the engine tries the plans</button>' in html
    assert 'data-goto="why"' in html and "Next: what helps" in html
    assert "const TABKEYS = { m: 'crowd', p: 'search', w: 'why', c: 'recap' };" in html


def test_what_helps_is_trimmed_and_keeps_the_details() -> None:
    html = _html()
    for needle in ("Strongest plan in these tests", "Other useful plans", "Why plans drop out", "Why timing matters",
                   "What the result depends on", '<details id="whyMore"', "function alsoUseful", "function knockCounts"):
        assert needle in html, needle
    assert "Why each plan was knocked out" not in html


def test_plain_words_on_screen() -> None:
    html = _html()
    for old in ("${N} groups × ", "fillText(\"Each group is one version", "`Group ${g.seed}`", "Same rumor, more seeds",
                "runs per plan: open in", "rumor strengths (believability", "simulated days that looked most like",
                "'credit-line': 'credit line'", ">Groups<", "the group's #1"):
        assert old not in html, old
    for new in ("versions of the day × ", "Versions of the day</span>", "weaker and stronger versions of the rumor"):
        assert new in html, new


def test_rescore_counts_why_plans_are_out_and_words_the_tipping_point_plainly() -> None:
    from ui.live.useful import _tipping, rescore
    from tests.test_au_automatic import _search

    out = rescore(_search(seeds=tuple(range(1, 11))), checks=False)
    k = out["considered"]["knocked"]
    assert set(k) == {"early", "not-us", "fails"}
    assert (out["considered"]["useful"] + sum(k.values())
            + out["considered"]["unverified"]) == out["considered"]["plans"]
    seeds = [1, 2]
    tips = _tipping([0.2, 0.3], seeds, {(0.2, 1): False, (0.2, 2): False, (0.3, 1): True, (0.3, 2): True})
    flat = _tipping([0.2, 0.3], seeds, {(c, s): False for c in (0.2, 0.3) for s in seeds})
    assert tips["between"] == [0.2, 0.3] and "slightly weaker" in tips["text"] and "versions of the day" in tips["text"]
    assert flat["between"] is None
    assert "strength" not in tips["text"] + flat["text"]
