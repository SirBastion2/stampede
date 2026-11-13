"""VF-6: a wall tile names the days a plan stepped in early, instead of a bare "useful on 36 of 50 days" that
reads as 36 open days next to OPEN 50/50."""

from __future__ import annotations

from pathlib import Path


def _html() -> str:
    return Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")


def test_the_tally_counts_the_early_days() -> None:
    html = _html()
    assert "early: G.filter(g => g.cells[k].k === 'early').length" in html
    assert "q.groups_early = r.early;" in html


def test_the_tile_names_the_other_days() -> None:
    html = _html()
    assert "function tileDays(p)" in html
    assert "`useful on all ${p.groups_n} days`" in html
    assert "`stepped in early on ${p.groups_early} of ${p.groups_n} days`" in html
    assert "wrapFill(ctx, howItWorks(p), x + s / 2, 48, s - 8, 16);" in html   # UI-8: the finale explains the plan instead
    assert "ctx.fillText(`useful on ${p.groups_useful} of ${p.groups_n} days" not in html
