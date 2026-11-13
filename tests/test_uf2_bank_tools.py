"""UF-2: the bank's balance-sheet tools (bonds pledged in advance, deposits from big banks), computed from the runs,
and the groups of 31+ on the wall."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.system.can_pay import CanPay, can_pay_fractions
from ui.live import useful


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")


def _series(final: float) -> list:
    return [[m, round(final * m / 1441, 6), round(max(0.0, 1.0 - final * m / 1441 / 0.3), 6), 0.0, 0.0] for m in range(10, 1442, 10)]


FR = CanPay(none=0.2547, fed=0.7759, prepared=0.6529, big_banks=0.1714)


def test_what_the_new_tools_let_the_bank_pay() -> None:
    from params.loader import get_parameter

    fr = can_pay_fractions()
    assert round(fr.prepared, 3) == 0.653 and round(fr.big_banks, 3) == 0.171
    assert get_parameter("M20 deposits from big banks fraction of deposits").status == "SOURCED"
    assert "$30B" in get_parameter("M20 deposits from big banks fraction of deposits").source
    assert get_parameter("M20 HTM bonds fair value fraction of deposits").status == "SOURCED"
    assert get_parameter("M20 discount window margin on pledged bonds").status == "ASSUMED"


def test_the_tools_change_what_the_bank_can_pay_not_what_is_asked() -> None:
    day = _series(0.8)
    fired, demand, cap, first = useful.predict([("prepared", 0)], day, FR)
    assert fired == {} and first is None and cap == FR.prepared and demand == pytest.approx(0.8, abs=1e-3)   # not an action
    _, _, cap, _ = useful.predict([("big-bank-deposits", ("requested", 0.05))], day, FR)
    assert cap == pytest.approx(FR.none + FR.big_banks)      # in time
    _, _, cap, _ = useful.predict([("big-bank-deposits", ("requested", 0.5))], day, FR)
    assert cap == FR.none                                    # too late: already past what it could pay
    _, _, cap, _ = useful.predict([("prepared", 0), ("big-bank-deposits", ("requested", 0.5))], day, FR)
    assert cap == pytest.approx(FR.prepared + FR.big_banks)  # with bonds pledged, 50% is still in time


def test_the_computed_plans_and_who_can_act() -> None:
    plans = useful.computed_plans()
    assert len(plans) == 1 + 2 * len(useful.BIG_TRIGGERS)
    fams = {p["plan"]: useful._family([tuple(x) for x in p["steps"]]) for p in plans}
    assert fams["prepared@0"] == "bank" and fams["big-bank-deposits@req0.05"] == "other-banks"
    assert fams["prepared@0+big-bank-deposits@req0.05"] == "other-banks"
    assert useful._family([("guarantee", ("requested", 0.1))]) == "regulator"
    assert useful.FAMRANK["bank"] < useful.FAMRANK["other-banks"] < useful.FAMRANK["regulator"]


def _search(seeds=(1, 2)) -> dict:
    finals = {0.2: 0.01, 0.25: 0.04, 0.3: 0.36, 0.35: 0.6, 0.4: 0.69, 0.5: 0.77, 0.6: 0.82, 0.75: 0.82, 0.9: 0.82}
    series = {f"strength:{c:g}|{s}": _series(f) for c, f in finals.items() for s in seeds}
    plans = [[], [["guarantee", ["requested", 0.1]]]]
    ids = ["none", "g0.1"]
    for i, p in zip(ids, plans):
        for s in seeds:
            series[f"{i}|{s}"] = series[f"strength:0.9|{s}"]
    rows = [{"plan": i, "label": i, "steps": p, "mean_series": series[f"strength:0.9|{seeds[0]}"], "fired_by_seed": [{}] * len(seeds),
             "open_by_seed": [False] * len(seeds), "rank": 0, "survival": 0.0, "deposits_lost": 0.0} for i, p in zip(ids, plans)]
    return {"seeds": list(seeds), "plans": rows, "series": series, "pick": "g0.1", "what_happened": "none", "action_window": {}}


def test_rescoring_adds_the_tools_ranks_the_banks_first_and_writes_groups() -> None:
    out = useful.rescore(_search(), FR, generated=useful.computed_plans(), wall_size=99)   # AU-2: UF-2's 13 only
    by = {p["plan"]: p for p in out["plans"]}
    both = by["prepared@0+big-bank-deposits@req0.08"]
    assert both["computed"] and both["family"] == "other-banks" and both["knocked"] is None    # 0.82 asked < 0.824
    assert by["prepared@0"]["knocked"] == "fails"                                               # 0.82 > 0.653
    assert by["big-bank-deposits@req0.02"]["knocked"] == "early"                                # the 4% rumor reaches 2%
    ranks = {i: by[i]["useful_rank"] for i in out["useful"]}
    assert ranks["prepared@0+big-bank-deposits@req0.08"] < ranks["g0.1"]   # the banks' own tools before a regulator's
    assert "close call" in both["reason"] and "computed from the runs" in both["reason"]
    assert len(out["groups"]) == 2 and len(out["groups"][0]["cells"]) == len(out["groupPlans"]) == len(out["plans"]) - 1
    cell = out["groups"][0]["cells"][[p["plan"] for p in out["groupPlans"]].index("prepared@0+big-bank-deposits@req0.08")]
    assert cell["k"] == "useful" and cell["c"] and len(cell["s"]) == 48 and cell.get("r")
    assert [p["computed"] for p in out["groupPlans"]].count(False) == 1   # g0.1 is the only simulated plan here


def test_the_page_draws_groups_and_marks_computed_plans() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    for needle in ("function drawGroups", "versions of the day × ", "Versions of the day</span>", "≈", "worked out from the runs",
                   "not separately simulated on every day", "other plans dropped out"):
        assert needle in html, needle
