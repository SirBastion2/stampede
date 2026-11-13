"""Funding-dependent cash conditions must not inherit a no-response cash path."""

from __future__ import annotations

import copy

import pytest

from tests.test_au_automatic import FR, _search, _series
from ui.live import useful


def test_pledging_bonds_invalidates_a_later_cash_path_prediction() -> None:
    day = _series(0.82)
    with pytest.raises(useful.CashPathRequired, match="pledged"):
        useful.predict([("prepared", 0), ("big-bank-deposits", ("cash", 0.75))], day, FR)


def test_request_rule_after_pledging_keeps_its_observed_demand_condition() -> None:
    fired, demand, capacity, _ = useful.predict(
        [("prepared", 0), ("big-bank-deposits", ("requested", 0.2))], _series(0.82), FR
    )
    assert fired["big-bank-deposits"] > 0
    assert demand <= capacity == pytest.approx(FR.prepared + FR.big_banks)


def test_first_cash_condition_is_supported_but_later_cash_needs_its_own_path() -> None:
    day = _series(0.82)
    assert useful.predict([("credit-line", ("cash", 0.75))], day, FR)[0]["credit-line"] > 0
    with pytest.raises(useful.CashPathRequired, match="another action"):
        useful.predict(
            [("big-bank-deposits", ("requested", 0.02)), ("guarantee", ("cash", 0.5))], day, FR
        )


def test_unverified_cash_rules_are_retained_without_survival_claims_or_a_rank() -> None:
    data = _search()
    original = copy.deepcopy(data)
    plan = {
        "plan": "pledged-cash",
        "label": "Pledged bonds then peer deposits on cash",
        "steps": [["prepared", 0], ["big-bank-deposits", ["cash", 0.75]]],
    }
    out = useful.rescore(data, FR, generated=[plan])
    assert data == original
    assert out["considered"]["unverified"] == 1
    assert out["unverified_plans"] == [
        {
            **plan,
            "computed": True,
            "status": "needs-simulation",
            "reason": ("A cash condition after bonds are pledged "
                       "needs a funded cash-path simulation."),
        }
    ]
    assert out["pick"] != plan["plan"]
    assert plan["plan"] not in [
        p["plan"] for p in out["plans"] + out["useful_all"] + out["groupPlans"]
    ]
    assert all(plan["plan"] not in v["top_plans"] for v in out["sensitivity"]["variants"])
    assert out["considered"]["plans"] == (
        out["considered"]["simulated"] + out["considered"]["worked_out"] + 1
    )


def test_no_useful_plan_does_not_fall_back_to_an_unverified_or_failing_pick() -> None:
    data = _search()
    data["plans"] = [data["plans"][0]]
    out = useful.rescore(
        data,
        FR,
        generated=[
            {
                "plan": "pledged-cash",
                "label": "Unverified cash rule",
                "steps": [["prepared", 0], ["big-bank-deposits", ["cash", 0.75]]],
            }
        ],
    )
    assert out["pick"] is None and out["useful"] == []


def test_independent_check_never_counts_generated_paths_as_simulated_runs() -> None:
    data = _search()
    # A previously generated path is not independent evidence, even if it is present in the input.
    data["plans"].append(
        {"plan": "generated", "label": "Generated", "steps": [["prepared", 0]], "computed": True}
    )
    out = useful.rescore(data, FR, generated=[])
    assert out["checks"]["runs_checked"] == len(data["seeds"])
    assert out["considered"]["simulated"] == 1


def test_holdout_for_the_old_pick_is_preserved_but_not_claimed_for_the_new_one() -> None:
    data = _search()
    holdout = {"plan": "old-pick", "runs": 10, "survival": 1.0}
    data["holdout"] = holdout
    out = useful.rescore(data, FR, generated=[])
    assert out["holdout"] is None
    assert out["previous_holdout"] == holdout
    assert data["holdout"] == holdout
