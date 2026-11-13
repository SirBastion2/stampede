"""UI-3: the response search (plans, successive halving, held-out check) and the decision deadlines."""

from __future__ import annotations

import pytest

from ui.live import search
from ui.live.presets import RESPONSES


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _fake_engine(monkeypatch: pytest.MonkeyPatch, deadline: int = 300) -> list:
    """A stand-in run: the guarantee saves the bank if it comes by ``deadline``; nothing else does."""
    calls = []

    def run(args):
        plan, seed, minutes = args
        calls.append((search.plan_id(plan), seed))
        g = dict(plan).get("guarantee")
        saved = g is not None and g <= deadline
        lost = (0.02 + g / 2000.0 if g is not None else 0.82) - (0.5 if "withdrawal-limit" in dict(plan) and not saved else 0.0)
        lost += seed * 1e-4
        return {"plan": search.plan_id(plan), "seed": seed, "minutes": minutes or 1441, "open": saved,
                "closure_minute": None if saved else 1440.0, "cash_out_minute": None, "requested": lost,
                "requested_usd": lost * 175e9, "emergency_usd": 0.0, "series": [[10, lost / 2, 1.0]], "seconds": 0.0}

    monkeypatch.setattr(search, "run_plan", run)
    return calls


def test_plans_are_unique_and_readable() -> None:
    plans = search.all_plans()
    assert len(plans) == 90 and plans[0] == [] and len({search.plan_id(p) for p in plans}) == 90
    assert search.plan_label([]) == "The 2023 replay (no response)"
    assert search.plan_label([("guarantee", 240)]) == "Deposit guarantee at 1:30 PM"
    assert search.clock(930) == "1:00 AM (day 2)" and search.clock(390) == "4:00 PM"


def test_tournament_keeps_the_best_third_and_checks_on_new_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_engine(monkeypatch)
    runner = search.Runner(1, None)
    out = search.tournament(runner, search.all_plans())
    sizes = [len(r["plans"]) for r in out["rounds"]]
    kept = [sum(p["kept"] for p in r["plans"]) for r in out["rounds"]]
    assert sizes == [90, 30, 10] and kept == [30, 10, 10]
    assert [r["seeds"] for r in out["rounds"]] == [[1], [1, 2, 3], list(range(1, 11))]
    assert out["holdout"]["seeds"] == list(range(11, 21)) and len(out["holdout"]["plans"]) == 3
    winner = out["winner"]
    assert winner["survival"] == 1.0 and winner["plan"] == "guarantee@0"  # the earliest guarantee loses the least
    first = out["rounds"][0]["plans"]
    assert all(p["survival"] == 1.0 for p in first[:20])  # survivors always outrank closures
    assert len(calls) == len(set(calls))  # each (plan, seed) ran once


def test_ties_go_to_fewer_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_engine(monkeypatch)
    runner = search.Runner(1, None)
    one, two = [("guarantee", 90)], [("guarantee", 90), ("credit-line", 90)]
    out = search.tournament(runner, [two, one, []], round_seeds=((1,),), holdout=(2,))
    assert [p["plan"] for p in out["rounds"][0]["plans"]][:2] == [search.plan_id(one), search.plan_id(two)]


def test_deadline_binary_search(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_engine(monkeypatch, deadline=300)
    runner = search.Runner(1, None)
    g = search.deadline(runner, "guarantee")
    assert g["saves_bank"] and 295 <= g["deadline_minute"] <= 300 and g["survival_at_deadline"] == 1.0
    assert g["deadline_clock"] in {search.clock(m) for m in range(295, 301)}
    c = search.deadline(runner, "credit-line")
    assert not c["saves_bank"] and "never" in c["text"]


def test_run_search_writes_a_replayable_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _fake_engine(monkeypatch)
    data = search.run_search(plans=search.all_plans()[:12], round_seeds=((1,), (1, 2)), holdout=(3,), deadline_seeds=(1,))
    path = search.write_search(data, str(tmp_path / "s.json"))
    back = search.load_search(path)
    assert back["plans"] == 12 and len(back["deadlines"]) == len(RESPONSES)
    assert all(f"{p['plan']}|1" in back["series"] for p in back["rounds"][0]["plans"])
    assert search.load_search(str(tmp_path / "missing.json")) is None


def test_session_runs_a_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    from contracts.events import PolicyEnacted
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=1)
    session.load_preset("lost-bonds", plan=[("credit-line", 2), ("withdrawal-limit", 4)])
    for _ in range(6):
        session.step_once()
    assert [(e.t, e.lever) for e in session.engine.bus.log if isinstance(e, PolicyEnacted)] == [(2.0, "emergency_lending"), (4.0, "cap_withdrawals")]
    assert session.run_label()["responses"] == ["credit-line", "withdrawal-limit"]
    with pytest.raises(KeyError):
        LiveSession(seed=1).load_preset("lost-bonds", plan=[("guarantee", 5), ("guarantee", 9)])


def test_server_routes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    monkeypatch.setenv("STAMPEDE_SEARCH_FILE", str(tmp_path / "none.json"))
    client = TestClient(server.app)
    assert client.get("/api/search").json()["ok"] is False


def _curve_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in runs with a curve: requests climb all day; a guarantee freezes them; only it keeps the bank open."""
    def run(args):
        plan, seed, minutes = args
        g = dict(plan).get("guarantee")
        rate = 0.82 / 1440 * (0.33 if "withdrawal-limit" in dict(plan) else 1.0)
        series, req = [], 0.0
        for m in range(10, 1441, 10):
            req = (rate * min(m, g) if g is not None else rate * m) + seed * 1e-5
            series.append([m, round(req, 5), 1.0, 0.0, 0.0])
        saved = g is not None and g <= 300
        return {"plan": search.plan_id(plan), "seed": seed, "minutes": 1441, "open": saved,
                "closure_minute": None if saved else 1440.0, "cash_out_minute": None, "requested": req,
                "requested_usd": req * 175e9, "emergency_usd": 0.0, "series": series, "seconds": 0.0}

    monkeypatch.setattr(search, "run_plan", run)


def test_continuous_elimination_knocks_out_each_plan_once_it_is_certain_to_lose(monkeypatch: pytest.MonkeyPatch) -> None:
    _curve_engine(monkeypatch)
    plans = [[], [("guarantee", 0)], [("guarantee", 90)], [("guarantee", 390)], [("withdrawal-limit", 90)],
             [("guarantee", 0), ("credit-line", 90)]]
    data = search.run_full_search(plans=plans, seeds=(1, 2, 3), holdout=(4, 5), deadline_seeds=(1, 2))
    rows = {r["plan"]: r for r in data["plans"]}
    assert data["pick"] == "guarantee@0" and rows["guarantee@0"]["elim_minute"] is None
    # requests only add up: "no response" is out the moment it passes what the pick loses all day (~0)
    assert rows["none"]["elim_minute"] == 10 and "more than the best plan loses all day" in rows["none"]["reason"]
    assert rows["guarantee@390"]["elim_minute"] < 390 and rows["withdrawal-limit@90"]["elim_minute"] < 100
    tie = rows[search.plan_id([("guarantee", 0), ("credit-line", 90)])]
    assert tie["elim_minute"] == 1441 and "more responses" in tie["reason"]  # same curve: out at the end, on a tie-break
    order = sorted(data["plans"], key=lambda r: r["elim_order"])
    assert order[-1]["plan"] == data["pick"] and [r["elim_order"] for r in order] == list(range(1, len(plans) + 1))
    assert all(r["elim_minute"] <= s["elim_minute"] for r, s in zip(order[:-2], order[1:-1]))  # in time order
    assert data["holdout"]["seeds"] == [4, 5] and data["holdout"]["plan"] == "guarantee@0"
    assert len(rows["none"]["mean_series"][0]) == 5 and rows["none"]["open_by_seed"] == [False] * 3


def test_a_closing_plan_goes_out_when_its_closures_are_known(monkeypatch: pytest.MonkeyPatch) -> None:
    _curve_engine(monkeypatch)
    runner = search.Runner(1, None)
    a, b = [("guarantee", 200)], [("guarantee", 310)]   # b freezes later and closes; a survives
    runner.run([(p, s) for p in (a, b) for s in (1, 2)])
    out = {r["plan"]: r for r in search.elimination([a, b], runner, (1, 2))["plans"]}
    assert out[search.plan_id(b)]["elim_minute"] is not None and out[search.plan_id(a)]["reason"] == "the engine's pick"


def test_earliest_moment_to_act() -> None:
    late = search.all_plans(90)
    assert late and all(m >= 90 for p in late for _, m in p) and [] in late
    assert len(late) < len(search.all_plans())


def _trigger_engine(monkeypatch: pytest.MonkeyPatch, noise: float = 0.0012, deadline: int = 300) -> None:
    """Stand-in runs for trigger rules: requests climb 0.82 over the day; a normal day peaks at ``noise``;
    a guarantee freezes requests and keeps the bank open if it lands by ``deadline``."""
    rate = 0.82 / 1440

    def fire(when):
        if isinstance(when, (int, float)):
            return float(when)
        kind, level = when
        return level / rate if kind == "requested" else (1 - level) / (rate * 1.25)

    def run(args):
        plan, seed, minutes = args[:3]
        preset = args[3] if len(args) > 3 else search.PRESET
        if len(args) > 4 and args[4] is not None:   # a rumor-strength run (UI-3e): here every strength sinks the bank
            cred = args[4]
            series = [[m, round(rate * m, 6), 1.0, 0.0, 0.0] for m in range(10, 1441, 10)]
            return {"plan": f"strength:{cred:g}", "seed": seed, "credibility": cred, "minutes": 1441, "open": False,
                    "closure_minute": 1440.0, "fired": {}, "fired_requested": {}, "cash_out_minute": None, "requested": 0.82,
                    "requested_usd": 0.82 * 175e9, "emergency_usd": 0.0, "series": series, "seconds": 0.0}
        if preset != search.PRESET:   # the no-rumor day
            series = [[m, round(noise * m / 1440, 6), 1.0, 0.0, 0.0] for m in range(10, 1441, 10)]
            return {"plan": f"baseline:{preset}", "seed": seed, "minutes": 1441, "open": True, "closure_minute": None,
                    "fired": {}, "cash_out_minute": None, "requested": noise, "requested_usd": noise * 175e9,
                    "emergency_usd": 0.0, "series": series, "seconds": 0.0}
        g = next((fire(w) for n, w in plan if n == "guarantee"), None)
        series, req = [], 0.0
        for m in range(10, 1441, 10):
            req = rate * (min(m, g) if g is not None else m) + seed * 1e-6
            series.append([m, round(req, 6), round(max(0.0, 1 - 1.25 * req), 4), 0.0, 0.0])
        saved = g is not None and g <= deadline
        return {"plan": search.plan_id(plan), "seed": seed, "minutes": 1441, "open": saved,
                "closure_minute": None if saved else 1440.0, "fired": {"guarantee": g} if g is not None else {},
                "fired_requested": {"guarantee": rate * g} if g is not None else {},
                "cash_out_minute": None, "requested": req, "requested_usd": req * 175e9, "emergency_usd": 0.0,
                "series": series, "seconds": 0.0}

    monkeypatch.setattr(search, "run_plan", run)


def test_trigger_labels_and_ids() -> None:
    assert search.plan_label([("guarantee", ("requested", 0.0025))]) == "Deposit guarantee once withdrawals pass 0.25% of deposits"
    assert search.plan_label([("credit-line", ("cash", 0.75))]) == "Emergency credit line once cash falls below 75%"
    assert search.plan_id([("guarantee", ("requested", 0.05)), ("withdrawal-limit", ("requested", 0.005))]) == "withdrawal-limit@req0.005+guarantee@req0.05"
    assert len(search.trigger_plans()) == 32 and [] in search.trigger_plans()


def test_trigger_search_rules_out_false_alarms_and_finds_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _trigger_engine(monkeypatch, noise=0.0012, deadline=300)
    plans = [[], [("guarantee", ("requested", 0.001))], [("guarantee", ("requested", 0.0025))],
             [("guarantee", ("requested", 0.05))], [("guarantee", ("requested", 0.2))], [("withdrawal-limit", ("requested", 0.02))]]
    data = search.run_trigger_search(plans=plans, seeds=(1, 2), holdout=(3,), deadline_seeds=(1, 2))
    rows = {r["plan"]: r for r in data["plans"]}
    early = rows["guarantee@req0.001"]
    assert early["false_alarm"] and early["elim_minute"] == 0 and "normal day" in early["reason"]
    assert data["pick"] == "guarantee@req0.0025"   # the earliest rule that never fires on a normal day
    assert rows["guarantee@req0.2"]["survival"] == 0.0   # waited too long: past the deadline
    w = data["action_window"]
    assert w["detect_level"] == 0.0025 and w["detect_minute"] is not None and w["deadline_minute"] is not None
    assert w["detect_minute"] < w["deadline_minute"]
    assert "Failure became likely" in w["text"]   # UI-3e: the window now opens when failure is likely, not merely visible
    assert data["mode"] == "triggers" and data["baseline"]["preset"] == "stadium"


def test_a_trigger_fires_the_minute_after_its_level_is_crossed(monkeypatch: pytest.MonkeyPatch) -> None:
    from contracts.events import PolicyEnacted
    from engine.bank.ledger import registry_deposits_total
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=1)
    session.load_preset("lost-bonds", plan=[("guarantee", ("requested", 0.0005)), ("credit-line", ("cash", 0.01))])
    shares = []
    for _ in range(110):
        shares.append(session._ui_requested / registry_deposits_total())
        session.step_once()
    fired = session.trigger_minutes["guarantee"]
    assert shares[int(fired)] >= 0.0005 > shares[int(fired) - 1]   # reported by the start of that minute, not before
    assert [e.t for e in session.engine.bus.log if isinstance(e, PolicyEnacted)] == [fired]
    assert "credit-line" not in session.trigger_minutes   # cash never fell that far
    with pytest.raises(KeyError):
        LiveSession(seed=1).load_preset("lost-bonds", plan=[("guarantee", ("price", 0.5))])


def test_likely_failure_is_where_nine_in_ten_runs_that_got_there_failed() -> None:
    weak = [{"requested": 0.006, "open": True, "credibility": 0.2}] * 6           # fizzle: peak 0.6%
    strong = [{"requested": 0.8, "open": False, "credibility": 0.9}] * 14          # sink the bank
    out = search.likely_failure(weak + strong)
    by = {r["level"]: r for r in out["table"]}
    assert by[0.005]["reached"] == 20 and by[0.005]["rate"] == 0.7      # weak runs got this far too
    assert by[0.01]["reached"] == 14 and by[0.01]["rate"] == 1.0
    assert out["level"] == 0.01 and out["strengths"] == [0.2, 0.9]
    assert search.likely_failure(weak)["level"] is None                  # nothing fails: no level qualifies


def _strength_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in: weak rumors (credibility < 0.3) peak under 1% and the bank stays open; stronger ones sink it."""
    rate = 0.82 / 1440
    base = search.run_plan

    def run(args):
        plan, seed, minutes = args[:3]
        preset = args[3] if len(args) > 3 else search.PRESET
        cred = args[4] if len(args) > 4 else None
        if cred is None:
            return base_run((plan, seed, minutes, preset))
        peak = 0.006 if cred < 0.3 else 0.8
        series = [[m, round(min(peak, rate * m), 6), 1.0, 0.0, 0.0] for m in range(10, 1441, 10)]
        return {"plan": f"strength:{cred:g}", "seed": seed, "credibility": cred, "minutes": 1441, "open": peak < 0.01,
                "closure_minute": None if peak < 0.01 else 1440.0, "fired": {}, "cash_out_minute": None, "requested": peak,
                "requested_usd": peak * 175e9, "emergency_usd": 0.0, "series": series, "seconds": 0.0}

    _trigger_engine(monkeypatch, noise=0.0004, deadline=300)
    base_run = search.run_plan
    monkeypatch.setattr(search, "run_plan", run)


def test_a_guarantee_is_justified_only_once_failure_is_likely(monkeypatch: pytest.MonkeyPatch) -> None:
    _strength_engine(monkeypatch)
    plans = [[], [("guarantee", ("requested", 0.001))], [("guarantee", ("requested", 0.005))],
             [("guarantee", ("requested", 0.01))], [("guarantee", ("requested", 0.05))], [("guarantee", ("requested", 0.2))]]
    data = search.run_trigger_search(plans=plans, seeds=(1, 2), holdout=(3,), deadline_seeds=(1, 2))
    rows = {r["plan"]: r for r in data["plans"]}
    assert data["likely_failure"]["level"] == 0.01
    for early in ("guarantee@req0.001", "guarantee@req0.005"):   # pass the normal-day test, but act before failure is likely
        assert rows[early]["false_alarm"] and "before failure was likely" in rows[early]["reason"]
    assert data["pick"] == "guarantee@req0.01"                    # the first justified rule that still saves the bank
    w = data["action_window"]
    assert w["likely_level"] == 0.01 and w["likely_minute"] is not None and w["likely_minute"] <= w["deadline_minute"]
    assert "Failure became likely" in w["text"]


def test_rumor_strength_reaches_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=1)
    session.load_preset("lost-bonds", credibility=0.3)
    assert [round(p.credibility, 2) for p in session._scheduled] == [0.3]
    assert LiveSession(seed=1).load_preset("lost-bonds") is not None


def test_crossings_are_interpolated_between_samples() -> None:
    series = [[10, 0.004, 1, 0, 0], [20, 0.008, 1, 0, 0], [30, 0.016, 1, 0, 0]]
    assert search.first_cross(series, 0.012) == 25.0 and search.first_cross(series, 0.002) == 5.0
    assert search.first_cross(series, 0.5) is None
