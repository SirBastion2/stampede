"""AU-1..7: the automatic loop: one-button pipeline, generated plans, tipping point and sensitivity, live tools,
checks, early warning and plain summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.system.can_pay import CanPay
from ui.live import plangen, useful

FR = CanPay(none=0.2547, fed=0.7759, prepared=0.6529, big_banks=0.1714)


def _series(final: float) -> list:
    return [[m, round(final * m / 1441, 6), round(max(0.0, 1.0 - final * m / 1441 / 0.3), 6), 0.1, 0.0] for m in range(10, 1442, 10)]


def _search(seeds=(1, 2), finals=None) -> dict:
    finals = finals or {0.2: 0.01, 0.25: 0.04, 0.3: 0.36, 0.5: 0.77, 0.9: 0.82}
    series = {f"strength:{c:g}|{s}": _series(f) for c, f in finals.items() for s in seeds}
    for s in seeds:
        series[f"none|{s}"] = series[f"strength:0.9|{s}"]
        series[f"guarantee@req0.1|{s}"] = useful.day_path([("guarantee", ("requested", 0.1))], series[f"strength:0.9|{s}"], FR)
    rows = [{"plan": i, "label": i, "steps": p, "mean_series": series[f"strength:0.9|{seeds[0]}"], "fired_by_seed": [{}] * len(seeds),
             "open_by_seed": [False] * len(seeds), "rank": 0, "survival": 0.0, "deposits_lost": 0.0}
            for i, p in (("none", []), ("guarantee@req0.1", [["guarantee", ["requested", 0.1]]]))]
    return {"seeds": list(seeds), "plans": rows, "series": series, "pick": "none", "what_happened": "none", "action_window": {}}


def test_the_generator_writes_hundreds_of_distinct_plans() -> None:
    plans = plangen.generate()
    ids = [p["plan"] for p in plans]
    assert len(plans) > 300 and len(set(ids)) == len(ids) and plans[0]["plan"] == "prepared@0"
    for p in plans:
        steps = [(n, tuple(w) if isinstance(w, list) else w) for n, w in p["steps"]]
        names = [n for n, _ in steps]
        if "prepared" in names:
            assert names[0] == "prepared" and "withdrawal-limit" not in names
        tail = [s for s in steps if s[0] != "prepared"]
        if len(tail) == 2:
            assert plangen._order(tail[1][1]) > plangen._order(tail[0][1])   # the bigger step comes later
    assert plangen.plan_label([("prepared", 0), ("big-bank-deposits", ("requested", 0.12))]) == \
        "Bonds pledged in advance + Deposits from big banks once withdrawals pass 12% of deposits"
    msgs = plangen.messaging_plans([[("guarantee", ("requested", 0.2))]], 1)
    assert len(msgs) == 3 and all(any(n in ("statement", "rate-rise") for n, _ in m) for m in msgs)


def test_a_worked_out_day_freezes_or_cuts_requests() -> None:
    day = _series(0.8)
    g = useful.day_path([("guarantee", ("requested", 0.1))], day, FR)
    assert max(r[1] for r in g) == pytest.approx(0.1 + useful.OVERSHOOT, abs=2e-3)
    lim = useful.day_path([("withdrawal-limit", ("requested", 0.1))], day, FR)
    assert lim[-1][1] == pytest.approx(0.1 + 0.3 * (0.8 - 0.1), abs=2e-3)
    pre = useful.day_path([("prepared", 0)], day, FR)
    assert pre[-1][2] == 0.0 and min(r[2] for r in pre if r[1] <= FR.prepared) >= 0.3   # held up while it can pay


def test_the_rescoring_runs_the_whole_automatic_loop() -> None:
    out = useful.rescore(_search(), FR, wall_size=12, wall_useful=4)
    c = out["considered"]
    assert c["simulated"] == 1 and c["worked_out"] + c["unverified"] > 300 and c["plans"] == c["simulated"] + c["worked_out"] + c["unverified"]
    assert len(out["plans"]) <= 13 and out["plans"][-1]["plan"] == "none" and out["useful"][0] == out["pick"]
    assert out["checks"]["ok"] and out["checks"]["runs_checked"] == 2 and out["checks"]["prefix_max_difference"] < 1e-4
    assert out["sensitivity"]["of"] == 7 and 1 <= out["sensitivity"]["top1_holds"] <= 7
    assert out["tipping"]["between"] == [0.25, 0.3] and out["tipping"]["suggest"] == [0.26, 0.27, 0.28, 0.29]
    w = out["warning"]
    assert len(w["days"]) == 10 and len(w["days"][0]) == 1 + len(w["minutes"]) and {d[0] for d in w["days"]} == {0, 1}
    assert all(p["summary"] for p in out["plans"]) and out["useful_all"][0]["summary"].startswith(out["plans"][0]["label"])
    assert len(out["groups"]) == 2 and len(out["groups"][0]["cells"]) == len(out["groupPlans"])


def test_extra_strengths_are_used_once_every_seed_has_them() -> None:
    data = _search()
    for s in (1, 2):
        data["series"][f"strength:0.27|{s}"] = _series(0.2)
    data["series"]["strength:0.28|1"] = _series(0.3)   # only one seed: not used yet
    out = useful.rescore(data, FR, checks=False, generated=[])
    assert 0.27 in out["strengths"] and 0.28 not in out["strengths"]


def test_live_tools_change_what_the_regulator_says_the_bank_can_pay(monkeypatch: pytest.MonkeyPatch) -> None:
    from contracts.events import BankClosed, PolicyEnacted, WithdrawalRequested
    from contracts.schemas import Scenario
    from engine.live_overrides import live_flags
    from mocks.bus import EventBus
    from mocks.l4_system import RegulatorFake
    from mocks.support import deposits_base

    D = deposits_base()
    for levers, asked, closes in (([], 0.5, True), (["pledge_collateral"], 0.5, False), (["pledge_collateral"], 0.8, True),
                                  (["pledge_collateral", "big_bank_deposits"], 0.8, False)):
        bus, reg = EventBus(), RegulatorFake("M20")
        with live_flags(["M20 closes on what the bank can pay"]):
            reg.attach(bus, Scenario(preset="tech-concentrated", seed=1))
        for lever in levers:
            bus.publish(PolicyEnacted(t=1.0, seed=1, bank_id="bank-fvb", lever=lever, scope="bank-fvb", cost_model={}))
        bus.publish(WithdrawalRequested(t=2.0, seed=1, bank_id="bank-fvb", depositor_id="x", amount=asked * D, channel="wire"))
        assert any(isinstance(e, BankClosed) for e in reg.step(3.0)) is closes, (levers, asked)


def test_big_bank_deposits_arrive_and_pledged_bonds_lend() -> None:
    from contracts.events import LiquidityDrawn, PaymentSettled, PolicyEnacted, Tick
    from contracts.schemas import Scenario
    from engine.bank.ledger import svb_like_opening_book
    from engine.live_overrides import live_flags
    from mocks.bus import EventBus
    from mocks.l2_banks import LiquidityFake
    from mocks.support import deposits_base

    bus, m12 = EventBus(), LiquidityFake("M12")
    with svb_like_opening_book(), live_flags([]):
        m12.attach(bus, Scenario(preset="tech-concentrated", seed=1))
    bus.publish(PolicyEnacted(t=0.0, seed=1, bank_id="bank-fvb", lever="big_bank_deposits", scope="bank-fvb", cost_model={}))
    bus.publish(Tick(t=1.0, seed=1))
    draw, = m12.step(1.0)
    assert isinstance(draw, LiquidityDrawn) and draw.source == "big-bank-deposits" and draw.amount == pytest.approx(30e9, rel=1e-3)
    assert m12.step(2.0) == []   # once
    bus2, m = EventBus(), LiquidityFake("M12")
    with svb_like_opening_book(), live_flags([]):
        m.attach(bus2, Scenario(preset="tech-concentrated", seed=1))
    bus2.publish(PolicyEnacted(t=0.0, seed=1, bank_id="bank-fvb", lever="pledge_collateral", scope="bank-fvb", cost_model={}))
    bus2.publish(PaymentSettled(t=1.0, seed=1, bank_id="bank-fvb", amount=m._opening_cash * deposits_base() + 2e8, channel="wire", queued_remaining=0))
    bus2.publish(Tick(t=1.0, seed=1))
    d, = m.step(1.0)
    assert d.source == "discount_window" and d.amount == pytest.approx(2e8, rel=1e-3)


def test_the_session_accepts_the_new_steps_and_publishes_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    from contracts.events import DepositRateChanged, PolicyEnacted, StatementIssued
    from ui.live.session import LiveSession

    s = LiveSession(seed=1)
    s.load_preset("lost-bonds", plan=[("statement", 1), ("rate-rise", 1), ("prepared", 0), ("big-bank-deposits", 2)])
    for _ in range(4):
        s.step_once()
    log = s.engine.bus.log
    assert any(isinstance(e, StatementIssued) for e in log)
    assert {e.segment for e in log if isinstance(e, DepositRateChanged)} >= {"startup", "vc", "retail"}
    assert {e.lever for e in log if isinstance(e, PolicyEnacted)} >= {"pledge_collateral", "big_bank_deposits"}


def test_the_pipeline_reuses_its_simulations_when_nothing_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    monkeypatch.setenv("STAMPEDE_SEARCH_FILE", str(tmp_path / "live_search.json"))
    spec = importlib.util.spec_from_file_location("pipeline", "scripts/pipeline.py")
    pipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipe)
    key = pipe.input_key((10,))
    assert key == pipe.input_key((10,)) and key != pipe.input_key((10, 25))
    data = _search(seeds=tuple(range(1, 11)))
    data["input_key"] = key
    for name in ("live_search.json", "live_search_10.json"):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    import ui.live.search as search
    monkeypatch.setattr(search, "run_trigger_options", lambda **k: pytest.fail("resimulated although nothing changed"))
    monkeypatch.setattr(pipe, "_tipping_runs", lambda *a, **k: None)
    assert pipe.main(["--runs", "10", "--messaging-plans", "0"]) == 0
    st = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert st["stage"] == "done" and st["finished"]
    out = json.loads((tmp_path / "live_search_10.json").read_text())
    assert out["mode"] == "useful" and out["considered"]["worked_out"] + out["considered"]["unverified"] > 300 and (tmp_path / "live_search_10.raw.json").exists()


def test_the_pipeline_reuse_flag_keeps_files_without_an_input_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI-4 pool predates the input key: --reuse ranks it without the ~5 h resimulation."""
    import importlib.util

    monkeypatch.setenv("STAMPEDE_SEARCH_FILE", str(tmp_path / "live_search.json"))
    spec = importlib.util.spec_from_file_location("pipeline_reuse", "scripts/pipeline.py")
    pipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipe)
    data = _search(seeds=tuple(range(1, 11)))
    assert "input_key" not in data
    for name in ("live_search.json", "live_search_10.json"):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    import ui.live.search as search
    monkeypatch.setattr(search, "run_trigger_options", lambda **k: pytest.fail("resimulated with --reuse"))
    monkeypatch.setattr(pipe, "_tipping_runs", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        pipe.main(["--runs", "10", "--reuse", "--force"])
    assert pipe.main(["--runs", "10", "--messaging-plans", "0", "--reuse"]) == 0
    st = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert st["stage"] == "done"
    assert json.loads((tmp_path / "live_search_10.json").read_text())["mode"] == "useful"
    (tmp_path / "live_search_10.json").unlink()
    (tmp_path / "live_search_10.raw.json").unlink()
    with pytest.raises(SystemExit):
        pipe.main(["--runs", "10", "--reuse"])


def test_the_page_and_server_have_the_button_the_meter_and_the_routes() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    for needle in ('id="wallRun"', "/api/pipeline/run", "/api/pipeline/status", "function warningLine", "similar simulated days ended in closure",
                   "plans considered", "p.summary || p.reason"):
        assert needle in html, needle
    server = Path("ui/server.py").read_text(encoding="utf-8")
    assert '@app.post("/api/pipeline/run")' in server and '@app.get("/api/pipeline/status")' in server
