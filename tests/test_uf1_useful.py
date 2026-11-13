"""UF-1: plans ranked by how useful they are to the bank, and the live regulator on the same can-pay rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.system.can_pay import CanPay, can_pay_fractions
from ui.live import useful


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _series(final: float, start: int = 0) -> list:
    """Demand rising linearly from ``start`` to ``final`` by the next open; cash falls as it is paid."""
    rows = []
    for m in range(10, 1442, 10):
        r = 0.0 if m <= start else final * (m - start) / (1441 - start)
        rows.append([m, round(r, 6), round(max(0.0, 1.0 - r / 0.3), 6), 0.0, 0.0])
    return rows


FR = CanPay(none=0.2547, fed=0.7759)


def test_what_the_bank_can_pay() -> None:
    from params.loader import get_parameter

    fr = can_pay_fractions()
    assert fr.none == pytest.approx(0.0797359 + 0.026 + 26.069 / 175, abs=1e-6) and round(fr.none, 3) == 0.255
    assert fr.fed == pytest.approx(0.0797359 + 0.678128 + 0.018, abs=1e-6) and round(fr.fed, 3) == 0.776
    row = get_parameter("M20 bonds sellable near book value fraction of deposits")
    assert row.status == "SOURCED" and "$26.069B" in row.source
    assert get_parameter("M20 closes on what the bank can pay").value == "false"
    assert get_parameter("M29 live regulator closes on what the bank can pay").value == "true"


def test_a_plans_day_follows_from_the_no_response_run() -> None:
    day = _series(0.8)
    fired, demand, cap, first = useful.predict([("guarantee", ("requested", 0.05))], day, FR)
    assert demand == pytest.approx(0.05 + useful.OVERSHOOT) and cap == FR.none and first == pytest.approx(0.05 / 0.8 * 1441, abs=1)
    _, demand, _, _ = useful.predict([("withdrawal-limit", ("requested", 0.1))], day, FR)
    assert demand == pytest.approx(0.1 + 0.3 * (0.8 - 0.1), abs=1e-3)
    _, demand, cap, _ = useful.predict([("credit-line", ("requested", 0.02))], day, FR)
    assert demand == pytest.approx(0.8, abs=1e-3) and cap == FR.fed   # on before the bank runs short
    _, _, cap, _ = useful.predict([("credit-line", ("requested", 0.5))], day, FR)
    assert cap == FR.none                                              # too late: past what it could pay
    _, demand, _, _ = useful.predict([("guarantee", ("cash", 0.75))], day, FR)
    assert demand == pytest.approx(0.075, abs=0.002)                   # cash 75% is 7.5% asked here
    fired, demand, cap, _ = useful.predict([("credit-line", ("cash", 0.75)), ("guarantee", ("requested", 0.08))], day, FR)
    assert set(fired) == {"credit-line", "guarantee"} and demand == pytest.approx(0.08 + useful.OVERSHOOT) and cap == FR.fed


def _search(seeds=(1, 2)) -> dict:
    finals = {0.2: 0.01, 0.25: 0.04, 0.3: 0.36, 0.35: 0.6, 0.4: 0.69, 0.5: 0.77, 0.6: 0.8, 0.75: 0.82, 0.9: 0.82}
    series = {f"strength:{c:g}|{s}": _series(f * (1 + 0.01 * s)) for c, f in finals.items() for s in seeds}
    plans = [[], [["guarantee", ["requested", 0.001]]], [["guarantee", ["requested", 0.1]]], [["guarantee", ["requested", 0.3]]],
             [["withdrawal-limit", ["requested", 0.05]]], [["credit-line", ["requested", 0.02]]],
             [["credit-line", ["cash", 0.75]], ["guarantee", ["requested", 0.08]]]]
    ids = ["none", "g0.001", "g0.1", "g0.3", "lim", "line", "line+g"]
    rows = [{"plan": i, "label": i, "steps": p, "mean_series": series[f"strength:0.9|{seeds[0]}"], "fired_by_seed": [{}] * len(seeds),
             "open_by_seed": [False] * len(seeds), "rank": 0, "survival": 0.0, "deposits_lost": 0.0} for i, p in zip(ids, plans)]
    return {"seeds": list(seeds), "plans": rows, "series": series, "pick": "g0.001", "what_happened": "none",
            "action_window": {"text": "old"}, "holdout": None}


def test_rescoring_knocks_out_the_useless_and_ranks_the_rest() -> None:
    out = useful.rescore(_search(), FR, generated=[], wall_size=99)   # AU-2: the simulated plans only
    by = {p["plan"]: p for p in out["plans"]}
    assert out["mode"] == "useful" and "series" not in out
    assert by["g0.001"]["knocked"] == "early" and "acts before the run is a threat" in by["g0.001"]["reason"]
    assert by["lim"]["knocked"] == "not-us"
    assert by["g0.3"]["knocked"] == "fails"   # frozen at 30%, past what the bank can pay
    assert by["line"]["knocked"] == "early"   # once 2% is asked it also acts on the rumors that reach 4% and fizzle
    assert out["useful"] == ["line+g", "g0.1"]                                     # escalation first, then the later guarantee
    assert by["line+g"]["useful_rank"] == 1 and by["line+g"]["escalates"] and out["pick"] == "line+g"
    assert by["g0.1"]["saves_dangerous"] == 1.0 and by["g0.1"]["acts_harmless"] == 0.0
    assert out["threat"]["level"] == 0.05 and out["threat"]["no_return_level"] == FR.none
    assert by["g0.001"]["elim_minute"] < by["g0.3"]["elim_minute"]                # too early goes before can't save
    assert by["none"]["elim_minute"] == 1441 and all(by[i]["elim_minute"] is None for i in out["useful"])
    assert [p["plan"] for p in out["plans"]][:2] == ["line+g", "g0.1"] and out["plans"][0]["rank"] == 1
    assert "became a threat at 5% asked" in out["action_window"]["text"]
    assert by["g0.1"]["open_by_seed"] == [True, True] and by["none"]["open_by_seed"] == [False, False]


def test_the_script_rescores_in_place_and_keeps_the_original(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("rescore_useful", "scripts/rescore_useful.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = tmp_path / "live_search_10.json"
    path.write_text(json.dumps(_search()), encoding="utf-8")
    first = mod.rescore_file(str(path))
    assert (tmp_path / "live_search_10.raw.json").exists() and first["mode"] == "useful" and first["series"]
    second = mod.rescore_file(str(path))   # twice: the original is reused, nothing changes
    assert second["useful"] == first["useful"] and json.loads((tmp_path / "live_search_10.raw.json").read_text())["pick"] == "g0.001"


def _regulator(flags: list[str]):
    from contracts.schemas import Scenario
    from engine.live_overrides import live_flags
    from mocks.bus import EventBus
    from mocks.l4_system import RegulatorFake

    bus, reg = EventBus(), RegulatorFake("M20")
    with live_flags(flags):
        reg.attach(bus, Scenario(preset="tech-concentrated", seed=1))
    return reg, bus


def test_the_regulator_closes_on_what_the_bank_can_pay() -> None:
    from contracts.events import BankClosed, PolicyEnacted, WithdrawalRequested
    from mocks.support import deposits_base

    flag = "M20 closes on what the bank can pay"
    D = deposits_base()
    for asked, line, closes in ((0.20, False, False), (0.30, False, True), (0.60, True, False), (0.82, True, True)):
        reg, bus = _regulator([flag])
        if line:
            bus.publish(PolicyEnacted(t=1.0, seed=1, bank_id="bank-fvb", lever="emergency_lending", scope="bank-fvb", cost_model={}))
        bus.publish(WithdrawalRequested(t=2.0, seed=1, bank_id="bank-fvb", depositor_id="x", amount=asked * D, channel="wire"))
        out = reg.step(3.0)
        assert any(isinstance(e, BankClosed) for e in out) is closes, (asked, line)
    off, _ = _regulator([])
    assert off._can_pay is None   # off by default: the old test, as before


def test_the_page_shows_why_and_ranks_the_useful_plans() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    for needle in ("ACTS TOO EARLY", "not allowed for US banks", "didn't save the bank", "MOST USEFUL", "useful plans`",
                   "point of no return", "the run becomes dangerous", "How the plans are judged",
                   "out first: rules that act before the run is a threat"):
        assert needle in html, needle
