"""M20 regulators: closure rules, timing, determinism, real-vs-fake parity."""

from __future__ import annotations

from typing import Any

from contracts import parse_event
from contracts.events import BankClosed
from contracts.schemas import Rumor, Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from engine.system.regulators import (
    RegulatorConfig,
    RegulatorModule,
    closure_reason,
    closure_timing_allows,
    load_regulator_config,
    simulated_day_index,
    watch_recommended,
)
from mocks.l4_system import RegulatorFake
from mocks.support import Book, calendar_at, deposits_base
from tests.test_fakes import panic
from ui.live.pacing import crisis_sim_minutes


def _cfg() -> RegulatorConfig:
    return load_regulator_config()


def _m20_payloads(log: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in log:
        if event.event_type in {"BankClosed", "SupervisoryAction"}:
            payload = event.model_dump(mode="json", by_alias=True)
            parse_event(payload)
            out.append(payload)
    return out


def test_svb_like_closes_without_policy_by_sim_day_two() -> None:
    scenario = Scenario(
        preset="tech-concentrated",
        seed=3,
        rate_path=[(0.0, 0.04), (3.0, 0.06)],
        rumors=[
            Rumor(
                id="rumor-1",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                credibility=0.85,
                reach=12000,
            )
        ],
        policies=[],
    )
    closed_t: float | None = None
    for steps in range(1, 4 * 24 * 60):
        log = run_scenario(scenario, steps, modes={"M20": ModuleMode.REAL})
        closed = [e for e in log if isinstance(e, BankClosed)]
        if closed:
            closed_t = closed[-1].t
            break
    assert closed_t is not None
    assert simulated_day_index(closed_t) <= 1


def test_real_matches_fake_regulator_events() -> None:
    scenario = panic(9)
    steps = 12
    fake_log = run_scenario(scenario, steps, modes={"M20": ModuleMode.FAKE})
    real_log = run_scenario(scenario, steps, modes={"M20": ModuleMode.REAL})
    assert _m20_payloads(fake_log) == _m20_payloads(real_log)


def test_closure_rules_unit() -> None:
    cfg = _cfg()
    solvent = Book(
        cash=1e6,
        securities=1e9,
        deposits_remaining=deposits_base(),
        borrowings=0.0,
        equity=deposits_base() * cfg.opening_cet1,
        policy_rate=0.04,
        cet1=cfg.opening_cet1,
        unpaid=0.0,
    )
    assert closure_reason(solvent, cfg) is None
    assert not watch_recommended(solvent, cfg)

    illiquid = Book(
        cash=-5.0,
        securities=0.0,
        deposits_remaining=deposits_base(),
        borrowings=0.0,
        equity=deposits_base() * cfg.opening_cet1,
        policy_rate=0.04,
        cet1=cfg.opening_cet1,
        unpaid=5.0,
    )
    assert closure_reason(illiquid, cfg) == "unable to pay obligations"

    weak_capital = Book(
        cash=1e6,
        securities=1e9,
        deposits_remaining=deposits_base(),
        borrowings=0.0,
        equity=deposits_base() * (cfg.cet1_floor - 0.01),
        policy_rate=0.04,
        cet1=cfg.cet1_floor - 0.01,
        unpaid=0.0,
    )
    assert closure_reason(weak_capital, cfg) == "capital below threshold"


def test_weekend_timing_for_crypto_preset() -> None:
    business_t = 120.0
    assert calendar_at(business_t).weekend is False
    assert closure_timing_allows(business_t, "crypto-heavy follower") is False
    weekend_t = 24 * 60 * 5 + 600.0
    assert calendar_at(weekend_t).weekend is True
    assert closure_timing_allows(weekend_t, "crypto-heavy follower") is True
    assert closure_timing_allows(business_t, "tech-concentrated") is True


def test_build_engine_m20_real_mode() -> None:
    engine = build_engine(modes={"M20": ModuleMode.REAL}, scenario=panic(1))
    assert isinstance(engine["M20"], RegulatorModule)
    assert not isinstance(engine["M20"], RegulatorFake)
    log = engine.run(int(crisis_sim_minutes()))
    assert any(isinstance(event, BankClosed) for event in log)


def test_regulator_deterministic() -> None:
    scenario = panic(4)
    first = _m20_payloads(run_scenario(scenario, 10, modes={"M20": ModuleMode.REAL}))
    second = _m20_payloads(run_scenario(scenario, 10, modes={"M20": ModuleMode.REAL}))
    assert first == second
