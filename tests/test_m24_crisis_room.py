"""M24 crisis room: truth filter, determinism, real-vs-fake contract."""

from __future__ import annotations

import json

from agents.crisis_room import (
    CrisisRoomModule,
    build_claims_ledger,
    draft_statements,
    filter_statement,
    load_crisis_config,
    run_crisis_pipeline,
)
from contracts.events import FutureCompleted, StatementIssued
from contracts.schemas import Scenario, Statement
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l5_grok import CrisisRoomFake


def _scenario(seed: int = 3, *, policies: list[dict[str, object]] | None = None) -> Scenario:
    return Scenario(
        preset="tech-concentrated",
        seed=seed,
        rate_path=[(0.0, 0.05)],
        policies=policies or [],
    )


def _sample_futures(scenario: Scenario) -> list[FutureCompleted]:
    return [
        FutureCompleted(
            t=5.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="f-0",
            future_id="f00",
            metrics={"failure_probability": 0.2, "deposits_kept": 1.0},
        )
    ]


def test_load_crisis_config_reads_registry_count() -> None:
    cfg = load_crisis_config()
    assert cfg.statement_count == 32


def test_false_cash_statement_rejected() -> None:
    scenario = _scenario()
    ledger = build_claims_ledger(scenario, [], 0.0)
    drafts = draft_statements(scenario.seed, ledger)
    false_stmt = next(s for s in drafts if s.id == "stmt-crisis-false-cash")
    row = filter_statement(false_stmt, ledger, scenario, [])
    assert row.verdict == "rejected"
    assert row.reason == "cash claim exceeds the ledger"


def test_guarantee_claim_rejected_without_policy() -> None:
    scenario = _scenario()
    ledger = build_claims_ledger(scenario, [], 0.0)
    stmt = Statement(
        id="guarantee-claim",
        text="All deposits are fully guaranteed.",
        claims=[json.dumps({"type": "policy", "lever": "guarantee_deposits"})],
        channel="press_release",
    )
    row = filter_statement(stmt, ledger, scenario, [])
    assert row.verdict == "rejected"
    assert "guarantee" in row.reason


def test_guarantee_claim_accepted_when_policy_present() -> None:
    scenario = _scenario(policies=[{"lever": "guarantee_deposits", "at": 0}])
    ledger = build_claims_ledger(scenario, [], 0.0)
    stmt = Statement(
        id="guarantee-ok",
        text="Deposits are guaranteed.",
        claims=[json.dumps({"type": "policy", "lever": "guarantee_deposits"})],
        channel="press_release",
    )
    row = filter_statement(stmt, ledger, scenario, [])
    assert row.verdict == "accepted"


def test_pipeline_is_deterministic() -> None:
    scenario = _scenario(9)
    futures = _sample_futures(scenario)
    first = run_crisis_pipeline(scenario, [], futures, 5.0)
    second = run_crisis_pipeline(scenario, [], futures, 5.0)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[1] is not None
    assert first[1].id == "stmt-crisis-1"


def test_real_module_matches_fake_after_futures() -> None:
    scenario = _scenario(7)
    fake_engine = build_engine(scenario=scenario)
    fake_engine.run(6)
    fake = fake_engine["M24"]
    assert isinstance(fake, CrisisRoomFake)

    real_engine = build_engine(modes={"M24": ModuleMode.REAL}, scenario=scenario)
    real_engine.run(6)
    real = real_engine["M24"]
    assert isinstance(real, CrisisRoomModule)

    def crisis_stmt(log: list[object]) -> StatementIssued:
        return next(
            e
            for e in log
            if isinstance(e, StatementIssued) and e.stmt_id == "stmt-crisis-1"
        )

    fake_stmt = crisis_stmt(fake_engine.bus.log)
    real_stmt = crisis_stmt(real_engine.bus.log)
    assert real_stmt.channel == fake_stmt.channel == "x_post"
    assert real_stmt.segments == fake_stmt.segments

    fake_reject = next(r for r in fake.verdicts if r["verdict"] == "rejected")
    real_reject = next(r for r in real.verdicts if r["verdict"] == "rejected")
    assert fake_reject["stmt_id"] == real_reject["stmt_id"] == "stmt-crisis-false-cash"
    assert fake_reject["reason"] == real_reject["reason"]
