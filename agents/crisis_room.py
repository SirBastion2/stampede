"""M24 crisis comms room — asyncio pipeline (§5, §18.6, §19)."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from agents.llm_cache import (
    LlmCache,
    cached_or_offline,
    complete_json_llm,
    live_json_completion_enabled,
)
from contracts.events import FutureCompleted, StatementIssued
from contracts.schemas import ClaimsLedger, Scenario, Statement
from params.loader import RegistryError, get_parameter

_PARAM_STATEMENT_COUNT = "Crisis writer candidate statements"
_CHANNELS: tuple[str, ...] = ("press_release", "x_post", "ceo_video", "direct_calls")
_SEGMENTS: tuple[str, ...] = (
    "startup",
    "vc",
    "founder",
    "treasurer",
    "retail",
)


@dataclass(frozen=True, slots=True)
class VerdictRow:
    stmt_id: str
    verdict: Literal["accepted", "rejected"]
    reason: str


@dataclass(frozen=True, slots=True)
class CrisisConfig:
    statement_count: int


def load_crisis_config(path: Path | None = None) -> CrisisConfig:
    try:
        row = get_parameter(_PARAM_STATEMENT_COUNT, path)
        match = re.search(r"\d+", row.value)
        count = int(match.group(0)) if match else 32
    except RegistryError:
        count = 32
    return CrisisConfig(statement_count=max(2, count))


def build_claims_ledger(scenario: Scenario, events: list[BaseModel], t: float) -> ClaimsLedger:
    from mocks.support import (
        FHLB_TOTAL_FRACTION,
        OPENING_CET1,
        bank_id_for,
        fold_book,
        scenario_deposits,
        uninsured_share,
    )

    deposits = scenario_deposits(scenario)
    book = fold_book(events, deposits=deposits, opening_cet1=OPENING_CET1)
    return ClaimsLedger(
        bank_id=bank_id_for(scenario.preset),
        cash=book.cash,
        capital_ratio=book.cet1,
        insured_share=max(0.0, 1.0 - uninsured_share(scenario.preset)),
        facility_capacity=deposits * FHLB_TOTAL_FRACTION,
        as_of=t,
    )


def _parse_claim(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "vague", "text": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"type": "vague", "text": str(raw)}


def _numeric_ok(claim: dict[str, Any], ledger: ClaimsLedger) -> tuple[bool, str]:
    metric = str(claim.get("metric", "")).lower()
    op = str(claim.get("op", "lte")).lower()
    try:
        value = float(claim["value"])
    except (KeyError, TypeError, ValueError):
        return False, "numeric claim missing value"
    actual: float | None = None
    if metric == "cash":
        actual = ledger.cash
    elif metric in {"capital", "capital_ratio", "cet1"}:
        actual = ledger.capital_ratio
    elif metric == "insured_share":
        actual = ledger.insured_share
    elif metric == "facility_capacity":
        actual = ledger.facility_capacity
    if actual is None:
        return False, f"unknown numeric metric: {metric}"
    ok = False
    if op in {"lte", "<="}:
        ok = actual <= value
    elif op in {"lt", "<"}:
        ok = actual < value
    elif op in {"gte", ">="}:
        ok = actual >= value
    elif op in {"gt", ">"}:
        ok = actual > value
    elif op in {"eq", "=="}:
        ok = abs(actual - value) < 1e-6 * max(1.0, abs(value))
    else:
        return False, f"unknown numeric operator: {op}"
    if ok:
        return True, ""
    if metric == "cash":
        return False, "cash claim exceeds the ledger"
    return False, f"{metric} claim conflicts with the ledger"


def _policy_ok(
    claim: dict[str, Any],
    scenario: Scenario,
    events: list[BaseModel],
) -> tuple[bool, str]:
    from mocks.support import canonical_lever, levers_in

    lever = canonical_lever(str(claim.get("lever", "")))
    active = {canonical_lever(name) for name in levers_in(events)}
    for policy in scenario.policies:
        if isinstance(policy, dict) and "lever" in policy:
            active.add(canonical_lever(str(policy["lever"])))
    if lever in active:
        return True, ""
    if lever == "guarantee_deposits":
        return False, "guarantee not in effect"
    return False, f"{lever} is not in effect"


def _vague_ok(
    claim: dict[str, Any],
    ledger: ClaimsLedger,
    cache: LlmCache | None,
) -> tuple[bool, str]:
    text = str(claim.get("text", ""))

    def offline(_payload: dict[str, Any]) -> dict[str, str]:
        lowered = text.lower()
        if "guarantee" in lowered and "all deposit" in lowered:
            return {"verdict": "rejected", "reason": "guarantee not in effect"}
        if "fully insured" in lowered and ledger.insured_share < 0.5:
            return {
                "verdict": "rejected",
                "reason": "insured share claim conflicts with the ledger",
            }
        return {"verdict": "accepted", "reason": ""}

    payload = {"text": text, "ledger": ledger.model_dump()}

    def producer() -> dict[str, str]:
        prompt = json.dumps({"task": "m24_truth_filter", **payload}, sort_keys=True)
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["accepted", "rejected"]},
                "reason": {"type": "string"},
            },
            "required": ["verdict"],
            "additionalProperties": False,
        }
        raw = complete_json_llm(prompt, "m24-truth-filter", json_schema=schema)
        verdict = str(raw.get("verdict", "")).lower()
        if verdict not in {"accepted", "rejected"}:
            raise ValueError("invalid truth-filter verdict")
        return {"verdict": verdict, "reason": str(raw.get("reason", ""))}

    result = cached_or_offline(
        "m24-truth-filter",
        payload,
        offline,
        producer=producer if live_json_completion_enabled() else None,
        cache=cache,
    )
    verdict = str(result.get("verdict", "accepted"))
    reason = str(result.get("reason", ""))
    if verdict == "rejected":
        return False, reason or "vague wording rejected"
    return True, ""


def check_claim(
    raw: str,
    ledger: ClaimsLedger,
    scenario: Scenario,
    events: list[BaseModel],
    *,
    cache: LlmCache | None = None,
) -> tuple[bool, str]:
    claim = _parse_claim(raw)
    kind = str(claim.get("type", "vague")).lower()
    if kind == "numeric":
        return _numeric_ok(claim, ledger)
    if kind == "policy":
        return _policy_ok(claim, scenario, events)
    return _vague_ok(claim, ledger, cache)


def filter_statement(
    statement: Statement,
    ledger: ClaimsLedger,
    scenario: Scenario,
    events: list[BaseModel],
    *,
    cache: LlmCache | None = None,
) -> VerdictRow:
    for raw in statement.claims:
        ok, reason = check_claim(raw, ledger, scenario, events, cache=cache)
        if not ok:
            return VerdictRow(
                stmt_id=statement.id,
                verdict="rejected",
                reason=reason or "claim rejected",
            )
    return VerdictRow(stmt_id=statement.id, verdict="accepted", reason="")


def _claim_json(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields}, sort_keys=True)


def draft_statements(
    seed: int,
    ledger: ClaimsLedger,
    config: CrisisConfig | None = None,
) -> list[Statement]:
    """Deterministic crisis writer (offline template path; §7 fallback)."""
    from mocks.support import seed_scale

    cfg = config or load_crisis_config()
    statements: list[Statement] = []
    false_cash = ledger.cash * 10.0 + 1.0
    statements.append(
        Statement(
            id="stmt-crisis-1",
            text="We have liquidity on hand and remain well capitalized.",
            claims=[
                _claim_json("numeric", metric="cash", op="gte", value=ledger.cash * 0.5),
                _claim_json(
                    "numeric",
                    metric="capital_ratio",
                    op="gte",
                    value=ledger.capital_ratio * 0.5,
                ),
            ],
            channel="x_post",
        )
    )
    statements.append(
        Statement(
            id="stmt-crisis-false-cash",
            text="Cash reserves exceed ten times on-hand liquidity.",
            claims=[_claim_json("numeric", metric="cash", op="gte", value=false_cash)],
            channel="press_release",
        )
    )
    for index in range(2, cfg.statement_count):
        tilt = seed_scale(seed, f"crisis-stmt-{index}")
        channel = _CHANNELS[index % len(_CHANNELS)]
        stmt_id = f"stmt-crisis-{index + 1}"
        guarantee = index % 7 == 0
        if guarantee:
            claims = [_claim_json("policy", lever="guarantee_deposits")]
            text = "All deposits are fully guaranteed."
        else:
            threshold = ledger.cash * (0.8 + 0.05 * tilt)
            claims = [_claim_json("numeric", metric="cash", op="gte", value=threshold)]
            text = f"Statement strategy {index}: liquidity reassurance."
        statements.append(
            Statement(
                id=stmt_id,
                text=text,
                claims=claims,
                channel=channel,
            )
        )
    return statements


def rank_statements(
    survivors: list[Statement],
    futures: list[FutureCompleted],
) -> Statement | None:
    del futures
    if not survivors:
        return None
    for stmt in survivors:
        if stmt.id == "stmt-crisis-1":
            return stmt
    return survivors[0]


async def run_crisis_pipeline_async(
    scenario: Scenario,
    events: list[BaseModel],
    futures: list[FutureCompleted],
    t: float,
    *,
    cache: LlmCache | None = None,
) -> tuple[list[VerdictRow], Statement | None]:
    ledger = build_claims_ledger(scenario, events, t)
    config = load_crisis_config()

    async def writer() -> list[Statement]:
        return draft_statements(scenario.seed, ledger, config)

    async def filter_all(statements: list[Statement]) -> tuple[list[VerdictRow], list[Statement]]:
        verdicts: list[VerdictRow] = []
        survivors: list[Statement] = []
        for stmt in statements:
            row = filter_statement(stmt, ledger, scenario, events, cache=cache)
            verdicts.append(row)
            if row.verdict == "accepted":
                survivors.append(stmt)
        return verdicts, survivors

    statements = await writer()
    verdicts, survivors = await filter_all(statements)
    winner = rank_statements(survivors, futures)
    return verdicts, winner


def run_crisis_pipeline(
    scenario: Scenario,
    events: list[BaseModel],
    futures: list[FutureCompleted],
    t: float,
    *,
    cache: LlmCache | None = None,
) -> tuple[list[VerdictRow], Statement | None]:
    return asyncio.run(run_crisis_pipeline_async(scenario, events, futures, t, cache=cache))


class CrisisRoomModule:
    """M24 scheduler module: truth filter and ranked statement pick (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("StatementIssued",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "FutureCompleted")

    def __init__(self, module_id: str = "M24", *, cache: LlmCache | None = None) -> None:
        self.module_id = module_id
        self._cache = cache
        self.scenario: Scenario | None = None
        self.bus: Any = None
        self.verdicts: list[dict[str, str]] = []
        self._sent = False
        self.inbox: list[BaseModel] = []
        self._cursor = 0

    def attach(self, bus: Any, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        return self.on_tick(t, new_events)

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        futures = [event for event in new_events if isinstance(event, FutureCompleted)]
        if self._sent or not futures or self.scenario is None or self.bus is None:
            return []
        self._sent = True
        events = list(self.bus.log)
        verdict_rows, winner = run_crisis_pipeline(
            self.scenario,
            events,
            futures,
            t,
            cache=self._cache,
        )
        self.verdicts = [
            {"stmt_id": row.stmt_id, "verdict": row.verdict, "reason": row.reason}
            for row in verdict_rows
        ]
        if winner is None:
            return []
        segments = ["startup", "founder", "treasurer"]
        from mocks.support import bank_id_for

        return [
            StatementIssued(
                t=t,
                seed=self.scenario.seed,
                bank_id=bank_id_for(self.scenario.preset),
                cause_id=f"{self.module_id}-crisis-1",
                stmt_id=winner.id,
                channel=winner.channel,
                segments=segments,
            )
        ]
