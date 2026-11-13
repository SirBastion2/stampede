"""Contract tests: one §6.1 round-trip per event; §6.2 schemas match the doc."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contracts import EVENT_MODELS, parse_event
from contracts.events import EventEnvelope, Tick
from contracts.schemas import (
    Bank,
    BankAssets,
    BankCapital,
    BankLiabilities,
    ClaimsLedger,
    CrowdFrame,
    CrowdPanels,
    DepositorArrays,
    FutureResult,
    MarketScreenFrame,
    Rumor,
    Scenario,
    ScenarioShock,
    Security,
    Statement,
)

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "ARCHITECTURE.md"
DOC_TEXT = DOC_PATH.read_text()

ENVELOPE = {"t", "seed", "bank_id", "cause_id", "event_type"}

# Top-level fields named in the §6.2 rows (nested object keys are checked apart).
SCHEMA_FIELDS: dict[str, set[str]] = {
    "Bank": {
        "id",
        "name",
        "ticker",
        "preset",
        "assets",
        "liabilities",
        "capital",
        "ladder",
        "stock_ref",
    },
    "DepositorArrays": {
        "bank_idx",
        "segment",
        "balance",
        "insured",
        "belief",
        "threshold",
        "attention",
        "needs_flag",
        "has_second_bank",
        "state",
    },
    "Security": {
        "class",
        "maturity_bucket",
        "duration",
        "book_value",
        "market_value",
        "htm",
    },
    "Rumor": {"id", "claim", "lineage", "credibility", "reach", "notes"},
    "Statement": {"id", "text", "claims", "channel", "verdict", "reasons"},
    "ClaimsLedger": {
        "bank_id",
        "cash",
        "capital_ratio",
        "insured_share",
        "facility_capacity",
        "as_of",
    },
    "Scenario": {
        "preset",
        "seed",
        "rate_path",
        "shocks",
        "rumors",
        "policies",
        "fidelity",
    },
    "FutureResult": {
        "future_id",
        "failed",
        "t_fail",
        "deposits_kept",
        "price_path_band",
        "cost",
    },
    "CrowdPanels": {
        "cash_gauge",
        "time_to_failure_ticks",
        "tipping_point",
        "rumor",
        "state_ribbon",
        "calm_count",
        "wavering_count",
        "withdrew_count",
    },
    "CrowdFrame": {
        "t",
        "seed",
        "bank_id",
        "point_count",
        "positions_xy",
        "state",
        "balance_norm",
        "uninsured_mask",
        "panels",
    },
    "MarketScreenFrame": {
        "module",
        "t",
        "bank_id",
        "prices",
        "volumes",
        "depth",
        "halt",
        "markers",
        "statements",
        "futures",
    },
}

NESTED_FIELDS: dict[str, set[str]] = {
    "BankAssets": {"cash", "reserves", "securities", "loans", "other"},
    "BankLiabilities": {"deposits_by_segment", "fhlb", "brokered", "other"},
    "BankCapital": {"cet1", "aoci_toggle"},
}


def _slice(start: str, end: str) -> str:
    begin = DOC_TEXT.index(start)
    return DOC_TEXT[begin : DOC_TEXT.index(end, begin)]


def _payload_names(cell: str) -> set[str]:
    cell = re.sub(r"\([^)]*\)", "", cell)
    names: set[str] = set()
    for part in cell.split(","):
        token = part.strip().replace("[]", "").replace("{}", "").replace("?", "")
        if token:
            names.add(token)
    return names


def event_payloads() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in _slice("## 6.1", "## 6.2").splitlines():
        if not line.startswith("| ") or line.startswith("| Event") or line.startswith("| :"):
            continue
        cells = [c.strip().replace("\\", "") for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1]
    return rows


def schema_cells() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in _slice("## 6.2", "# 7\\").splitlines():
        match = re.match(r"\| `([^`]+)` \| `([^`]+)` \|", line)
        if not match:
            continue
        name = match.group(1).split(" (", 1)[0]
        if name == "Schema":
            continue
        rows[name] = match.group(2)
    return rows


def alias_names(model: type) -> set[str]:
    return {(info.alias or name) for name, info in model.model_fields.items()}


DOC_EVENTS = event_payloads()
DOC_SCHEMAS = schema_cells()

_SAMPLES: dict[str, dict[str, Any]] = {
    "Tick": {},
    "RatesChanged": {"curve": [0.01, 0.02], "policy_rate": 0.05},
    "RumorPosted": {
        "rumor_id": "r1",
        "claim": "test",
        "source_node": "n1",
        "credibility": 0.5,
    },
    "AttentionTick": {"level": 0.1, "delta_4h": 0.01},
    "HeadlinePublished": {"text": "h", "reach": 100},
    "WithdrawalRequested": {"depositor_id": "d1", "amount": 1.0, "channel": "wire"},
    "PaymentSettled": {"amount": 1.0, "channel": "wire", "queued_remaining": 0},
    "LiquidityDrawn": {"source": "fhlb", "amount": 1e6, "eta": 1.0},
    "AssetSold": {"class": "ust", "amount": 1e6, "price_impact": 0.001},
    "LossRealized": {"amount": 1e5, "kind": "securities"},
    "CapitalChanged": {"cet1_ratio": 0.12},
    "PriceTick": {"price": 10.0, "volume": 1000.0},
    "TradingHalt": {"reason": "volatility", "duration": 300.0},
    "StatementIssued": {"stmt_id": "s1", "channel": "press", "segments": ["all"]},
    "PolicyEnacted": {"lever": "guarantee", "scope": "deposits", "cost_model": {}},
    "BankClosed": {"reason": "liquidity"},
    "FutureCompleted": {"future_id": "f0", "metrics": {"failed": False}},
    "DepositRateChanged": {"segment": "retail", "rate": 0.04},
    "DirectiveIssued": {"vc_id": "vc1", "portfolio_ids": ["p1"], "action": "move"},
    "VIPDecision": {
        "agent_id": "v1",
        "action": "withdraw",
        "amount": 1e6,
        "rationale": "risk",
    },
    "CEOAction": {"action": "statement", "params": {}, "rationale": "calm"},
    "ThesisPublished": {"author_id": "a1", "reach": 500},
    "RumorMutated": {"rumor_id": "r1", "claim": "mut", "exaggeration_score": 0.2},
    "AnalystNote": {"text": "note", "cited_event_ids": ["e1"]},
    "SupervisoryAction": {"action": "consent-order"},
}


def _event_payload(event_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "t": 0.0,
        "seed": 1,
        "bank_id": "bank-1",
        "cause_id": None,
        "event_type": event_name,
    }
    payload.update(_SAMPLES[event_name])
    return payload


@pytest.mark.parametrize("event_name", sorted(DOC_EVENTS))
def test_event_roundtrip_via_parse_event(event_name: str) -> None:
    """Build, serialize, and parse_event one §6.1 event; fields match the doc row."""
    model = EVENT_MODELS[event_name]
    documented = _payload_names(DOC_EVENTS[event_name])
    assert alias_names(model) == ENVELOPE | documented
    assert model.model_fields["bank_id"].is_required() is False
    assert str(model.model_fields["event_type"].default) == event_name

    if event_name == "LossRealized":
        match = re.search(r"\(([^)]*)\)", DOC_EVENTS[event_name])
        assert match is not None
        kinds = {part.strip() for part in match.group(1).split(",")}
        assert set(get_args(model.model_fields["kind"].annotation)) == kinds

    instance = model.model_validate(_event_payload(event_name))
    blob = json.loads(instance.model_dump_json(by_alias=True))
    parsed = parse_event(blob)
    assert type(parsed) is model
    assert parsed.model_dump() == instance.model_dump()
    assert set(blob) == alias_names(model)


def test_event_registry_matches_doc() -> None:
    assert set(EVENT_MODELS) == set(DOC_EVENTS)
    assert set(_SAMPLES) == set(DOC_EVENTS)


def test_parse_event_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown event_type"):
        parse_event({"event_type": "NotAnEvent", "t": 0.0, "seed": 1})


@given(st.integers(min_value=0, max_value=10_000), st.integers(min_value=0, max_value=2**31 - 1))
def test_tick_hypothesis(t: int, seed: int) -> None:
    ev = Tick(t=float(t), seed=seed)
    assert EventEnvelope.model_validate(ev.model_dump()).t == float(t)


def _tokens(cell: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cell.replace("?", "")))


def test_schema_table_covers_section_6_2() -> None:
    assert set(DOC_SCHEMAS) == set(SCHEMA_FIELDS)


@pytest.mark.parametrize("name", sorted(SCHEMA_FIELDS))
def test_schema_fields_match_doc(name: str) -> None:
    model = {
        "Bank": Bank,
        "DepositorArrays": DepositorArrays,
        "Security": Security,
        "Rumor": Rumor,
        "Statement": Statement,
        "ClaimsLedger": ClaimsLedger,
        "Scenario": Scenario,
        "FutureResult": FutureResult,
        "CrowdPanels": CrowdPanels,
        "CrowdFrame": CrowdFrame,
        "MarketScreenFrame": MarketScreenFrame,
    }[name]
    cell = DOC_SCHEMAS[name]
    aliases = alias_names(model)
    assert aliases == SCHEMA_FIELDS[name]
    assert aliases <= _tokens(cell)
    if name != "Bank":
        assert aliases == _tokens(cell)


def test_bank_nested_fields_match_doc() -> None:
    tokens = _tokens(DOC_SCHEMAS["Bank"])
    assert alias_names(BankAssets) == NESTED_FIELDS["BankAssets"]
    assert alias_names(BankLiabilities) == NESTED_FIELDS["BankLiabilities"]
    assert alias_names(BankCapital) == NESTED_FIELDS["BankCapital"]
    assert NESTED_FIELDS["BankAssets"] <= tokens
    assert NESTED_FIELDS["BankLiabilities"] <= tokens
    assert NESTED_FIELDS["BankCapital"] <= tokens
    assert {"insured", "uninsured"} <= tokens
    assert "t_fail?" in DOC_SCHEMAS["FutureResult"]
    assert FutureResult.model_fields["t_fail"].is_required() is False


def _bank() -> Bank:
    return Bank(
        id="b1",
        name="Fictional Valley Bank",
        ticker="FVB",
        preset="tech-concentrated",
        assets=BankAssets(
            cash=1.0,
            reserves=2.0,
            securities=[
                Security(
                    class_="ust",
                    maturity_bucket="5y",
                    duration=6.0,
                    book_value=10.0,
                    market_value=8.0,
                    htm=True,
                )
            ],
            loans=[{"type": "cre", "balance": 3.0}],
            other=0.5,
        ),
        liabilities=BankLiabilities(
            deposits_by_segment={"insured": 1.0, "uninsured": 9.0},
            fhlb=0.2,
            brokered=0.1,
            other=0.0,
        ),
        capital=BankCapital(cet1=0.1, aoci_toggle=True),
        ladder=[],
        stock_ref="FVB",
    )


def _depositor_arrays() -> DepositorArrays:
    return DepositorArrays(
        bank_idx=[0],
        segment=["founder"],
        balance=[1.0],
        insured=[250_000.0],
        belief=[0.2],
        threshold=[0.5],
        attention=[0.1],
        needs_flag=[False],
        has_second_bank=[True],
        state=["calm"],
    )


def _security() -> Security:
    return Security(
        class_="ust",
        maturity_bucket="5y",
        duration=6.0,
        book_value=10.0,
        market_value=8.0,
        htm=False,
    )


def _rumor() -> Rumor:
    return Rumor(
        id="r1",
        claim="queues are long",
        lineage=["r0"],
        credibility=0.4,
        reach=12,
        notes=["n"],
    )


def _statement() -> Statement:
    return Statement(
        id="s1",
        text="cash is on hand",
        claims=["cash"],
        channel="press",
        verdict="accepted",
        reasons=[],
    )


def _ledger() -> ClaimsLedger:
    return ClaimsLedger(
        bank_id="b1",
        cash=1.0,
        capital_ratio=0.12,
        insured_share=0.07,
        facility_capacity=5.0,
        as_of=9.0,
    )


def _scenario() -> Scenario:
    return Scenario(
        preset="tech-concentrated",
        seed=42,
        rate_path=[(0.0, 0.01), (1.0, 0.02)],
        shocks=[ScenarioShock(kind="rumor", at=1.0, params={"who": "hub"})],
        rumors=[_rumor()],
        policies=[{"lever": "guarantee"}],
        fidelity="demo",
    )


def _future() -> FutureResult:
    return FutureResult(
        future_id="f0",
        failed=False,
        t_fail=None,
        deposits_kept=0.8,
        price_path_band=(9.0, 11.0),
        cost=1.5,
    )


def _market_frame() -> MarketScreenFrame:
    return MarketScreenFrame(
        t=1.0,
        bank_id="bank-fvb",
        prices=[(0.0, 42.0), (1.0, 41.0)],
        volumes=[1000.0, 1200.0],
    )


_SCHEMA_BUILDERS = {
    "Bank": _bank,
    "DepositorArrays": _depositor_arrays,
    "Security": _security,
    "Rumor": _rumor,
    "Statement": _statement,
    "ClaimsLedger": _ledger,
    "Scenario": _scenario,
    "FutureResult": _future,
    "MarketScreenFrame": _market_frame,
}


@pytest.mark.parametrize("name", sorted(_SCHEMA_BUILDERS))
def test_schema_roundtrip(name: str) -> None:
    instance = _SCHEMA_BUILDERS[name]()
    again = type(instance).model_validate(instance.model_dump(by_alias=True))
    assert again == instance
    assert alias_names(type(instance)) == SCHEMA_FIELDS[name]
