"""§6.1 event bus — typed payloads; every event carries t, seed, bank_id?, cause_id?."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

_EVENT_CLASSES: tuple[type[BaseModel], ...] = (
    # defined below after classes
)


class EventEnvelope(BaseModel):
    """Fields present on every published event."""

    t: float
    seed: int
    bank_id: str | None = None
    cause_id: str | None = None


class Tick(EventEnvelope):
    event_type: Literal["Tick"] = "Tick"


class RatesChanged(EventEnvelope):
    event_type: Literal["RatesChanged"] = "RatesChanged"
    curve: list[float]
    policy_rate: float


class RumorPosted(EventEnvelope):
    event_type: Literal["RumorPosted"] = "RumorPosted"
    rumor_id: str
    claim: str
    source_node: str
    credibility: float


class AttentionTick(EventEnvelope):
    event_type: Literal["AttentionTick"] = "AttentionTick"
    level: float
    delta_4h: float


class HeadlinePublished(EventEnvelope):
    event_type: Literal["HeadlinePublished"] = "HeadlinePublished"
    text: str
    reach: int


class WithdrawalRequested(EventEnvelope):
    event_type: Literal["WithdrawalRequested"] = "WithdrawalRequested"
    depositor_id: str
    amount: float
    channel: str


class PaymentSettled(EventEnvelope):
    event_type: Literal["PaymentSettled"] = "PaymentSettled"
    amount: float
    channel: str
    queued_remaining: int


class LiquidityDrawn(EventEnvelope):
    event_type: Literal["LiquidityDrawn"] = "LiquidityDrawn"
    source: str
    amount: float
    eta: float


class AssetSold(EventEnvelope):
    event_type: Literal["AssetSold"] = "AssetSold"
    asset_class: str = Field(alias="class")
    amount: float
    price_impact: float

    model_config = {"populate_by_name": True}


class LossRealized(EventEnvelope):
    event_type: Literal["LossRealized"] = "LossRealized"
    amount: float
    kind: Literal["securities", "credit"]


class CapitalChanged(EventEnvelope):
    event_type: Literal["CapitalChanged"] = "CapitalChanged"
    cet1_ratio: float


class PriceTick(EventEnvelope):
    event_type: Literal["PriceTick"] = "PriceTick"
    price: float
    volume: float


class TradingHalt(EventEnvelope):
    event_type: Literal["TradingHalt"] = "TradingHalt"
    reason: str
    duration: float


class StatementIssued(EventEnvelope):
    event_type: Literal["StatementIssued"] = "StatementIssued"
    stmt_id: str
    channel: str
    segments: list[str]


class PolicyEnacted(EventEnvelope):
    event_type: Literal["PolicyEnacted"] = "PolicyEnacted"
    lever: str
    scope: str
    cost_model: dict[str, Any]


class BankClosed(EventEnvelope):
    event_type: Literal["BankClosed"] = "BankClosed"
    reason: str


class FutureCompleted(EventEnvelope):
    event_type: Literal["FutureCompleted"] = "FutureCompleted"
    future_id: str
    metrics: dict[str, Any]


class DepositRateChanged(EventEnvelope):
    event_type: Literal["DepositRateChanged"] = "DepositRateChanged"
    segment: str
    rate: float


class DirectiveIssued(EventEnvelope):
    event_type: Literal["DirectiveIssued"] = "DirectiveIssued"
    vc_id: str
    portfolio_ids: list[str]
    action: str


class VIPDecision(EventEnvelope):
    event_type: Literal["VIPDecision"] = "VIPDecision"
    agent_id: str
    action: str
    amount: float
    rationale: str


class CEOAction(EventEnvelope):
    event_type: Literal["CEOAction"] = "CEOAction"
    action: str
    params: dict[str, Any]
    rationale: str


class ThesisPublished(EventEnvelope):
    event_type: Literal["ThesisPublished"] = "ThesisPublished"
    author_id: str
    reach: int


class RumorMutated(EventEnvelope):
    event_type: Literal["RumorMutated"] = "RumorMutated"
    rumor_id: str
    claim: str
    exaggeration_score: float


class AnalystNote(EventEnvelope):
    event_type: Literal["AnalystNote"] = "AnalystNote"
    text: str
    cited_event_ids: list[str]


class SupervisoryAction(EventEnvelope):
    event_type: Literal["SupervisoryAction"] = "SupervisoryAction"
    action: str


_EVENT_CLASSES = (
    Tick,
    RatesChanged,
    RumorPosted,
    AttentionTick,
    HeadlinePublished,
    WithdrawalRequested,
    PaymentSettled,
    LiquidityDrawn,
    AssetSold,
    LossRealized,
    CapitalChanged,
    PriceTick,
    TradingHalt,
    StatementIssued,
    PolicyEnacted,
    BankClosed,
    FutureCompleted,
    DepositRateChanged,
    DirectiveIssued,
    VIPDecision,
    CEOAction,
    ThesisPublished,
    RumorMutated,
    AnalystNote,
    SupervisoryAction,
)

SimEvent = Annotated[
    Tick
    | RatesChanged
    | RumorPosted
    | AttentionTick
    | HeadlinePublished
    | WithdrawalRequested
    | PaymentSettled
    | LiquidityDrawn
    | AssetSold
    | LossRealized
    | CapitalChanged
    | PriceTick
    | TradingHalt
    | StatementIssued
    | PolicyEnacted
    | BankClosed
    | FutureCompleted
    | DepositRateChanged
    | DirectiveIssued
    | VIPDecision
    | CEOAction
    | ThesisPublished
    | RumorMutated
    | AnalystNote
    | SupervisoryAction,
    Field(discriminator="event_type"),
]

EVENT_MODELS: dict[str, type[BaseModel]] = {
    str(cls.model_fields["event_type"].default): cls for cls in _EVENT_CLASSES
}


def event_type_for_model(model: type[BaseModel]) -> str:
    default = model.model_fields.get("event_type")
    if default is None or default.default is None:
        raise ValueError(f"No event_type on {model.__name__}")
    return str(default.default)


def parse_event(data: dict[str, Any]) -> BaseModel:
    event_type = data.get("event_type")
    if not isinstance(event_type, str) or event_type not in EVENT_MODELS:
        raise ValueError(f"Unknown event_type: {event_type!r}")
    return EVENT_MODELS[event_type].model_validate(data)
