"""§6.2 core schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Security(BaseModel):
    class_: str = Field(alias="class")
    maturity_bucket: str
    duration: float
    book_value: float
    market_value: float
    htm: bool

    model_config = {"populate_by_name": True}


class BankAssets(BaseModel):
    cash: float
    reserves: float
    securities: list[Security]
    loans: list[dict[str, Any]]
    other: float = 0.0


class BankLiabilities(BaseModel):
    deposits_by_segment: dict[str, float]
    fhlb: float = 0.0
    brokered: float = 0.0
    other: float = 0.0


class BankCapital(BaseModel):
    cet1: float
    aoci_toggle: bool = False


class LadderStep(BaseModel):
    source: str
    capacity: float
    eta_hours: float


class Bank(BaseModel):
    id: str
    name: str
    ticker: str
    preset: str
    assets: BankAssets
    liabilities: BankLiabilities
    capital: BankCapital
    ladder: list[LadderStep] = Field(default_factory=list)
    stock_ref: str | None = None


class DepositorArrays(BaseModel):
    """Structure-of-arrays layout (GPU in later waves); one row per depositor."""

    bank_idx: list[int]
    segment: list[str]
    balance: list[float]
    insured: list[float]
    belief: list[float]
    threshold: list[float]
    attention: list[float]
    needs_flag: list[bool]
    has_second_bank: list[bool]
    state: list[str]


class Rumor(BaseModel):
    id: str
    claim: str
    lineage: list[str] = Field(default_factory=list)
    credibility: float
    reach: int = 0
    notes: list[str] = Field(default_factory=list)


class Statement(BaseModel):
    id: str
    text: str
    claims: list[str]
    channel: str
    verdict: Literal["pending", "accepted", "rejected"] = "pending"
    reasons: list[str] = Field(default_factory=list)


class ClaimsLedger(BaseModel):
    bank_id: str
    cash: float
    capital_ratio: float
    insured_share: float
    facility_capacity: float
    as_of: float


class ScenarioShock(BaseModel):
    kind: str
    at: float
    params: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    preset: str
    seed: int
    rate_path: list[tuple[float, float]] = Field(default_factory=list)
    shocks: list[ScenarioShock] = Field(default_factory=list)
    rumors: list[Rumor] = Field(default_factory=list)
    policies: list[dict[str, Any]] = Field(default_factory=list)
    fidelity: str = "demo"


class FutureResult(BaseModel):
    future_id: str
    failed: bool
    t_fail: float | None = None
    deposits_kept: float
    price_path_band: tuple[float, float]
    cost: float


class CrowdPanels(BaseModel):
    """Side-panel gauges for the M29 crowd screen (§5)."""

    cash_gauge: float
    time_to_failure_ticks: float | None = None
    tipping_point: float
    rumor: str = ""
    state_ribbon: Literal["calm", "wavering", "crisis"] = "calm"
    calm_count: int = 0
    wavering_count: int = 0
    withdrew_count: int = 0


class CrowdFrame(BaseModel):
    """One crowd-screen snapshot: layout once, state per frame (§6.2, §18.7 M29)."""

    t: float
    seed: int
    bank_id: str | None = None
    point_count: int
    positions_xy: list[float]
    state: list[int]
    balance_norm: list[float]
    uninsured_mask: list[bool]
    panels: CrowdPanels


class DepthLevel(BaseModel):
    price: float
    size: float


class ChartMarker(BaseModel):
    t: float
    kind: Literal["halt", "statement"]
    label: str


class CrisisStatementRow(BaseModel):
    stmt_id: str
    channel: str
    verdict: Literal["accepted", "rejected", "pending"]
    reason: str = ""


class FutureFanBar(BaseModel):
    future_id: str
    failure_probability: float
    price_low: float | None = None
    price_high: float | None = None


class MarketScreenFrame(BaseModel):
    """§6.2 WebSocket payload for M30 (msgpack or JSON)."""

    module: Literal["M30"] = "M30"
    t: float
    bank_id: str | None = None
    prices: list[tuple[float, float]] = Field(default_factory=list)
    volumes: list[float] = Field(default_factory=list)
    depth: list[DepthLevel] = Field(default_factory=list)
    halt: dict[str, float | str] | None = None
    markers: list[ChartMarker] = Field(default_factory=list)
    statements: list[CrisisStatementRow] = Field(default_factory=list)
    futures: list[FutureFanBar] = Field(default_factory=list)
