"""M44 short-seller and activist agents (§5, §18.4, §19)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from agents.guardrails import statement_allowed
from agents.llm_cache import LLMCache, cached_text, effective_cache_model
from contracts.events import AttentionTick, CapitalChanged, PriceTick, ThesisPublished
from contracts.schemas import Scenario
from params.loader import RegistryError, get_parameter

_PARAM_THRESHOLD = "M44 assumed vulnerability publish threshold"
_PARAM_CAPITAL_BUMP = "M44 assumed capital-stress bump"
_PARAM_REFERENCE_CET1 = "M44 assumed reference CET1"
_PARAM_COMMUNITY_UNINSURED = "M44 assumed community uninsured share"
_PARAM_SVB_UNINSURED = "SVB-like: uninsured share"
_PARAM_REACH_BASE = "M44 assumed thesis reach base"
_PARAM_REACH_FLOOR = "M44 assumed thesis reach floor"
_PARAM_REACH_LOW = "M44 assumed thesis reach scale low"
_PARAM_REACH_HIGH = "M44 assumed thesis reach scale high"
_PARAM_SHORT_QTY = "M44 assumed short order quantity"
_PARAM_MODEL = "M44 assumed thesis model"

AUTHOR_ID = "short-1"


@dataclass(frozen=True, slots=True)
class ShortParams:
    """M44 numbers loaded from the parameter registry. Nothing here is inlined."""

    threshold: float
    capital_stress_bump: float
    reference_cet1: float
    community_uninsured_share: float
    svb_uninsured_share: float
    reach_base: float
    reach_floor: float
    reach_scale_low: float
    reach_scale_high: float
    short_quantity: float
    model_name: str


@dataclass(frozen=True, slots=True)
class PublicMetrics:
    """Public figures a short seller can read. The score uses the first three."""

    attention: float
    uninsured_share: float
    cet1_ratio: float
    price: float | None
    opening_price: float | None


@dataclass(frozen=True, slots=True)
class ShortOrder:
    """Order recorded for M14. M44 does not write the L3 book."""

    author_id: str
    bank_id: str
    quantity: float
    limit_price: float | None


@dataclass(frozen=True, slots=True)
class ThesisRecord:
    """Grok thesis text plus the simulation numbers published on the bus."""

    text: str
    score: float
    author_id: str
    bank_id: str
    reach: int


def _parse_float(name: str, value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if match is None:
        raise RegistryError(f"cannot parse registry value for {name}: {value!r}")
    number = float(match.group(0))
    lowered = value.lower()
    if "%" in value or "percent" in lowered:
        return number / 100.0
    return number


def _require_float(name: str, path: Path | None) -> float:
    row = get_parameter(name, path)
    return _parse_float(name, row.value)


def load_short_params(path: Path | None = None) -> ShortParams:
    """Load M44 parameters. The threshold row is ASSUMED, not the TBD/UNSET row."""
    return ShortParams(
        threshold=_require_float(_PARAM_THRESHOLD, path),
        capital_stress_bump=_require_float(_PARAM_CAPITAL_BUMP, path),
        reference_cet1=_require_float(_PARAM_REFERENCE_CET1, path),
        community_uninsured_share=_require_float(_PARAM_COMMUNITY_UNINSURED, path),
        svb_uninsured_share=_require_float(_PARAM_SVB_UNINSURED, path),
        reach_base=_require_float(_PARAM_REACH_BASE, path),
        reach_floor=_require_float(_PARAM_REACH_FLOOR, path),
        reach_scale_low=_require_float(_PARAM_REACH_LOW, path),
        reach_scale_high=_require_float(_PARAM_REACH_HIGH, path),
        short_quantity=_require_float(_PARAM_SHORT_QTY, path),
        model_name=get_parameter(_PARAM_MODEL, path).value.strip(),
    )


def uninsured_share_for(preset: str, params: ShortParams | None = None) -> float:
    """Community presets use the ASSUMED share. Every other preset uses the SVB-like row."""
    cfg = params if params is not None else load_short_params()
    if "community" in preset.lower():
        return cfg.community_uninsured_share
    return cfg.svb_uninsured_share


def bank_id_for(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "bank-community"
    if "crypto" in lowered:
        return "bank-crypto-fictional"
    return "bank-fvb"


def bank_name_for(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "Fictional Community Bank"
    if "crypto" in lowered:
        return "Fictional Harbor Bank"
    return "Fictional Valley Bank"


def vulnerability_score(
    metrics: PublicMetrics,
    params: ShortParams | None = None,
) -> float:
    """Attention times the uninsured share, plus a bump when CET1 is below the reference.

    The product is the public-metric score from §18.4. The bump and the reference
    CET1 are ASSUMED registry rows. Grok does not choose this number.
    """
    cfg = params if params is not None else load_short_params()
    score = metrics.attention * metrics.uninsured_share
    if metrics.cet1_ratio < cfg.reference_cet1:
        score += cfg.capital_stress_bump
    return score


def metrics_from_events(
    events: list[BaseModel] | tuple[BaseModel, ...],
    preset: str,
    params: ShortParams | None = None,
) -> PublicMetrics:
    """Latest public prints already on the bus. Missing capital stays at the reference."""
    cfg = params if params is not None else load_short_params()
    attention = 0.0
    cet1 = cfg.reference_cet1
    price: float | None = None
    opening: float | None = None
    for event in events:
        if isinstance(event, AttentionTick):
            attention = event.level
        elif isinstance(event, CapitalChanged):
            cet1 = event.cet1_ratio
        elif isinstance(event, PriceTick):
            if opening is None:
                opening = event.price
            price = event.price
    return PublicMetrics(
        attention=attention,
        uninsured_share=uninsured_share_for(preset, cfg),
        cet1_ratio=cet1,
        price=price,
        opening_price=opening,
    )


def _unit_interval(seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def thesis_reach(seed: int, params: ShortParams | None = None) -> int:
    """Simulation-owned reach. The model does not set it (§7)."""
    cfg = params if params is not None else load_short_params()
    span = cfg.reach_scale_high - cfg.reach_scale_low
    scale = cfg.reach_scale_low + span * _unit_interval(seed, "thesis-reach")
    return max(int(cfg.reach_floor), int(cfg.reach_base * scale))


def _offline_thesis(metrics: PublicMetrics, bank_name: str, score: float) -> str:
    """Deterministic rule used when the cache misses and live Grok is off."""
    return (
        f"{bank_name}: public vulnerability score {score:.4f}. "
        f"Attention {metrics.attention:.4f}, "
        f"uninsured share {metrics.uninsured_share:.4f}, "
        f"CET1 {metrics.cet1_ratio:.4f}. "
        "Offline template thesis."
    )


def _validate_thesis_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError("empty thesis")
    if not statement_allowed(cleaned):
        raise ValueError("thesis failed guardrails")
    return cleaned


def _prompt(metrics: PublicMetrics, bank_name: str, score: float, seed: int) -> str:
    return (
        "Write one short-seller thesis about a fictional bank. "
        f"Bank: {bank_name}. Vulnerability score: {score:.4f}. "
        f"Attention: {metrics.attention:.4f}. "
        f"Uninsured share: {metrics.uninsured_share:.4f}. "
        f"CET1: {metrics.cet1_ratio:.4f}. "
        f"Scenario seed: {seed}. "
        "Use only that bank name. Do not name a real institution."
    )


def thesis_text(
    metrics: PublicMetrics,
    *,
    seed: int,
    bank_name: str,
    score: float,
    params: ShortParams | None = None,
    cache: LLMCache | None = None,
) -> str:
    """Grok thesis text through the cache. Offline mode uses the rule template.

    ``seed`` is in the prompt so two seeds do not share a cache row. The offline
    template states only public metrics. The returned text is validated.
    """
    cfg = params if params is not None else load_short_params()
    store = cache if cache is not None else LLMCache()
    prompt = _prompt(metrics, bank_name, score, seed)

    def fallback() -> str:
        return _offline_thesis(metrics, bank_name, score)

    return cached_text(
        model=effective_cache_model(cfg.model_name),
        prompt=prompt,
        fallback=fallback,
        validate=_validate_thesis_text,
        cache=store,
    )


def short_order_for(
    metrics: PublicMetrics,
    *,
    author_id: str,
    bank_id: str,
    params: ShortParams | None = None,
) -> ShortOrder:
    """Build the short order M14 will apply. Quantity comes from the registry."""
    cfg = params if params is not None else load_short_params()
    return ShortOrder(
        author_id=author_id,
        bank_id=bank_id,
        quantity=cfg.short_quantity,
        limit_price=metrics.price,
    )


class ShortSellerModule:
    """Publishes ThesisPublished once when the vulnerability score clears the threshold.

    Consumes Tick (§19). Public metrics are read from events already on the bus
    (AttentionTick, CapitalChanged, PriceTick) because those prints are public
    and the same-tick publishers run later in the phase order.
    """

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ("ThesisPublished",)

    def __init__(
        self,
        module_id: str = "M44",
        *,
        params: ShortParams | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.module_id = module_id
        self.params = params if params is not None else load_short_params()
        self.cache = cache if cache is not None else LLMCache()
        self.scenario: Scenario | None = None
        self.bus: object | None = None
        self.inbox: list[BaseModel] = []
        self._cursor = 0
        self._seq = 0
        self._published = False
        self.score = 0.0
        self.metrics: PublicMetrics | None = None
        self.theses: list[ThesisRecord] = []
        self.short_orders: list[ShortOrder] = []

    def attach(self, bus: object, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        subscribe = getattr(bus, "subscribe", None)
        if callable(subscribe):
            for event_type in self.consumes:
                subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        return self.on_tick(t, new_events)

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self._published or self.scenario is None or self.bus is None:
            return []
        log = getattr(self.bus, "log", None)
        events: list[BaseModel] = list(log) if isinstance(log, list) else []
        metrics = metrics_from_events(events, self.scenario.preset, self.params)
        score = vulnerability_score(metrics, self.params)
        self.metrics = metrics
        self.score = score
        if score < self.params.threshold:
            return []
        name = bank_name_for(self.scenario.preset)
        text = thesis_text(
            metrics,
            seed=self.scenario.seed,
            bank_name=name,
            score=score,
            params=self.params,
            cache=self.cache,
        )
        bank_id = bank_id_for(self.scenario.preset)
        order = short_order_for(
            metrics,
            author_id=AUTHOR_ID,
            bank_id=bank_id,
            params=self.params,
        )
        reach = thesis_reach(self.scenario.seed, self.params)
        self._seq += 1
        self._published = True
        self.short_orders.append(order)
        self.theses.append(
            ThesisRecord(
                text=text,
                score=score,
                author_id=AUTHOR_ID,
                bank_id=bank_id,
                reach=reach,
            )
        )
        return [
            ThesisPublished(
                t=t,
                seed=self.scenario.seed,
                bank_id=bank_id,
                cause_id=f"{self.module_id}-thesis-{self._seq}",
                author_id=AUTHOR_ID,
                reach=reach,
            )
        ]
