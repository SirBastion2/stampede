"""M04 media and analyst agents — headlines and notes from public signals (§5, §19)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

from pydantic import BaseModel, Field, ValidationError

from agents.guardrails import statement_allowed
from agents.llm_cache import (
    LLMCache,
    RemoteJsonBridge,
    cached_json_completion,
    effective_cache_model,
    live_json_completion_enabled,
    offline_mode,
)
from contracts.events import (
    AnalystNote,
    AttentionTick,
    HeadlinePublished,
    LossRealized,
    PriceTick,
)
from contracts.schemas import Scenario
from engine.bank.ledger import bank_id_for
from params.loader import RegistryError, get_parameter

_PARAM_ATTENTION = "Media agent attention threshold"
_PARAM_ATTENTION_ASSUMED = "Media agent attention threshold (ASSUMED)"
_PARAM_PRICE = "Media agent price-move threshold"
_PARAM_PRICE_ASSUMED = "Media agent price-move threshold (ASSUMED)"
_PARAM_LOSS = "Media agent loss-surprise threshold"
_PARAM_LOSS_ASSUMED = "Media agent loss-surprise threshold (ASSUMED)"
_PARAM_MODEL = "M04 headline writer Grok model"
_PARAM_LATENCY = "Media headline latency max minutes (ASSUMED)"
_PARAM_REACH_BASE = "M04 headline reach base (ASSUMED)"

_TRIGGER_ATTENTION = "attention"
_TRIGGER_PRICE = "price"
_TRIGGER_LOSS = "loss"


def _unit_interval(seed: int, salt: str) -> float:
    acc = 2166136261
    for ch in f"{seed}:{salt}":
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return (acc % 1_000_000) / 1_000_000.0


def _seed_scale(seed: int, salt: str) -> float:
    return 0.8 + 0.4 * _unit_interval(seed, salt)


def _bank_name_for(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "Fictional Community Bank"
    if "crypto" in lowered:
        return "Fictional Harbor Bank"
    return "Fictional Valley Bank"


class HeadlinePayload(BaseModel):
    text: str = Field(min_length=1)
    cited_event_ids: list[str] = Field(min_length=1)


class AnalystPayload(BaseModel):
    text: str = Field(min_length=1)
    cited_event_ids: list[str] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class MediaThresholds:
    attention: float
    price_move_fraction: float
    loss_surprise_amount: float
    max_latency_minutes: float
    reach_base: int
    attention_assumed: bool
    price_assumed: bool
    loss_assumed: bool


@dataclass(frozen=True, slots=True)
class PendingCoverage:
    trigger: str
    queued_at: float
    bank_id: str
    cited_event_ids: tuple[str, ...]
    context: dict[str, Any]


def _parse_float_cell(value: str) -> float | None:
    text = value.strip()
    if not text or text.upper() == "TBD":
        return None
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", text.replace("%", "").replace(",", ""))
    if match is None:
        return None
    number = float(match.group(1))
    if "%" in text and number > 1.0:
        return number / 100.0
    return number


def _resolve_threshold(primary: str, assumed_name: str, path: Path | None) -> tuple[float, bool]:
    primary_row = get_parameter(primary, path)
    parsed = _parse_float_cell(primary_row.value)
    if parsed is not None:
        return parsed, False
    assumed_row = get_parameter(assumed_name, path)
    assumed = _parse_float_cell(assumed_row.value)
    if assumed is None:
        raise RegistryError(f"cannot parse assumed threshold {assumed_name!r}")
    return assumed, True


def load_media_thresholds(path: Path | None = None) -> MediaThresholds:
    """Load M04 thresholds; TBD registry cells use ASSUMED rows (§19)."""
    attention, att_assumed = _resolve_threshold(
        _PARAM_ATTENTION, _PARAM_ATTENTION_ASSUMED, path
    )
    price_frac, price_assumed = _resolve_threshold(_PARAM_PRICE, _PARAM_PRICE_ASSUMED, path)
    loss_amount, loss_assumed = _resolve_threshold(_PARAM_LOSS, _PARAM_LOSS_ASSUMED, path)
    latency = float(get_parameter(_PARAM_LATENCY, path).value)
    reach_base = int(float(get_parameter(_PARAM_REACH_BASE, path).value))
    return MediaThresholds(
        attention=attention,
        price_move_fraction=price_frac,
        loss_surprise_amount=loss_amount,
        max_latency_minutes=max(0.0, latency),
        reach_base=max(1, reach_base),
        attention_assumed=att_assumed,
        price_assumed=price_assumed,
        loss_assumed=loss_assumed,
    )


def media_model_name(path: Path | None = None) -> str:
    return get_parameter(_PARAM_MODEL, path).value.strip()


def attention_crossed(previous: float, level: float, threshold: float) -> bool:
    return previous < threshold <= level


def price_move_crossed(open_price: float, price: float, threshold_fraction: float) -> bool:
    if open_price <= 0.0:
        return False
    drop = (open_price - price) / open_price
    return drop >= threshold_fraction


def loss_surprises(amount: float, threshold_amount: float) -> bool:
    return amount >= threshold_amount


def headline_reach(seed: int, level: float, *, base: int, salt: str) -> int:
    return max(1, int(base * level * _seed_scale(seed, salt)))


class HeadlineProvider(Protocol):
    def write_headline(
        self,
        *,
        preset: str,
        seed: int,
        trigger: str,
        cited_event_ids: list[str],
        context: dict[str, Any],
    ) -> HeadlinePayload: ...


class AnalystProvider(Protocol):
    def write_note(
        self,
        *,
        preset: str,
        seed: int,
        cited_event_ids: list[str],
        context: dict[str, Any],
    ) -> AnalystPayload: ...


class RuleBasedHeadlineProvider:
    """Deterministic offline headline writer; mirrors MediaFake templates."""

    def write_headline(
        self,
        *,
        preset: str,
        seed: int,
        trigger: str,
        cited_event_ids: list[str],
        context: dict[str, Any],
    ) -> HeadlinePayload:
        del seed
        name = _bank_name_for(preset)
        if trigger == _TRIGGER_ATTENTION:
            text = f"Depositors focus on {name}"
        elif trigger == _TRIGGER_PRICE:
            text = f"{name} shares slide amid deposit concerns"
        else:
            text = f"{name} books a loss on securities it sold"
        return HeadlinePayload(text=text, cited_event_ids=list(cited_event_ids))


class RuleBasedAnalystProvider:
    def write_note(
        self,
        *,
        preset: str,
        seed: int,
        cited_event_ids: list[str],
        context: dict[str, Any],
    ) -> AnalystPayload:
        del preset, seed, context
        text = "Securities sale realized a loss. Template note, not a model call."
        return AnalystPayload(text=text, cited_event_ids=list(cited_event_ids))


class _GrokHeadlineProvider:
    def __init__(self, model: str, cache: LLMCache | None) -> None:
        self.model = model
        self.cache = cache

    def write_headline(
        self,
        *,
        preset: str,
        seed: int,
        trigger: str,
        cited_event_ids: list[str],
        context: dict[str, Any],
    ) -> HeadlinePayload:
        prompt = json.dumps(
            {
                "task": "headline_writer",
                "preset": preset,
                "seed": seed,
                "trigger": trigger,
                "cited_event_ids": cited_event_ids,
                "context": context,
                "output_schema": {"text": "string", "cited_event_ids": ["string"]},
            },
            sort_keys=True,
        )
        raw = cached_json_completion(
            prompt=prompt,
            model=effective_cache_model(self.model),
            provider=_GrokJsonBridge(prompt, self.model),
            cache=self.cache,
        )
        return validate_headline_payload(raw, cited_event_ids=cited_event_ids)


class _GrokAnalystProvider:
    def __init__(self, model: str, cache: LLMCache | None) -> None:
        self.model = model
        self.cache = cache

    def write_note(
        self,
        *,
        preset: str,
        seed: int,
        cited_event_ids: list[str],
        context: dict[str, Any],
    ) -> AnalystPayload:
        prompt = json.dumps(
            {
                "task": "analyst_note",
                "preset": preset,
                "seed": seed,
                "cited_event_ids": cited_event_ids,
                "context": context,
                "output_schema": {"text": "string", "cited_event_ids": ["string"]},
            },
            sort_keys=True,
        )
        raw = cached_json_completion(
            prompt=prompt,
            model=effective_cache_model(self.model),
            provider=_GrokJsonBridge(prompt, self.model),
            cache=self.cache,
        )
        return validate_analyst_payload(raw, cited_event_ids=cited_event_ids)


class _GrokJsonBridge(RemoteJsonBridge):
    pass


def validate_headline_payload(
    payload: dict[str, Any],
    *,
    cited_event_ids: list[str],
) -> HeadlinePayload:
    try:
        parsed = HeadlinePayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid headline JSON: {exc}") from exc
    if not set(cited_event_ids).issubset(set(parsed.cited_event_ids)):
        raise ValueError("headline must cite triggering event ids")
    if not statement_allowed(parsed.text):
        raise ValueError("headline failed guardrails")
    return parsed


def validate_analyst_payload(
    payload: dict[str, Any],
    *,
    cited_event_ids: list[str],
) -> AnalystPayload:
    try:
        parsed = AnalystPayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid analyst note JSON: {exc}") from exc
    if not set(cited_event_ids).issubset(set(parsed.cited_event_ids)):
        raise ValueError("analyst note must cite triggering event ids")
    if not statement_allowed(parsed.text):
        raise ValueError("analyst note failed guardrails")
    return parsed


def default_headline_provider(cache: LLMCache | None = None) -> HeadlineProvider:
    rule = RuleBasedHeadlineProvider()
    if not live_json_completion_enabled():
        return rule
    return _GrokHeadlineProvider(media_model_name(), cache)


def default_analyst_provider(cache: LLMCache | None = None) -> AnalystProvider:
    rule = RuleBasedAnalystProvider()
    if not live_json_completion_enabled():
        return rule
    return _GrokAnalystProvider(media_model_name(), cache)


class MediaModule:
    """M04 scheduler module: threshold crossings → HeadlinePublished / AnalystNote (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("HeadlinePublished", "AnalystNote")
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "AttentionTick",
        "PriceTick",
        "LossRealized",
    )

    def __init__(
        self,
        module_id: str = "M04",
        *,
        thresholds: MediaThresholds | None = None,
        headline_provider: HeadlineProvider | None = None,
        analyst_provider: AnalystProvider | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.module_id = module_id
        self.thresholds = thresholds
        self._headline_provider = headline_provider
        self._analyst_provider = analyst_provider
        self._cache = cache
        self.scenario: Scenario | None = None
        self.bus: Any | None = None
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self._cursor = 0
        self._seq = 0
        self.attention = 0.0
        self.open_price: float | None = None
        self._attention_headline_sent = False
        self._price_headline_sent = False
        self._loss_sent = False
        self._pending: list[PendingCoverage] = []

    def attach(self, bus: Any, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        if self.thresholds is None:
            try:
                self.thresholds = load_media_thresholds()
            except RegistryError:
                self.thresholds = MediaThresholds(
                    attention=0.30,
                    price_move_fraction=0.05,
                    loss_surprise_amount=0.0,
                    max_latency_minutes=2.0,
                    reach_base=8000,
                    attention_assumed=True,
                    price_assumed=True,
                    loss_assumed=True,
                )
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None or self.thresholds is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        out = self._on_tick(t, new_events)
        self.published.extend(out)
        return out

    def _on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        thresholds = self.thresholds
        assert thresholds is not None
        scenario = self.scenario
        assert scenario is not None

        for event in new_events:
            if isinstance(event, PriceTick) and self.open_price is None:
                self.open_price = event.price
            if isinstance(event, AttentionTick):
                bank_id = event.bank_id or self.bank_id
                cite = event.cause_id or self._cause("attention-cite")
                if (
                    attention_crossed(self.attention, event.level, thresholds.attention)
                    and not self._attention_headline_sent
                ):
                    self._attention_headline_sent = True
                    self._enqueue(
                        t,
                        trigger=_TRIGGER_ATTENTION,
                        bank_id=bank_id,
                        cited_event_ids=(cite,),
                        context={"level": event.level, "delta_4h": event.delta_4h},
                    )
                self.attention = event.level
            elif isinstance(event, PriceTick):
                bank_id = event.bank_id or self.bank_id
                open_px = self.open_price if self.open_price is not None else event.price
                cite = event.cause_id or self._cause("price-cite")
                if (
                    not self._price_headline_sent
                    and price_move_crossed(open_px, event.price, thresholds.price_move_fraction)
                ):
                    self._price_headline_sent = True
                    self._enqueue(
                        t,
                        trigger=_TRIGGER_PRICE,
                        bank_id=bank_id,
                        cited_event_ids=(cite,),
                        context={"price": event.price, "open": open_px},
                    )
            elif isinstance(event, LossRealized) and not self._loss_sent:
                bank_id = event.bank_id or self.bank_id
                cite = event.cause_id or self._cause("loss-cite")
                if loss_surprises(event.amount, thresholds.loss_surprise_amount):
                    self._loss_sent = True
                    self._enqueue(
                        t,
                        trigger=_TRIGGER_LOSS,
                        bank_id=bank_id,
                        cited_event_ids=(cite,),
                        context={"amount": event.amount, "kind": event.kind},
                    )

        return self._flush_ready(t, thresholds)

    def _enqueue(
        self,
        t: float,
        *,
        trigger: str,
        bank_id: str,
        cited_event_ids: tuple[str, ...],
        context: dict[str, Any],
    ) -> None:
        self._pending.append(
            PendingCoverage(
                trigger=trigger,
                queued_at=t,
                bank_id=bank_id,
                cited_event_ids=cited_event_ids,
                context=context,
            )
        )

    def _flush_ready(self, t: float, thresholds: MediaThresholds) -> list[BaseModel]:
        scenario = self.scenario
        assert scenario is not None
        headline_provider = self._headline_provider or default_headline_provider(self._cache)
        analyst_provider = self._analyst_provider or default_analyst_provider(self._cache)
        out: list[BaseModel] = []
        still_pending: list[PendingCoverage] = []
        for job in self._pending:
            elapsed = t - job.queued_at
            if elapsed < 0:
                still_pending.append(job)
                continue
            cited = list(job.cited_event_ids)
            headline = headline_provider.write_headline(
                preset=scenario.preset,
                seed=scenario.seed,
                trigger=job.trigger,
                cited_event_ids=cited,
                context=job.context,
            )
            if job.trigger == _TRIGGER_ATTENTION:
                level = float(job.context.get("level", 1.0))
                reach = headline_reach(
                    scenario.seed, level, base=thresholds.reach_base, salt="reach"
                )
            elif job.trigger == _TRIGGER_LOSS:
                reach = max(
                    1,
                    int((thresholds.reach_base + 4000) * _seed_scale(scenario.seed, "loss-reach")),
                )
            else:
                reach = headline_reach(
                    scenario.seed,
                    1.0,
                    base=thresholds.reach_base,
                    salt="price-reach",
                )
            out.append(
                HeadlinePublished(
                    t=t,
                    seed=scenario.seed,
                    bank_id=job.bank_id,
                    cause_id=self._cause(f"headline-{job.trigger}"),
                    text=headline.text,
                    reach=reach,
                )
            )
            if job.trigger == _TRIGGER_LOSS:
                note = analyst_provider.write_note(
                    preset=scenario.preset,
                    seed=scenario.seed,
                    cited_event_ids=cited,
                    context=job.context,
                )
                out.append(
                    AnalystNote(
                        t=t,
                        seed=scenario.seed,
                        bank_id=job.bank_id,
                        cause_id=self._cause("analyst-note"),
                        text=note.text,
                        cited_event_ids=list(note.cited_event_ids),
                    )
                )
        self._pending = still_pending
        return out

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_id_for(self.scenario.preset)

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"
