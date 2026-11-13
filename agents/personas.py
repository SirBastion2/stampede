"""M25 persona panel — batched depositor reactions to bank statements (§5, §19)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

from pydantic import BaseModel, Field, ValidationError

from agents.llm_cache import (
    LLMCache,
    RemoteJsonBridge,
    cached_json_completion,
    effective_cache_model,
    live_json_completion_enabled,
    record_local_validation_failure,
)
from contracts.events import StatementIssued
from contracts.schemas import Scenario
from params.loader import RegistryError, get_parameter

_PARAM_ARCHETYPE_COUNT = "Persona panel archetypes per bank type"
_PARAM_ARCHETYPE_VERSION = "M25 archetype catalog version"
_PARAM_MODEL = "M25 persona panel Grok model"
_PARAM_GENERIC_WORRY = "M25 rule generic reassurance worry delta"
_PARAM_PERSONAL_WORRY = "M25 rule personalized reassurance worry delta"
_PARAM_THRESHOLD_SCALE = "M25 rule threshold delta scale"
_PARAM_ACT_NOW_BASE = "M25 rule act-now share baseline"

_PERSONALIZED_MARKERS = re.compile(
    r"\b(your|you\b|personally|by name|direct(ly)?|treasurer|founder|startup|vc\b)",
    re.IGNORECASE,
)


class ArchetypeScore(BaseModel):
    worry_delta: float = Field(ge=-1.0, le=1.0)
    threshold_delta: float = Field(ge=-1.0, le=1.0)
    act_now_share: float = Field(ge=0.0, le=1.0)


class PersonaPanelBatch(BaseModel):
    archetype_version: str
    scores: dict[str, ArchetypeScore]


@dataclass(frozen=True, slots=True)
class ArchetypeCard:
    archetype_id: str
    segment: str
    label: str


def _parse_about_count(value: str) -> int:
    match = re.search(r"\d+", value.strip())
    if match is None:
        raise RegistryError(f"cannot parse archetype count from {value!r}")
    return int(match.group(0))


def archetype_catalog_version(path: Any = None) -> str:
    return get_parameter(_PARAM_ARCHETYPE_VERSION, path).value.strip()


def persona_model_name(path: Any = None) -> str:
    return get_parameter(_PARAM_MODEL, path).value.strip()


def _tech_archetypes(limit: int) -> tuple[ArchetypeCard, ...]:
    base: list[ArchetypeCard] = [
        ArchetypeCard("startup-ceo", "startup", "Startup CEO with payroll at risk"),
        ArchetypeCard("vc-partner", "vc", "VC partner managing fund liquidity"),
        ArchetypeCard("founder-single", "founder", "Solo founder with one operating account"),
        ArchetypeCard("corp-treasurer", "treasurer", "Corporate treasurer with wires due"),
        ArchetypeCard("retail-saver", "retail", "Retail saver under the insurance cap"),
        ArchetypeCard("municipal-finance", "municipal", "Municipal finance officer"),
        ArchetypeCard("crypto-treasury", "crypto", "Crypto treasury desk"),
        ArchetypeCard("payroll-manager", "startup", "Payroll manager with same-day ACH"),
        ArchetypeCard("angel-investor", "vc", "Angel with uninsured balance"),
        ArchetypeCard("pe-lp", "vc", "PE limited partner awaiting capital call"),
        ArchetypeCard("family-office", "treasurer", "Family office cash desk"),
        ArchetypeCard("nonprofit-cfo", "municipal", "Nonprofit CFO with grants in flight"),
        ArchetypeCard("saas-founder", "founder", "SaaS founder with burn runway"),
        ArchetypeCard("hardware-founder", "founder", "Hardware founder with inventory payables"),
        ArchetypeCard("international-subsidiary", "treasurer", "International subsidiary CFO"),
        ArchetypeCard(
            "insured-retail-jumbo",
            "retail",
            "Retail depositor just above insurance limit",
        ),
        ArchetypeCard("uninsured-whale", "startup", "Top-10 uninsured depositor"),
        ArchetypeCard("secondary-bank-user", "retail", "Depositor with a backup bank"),
        ArchetypeCard("vc-fund-admin", "vc", "Fund administrator tracking wires"),
        ArchetypeCard("community-org", "municipal", "Community organization bookkeeper"),
    ]
    if limit <= len(base):
        return tuple(base[:limit])
    extras = [
        ArchetypeCard(f"extra-{index}", "retail", f"Additional retail archetype {index}")
        for index in range(len(base), limit)
    ]
    return tuple(base + extras)


def _community_archetypes(limit: int) -> tuple[ArchetypeCard, ...]:
    cards = [
        ArchetypeCard("local-retail", "retail", "Local retail depositor"),
        ArchetypeCard("small-business", "founder", "Main-street small business owner"),
        ArchetypeCard("municipal-clerk", "municipal", "Town municipal clerk"),
        ArchetypeCard("nonprofit-board", "municipal", "Nonprofit board treasurer"),
        ArchetypeCard("insured-household", "retail", "Fully insured household"),
        ArchetypeCard("church-bookkeeper", "municipal", "Church bookkeeper"),
        ArchetypeCard("farm-operator", "founder", "Farm operator with seasonal deposits"),
        ArchetypeCard("school-district", "municipal", "School district finance lead"),
    ]
    pool = list(cards)
    index = 0
    while len(pool) < limit:
        pool.append(ArchetypeCard(f"community-{index}", "retail", f"Community depositor {index}"))
        index += 1
    return tuple(pool[:limit])


def _crypto_archetypes(limit: int) -> tuple[ArchetypeCard, ...]:
    cards = list(_tech_archetypes(limit))
    for card in cards:
        if card.archetype_id == "crypto-treasury":
            break
    else:
        cards.insert(0, ArchetypeCard("crypto-exchange", "crypto", "Exchange treasury team"))
    return tuple(cards[:limit])


def archetypes_for_preset(preset: str, path: Any = None) -> tuple[ArchetypeCard, ...]:
    limit = _parse_about_count(get_parameter(_PARAM_ARCHETYPE_COUNT, path).value)
    lowered = preset.lower()
    if "community" in lowered:
        return _community_archetypes(limit)
    if "crypto" in lowered:
        return _crypto_archetypes(limit)
    return _tech_archetypes(limit)


def statement_cache_key(
    stmt_id: str,
    channel: str,
    segments: list[str],
    text: str,
    archetype_version: str,
) -> str:
    payload = json.dumps(
        {
            "stmt_id": stmt_id,
            "channel": channel,
            "segments": segments,
            "text": text,
            "archetype_version": archetype_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _unit_interval(seed: int, salt: str) -> float:
    acc = 2166136261
    for ch in f"{seed}:{salt}":
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return (acc % 1_000_000) / 1_000_000.0


def _seed_scale(seed: int, salt: str) -> float:
    return 0.8 + 0.4 * _unit_interval(seed, salt)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def personalization_strength(text: str, channel: str, segments: list[str]) -> float:
    """Higher when the statement reads as direct or segment-specific (E11)."""
    strength = 0.0
    if _PERSONALIZED_MARKERS.search(text):
        strength += 0.55
    if channel in {"direct_call", "ceo_video", "text_top_clients"}:
        strength += 0.25
    if segments and any(segment.lower() in text.lower() for segment in segments):
        strength += 0.2
    return _clamp(strength, 0.0, 1.0)


def _rule_params(path: Any = None) -> tuple[float, float, float, float]:
    generic = float(get_parameter(_PARAM_GENERIC_WORRY, path).value)
    personal = float(get_parameter(_PARAM_PERSONAL_WORRY, path).value)
    threshold = float(get_parameter(_PARAM_THRESHOLD_SCALE, path).value)
    act_base = float(get_parameter(_PARAM_ACT_NOW_BASE, path).value)
    return generic, personal, threshold, act_base


class PersonaProvider(Protocol):
    def score(
        self,
        *,
        preset: str,
        seed: int,
        stmt_id: str,
        channel: str,
        segments: list[str],
        text: str,
        archetypes: tuple[ArchetypeCard, ...],
        archetype_version: str,
    ) -> PersonaPanelBatch: ...


class RuleBasedPersonaProvider:
    """Deterministic offline scorer; mirrors PersonaFake for listed segments."""

    def score(
        self,
        *,
        preset: str,
        seed: int,
        stmt_id: str,
        channel: str,
        segments: list[str],
        text: str,
        archetypes: tuple[ArchetypeCard, ...],
        archetype_version: str,
    ) -> PersonaPanelBatch:
        del preset
        generic_worry, personal_worry, threshold_scale, act_base = _rule_params()
        strength = personalization_strength(text, channel, segments)
        worry_base = generic_worry + (personal_worry - generic_worry) * strength
        scores: dict[str, ArchetypeScore] = {}
        segment_set = {segment.lower() for segment in segments}
        for index, card in enumerate(archetypes):
            tilt = _seed_scale(seed, f"{stmt_id}:{card.archetype_id}")
            targeted = card.segment.lower() in segment_set and strength > 0.2
            worry = worry_base * tilt * (1.15 if targeted else 1.0)
            threshold = threshold_scale * tilt * (1.1 if targeted else 1.0)
            act = _clamp(act_base / (index + 1), 0.0, 1.0)
            scores[card.archetype_id] = ArchetypeScore(
                worry_delta=worry,
                threshold_delta=threshold,
                act_now_share=act,
            )
        return PersonaPanelBatch(archetype_version=archetype_version, scores=scores)


class _GrokPersonaProvider:
    """Live Grok path; only constructed when offline mode is off and API key is set."""

    def __init__(self, model: str, cache: LLMCache | None) -> None:
        self.model = model
        self.cache = cache

    def score(
        self,
        *,
        preset: str,
        seed: int,
        stmt_id: str,
        channel: str,
        segments: list[str],
        text: str,
        archetypes: tuple[ArchetypeCard, ...],
        archetype_version: str,
    ) -> PersonaPanelBatch:
        del seed
        prompt = _build_prompt(
            preset=preset,
            stmt_id=stmt_id,
            channel=channel,
            segments=segments,
            text=text,
            archetypes=archetypes,
            archetype_version=archetype_version,
        )
        raw = cached_json_completion(
            prompt=prompt,
            model=effective_cache_model(self.model),
            provider=_GrokJsonBridge(prompt, self.model),
            cache=self.cache,
        )
        return validate_persona_batch(raw, archetype_version=archetype_version)


class _GrokJsonBridge(RemoteJsonBridge):
    pass


def _build_prompt(
    *,
    preset: str,
    stmt_id: str,
    channel: str,
    segments: list[str],
    text: str,
    archetypes: tuple[ArchetypeCard, ...],
    archetype_version: str,
) -> str:
    cards = [
        {"id": card.archetype_id, "segment": card.segment, "label": card.label}
        for card in archetypes
    ]
    return json.dumps(
        {
            "task": "persona_panel",
            "preset": preset,
            "stmt_id": stmt_id,
            "channel": channel,
            "segments": segments,
            "text": text,
            "archetype_version": archetype_version,
            "archetypes": cards,
            "output_schema": {
                "archetype_version": "string",
                "scores": {
                    "<archetype_id>": {
                        "worry_delta": "float -1..1",
                        "threshold_delta": "float -1..1",
                        "act_now_share": "float 0..1",
                    }
                },
            },
        },
        sort_keys=True,
    )


def validate_persona_batch(
    payload: dict[str, Any],
    *,
    archetype_version: str | None = None,
) -> PersonaPanelBatch:
    try:
        batch = PersonaPanelBatch.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid persona panel JSON: {exc}") from exc
    if archetype_version is not None and batch.archetype_version != archetype_version:
        raise ValueError("persona panel archetype_version mismatch")
    return batch


def default_provider(cache: LLMCache | None = None) -> PersonaProvider:
    rule = RuleBasedPersonaProvider()
    if not live_json_completion_enabled():
        return rule
    return _GrokPersonaProvider(persona_model_name(), cache)


def score_statement(
    *,
    preset: str,
    seed: int,
    stmt_id: str,
    channel: str,
    segments: list[str],
    text: str = "",
    provider: PersonaProvider | None = None,
    cache: LLMCache | None = None,
) -> PersonaPanelBatch:
    archetypes = archetypes_for_preset(preset)
    version = archetype_catalog_version()
    scorer = provider or default_provider(cache)
    kwargs = {
        "preset": preset,
        "seed": seed,
        "stmt_id": stmt_id,
        "channel": channel,
        "segments": segments,
        "text": text,
        "archetypes": archetypes,
        "archetype_version": version,
    }
    try:
        batch = scorer.score(**kwargs)
        validate_persona_batch(batch.model_dump(), archetype_version=version)
        return batch
    except (ValueError, RuntimeError, OSError, ValidationError):
        if provider is not None:
            raise
        record_local_validation_failure()
        return RuleBasedPersonaProvider().score(**kwargs)


def scores_for_statement_event(
    event: StatementIssued,
    preset: str,
    *,
    text: str = "",
    provider: PersonaProvider | None = None,
    cache: LLMCache | None = None,
) -> dict[str, dict[str, float]]:
    """Shape consumed by M06/M24; matches PersonaFake ``scores`` layout."""
    batch = score_statement(
        preset=preset,
        seed=event.seed,
        stmt_id=event.stmt_id,
        channel=event.channel,
        segments=list(event.segments),
        text=text,
        provider=provider,
        cache=cache,
    )
    segment_scores: dict[str, dict[str, float]] = {}
    archetypes = archetypes_for_preset(preset)
    by_segment: dict[str, list[ArchetypeScore]] = {}
    for card in archetypes:
        score = batch.scores[card.archetype_id]
        by_segment.setdefault(card.segment, []).append(score)
    for segment in event.segments:
        if segment in segment_scores:
            continue
        members = by_segment.get(segment)
        if not members:
            continue
        segment_scores[segment] = {
            "worry_delta": sum(item.worry_delta for item in members) / len(members),
            "threshold_delta": sum(item.threshold_delta for item in members) / len(members),
            "act_now_share": sum(item.act_now_share for item in members) / len(members),
        }
    if not segment_scores:
        for segment in event.segments:
            tilt = _seed_scale(event.seed, f"{event.stmt_id}:{segment}")
            segment_scores[segment] = {
                "worry_delta": -0.08 * tilt,
                "threshold_delta": 0.04 * tilt,
                "act_now_share": _clamp(0.15 / (event.segments.index(segment) + 1), 0.0, 1.0),
            }
    return segment_scores


class PersonaPanelModule:
    """M25 scheduler module: scores statements; publishes nothing (§19)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick", "StatementIssued")
    publishes: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        module_id: str = "M25",
        *,
        provider: PersonaProvider | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.module_id = module_id
        self._provider = provider
        self._cache = cache
        self.scenario: Scenario | None = None
        self.bus: Any = None
        self.scores: dict[str, dict[str, float]] = {}
        self._statement_text: dict[str, str] = {}
        self.inbox: list[Any] = []
        self._cursor = 0

    def attach(self, bus: Any, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[Any]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        return self.on_tick(t, new_events)

    def on_tick(self, t: float, new_events: list[Any]) -> list[Any]:
        del t
        if self.scenario is None:
            return []
        for event in new_events:
            if not isinstance(event, StatementIssued):
                continue
            text = self._statement_text.get(event.stmt_id, "")
            segment_scores = scores_for_statement_event(
                event,
                self.scenario.preset,
                text=text,
                provider=self._provider,
                cache=self._cache,
            )
            self.scores.update(segment_scores)
        return []
