"""M41 named VIP depositor agents — batched Grok decisions reduced for futures (§5, §19)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, cast

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
from contracts.events import AttentionTick, DirectiveIssued, VIPDecision
from contracts.schemas import Scenario
from params.loader import RegistryError, get_parameter

_PARAM_COUNT = "VIP depositor Grok agents (M41)"
_PARAM_BATCH = "VIP agent batch size"
_PARAM_MEMORY = "VIP agent memory depth"
_PARAM_SEMAPHORE = "Grok API concurrency semaphore (initial)"
_PARAM_TOP10 = "SVB-like: top-10 depositor balance"
_PARAM_INSURANCE = "Insurance limit"
_PARAM_INTERVAL = "M41 decision interval minutes"
_PARAM_ATTENTION = "M41 attention trigger level"
_PARAM_WITHDRAW_FRACTION = "M41 withdraw fraction of balance"
_PARAM_EXAMPLE_AMOUNT = "M41 acceptance example move amount"
_PARAM_MODEL = "M41 Grok model"
_PARAM_SENSITIVITIES = "M41 rule sensitivities (ASSUMED)"
_PARAM_MARGIN = "M41 reduced-rule margin"
_PARAM_TAIL = "M41 tail rank balance fraction of tenth"
_PARAM_BELIEF = "M41 baseline belief"

KESTREL_DISPLAY = "Founder of Kestrel Labs"
_SAFE_RATIONALE = "Fictional depositor reviewed uninsured cash."
_FEATURE_KEYS = ("attention", "uninsured_gap", "belief", "stock_return")

_ACTION_TOOL = {
    "withdraw": "move_funds",
    "stay": "check_exposure",
    "post": "post",
    "message_group": "message_group",
}

_BASE_PERSONAS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (KESTREL_DISPLAY, "founder", "Kestrel Labs", ()),
    ("CFO of Northwind Robotics", "treasurer", "Northwind Robotics", ()),
    (
        "Partner at Lumen Ridge Capital",
        "vc",
        "Lumen Ridge Capital",
        ("port-lumen-a", "port-lumen-b"),
    ),
    ("Treasurer of Harborline Systems", "treasurer", "Harborline Systems", ()),
    ("CEO of Pinion Health", "founder", "Pinion Health", ()),
    ("GP at Bramble Ventures", "vc", "Bramble Ventures", ("port-bramble-a", "port-bramble-b")),
    ("Founder of Quill Circuit", "founder", "Quill Circuit", ()),
    ("CFO of Marble Pine", "treasurer", "Marble Pine", ()),
    ("Treasurer of Sable Freight", "treasurer", "Sable Freight", ()),
    ("Partner at Otterglass Capital", "vc", "Otterglass Capital", ("port-otter-a",)),
    ("Founder of Redcedar Analytics", "founder", "Redcedar Analytics", ()),
    ("CFO of Kindling Foods", "treasurer", "Kindling Foods", ()),
    ("Treasurer of Vellum Payments", "treasurer", "Vellum Payments", ()),
    ("GP at Moss Meridian", "vc", "Moss Meridian", ("port-moss-a", "port-moss-b")),
    ("Founder of Lantern Grid", "founder", "Lantern Grid", ()),
)


class VipRateLimited(RuntimeError):
    """Raised when the live Grok call reports HTTP 429 (§7.1)."""


class VipAgentDecision(BaseModel):
    agent_id: str
    action: Literal["withdraw", "stay", "post", "message_group"]
    amount: float = Field(ge=0)
    rationale: str = Field(min_length=1)
    tool: Literal["check_exposure", "move_funds", "post", "message_group"]


class VipBatchReply(BaseModel):
    decisions: list[VipAgentDecision]


class ReducedRule(BaseModel):
    """Threshold and sensitivities M26 replays without a new Grok call (§7.1)."""

    agent_id: str
    action: str
    amount: float = Field(ge=0)
    threshold: float
    sensitivities: dict[str, float]
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class VipConfig:
    agent_count: int
    batch_size: int
    memory_depth: int
    semaphore_initial: int
    interval_minutes: int
    attention_trigger: float
    withdraw_fraction: float
    example_amount: float
    model: str
    sensitivities: dict[str, float]
    margin: float
    tail_fraction: float
    baseline_belief: float
    top10_balance: float
    insurance_limit: float


@dataclass(frozen=True, slots=True)
class PersonaCard:
    agent_id: str
    display_name: str
    role: str
    company: str
    portfolio_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExposureContext:
    balance: float
    insured: float
    attention: float


@dataclass(frozen=True, slots=True)
class ExposureView:
    agent_id: str
    balance: float
    insured: float
    uninsured_gap: float
    attention: float


@dataclass(frozen=True, slots=True)
class MoveResult:
    agent_id: str
    amount: float
    destination: str


@dataclass(frozen=True, slots=True)
class PostResult:
    agent_id: str
    text: str


@dataclass(frozen=True, slots=True)
class GroupMessage:
    agent_id: str
    portfolio_ids: tuple[str, ...]
    text: str


class VipMemory:
    """Last N tool results for one VIP (§18.2)."""

    def __init__(self, depth: int) -> None:
        self.depth = max(1, depth)
        self.events: list[str] = []

    def add(self, item: str) -> None:
        self.events.append(item)
        if len(self.events) > self.depth:
            self.events = self.events[-self.depth :]

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self.events)


class AdaptiveSemaphore:
    """§7.1 concurrency gate: start at the registry limit, +1 on success, halve on 429."""

    def __init__(self, initial: int) -> None:
        self.limit = max(1, initial)

    def on_success(self) -> None:
        self.limit += 1

    def on_rate_limit(self) -> None:
        self.limit = max(1, self.limit // 2)


class _JsonProvider(Protocol):
    def complete_json(self, prompt: str) -> dict[str, Any]: ...


def _parse_number(value: str) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([KkMmBbTt])?", value.replace(",", ""))
    if match is None:
        raise RegistryError(f"cannot parse number from {value!r}")
    number = float(match.group(1))
    suffix = match.group(2)
    if suffix:
        number *= {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[suffix.lower()]
    return number


def _parse_sensitivities(value: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for part in value.split(","):
        piece = part.strip()
        if not piece:
            continue
        key, raw = piece.split(":", 1)
        parsed[key.strip()] = float(raw.strip())
    missing = [key for key in _FEATURE_KEYS if key not in parsed]
    if missing:
        raise RegistryError(f"M41 sensitivities missing {missing}")
    return parsed


def load_vip_config(path: Any = None) -> VipConfig:
    """Load M41 rows from the parameter registry (§8.1)."""
    sensitivities = _parse_sensitivities(get_parameter(_PARAM_SENSITIVITIES, path).value)
    return VipConfig(
        agent_count=int(_parse_number(get_parameter(_PARAM_COUNT, path).value)),
        batch_size=max(1, int(_parse_number(get_parameter(_PARAM_BATCH, path).value))),
        memory_depth=max(1, int(_parse_number(get_parameter(_PARAM_MEMORY, path).value))),
        semaphore_initial=max(1, int(_parse_number(get_parameter(_PARAM_SEMAPHORE, path).value))),
        interval_minutes=max(1, int(_parse_number(get_parameter(_PARAM_INTERVAL, path).value))),
        attention_trigger=float(get_parameter(_PARAM_ATTENTION, path).value),
        withdraw_fraction=float(get_parameter(_PARAM_WITHDRAW_FRACTION, path).value),
        example_amount=float(get_parameter(_PARAM_EXAMPLE_AMOUNT, path).value),
        model=get_parameter(_PARAM_MODEL, path).value.strip(),
        sensitivities=sensitivities,
        margin=float(get_parameter(_PARAM_MARGIN, path).value),
        tail_fraction=float(get_parameter(_PARAM_TAIL, path).value),
        baseline_belief=float(get_parameter(_PARAM_BELIEF, path).value),
        top10_balance=_parse_number(get_parameter(_PARAM_TOP10, path).value),
        insurance_limit=_parse_number(get_parameter(_PARAM_INSURANCE, path).value),
    )


def persona_cards(count: int | None = None) -> tuple[PersonaCard, ...]:
    """Named fictional VIPs. Count comes from the registry when omitted."""
    limit = load_vip_config().agent_count if count is None else count
    if limit < 1:
        raise RegistryError("VIP count must be positive")
    rows = list(_BASE_PERSONAS)
    extra = 0
    while len(rows) < limit:
        extra += 1
        rows.append((f"Founder of Spare Studio {extra}", "founder", f"Spare Studio {extra}", ()))
    cards: list[PersonaCard] = []
    for index, (display, role, company, portfolios) in enumerate(rows[:limit], start=1):
        cards.append(
            PersonaCard(
                agent_id=f"vip-{index}",
                display_name=display,
                role=role,
                company=company,
                portfolio_ids=portfolios,
            )
        )
    return tuple(cards)


def allocate_balances(config: VipConfig | None = None) -> tuple[float, ...]:
    """Split the top-10 total across the first ten VIPs; later ranks are a fraction of the tenth."""
    cfg = config or load_vip_config()
    head = min(10, cfg.agent_count)
    weights = [float(head - index) for index in range(head)]
    weight_sum = sum(weights) or 1.0
    balances = [cfg.top10_balance * weight / weight_sum for weight in weights]
    if head and cfg.agent_count > head:
        tenth = balances[-1]
        balances.extend([cfg.tail_fraction * tenth] * (cfg.agent_count - head))
    return tuple(balances)


def bank_id_for_preset(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "bank-community"
    if "crypto" in lowered:
        return "bank-crypto-fictional"
    return "bank-fvb"


def is_community_preset(preset: str) -> bool:
    return "community" in preset.lower()


def live_grok_enabled() -> bool:
    """Live JSON completions (Grok or local HTTP) when the backend is enabled."""
    return live_json_completion_enabled()


def uninsured_gap(balance: float, insured: float) -> float:
    if balance <= 0.0:
        return 0.0
    return max(0.0, balance - insured) / balance


def features_for(context: ExposureContext, baseline_belief: float) -> dict[str, float]:
    return {
        "attention": context.attention,
        "uninsured_gap": uninsured_gap(context.balance, context.insured),
        "belief": baseline_belief,
        "stock_return": 0.0,
    }


def check_exposure(card: PersonaCard, context: ExposureContext) -> ExposureView:
    """Tool: read this VIP's balance, insurance, and attention."""
    return ExposureView(
        agent_id=card.agent_id,
        balance=context.balance,
        insured=context.insured,
        uninsured_gap=uninsured_gap(context.balance, context.insured),
        attention=context.attention,
    )


def move_funds(card: PersonaCard, amount: float) -> MoveResult:
    """Tool: move cash off the fictional bank."""
    return MoveResult(agent_id=card.agent_id, amount=max(0.0, amount), destination="external")


def post(card: PersonaCard, text: str) -> PostResult:
    """Tool: public post. Text must stay fictional."""
    cleaned = _safe_text(text)
    return PostResult(agent_id=card.agent_id, text=cleaned)


def message_group(card: PersonaCard, text: str) -> GroupMessage:
    """Tool: message the VIP's portfolio group."""
    return GroupMessage(
        agent_id=card.agent_id,
        portfolio_ids=card.portfolio_ids,
        text=_safe_text(text),
    )


def _safe_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned or not statement_allowed(cleaned):
        return _SAFE_RATIONALE
    return cleaned


def format_compact_dollars(amount: float) -> str:
    if amount >= 1_000_000_000:
        return f"${_trim(amount / 1_000_000_000)}B"
    if amount >= 1_000_000:
        return f"${_trim(amount / 1_000_000)}M"
    return f"${amount:,.0f}"


def _trim(scaled: float) -> str:
    return f"{scaled:.2f}".rstrip("0").rstrip(".")


def format_feed_line(display_name: str, action: str, amount: float) -> str:
    """VIP feed panel copy. The acceptance line is the Kestrel Labs move."""
    if action == "withdraw" and amount > 0.0:
        return f"{display_name} moved {format_compact_dollars(amount)}"
    if action == "post":
        return f"{display_name} posted"
    if action == "message_group":
        return f"{display_name} messaged their group"
    return f"{display_name} stayed"


def feature_score(features: dict[str, float], sensitivities: dict[str, float]) -> float:
    return sum(sensitivities[key] * float(features.get(key, 0.0)) for key in sensitivities)


def reduce_decision(
    decision: VipAgentDecision,
    features: dict[str, float],
    *,
    sensitivities: dict[str, float],
    margin: float,
) -> ReducedRule:
    """Place the threshold on the deciding side of the score so replay matches."""
    score = feature_score(features, sensitivities)
    if decision.action == "stay":
        threshold = score + margin
    else:
        threshold = score - margin
    return ReducedRule(
        agent_id=decision.agent_id,
        action=decision.action,
        amount=decision.amount,
        threshold=threshold,
        sensitivities=dict(sensitivities),
        features=dict(features),
    )


def replay_reduced_rule(rule: ReducedRule, features: dict[str, float]) -> str:
    """Futures replay: the stored action when the score clears the threshold, otherwise stay."""
    score = feature_score(features, rule.sensitivities)
    if score >= rule.threshold:
        return rule.action
    return "stay"


def rules_for_futures(rules: list[ReducedRule]) -> list[dict[str, Any]]:
    """Plain records for M26. Replaying them does not call Grok."""
    return [rule.model_dump() for rule in rules]


def validate_vip_batch(
    payload: dict[str, Any],
    *,
    expected_ids: tuple[str, ...] | None = None,
) -> VipBatchReply:
    try:
        batch = VipBatchReply.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid VIP batch JSON: {exc}") from exc
    for item in batch.decisions:
        expected_tool = _ACTION_TOOL.get(item.action)
        if expected_tool is None or item.tool != expected_tool:
            raise ValueError(f"VIP tool does not match action for {item.agent_id}")
    if expected_ids is not None:
        got = tuple(item.agent_id for item in batch.decisions)
        if got != expected_ids:
            raise ValueError("VIP batch agent_id list does not match the batch")
    return batch


def batched_cards(cards: tuple[PersonaCard, ...], size: int) -> tuple[tuple[PersonaCard, ...], ...]:
    width = max(1, size)
    return tuple(cards[index : index + width] for index in range(0, len(cards), width))


def build_vip_prompt(
    *,
    preset: str,
    seed: int,
    community: bool,
    attention: float,
    agents: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "task": "vip_batch",
            "preset": preset,
            "seed": seed,
            "community": community,
            "attention": attention,
            "agents": agents,
            "tools": list(_ACTION_TOOL.values()),
            "output_schema": {
                "decisions": [
                    {
                        "agent_id": "string",
                        "action": "withdraw|stay|post|message_group",
                        "amount": "float >= 0",
                        "rationale": "string",
                        "tool": "move_funds|check_exposure|post|message_group",
                    }
                ]
            },
        },
        sort_keys=True,
    )


def _rationale_for(action: str) -> str:
    if action == "withdraw":
        return "Uninsured balance at a fictional bank is large enough to move."
    if action == "message_group":
        return "Portfolio companies should move uninsured cash."
    if action == "post":
        return "Posted a note about fictional cash moving."
    return "Insured balance stays."


class RuleBasedVipProvider:
    """Deterministic offline batch. Community presets stay; other presets move or message."""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        request = json.loads(prompt)
        config = load_vip_config()
        community = bool(request["community"])
        decisions: list[dict[str, Any]] = []
        for agent in request["agents"]:
            role = str(agent["role"])
            if community:
                action = "stay"
            elif role == "vc":
                action = "message_group"
            else:
                action = "withdraw"
            balance = float(agent["balance"])
            display = str(agent["display_name"])
            if action == "stay":
                amount = 0.0
            elif display == KESTREL_DISPLAY and action == "withdraw":
                amount = min(config.example_amount, balance)
            else:
                amount = max(0.0, balance * config.withdraw_fraction)
            tool = _ACTION_TOOL[action]
            decisions.append(
                {
                    "agent_id": agent["agent_id"],
                    "action": action,
                    "amount": amount,
                    "rationale": _rationale_for(action),
                    "tool": tool,
                }
            )
        return {"decisions": decisions}


class _GrokVipBridge(RemoteJsonBridge):
    """Live path. Constructed only when the remote JSON backend is enabled."""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        raw = super().complete_json(prompt)
        validate_vip_batch(raw)
        return raw


def complete_vip_json(
    *,
    prompt: str,
    model: str,
    provider: _JsonProvider,
    cache: LLMCache,
    expected_ids: tuple[str, ...] | None = None,
) -> VipBatchReply:
    """One cached completion. Callers validate again if they pass expected ids."""
    raw = cached_json_completion(prompt=prompt, model=model, provider=provider, cache=cache)
    return validate_vip_batch(raw, expected_ids=expected_ids)


def _agent_payload(
    card: PersonaCard,
    context: ExposureContext,
    features: dict[str, float],
    memory: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "agent_id": card.agent_id,
        "display_name": card.display_name,
        "role": card.role,
        "company": card.company,
        "portfolio_ids": list(card.portfolio_ids),
        "balance": context.balance,
        "insured": context.insured,
        "memory": list(memory),
        "features": features,
    }


class VIPModule:
    """M41 scheduler module. Publishes DirectiveIssued and VIPDecision (§19)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ("DirectiveIssued", "VIPDecision")

    def __init__(
        self,
        module_id: str = "M41",
        *,
        provider: _JsonProvider | None = None,
        cache: LLMCache | None = None,
        config: VipConfig | None = None,
    ) -> None:
        self.module_id = module_id
        self.config = config or load_vip_config()
        self._provider = provider
        self._rule = RuleBasedVipProvider()
        self._cache = cache
        self.gate = AdaptiveSemaphore(self.config.semaphore_initial)
        self.scenario: Scenario | None = None
        self.bus: Any = None
        self.inbox: list[BaseModel] = []
        self._cursor = 0
        self._seq = 0
        self._pending: list[str] = []
        self.attention = 0.0
        self._last_decision_t: float | None = None
        self.cards = persona_cards(self.config.agent_count)
        self.balances = allocate_balances(self.config)
        self.memory = {card.agent_id: VipMemory(self.config.memory_depth) for card in self.cards}
        self.rules: dict[str, ReducedRule] = {}
        self.feed: list[str] = []

    def attach(self, bus: Any, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def note_trigger(self, reason: str) -> None:
        """Fire a decision on the next step, ahead of the cadence (§7.1)."""
        self._pending.append(reason)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        return self.on_tick(t, new_events)

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        if self.scenario is None:
            return []
        triggered = bool(self._pending)
        self._pending.clear()
        for event in new_events:
            if isinstance(event, AttentionTick):
                crossed = self.attention < self.config.attention_trigger <= event.level
                self.attention = float(event.level)
                if crossed:
                    triggered = True
        due = int(t) > 0 and int(t) % self.config.interval_minutes == 0
        if not due and not triggered:
            return []
        if self._last_decision_t == t:
            return []
        self._last_decision_t = t
        return self._decide(t)

    def _decide(self, t: float) -> list[BaseModel]:
        if self.scenario is None:
            return []
        preset = self.scenario.preset
        community = is_community_preset(preset)
        contexts = self._contexts()
        events: list[BaseModel] = []
        for group in batched_cards(self.cards, self.config.batch_size):
            batch = self._complete_group(group, contexts, community)
            events.extend(self._apply_batch(t, group, contexts, batch))
        return events

    def _contexts(self) -> dict[str, ExposureContext]:
        contexts: dict[str, ExposureContext] = {}
        for card, balance in zip(self.cards, self.balances, strict=True):
            insured = min(balance, self.config.insurance_limit)
            contexts[card.agent_id] = ExposureContext(
                balance=balance,
                insured=insured,
                attention=self.attention,
            )
        return contexts

    def _complete_group(
        self,
        group: tuple[PersonaCard, ...],
        contexts: dict[str, ExposureContext],
        community: bool,
    ) -> VipBatchReply:
        if self.scenario is None:
            raise RuntimeError("M41 is not attached")
        agents: list[dict[str, Any]] = []
        for card in group:
            context = contexts[card.agent_id]
            features = features_for(context, self.config.baseline_belief)
            agents.append(
                _agent_payload(card, context, features, self.memory[card.agent_id].snapshot())
            )
        prompt = build_vip_prompt(
            preset=self.scenario.preset,
            seed=self.scenario.seed,
            community=community,
            attention=self.attention,
            agents=agents,
        )
        expected = tuple(card.agent_id for card in group)
        return self._complete_prompt(prompt, expected)

    def _complete_prompt(self, prompt: str, expected: tuple[str, ...]) -> VipBatchReply:
        try:
            if self._provider is not None:
                raw = self._provider.complete_json(prompt)
            elif live_grok_enabled():
                raw = cached_json_completion(
                    prompt=prompt,
                    model=effective_cache_model(self.config.model),
                    provider=_GrokVipBridge(prompt, self.config.model),
                    cache=self._cache,
                )
            else:
                raw = self._rule.complete_json(prompt)
            batch = validate_vip_batch(raw, expected_ids=expected)
        except VipRateLimited:
            self.gate.on_rate_limit()
            batch = self._fallback(prompt, expected)
        except (ValueError, RuntimeError):
            batch = self._fallback(prompt, expected)
        else:
            self.gate.on_success()
        return batch

    def _fallback(self, prompt: str, expected: tuple[str, ...]) -> VipBatchReply:
        request = json.loads(prompt)
        if self.rules and all(agent["agent_id"] in self.rules for agent in request["agents"]):
            decisions: list[dict[str, Any]] = []
            for agent in request["agents"]:
                rule = self.rules[agent["agent_id"]]
                features = {key: float(agent["features"][key]) for key in agent["features"]}
                action = replay_reduced_rule(rule, features)
                if action not in _ACTION_TOOL:
                    action = "stay"
                decisions.append(
                    {
                        "agent_id": rule.agent_id,
                        "action": action,
                        "amount": rule.amount if action == rule.action else 0.0,
                        "rationale": _rationale_for(action),
                        "tool": _ACTION_TOOL[action],
                    }
                )
            return validate_vip_batch({"decisions": decisions}, expected_ids=expected)
        raw = self._rule.complete_json(prompt)
        return validate_vip_batch(raw, expected_ids=expected)

    def _apply_batch(
        self,
        t: float,
        group: tuple[PersonaCard, ...],
        contexts: dict[str, ExposureContext],
        batch: VipBatchReply,
    ) -> list[BaseModel]:
        if self.scenario is None:
            return []
        by_id = {item.agent_id: item for item in batch.decisions}
        out: list[BaseModel] = []
        for card in group:
            decision = by_id[card.agent_id]
            context = contexts[card.agent_id]
            features = features_for(context, self.config.baseline_belief)
            rationale = _safe_text(decision.rationale)
            decision = decision.model_copy(update={"rationale": rationale})
            self._execute(card, decision, context)
            rule = reduce_decision(
                decision,
                features,
                sensitivities=self.config.sensitivities,
                margin=self.config.margin,
            )
            self.rules[card.agent_id] = rule
            if card.portfolio_ids and decision.action != "stay":
                if decision.action == "message_group":
                    directive_action = "withdraw"
                else:
                    directive_action = decision.action
                out.append(
                    DirectiveIssued(
                        t=t,
                        seed=self.scenario.seed,
                        bank_id=bank_id_for_preset(self.scenario.preset),
                        cause_id=self._cause(f"directive-{card.agent_id}"),
                        vc_id=card.agent_id,
                        portfolio_ids=list(card.portfolio_ids),
                        action=directive_action,
                    )
                )
            out.append(
                VIPDecision(
                    t=t,
                    seed=self.scenario.seed,
                    bank_id=bank_id_for_preset(self.scenario.preset),
                    cause_id=self._cause(f"vip-{card.agent_id}"),
                    agent_id=card.agent_id,
                    action=decision.action,
                    amount=decision.amount,
                    rationale=rationale,
                )
            )
            self.feed.append(format_feed_line(card.display_name, decision.action, decision.amount))
        return out

    def _execute(
        self,
        card: PersonaCard,
        decision: VipAgentDecision,
        context: ExposureContext,
    ) -> None:
        if decision.tool == "check_exposure":
            view = check_exposure(card, context)
            line = f"check_exposure gap={view.uninsured_gap:.4f}"
        elif decision.tool == "move_funds":
            moved = move_funds(card, decision.amount)
            line = f"move_funds {moved.amount:.0f}"
        elif decision.tool == "post":
            posted = post(card, decision.rationale)
            line = f"post {posted.text}"
        else:
            sent = message_group(card, decision.rationale)
            line = f"message_group {','.join(sent.portfolio_ids)}"
        self.memory[card.agent_id].add(line)

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"
