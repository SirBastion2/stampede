"""M42 bank CEO agents — Grok strategists on the M23 menu (§5, §18.3, §19)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

from pydantic import BaseModel, Field, ValidationError

from agents.llm_cache import (
    LLMCache,
    RemoteJsonBridge,
    cached_json_completion,
    effective_cache_model,
    live_json_completion_enabled,
    offline_mode,
)
from contracts.events import (
    AssetSold,
    AttentionTick,
    BankClosed,
    CapitalChanged,
    CEOAction,
    DepositRateChanged,
    HeadlinePublished,
    LiquidityDrawn,
    LossRealized,
    PaymentSettled,
    PriceTick,
    RatesChanged,
    TradingHalt,
)
from contracts.schemas import Scenario
from engine.bank.ledger import (
    LedgerState,
    bank_id_for,
    cet1_ratio,
    initial_ledger_state,
    policy_rate_at,
    post_asset_sale,
    post_liquidity_drawn,
    post_loss_realized,
    post_payment_settled,
    registry_deposits_total,
    securities_sale_impact,
)
from engine.bank.management import (
    MANAGEMENT_ACTIONS,
    ManagementPolicy,
    capital_raise_success_probability,
    load_management_policy,
    statement_template,
)
from mocks.bus import EventBus
from mocks.support import bank_name_for
from params.loader import get_parameter

_PARAM_ATTENTION = "M42 attention trigger (ASSUMED)"
_PARAM_SLIDE = "M42 stock-slide trigger (ASSUMED)"
_PARAM_RATE = "M42 deposit rate (ASSUMED)"
_PARAM_RATE_CAP = "M42 deposit rate cap (ASSUMED)"
_PARAM_RATE_FLOOR = "M42 deposit rate floor (ASSUMED)"
_PARAM_SEGMENT = "M42 deposit rate segment (ASSUMED)"
_PARAM_SEGMENT_FRACTION = "M42 deposit segment balance fraction (ASSUMED)"
_PARAM_HAIRCUT = "M42 borrow collateral haircut (ASSUMED)"
_PARAM_SALE_FRACTION = "M42 securities sale fraction (ASSUMED)"
_PARAM_MODEL = "M42 CEO Grok model (ASSUMED)"
_PARAM_FLEET = "M42 CEO fleet size (ASSUMED)"
_PARAM_COOLDOWN = "M42 decision cooldown minutes (ASSUMED)"
_PARAM_SENS_ATTENTION = "M42 sensitivity attention (ASSUMED)"
_PARAM_SENS_CET1 = "M42 sensitivity cet1 (ASSUMED)"
_PARAM_SENS_STOCK = "M42 sensitivity stock return (ASSUMED)"
_PARAM_SENS_CASH = "M42 sensitivity cash (ASSUMED)"
_PARAM_GROK = "M42 grok live calls enabled (ASSUMED)"

_MENU = frozenset(MANAGEMENT_ACTIONS)
_HIT_TYPES = (
    AttentionTick,
    PriceTick,
    CapitalChanged,
    LossRealized,
    HeadlinePublished,
    TradingHalt,
    LiquidityDrawn,
    PaymentSettled,
)
_LEDGER_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class CEOPolicy:
    """ASSUMED knobs from the parameter registry (§8.1)."""

    attention_trigger: float
    stock_slide_trigger: float
    deposit_rate: float
    deposit_rate_cap: float
    deposit_rate_floor: float
    deposit_segment: str
    segment_balance_fraction: float
    borrow_haircut: float
    sale_fraction: float
    model: str
    ceo_count: int
    cooldown_minutes: float
    sensitivity_attention: float
    sensitivity_cet1: float
    sensitivity_stock: float
    sensitivity_cash: float
    grok_enabled: bool


@dataclass(frozen=True, slots=True)
class LedgerVerdict:
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CEODecision:
    """One mapped tool call, plus an optional deposit-rate change."""

    action: str
    params: dict[str, Any]
    rationale: str
    deposit_segment: str | None
    deposit_rate: float | None
    on_menu: bool
    menu_reason: str


@dataclass(frozen=True, slots=True)
class ReducedCEORule:
    """Thresholds, sensitivities, and the action futures replay without Grok (§7.1)."""

    action: str
    attention_threshold: float
    cet1_threshold: float
    stock_return_threshold: float
    cash_threshold: float
    sensitivities: dict[str, float]
    deposit_segment: str
    deposit_rate: float


@dataclass(frozen=True, slots=True)
class CEOLogEntry:
    t: float
    bank_id: str
    action: str
    params: dict[str, Any]
    rationale: str
    accepted: bool
    refusal_reason: str


@dataclass(frozen=True, slots=True)
class BankSnapshot:
    preset: str
    seed: int
    bank_id: str
    t: float
    cash: float
    cet1: float
    securities: float
    deposits: float
    attention: float
    stock_return: float
    policy_rate: float
    loss_hit: bool
    hit_names: tuple[str, ...]


class _ToolBody(BaseModel):
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    deposit_segment: str | None = None
    deposit_rate: float | None = None


class CEOProvider(Protocol):
    def choose(self, snapshot: BankSnapshot) -> CEODecision: ...


def _float_param(name: str, path: Path | None) -> float:
    return float(get_parameter(name, path).value)


def _bool_param(name: str, path: Path | None) -> bool:
    return get_parameter(name, path).value.strip().lower() in {"1", "true", "yes", "on"}


def load_ceo_policy(path: Path | None = None) -> CEOPolicy:
    """Load M42 ASSUMED rows through ``params/loader.py`` (§19)."""
    return CEOPolicy(
        attention_trigger=_float_param(_PARAM_ATTENTION, path),
        stock_slide_trigger=_float_param(_PARAM_SLIDE, path),
        deposit_rate=_float_param(_PARAM_RATE, path),
        deposit_rate_cap=_float_param(_PARAM_RATE_CAP, path),
        deposit_rate_floor=_float_param(_PARAM_RATE_FLOOR, path),
        deposit_segment=get_parameter(_PARAM_SEGMENT, path).value.strip(),
        segment_balance_fraction=_float_param(_PARAM_SEGMENT_FRACTION, path),
        borrow_haircut=_float_param(_PARAM_HAIRCUT, path),
        sale_fraction=_float_param(_PARAM_SALE_FRACTION, path),
        model=get_parameter(_PARAM_MODEL, path).value.strip(),
        ceo_count=int(float(get_parameter(_PARAM_FLEET, path).value)),
        cooldown_minutes=_float_param(_PARAM_COOLDOWN, path),
        sensitivity_attention=_float_param(_PARAM_SENS_ATTENTION, path),
        sensitivity_cet1=_float_param(_PARAM_SENS_CET1, path),
        sensitivity_stock=_float_param(_PARAM_SENS_STOCK, path),
        sensitivity_cash=_float_param(_PARAM_SENS_CASH, path),
        grok_enabled=_bool_param(_PARAM_GROK, path),
    )


def _for_bank(event: BaseModel, bank_id: str | None) -> bool:
    if bank_id is None:
        return True
    owner = getattr(event, "bank_id", None)
    return owner is None or owner == bank_id


def ledger_state_from_events(
    events: list[BaseModel],
    *,
    bank_id: str | None = None,
    deposits: float | None = None,
) -> LedgerState:
    """Replay M08 postings so affordability sees the same book as the ledger."""
    total = registry_deposits_total() if deposits is None else deposits
    state = initial_ledger_state(deposits=total)
    for event in events:
        if not _for_bank(event, bank_id):
            continue
        try:
            if isinstance(event, PaymentSettled) and event.amount > 0.0:
                state = post_payment_settled(state, event.amount)
            elif isinstance(event, LiquidityDrawn) and event.amount > 0.0:
                state = post_liquidity_drawn(state, event.amount)
            elif isinstance(event, LossRealized) and event.amount > 0.0:
                state = post_loss_realized(state, event.amount)
            elif isinstance(event, AssetSold) and event.amount > 0.0:
                state = post_asset_sale(
                    state,
                    book_amount=event.amount,
                    price_impact=event.price_impact,
                )
        except ValueError:
            continue
    return state


def _amount(params: dict[str, Any]) -> float | None:
    raw = params.get("amount", params.get("book_amount"))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def ledger_allows_action(
    state: LedgerState,
    action: str,
    params: dict[str, Any],
    *,
    policy: CEOPolicy | None = None,
) -> LedgerVerdict:
    """Refuse menu actions the balance sheet cannot fund (§18 M42)."""
    rules = policy or load_ceo_policy()
    if action not in _MENU:
        return LedgerVerdict(False, "not on M23 action menu")
    if action == "sell_assets":
        amount = _amount(params)
        if amount is None or amount <= 0.0:
            return LedgerVerdict(False, "sale amount must be positive")
        if amount > state.securities + _LEDGER_EPS:
            return LedgerVerdict(False, "sale exceeds securities on the ledger")
        impact = params.get("price_impact", 0.0)
        price_impact = float(impact) if isinstance(impact, (int, float)) else 0.0
        try:
            post_asset_sale(state, book_amount=amount, price_impact=price_impact)
        except ValueError as exc:
            return LedgerVerdict(False, str(exc))
        return LedgerVerdict(True, "")
    if action == "borrow_facility":
        amount = _amount(params)
        if amount is None or amount <= 0.0:
            return LedgerVerdict(False, "borrow amount must be positive")
        capacity = max(0.0, state.securities * (1.0 - rules.borrow_haircut))
        if amount > capacity + _LEDGER_EPS:
            return LedgerVerdict(False, "borrow exceeds unpledged collateral")
        try:
            post_liquidity_drawn(state, amount)
        except ValueError as exc:
            return LedgerVerdict(False, str(exc))
        return LedgerVerdict(True, "")
    return LedgerVerdict(True, "")


def ledger_allows_deposit_rate(
    state: LedgerState,
    rate: float,
    *,
    policy: CEOPolicy | None = None,
) -> LedgerVerdict:
    """Refuse a deposit rate the cash balance cannot carry."""
    rules = policy or load_ceo_policy()
    if rate < rules.deposit_rate_floor or rate > rules.deposit_rate_cap:
        return LedgerVerdict(False, "deposit rate outside ledger bounds")
    cost = rate * state.deposits * rules.segment_balance_fraction
    if cost > state.cash + _LEDGER_EPS:
        return LedgerVerdict(False, "deposit rate interest exceeds cash")
    return LedgerVerdict(True, "")


def map_tool_call(payload: dict[str, Any]) -> CEODecision:
    """Map a Grok tool payload onto ``MANAGEMENT_ACTIONS`` (§19)."""
    try:
        parsed = _ToolBody.model_validate(payload)
    except ValidationError:
        return CEODecision(
            action="",
            params={},
            rationale="",
            deposit_segment=None,
            deposit_rate=None,
            on_menu=False,
            menu_reason="invalid tool payload",
        )
    params = dict(parsed.params)
    if parsed.action not in _MENU:
        return CEODecision(
            action=parsed.action,
            params=params,
            rationale=parsed.rationale,
            deposit_segment=parsed.deposit_segment,
            deposit_rate=parsed.deposit_rate,
            on_menu=False,
            menu_reason="not on M23 action menu",
        )
    return CEODecision(
        action=parsed.action,
        params=params,
        rationale=parsed.rationale,
        deposit_segment=parsed.deposit_segment,
        deposit_rate=parsed.deposit_rate,
        on_menu=True,
        menu_reason="",
    )


def _capital_gap(cet1: float, management: ManagementPolicy) -> float:
    return max(0.0, management.capital_raise_cet1_threshold - cet1)


def _decision(
    action: str,
    params: dict[str, Any],
    rationale: str,
    policy: CEOPolicy,
) -> CEODecision:
    return CEODecision(
        action=action,
        params=params,
        rationale=rationale,
        deposit_segment=policy.deposit_segment,
        deposit_rate=policy.deposit_rate,
        on_menu=True,
        menu_reason="",
    )


def offline_ceo_decision(
    snapshot: BankSnapshot,
    *,
    policy: CEOPolicy | None = None,
    management: ManagementPolicy | None = None,
) -> CEODecision:
    """Deterministic menu choice from M23 policy helpers. No network (§7 fallback)."""
    rules = policy or load_ceo_policy()
    management_rules = management or load_management_policy()
    name = bank_name_for(snapshot.preset)
    segment = rules.deposit_segment
    if snapshot.cash < management_rules.cash_borrow_threshold:
        shortfall = management_rules.cash_borrow_threshold - snapshot.cash
        capacity = max(0.0, snapshot.securities * (1.0 - rules.borrow_haircut))
        if shortfall > 0.0 and capacity > 0.0:
            amount = min(shortfall, capacity)
            return _decision(
                "borrow_facility",
                {"amount": amount, "segment": segment},
                f"{name} draws on facilities because cash is below the management threshold",
                rules,
            )
    if snapshot.cet1 < management_rules.capital_raise_cet1_threshold:
        gap = _capital_gap(snapshot.cet1, management_rules)
        probability = capital_raise_success_probability(
            snapshot.stock_return,
            snapshot.attention,
            gap,
        )
        return _decision(
            "announce_capital_raise",
            {"probability": probability, "capital_gap": gap, "segment": segment},
            (
                f"{name} proposes a capital raise; "
                f"success probability {probability:.3f} from stock, attention, and the capital gap"
            ),
            rules,
        )
    if snapshot.cet1 < management_rules.dividend_cut_cet1_threshold:
        return _decision(
            "cut_dividend",
            {"cet1": snapshot.cet1, "segment": segment},
            f"{name} cuts the dividend because capital is below the management threshold",
            rules,
        )
    if snapshot.stock_return <= rules.stock_slide_trigger and snapshot.securities > 0.0:
        amount = snapshot.securities * rules.sale_fraction
        if amount > 0.0:
            impact = securities_sale_impact(snapshot.policy_rate)
            params = {
                "amount": amount,
                "price_impact": impact,
                "asset_class": "afs",
                "segment": segment,
            }
            return _decision(
                "sell_assets",
                params,
                f"{name} sells securities after the stock slide",
                rules,
            )
    _channel, text = statement_template(
        preset=snapshot.preset,
        action="issue_statement",
        succeeded=True,
    )
    return _decision(
        "issue_statement",
        {"segment": segment},
        text,
        rules,
    )


def reduce_ceo_decision(
    decision: CEODecision,
    *,
    policy: CEOPolicy | None = None,
    management: ManagementPolicy | None = None,
) -> ReducedCEORule:
    """Store the threshold, sensitivities, and action for futures replay (§7.1)."""
    rules = policy or load_ceo_policy()
    management_rules = management or load_management_policy()
    segment = decision.deposit_segment or rules.deposit_segment
    rate = rules.deposit_rate if decision.deposit_rate is None else decision.deposit_rate
    return ReducedCEORule(
        action=decision.action,
        attention_threshold=rules.attention_trigger,
        cet1_threshold=management_rules.capital_raise_cet1_threshold,
        stock_return_threshold=rules.stock_slide_trigger,
        cash_threshold=management_rules.cash_borrow_threshold,
        sensitivities={
            "attention": rules.sensitivity_attention,
            "cet1": rules.sensitivity_cet1,
            "stock_return": rules.sensitivity_stock,
            "cash": rules.sensitivity_cash,
        },
        deposit_segment=segment,
        deposit_rate=rate,
    )


def replay_reduced_rule(
    rule: ReducedCEORule,
    *,
    attention: float,
    cet1: float,
    stock_return: float,
    cash: float,
) -> str | None:
    """Replay a stored CEO rule. Does not call Grok."""
    if (
        attention >= rule.attention_threshold
        or cet1 < rule.cet1_threshold
        or stock_return <= rule.stock_return_threshold
        or cash < rule.cash_threshold
    ):
        return rule.action
    return None


def _stress(
    snapshot: BankSnapshot,
    policy: CEOPolicy,
    management: ManagementPolicy,
) -> bool:
    if snapshot.loss_hit:
        return True
    if snapshot.attention >= policy.attention_trigger:
        return True
    if snapshot.cash < management.cash_borrow_threshold:
        return True
    if snapshot.cet1 < management.dividend_cut_cet1_threshold:
        return True
    return snapshot.stock_return <= policy.stock_slide_trigger


def bank_snapshot(
    events: list[BaseModel],
    new_events: list[BaseModel],
    *,
    preset: str,
    seed: int,
    bank_id: str,
    t: float,
    scenario: Scenario | None = None,
) -> BankSnapshot:
    """Bank state from the ledger replay plus events that just hit the bank."""
    state = ledger_state_from_events(events, bank_id=bank_id)
    attention = 0.0
    open_px: float | None = None
    last_px: float | None = None
    seen_rate: float | None = None
    names: list[str] = []
    loss_hit = False
    for event in events:
        if not _for_bank(event, bank_id):
            continue
        if isinstance(event, AttentionTick):
            attention = event.level
        elif isinstance(event, PriceTick):
            if open_px is None:
                open_px = event.price
            last_px = event.price
        elif isinstance(event, RatesChanged):
            seen_rate = event.policy_rate
    for event in new_events:
        if not _for_bank(event, bank_id):
            continue
        if isinstance(event, _HIT_TYPES):
            names.append(event.event_type)
        if isinstance(event, LossRealized):
            loss_hit = True
    if open_px is not None and last_px is not None and open_px > 0.0:
        stock_return = (last_px - open_px) / open_px
    else:
        stock_return = 0.0
    if seen_rate is None:
        path = scenario.rate_path if scenario is not None else []
        seen_rate = policy_rate_at(path, t)
    return BankSnapshot(
        preset=preset,
        seed=seed,
        bank_id=bank_id,
        t=t,
        cash=state.cash,
        cet1=cet1_ratio(state),
        securities=state.securities,
        deposits=state.deposits,
        attention=attention,
        stock_return=stock_return,
        policy_rate=seen_rate,
        loss_hit=loss_hit,
        hit_names=tuple(names),
    )


def _prompt(snapshot: BankSnapshot) -> str:
    return json.dumps(
        {
            "task": "bank_ceo",
            "preset": snapshot.preset,
            "bank_id": snapshot.bank_id,
            "t": snapshot.t,
            "menu": list(MANAGEMENT_ACTIONS),
            "state": {
                "cash": snapshot.cash,
                "cet1": snapshot.cet1,
                "securities": snapshot.securities,
                "deposits": snapshot.deposits,
                "attention": snapshot.attention,
                "stock_return": snapshot.stock_return,
            },
            "hits": list(snapshot.hit_names),
            "output_schema": {
                "action": "one of the menu names",
                "params": "object",
                "rationale": "string",
                "deposit_segment": "string or null",
                "deposit_rate": "float or null",
            },
        },
        sort_keys=True,
    )


class _GrokJsonBridge(RemoteJsonBridge):
    pass


class RuleBasedCEOProvider:
    """Offline strategist. Deterministic for a fixed snapshot."""

    def choose(self, snapshot: BankSnapshot) -> CEODecision:
        return offline_ceo_decision(snapshot)


class _GrokCEOProvider:
    """Live tool-call path. Only built when offline mode is off and the flag is on."""

    def __init__(self, model: str, cache: LLMCache | None) -> None:
        self.model = model
        self.cache = cache

    def choose(self, snapshot: BankSnapshot) -> CEODecision:
        prompt = _prompt(snapshot)
        try:
            raw = cached_json_completion(
                prompt=prompt,
                model=effective_cache_model(self.model),
                provider=_GrokJsonBridge(prompt, self.model),
                cache=self.cache,
            )
            decision = map_tool_call(raw)
        except (RuntimeError, ValueError, TypeError):
            return offline_ceo_decision(snapshot)
        if not decision.on_menu or not decision.rationale:
            fallback = offline_ceo_decision(snapshot)
            if decision.on_menu and not decision.rationale:
                return CEODecision(
                    action=decision.action,
                    params=decision.params,
                    rationale=fallback.rationale,
                    deposit_segment=decision.deposit_segment or fallback.deposit_segment,
                    deposit_rate=(
                        fallback.deposit_rate
                        if decision.deposit_rate is None
                        else decision.deposit_rate
                    ),
                    on_menu=True,
                    menu_reason="",
                )
            return fallback
        return decision


def default_provider(cache: LLMCache | None = None) -> CEOProvider:
    policy = load_ceo_policy()
    if not policy.grok_enabled or not live_json_completion_enabled():
        return RuleBasedCEOProvider()
    return _GrokCEOProvider(policy.model, cache)


def choose_ceo_action(
    snapshot: BankSnapshot,
    *,
    provider: CEOProvider | None = None,
    cache: LLMCache | None = None,
) -> CEODecision:
    chooser = provider or default_provider(cache)
    return chooser.choose(snapshot)


def _hits_bank(event: BaseModel, bank_id: str) -> bool:
    return _for_bank(event, bank_id) and isinstance(event, _HIT_TYPES)


class CEOModule:
    """One Grok strategist for the scenario bank. Publishes CEOAction (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("DepositRateChanged", "CEOAction")
    consumes: ClassVar[tuple[str, ...]] = ("Tick",)

    def __init__(
        self,
        module_id: str = "M42",
        *,
        policy: CEOPolicy | None = None,
        management: ManagementPolicy | None = None,
        provider: CEOProvider | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.module_id = module_id
        self.policy = policy
        self.management = management
        self._provider = provider
        self._cache = cache
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self.log_by_bank: dict[str, list[CEOLogEntry]] = {}
        self.reduced_rules: dict[str, ReducedCEORule] = {}
        self._log_cursor = 0
        self._seq = 0
        self._last_decision_t: float | None = None

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        if self.policy is None:
            self.policy = load_ceo_policy()
        if self.management is None:
            self.management = load_management_policy()
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.bus.log[self._log_cursor :]
        self._log_cursor = len(self.bus.log)
        out = self.on_tick(t, new_events)
        self.published.extend(out)
        return out

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        if self.scenario is None or self.policy is None or self.management is None:
            return []
        bank_id = self.bank_id
        if any(isinstance(event, BankClosed) and _for_bank(event, bank_id) for event in new_events):
            return []
        if self._cooling(t):
            return []
        hits = [event for event in new_events if _hits_bank(event, bank_id)]
        if not hits:
            return []
        history = list(self.bus.log) if self.bus is not None else []
        seen = {id(event) for event in history}
        combined = history + [event for event in new_events if id(event) not in seen]
        snapshot = bank_snapshot(
            combined,
            new_events,
            preset=self.scenario.preset,
            seed=self.scenario.seed,
            bank_id=bank_id,
            t=t,
            scenario=self.scenario,
        )
        if not _stress(snapshot, self.policy, self.management):
            return []
        decision = choose_ceo_action(snapshot, provider=self._provider, cache=self._cache)
        state = ledger_state_from_events(combined, bank_id=bank_id)
        return self._apply(t, decision, state)

    def _cooling(self, t: float) -> bool:
        policy = self.policy
        if policy is None or self._last_decision_t is None:
            return False
        return (t - self._last_decision_t) < policy.cooldown_minutes

    def _apply(self, t: float, decision: CEODecision, state: LedgerState) -> list[BaseModel]:
        policy = self.policy
        assert policy is not None
        bank_id = self.bank_id
        if not decision.on_menu:
            self._record(
                t,
                bank_id,
                decision.action or "unknown",
                decision.params,
                decision.rationale,
                accepted=False,
                reason=decision.menu_reason or "not on M23 action menu",
            )
            self._last_decision_t = t
            return []
        verdict = ledger_allows_action(state, decision.action, decision.params, policy=policy)
        if not verdict.allowed:
            self._record(
                t,
                bank_id,
                decision.action,
                decision.params,
                decision.rationale,
                accepted=False,
                reason=verdict.reason,
            )
            self._last_decision_t = t
            return []
        out: list[BaseModel] = []
        if decision.deposit_rate is not None and decision.deposit_segment:
            rate_verdict = ledger_allows_deposit_rate(state, decision.deposit_rate, policy=policy)
            if rate_verdict.allowed:
                out.append(
                    DepositRateChanged(
                        t=t,
                        seed=self.seed,
                        bank_id=bank_id,
                        cause_id=self._cause("rate"),
                        segment=decision.deposit_segment,
                        rate=decision.deposit_rate,
                    )
                )
            else:
                self._record(
                    t,
                    bank_id,
                    "set_deposit_rate",
                    {"segment": decision.deposit_segment, "rate": decision.deposit_rate},
                    decision.rationale,
                    accepted=False,
                    reason=rate_verdict.reason,
                )
        out.append(
            CEOAction(
                t=t,
                seed=self.seed,
                bank_id=bank_id,
                cause_id=self._cause("ceo"),
                action=decision.action,
                params=dict(decision.params),
                rationale=decision.rationale,
            )
        )
        self._record(
            t,
            bank_id,
            decision.action,
            decision.params,
            decision.rationale,
            accepted=True,
            reason="",
        )
        self.reduced_rules[bank_id] = reduce_ceo_decision(
            decision,
            policy=policy,
            management=self.management,
        )
        self._last_decision_t = t
        return out

    def _record(
        self,
        t: float,
        bank_id: str,
        action: str,
        params: dict[str, Any],
        rationale: str,
        *,
        accepted: bool,
        reason: str,
    ) -> None:
        self.log_by_bank.setdefault(bank_id, []).append(
            CEOLogEntry(
                t=t,
                bank_id=bank_id,
                action=action,
                params=dict(params),
                rationale=rationale,
                accepted=accepted,
                refusal_reason=reason,
            )
        )

    @property
    def seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return self.scenario.seed

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_id_for(self.scenario.preset)

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"
