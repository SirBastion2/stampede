"""M23 bank management rule core — Wave C (§5, §18.3, §19)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from contracts.events import (
    AttentionTick,
    CEOAction,
    PriceTick,
    StatementIssued,
    Tick,
)
from contracts.schemas import Scenario
from engine.bank.ledger import (
    bank_id_for,
    cet1_ratio,
    initial_ledger_state,
    registry_deposits_total,
)
from mocks.bus import EventBus
from mocks.support import (
    OPENING_CET1,
    OPENING_PRICE,
    bank_name_for,
    deposits_base,
    fold_book,
    unit_interval,
)
from params.loader import RegistryError, get_parameter

_PARAM_CASH_BORROW = "Bank management: borrow when cash below threshold"
_PARAM_CAPITAL_RAISE = "Bank management: raise capital when capital below threshold"

# ASSUMED fallbacks when registry Value is still TBD (§18.3; ADR-022).
_ASSUMED_CASH_BORROW_THRESHOLD = 0.0
_ASSUMED_CAPITAL_RAISE_CET1 = 0.10
_ASSUMED_DIVIDEND_CUT_CET1 = 0.11
_STOCK_SLIDE_FRACTION = 0.95
_TARGET_CET1 = OPENING_CET1

MANAGEMENT_ACTIONS: tuple[str, ...] = (
    "borrow_facility",
    "announce_capital_raise",
    "cut_dividend",
    "sell_assets",
    "issue_statement",
    "call_top_clients",
)

_DEFAULT_SEGMENTS: tuple[str, ...] = (
    "startup",
    "vc",
    "founder",
    "treasurer",
    "retail",
)


@dataclass(frozen=True, slots=True)
class ManagementPolicy:
    """Thresholds read from §8.1; TBD registry cells use ASSUMED defaults."""

    cash_borrow_threshold: float
    capital_raise_cet1_threshold: float
    dividend_cut_cet1_threshold: float
    cash_threshold_assumed: bool
    capital_threshold_assumed: bool


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    t: float
    action: str
    rationale: str
    params: dict[str, Any]


def _parse_threshold_cell(value: str, *, assumed_default: float) -> tuple[float, bool]:
    text = value.strip()
    if not text or text.upper() == "TBD":
        return assumed_default, True
    lowered = text.lower()
    if lowered.endswith("%") or "cet1" in lowered:
        match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", lowered.replace("%", ""))
        if match is None:
            return assumed_default, True
        number = float(match.group(1))
        if "%" in text or number > 1.0:
            return number / 100.0, False
        return number, False
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", lowered.replace(",", ""))
    if match is None:
        return assumed_default, True
    return float(match.group(1)), False


def load_management_policy(path: Path | None = None) -> ManagementPolicy:
    """Load M23 policy thresholds from the registry (§19)."""
    cash_row = get_parameter(_PARAM_CASH_BORROW, path)
    cap_row = get_parameter(_PARAM_CAPITAL_RAISE, path)
    cash, cash_assumed = _parse_threshold_cell(
        cash_row.value, assumed_default=_ASSUMED_CASH_BORROW_THRESHOLD
    )
    cap, cap_assumed = _parse_threshold_cell(
        cap_row.value, assumed_default=_ASSUMED_CAPITAL_RAISE_CET1
    )
    if cash_assumed and cash_row.value.strip().upper() != "TBD":
        cash_assumed = False
    if cap_assumed and cap_row.value.strip().upper() != "TBD":
        cap_assumed = False
    return ManagementPolicy(
        cash_borrow_threshold=cash,
        capital_raise_cet1_threshold=cap,
        dividend_cut_cet1_threshold=_ASSUMED_DIVIDEND_CUT_CET1,
        cash_threshold_assumed=cash_assumed,
        capital_threshold_assumed=cap_assumed,
    )


def capital_raise_success_probability(
    stock_return: float,
    attention: float,
    capital_gap: float,
) -> float:
    """Success odds from stock price, attention, and the capital shortfall (§5 M23)."""
    raw = 0.55 + (0.25 * stock_return) - (0.35 * attention) - (2.0 * capital_gap)
    return max(0.05, min(0.95, raw))


def roll_capital_raise_success(
    seed: int,
    bank_id: str,
    salt: str,
    probability: float,
) -> bool:
    """Deterministic Bernoulli draw for a capital raise (no LLM)."""
    draw = unit_interval(seed, f"{bank_id}:{salt}:capital-raise")
    return draw < probability


def _latest_attention(events: list[BaseModel], bank_id: str) -> float:
    level = 0.0
    for event in events:
        if isinstance(event, AttentionTick) and event.bank_id == bank_id:
            level = event.level
    return level


def _book_snapshot(events: list[BaseModel], *, opening_cet1: float) -> tuple[float, float]:
    book = fold_book(events, deposits=deposits_base(), opening_cet1=opening_cet1)
    return book.cash, book.cet1


def statement_template(
    *,
    preset: str,
    action: str,
    succeeded: bool,
) -> tuple[str, str]:
    """Deterministic press-release copy (template path, §7)."""
    name = bank_name_for(preset)
    if action == "announce_capital_raise":
        if succeeded:
            return (
                "press_release",
                f"{name} announces a planned capital raise to strengthen its balance sheet",
            )
        return (
            "failed_capital_raise",
            f"{name} struggles to secure investors for its announced capital raise",
        )
    if action == "issue_statement":
        return (
            "press_release",
            f"{name} reaffirms liquidity and client support",
        )
    return (
        "press_release",
        f"{name} addresses market speculation",
    )


def build_statement_issued(
    *,
    t: float,
    seed: int,
    bank_id: str,
    cause_id: str,
    stmt_id: str,
    channel: str,
    segments: list[str],
) -> StatementIssued:
    return StatementIssued(
        t=t,
        seed=seed,
        bank_id=bank_id,
        cause_id=cause_id,
        stmt_id=stmt_id,
        channel=channel,
        segments=segments,
    )


class ManagementModule:
    """M23 rule core: policy actions, CEO menu, capital-raise model (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("StatementIssued",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "PriceTick", "CEOAction")

    def __init__(
        self,
        module_id: str = "M23",
        *,
        policy: ManagementPolicy | None = None,
    ) -> None:
        self.module_id = module_id
        self.policy = policy
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self.actions: list[ActionLogEntry] = []
        self._cursor = 0
        self._seq = 0
        self.open_price: float | None = None
        self._statement_sent = False
        self._policy_flags: dict[str, bool] = {
            "borrow": False,
            "raise": False,
            "dividend": False,
        }

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        if self.policy is None:
            try:
                self.policy = load_management_policy()
            except RegistryError:
                self.policy = ManagementPolicy(
                    cash_borrow_threshold=_ASSUMED_CASH_BORROW_THRESHOLD,
                    capital_raise_cet1_threshold=_ASSUMED_CAPITAL_RAISE_CET1,
                    dividend_cut_cet1_threshold=_ASSUMED_DIVIDEND_CUT_CET1,
                    cash_threshold_assumed=True,
                    capital_threshold_assumed=True,
                )
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None or self.policy is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        out = self._on_tick(t, new_events)
        self.published.extend(out)
        return out

    def _on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        scenario = self.scenario
        assert scenario is not None
        bus = self.bus
        assert bus is not None
        policy = self.policy
        assert policy is not None

        ceo_action: CEOAction | None = None
        for event in new_events:
            if isinstance(event, PriceTick) and self.open_price is None:
                self.open_price = event.price
            elif isinstance(event, CEOAction):
                ceo_action = event

        ceo = ceo_action is not None
        sliding = False
        if self.open_price is not None:
            for event in new_events:
                if isinstance(event, PriceTick):
                    floor = self.open_price * _STOCK_SLIDE_FRACTION
                    if event.price <= floor:
                        sliding = True

        out: list[BaseModel] = []
        if ceo_action is not None:
            self._log(ceo_action.action, t, ceo_action.rationale, dict(ceo_action.params))
            if ceo_action.action == "announce_capital_raise":
                self._record_capital_raise_attempt(t, salt=f"ceo-{ceo_action.cause_id or 'menu'}")

        if not self._statement_sent and (ceo or sliding):
            out.extend(self._issue_fake_aligned_statement(t))

        if any(isinstance(event, Tick) for event in new_events):
            out.extend(self._apply_policy_rules(t, bus.log))

        return out

    def _issue_fake_aligned_statement(self, t: float) -> list[BaseModel]:
        """Same payload shape as ``ManagementFake`` (Wave A contract)."""
        if self._statement_sent:
            return []
        scenario = self.scenario
        assert scenario is not None
        self._statement_sent = True
        return [
            build_statement_issued(
                t=t,
                seed=scenario.seed,
                bank_id=self.bank_id,
                cause_id=self._cause("stmt"),
                stmt_id="stmt-management-1",
                channel="press_release",
                segments=list(_DEFAULT_SEGMENTS),
            )
        ]

    def _record_capital_raise_attempt(self, t: float, *, salt: str) -> None:
        scenario = self.scenario
        assert scenario is not None
        bus = self.bus
        assert bus is not None
        cash, cet1 = _book_snapshot(bus.log, opening_cet1=OPENING_CET1)
        attention = _latest_attention(bus.log, self.bank_id)
        open_px = self.open_price if self.open_price is not None else OPENING_PRICE
        last_px = open_px
        for event in bus.log:
            if isinstance(event, PriceTick) and event.bank_id == self.bank_id:
                last_px = event.price
        stock_return = (last_px - open_px) / open_px if open_px > 0 else 0.0
        gap = max(0.0, _TARGET_CET1 - cet1)
        probability = capital_raise_success_probability(stock_return, attention, gap)
        succeeded = roll_capital_raise_success(
            scenario.seed,
            self.bank_id,
            salt,
            probability,
        )
        self._log(
            "announce_capital_raise",
            t,
            (
                f"capital raise attempt; p_success={probability:.3f}; "
                f"outcome={'ok' if succeeded else 'failed'}"
            ),
            {"cash": cash, "cet1": cet1, "probability": probability, "succeeded": succeeded},
        )

    def _apply_policy_rules(self, t: float, log: list[BaseModel]) -> list[BaseModel]:
        policy = self.policy
        assert policy is not None
        cash, cet1 = _book_snapshot(log, opening_cet1=OPENING_CET1)
        out: list[BaseModel] = []
        if cash < policy.cash_borrow_threshold and not self._policy_flags["borrow"]:
            self._policy_flags["borrow"] = True
            self._log(
                "borrow_facility",
                t,
                "cash fell below policy threshold; draw from funding ladder (M12)",
                {"cash": cash, "threshold": policy.cash_borrow_threshold},
            )
        if cet1 < policy.capital_raise_cet1_threshold and not self._policy_flags["raise"]:
            self._policy_flags["raise"] = True
            self._record_capital_raise_attempt(t, salt="policy")
            channel, _ = statement_template(
                preset=self.scenario.preset if self.scenario else "tech-concentrated",
                action="announce_capital_raise",
                succeeded=False,
            )
            if not self._statement_sent:
                scenario = self.scenario
                assert scenario is not None
                self._statement_sent = True
                out.append(
                    build_statement_issued(
                        t=t,
                        seed=scenario.seed,
                        bank_id=self.bank_id,
                        cause_id=self._cause("stmt-policy"),
                        stmt_id="stmt-management-policy-raise",
                        channel=channel,
                        segments=list(_DEFAULT_SEGMENTS),
                    )
                )
        if cet1 < policy.dividend_cut_cet1_threshold and not self._policy_flags["dividend"]:
            self._policy_flags["dividend"] = True
            self._log("cut_dividend", t, "capital cushion low; dividend cut", {"cet1": cet1})
        return out

    def _log(self, action: str, t: float, rationale: str, params: dict[str, Any]) -> None:
        self.actions.append(ActionLogEntry(t=t, action=action, rationale=rationale, params=params))

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


def opening_cet1_from_registry() -> float:
    """Helper for tests: CET1 on the opening book."""
    state = initial_ledger_state(deposits=registry_deposits_total())
    return cet1_ratio(state)
