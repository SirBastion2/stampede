"""M20 regulators and supervision triggers (§5, §19)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import BankClosed, SupervisoryAction
from contracts.schemas import Scenario
from mocks.bus import EventBus
from mocks.support import (
    Book,
    bank_id_for,
    calendar_at,
    deposits_base,
    fold_book,
    is_community,
)
from params.loader import RegistryError, get_parameter

_PARAM_CET1_FLOOR = "M20 CET1 closure floor"
_PARAM_OPENING_CET1 = "M20 opening CET1 ratio"
_PARAM_UNPAID_THRESHOLD = "M20 unpaid obligation threshold"
_PARAM_SECURITIES_FLOOR = "M20 securities exhaustion threshold"


@dataclass(frozen=True, slots=True)
class RegulatorConfig:
    cet1_floor: float
    opening_cet1: float
    unpaid_threshold: float
    securities_floor: float


def _parse_fraction(value: str) -> float:
    text = value.strip().lower().replace("%", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        raise RegistryError(f"cannot parse fraction registry value: {value!r}")
    number = float(match.group(1))
    if "%" in value or number > 1.0:
        return number / 100.0
    return number


def _parse_money_threshold(value: str) -> float:
    text = value.strip().lower().replace(",", "").replace("$", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        raise RegistryError(f"cannot parse money threshold: {value!r}")
    return float(match.group(1))


def load_regulator_config() -> RegulatorConfig:
    """Load M20 thresholds from the parameter registry (§8.1)."""
    return RegulatorConfig(
        cet1_floor=_parse_fraction(get_parameter(_PARAM_CET1_FLOOR).value),
        opening_cet1=_parse_fraction(get_parameter(_PARAM_OPENING_CET1).value),
        unpaid_threshold=_parse_money_threshold(
            get_parameter(_PARAM_UNPAID_THRESHOLD).value
        ),
        securities_floor=_parse_money_threshold(
            get_parameter(_PARAM_SECURITIES_FLOOR).value
        ),
    )


def closure_reason(book: Book, cfg: RegulatorConfig) -> str | None:
    """Return a closure reason when liquidity or capital triggers fire (§5 M20)."""
    if book.unpaid > cfg.unpaid_threshold and book.securities <= cfg.securities_floor:
        return "unable to pay obligations"
    if book.cet1 < cfg.cet1_floor:
        return "capital below threshold"
    return None


def watch_recommended(book: Book, cfg: RegulatorConfig) -> bool:
    """Stress short of closure: unpaid pressure or CET1 below opening level."""
    if closure_reason(book, cfg) is not None:
        return False
    return book.unpaid > cfg.unpaid_threshold or book.cet1 < cfg.opening_cet1


def closure_timing_allows(t: float, preset: str) -> bool:
    """SVB-like presets close on business days; Signature-like over a weekend (§5)."""
    cal = calendar_at(t)
    lowered = preset.lower()
    if "crypto" in lowered:
        return cal.weekend
    return cal.business_hours


def simulated_day_index(t: float) -> int:
    """Calendar day index for acceptance checks (same origin as ``calendar_at``)."""
    absolute = 9 * 60 + int(t)
    return absolute // (24 * 60)


class RegulatorModule:
    """M20 supervision and closure rule engine (§19)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick", "CapitalChanged", "PolicyEnacted")
    publishes: ClassVar[tuple[str, ...]] = ("BankClosed", "SupervisoryAction")

    def __init__(
        self,
        module_id: str = "M20",
        *,
        config: RegulatorConfig | None = None,
    ) -> None:
        self.module_id = module_id
        self._config = config
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.status = "open"
        self.closed = False
        self._pending_reason: str | None = None
        self.inbox: list[BaseModel] = []
        self._cursor = 0
        self._seq = 0
        self.published: list[BaseModel] = []

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        out = self.on_tick(t, new_events)
        self.published.extend(out)
        return out

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.scenario is None or self.bus is None or self.closed:
            return []
        if is_community(self.scenario.preset):
            return []
        cfg = self._config or load_regulator_config()
        book = fold_book(
            self.bus.log,
            deposits=deposits_base(),
            opening_cet1=cfg.opening_cet1,
        )
        reason = closure_reason(book, cfg)
        if reason is not None:
            if not closure_timing_allows(t, self.scenario.preset):
                if self._pending_reason != reason:
                    self._pending_reason = reason
                    if self.status != "closure-pending":
                        self.status = "closure-pending"
                        return [self._supervisory(t, "closure-pending")]
                return []
            return self._close(t, reason)
        if self._pending_reason is not None and closure_timing_allows(
            t, self.scenario.preset
        ):
            return self._close(t, self._pending_reason)
        if watch_recommended(book, cfg):
            if self.status != "watch":
                self.status = "watch"
                return [self._supervisory(t, "watch")]
        return []

    def _close(self, t: float, reason: str) -> list[BaseModel]:
        self.closed = True
        self.status = "closed"
        self._pending_reason = None
        return [
            BankClosed(
                t=t,
                seed=self._seed,
                bank_id=self._bank_id,
                cause_id=self._cause("close"),
                reason=reason,
            )
        ]

    def _supervisory(self, t: float, action: str) -> SupervisoryAction:
        return SupervisoryAction(
            t=t,
            seed=self._seed,
            bank_id=self._bank_id,
            cause_id=self._cause(action),
            action=action,
        )

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"

    @property
    def _seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return self.scenario.seed

    @property
    def _bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_id_for(self.scenario.preset)


__all__ = [
    "RegulatorConfig",
    "RegulatorModule",
    "closure_reason",
    "closure_timing_allows",
    "load_regulator_config",
    "simulated_day_index",
    "watch_recommended",
]
