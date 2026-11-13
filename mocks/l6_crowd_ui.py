"""L6 fake: crowd-screen snapshot. Can also post a rumor from a ui_rumor shock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import AttentionTick, RumorPosted, WithdrawalRequested
from mocks.base import FakeModule
from mocks.demo_arc import (
    cash_gauge_value,
    rumor_strength,
    time_to_failure_ticks,
    wavering_fraction,
    withdrawn_fraction,
)
from mocks.support import (
    DEPOSITORS_TIER0,
    OPENING_CASH_FRACTION,
    OPENING_CET1,
    claim_allowed,
    clamp,
    deposits_base,
    fold_book,
)


@dataclass(slots=True)
class _QueuedRumor:
    claim: str
    source_node: str
    bank_id: str
    rumor_id: str
    credibility: float


class CrowdFake(FakeModule):
    """M29. Point-cloud counts and gauges derived from the log. Optional UI rumor."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick", "PaymentSettled")
    publishes: ClassVar[tuple[str, ...]] = ("RumorPosted",)

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.calm = DEPOSITORS_TIER0
        self.wavering = 0
        self.withdrew = 0
        self.cash_gauge = 1.0
        self.tipping_point = 0.0
        self.time_to_failure_ticks: float | None = None
        self.rumor = ""
        self._posted: list[int] = []
        self._live_queue: list[_QueuedRumor] = []
        # SVB-50: the live server's real mode sets this and writes the counts from
        # engine state (ui/live/engine_crowd.py). Off keeps the scripted arc.
        self.follow_engine = False

    def queue_live_rumor(
        self,
        *,
        claim: str,
        source_node: str,
        bank_id: str,
        rumor_id: str,
        credibility: float = 0.45,
    ) -> None:
        """Queue a UI rumor for publication on the next ``on_tick`` (M29 publisher)."""
        self._live_queue.append(
            _QueuedRumor(
                claim=claim,
                source_node=source_node,
                bank_id=bank_id,
                rumor_id=rumor_id,
                credibility=float(credibility),
            )
        )

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.follow_engine:
            for event in self.log():
                if isinstance(event, RumorPosted):
                    self.rumor = event.claim
            out = self._maybe_post(t)
            out.extend(self._drain_live_queue(t))
            return out
        attention = 0.0
        withdrawn = 0.0
        for event in self.log():
            if isinstance(event, AttentionTick):
                attention = event.level
            elif isinstance(event, WithdrawalRequested):
                withdrawn += event.amount
            elif isinstance(event, RumorPosted):
                self.rumor = event.claim
        deposits = deposits_base()
        book = fold_book(self.log(), deposits=deposits, opening_cet1=OPENING_CET1)
        log = self.log()
        strength = rumor_strength(log)
        if strength > 0.0:
            left = withdrawn_fraction(t, strength, log)
            wobble = wavering_fraction(t, strength, log)
            self.withdrew = int(DEPOSITORS_TIER0 * left)
            self.wavering = int(DEPOSITORS_TIER0 * wobble)
            if self.withdrew + self.wavering > DEPOSITORS_TIER0:
                self.wavering = DEPOSITORS_TIER0 - self.withdrew
            self.calm = DEPOSITORS_TIER0 - self.withdrew - self.wavering
            self.cash_gauge = cash_gauge_value(t, strength)
            self.time_to_failure_ticks = time_to_failure_ticks(t, strength, self.cash_gauge)
        else:
            withdrawn_share = clamp(withdrawn / deposits if deposits else 0.0, 0.0, 1.0)
            self.withdrew = int(DEPOSITORS_TIER0 * withdrawn_share)
            self.wavering = int(DEPOSITORS_TIER0 * attention * (1.0 - withdrawn_share) * 0.5)
            self.calm = max(0, DEPOSITORS_TIER0 - self.withdrew - self.wavering)
            opening_cash = deposits * OPENING_CASH_FRACTION
            self.cash_gauge = clamp(book.cash / opening_cash if opening_cash else 0.0, 0.0, 1.5)
            if book.unpaid > 1.0:
                self.time_to_failure_ticks = 0.0
        self.tipping_point = attention
        out = self._maybe_post(t)
        out.extend(self._drain_live_queue(t))
        return out

    def _drain_live_queue(self, t: float) -> list[BaseModel]:
        out: list[BaseModel] = []
        while self._live_queue:
            item = self._live_queue.pop(0)
            if not claim_allowed(item.claim):
                continue
            self.rumor = item.claim
            out.append(
                RumorPosted(
                    t=t,
                    seed=self.seed,
                    bank_id=item.bank_id,
                    cause_id=self.cause(item.rumor_id),
                    rumor_id=item.rumor_id,
                    claim=item.claim,
                    source_node=item.source_node,
                    credibility=clamp(item.credibility, 0.0, 1.0),
                )
            )
        return out

    def _maybe_post(self, t: float) -> list[BaseModel]:
        if self.scenario is None:
            return []
        out: list[BaseModel] = []
        for index, shock in enumerate(self.scenario.shocks):
            if shock.kind != "ui_rumor" or index in self._posted or t < shock.at:
                continue
            claim = shock.params.get("claim", "Group chat says the fictional bank is raising cash")
            if not isinstance(claim, str) or not claim_allowed(claim):
                continue
            rumor_id = shock.params.get("rumor_id", f"rumor-ui-{index}")
            source = shock.params.get("source_node", "ui-user")
            credibility = shock.params.get("credibility", 0.4)
            if not isinstance(rumor_id, str):
                rumor_id = f"rumor-ui-{index}"
            if not isinstance(source, str):
                source = "ui-user"
            if not isinstance(credibility, (int, float)) or isinstance(credibility, bool):
                credibility = 0.4
            self._posted.append(index)
            self.rumor = claim
            out.append(
                RumorPosted(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause("ui-rumor"),
                    rumor_id=rumor_id,
                    claim=claim,
                    source_node=source,
                    credibility=clamp(float(credibility), 0.0, 1.0),
                )
            )
        return out


def l6_modules() -> dict[str, FakeModule]:
    return {"M29": CrowdFake("M29")}
