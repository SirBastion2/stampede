"""L5 fakes: media, management, crisis room, personas, guardrails, VIP, CEO, shorts.

No Grok calls. Text is a template so a run stays deterministic and offline.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from contracts.events import (
    AnalystNote,
    AttentionTick,
    CapitalChanged,
    CEOAction,
    DepositRateChanged,
    DirectiveIssued,
    FutureCompleted,
    HeadlinePublished,
    LossRealized,
    PriceTick,
    StatementIssued,
    ThesisPublished,
    VIPDecision,
)
from mocks.base import FakeModule
from mocks.params_access import get_param
from mocks.support import (
    ASSUMED_HEADLINE_ATTENTION,
    OPENING_CASH_FRACTION,
    OPENING_CET1,
    bank_name_for,
    claim_allowed,
    clamp,
    deposits_base,
    is_community,
    seed_scale,
    uninsured_share,
)


class MediaFake(FakeModule):
    """M04. One headline when attention crosses the line, and one when a loss lands.

    """

    publishes: ClassVar[tuple[str, ...]] = ("HeadlinePublished", "AnalystNote")
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "AttentionTick", "LossRealized")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.attention = 0.0
        self.headlines = 0
        self.note: str | None = None
        self._headline_sent = False
        self._loss_sent = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        out: list[BaseModel] = []
        name = bank_name_for(self.scenario.preset) if self.scenario else "Fictional Valley Bank"
        for event in new_events:
            if isinstance(event, AttentionTick):
                crossed = (
                    self.attention < ASSUMED_HEADLINE_ATTENTION <= event.level
                    and not self._headline_sent
                )
                self.attention = event.level
                if crossed:
                    self._headline_sent = True
                    self.headlines += 1
                    reach = int(8_000 * event.level * seed_scale(self.seed, "reach"))
                    out.append(
                        HeadlinePublished(
                            t=t,
                            seed=self.seed,
                            bank_id=self.bank_id,
                            cause_id=self.cause("headline"),
                            text=f"Depositors focus on {name}",
                            reach=max(1, reach),
                        )
                    )
            elif isinstance(event, LossRealized) and not self._loss_sent:
                self._loss_sent = True
                note_text = "Securities sale realized a loss. Template note, not a model call."
                self.note = note_text
                self.headlines += 1
                out.append(
                    HeadlinePublished(
                        t=t,
                        seed=self.seed,
                        bank_id=self.bank_id,
                        cause_id=self.cause("loss-headline"),
                        text=f"{name} books a loss on securities it sold",
                        reach=max(1, int(12_000 * seed_scale(self.seed, "loss-reach"))),
                    )
                )
                out.append(
                    AnalystNote(
                        t=t,
                        seed=self.seed,
                        bank_id=self.bank_id,
                        cause_id=self.cause("analyst-note"),
                        text=note_text,
                        cited_event_ids=[event.cause_id or self.cause("loss-cite")],
                    )
                )
        return out


class ManagementFake(FakeModule):
    """M23. Issues a statement when the CEO acts or the stock is already sliding."""

    publishes: ClassVar[tuple[str, ...]] = ("StatementIssued",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "PriceTick", "CEOAction")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.actions: list[str] = []
        self.open_price: float | None = None
        self._sent = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        ceo = False
        for event in new_events:
            if isinstance(event, PriceTick) and self.open_price is None:
                self.open_price = event.price
            elif isinstance(event, CEOAction):
                self.actions.append(event.action)
                ceo = True
        sliding = False
        if self.open_price:
            for event in new_events:
                if isinstance(event, PriceTick) and event.price <= self.open_price * 0.95:
                    sliding = True
        if self._sent or not (ceo or sliding):
            return []
        self._sent = True
        return [
            StatementIssued(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("stmt"),
                stmt_id="stmt-management-1",
                channel="press_release",
                segments=["startup", "vc", "founder", "treasurer", "retail"],
            )
        ]


class CrisisRoomFake(FakeModule):
    """M24. After futures finish, keep the statement whose failure odds are lowest.

    The truth filter rejects one false cash claim in ``verdicts``. Only the accepted
    statement is published.
    """

    publishes: ClassVar[tuple[str, ...]] = ("StatementIssued",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "FutureCompleted")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.verdicts: list[dict[str, str]] = []
        self._sent = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        futures = [event for event in new_events if isinstance(event, FutureCompleted)]
        if self._sent or not futures:
            return []
        self._sent = True
        self.verdicts = [
            {"stmt_id": "stmt-crisis-1", "verdict": "accepted", "reason": ""},
            {
                "stmt_id": "stmt-crisis-false-cash",
                "verdict": "rejected",
                "reason": "cash claim exceeds the ledger",
            },
        ]
        return [
            StatementIssued(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("crisis"),
                stmt_id="stmt-crisis-1",
                channel="x_post",
                segments=["startup", "founder", "treasurer"],
            )
        ]


class PersonaFake(FakeModule):
    """M25. Segment scores move worry down, matching the reassurance direction. No Grok call."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick", "StatementIssued")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.scores: dict[str, dict[str, float]] = {}

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del t
        for event in new_events:
            if not isinstance(event, StatementIssued):
                continue
            for index, segment in enumerate(event.segments):
                tilt = seed_scale(self.seed, f"{event.stmt_id}:{segment}")
                self.scores[segment] = {
                    "worry_delta": -0.08 * tilt,
                    "threshold_delta": 0.04 * tilt,
                    "act_now_share": clamp(0.15 / (index + 1), 0.0, 1.0),
                }
        return []


class GuardrailFake(FakeModule):
    """M34. Flags real-bank names. Publishes nothing; blocked claims are not rumors we start."""

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.blocked: list[str] = []
        self._scanned = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del t, new_events
        if self._scanned or self.scenario is None:
            return []
        self._scanned = True
        for rumor in self.scenario.rumors:
            if not claim_allowed(rumor.claim):
                self.blocked.append(rumor.id)
        return []


class VIPFake(FakeModule):
    """M41. Top depositors act when attention is already high, or every 30 minutes."""

    publishes: ClassVar[tuple[str, ...]] = ("DirectiveIssued", "VIPDecision")
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "AttentionTick")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.attention = 0.0
        self._acted = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        for event in new_events:
            if isinstance(event, AttentionTick):
                self.attention = event.level
        due = int(t) > 0 and int(t) % 30 == 0
        if self._acted or self.scenario is None:
            return []
        if self.attention < 0.25 and not due:
            return []
        self._acted = True
        community = is_community(self.scenario.preset)
        action = "stay" if community else "withdraw"
        opening_cash = deposits_base() * OPENING_CASH_FRACTION
        pitch = get_param("M41 fake pitch withdrawal cash fraction (ASSUMED)")
        amount = 0.0 if community else opening_cash * pitch * seed_scale(self.seed, "vip")
        return [
            DirectiveIssued(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("directive"),
                vc_id="vip-1",
                portfolio_ids=["port-1", "port-2"],
                action=action,
            ),
            VIPDecision(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("vip"),
                agent_id="vip-1",
                action=action,
                amount=amount,
                rationale=(
                    "insured balance stays"
                    if community
                    else "uninsured balance is large and attention is elevated"
                ),
            ),
        ]


class CEOFake(FakeModule):
    """M42. Picks one menu action and nudges a deposit rate. No Grok call."""

    publishes: ClassVar[tuple[str, ...]] = ("DepositRateChanged", "CEOAction")
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "AttentionTick", "PriceTick", "CapitalChanged")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.attention = 0.0
        self.cet1 = OPENING_CET1
        self._acted = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        for event in new_events:
            if isinstance(event, AttentionTick):
                self.attention = event.level
            elif isinstance(event, CapitalChanged):
                self.cet1 = event.cet1_ratio
        if self._acted or self.attention < 0.25:
            return []
        self._acted = True
        if self.cet1 < OPENING_CET1:
            action = "announce_capital_raise"
        else:
            action = "issue_statement"
        rate = clamp(0.03 * seed_scale(self.seed, "dep-rate"), 0.0, 0.08)
        return [
            DepositRateChanged(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("rate"),
                segment="startup",
                rate=rate,
            ),
            CEOAction(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("ceo"),
                action=action,
                params={"segment": "startup"},
                rationale="attention is elevated; the action set is the management menu",
            ),
        ]


class ShortFake(FakeModule):
    """M44. Publishes a thesis when attention times the uninsured share is high."""

    publishes: ClassVar[tuple[str, ...]] = ("ThesisPublished",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "AttentionTick", "CapitalChanged")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self._sent = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        if self._sent or self.scenario is None:
            return []
        attention = 0.0
        cet1 = OPENING_CET1
        for event in new_events:
            if isinstance(event, AttentionTick):
                attention = event.level
            elif isinstance(event, CapitalChanged):
                cet1 = event.cet1_ratio
        vulnerability = attention * uninsured_share(self.scenario.preset)
        if cet1 < OPENING_CET1:
            vulnerability += 0.05
        if vulnerability < 0.2:
            return []
        self._sent = True
        reach = int(5_000 * seed_scale(self.seed, "thesis-reach"))
        return [
            ThesisPublished(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("thesis"),
                author_id="short-1",
                reach=max(1, reach),
            )
        ]


def l5_modules() -> dict[str, FakeModule]:
    return {
        "M04": MediaFake("M04"),
        "M23": ManagementFake("M23"),
        "M24": CrisisRoomFake("M24"),
        "M25": PersonaFake("M25"),
        "M34": GuardrailFake("M34"),
        "M41": VIPFake("M41"),
        "M42": CEOFake("M42"),
        "M44": ShortFake("M44"),
    }
