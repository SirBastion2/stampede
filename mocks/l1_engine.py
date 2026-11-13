"""L1 fakes: clock, network, attention, population, depositor decisions, futures."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from contracts.events import (
    AttentionTick,
    BankClosed,
    FutureCompleted,
    HeadlinePublished,
    LiquidityDrawn,
    PriceTick,
    RumorMutated,
    RumorPosted,
    StatementIssued,
    ThesisPublished,
    Tick,
    TradingHalt,
    VIPDecision,
    WithdrawalRequested,
)
from mocks.base import FakeModule
from mocks.demo_arc import buffer_dollars, rumor_strength, stress_dollars
from mocks.params_access import get_param
from mocks.support import (
    DEPOSITORS_TIER0,
    FUTURE_COUNT,
    SEGMENT_WEIGHTS,
    UNINSURED_WITHDRAW_SPLIT,
    attention_window_ticks,
    calendar_at,
    clamp,
    deposits_base,
    is_community,
    seed_scale,
    uninsured_share,
)


class ClockFake(FakeModule):
    """M00. One step is one simulated minute starting Monday 09:00."""

    publishes: ClassVar[tuple[str, ...]] = ("Tick",)
    consumes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.business_hours = True
        self.weekend = False
        self.facility_open = True

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        flags = calendar_at(t)
        self.business_hours = flags.business_hours
        self.weekend = flags.weekend
        self.facility_open = flags.facility_open
        return [Tick(t=t, seed=self.seed, bank_id=None, cause_id=None)]


class NetworkFake(FakeModule):
    """M02. Tiny deterministic stand-in for the three graph layers. Publishes nothing."""

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.built = False
        self.hub_weight = 0.0
        self.group_blocks = 0
        self.advisor_links = 0

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del t, new_events
        if self.built:
            return []
        self.built = True
        self.hub_weight = 0.15 + 0.2 * seed_scale(self.seed, "hub")
        self.group_blocks = 8 + int(seed_scale(self.seed, "blocks") * 6)
        self.advisor_links = 40
        return []


class AttentionFake(FakeModule):
    """M03. Attention rises with rumors and coverage and decays over the memory window."""

    publishes: ClassVar[tuple[str, ...]] = ("AttentionTick", "RumorMutated")
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "RumorPosted",
        "HeadlinePublished",
        "LiquidityDrawn",
        "TradingHalt",
        "StatementIssued",
        "BankClosed",
        "VIPDecision",
        "ThesisPublished",
    )

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.level = 0.0
        self.base = 0.0
        self.history: list[float] = []
        self.mutations: list[dict[str, float | str]] = []
        self._mutated: set[str] = set()

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        if not self.history and self.base == 0.0:
            self.base = 0.04 * seed_scale(self.seed, "attn-base")
            self.level = self.base
        window = attention_window_ticks()
        decay = max(0.0, 1.0 - 1.0 / float(window))
        reference = self.history[-window] if len(self.history) >= window else self.base
        level = self.level * decay
        mutated: list[RumorMutated] = []
        for event in new_events:
            if isinstance(event, RumorPosted):
                gain = (
                    clamp(event.credibility, 0.0, 1.0)
                    * 0.55
                    * seed_scale(self.seed, "attn-rumor")
                )
                level += gain
                if event.rumor_id not in self._mutated:
                    self._mutated.add(event.rumor_id)
                    exaggeration = 0.25 * seed_scale(self.seed, event.rumor_id)
                    self.mutations.append(
                        {
                            "rumor_id": event.rumor_id,
                            "claim": event.claim,
                            "exaggeration_score": exaggeration,
                        }
                    )
                    mutated.append(
                        RumorMutated(
                            t=t,
                            seed=self.seed,
                            bank_id=self.bank_id,
                            cause_id=self.cause(f"mut-{event.rumor_id}"),
                            rumor_id=event.rumor_id,
                            claim=event.claim,
                            exaggeration_score=exaggeration,
                        )
                    )
            elif isinstance(event, HeadlinePublished):
                level += min(0.12, event.reach / 200_000.0)
            elif isinstance(event, LiquidityDrawn):
                level += 0.04
            elif isinstance(event, TradingHalt):
                level += 0.05
            elif isinstance(event, ThesisPublished):
                level += 0.08 * seed_scale(self.seed, "thesis-attn")
            elif isinstance(event, StatementIssued):
                level -= 0.06
            elif isinstance(event, BankClosed):
                level += 0.15
            elif isinstance(event, VIPDecision) and event.action == "withdraw":
                level += 0.03
        level = clamp(level, 0.0, 1.0)
        delta = level - reference
        self.level = level
        self.history.append(level)
        return [
            *mutated,
            AttentionTick(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("attn"),
                level=level,
                delta_4h=delta,
            ),
        ]


class PopulationFake(FakeModule):
    """M05. Summary population. Insured amounts use the registry cap. Publishes nothing."""

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.ready = False
        self.depositor_count = 0
        self.deposits = 0.0
        self.insurance_limit = 0.0
        self.insured_share = 0.0
        self.uninsured_share = 0.0
        self.top10_balance = 0.0
        self.example_top_insured = 0.0
        self.example_top_uninsured = 0.0
        self.segments: dict[str, float] = {}

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del t, new_events
        if self.ready or self.scenario is None:
            return []
        self.ready = True
        limit = get_param("Insurance limit")
        top10 = get_param("SVB-like: top-10 depositor balance")
        share = uninsured_share(self.scenario.preset)
        self.depositor_count = DEPOSITORS_TIER0
        self.deposits = deposits_base()
        self.insurance_limit = limit
        self.uninsured_share = share
        self.insured_share = clamp(1.0 - share, 0.0, 1.0)
        self.top10_balance = top10
        each = top10 / 10.0
        self.example_top_insured = min(each, limit)
        self.example_top_uninsured = max(0.0, each - limit)
        weights = dict(SEGMENT_WEIGHTS)
        if "crypto" not in self.scenario.preset.lower():
            crypto = weights.pop("crypto", 0.0)
            weights["startup"] = weights.get("startup", 0.0) + crypto
        self.segments = weights
        return []


class DecisionFake(FakeModule):
    """M06 fake. Headcount pacing lives in the crowd gauge; dollars follow the cash arc."""

    publishes: ClassVar[tuple[str, ...]] = ("WithdrawalRequested",)
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "AttentionTick",
        "PriceTick",
        "StatementIssued",
        "PolicyEnacted",
        "DepositRateChanged",
        "DirectiveIssued",
        "VIPDecision",
    )

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.attention = 0.0
        self.withdrawn = 0.0
        self.price_return = 0.0
        self._open_price: float | None = None

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        if self.scenario is None:
            return []
        for event in new_events:
            if isinstance(event, AttentionTick):
                self.attention = event.level
            elif isinstance(event, PriceTick):
                if self._open_price is None:
                    self._open_price = event.price
                if self._open_price > 0.0:
                    self.price_return = event.price / self._open_price - 1.0
        if is_community(self.scenario.preset):
            return []
        if any(isinstance(event, BankClosed) for event in self.log()):
            return []
        log = self.log()
        strength = rumor_strength(log)
        if strength <= 0.0:
            return []
        # A 1% seed tilt keeps two seeds apart without moving the gauge off the arc.
        jitter = 1.0 + (seed_scale(self.seed, "withdraw") - 1.0) * 0.01
        target = buffer_dollars(t, strength, jitter=jitter)
        target += stress_dollars(t, strength, seed_factor=seed_scale(self.seed, "withdraw-hole"))
        raw = target - self.withdrawn
        if raw <= 1.0:
            return []
        self.withdrawn += raw
        channel = "wire"
        out: list[BaseModel] = []
        for segment, weight in UNINSURED_WITHDRAW_SPLIT:
            amount = raw * weight
            out.append(
                WithdrawalRequested(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause(segment),
                    depositor_id=f"dep-{segment}",
                    amount=amount,
                    channel=channel,
                )
            )
        return out


class FuturesFake(FakeModule):
    """M26. One batch of futures once attention and a price exist. No extra Grok calls."""

    publishes: ClassVar[tuple[str, ...]] = ("FutureCompleted",)

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.emitted = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.emitted:
            return []
        attention = 0.0
        price = 0.0
        withdrawn = 0.0
        seen_price = False
        for event in self.log():
            if isinstance(event, AttentionTick):
                attention = event.level
            elif isinstance(event, PriceTick):
                price = event.price
                seen_price = True
            elif isinstance(event, WithdrawalRequested):
                withdrawn += event.amount
        if not seen_price:
            return []
        self.emitted = True
        deposits = deposits_base()
        out: list[BaseModel] = []
        for index in range(FUTURE_COUNT):
            tilt = seed_scale(self.seed, f"future-{index}")
            failure = clamp(attention * tilt, 0.0, 1.0)
            kept = max(0.0, deposits - withdrawn * tilt)
            low = max(0.5, price * (1.0 - 0.15 * tilt))
            high = price * (1.0 + 0.05 * (2.0 - tilt))
            out.append(
                FutureCompleted(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause(f"f{index}"),
                    future_id=f"f{index:02d}",
                    metrics={
                        "failure_probability": failure,
                        "deposits_kept": kept,
                        "price_low": low,
                        "price_high": high,
                        "t_fail": None,
                    },
                )
            )
        return out


def l1_modules() -> dict[str, FakeModule]:
    return {
        "M00": ClockFake("M00"),
        "M02": NetworkFake("M02"),
        "M03": AttentionFake("M03"),
        "M05": PopulationFake("M05"),
        "M06": DecisionFake("M06"),
        "M26": FuturesFake("M26"),
    }
