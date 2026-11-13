"""L3 fake: one bank's stock, a widening spread, and a volatility halt."""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import (
    AttentionTick,
    CapitalChanged,
    HeadlinePublished,
    PriceTick,
    ThesisPublished,
    TradingHalt,
)
from mocks.base import FakeModule
from mocks.demo_arc import price_quote, rumor_strength
from mocks.support import (
    ASSUMED_HALT_DRAWDOWN,
    ASSUMED_HALT_MINUTES,
    ASSUMED_HALT_WINDOW,
    OPENING_CET1,
    OPENING_PRICE,
    levers_in,
    seed_scale,
)


class MarketFake(FakeModule):
    """M14 fake. Price follows the pitch curve plus a small attention discount."""

    publishes: ClassVar[tuple[str, ...]] = ("PriceTick", "TradingHalt")
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "RatesChanged",
        "AttentionTick",
        "HeadlinePublished",
        "CapitalChanged",
        "ThesisPublished",
        "PolicyEnacted",
    )

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.price = OPENING_PRICE
        self.attention = 0.0
        self.cet1 = OPENING_CET1
        self.headline_pressure = 0.0
        self.short_interest = 0.0
        self.spread = 0.02
        self.history: list[float] = []
        self.halted = False
        self.depth: list[tuple[float, float]] = []

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        for event in new_events:
            if isinstance(event, AttentionTick):
                self.attention = event.level
            elif isinstance(event, CapitalChanged):
                self.cet1 = event.cet1_ratio
            elif isinstance(event, HeadlinePublished):
                self.headline_pressure = min(1.0, event.reach / 50_000.0)
            elif isinstance(event, ThesisPublished):
                self.short_interest += 0.05 * seed_scale(self.seed, "short")
        from engine.gpu_sim import calibration_uses_scripted_drawdown, gpu_sim_enabled

        # CPU real-engine and gpu-sim both read the M27 row; demo ticks stay scripted.
        in_calibration = gpu_sim_enabled() or (
            os.environ.get("STAMPEDE_CALIBRATION_ENGINE", "fake").strip().lower()
            == "real"
        )
        scripted = calibration_uses_scripted_drawdown() if in_calibration else True
        self.price = price_quote(
            t, rumor_strength(self.log()), self.attention, scripted=scripted
        )
        volume = 1_000.0 + 80_000.0 * max(self.attention, 0.05) * seed_scale(self.seed, "volume")
        self.spread = 0.02 * (1.0 + 6.0 * self.attention + 4.0 * self.short_interest)
        self.history.append(self.price)
        self.depth = _depth(self.price, self.spread, volume)
        out: list[BaseModel] = [
            PriceTick(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("px"),
                price=self.price,
                volume=volume,
            )
        ]
        if not self.halted and _drawdown(self.history) <= -ASSUMED_HALT_DRAWDOWN:
            self.halted = True
            out.append(
                TradingHalt(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause("halt"),
                    reason="volatility",
                    duration=ASSUMED_HALT_MINUTES,
                )
            )
        elif not self.halted and "suspend_trading" in levers_in(self.log()):
            self.halted = True
            out.append(
                TradingHalt(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause("halt-policy"),
                    reason="policy",
                    duration=ASSUMED_HALT_MINUTES,
                )
            )
        return out


def _drawdown(history: list[float]) -> float:
    if len(history) < 2:
        return 0.0
    window = history[-ASSUMED_HALT_WINDOW :]
    start = window[0]
    if start <= 0.0:
        return 0.0
    return history[-1] / start - 1.0


def _depth(price: float, spread: float, volume: float) -> list[tuple[float, float]]:
    half = max(spread, 0.01)
    bid = max(0.5, price - half)
    ask = price + half
    size = max(1.0, volume * 0.1)
    return [(bid, size), (ask, size * 0.8)]


def l3_modules() -> dict[str, FakeModule]:
    return {"M14": MarketFake("M14")}
