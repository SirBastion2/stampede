"""L7 fake: market and crisis-room snapshot. Publishes nothing."""

from __future__ import annotations

from pydantic import BaseModel

from contracts.events import FutureCompleted, PriceTick, StatementIssued, TradingHalt
from mocks.base import FakeModule
from mocks.l3_markets import _depth


class MarketScreenFake(FakeModule):
    """M30. Series for the price chart, halt banner, statements, and futures fan."""

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.prices: list[tuple[float, float]] = []
        self.volumes: list[float] = []
        self.depth: list[tuple[float, float]] = []
        self.markers: list[dict[str, str | float]] = []
        self.halt: dict[str, float | str] | None = None
        self.statements: list[dict[str, str | list[str]]] = []
        self.futures: list[dict[str, float | str | None]] = []

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        """Series from the whole log; SVB-69: extended with the new events instead of rebuilt each tick."""
        del t, new_events
        log = self.log()
        n = getattr(self, "_n", 0)
        if log is not getattr(self, "_seen_log", None) or n > len(log) or (n and log[n - 1] is not self._last):
            self.prices, self.volumes, self.markers, self.statements, self.futures = [], [], [], [], []
            self.halt = None
            self._last_price, self._last_volume, self._seen_log, n = 0.0, 0.0, log, 0
        self.depth = []
        last_price = self._last_price
        last_volume = self._last_volume
        for event in log[n:]:
            if isinstance(event, PriceTick):
                self.prices.append((event.t, event.price))
                self.volumes.append(event.volume)
                last_price = event.price
                last_volume = event.volume
            elif isinstance(event, TradingHalt):
                self.halt = {"reason": event.reason, "duration": event.duration, "t": event.t}
                self.markers.append(
                    {"t": event.t, "kind": "halt", "label": f"halt: {event.reason}"}
                )
            elif isinstance(event, StatementIssued):
                self.statements.append(
                    {"stmt_id": event.stmt_id, "channel": event.channel, "segments": event.segments}
                )
                self.markers.append(
                    {"t": event.t, "kind": "statement", "label": event.stmt_id}
                )
            elif isinstance(event, FutureCompleted):
                failure = event.metrics.get("failure_probability", 0.0)
                probability = float(failure) if isinstance(failure, (int, float)) else 0.0
                low = event.metrics.get("price_low")
                high = event.metrics.get("price_high")
                self.futures.append(
                    {
                        "future_id": event.future_id,
                        "failure_probability": probability,
                        "price_low": float(low) if isinstance(low, (int, float)) else None,
                        "price_high": float(high) if isinstance(high, (int, float)) else None,
                    }
                )
        self._n = len(log)
        self._last = log[-1] if log else None
        self._last_price, self._last_volume = last_price, last_volume
        if last_price > 0.0:
            self.depth = _depth(last_price, 0.02, last_volume)
        return []


def l7_modules() -> dict[str, FakeModule]:
    return {"M30": MarketScreenFake("M30")}
