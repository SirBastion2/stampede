"""Fake modules that publish contract events when a scenario is attached."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from contracts.schemas import Scenario
from mocks.bus import EventBus
from mocks.support import bank_id_for


class FakeModule:
    """Deterministic stand-in. ``step`` is empty until ``attach`` gets a scenario."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0

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
        del t, new_events
        return []

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

    def cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"

    def log(self) -> list[BaseModel]:
        if self.bus is None:
            return []
        return self.bus.log
