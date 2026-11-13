"""M28 scheduler module: posts scenario rumors as RumorPosted events (§19)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from agents.guardrails import claim_allowed
from contracts.events import RumorPosted
from contracts.schemas import Scenario
from mocks.bus import EventBus
from scenarios.builder import poster_for_rumor
from scenarios.presets import bank_for_preset


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ScenarioModule:
    """Real M28: publish each scenario rumor once; drop blocklisted claims."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ("RumorPosted",)

    def __init__(self, module_id: str = "M28") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0
        self._posted: set[int] = set()
        self.dropped: list[str] = []

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
        out = self._on_tick(t, new_events)
        self.published.extend(out)
        return out

    def _on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.scenario is None:
            return []
        out: list[BaseModel] = []
        for index, rumor in enumerate(self.scenario.rumors):
            if index in self._posted:
                continue
            self._posted.add(index)
            if not claim_allowed(rumor.claim):
                self.dropped.append(rumor.id)
                continue
            out.append(
                RumorPosted(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self._cause(rumor.id),
                    rumor_id=rumor.id,
                    claim=rumor.claim,
                    source_node=poster_for_rumor(rumor),
                    credibility=_clamp(rumor.credibility, 0.0, 1.0),
                )
            )
        return out

    @property
    def seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return self.scenario.seed

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_for_preset(self.scenario.preset).id

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"
