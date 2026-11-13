"""Scheduler-facing module interface (§19: step(t) per module)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class SimModule(Protocol):
    module_id: str

    def step(self, t: float) -> list[BaseModel]:
        """Advance one tick and return events for the bus to publish, in order.

        The scheduler publishes each list before the next module steps.
        Fakes that have not been given a scenario return an empty list.
        """
        ...
