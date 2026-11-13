"""In-memory event bus: publish/subscribe, append-only, deterministic.

Wired by ``engine.registry.build_engine``. There is no wall clock and no
reordering. Subscribers see an event only after ``publish`` returns, which the
scheduler does between module steps.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class EventSink(Protocol):
    inbox: list[BaseModel]


class EventBus:
    def __init__(self) -> None:
        self.log: list[BaseModel] = []
        self._subs: dict[str, list[EventSink]] = {}

    def subscribe(self, event_type: str, sink: EventSink) -> None:
        self._subs.setdefault(event_type, []).append(sink)

    def publish(self, event: BaseModel) -> None:
        self.log.append(event)
        kind = getattr(event, "event_type", None)
        if not isinstance(kind, str):
            return
        for sink in self._subs.get(kind, ()):
            sink.inbox.append(event)
