"""Bridge M29 frames to the fake engine and event bus (Tick, PaymentSettled)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contracts.events import PaymentSettled, Tick
from contracts.schemas import CrowdFrame
from mocks.bus import EventBus
from mocks.l6_crowd_ui import CrowdFake
from ui.crowd.frames import (
    build_crowd_frame,
    layout_community_count,
    render_point_count,
)
from ui.crowd.layout import CrowdLayout, build_layout

if TYPE_CHECKING:
    from engine.registry import Engine


class CrowdFeed:
    """Builds CrowdFrame snapshots from the M29 fake module and bus events."""

    def __init__(self) -> None:
        self._layout: CrowdLayout | None = None
        self._bus: EventBus | None = None

    def attach(self, engine: Engine) -> None:
        crowd = engine.get("M29")
        if not isinstance(crowd, CrowdFake):
            raise TypeError("engine has no M29 CrowdFake module")
        seed = crowd.seed
        self._layout = build_layout(seed, render_point_count(), layout_community_count())
        self._bus = engine.bus

    @property
    def payments_settled(self) -> int:
        if self._bus is None:
            return 0
        return sum(1 for event in self._bus.log if isinstance(event, PaymentSettled))

    def snapshot(self, crowd: CrowdFake, t: float | None = None) -> CrowdFrame:
        if self._layout is None:
            self._layout = build_layout(
                crowd.seed, render_point_count(), layout_community_count()
            )
        if t is not None:
            tick = t
        elif self._bus is not None:
            ticks = [event for event in self._bus.log if isinstance(event, Tick)]
            tick = ticks[-1].t if ticks else 0.0
        else:
            tick = 0.0
        return build_crowd_frame(crowd, tick, self._layout)


def snapshot_from_engine(engine: Engine, t: float | None = None) -> CrowdFrame:
    crowd = engine["M29"]
    if not isinstance(crowd, CrowdFake):
        raise TypeError("engine M29 is not CrowdFake")
    layout = build_layout(crowd.seed, render_point_count(), layout_community_count())
    tick = t if t is not None else float(max(0, len(engine.bus.log) - 1))
    return build_crowd_frame(crowd, tick, layout)
