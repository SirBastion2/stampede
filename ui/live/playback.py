"""P-A2: buffered playback. The engine steps ahead into a frame buffer; the screens play from it.

Playback starts once the buffer holds the head start (``M29 live playback head start seconds`` of
1x playback) and never passes the newest minute the engine has computed. If the engine is slower
than 1x, playback slows to the engine's pace and the screens hold the newest frame; they never
jump ahead or fall out of step. The producer never runs more than ``max_lead_ticks`` ahead, so a
post made mid-run takes effect at most that many simulated minutes after the minute on screen.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    tick: float
    crowd: Any
    market: Any
    info: Any = None  # UI-1: the screen's numbers at this minute (LiveSession._ui_info)


class PlaybackBuffer:
    """Thread-safe frame queue plus the playback position (simulated minutes)."""

    def __init__(self, head_ticks: float, max_lead_ticks: float) -> None:
        if head_ticks < 0 or max_lead_ticks < head_ticks:
            raise ValueError("need 0 <= head_ticks <= max_lead_ticks")
        self.head_ticks = float(head_ticks)
        self.max_lead_ticks = float(max_lead_ticks)
        self._lock = threading.Lock()
        self._frames: deque[BufferedFrame] = deque()
        self.position = 0.0
        self.playing = False

    def clear(self, position: float = 0.0) -> None:
        with self._lock:
            self._frames.clear()
            self.position = float(position)
            self.playing = False

    def push(self, frame: BufferedFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def newest_tick(self) -> float | None:
        with self._lock:
            return self._frames[-1].tick if self._frames else None

    def lead(self, engine_tick: float) -> float:
        return float(engine_tick) - self.position

    def advance(self, minutes: float, *, finished: bool = False) -> int:
        """Move playback ``minutes`` forward, clamped to the newest frame. Returns whole minutes moved."""
        with self._lock:
            if not self._frames:
                return 0
            newest = self._frames[-1].tick
            if not self.playing:
                if newest - self.position >= self.head_ticks or finished:
                    self.playing = True
                else:
                    return 0
            new = min(self.position + max(0.0, float(minutes)), newest)
            moved = int(new) - int(self.position)
            self.position = new
            while len(self._frames) > 1 and self._frames[1].tick <= self.position:
                self._frames.popleft()
            return moved

    def jump_to_newest(self) -> None:
        with self._lock:
            if self._frames:
                self.position = self._frames[-1].tick
                while len(self._frames) > 1:
                    self._frames.popleft()
                self.playing = True

    def current(self) -> BufferedFrame | None:
        """The newest frame at or before the playback position (the oldest frame if none is)."""
        with self._lock:
            return self._frames[0] if self._frames else None
