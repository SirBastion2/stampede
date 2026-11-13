"""Work-stream mock generators (§10.2 phase 0)."""

from __future__ import annotations

__all__ = ["stream_modules_for_mode"]


def __getattr__(name: str) -> object:
    if name == "stream_modules_for_mode":
        from mocks.stream_loader import stream_modules_for_mode

        return stream_modules_for_mode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
