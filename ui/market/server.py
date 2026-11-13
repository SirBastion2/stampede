"""Backward-compatible entry for M30 WebSocket tests (delegates to ``ui.server``)."""

from __future__ import annotations

from ui.live.config import server_port
from ui.live.session import step_engine
from ui.server import app, main

__all__ = ["app", "main", "step_engine"]

if __name__ == "__main__":
    main()
