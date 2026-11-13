"""Serialize M30 frames for WebSocket transport."""

from __future__ import annotations

import json
from typing import Any

from contracts.schemas import MarketScreenFrame


def frame_to_json(frame: MarketScreenFrame) -> str:
    return json.dumps(frame.model_dump(mode="json"), separators=(",", ":"))


def frame_from_json(payload: str) -> MarketScreenFrame:
    data: dict[str, Any] = json.loads(payload)
    return MarketScreenFrame.model_validate(data)


def frame_to_msgpack(frame: MarketScreenFrame) -> bytes:
    import msgpack

    return msgpack.packb(frame.model_dump(mode="json"), use_bin_type=True)


def frame_from_msgpack(payload: bytes) -> MarketScreenFrame:
    import msgpack

    data: dict[str, Any] = msgpack.unpackb(payload, raw=False)
    return MarketScreenFrame.model_validate(data)
