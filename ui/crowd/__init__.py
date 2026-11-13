"""M29 crowd screen — layout, frames, and feed (§5, stream L6)."""

from ui.crowd.feed import CrowdFeed, snapshot_from_engine
from ui.crowd.frames import (
    build_crowd_frame,
    decode_crowd_frame,
    encode_crowd_frame,
    state_ribbon,
)
from ui.crowd.layout import build_layout

__all__ = [
    "CrowdFeed",
    "build_crowd_frame",
    "build_layout",
    "decode_crowd_frame",
    "encode_crowd_frame",
    "snapshot_from_engine",
    "state_ribbon",
]
