"""CrowdFrame builders and msgpack wire codec (§6.2)."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Literal

import msgpack

from contracts.schemas import CrowdFrame, CrowdPanels
from params.loader import RegistryError, get_parameter
from ui.crowd.layout import CrowdLayout, build_layout
from ui.crowd.orbs import aggregate_orb_states

if TYPE_CHECKING:
    from mocks.l6_crowd_ui import CrowdFake

_PARAM_RENDER_POINTS = "M29 UI fake-mode render points"
_PARAM_COMMUNITIES = "M29 layout community count"


def _parse_int_param(name: str, default: int) -> int:
    try:
        row = get_parameter(name)
    except RegistryError:
        return default
    digits = "".join(ch for ch in row.value if ch.isdigit())
    return int(digits) if digits else default


def render_point_count() -> int:
    return max(64, _parse_int_param(_PARAM_RENDER_POINTS, 2048))


def layout_community_count() -> int:
    return max(1, _parse_int_param(_PARAM_COMMUNITIES, 8))


def state_ribbon(
    calm: int, wavering: int, withdrew: int
) -> Literal["calm", "wavering", "crisis"]:
    total = max(1, calm + wavering + withdrew)
    withdrew_share = withdrew / total
    wavering_share = wavering / total
    if withdrew_share >= 0.12 or (withdrew_share >= 0.05 and wavering_share >= 0.2):
        return "crisis"
    if wavering_share >= 0.08:
        return "wavering"
    return "calm"


def build_crowd_frame(
    crowd: CrowdFake,
    t: float,
    layout: CrowdLayout | None = None,
) -> CrowdFrame:
    budget = render_point_count()
    layout = layout or build_layout(crowd.seed, budget, layout_community_count())
    if layout.point_count > budget:
        raise ValueError("crowd layout exceeds the registry orb budget")
    total = max(0, crowd.calm) + max(0, crowd.wavering) + max(0, crowd.withdrew)
    states = aggregate_orb_states(
        total if total else 1,
        layout.point_count,
        crowd.calm,
        crowd.wavering,
        crowd.withdrew,
        layout.community_count,
        crowd.seed,
    )
    panels = CrowdPanels(
        cash_gauge=crowd.cash_gauge,
        time_to_failure_ticks=crowd.time_to_failure_ticks,
        tipping_point=crowd.tipping_point,
        rumor=crowd.rumor,
        state_ribbon=state_ribbon(crowd.calm, crowd.wavering, crowd.withdrew),
        calm_count=crowd.calm,
        wavering_count=crowd.wavering,
        withdrew_count=crowd.withdrew,
    )
    return CrowdFrame(
        t=t,
        seed=crowd.seed,
        bank_id=crowd.bank_id,
        point_count=layout.point_count,
        positions_xy=list(layout.positions_xy),
        state=states,
        balance_norm=list(layout.balance_norm),
        uninsured_mask=list(layout.uninsured_mask),
        panels=panels,
    )


def encode_crowd_frame(frame: CrowdFrame) -> bytes:
    xy_blob = struct.pack(f"{len(frame.positions_xy)}f", *frame.positions_xy)
    state_blob = bytes(frame.state)
    balance_blob = struct.pack(f"{len(frame.balance_norm)}f", *frame.balance_norm)
    uninsured_blob = bytes(1 if flag else 0 for flag in frame.uninsured_mask)
    payload = {
        "v": 1,
        "t": frame.t,
        "seed": frame.seed,
        "bank_id": frame.bank_id,
        "point_count": frame.point_count,
        "xy": xy_blob,
        "state": state_blob,
        "balance": balance_blob,
        "uninsured": uninsured_blob,
        "panels": frame.panels.model_dump(),
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_crowd_frame(data: bytes) -> CrowdFrame:
    raw = msgpack.unpackb(data, raw=False)
    point_count = int(raw["point_count"])
    xy_blob = raw["xy"]
    state_blob = raw["state"]
    balance_blob = raw["balance"]
    uninsured_blob = raw["uninsured"]
    xy_count = point_count * 2
    positions = struct.unpack(f"{xy_count}f", xy_blob)
    balance = struct.unpack(f"{point_count}f", balance_blob)
    state = list(state_blob[:point_count])
    uninsured = [bool(byte) for byte in uninsured_blob[:point_count]]
    panels = CrowdPanels.model_validate(raw["panels"])
    return CrowdFrame(
        t=float(raw["t"]),
        seed=int(raw["seed"]),
        bank_id=raw.get("bank_id"),
        point_count=point_count,
        positions_xy=list(positions),
        state=state,
        balance_norm=list(balance),
        uninsured_mask=uninsured,
        panels=panels,
    )
