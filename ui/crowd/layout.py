"""CPU layout for M29: community clusters, computed once per seed (no GPU)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mocks.support import unit_interval
from ui.crowd.orbs import uninsured_orb_fraction

# Loose scatter matching the monitor comp (not a regular ring). Index order is
# the infection ring: neighbors in this list sit next to each other on screen.
_CENTERS: tuple[tuple[float, float], ...] = (
    (-3.55, -1.55),
    (-3.15, 1.35),
    (-1.15, 0.15),
    (0.15, 1.7),
    (1.45, -0.15),
    (3.35, -1.45),
    (3.45, 1.15),
    (1.15, -1.85),
)


@dataclass(frozen=True, slots=True)
class CrowdLayout:
    """Float32-ready xy pairs, one row per rendered orb (not per depositor)."""

    point_count: int
    community_count: int
    positions_xy: tuple[float, ...]
    balance_norm: tuple[float, ...]
    uninsured_mask: tuple[bool, ...]


def build_layout(seed: int, point_count: int, community_count: int) -> CrowdLayout:
    if point_count < 1:
        raise ValueError("point_count must be >= 1")
    if community_count < 1:
        raise ValueError("community_count must be >= 1")
    communities = min(community_count, point_count)
    xy: list[float] = []
    balance: list[float] = []
    uninsured: list[bool] = []
    per_community = point_count // communities
    remainder = point_count % communities
    insured_cut = 1.0 - uninsured_orb_fraction()
    for community in range(communities):
        count = per_community + (1 if community < remainder else 0)
        if community < len(_CENTERS):
            base_x, base_y = _CENTERS[community]
        else:
            angle = (2.0 * math.pi * community) / communities
            base_x = math.cos(angle) * 3.2
            base_y = math.sin(angle) * 2.0
        jitter_x = (unit_interval(seed, f"cx-{community}") - 0.5) * 0.28
        jitter_y = (unit_interval(seed, f"cy-{community}") - 0.5) * 0.28
        center_x = base_x + jitter_x
        center_y = base_y + jitter_y
        radius_x = 0.62 + unit_interval(seed, f"rx-{community}") * 0.48
        radius_y = 0.42 + unit_interval(seed, f"ry-{community}") * 0.36
        rotation = (unit_interval(seed, f"rot-{community}") - 0.5) * 1.2
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        for local in range(count):
            salt = f"layout-{community}-{local}"
            theta = unit_interval(seed, f"{salt}-a") * math.tau
            radial = math.sqrt(unit_interval(seed, salt))
            radial *= 0.72 + 0.5 * unit_interval(seed, f"{salt}-f")
            dx = math.cos(theta) * radial * radius_x
            dy = math.sin(theta) * radial * radius_y
            xy.extend(
                (
                    center_x + dx * cos_r - dy * sin_r,
                    center_y + dx * sin_r + dy * cos_r,
                )
            )
            balance.append(0.15 + unit_interval(seed, f"{salt}-bal") * 0.85)
            uninsured.append(unit_interval(seed, f"{salt}-ins") >= insured_cut)
    return CrowdLayout(
        point_count=point_count,
        community_count=communities,
        positions_xy=tuple(xy),
        balance_norm=tuple(balance),
        uninsured_mask=tuple(uninsured),
    )
