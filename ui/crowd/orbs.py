"""Aggregate depositors into fixed-size orbs for the M29 crowd screen.

One orb is a block of N people, N = round(total / orb count). A 1M crowd at the
registry render budget (2048) is a few hundred people per orb. The wire frame
never carries one item per depositor.
"""

from __future__ import annotations

import math

from params.loader import RegistryError, get_parameter

STATE_CALM = 0
STATE_WAVERING = 1
STATE_WITHDREW = 2

_PARAM_BASE_RADIUS = "M29 orb base radius px"
_PARAM_UNINSURED_EXTRA = "M29 uninsured orb extra radius px"
_PARAM_RADIUS_CAP = "M29 orb radius cap px"
_PARAM_PER_TICK = "M29 orbs transitioning per tick"
_PARAM_UNINSURED_FRACTION = "M29 uninsured orb fraction"
_PARAM_DEMO_POPULATION = "M29 demo crowd population"
_PARAM_DEMO_WITHDREW = "M29 demo withdrew share"
_PARAM_DEMO_WAVERING = "M29 demo wavering share"
_PARAM_DEMO_CASH = "M29 demo cash gauge"
_PARAM_DEMO_TTF = "M29 demo time to failure ticks"
_PARAM_DEMO_TIP = "M29 demo tipping point"


def _parse_float(name: str, default: float) -> float:
    try:
        row = get_parameter(name)
    except RegistryError:
        return default
    token = ""
    for ch in row.value:
        if ch.isdigit() or ch == ".":
            token += ch
        elif token:
            break
    try:
        return float(token) if token else default
    except ValueError:
        return default


def _round_half_up(value: float) -> int:
    """Match JavaScript ``Math.round`` for non-negative values (half away from 0)."""
    return int(math.floor(value + 0.5))


def block_size(total_depositors: int, orb_count: int) -> int:
    """People represented by one orb. ``N = round(total / orb count)``, at least 1."""
    if total_depositors < 1:
        raise ValueError("total_depositors must be >= 1")
    if orb_count < 1:
        raise ValueError("orb_count must be >= 1")
    return max(1, _round_half_up(total_depositors / orb_count))


def majority_state(calm: int, wavering: int, withdrew: int) -> int:
    """Mode of one block. Ties break toward withdrew, then wavering.

    An empty block (no people) stays calm so unused orb slots do not paint red.
    """
    if calm <= 0 and wavering <= 0 and withdrew <= 0:
        return STATE_CALM
    best_count = -1
    best_state = STATE_CALM
    for count, state in (
        (calm, STATE_CALM),
        (wavering, STATE_WAVERING),
        (withdrew, STATE_WITHDREW),
    ):
        if count > best_count or (count == best_count and state > best_state):
            best_count = count
            best_state = state
    return best_state


def orb_radius(
    uninsured: bool,
    balance_norm: float,
    *,
    base: float,
    extra: float,
    cap: float,
) -> float:
    """Pixel radius. Uninsured orbs are larger. Nothing exceeds ``cap``."""
    bal = min(1.0, max(0.0, balance_norm))
    radius = base + bal * (base * 0.65)
    if uninsured:
        radius += extra
    return min(cap, radius)


def styled_orb_radius(uninsured: bool, balance_norm: float) -> float:
    """``orb_radius`` using the M29 rows in ``params/registry.csv``."""
    return orb_radius(
        uninsured,
        balance_norm,
        base=_parse_float(_PARAM_BASE_RADIUS, 2.1),
        extra=_parse_float(_PARAM_UNINSURED_EXTRA, 1.9),
        cap=_parse_float(_PARAM_RADIUS_CAP, 5.2),
    )


def orbs_per_tick() -> int:
    return max(1, int(_parse_float(_PARAM_PER_TICK, 5)))


def uninsured_orb_fraction() -> float:
    return min(1.0, max(0.0, _parse_float(_PARAM_UNINSURED_FRACTION, 0.08)))


def demo_crowd_counts() -> tuple[int, int, int]:
    """Calm, wavering, withdrew for the bundled fake frame (pitch population)."""
    total = max(1, int(_parse_float(_PARAM_DEMO_POPULATION, 1_000_000)))
    withdrew = _round_half_up(total * _parse_float(_PARAM_DEMO_WITHDREW, 0.28))
    wavering = _round_half_up(total * _parse_float(_PARAM_DEMO_WAVERING, 0.22))
    withdrew = min(total, max(0, withdrew))
    wavering = min(total - withdrew, max(0, wavering))
    calm = total - withdrew - wavering
    return calm, wavering, withdrew


def demo_gauges() -> tuple[float, float, float]:
    """Cash gauge, time-to-failure ticks, tipping point for the fake frame."""
    return (
        _parse_float(_PARAM_DEMO_CASH, 0.62),
        _parse_float(_PARAM_DEMO_TTF, 116),
        _parse_float(_PARAM_DEMO_TIP, 1),
    )


def _community_ranges(orb_count: int, community_count: int) -> list[tuple[int, int]]:
    communities = max(1, min(community_count, orb_count))
    per = orb_count // communities
    remainder = orb_count % communities
    ranges: list[tuple[int, int]] = []
    start = 0
    for community in range(communities):
        count = per + (1 if community < remainder else 0)
        ranges.append((start, count))
        start += count
    return ranges


def _fill_order(orb_count: int, community_count: int, seed: int) -> list[int]:
    """Orb indexes, hottest community first, sprinkled inside each cluster."""
    ranges = _community_ranges(orb_count, community_count)
    communities = len(ranges)
    origin = seed % communities

    def dist(community: int) -> int:
        raw = abs(community - origin)
        return min(raw, communities - raw)

    scattered: list[list[int]] = []
    for start, count in ranges:
        step = max(1, int(math.sqrt(count)))
        order: list[int] = []
        for phase in range(step):
            index = phase
            while index < count:
                order.append(start + index)
                index += step
        scattered.append(order)

    hottest = sorted(range(communities), key=dist)
    cycle: list[int] = []
    for community in hottest:
        cycle.extend([community] * max(1, 4 - dist(community)))

    cursors = [0] * communities
    picked: list[int] = []
    while len(picked) < orb_count:
        progressed = False
        for community in cycle:
            cursor = cursors[community]
            if cursor >= len(scattered[community]):
                continue
            picked.append(scattered[community][cursor])
            cursors[community] = cursor + 1
            progressed = True
            if len(picked) >= orb_count:
                break
        if not progressed:
            break
    if len(picked) < orb_count:
        seen = set(picked)
        for index in range(orb_count):
            if index not in seen:
                picked.append(index)
    return picked[:orb_count]


def aggregate_orb_states(
    total_depositors: int,
    orb_count: int,
    calm: int,
    wavering: int,
    withdrew: int,
    community_count: int,
    seed: int,
) -> list[int]:
    """One majority color per orb. Length is the orb budget, never the population.

    Each orb stands for ``block_size`` people of a single state, so the color is
    that block's majority. Withdrew orbs are sprinkled toward the seed's origin
    community; raising the withdrew count by one block flips only a few orbs.
    """
    if orb_count < 1:
        raise ValueError("orb_count must be >= 1")
    counted = max(0, calm) + max(0, wavering) + max(0, withdrew)
    people = counted if counted > 0 else max(1, total_depositors)
    # Never draw one orb per depositor when the crowd is larger than the budget.
    drawn = min(orb_count, people)
    size = block_size(people, drawn)
    red = max(0, min(drawn, _round_half_up(max(0, withdrew) / size)))
    gray = max(0, min(drawn - red, _round_half_up(max(0, wavering) / size)))
    order = _fill_order(drawn, max(1, community_count), seed)
    states = [STATE_CALM] * drawn
    for index, orb_index in enumerate(order):
        if index < red:
            slot = STATE_WITHDREW
        elif index < red + gray:
            slot = STATE_WAVERING
        else:
            slot = STATE_CALM
        # Pure block of ``size`` people: majority is the slot itself.
        block = [0, 0, 0]
        block[slot] = size
        states[orb_index] = majority_state(block[0], block[1], block[2])
    if drawn < orb_count:
        states.extend([STATE_CALM] * (orb_count - drawn))
    return states[:orb_count]
