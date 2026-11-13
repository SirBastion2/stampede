"""Orb aggregation: block size, majority color, radius cap. Offline, no network."""

from __future__ import annotations

from pathlib import Path

from params.loader import get_parameter
from ui.crowd.orbs import (
    STATE_CALM,
    STATE_WAVERING,
    STATE_WITHDREW,
    aggregate_orb_states,
    block_size,
    majority_state,
    styled_orb_radius,
)

_ROOT = Path(__file__).resolve().parents[1]


def _orb_budget() -> int:
    return int(get_parameter("M29 UI fake-mode render points").value)


def test_block_size_for_one_million_is_a_few_hundred() -> None:
    orb_count = _orb_budget()
    size = block_size(1_000_000, orb_count)
    assert orb_count == 2048
    assert size == 488
    assert 250 <= size <= 1000
    assert size == max(1, int((1_000_000 / orb_count) + 0.5))


def test_majority_color() -> None:
    assert majority_state(500, 10, 10) == STATE_CALM
    assert majority_state(10, 500, 10) == STATE_WAVERING
    assert majority_state(10, 10, 500) == STATE_WITHDREW
    assert majority_state(5, 5, 5) == STATE_WITHDREW
    assert majority_state(5, 5, 0) == STATE_WAVERING
    assert majority_state(0, 0, 0) == STATE_CALM


def test_aggregate_draws_orbs_not_depositors() -> None:
    orb_count = _orb_budget()
    states = aggregate_orb_states(1_000_000, orb_count, 500_000, 220_000, 280_000, 8, 1)
    assert len(states) == orb_count
    assert len(states) != 1_000_000
    size = block_size(1_000_000, orb_count)
    assert states.count(STATE_WITHDREW) == min(orb_count, int(280_000 / size + 0.5))
    origin = states[256:512]
    farthest = states[5 * 256 : 6 * 256]
    assert origin.count(STATE_WITHDREW) > farthest.count(STATE_WITHDREW)


def test_orbs_flip_a_few_when_one_block_withdraws() -> None:
    orb_count = _orb_budget()
    size = block_size(1_000_000, orb_count)
    before = aggregate_orb_states(1_000_000, orb_count, 700_000, 200_000, 100_000, 8, 1)
    after = aggregate_orb_states(
        1_000_000, orb_count, 700_000 - size, 200_000, 100_000 + size, 8, 1
    )
    changed = sum(left != right for left, right in zip(before, after, strict=True))
    assert 1 <= changed <= 4


def test_uninsured_radius_is_larger_and_capped() -> None:
    cap = float(get_parameter("M29 orb radius cap px").value)
    huge = styled_orb_radius(True, 10_000.0)
    assert huge == cap
    insured = styled_orb_radius(False, 0.4)
    uninsured = styled_orb_radius(True, 0.4)
    assert insured < uninsured <= cap
    assert styled_orb_radius(False, 1.0) <= cap


def test_client_style_matches_registry() -> None:
    text = (_ROOT / "ui" / "crowd" / "src" / "orbStyle.ts").read_text(encoding="utf-8")
    expected = {
        "ORB_BASE_RADIUS": get_parameter("M29 orb base radius px").value,
        "ORB_UNINSURED_EXTRA": get_parameter("M29 uninsured orb extra radius px").value,
        "ORB_RADIUS_CAP": get_parameter("M29 orb radius cap px").value,
        "ORBS_PER_TICK": str(int(float(get_parameter("M29 orbs transitioning per tick").value))),
        "MAX_DRAWN_ORBS": get_parameter("M29 UI fake-mode render points").value,
    }
    for name, value in expected.items():
        assert f"export const {name} = {value};" in text
