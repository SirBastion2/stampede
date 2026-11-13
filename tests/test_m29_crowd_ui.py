"""M29 crowd UI shell: layout, frames, feed, fake-engine snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.schemas import CrowdFrame
from engine.registry import build_engine, run_scenario
from mocks.l6_crowd_ui import CrowdFake
from tests.test_fakes import panic
from ui.crowd.feed import CrowdFeed, snapshot_from_engine
from ui.crowd.frames import (
    build_crowd_frame,
    decode_crowd_frame,
    encode_crowd_frame,
    render_point_count,
    state_ribbon,
)
from ui.crowd.layout import build_layout


def test_layout_is_deterministic_for_seed() -> None:
    first = build_layout(7, 128, 4)
    second = build_layout(7, 128, 4)
    assert first == second
    assert len(first.positions_xy) == 128 * 2


def test_state_ribbon_thresholds() -> None:
    assert state_ribbon(90_000, 5_000, 1_000) == "calm"
    assert state_ribbon(82_000, 15_000, 2_000) == "wavering"
    assert state_ribbon(50_000, 20_000, 20_000) == "crisis"


def test_crowd_frame_round_trip_msgpack() -> None:
    engine = build_engine(scenario=panic(3))
    engine.run(4)
    crowd = engine["M29"]
    assert isinstance(crowd, CrowdFake)
    frame = build_crowd_frame(crowd, 4.0)
    assert frame.point_count == render_point_count()
    wire = encode_crowd_frame(frame)
    restored = decode_crowd_frame(wire)
    assert restored.panels == frame.panels
    assert restored.state == frame.state
    assert restored.point_count == frame.point_count


def test_feed_counts_payment_settled_events() -> None:
    engine = build_engine(scenario=panic(1))
    feed = CrowdFeed()
    feed.attach(engine)
    engine.run(6)
    assert feed.payments_settled >= 1
    frame = feed.snapshot(engine["M29"], 5.0)
    assert isinstance(frame, CrowdFrame)
    assert frame.panels.withdrew_count > 0


def test_snapshot_from_engine_after_run() -> None:
    log = run_scenario(panic(2), 5)
    engine = build_engine(scenario=panic(2))
    engine.run(5)
    frame = snapshot_from_engine(engine, t=4.0)
    assert frame.t == 4.0
    assert frame.panels.rumor or any(e.event_type == "RumorPosted" for e in log)


def test_fake_fixture_file_matches_engine_snapshot() -> None:
    engine = build_engine(scenario=panic(1))
    engine.run(3)
    frame = snapshot_from_engine(engine, t=2.0)
    assert frame.point_count == render_point_count()
    root = Path(__file__).resolve().parents[1]
    fixture_path = root / "ui" / "crowd" / "src" / "fixtures" / "fake_frame.json"
    if fixture_path.is_file():
        loaded = CrowdFrame.model_validate(json.loads(fixture_path.read_text(encoding="utf-8")))
        assert loaded.point_count == frame.point_count
        assert loaded.seed == frame.seed
