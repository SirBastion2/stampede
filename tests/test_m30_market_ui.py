"""M30 market screen shell: snapshot, codec, and fake-engine wiring."""

from __future__ import annotations

from engine.registry import build_engine
from mocks.l5_grok import CrisisRoomFake
from mocks.l7_market_ui import MarketScreenFake
from ui.market.codec import frame_from_json, frame_to_json
from ui.market.demo import demo_scenario
from ui.market.snapshot import build_market_frame, stream_hz


def test_stream_hz_reads_registry() -> None:
    assert stream_hz() == 10.0


def test_market_frame_from_panic_scenario() -> None:
    engine = build_engine(scenario=demo_scenario(1))
    engine.run(6)
    frame = build_market_frame(engine)
    assert frame.module == "M30"
    assert frame.prices
    assert frame.futures
    assert len(frame.futures) == 16
    assert frame.depth
    crisis = engine["M24"]
    assert isinstance(crisis, CrisisRoomFake)
    rejected = [row for row in frame.statements if row.verdict == "rejected"]
    assert rejected
    assert rejected[0].reason


def test_frame_json_round_trip() -> None:
    engine = build_engine(scenario=demo_scenario(1))
    engine.run(6)
    screen = engine["M30"]
    assert isinstance(screen, MarketScreenFake)
    original = build_market_frame(engine)
    restored = frame_from_json(frame_to_json(original))
    assert restored.model_dump() == original.model_dump()
