"""Mock generators per work stream."""

from engine.modes import ModuleMode
from mocks import stream_modules_for_mode


def test_each_stream_exposes_fakes() -> None:
    counts = {
        stream: len(stream_modules_for_mode(stream, None, ModuleMode.FAKE))
        for stream in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")
    }
    assert counts["L1"] == 6
    assert counts["L5"] == 8
    assert sum(counts.values()) == 27


def test_off_mode_drops_modules() -> None:
    mods = stream_modules_for_mode("L3", {"M14": ModuleMode.OFF}, ModuleMode.FAKE)
    assert mods == {}
