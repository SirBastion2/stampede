"""Skeleton smoke test (§10.2 phase 0)."""

from contracts.events import Tick
from engine import ModuleMode, build_engine


def test_fake_engine_wires_all_modules() -> None:
    modules = build_engine(default_mode=ModuleMode.FAKE)
    assert len(modules) == 27
    assert "M00" in modules


def test_tick_contract_example() -> None:
    tick = Tick(t=0.0, seed=42)
    assert tick.event_type == "Tick"


def test_fake_modules_step_without_error() -> None:
    modules = build_engine(default_mode=ModuleMode.FAKE)
    for mod in modules.values():
        assert mod.step(0.0) == []
