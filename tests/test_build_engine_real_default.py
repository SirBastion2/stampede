"""E1 (SVB-54): build_engine(default_mode=REAL) builds for every caller (ADR-101).

Real where a real module exists (REAL_MODULE_IDS), fake for the rest. Explicitly
requesting a module without a real implementation still raises. CPU, offline.
"""

from __future__ import annotations

import pytest

from engine.modes import ModuleMode
from engine.registry import REAL_MODULE_IDS, build_engine
from tests.test_fakes import panic


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _is_fake(module: object) -> bool:
    return type(module).__name__.endswith("Fake") or type(module).__module__.startswith("mocks.")


def test_default_mode_real_builds_every_real_module() -> None:
    engine = build_engine(default_mode=ModuleMode.REAL, scenario=panic(1))
    still_fake = [mid for mid in REAL_MODULE_IDS if mid in engine.order and _is_fake(engine[mid])]
    assert still_fake == []
    assert _is_fake(engine["M29"]) and _is_fake(engine["M30"])


def test_default_mode_real_runs_a_tick() -> None:
    engine = build_engine(default_mode=ModuleMode.REAL, scenario=panic(1))
    log = engine.run(1)
    assert log


def test_explicit_modes_override_the_real_default() -> None:
    engine = build_engine(
        modes={"M09": ModuleMode.OFF, "M14": ModuleMode.FAKE},
        default_mode=ModuleMode.REAL,
        scenario=panic(1),
    )
    assert "M09" not in engine.order
    assert _is_fake(engine["M14"])
    assert not _is_fake(engine["M06"])


def test_explicit_real_request_without_implementation_still_raises() -> None:
    with pytest.raises(NotImplementedError, match="M29"):
        build_engine(modes={"M29": ModuleMode.REAL})
    with pytest.raises(NotImplementedError, match="M30"):
        build_engine(modes={"M30": ModuleMode.REAL}, default_mode=ModuleMode.REAL)


def test_fake_default_is_unchanged() -> None:
    engine = build_engine(scenario=panic(1))
    assert all(_is_fake(engine[mid]) or mid == "M00" for mid in engine.order if mid in REAL_MODULE_IDS)
