"""SVB-43: mutated-rumor exaggeration feeds repost exposure (ADR-095). Flag off by default."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import engine.agents.attention as attention
from contracts.events import RumorMutated, RumorPosted
from contracts.schemas import Scenario
from engine.agents.attention import AttentionEngine, load_rumor_amplifiers
from mocks.bus import EventBus
from params.loader import RegistryError, get_parameter

ROWS = (
    "M03 exaggeration feedback enabled",
    "M03 exaggeration feedback gain",
    "M03 exaggeration feedback cap",
)


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def _primed(seed: int = 11, *, sample: float = 1.0, **flags: object) -> AttentionEngine:
    """Rumor posted at t=0; then every node holds it at attention 1 (all repost)."""
    engine = AttentionEngine("M03")
    engine.attach(EventBus(), Scenario(preset="tech-concentrated", seed=seed))
    engine.params = dataclasses.replace(engine.params, sample_fraction=sample)
    if flags:
        engine.amplifiers = dataclasses.replace(engine.amplifiers, **flags)
    engine.inbox.append(
        RumorPosted(
            t=0.0,
            seed=seed,
            bank_id="bank-fvb",
            cause_id="t",
            rumor_id="r1",
            claim="Uninsured cash is moving at Fictional Valley Bank",
            source_node="hub-public-1",
            credibility=1.0,
        )
    )
    engine.step(0.0)
    engine._rumors[0].holders[:] = True
    engine.attention[:] = 1.0
    return engine


def _added(engine: AttentionEngine) -> tuple[np.ndarray, list[RumorMutated]]:
    """Attention added by one tick's reposts (decayed prior attention removed)."""
    events = engine.step(1.0)
    added = engine.attention - engine._decay * 1.0
    return added, [e for e in events if isinstance(e, RumorMutated)]


def test_rows_are_assumed_and_flag_defaults_off() -> None:
    for name in ROWS:
        row = get_parameter(name)
        assert row.status == "ASSUMED" and row.module == "M03"
    assert get_parameter(ROWS[0]).value == "false"
    amp = load_rumor_amplifiers()
    assert amp.exaggeration_enabled is False
    assert amp.exaggeration_factor(1.0) == 1.0


@pytest.mark.parametrize(("name", "value"), [(ROWS[1], "-1"), (ROWS[2], "0.9"), (ROWS[1], "inf")])
def test_bad_gain_or_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    real = attention.list_by_module

    def broken(module: str, path=None):
        return [
            dataclasses.replace(row, value=value) if row.parameter == name else row
            for row in real(module)
        ]

    monkeypatch.setattr(attention, "list_by_module", broken)
    with pytest.raises(RegistryError):
        load_rumor_amplifiers()


def test_factor_formula_and_cap() -> None:
    amp = dataclasses.replace(
        load_rumor_amplifiers(), exaggeration_enabled=True, exaggeration_gain=1.0, exaggeration_cap=1.5
    )
    assert amp.exaggeration_factor(0.0) == 1.0
    assert amp.exaggeration_factor(0.25) == pytest.approx(1.25)
    assert amp.exaggeration_factor(1.0) == pytest.approx(1.5)
    big = dataclasses.replace(amp, exaggeration_gain=10.0)
    assert big.exaggeration_factor(0.25) == pytest.approx(1.5)


def test_flag_off_mutations_do_not_change_exposure() -> None:
    with_mutations, mutated = _added(_primed(sample=1.0))
    without, none = _added(_primed(sample=0.0))
    assert mutated and not none
    np.testing.assert_array_equal(with_mutations, without)


def test_flag_on_scales_reposts_after_a_mutation() -> None:
    amp = load_rumor_amplifiers()
    on, mutated = _added(_primed(sample=1.0, exaggeration_enabled=True))
    off, _ = _added(_primed(sample=1.0))
    assert mutated
    score = max(event.exaggeration_score for event in mutated)
    expected = min(1.0 + amp.exaggeration_gain * score, amp.exaggeration_cap)
    np.testing.assert_allclose(on, off * expected, rtol=1e-12)
    assert expected > 1.0


def test_cap_is_respected_for_a_large_gain() -> None:
    on, _ = _added(_primed(sample=1.0, exaggeration_enabled=True, exaggeration_gain=10.0))
    off, _ = _added(_primed(sample=1.0))
    cap = load_rumor_amplifiers().exaggeration_cap
    np.testing.assert_allclose(on, off * cap, rtol=1e-12)


def test_flag_on_without_a_mutation_changes_nothing() -> None:
    on, mutated = _added(_primed(sample=0.0, exaggeration_enabled=True))
    off, _ = _added(_primed(sample=0.0))
    assert not mutated
    np.testing.assert_array_equal(on, off)


def test_exaggeration_tracks_the_highest_accepted_score() -> None:
    engine = _primed(sample=1.0, exaggeration_enabled=True)
    _added(engine)
    assert engine._rumors[0].exaggeration == pytest.approx(engine.params.offline_exaggeration)


def test_flag_on_is_deterministic() -> None:
    def run() -> np.ndarray:
        engine = _primed(5, sample=1.0, exaggeration_enabled=True)
        for tick in range(1, 15):
            engine.step(float(tick))
        return engine.attention.copy()

    np.testing.assert_array_equal(run(), run())


def test_exaggeration_flag_counts_for_the_gpu_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    on = dataclasses.replace(load_rumor_amplifiers(), exaggeration_enabled=True)
    monkeypatch.setattr(attention, "load_rumor_amplifiers", lambda: on)
    with pytest.raises(RegistryError):
        AttentionEngine("M03")
