"""SVB-42: claim strength into the M03 post impulse (ADR-094). Flag off by default.

M03 uses the offline lexicon strength of ``RumorPosted.claim``. That is the
same rule M34 classification uses offline (one shared function), so a replay
of the log gives the same impulse. Live judged strength does not reach M03.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import engine.agents.attention as attention
from agents.rumor_input import classify_rumor_market
from agents.rumor_lexicon import lexicon_tone_and_strength, normalize
from contracts.events import RumorPosted
from contracts.schemas import Scenario
from engine.agents.attention import AttentionEngine, claim_strength, load_rumor_amplifiers
from mocks.bus import EventBus
from params.loader import RegistryError, get_parameter

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "rumor_classification_77db0e2.json"
HIGH = "Kestrel Bank is out of cash and may delay uninsured wires, pull your money now"
MEDIUM = "Kestrel may delay wires"
LOW = "Kestrel Bank is well capitalized and stable"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def _mass(claim: str, source: str = "hub-public-1", **flags: bool) -> float:
    engine = AttentionEngine("M03")
    engine.attach(EventBus(), Scenario(preset="tech-concentrated", seed=1))
    if flags:
        engine.amplifiers = dataclasses.replace(engine.amplifiers, **flags)
    engine.inbox.append(
        RumorPosted(
            t=0.0,
            seed=1,
            bank_id="bank-fvb",
            cause_id="test",
            rumor_id="r1",
            claim=claim,
            source_node=source,
            credibility=0.6,
        )
    )
    engine.step(0.0)
    return float(engine.attention.sum())


def test_rows_are_assumed_and_flag_defaults_off() -> None:
    for name in ("M03 rumor strength impulse enabled", "M03 rumor strength impulse multipliers"):
        row = get_parameter(name)
        assert row.status == "ASSUMED" and row.module == "M03"
    assert get_parameter("M03 rumor strength impulse enabled").value == "false"
    amplifiers = load_rumor_amplifiers()
    assert amplifiers.strength_enabled is False
    assert set(amplifiers.strength_multipliers) == {"low", "medium", "high"}
    assert amplifiers.strength_multipliers["medium"] == 1.0


def test_strength_map_must_have_exactly_three_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    real = attention.list_by_module

    def broken(module: str, path=None):
        return [
            dataclasses.replace(row, value="low:1,high:2")
            if row.parameter == "M03 rumor strength impulse multipliers"
            else row
            for row in real(module)
        ]

    monkeypatch.setattr(attention, "list_by_module", broken)
    with pytest.raises(RegistryError, match="low, medium and high"):
        load_rumor_amplifiers()


def test_claim_strength_matches_offline_classification_on_the_golden_table() -> None:
    rows = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for row in rows:
        if row["affects_simulation"]:
            assert claim_strength(row["text"]) == row["strength"], row["text"]


def test_classification_and_m03_share_one_rule() -> None:
    for text in (HIGH, MEDIUM, LOW):
        tone, strength = lexicon_tone_and_strength(normalize(text))
        result = classify_rumor_market(text, "journalist")
        assert (result.tone, result.strength_lexicon) == (tone, strength)
        assert claim_strength(text) == strength


def test_example_claims_have_the_expected_levels() -> None:
    assert [claim_strength(t) for t in (HIGH, MEDIUM, LOW)] == ["high", "medium", "low"]


def test_flag_off_ignores_claim_text() -> None:
    assert _mass(HIGH) == _mass(LOW) == _mass(MEDIUM)


def test_flag_on_scales_impulse_by_strength_exactly() -> None:
    mult = load_rumor_amplifiers().strength_multipliers
    high = _mass(HIGH, strength_enabled=True)
    low = _mass(LOW, strength_enabled=True)
    medium = _mass(MEDIUM, strength_enabled=True)
    assert high > medium > low
    assert high / low == pytest.approx(mult["high"] / mult["low"])
    assert medium == _mass(MEDIUM)


def test_strength_and_role_multipliers_compose() -> None:
    amp = load_rumor_amplifiers()
    both = _mass(HIGH, "journalist", strength_enabled=True, role_multiplier_enabled=True)
    base = _mass(HIGH, "journalist")
    assert both / base == pytest.approx(
        amp.strength_multipliers["high"] * amp.role_multipliers["journalist"]
    )


def test_strength_flag_counts_for_the_gpu_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    on = dataclasses.replace(load_rumor_amplifiers(), strength_enabled=True)
    monkeypatch.setattr(attention, "load_rumor_amplifiers", lambda: on)
    with pytest.raises(RegistryError):
        AttentionEngine("M03")


def test_flag_on_is_deterministic_over_many_ticks() -> None:
    def run() -> np.ndarray:
        engine = AttentionEngine("M03")
        engine.attach(EventBus(), Scenario(preset="tech-concentrated", seed=4))
        engine.amplifiers = dataclasses.replace(engine.amplifiers, strength_enabled=True)
        engine.inbox.append(
            RumorPosted(
                t=0.0,
                seed=4,
                bank_id="bank-fvb",
                cause_id="t",
                rumor_id="r1",
                claim=HIGH,
                source_node="hub-public-1",
                credibility=0.9,
            )
        )
        for tick in range(20):
            engine.step(float(tick))
        return engine.attention.copy()

    np.testing.assert_array_equal(run(), run())
