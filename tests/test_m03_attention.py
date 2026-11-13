"""M03 attention and rumor engine: half-life, hub spread, mutator, determinism."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from contracts import parse_event
from contracts.events import (
    AnalystNote,
    AttentionTick,
    BankClosed,
    HeadlinePublished,
    LiquidityDrawn,
    RumorMutated,
    RumorPosted,
    StatementIssued,
    ThesisPublished,
    TradingHalt,
    VIPDecision,
)
from contracts.schemas import Scenario
from engine.agents.attention import (
    AttentionEngine,
    csr_matvec,
    decay_multiplier,
    fallback_layers,
    load_attention_params,
    numpy_csr_matvec,
    resolve_backend,
    select_mutator_indices,
    update_attention,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.bus import EventBus
from mocks.l1_engine import AttentionFake
from params.loader import get_parameter
from tests.test_fakes import panic


def _scenario(seed: int = 1) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed)


def _attached(seed: int = 1) -> AttentionEngine:
    engine = AttentionEngine("M03")
    engine.attach(EventBus(), _scenario(seed))
    return engine


def _rumor(
    source: str,
    *,
    seed: int = 1,
    credibility: float = 1.0,
    rumor_id: str = "r1",
) -> RumorPosted:
    return RumorPosted(
        t=0.0,
        seed=seed,
        bank_id="bank-fvb",
        cause_id="src",
        rumor_id=rumor_id,
        claim="Uninsured cash is moving at Fictional Valley Bank",
        source_node=source,
        credibility=credibility,
    )


def test_operational_rows_are_assumed_and_window_stays_verified() -> None:
    window = get_parameter("Attention memory window")
    assert window.value == "4 hours"
    assert window.module == "M03"
    assert "VERIFIED" in window.status
    names = (
        "M03 sim tick minutes",
        "M03 minutes per hour",
        "M03 rumor exposure",
        "M03 context note credibility factor",
        "M03 rumor mutator sample fraction",
        "M03 rumor mutator max calls per rumor",
        "M03 offline mutator exaggeration",
        "M03 grok mutator enabled",
        "M03 fallback node count",
        "M03 repost attention scale",
    )
    for name in names:
        row = get_parameter(name)
        assert row.status == "ASSUMED"
        assert row.module == "M03"


def test_backend_follows_cuda_probe() -> None:
    from engine.agents.attention import cuda_device_available

    backend = resolve_backend()
    if cuda_device_available():
        assert backend.name == "cupy"
    else:
        assert backend.name == "numpy"
        assert backend.xp is np


def test_numpy_csr_matvec_matches_hand_product() -> None:
    indptr = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    indices = np.asarray([0, 0, 0], dtype=np.int64)
    data = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    vector = np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    got = numpy_csr_matvec(indptr, indices, data, vector)
    assert got.tolist() == [0.0, 2.0, 2.0, 2.0]
    via_switch = csr_matvec(indptr, indices, data, vector, backend=resolve_backend())
    np.testing.assert_allclose(via_switch, got)


def test_half_life_matches_memory_window() -> None:
    params = load_attention_params()
    steps = params.window_ticks
    factor = decay_multiplier(params.half_life_minutes, params.tick_minutes)
    assert factor**steps == pytest.approx(0.5)
    engine = _attached(9)
    engine.attention[:] = 1.0
    for tick in range(steps):
        engine.step(float(tick))
    np.testing.assert_allclose(engine.attention, 0.5, rtol=1e-8, atol=1e-8)


def test_hub_post_outruns_small_account() -> None:
    params = load_attention_params()
    layers, node_ids, banks = fallback_layers(params, "bank-fvb")
    backend = resolve_backend()
    n_nodes = len(node_ids)
    zeros = np.zeros(n_nodes, dtype=np.float64)
    hub = zeros.copy()
    hub[node_ids.index("hub-public-1")] = 1.0
    small = zeros.copy()
    small[node_ids.index("small-account")] = 1.0
    hub_mass = float(update_attention(zeros, 1.0, layers, hub, backend=backend).sum())
    small_mass = float(update_attention(zeros, 1.0, layers, small, backend=backend).sum())
    assert hub_mass > small_mass

    hub_engine = _attached(5)
    small_engine = _attached(5)
    hub_engine.inbox.append(_rumor("hub-public-1"))
    small_engine.inbox.append(_rumor("small-account"))
    hub_engine.step(0.0)
    small_engine.step(0.0)
    assert float(hub_engine.attention.sum()) > float(small_engine.attention.sum())


def test_attention_tick_fires_for_each_bank() -> None:
    engine = AttentionEngine("M03")
    indptr = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    indices = np.asarray([0, 0, 0], dtype=np.int64)
    data = np.ones(3, dtype=np.float64)
    from engine.agents.attention import LayerCSR

    layer = LayerCSR("public", indptr, indices, data, 1.0)
    engine.set_layers(
        [layer],
        ["n0", "n1", "n2", "n3"],
        ["bank-a", "bank-a", "bank-b", "bank-b"],
    )
    engine.attach(EventBus(), _scenario(3))
    events = engine.step(0.0)
    ticks = [event for event in events if isinstance(event, AttentionTick)]
    assert [event.bank_id for event in ticks] == ["bank-a", "bank-b"]
    for event in ticks:
        parsed = parse_event(event.model_dump(mode="json"))
        assert isinstance(parsed, AttentionTick)


def test_context_note_multiplies_credibility_before_spread() -> None:
    params = load_attention_params()

    def mass(with_note: bool) -> float:
        engine = _attached(2)
        engine.inbox.append(_rumor("hub-public-1", credibility=1.0))
        if with_note:
            engine.inbox.append(
                AnalystNote(
                    t=0.0,
                    seed=2,
                    bank_id="bank-fvb",
                    cause_id="note",
                    text="The cash figure is the quarter-end balance",
                    cited_event_ids=["src"],
                )
            )
        engine.step(0.0)
        assert engine._rumors[0].credibility == pytest.approx(
            params.context_factor if with_note else 1.0
        )
        return float(engine.attention.sum())

    assert mass(True) == pytest.approx(mass(False) * params.context_factor)


def test_mutator_sample_matches_fraction_and_budget() -> None:
    nodes = np.arange(1000, dtype=np.int64)
    draws = np.linspace(0.0, 0.999, 1000, dtype=np.float64)
    picked = select_mutator_indices(nodes, 0.01, 20, draws)
    assert int(picked.size) == 10
    assert int(picked[-1]) == 9
    capped = select_mutator_indices(nodes, 1.0, 20, np.zeros(1000, dtype=np.float64))
    assert int(capped.size) == 20
    assert capped.tolist() == list(range(20))
    assert int(select_mutator_indices(nodes, 0.01, 0, draws).size) == 0


def test_engine_mutates_sampled_reposts_and_respects_budget() -> None:
    params = load_attention_params()
    engine = _attached(11)
    engine.inbox.append(_rumor("hub-public-1", credibility=1.0))
    engine.step(0.0)
    rumor = engine._rumors[0]
    n_nodes = int(engine.attention.shape[0])
    # Step 0 drew one uniform vector (the source was a holder and attention was 0).
    probe = np.random.Generator(np.random.Philox(11))
    probe.random(n_nodes)
    rumor.holders[:] = True
    engine.attention[:] = 1.0
    repost_draws = probe.random(n_nodes)
    repost_nodes = np.asarray(np.flatnonzero(repost_draws < 1.0), dtype=np.int64)
    mutator_draws = probe.random(int(repost_nodes.size))
    expected = select_mutator_indices(
        repost_nodes,
        params.sample_fraction,
        params.max_calls,
        mutator_draws,
    )
    events = engine.step(1.0)
    mutated = [event for event in events if isinstance(event, RumorMutated)]
    assert len(mutated) == int(expected.size)
    for event in mutated:
        assert event.claim.endswith(params.offline_suffix)
        assert event.exaggeration_score == pytest.approx(params.offline_exaggeration)
        assert event.rumor_id == "r1"
        parsed = parse_event(event.model_dump(mode="json"))
        assert isinstance(parsed, RumorMutated)
    rumor.mutator_calls = params.max_calls
    engine.attention[:] = 1.0
    rumor.holders[:] = True
    again = [event for event in engine.step(2.0) if isinstance(event, RumorMutated)]
    assert again == []


def test_consumed_shocks_do_not_raise() -> None:
    engine = _attached(4)
    seed = 4
    bank = "bank-fvb"
    engine.inbox.extend(
        [
            HeadlinePublished(
                t=0.0, seed=seed, bank_id=bank, cause_id="h", text="Markets open", reach=1000
            ),
            ThesisPublished(
                t=0.0, seed=seed, bank_id=bank, cause_id="th", author_id="short-1", reach=500
            ),
            LiquidityDrawn(
                t=0.0,
                seed=seed,
                bank_id=bank,
                cause_id="l",
                source="fhlb",
                amount=10.0,
                eta=1.0,
            ),
            TradingHalt(
                t=0.0,
                seed=seed,
                bank_id=bank,
                cause_id="halt",
                reason="move",
                duration=15.0,
            ),
            StatementIssued(
                t=0.0,
                seed=seed,
                bank_id=bank,
                cause_id="st",
                stmt_id="s1",
                channel="wire",
                segments=["all"],
            ),
            BankClosed(t=0.0, seed=seed, bank_id=bank, cause_id="cl", reason="cash"),
            VIPDecision(
                t=0.0,
                seed=seed,
                bank_id=bank,
                cause_id="v",
                agent_id="vip-1",
                action="withdraw",
                amount=1.0,
                rationale="payroll",
            ),
        ]
    )
    events = engine.step(0.0)
    ticks = [event for event in events if isinstance(event, AttentionTick)]
    assert len(ticks) == 1
    assert ticks[0].level > 0.0
    assert ticks[0].bank_id == bank


def test_real_mode_is_deterministic_and_fake_stays_default() -> None:
    scenario = panic(6)
    modes = {"M03": ModuleMode.REAL}
    first = run_scenario(scenario, 6, modes=modes)
    second = run_scenario(scenario, 6, modes=modes)
    assert _dump(first) == _dump(second)
    other = run_scenario(panic(7), 6, modes=modes)
    assert _dump(first) != _dump(other)
    ticks = [event for event in first if isinstance(event, AttentionTick)]
    assert len(ticks) == 6
    assert all(event.bank_id == "bank-fvb" for event in ticks)
    assert ticks[0].delta_4h > 0.0
    assert ticks[0].level > 0.0

    default = build_engine(scenario=_scenario())
    assert isinstance(default["M03"], AttentionFake)
    real = build_engine(modes=modes, scenario=scenario)
    assert isinstance(real["M03"], AttentionEngine)
    assert "M03" not in build_engine(modes={"M03": ModuleMode.OFF}, scenario=scenario)


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json") for event in events]
