"""SVB-57 (D2): M03 can run on M02's depositor network (flag off by default, ADR-103)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import engine.agents.attention as attention
from contracts.events import RumorPosted
from engine.agents.attention import M02_ROLE_SEGMENTS, AttentionEngine, load_rumor_amplifiers
from engine.agents.network import NetworkModule
from engine.modes import ModuleMode
from engine.registry import build_engine
from params.loader import RegistryError, get_parameter
from tests.test_fakes import panic

MODES = {"M02": ModuleMode.REAL, "M03": ModuleMode.REAL}


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def _engine(*, on: bool, role_nodes: bool = False, seed: int = 1):
    engine = build_engine(modes=MODES, scenario=panic(seed))
    m03 = engine["M03"]
    if on or role_nodes:
        m03.amplifiers = dataclasses.replace(
            m03.amplifiers, m02_network_enabled=on, role_node_enabled=role_nodes
        )
    return engine


def _post(engine, source: str) -> np.ndarray:
    m03 = engine["M03"]
    m03.inbox.append(
        RumorPosted(t=0.0, seed=1, bank_id="bank-fvb", cause_id="t", rumor_id="r1",
                    claim="Kestrel Bank may delay wires", source_node=source, credibility=0.6)
    )
    m03.step(0.0)
    return m03.attention.copy()


def test_row_is_assumed_and_off() -> None:
    row = get_parameter("M03 uses M02 network")
    assert row.status == "ASSUMED" and row.module == "M03" and row.value == "false"
    assert load_rumor_amplifiers().m02_network_enabled is False


def test_build_engine_hands_m02_to_m03() -> None:
    engine = build_engine(modes=MODES, scenario=panic(1))
    assert isinstance(engine["M02"], NetworkModule)
    assert engine["M03"].network_source is engine["M02"]


def test_flag_off_keeps_the_fallback_graph() -> None:
    engine = _engine(on=False)
    engine["M03"].step(0.0)
    assert len(engine["M03"]._node_ids) == 16
    assert engine["M03"]._m02_installed is False


def test_flag_on_installs_the_m02_network_on_the_first_step() -> None:
    engine = _engine(on=True)
    m03 = engine["M03"]
    m03.step(0.0)
    network = engine["M02"].network
    assert m03._m02_installed is True
    assert len(m03._node_ids) == network.n
    assert {layer.name for layer in m03._layers} == set(network.layers)
    assert all(node.startswith("m02-") for node in m03._node_ids[:5])


def test_role_nodes_sit_in_the_documented_segments() -> None:
    engine = _engine(on=True)
    m03 = engine["M03"]
    m03.step(0.0)
    segments = list(engine["M02"].network.segments)
    for role, node in m03._m02_role_nodes.items():
        assert segments[m03._index[node]] == M02_ROLE_SEGMENTS[role]


def test_media_role_reaches_more_than_a_regular_user() -> None:
    journalist = _post(_engine(on=True, role_nodes=True), "journalist")
    user = _post(_engine(on=True, role_nodes=True), "ui-user")
    assert int(np.count_nonzero(journalist)) > int(np.count_nonzero(user))


def test_unmapped_sources_post_from_the_hub() -> None:
    engine = _engine(on=True)
    m03 = engine["M03"]
    m03.step(0.0)
    assert m03._source_index("hub-public-1") == m03._hub
    assert m03._source_index("journalist") == m03._hub  # role-node flag off


def test_flag_on_is_deterministic() -> None:
    a = _post(_engine(on=True, role_nodes=True, seed=3), "vc-partner")
    b = _post(_engine(on=True, role_nodes=True, seed=3), "vc-partner")
    np.testing.assert_array_equal(a, b)


def test_flag_counts_for_the_gpu_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    on = dataclasses.replace(load_rumor_amplifiers(), m02_network_enabled=True)
    monkeypatch.setattr(attention, "load_rumor_amplifiers", lambda: on)
    with pytest.raises(RegistryError):
        AttentionEngine("M03")
