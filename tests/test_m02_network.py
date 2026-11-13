"""M02 real mode: heavy tail, communities, bipartite ties, CPU determinism."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from contracts import parse_event
from contracts.schemas import DepositorArrays, Scenario
from engine.agents.network import (
    GROUP,
    PROFESSIONAL,
    PUBLIC,
    InformationNetwork,
    NetworkModule,
    as_numpy,
    build_information_network,
    depositor_segments,
    load_network_params,
    select_backend,
)
from engine.agents.population import parse_registry_number, sample_population
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l1_engine import NetworkFake
from params.loader import get_parameter
from tests.test_fakes import panic

_FOUNDER = 0
_VC = 1
_OTHER = 2


def _param(name: str) -> float:
    return float(get_parameter(name).value)


def _segments(n: int) -> list[str]:
    n_founder = max(1, int(0.18 * n))
    n_vc = max(1, int(0.15 * n))
    n_treasurer = max(1, int(0.16 * n))
    n_startup = n - n_founder - n_vc - n_treasurer
    if n_startup < 1:
        raise AssertionError("segment fixture needs at least one startup")
    return (
        ["founder"] * n_founder
        + ["vc"] * n_vc
        + ["treasurer"] * n_treasurer
        + ["startup"] * n_startup
    )


def _labels(segments: tuple[str, ...]) -> np.ndarray[Any, np.dtype[Any]]:
    coded = np.full(len(segments), _OTHER, dtype=np.int8)
    for index, name in enumerate(segments):
        if name == "founder":
            coded[index] = _FOUNDER
        elif name == "vc":
            coded[index] = _VC
    return coded


def _rows(layer: Any) -> np.ndarray[Any, np.dtype[Any]]:
    indptr = as_numpy(layer.indptr)
    return np.repeat(np.arange(layer.n, dtype=np.int64), np.diff(indptr))


def _assert_csr(layer: Any) -> None:
    indptr = as_numpy(layer.indptr)
    indices = as_numpy(layer.indices)
    data = as_numpy(layer.data)
    assert indptr.shape == (layer.n + 1,)
    assert indptr[0] == 0
    assert indptr[-1] == layer.nnz
    assert np.all(np.diff(indptr) >= 0)
    assert indices.shape == data.shape
    if layer.nnz:
        assert int(indices.min()) >= 0
        assert int(indices.max()) < layer.n
        assert np.all(data == 1.0)


def _dense(layer: Any) -> np.ndarray[Any, np.dtype[Any]]:
    matrix = np.zeros((layer.n, layer.n), dtype=np.float64)
    indptr = as_numpy(layer.indptr)
    indices = as_numpy(layer.indices)
    data = as_numpy(layer.data)
    for row in range(layer.n):
        start = int(indptr[row])
        stop = int(indptr[row + 1])
        matrix[row, indices[start:stop]] = data[start:stop]
    return matrix


def _recovery(network: InformationNetwork, code: int) -> float:
    layer = network.layers[GROUP]
    labels = _labels(network.segments)
    indptr = as_numpy(layer.indptr)
    indices = as_numpy(layer.indices)
    nodes = np.flatnonzero(labels == code)
    correct = 0
    for node in nodes:
        start = int(indptr[node])
        stop = int(indptr[node + 1])
        if start == stop:
            continue
        counts = np.bincount(labels[indices[start:stop]], minlength=3)
        if int(counts.argmax()) == code:
            correct += 1
    return correct / float(len(nodes))


def test_backend_switch_isolates_gpu(monkeypatch: Any) -> None:
    from engine.agents import network as network_module

    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)  # CPY-1: the default prefers the GPU
    xp, kind = select_backend()
    if network_module._cupy_module() is None:
        assert kind == "cpu"
        assert xp is np
    else:
        assert kind == "gpu"

    monkeypatch.setattr(network_module, "_cupy_module", lambda: None)
    xp, kind = select_backend()
    assert kind == "cpu"
    assert xp is np

    sentinel = object()
    monkeypatch.setattr(network_module, "_cupy_module", lambda: sentinel)
    xp, kind = select_backend()
    assert kind == "gpu"
    assert xp is sentinel
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")  # CPY-1: the NumPy reference, even with a device
    xp, kind = select_backend()
    assert kind == "cpu"
    assert xp is np


def test_params_come_from_the_registry() -> None:
    params = load_network_params()
    assert params.media_count == int(_param("M02 media and analyst node count"))
    assert params.public_m == int(_param("M02 public feed edges per new node"))
    assert params.homophily == _param("M02 group chat homophily")
    assert params.population_tier == int(_param("M02 population tier"))


def test_public_feed_has_a_heavy_tail() -> None:
    network = build_information_network(_segments(8000), seed=11)
    assert network.backend in {"cpu", "gpu"}
    _assert_csr(network.layers[PUBLIC])
    degrees = as_numpy(network.follower_counts())
    assert int(degrees.sum()) == network.layers[PUBLIC].nnz
    total = float(degrees.sum())
    fraction = _param("M02 test heavy-tail top fraction")
    top_n = max(1, int(degrees.size * fraction))
    share = float(np.sort(degrees)[-top_n:].sum()) / total
    assert share >= _param("M02 test heavy-tail top share minimum")
    assert float(degrees.max()) >= _param("M02 test heavy-tail max over mean minimum") * float(
        degrees.mean()
    )
    media = np.array([name == "media" for name in network.segments])
    assert int(media.sum()) == network.params.media_count
    assert float(np.median(degrees[media])) > float(np.median(degrees))


def test_group_communities_can_be_recovered() -> None:
    network = build_information_network(_segments(4000), seed=5)
    layer = network.layers[GROUP]
    _assert_csr(layer)
    labels = _labels(network.segments)
    rows = _rows(layer)
    indices = as_numpy(layer.indices)
    within = float(np.mean(labels[rows] == labels[indices]))
    assert within >= _param("M02 test within-community edge share minimum")
    # Both directions are stored, so each undirected chat edge appears twice.
    paired = {(int(rows[i]), int(indices[i])) for i in range(rows.size)}
    assert all((dst, src) in paired for src, dst in paired)
    floor = _param("M02 test community recovery minimum")
    assert _recovery(network, _FOUNDER) >= floor
    assert _recovery(network, _VC) >= floor


def test_professional_ties_are_bipartite() -> None:
    network = build_information_network(_segments(3000), seed=9)
    layer = network.layers[PROFESSIONAL]
    _assert_csr(layer)
    assert layer.nnz > 0
    rows = _rows(layer)
    indices = as_numpy(layer.indices)
    treasurer = np.array([name == "treasurer" for name in network.segments])
    advisor = as_numpy(network.is_advisor).astype(bool)
    assert not np.any(treasurer & advisor)
    assert np.all(treasurer[rows] == advisor[indices])
    assert np.all(advisor[rows] == treasurer[indices])
    media = np.array([name == "media" for name in network.segments])
    assert int(advisor[media].sum()) == int(media.sum())


def test_spread_matches_a_dense_channel_weighted_product() -> None:
    network = build_information_network(_segments(48), seed=3)
    exposure = np.linspace(0.1, 1.0, network.n)
    channel = as_numpy(network.channel_weight)
    weighted = channel * exposure
    expected = np.zeros(network.n, dtype=np.float64)
    for name, layer in network.layers.items():
        expected += network.layer_weight[name] * (_dense(layer) @ weighted)
    np.testing.assert_allclose(as_numpy(network.spread(exposure)), expected)
    np.testing.assert_allclose(
        as_numpy(network.spread(exposure, layer=PUBLIC)),
        network.layer_weight[PUBLIC] * (_dense(network.layers[PUBLIC]) @ weighted),
    )
    credibility = as_numpy(network.credibility)
    assert credibility.min() >= network.params.credibility_floor - 1e-12
    assert credibility.max() <= network.params.credibility_cap + 1e-12
    assert np.all(channel == network.params.node_channel_weight)


def test_cpu_path_is_deterministic_for_a_fixed_seed() -> None:
    segments = _segments(1500)
    first = build_information_network(segments, seed=21)
    second = build_information_network(segments, seed=21)
    other = build_information_network(segments, seed=22)
    assert first.backend == "cpu" or second.backend == first.backend
    for name in (PUBLIC, GROUP, PROFESSIONAL):
        np.testing.assert_array_equal(
            as_numpy(first.layers[name].indices),
            as_numpy(second.layers[name].indices),
        )
        np.testing.assert_array_equal(
            as_numpy(first.layers[name].indptr),
            as_numpy(second.layers[name].indptr),
        )
    np.testing.assert_array_equal(as_numpy(first.credibility), as_numpy(second.credibility))
    assert not np.array_equal(
        as_numpy(first.layers[PUBLIC].indices),
        as_numpy(other.layers[PUBLIC].indices),
    )
    assert first.segments[: len(segments)] == tuple(segments)
    assert first.segments[len(segments) :] == ("media",) * first.params.media_count


def test_depositor_segments_follow_m05(monkeypatch: Any) -> None:
    seen: dict[str, int] = {}

    def fake_sample(scenario: Scenario, tier: int = 0) -> tuple[DepositorArrays, None]:
        del scenario
        seen["tier"] = tier
        segments = ["founder", "vc", "treasurer", "startup"]
        return (
            DepositorArrays(
                bank_idx=[0, 0, 0, 0],
                segment=segments,
                balance=[1.0, 1.0, 1.0, 1.0],
                insured=[1.0, 1.0, 1.0, 1.0],
                belief=[0.0, 0.0, 0.0, 0.0],
                threshold=[0.5, 0.5, 0.5, 0.5],
                attention=[0.0, 0.0, 0.0, 0.0],
                needs_flag=[False, False, False, False],
                has_second_bank=[False, False, False, False],
                state=["active", "active", "active", "active"],
            ),
            None,
        )

    monkeypatch.setattr("engine.agents.population.sample_population", fake_sample)
    scenario = panic(4)
    assert depositor_segments(scenario, 0) == ["founder", "vc", "treasurer", "startup"]
    assert seen["tier"] == 0

    def forbid(scenario: Scenario, tier: int = 0) -> tuple[DepositorArrays, None]:
        del scenario, tier
        raise AssertionError("community presets must not call sample_population")

    monkeypatch.setattr("engine.agents.population.sample_population", forbid)
    community = Scenario(preset="insured community bank", seed=1)
    retail = depositor_segments(community, 0)
    assert len(retail) == int(parse_registry_number("Depositor count (Tier 0)"))
    assert set(retail) == {"retail"}


def test_step_publishes_nothing_and_keeps_m05_indexes() -> None:
    scenario = panic(6)
    module = NetworkModule("M02")
    assert module.step(0.0) == []
    assert module.network is None
    from mocks.bus import EventBus

    module.attach(EventBus(), scenario)
    assert module.step(0.0) == []
    assert module.network is not None
    built = module.network
    assert module.step(1.0) == []
    assert module.network is built
    arrays, _summary = sample_population(scenario)
    assert built.segments[: built.n_depositors] == tuple(arrays.segment)
    assert built.n_depositors == len(arrays.segment)


def test_engine_real_mode_matches_fake_events_and_stays_opt_in() -> None:
    scenario = panic(2)
    fake = build_engine(scenario=scenario)
    real = build_engine(modes={"M02": ModuleMode.REAL}, scenario=scenario)
    assert isinstance(fake["M02"], NetworkFake)
    assert isinstance(real["M02"], NetworkModule)
    fake_log = fake.run(4)
    real_log = real.run(4)
    assert [event.model_dump(mode="json", by_alias=True) for event in real_log] == [
        event.model_dump(mode="json", by_alias=True) for event in fake_log
    ]
    for event in real_log:
        payload = event.model_dump(mode="json", by_alias=True)
        parsed = parse_event(payload)
        assert parsed.model_dump(mode="json", by_alias=True) == payload
    network = real["M02"].network
    assert isinstance(network, InformationNetwork)
    assert network.n_depositors == int(parse_registry_number("Depositor count (Tier 0)"))

    off = build_engine(modes={"M02": ModuleMode.OFF}, scenario=scenario)
    assert "M02" not in off
    with pytest.raises(NotImplementedError, match="M30"):
        build_engine(modes={"M30": ModuleMode.REAL})


def test_million_depositor_nodes_build_within_budget() -> None:
    count = int(parse_registry_number("Depositor count (Tier 2)"))
    n_founder = count // 5
    n_vc = count // 6
    n_treasurer = count // 6
    segments = (
        ["founder"] * n_founder
        + ["vc"] * n_vc
        + ["treasurer"] * n_treasurer
        + ["startup"] * (count - n_founder - n_vc - n_treasurer)
    )
    started = time.perf_counter()
    network = build_information_network(segments, seed=1)
    elapsed = time.perf_counter() - started
    assert network.n_depositors == count
    assert elapsed < _param("M02 build time budget")
    for name in (PUBLIC, GROUP, PROFESSIONAL):
        _assert_csr(network.layers[name])
        assert network.layers[name].nnz > 0
