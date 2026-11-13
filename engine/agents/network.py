"""M02 multilayer information network (§5, §18.1, §19, ADR-027).

Array code goes through one backend: CuPy when a CUDA device is available,
NumPy otherwise. The CPU path uses a seeded NumPy Generator and is
deterministic. The GPU path is isolated in ``_cupy_module`` and
``_gpu_generator`` and is not required to match the CPU stream.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from contracts.schemas import Scenario
from mocks.bus import EventBus
from params.loader import RegistryError, list_by_module

PUBLIC = "public"
GROUP = "group"
PROFESSIONAL = "professional"
LAYER_NAMES: tuple[str, ...] = (PUBLIC, GROUP, PROFESSIONAL)

_FOUNDER = 0
_VC = 1
_OTHER = 2

_MEDIA = "M02 media and analyst node count"
_PUBLIC_M = "M02 public feed edges per new node"
_SEED = "M02 public feed seed nodes"
_BATCH = "M02 attachment batch size"
_GROUP_DEGREE = "M02 group chat degree"
_HOMOPHILY = "M02 group chat homophily"
_ADVISORS = "M02 advisor count"
_PRO_DEGREE = "M02 professional ties per treasurer"
_W_PUBLIC = "M02 channel weight public"
_W_GROUP = "M02 channel weight group"
_W_PRO = "M02 channel weight professional"
_NODE_W = "M02 node channel weight"
_CRED_BASE = "M02 credibility base"
_CRED_MEDIA = "M02 credibility media bonus"
_CRED_HUB = "M02 credibility hub scale"
_CRED_FLOOR = "M02 credibility floor"
_CRED_CAP = "M02 credibility cap"
_TIER = "M02 population tier"


@dataclass(frozen=True, slots=True)
class NetworkParams:
    """ASSUMED M02 inputs loaded from the parameter registry."""

    media_count: int
    public_m: int
    seed_nodes: int
    batch_size: int
    group_degree: int
    homophily: float
    advisor_count: int
    professional_degree: int
    weight_public: float
    weight_group: float
    weight_professional: float
    node_channel_weight: float
    credibility_base: float
    credibility_media_bonus: float
    credibility_hub_scale: float
    credibility_floor: float
    credibility_cap: float
    population_tier: int


class CsrLayer:
    """One directed CSR adjacency. ``data`` is 1 on every stored edge."""

    def __init__(self, name: str, indptr: Any, indices: Any, data: Any, n: int) -> None:
        self.name = name
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.n = n
        self.nnz = int(indices.shape[0])

    def matvec(self, x: Any, xp: Any) -> Any:
        """Return A @ x. Empty rows contribute 0."""
        if self.nnz == 0 or self.n == 0:
            return xp.zeros(self.n, dtype=xp.float64)
        contrib = self.data * x[self.indices]
        repeats = xp.diff(self.indptr)
        rows = xp.repeat(xp.arange(self.n, dtype=xp.int64), repeats)
        return xp.bincount(rows, weights=contrib, minlength=self.n).astype(xp.float64, copy=False)


class InformationNetwork:
    """Three CSR layers plus per-node credibility and channel weight."""

    def __init__(
        self,
        *,
        segments: tuple[str, ...],
        n_depositors: int,
        credibility: Any,
        channel_weight: Any,
        is_advisor: Any,
        layers: dict[str, CsrLayer],
        layer_weight: dict[str, float],
        backend: str,
        xp: Any,
        params: NetworkParams,
    ) -> None:
        self.segments = segments
        self.n_depositors = n_depositors
        self.n = len(segments)
        self.credibility = credibility
        self.channel_weight = channel_weight
        self.is_advisor = is_advisor
        self.layers = layers
        self.layer_weight = layer_weight
        self.backend = backend
        self.xp = xp
        self.params = params

    def follower_counts(self) -> Any:
        """In-degree of the public feed (how many nodes follow each account)."""
        layer = self.layers[PUBLIC]
        return self.xp.bincount(layer.indices, minlength=self.n)

    def spread(self, exposure: Any, layer: str | None = None) -> Any:
        """Channel-weighted attention spread.

        ``sum_layer w_layer * A_layer @ (node_channel_weight * exposure)``.
        With a unit node weight this is the M03 form ``w · A · exposure``.
        """
        xp = self.xp
        signal = xp.asarray(exposure, dtype=xp.float64)
        if signal.shape != (self.n,):
            raise ValueError("exposure length must match the node count")
        weighted = self.channel_weight * signal
        names = (layer,) if layer is not None else LAYER_NAMES
        total = xp.zeros(self.n, dtype=xp.float64)
        for name in names:
            graph = self.layers[name]
            total = total + self.layer_weight[name] * graph.matvec(weighted, xp)
        return total


def as_numpy(array: Any) -> np.ndarray[Any, np.dtype[Any]]:
    """Copy a backend array onto NumPy. CuPy arrays use ``.get`` (GPU path)."""
    module_name = type(array).__module__.split(".", 1)[0]
    if module_name == "cupy":
        return np.asarray(array.get())
    return np.asarray(array)


def select_backend() -> tuple[Any, str]:
    """Return ``(cupy, "gpu")`` when CUDA is visible, else ``(numpy, "cpu")``.

    CPY-1: ``STAMPEDE_BACKEND=cpu`` forces NumPy even with a device (the reference path).
    """
    from engine.xp import backend_preference

    if backend_preference() == "cpu":
        return np, "cpu"
    cupy_module = _cupy_module()
    if cupy_module is not None:
        return cupy_module, "gpu"
    return np, "cpu"


def _cupy_module() -> Any | None:
    """Isolated GPU import (ADR-027).

    Returns CuPy only when it imports and a CUDA device is visible. The CPU
    path does not use the returned module for draws or arithmetic.
    """
    try:
        cupy = importlib.import_module("cupy")
    except ImportError:
        return None
    try:
        available = bool(cupy.cuda.is_available())
        devices = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return None
    if not available or devices < 1:
        return None
    return cupy


def _generator(xp: Any, backend: str, seed: int) -> Any:
    if backend == "cpu":
        return xp.random.default_rng(seed)
    return _gpu_generator(xp, seed)


def _permutation(xp: Any, rng: Any, n: int) -> Any:
    """Shuffle indices 0..n-1 with the seeded generator ``rng``.

    NumPy's Generator has ``permutation``, so the CPU stream is unchanged. CuPy's Generator lacks
    it (ADR-027), so the GPU path ranks one uniform draw per index from the same generator. It
    never falls back to the backend's global RNG, which no seed reaches (SVB-68).
    """
    draw = getattr(rng, "permutation", None)
    if callable(draw):
        return draw(int(n))
    return xp.argsort(rng.random(int(n)))


def _gpu_generator(xp: Any, seed: int) -> Any:
    """CuPy Generator. Not used on the CPU path (ADR-027)."""
    random = xp.random
    for name in ("Philox4x3210", "Philox"):
        bitgen_cls = getattr(random, name, None)
        if bitgen_cls is not None:
            return random.Generator(bitgen_cls(seed))
    return random.default_rng(seed)


def _argsort(xp: Any, backend: str, keys: Any) -> Any:
    if backend == "cpu":
        return xp.argsort(keys, kind="mergesort")
    return xp.argsort(keys)


def _as_float(name: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise RegistryError(f"{name} is not a float: {text!r}") from exc
    if not math.isfinite(value):
        raise RegistryError(f"{name} is not finite: {text!r}")
    return value


def _as_int(name: str, text: str) -> int:
    value = _as_float(name, text)
    if not value.is_integer():
        raise RegistryError(f"{name} is not an integer: {text!r}")
    if value < 0:
        raise RegistryError(f"{name} is negative: {text!r}")
    return int(value)


def _required(rows: dict[str, str], name: str, *, integer: bool) -> float:
    try:
        text = rows[name]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {name}") from exc
    if integer:
        return float(_as_int(name, text))
    return _as_float(name, text)


def load_network_params() -> NetworkParams:
    """Load M02 rows from ``params/registry.csv`` via ``params.loader``."""
    rows = {row.parameter: row.value for row in list_by_module("M02")}
    params = NetworkParams(
        media_count=int(_required(rows, _MEDIA, integer=True)),
        public_m=int(_required(rows, _PUBLIC_M, integer=True)),
        seed_nodes=int(_required(rows, _SEED, integer=True)),
        batch_size=int(_required(rows, _BATCH, integer=True)),
        group_degree=int(_required(rows, _GROUP_DEGREE, integer=True)),
        homophily=_required(rows, _HOMOPHILY, integer=False),
        advisor_count=int(_required(rows, _ADVISORS, integer=True)),
        professional_degree=int(_required(rows, _PRO_DEGREE, integer=True)),
        weight_public=_required(rows, _W_PUBLIC, integer=False),
        weight_group=_required(rows, _W_GROUP, integer=False),
        weight_professional=_required(rows, _W_PRO, integer=False),
        node_channel_weight=_required(rows, _NODE_W, integer=False),
        credibility_base=_required(rows, _CRED_BASE, integer=False),
        credibility_media_bonus=_required(rows, _CRED_MEDIA, integer=False),
        credibility_hub_scale=_required(rows, _CRED_HUB, integer=False),
        credibility_floor=_required(rows, _CRED_FLOOR, integer=False),
        credibility_cap=_required(rows, _CRED_CAP, integer=False),
        population_tier=int(_required(rows, _TIER, integer=True)),
    )
    if params.public_m < 1:
        raise RegistryError("M02 public feed edges per new node must be >= 1")
    if params.seed_nodes < 1:
        raise RegistryError("M02 public feed seed nodes must be >= 1")
    if params.batch_size < 1:
        raise RegistryError("M02 attachment batch size must be >= 1")
    if params.group_degree < 1:
        raise RegistryError("M02 group chat degree must be >= 1")
    if params.professional_degree < 1:
        raise RegistryError("M02 professional ties per treasurer must be >= 1")
    if not 0.0 <= params.homophily <= 1.0:
        raise RegistryError("M02 group chat homophily must be between 0 and 1")
    if params.credibility_floor > params.credibility_cap:
        raise RegistryError("M02 credibility floor exceeds cap")
    if params.population_tier not in (0, 2):
        raise RegistryError("M02 population tier must be 0 or 2")
    return params


def depositor_segments(scenario: Scenario, tier: int) -> list[str]:
    """Load M05 segments. Community presets stay all-retail, matching M05."""
    from engine.agents.population import parse_registry_number, sample_population
    from mocks.support import is_community

    if is_community(scenario.preset):
        count_name = "Depositor count (Tier 0)" if tier == 0 else "Depositor count (Tier 2)"
        count = int(parse_registry_number(count_name))
        return ["retail"] * count
    arrays, _summary = sample_population(scenario, tier=tier)
    return list(arrays.segment)


def _empty_edges(xp: Any) -> tuple[Any, Any]:
    empty = xp.zeros(0, dtype=xp.int32)
    return empty, empty


def _csr(xp: Any, backend: str, src: Any, dst: Any, n: int, name: str) -> CsrLayer:
    if n == 0 or int(src.shape[0]) == 0:
        indptr = xp.zeros(n + 1, dtype=xp.int64)
        indices = xp.zeros(0, dtype=xp.int32)
        data = xp.zeros(0, dtype=xp.float64)
        return CsrLayer(name, indptr, indices, data, n)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if int(src.shape[0]) == 0:
        indptr = xp.zeros(n + 1, dtype=xp.int64)
        indices = xp.zeros(0, dtype=xp.int32)
        data = xp.zeros(0, dtype=xp.float64)
        return CsrLayer(name, indptr, indices, data, n)
    keys = src.astype(xp.int64) * n + dst.astype(xp.int64)
    index = xp.unique(keys, return_index=True)[1]
    src_u = src[index]
    dst_u = dst[index]
    order = _argsort(xp, backend, src_u)
    src_u = src_u[order]
    dst_u = dst_u[order]
    counts = xp.bincount(src_u.astype(xp.int64), minlength=n)
    indptr = xp.zeros(n + 1, dtype=xp.int64)
    indptr[1:] = xp.cumsum(counts)
    data = xp.ones(int(dst_u.shape[0]), dtype=xp.float64)
    return CsrLayer(name, indptr, dst_u.astype(xp.int32), data, n)


def _remap(xp: Any, src: Any, dst: Any, n_media: int, n_depositors: int) -> tuple[Any, Any]:
    n = n_media + n_depositors
    mapping = xp.empty(n, dtype=xp.int32)
    if n_media:
        mapping[:n_media] = xp.arange(n_depositors, n_depositors + n_media, dtype=xp.int32)
    if n_depositors:
        mapping[n_media:] = xp.arange(n_depositors, dtype=xp.int32)
    if int(src.shape[0]) == 0:
        return src, dst
    return mapping[src], mapping[dst]


def _preferential_edges(
    xp: Any,
    rng: Any,
    n: int,
    m: int,
    m0: int,
    batch: int,
) -> tuple[Any, Any]:
    """Batched preferential attachment in construction-index space.

    Each batch samples targets from the degree bag built so far. Both
    endpoints are written back so later batches can attach to new nodes.
    """
    if n < 2 or m < 1 or m0 < 2:
        return _empty_edges(xp)
    m0 = min(m0, n)
    m = min(m, n - 1)
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    for i in range(m0):
        for j in range(m0):
            if i != j:
                pairs_i.append(i)
                pairs_j.append(j)
    src0 = xp.asarray(pairs_i, dtype=xp.int32)
    dst0 = xp.asarray(pairs_j, dtype=xp.int32)
    init = xp.repeat(xp.arange(m0, dtype=xp.int32), m0 - 1)
    bag_cap = int(init.shape[0]) + 2 * m * (n - m0)
    bag = xp.empty(bag_cap, dtype=xp.int32)
    bag[: int(init.shape[0])] = init
    bag_len = int(init.shape[0])
    src_chunks: list[Any] = [src0]
    dst_chunks: list[Any] = [dst0]
    cursor = m0
    while cursor < n:
        width = min(batch, n - cursor)
        drawn = width * m
        picks = rng.integers(0, bag_len, size=(width, m))
        targets = bag[picks].reshape(drawn)
        new_ids = xp.arange(cursor, cursor + width, dtype=xp.int32)
        src_chunks.append(xp.repeat(new_ids, m))
        dst_chunks.append(targets.astype(xp.int32, copy=False))
        bag[bag_len : bag_len + drawn] = targets
        bag_len += drawn
        bag[bag_len : bag_len + drawn] = xp.repeat(new_ids, m)
        bag_len += drawn
        cursor += width
    return xp.concatenate(src_chunks), xp.concatenate(dst_chunks)


def _indices(segments: Sequence[str], label: str) -> list[int]:
    return [index for index, name in enumerate(segments) if name == label]


def _block_edges(
    xp: Any,
    rng: Any,
    n: int,
    labels: Any,
    members: Any,
    degree: int,
    homophily: float,
) -> tuple[Any, Any]:
    k = int(members.shape[0])
    outside = xp.flatnonzero(labels != labels[members[0]]) if k else xp.zeros(0, dtype=xp.int32)
    outside_n = int(outside.shape[0])
    if k == 0 or degree < 1 or (k <= 1 and outside_n == 0):
        return _empty_edges(xp)
    use_within = rng.random((k, degree)) < homophily
    if k <= 1:
        use_within = xp.zeros((k, degree), dtype=bool)
    elif outside_n == 0:
        use_within = xp.ones((k, degree), dtype=bool)
    if k > 1:
        raw = rng.integers(0, k - 1, size=(k, degree))
        local = xp.arange(k)[:, None]
        within_local = raw + (raw >= local)
        within_global = members[within_local]
    else:
        within_global = xp.zeros((k, degree), dtype=xp.int32)
    if outside_n:
        out_global = outside[rng.integers(0, outside_n, size=(k, degree))]
    else:
        out_global = within_global
    neighbors = xp.where(use_within, within_global, out_global).astype(xp.int32, copy=False)
    src = xp.repeat(members, degree)
    dst = neighbors.reshape(k * degree)
    return xp.concatenate([src, dst]), xp.concatenate([dst, src])


def _group_edges(
    xp: Any,
    rng: Any,
    n: int,
    labels: Any,
    degree: int,
    homophily: float,
) -> tuple[Any, Any]:
    src_chunks: list[Any] = []
    dst_chunks: list[Any] = []
    for code in (_FOUNDER, _VC, _OTHER):
        members = xp.flatnonzero(labels == code)
        if int(members.shape[0]) == 0:
            continue
        src, dst = _block_edges(xp, rng, n, labels, members, degree, homophily)
        if int(src.shape[0]) == 0:
            continue
        src_chunks.append(src)
        dst_chunks.append(dst)
    if not src_chunks:
        return _empty_edges(xp)
    return xp.concatenate(src_chunks), xp.concatenate(dst_chunks)


def _professional_edges(
    xp: Any,
    rng: Any,
    treasurers: Any,
    advisors: Any,
    degree: int,
) -> tuple[Any, Any]:
    n_t = int(treasurers.shape[0])
    n_a = int(advisors.shape[0])
    if n_t == 0 or n_a == 0 or degree < 1:
        return _empty_edges(xp)
    picks = rng.integers(0, n_a, size=(n_t, degree))
    dst = advisors[picks].reshape(n_t * degree)
    src = xp.repeat(treasurers, degree)
    return xp.concatenate([src, dst]), xp.concatenate([dst, src])


def _credibility(
    xp: Any,
    followers: Any,
    n_depositors: int,
    n_media: int,
    params: NetworkParams,
) -> Any:
    n = n_depositors + n_media
    peak = float(followers.max()) if n else 0.0
    if peak <= 0.0:
        hub = xp.zeros(n, dtype=xp.float64)
    else:
        hub = xp.log1p(followers.astype(xp.float64)) / math.log1p(peak)
    media = xp.zeros(n, dtype=xp.float64)
    if n_media:
        media[n_depositors:] = 1.0
    raw = params.credibility_base + params.credibility_media_bonus * media
    raw = raw + params.credibility_hub_scale * hub
    return xp.clip(raw, params.credibility_floor, params.credibility_cap)


def build_information_network(
    segments: Sequence[str],
    seed: int,
    params: NetworkParams | None = None,
) -> InformationNetwork:
    """Build the three layers for one depositor segment list and seed."""
    resolved = params if params is not None else load_network_params()
    xp, backend = select_backend()
    rng = _generator(xp, backend, int(seed))
    depositor_segments_list = list(segments)
    n_depositors = len(depositor_segments_list)
    n_media = resolved.media_count
    n = n_depositors + n_media
    full_segments = tuple(depositor_segments_list) + ("media",) * n_media

    m0 = min(n, max(resolved.seed_nodes, n_media, 2)) if n >= 2 else n
    src, dst = _preferential_edges(xp, rng, n, resolved.public_m, m0, resolved.batch_size)
    src, dst = _remap(xp, src, dst, n_media, n_depositors)
    public = _csr(xp, backend, src, dst, n, PUBLIC)

    labels = xp.full(n, _OTHER, dtype=xp.int8)
    founder_idx = _indices(full_segments, "founder")
    vc_idx = _indices(full_segments, "vc")
    treasurer_idx = _indices(full_segments, "treasurer")
    if founder_idx:
        labels[xp.asarray(founder_idx, dtype=xp.int32)] = _FOUNDER
    if vc_idx:
        labels[xp.asarray(vc_idx, dtype=xp.int32)] = _VC
    group_src, group_dst = _group_edges(
        xp, rng, n, labels, resolved.group_degree, resolved.homophily
    )
    group = _csr(xp, backend, group_src, group_dst, n, GROUP)

    media_idx = list(range(n_depositors, n))
    pool_list = founder_idx + vc_idx + media_idx
    if pool_list:
        pool = xp.asarray(pool_list, dtype=xp.int32)
        if int(pool.shape[0]) > resolved.advisor_count:
            perm = _permutation(xp, rng, int(pool.shape[0]))
            advisors = pool[perm[: resolved.advisor_count]]
        else:
            advisors = pool
    else:
        advisors = xp.zeros(0, dtype=xp.int32)
    treasurers = (
        xp.asarray(treasurer_idx, dtype=xp.int32)
        if treasurer_idx
        else xp.zeros(0, dtype=xp.int32)
    )
    pro_src, pro_dst = _professional_edges(
        xp, rng, treasurers, advisors, resolved.professional_degree
    )
    professional = _csr(xp, backend, pro_src, pro_dst, n, PROFESSIONAL)

    is_advisor = xp.zeros(n, dtype=bool)
    if int(advisors.shape[0]):
        is_advisor[advisors] = True
    followers = xp.bincount(public.indices, minlength=n)
    credibility = _credibility(xp, followers, n_depositors, n_media, resolved)
    channel_weight = xp.full(n, resolved.node_channel_weight, dtype=xp.float64)
    return InformationNetwork(
        segments=full_segments,
        n_depositors=n_depositors,
        credibility=credibility,
        channel_weight=channel_weight,
        is_advisor=is_advisor,
        layers={PUBLIC: public, GROUP: group, PROFESSIONAL: professional},
        layer_weight={
            PUBLIC: resolved.weight_public,
            GROUP: resolved.weight_group,
            PROFESSIONAL: resolved.weight_professional,
        },
        backend=backend,
        xp=xp,
        params=resolved,
    )


class NetworkModule:
    """Network builder (§19 M02). Publishes nothing. Consumes Tick."""

    publishes: ClassVar[tuple[str, ...]] = ()
    consumes: ClassVar[tuple[str, ...]] = ("Tick",)

    def __init__(self, module_id: str = "M02") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.params = load_network_params()
        self.network: InformationNetwork | None = None

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        """Bind the scenario and subscribe to Tick. One attach per run."""
        self.bus = bus
        self.scenario = scenario
        self.inbox = []
        self.network = None
        bus.subscribe("Tick", self)

    def ensure_network(self) -> InformationNetwork | None:
        """Build the layers if they are not built yet (SVB-57: M03 may ask first)."""
        if self.scenario is None:
            return None
        if self.network is None:
            segments = depositor_segments(self.scenario, self.params.population_tier)
            self.network = build_information_network(segments, self.scenario.seed, self.params)
        return self.network

    def step(self, t: float) -> list[BaseModel]:
        """Build the layers on the first tick. Later ticks publish nothing."""
        del t
        if self.scenario is None or self.network is not None:
            return []
        self.ensure_network()
        return []
