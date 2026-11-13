"""M03 attention and rumor engine (§5, §18, §19, ADR-028).

Array code goes through ``resolve_backend``: CuPy sparse matvec when a CUDA
device is available, NumPy otherwise. The GPU helpers at the bottom of this
file are imported only on that path, so CPU CI never loads CuPy. Stored
attention stays on the host. CPU draws use NumPy Philox, so a fixed seed
repeats.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

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
    Tick,
    TradingHalt,
    VIPDecision,
)
from contracts.schemas import Scenario
from mocks.bus import EventBus
from mocks.support import bank_id_for
from params.loader import RegistryError, list_by_module

# Registry names. Values live in params/registry.csv. Nothing numeric is inlined.
_MEMORY = "Attention memory window"
_TICK = "M03 sim tick minutes"
_MINUTES_PER_HOUR = "M03 minutes per hour"
_RUMOR_EXPOSURE = "M03 rumor exposure"
_HEADLINE_DIVISOR = "M03 headline reach divisor"
_THESIS_DIVISOR = "M03 thesis reach divisor"
_LIQUIDITY = "M03 liquidity drawn exposure"
_HALT = "M03 trading halt exposure"
_STATEMENT = "M03 statement exposure"
_CLOSURE = "M03 bank closed exposure"
_VIP = "M03 vip decision exposure"
_CONTEXT = "M03 context note credibility factor"
_SAMPLE = "M03 rumor mutator sample fraction"
_MAX_CALLS = "M03 rumor mutator max calls per rumor"
_OFFLINE_SCORE = "M03 offline mutator exaggeration"
_OFFLINE_SUFFIX = "M03 offline mutator suffix"
_GROK = "M03 grok mutator enabled"
_FALLBACK_NODES = "M03 fallback node count"
_FALLBACK_EDGE = "M03 fallback edge weight"
_PUBLIC_W = "M03 fallback public channel weight"
_GROUP_W = "M03 fallback group channel weight"
_PROFESSIONAL_W = "M03 fallback professional channel weight"
_NODE_CRED = "M03 default node credibility"
_REPOST_SCALE = "M03 repost attention scale"
# Flagged rumor amplifiers (ADR-093). Every flag defaults off; off means the
# arithmetic below is skipped, so logs are byte-identical to b674fa7.
_ROLE_MULT_ON = "M03 role exposure multiplier enabled"
_ROLE_MULT = "M03 role exposure multipliers"
_ROLE_NODE_ON = "M03 role source node enabled"
_ROLE_NODES = "M03 role source nodes"
_STRENGTH_ON = "M03 rumor strength impulse enabled"
_STRENGTH_MULT = "M03 rumor strength impulse multipliers"
_STRENGTH_LEVELS = frozenset({"low", "medium", "high"})
_EXAG_ON = "M03 exaggeration feedback enabled"
_EXAG_GAIN = "M03 exaggeration feedback gain"
_EXAG_CAP = "M03 exaggeration feedback cap"
_M02_NETWORK_ON = "M03 uses M02 network"
_CURE_ON = "M03 cure rumor enabled"
_CURE_CUT = "M03 cure attention cut"
_GPU_CALIB_ENV = ("STAMPEDE_CALIB_GPU_SIM", "STAMPEDE_CALIB_GPU_FULL")

_HOURS = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*hours?\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AttentionParams:
    """M03 inputs loaded from the parameter registry."""

    half_life_hours: float
    half_life_minutes: float
    tick_minutes: float
    window_ticks: int
    rumor_exposure: float
    headline_reach_divisor: float
    thesis_reach_divisor: float
    liquidity_exposure: float
    halt_exposure: float
    statement_exposure: float
    closure_exposure: float
    vip_exposure: float
    context_factor: float
    sample_fraction: float
    max_calls: int
    offline_exaggeration: float
    offline_suffix: str
    grok_enabled: bool
    fallback_nodes: int
    fallback_edge_weight: float
    public_weight: float
    group_weight: float
    professional_weight: float
    node_credibility: float
    repost_scale: float


@dataclass(frozen=True, slots=True)
class RumorAmplifierParams:
    """Flagged M03 rumor amplifiers (ADR-093). All flags default off.

    The poster role reaches M03 as ``RumorPosted.source_node``: the live UI
    passes the role id (``vc-partner``, ``journalist``, ...). A source that is
    not a key of a map below gets the old behaviour. §6 payloads are unchanged.
    """

    role_multiplier_enabled: bool = False
    role_multipliers: dict[str, float] | None = None
    role_node_enabled: bool = False
    role_nodes: dict[str, str] | None = None
    strength_enabled: bool = False
    strength_multipliers: dict[str, float] | None = None
    exaggeration_enabled: bool = False
    exaggeration_gain: float = 0.0
    exaggeration_cap: float = 1.0
    m02_network_enabled: bool = False

    def any_enabled(self) -> bool:
        return (
            self.role_multiplier_enabled
            or self.role_node_enabled
            or self.strength_enabled
            or self.exaggeration_enabled
            or self.m02_network_enabled
        )

    def exaggeration_factor(self, exaggeration: float) -> float:
        """Repost exposure factor ``min(1 + gain·exaggeration, cap)`` (ADR-095)."""
        if not self.exaggeration_enabled or exaggeration <= 0.0:
            return 1.0
        return min(1.0 + self.exaggeration_gain * exaggeration, self.exaggeration_cap)

    def role_multiplier(self, source: str) -> float:
        if not self.role_multiplier_enabled or not self.role_multipliers:
            return 1.0
        return self.role_multipliers.get(source, 1.0)

    def role_node(self, source: str) -> str | None:
        if not self.role_node_enabled or not self.role_nodes:
            return None
        return self.role_nodes.get(source)

    def strength_multiplier(self, claim: str) -> float:
        """Multiplier for the claim's lexicon strength (ADR-094); 1.0 when off."""
        if not self.strength_enabled or not self.strength_multipliers:
            return 1.0
        return self.strength_multipliers[claim_strength(claim)]


def claim_tone(claim: str) -> str:
    """Offline lexicon tone of a claim (positive / fear / neutral), the shared M34 rule."""
    from agents.rumor_lexicon import lexicon_tone_and_strength, normalize

    return lexicon_tone_and_strength(normalize(claim.strip()))[0]


def claim_strength(claim: str) -> str:
    """Offline lexicon strength of a claim: the same rule M34 classification uses.

    Deterministic and recomputable from the event log (the claim is in
    ``RumorPosted``). A live model judgment does not reach M03 (§6 unchanged).
    """
    from agents.rumor_lexicon import lexicon_tone_and_strength, normalize

    return lexicon_tone_and_strength(normalize(claim.strip()))[1]


@dataclass(frozen=True, slots=True)
class ArrayBackend:
    """NumPy or CuPy. ``name == "cupy"`` is the isolated GPU path."""

    name: Literal["numpy", "cupy"]
    xp: Any


@dataclass(frozen=True, slots=True)
class LayerCSR:
    """One information layer in CSR. Row i lists the nodes i hears."""

    name: str
    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    data: NDArray[np.float64]
    channel_weight: float


@dataclass(slots=True)
class LiveRumor:
    """Rumor state from §6.2, plus the nodes that currently hold it."""

    rumor_id: str
    claim: str
    source_node: str
    credibility: float
    reach: int
    lineage: list[str]
    notes: list[str]
    bank_id: str | None
    mutator_calls: int
    holders: NDArray[np.bool_]
    exposure: NDArray[np.float64]
    exaggeration: float = 0.0


def _cell(rows: dict[str, str], name: str) -> str:
    try:
        return rows[name]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {name}") from exc


def _parse_float(name: str, text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise RegistryError(f"{name} is not a float: {text!r}") from exc


def _parse_int(name: str, text: str) -> int:
    value = _parse_float(name, text)
    if not value.is_integer():
        raise RegistryError(f"{name} is not an integer: {text!r}")
    return int(value)


def _parse_hours(name: str, text: str) -> float:
    match = _HOURS.match(text)
    if match is None:
        raise RegistryError(f"{name} is not an hour count: {text!r}")
    return float(match.group(1))


def _parse_bool(name: str, text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise RegistryError(f"{name} is not a boolean: {text!r}")


def load_attention_params() -> AttentionParams:
    """Load M03 rows from ``params/registry.csv`` via ``params.loader``."""
    rows = {row.parameter: row.value for row in list_by_module("M03")}
    hours = _parse_hours(_MEMORY, _cell(rows, _MEMORY))
    minutes_per_hour = _parse_float(_MINUTES_PER_HOUR, _cell(rows, _MINUTES_PER_HOUR))
    tick = _parse_float(_TICK, _cell(rows, _TICK))
    if hours <= 0.0 or minutes_per_hour <= 0.0 or tick <= 0.0:
        raise RegistryError("attention half-life and tick length must be positive")
    half_life_minutes = hours * minutes_per_hour
    window = int(round(half_life_minutes / tick))
    if window < 1:
        raise RegistryError("attention half-life is shorter than one tick")
    sample = _parse_float(_SAMPLE, _cell(rows, _SAMPLE))
    context = _parse_float(_CONTEXT, _cell(rows, _CONTEXT))
    score = _parse_float(_OFFLINE_SCORE, _cell(rows, _OFFLINE_SCORE))
    suffix = _cell(rows, _OFFLINE_SUFFIX).strip()
    headline_div = _parse_float(_HEADLINE_DIVISOR, _cell(rows, _HEADLINE_DIVISOR))
    thesis_div = _parse_float(_THESIS_DIVISOR, _cell(rows, _THESIS_DIVISOR))
    nodes = _parse_int(_FALLBACK_NODES, _cell(rows, _FALLBACK_NODES))
    if not 0.0 <= sample <= 1.0:
        raise RegistryError("M03 rumor mutator sample fraction must be in [0, 1]")
    if not 0.0 <= context <= 1.0:
        raise RegistryError("M03 context note credibility factor must be in [0, 1]")
    if not 0.0 <= score <= 1.0:
        raise RegistryError("M03 offline mutator exaggeration must be in [0, 1]")
    if not suffix:
        raise RegistryError("M03 offline mutator suffix must be non-empty")
    if headline_div <= 0.0 or thesis_div <= 0.0:
        raise RegistryError("M03 reach divisors must be positive")
    if nodes < 4:
        raise RegistryError("M03 fallback node count must be at least 4")
    weights = (
        _parse_float(_RUMOR_EXPOSURE, _cell(rows, _RUMOR_EXPOSURE)),
        _parse_float(_LIQUIDITY, _cell(rows, _LIQUIDITY)),
        _parse_float(_HALT, _cell(rows, _HALT)),
        _parse_float(_STATEMENT, _cell(rows, _STATEMENT)),
        _parse_float(_CLOSURE, _cell(rows, _CLOSURE)),
        _parse_float(_VIP, _cell(rows, _VIP)),
        _parse_float(_FALLBACK_EDGE, _cell(rows, _FALLBACK_EDGE)),
        _parse_float(_PUBLIC_W, _cell(rows, _PUBLIC_W)),
        _parse_float(_GROUP_W, _cell(rows, _GROUP_W)),
        _parse_float(_PROFESSIONAL_W, _cell(rows, _PROFESSIONAL_W)),
        _parse_float(_NODE_CRED, _cell(rows, _NODE_CRED)),
        _parse_float(_REPOST_SCALE, _cell(rows, _REPOST_SCALE)),
    )
    if any(item < 0.0 for item in weights):
        raise RegistryError("M03 weights and exposures must be non-negative")
    return AttentionParams(
        half_life_hours=hours,
        half_life_minutes=half_life_minutes,
        tick_minutes=tick,
        window_ticks=window,
        rumor_exposure=weights[0],
        headline_reach_divisor=headline_div,
        thesis_reach_divisor=thesis_div,
        liquidity_exposure=weights[1],
        halt_exposure=weights[2],
        statement_exposure=weights[3],
        closure_exposure=weights[4],
        vip_exposure=weights[5],
        context_factor=context,
        sample_fraction=sample,
        max_calls=_parse_int(_MAX_CALLS, _cell(rows, _MAX_CALLS)),
        offline_exaggeration=score,
        offline_suffix=suffix,
        grok_enabled=_parse_bool(_GROK, _cell(rows, _GROK)),
        fallback_nodes=nodes,
        fallback_edge_weight=weights[6],
        public_weight=weights[7],
        group_weight=weights[8],
        professional_weight=weights[9],
        node_credibility=weights[10],
        repost_scale=weights[11],
    )


def _parse_pairs(name: str, text: str) -> list[tuple[str, str]]:
    """``key:value,key:value`` with unique, non-empty keys and values."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise RegistryError(f"{name} entry is not key:value: {piece!r}")
        key, value = (part.strip() for part in piece.split(":", 1))
        if not key or not value:
            raise RegistryError(f"{name} entry has an empty key or value: {piece!r}")
        if key in seen:
            raise RegistryError(f"{name} repeats key {key!r}")
        seen.add(key)
        pairs.append((key, value))
    return pairs


def _parse_multiplier_map(name: str, text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in _parse_pairs(name, text):
        number = _parse_float(name, value)
        if not math.isfinite(number) or number < 0.0:
            raise RegistryError(f"{name} value for {key!r} must be finite and non-negative")
        out[key] = number
    return out


def load_rumor_amplifiers() -> RumorAmplifierParams:
    """Load the flagged amplifier rows. Maps are parsed even when a flag is off."""
    rows = {row.parameter: row.value for row in list_by_module("M03")}
    strength = _parse_multiplier_map(_STRENGTH_MULT, _cell(rows, _STRENGTH_MULT))
    if set(strength) != _STRENGTH_LEVELS:
        raise RegistryError(f"{_STRENGTH_MULT} must have exactly low, medium and high")
    gain = _parse_float(_EXAG_GAIN, _cell(rows, _EXAG_GAIN))
    cap = _parse_float(_EXAG_CAP, _cell(rows, _EXAG_CAP))
    if not math.isfinite(gain) or gain < 0.0:
        raise RegistryError(f"{_EXAG_GAIN} must be finite and non-negative")
    if not math.isfinite(cap) or cap < 1.0:
        raise RegistryError(f"{_EXAG_CAP} must be finite and at least 1")
    return RumorAmplifierParams(
        role_multiplier_enabled=_parse_bool(_ROLE_MULT_ON, _cell(rows, _ROLE_MULT_ON)),
        role_multipliers=_parse_multiplier_map(_ROLE_MULT, _cell(rows, _ROLE_MULT)),
        role_node_enabled=_parse_bool(_ROLE_NODE_ON, _cell(rows, _ROLE_NODE_ON)),
        role_nodes=dict(_parse_pairs(_ROLE_NODES, _cell(rows, _ROLE_NODES))),
        strength_enabled=_parse_bool(_STRENGTH_ON, _cell(rows, _STRENGTH_ON)),
        strength_multipliers=strength,
        exaggeration_enabled=_parse_bool(_EXAG_ON, _cell(rows, _EXAG_ON)),
        exaggeration_gain=gain,
        exaggeration_cap=cap,
        m02_network_enabled=_parse_bool(_M02_NETWORK_ON, _cell(rows, _M02_NETWORK_ON)),
    )


def gpu_calibration_requested() -> bool:
    """True when the batched GPU calibration kernels are switched on."""
    return any(os.environ.get(name, "").strip() == "1" for name in _GPU_CALIB_ENV)


def check_amplifiers_cpu_only(amplifiers: RumorAmplifierParams) -> None:
    """The flags change this CPU engine only; gpu_sim/gpu_full do not read them."""
    if amplifiers.any_enabled() and gpu_calibration_requested():
        raise RegistryError(
            "M03 rumor amplifier flags are on, but engine/gpu_sim.py and engine/gpu_full.py "
            "do not read them; turn the flags off for --gpu-sim / --gpu-full runs (ADR-093)"
        )


def _host(array: object) -> np.ndarray:
    get = getattr(array, "get", None)
    return np.asarray(get() if callable(get) else array)


# Role -> M02 segment for the posting node when M03 runs on the M02 graph (SVB-57).
M02_ROLE_SEGMENTS: dict[str, str] = {
    "journalist": "media", "vip": "media", "regulator": "media", "short-seller": "media",
    "bank-ceo": "media", "major-shareholder": "media",
    "vc-partner": "vc", "founder-circle": "founder", "treasury-desk": "treasurer",
    "ui-user": "retail", "group-chat-anon": "retail",
}
_LOW_REACH_SEGMENTS = frozenset({"retail"})


def m02_layers(network: object, bank_id: str) -> tuple[list[LayerCSR], list[str], list[str], np.ndarray]:
    """M02's CSR layers as M03 layers (same row-hears-column convention), on the host."""
    layers = []
    for name, layer in network.layers.items():  # type: ignore[attr-defined]
        layers.append(
            LayerCSR(
                name,
                _host(layer.indptr).astype(np.int64),
                _host(layer.indices).astype(np.int64),
                _host(layer.data).astype(np.float64),
                float(network.layer_weight[name]),  # type: ignore[attr-defined]
            )
        )
    segments = list(network.segments)  # type: ignore[attr-defined]
    node_ids = [f"m02-{segment}-{index}" for index, segment in enumerate(segments)]
    return layers, node_ids, [bank_id] * len(node_ids), _host(network.credibility).astype(np.float64)  # type: ignore[attr-defined]


def m02_role_nodes(
    segments: Sequence[str], node_ids: Sequence[str], follower_score: np.ndarray
) -> dict[str, str]:
    """Role id -> node id on the M02 graph: the most-followed node of the role's segment,
    or for regular users and anonymous chats the least-followed retail node (lowest index on ties)."""
    by_segment: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        by_segment.setdefault(str(segment), []).append(index)
    chosen: dict[str, str] = {}
    for role, segment in M02_ROLE_SEGMENTS.items():
        members = by_segment.get(segment)
        if not members:
            continue
        scores = follower_score[members]
        pick = members[int(np.argmin(scores))] if segment in _LOW_REACH_SEGMENTS else members[int(np.argmax(scores))]
        chosen[role] = node_ids[pick]
    return chosen


def cuda_device_available() -> bool:
    """True only when CuPy imports and a CUDA device is visible."""
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def resolve_backend() -> ArrayBackend:
    """CuPy when CUDA is available and ``STAMPEDE_BACKEND`` isn't ``cpu`` (CPY-1), otherwise NumPy."""
    from engine.xp import backend_preference

    return _resolve_backend(backend_preference())


@lru_cache(maxsize=2)
def _resolve_backend(preference: str) -> ArrayBackend:
    if preference == "gpu" and cuda_device_available():
        import cupy as cp

        return ArrayBackend("cupy", cp)
    return ArrayBackend("numpy", np)


def _to_host(array: Any, backend: ArrayBackend) -> NDArray[np.float64]:
    if backend.name == "cupy":
        host = np.asarray(backend.xp.asnumpy(array), dtype=np.float64)
        return host
    host = np.asarray(array, dtype=np.float64)
    return host


def decay_multiplier(half_life_minutes: float, tick_minutes: float) -> float:
    """Per-tick decay whose half-life is ``half_life_minutes``.

    One half remains after one half-life. The one-half is that definition,
    not a registry knob.
    """
    if half_life_minutes <= 0.0 or tick_minutes <= 0.0:
        raise ValueError("half-life and tick length must be positive")
    return float(math.pow(0.5, tick_minutes / half_life_minutes))


def numpy_csr_matvec(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """CPU CSR matrix-vector product. Deterministic for the same arrays."""
    rows_n = int(indptr.shape[0]) - 1
    out = np.zeros(rows_n, dtype=np.float64)
    if rows_n <= 0 or int(indices.size) == 0:
        return out
    counts = np.diff(indptr)
    row_ids = np.repeat(np.arange(rows_n, dtype=np.int64), counts)
    contributed = np.asarray(data, dtype=np.float64) * np.asarray(vector, dtype=np.float64)[
        indices
    ]
    np.add.at(out, row_ids, contributed)
    return out


def csr_matvec(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    vector: NDArray[np.float64],
    *,
    backend: ArrayBackend,
) -> NDArray[np.float64]:
    """Sparse matvec on the active backend. Result is a host float64 vector."""
    if backend.name == "cupy":
        return _gpu_csr_matvec(indptr, indices, data, vector)
    return numpy_csr_matvec(
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
        np.asarray(vector, dtype=np.float64),
    )


def propagate(
    layers: Sequence[LayerCSR],
    exposure: NDArray[np.float64],
    *,
    backend: ArrayBackend,
) -> NDArray[np.float64]:
    """``Σ_layer w · A_layer · exposure``."""
    acc = np.zeros(int(exposure.shape[0]), dtype=np.float64)
    for layer in layers:
        spread = csr_matvec(layer.indptr, layer.indices, layer.data, exposure, backend=backend)
        acc = acc + float(layer.channel_weight) * spread
    return acc


def update_attention(
    attention: NDArray[np.float64],
    decay: float,
    layers: Sequence[LayerCSR],
    exposure: NDArray[np.float64],
    *,
    backend: ArrayBackend,
) -> NDArray[np.float64]:
    """``a ← decay(a) + Σ_layer w · A_layer · exposure``."""
    xp = backend.xp
    spread = propagate(layers, np.asarray(exposure, dtype=np.float64), backend=backend)
    decayed = xp.asarray(attention, dtype=np.float64) * float(decay)
    added = decayed + xp.asarray(spread, dtype=np.float64)
    return _to_host(added, backend)


def repost_probability(
    credibility: float,
    node_credibility: NDArray[np.float64],
    attention: NDArray[np.float64],
    scale: float,
    *,
    backend: ArrayBackend,
) -> NDArray[np.float64]:
    """Clip credibility × node credibility × attention × scale into [0, 1]."""
    xp = backend.xp
    raw = (
        float(credibility)
        * xp.asarray(node_credibility, dtype=np.float64)
        * xp.asarray(attention, dtype=np.float64)
        * float(scale)
    )
    return _to_host(xp.clip(raw, 0.0, 1.0), backend)


def select_mutator_indices(
    repost_nodes: NDArray[np.int64],
    rate: float,
    budget: int,
    draws: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Keep reposts whose draw is below ``rate``, up to the remaining call budget."""
    if budget <= 0 or int(repost_nodes.size) == 0 or rate <= 0.0:
        return np.zeros(0, dtype=np.int64)
    if int(draws.shape[0]) != int(repost_nodes.shape[0]):
        raise ValueError("mutator draws must match repost count")
    chosen = repost_nodes[draws < rate]
    limit = min(int(budget), int(chosen.shape[0]))
    return np.asarray(chosen[:limit], dtype=np.int64)


def _accept_variant(original: str, variant: str, score: float) -> bool:
    if not variant.strip() or variant == original:
        return False
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        return False
    return True


def mutate_claim(
    claim: str,
    *,
    seed: int,
    enabled: bool,
    suffix: str,
    score: float,
    cache: dict[str, tuple[str, float]],
) -> tuple[str, float]:
    """Return a cached variant and exaggeration score.

    A Grok suggestion is used only when it validates. Otherwise the offline
    rewrite is the variant. The cache key is the seed plus the original claim.
    """
    key = f"{seed}:{claim}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    suggestion = grok_mutate_suggestion(claim, enabled=enabled)
    if suggestion is not None and _accept_variant(claim, suggestion[0], suggestion[1]):
        cache[key] = suggestion
        return suggestion
    variant = f"{claim} {suffix}".strip()
    if not _accept_variant(claim, variant, score):
        raise RuntimeError("offline mutator produced a variant that failed validation")
    result = (variant, float(score))
    cache[key] = result
    return result


def grok_mutate_suggestion(claim: str, *, enabled: bool) -> tuple[str, float] | None:
    """Isolated Grok seam (§7). Suggestions only. None selects the offline rewrite.

    Disabled, or no SDK, returns None. CPU CI leaves the flag off, so this
    does not import the SDK and does not make a network call.
    """
    if not enabled:
        return None
    return _grok_mutate_live(claim)


def csr_from_edges(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    weight: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Build a CSR matrix. Columns in each row are sorted."""
    buckets: list[list[int]] = [[] for _ in range(n_nodes)]
    for row, col in edges:
        buckets[row].append(col)
    indptr_list = [0]
    indices_list: list[int] = []
    for row in range(n_nodes):
        indices_list.extend(sorted(buckets[row]))
        indptr_list.append(len(indices_list))
    indptr = np.asarray(indptr_list, dtype=np.int64)
    indices = np.asarray(indices_list, dtype=np.int64)
    data = np.full(int(indices.shape[0]), float(weight), dtype=np.float64)
    return indptr, indices, data


def follower_scores(layers: Sequence[LayerCSR], n_nodes: int) -> NDArray[np.float64]:
    """Column sums across layers, weighted by channel weight. Higher means more heard."""
    scores = np.zeros(n_nodes, dtype=np.float64)
    for layer in layers:
        if int(layer.indices.size) == 0:
            continue
        weighted = np.asarray(layer.data, dtype=np.float64) * float(layer.channel_weight)
        np.add.at(scores, layer.indices, weighted)
    return scores


def fallback_layers(
    params: AttentionParams,
    bank_id: str,
) -> tuple[list[LayerCSR], list[str], list[str]]:
    """Deterministic stand-in used until M02 supplies CSR layers (ADR-028).

    Node ``hub-public-1`` is the public hub (the M28 stand-in source).
    Node ``small-account`` has a single public follower.
    """
    n = params.fallback_nodes
    node_ids = [f"node-{index}" for index in range(n)]
    node_ids[0] = "hub-public-1"
    node_ids[-1] = "small-account"
    banks = [bank_id for _ in range(n)]
    weight = params.fallback_edge_weight
    public_edges = [(follower, 0) for follower in range(1, n - 1)]
    public_edges.append((1, n - 1))
    group_edges = [(index, (index + 1) % n) for index in range(n)]
    professional_edges = [(index, 2) for index in range(n) if index % 2 == 0 and index != 2]
    layers = [
        _layer("public", n, public_edges, weight, params.public_weight),
        _layer("groups", n, group_edges, weight, params.group_weight),
        _layer("professional", n, professional_edges, weight, params.professional_weight),
    ]
    return layers, node_ids, banks


def _layer(
    name: str,
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    edge_weight: float,
    channel_weight: float,
) -> LayerCSR:
    indptr, indices, data = csr_from_edges(n_nodes, edges, edge_weight)
    return LayerCSR(name, indptr, indices, data, channel_weight)


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _nonnegative(value: float) -> float:
    if value < 0.0:
        return 0.0
    return float(value)


class AttentionEngine:
    """Attention propagator and rumor spreader (§19 M03)."""

    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "RumorPosted",
        "HeadlinePublished",
        "AnalystNote",
        "LiquidityDrawn",
        "TradingHalt",
        "StatementIssued",
        "BankClosed",
        "VIPDecision",
        "ThesisPublished",
    )
    publishes: ClassVar[tuple[str, ...]] = ("AttentionTick", "RumorMutated")

    def __init__(self, module_id: str = "M03") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.params = load_attention_params()
        self.amplifiers = load_rumor_amplifiers()
        check_amplifiers_cpu_only(self.amplifiers)
        # SVB-57: build_engine sets this to the real M02 module; used only when
        # ``M03 uses M02 network`` is on.
        self.network_source: object | None = None
        self._m02_installed = False
        self._m02_role_nodes: dict[str, str] = {}
        # SVB-59: a positive-tone post lowers this bank's attention instead of spreading.
        from engine.live_overrides import flag_enabled, refuse_under_gpu_calibration

        refuse_under_gpu_calibration(_CURE_ON)
        self.cure_enabled = flag_enabled(_CURE_ON)
        self.backend = resolve_backend()
        self._decay = decay_multiplier(self.params.half_life_minutes, self.params.tick_minutes)
        self._layers: list[LayerCSR] | None = None
        self._node_ids: list[str] = []
        self._index: dict[str, int] = {}
        self._node_banks: list[str] = []
        self._node_credibility = np.zeros(0, dtype=np.float64)
        self.attention = np.zeros(0, dtype=np.float64)
        self._follower_score = np.zeros(0, dtype=np.float64)
        self._hub = 0
        self._rumors: list[LiveRumor] = []
        self._history: dict[str, list[float]] = {}
        self._cache: dict[str, tuple[str, float]] = {}
        self._rng = np.random.Generator(np.random.Philox(0))
        self._gpu: _GpuRandom | None = None
        self._cursor = 0
        self._seq = 0
        self._subscribed = False

    def set_layers(
        self,
        layers: Sequence[LayerCSR],
        node_ids: Sequence[str],
        node_banks: Sequence[str],
        node_credibility: Sequence[float] | None = None,
    ) -> None:
        """Install CSR layers. Lengths must match. Attention resets to zero."""
        if len(node_ids) != len(node_banks):
            raise ValueError("node_ids and node_banks must have the same length")
        if len(node_ids) == 0:
            raise ValueError("attention graph needs at least one node")
        n_nodes = len(node_ids)
        for layer in layers:
            if int(layer.indptr.shape[0]) != n_nodes + 1:
                raise ValueError(f"layer {layer.name} indptr does not match node count")
        if node_credibility is None:
            cred = np.full(n_nodes, self.params.node_credibility, dtype=np.float64)
        else:
            cred = np.asarray(list(node_credibility), dtype=np.float64)
            if int(cred.shape[0]) != n_nodes:
                raise ValueError("node_credibility length must match nodes")
        self._layers = list(layers)
        self._node_ids = list(node_ids)
        self._index = {name: index for index, name in enumerate(self._node_ids)}
        self._node_banks = list(node_banks)
        self._node_credibility = cred
        self.attention = np.zeros(n_nodes, dtype=np.float64)
        self._follower_score = follower_scores(self._layers, n_nodes)
        self._hub = int(np.argmax(self._follower_score))
        self._history = {}
        self._rumors = []

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        """Bind the scenario, reset draws, and subscribe to consumed events."""
        self.bus = bus
        self.scenario = scenario
        self._rng = np.random.Generator(np.random.Philox(int(scenario.seed)))
        self._gpu = _GpuRandom(int(scenario.seed)) if self.backend.name == "cupy" else None
        self._cache = {}
        self._seq = 0
        self._cursor = 0
        if self._layers is None:
            layers, node_ids, banks = fallback_layers(self.params, bank_id_for(scenario.preset))
            self.set_layers(layers, node_ids, banks)
        else:
            self.attention = np.zeros(self.attention.shape[0], dtype=np.float64)
            self._history = {}
            self._rumors = []
        if not self._subscribed:
            for event_type in self.consumes:
                bus.subscribe(event_type, self)
            self._subscribed = True

    def _maybe_install_m02(self) -> None:
        """Swap the fallback graph for M02's network once (flag on, before any event is read)."""
        if self._m02_installed or not self.amplifiers.m02_network_enabled or self.network_source is None:
            return
        ensure = getattr(self.network_source, "ensure_network", None)
        network = ensure() if callable(ensure) else getattr(self.network_source, "network", None)
        if network is None or self.scenario is None:
            return
        layers, node_ids, banks, cred = m02_layers(network, bank_id_for(self.scenario.preset))
        self.set_layers(layers, node_ids, banks, cred)
        self._m02_role_nodes = m02_role_nodes(list(network.segments), self._node_ids, self._follower_score)
        self._m02_installed = True

    def step(self, t: float) -> list[BaseModel]:
        """Advance one minute-tick. No scenario → no events."""
        self._maybe_install_m02()
        scenario = self.scenario
        layers = self._layers
        if scenario is None or layers is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        shock = np.zeros(self.attention.shape[0], dtype=np.float64)
        posted: list[tuple[LiveRumor, int, float]] = []
        cures: list[RumorPosted] = []
        for event in new_events:
            if isinstance(event, RumorPosted) and self.cure_enabled and claim_tone(event.claim) == "positive":
                cures.append(event)
            elif isinstance(event, RumorPosted):
                posted.append(
                    (
                        self._rumor_for(event),
                        self._source_index(event.source_node),
                        self._post_scale(event),
                    )
                )
            elif isinstance(event, AnalystNote):
                self._apply_note(event)
            elif isinstance(event, Tick):
                continue
            else:
                amount = self._shock_amount(event)
                if amount is None:
                    continue
                bank_value = getattr(event, "bank_id", None)
                bank = bank_value if isinstance(bank_value, str) else None
                shock[self._hub_for(bank)] += amount
        impulse = self.params.rumor_exposure
        for rumor, src, scale in posted:
            amount = rumor.credibility * impulse
            if scale != 1.0:
                amount = amount * scale
            rumor.exposure[src] += amount
            rumor.holders[src] = True
        mutated = self._draw_reposts(t, scenario.seed)
        combined = shock.copy()
        for rumor in self._rumors:
            combined = combined + rumor.exposure
            spread = propagate(layers, rumor.exposure, backend=self.backend)
            rumor.holders = np.logical_or(rumor.holders, spread > 0.0)
            rumor.reach = int(np.count_nonzero(rumor.holders))
            rumor.exposure.fill(0.0)
        self.attention = update_attention(
            self.attention,
            self._decay,
            layers,
            combined,
            backend=self.backend,
        )
        if cures:
            self._apply_cures(cures)
        return [*mutated, *self._attention_ticks(t, scenario.seed)]

    def _apply_cures(self, cures: list[RumorPosted]) -> None:
        """Scale the bank's node attention by ``1 - cut * credibility`` per cure (floor 0)."""
        from params.loader import get_parameter

        cut = float(get_parameter(_CURE_CUT).value)
        banks = np.asarray(self._node_banks)
        for event in cures:
            mask = banks == event.bank_id
            if not mask.any():
                mask = np.ones(banks.shape[0], dtype=np.bool_)
            factor = max(0.0, 1.0 - cut * float(event.credibility))
            self.attention[mask] = self.attention[mask] * factor

    def _rumor_for(self, event: RumorPosted) -> LiveRumor:
        for rumor in self._rumors:
            if rumor.rumor_id == event.rumor_id:
                rumor.credibility = _clip01(event.credibility)
                if event.bank_id is not None:
                    rumor.bank_id = event.bank_id
                return rumor
        n_nodes = int(self.attention.shape[0])
        rumor = LiveRumor(
            rumor_id=event.rumor_id,
            claim=event.claim,
            source_node=event.source_node,
            credibility=_clip01(event.credibility),
            reach=0,
            lineage=[],
            notes=[],
            bank_id=event.bank_id,
            mutator_calls=0,
            holders=np.zeros(n_nodes, dtype=np.bool_),
            exposure=np.zeros(n_nodes, dtype=np.float64),
        )
        self._rumors.append(rumor)
        return rumor

    def _apply_note(self, event: AnalystNote) -> None:
        factor = self.params.context_factor
        for rumor in self._rumors:
            if event.bank_id is not None and rumor.bank_id != event.bank_id:
                continue
            rumor.credibility = _clip01(rumor.credibility * factor)
            rumor.notes.append(event.text)

    def _shock_amount(self, event: BaseModel) -> float | None:
        params = self.params
        if isinstance(event, HeadlinePublished):
            return _nonnegative(float(event.reach)) / params.headline_reach_divisor
        if isinstance(event, ThesisPublished):
            return _nonnegative(float(event.reach)) / params.thesis_reach_divisor
        if isinstance(event, LiquidityDrawn):
            return params.liquidity_exposure
        if isinstance(event, TradingHalt):
            return params.halt_exposure
        if isinstance(event, StatementIssued):
            return params.statement_exposure
        if isinstance(event, BankClosed):
            return params.closure_exposure
        if isinstance(event, VIPDecision):
            return params.vip_exposure
        return None

    def _draw_reposts(self, t: float, seed: int) -> list[RumorMutated]:
        out: list[RumorMutated] = []
        n_nodes = int(self.attention.shape[0])
        for rumor in self._rumors:
            probs = repost_probability(
                rumor.credibility,
                self._node_credibility,
                self.attention,
                self.params.repost_scale,
                backend=self.backend,
            )
            masked = np.where(rumor.holders, probs, 0.0)
            draws = self._uniform(n_nodes)
            repost_nodes = np.asarray(np.flatnonzero(draws < masked), dtype=np.int64)
            if int(repost_nodes.size) == 0:
                continue
            mutator_draws = self._uniform(int(repost_nodes.size))
            budget = self.params.max_calls - rumor.mutator_calls
            picked = select_mutator_indices(
                repost_nodes,
                self.params.sample_fraction,
                budget,
                mutator_draws,
            )
            for _node in picked:
                variant, score = mutate_claim(
                    rumor.claim,
                    seed=int(seed),
                    enabled=self.params.grok_enabled,
                    suffix=self.params.offline_suffix,
                    score=self.params.offline_exaggeration,
                    cache=self._cache,
                )
                rumor.mutator_calls += 1
                rumor.lineage.append(variant)
                if self.amplifiers.exaggeration_enabled:
                    rumor.exaggeration = max(rumor.exaggeration, float(score))
                out.append(
                    RumorMutated(
                        t=t,
                        seed=int(seed),
                        bank_id=rumor.bank_id,
                        cause_id=self._cause(f"mut-{rumor.rumor_id}"),
                        rumor_id=rumor.rumor_id,
                        claim=variant,
                        exaggeration_score=score,
                    )
                )
            amount = rumor.credibility * self.params.rumor_exposure
            if self.amplifiers.exaggeration_enabled:
                factor = self.amplifiers.exaggeration_factor(rumor.exaggeration)
                if factor != 1.0:
                    amount = amount * factor
            rumor.exposure[repost_nodes] += amount
        return out

    def _attention_ticks(self, t: float, seed: int) -> list[AttentionTick]:
        ticks: list[AttentionTick] = []
        seen: list[str] = []
        for bank in self._node_banks:
            if bank in seen:
                continue
            seen.append(bank)
            idxs = [index for index, name in enumerate(self._node_banks) if name == bank]
            level = float(np.mean(self.attention[np.asarray(idxs, dtype=np.int64)]))
            hist = self._history.setdefault(bank, [])
            if len(hist) >= self.params.window_ticks:
                reference = hist[-self.params.window_ticks]
            else:
                reference = 0.0
            hist.append(level)
            ticks.append(
                AttentionTick(
                    t=t,
                    seed=int(seed),
                    bank_id=bank,
                    cause_id=self._cause(f"attn-{bank}"),
                    level=level,
                    delta_4h=level - reference,
                )
            )
        return ticks

    def _source_index(self, source: str) -> int:
        """Posting node. With the role-node flag, a role id maps to a graph node."""
        if self._m02_installed:
            node = self._m02_role_nodes.get(source) if self.amplifiers.role_node_enabled else None
            return self._index[node] if node is not None else self._node_index(source)
        mapped = self.amplifiers.role_node(source)
        if mapped is not None and mapped in self._index:
            return self._index[mapped]
        return self._node_index(source)

    def _post_scale(self, event: RumorPosted) -> float:
        """Product of the enabled post-impulse multipliers; 1.0 when all are off."""
        scale = self.amplifiers.role_multiplier(event.source_node)
        if self.amplifiers.strength_enabled:
            scale = scale * self.amplifiers.strength_multiplier(event.claim)
        return scale

    def _node_index(self, name: str) -> int:
        found = self._index.get(name)
        if found is None:
            return self._hub
        return found

    def _hub_for(self, bank_id: str | None) -> int:
        if bank_id is None:
            return self._hub
        idxs = [index for index, bank in enumerate(self._node_banks) if bank == bank_id]
        if not idxs:
            return self._hub
        local = self._follower_score[np.asarray(idxs, dtype=np.int64)]
        return idxs[int(np.argmax(local))]

    def _uniform(self, n_draws: int) -> NDArray[np.float64]:
        if n_draws <= 0:
            return np.zeros(0, dtype=np.float64)
        if self.backend.name == "cupy":
            gpu = self._gpu
            if gpu is None:
                raise RuntimeError("CuPy backend has no random stream")
            return gpu.random(n_draws)
        return np.asarray(self._rng.random(n_draws), dtype=np.float64)

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"


# --- GPU path (isolated) ----------------------------------------------------
# Imported only when resolve_backend() selects CuPy. CPU CI does not enter
# these functions. The matvec is cupyx.scipy.sparse, the §18 CuPy product.


class _GpuRandom:
    """CuPy random stream. One stream per attached run, seeded like Philox."""

    def __init__(self, seed: int) -> None:
        import cupy as cp

        self._rs = cp.random.RandomState(int(seed))

    def random(self, n_draws: int) -> NDArray[np.float64]:
        import cupy as cp

        return _to_host(self._rs.rand(int(n_draws)), ArrayBackend("cupy", cp))


def _gpu_csr_matvec(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GPU path. CuPy sparse CSR matrix-vector product."""
    import cupy as cp
    import cupyx.scipy.sparse as sparse

    n_rows = int(indptr.shape[0]) - 1
    if n_rows <= 0 or int(indices.size) == 0:
        return np.zeros(max(n_rows, 0), dtype=np.float64)
    matrix = sparse.csr_matrix(
        (
            cp.asarray(data, dtype=cp.float64),
            cp.asarray(indices, dtype=cp.int32),
            cp.asarray(indptr, dtype=cp.int32),
        ),
        shape=(n_rows, int(vector.shape[0])),
    )
    product = matrix.dot(cp.asarray(vector, dtype=cp.float64))
    return np.asarray(cp.asnumpy(product), dtype=np.float64)


def _grok_mutate_live(claim: str) -> tuple[str, float] | None:
    """Live Grok or local HTTP suggestion. Isolated from the NumPy path."""
    if not claim.strip():
        return None
    from agents.llm_cache import (
        complete_json_llm,
        effective_cache_model,
        live_json_completion_enabled,
        record_local_validation_failure,
    )
    from params.loader import get_parameter

    if not live_json_completion_enabled():
        return None
    agent_model = get_parameter("M04 headline writer Grok model").value
    prompt = json.dumps(
        {
            "task": "rumor_mutator",
            "claim": claim,
            "output_schema": {"variant": "string", "exaggeration_score": "float 0..1"},
        },
        sort_keys=True,
    )
    schema = {
        "type": "object",
        "properties": {
            "variant": {"type": "string"},
            "exaggeration_score": {"type": "number"},
        },
        "required": ["variant", "exaggeration_score"],
        "additionalProperties": False,
    }
    model = effective_cache_model(agent_model)
    for attempt in range(2):
        try:
            raw = complete_json_llm(prompt, model, json_schema=schema)
            variant = str(raw.get("variant", ""))
            score = float(raw.get("exaggeration_score", 0.0))
            if _accept_variant(claim, variant, score):
                return variant, score
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            if attempt == 1:
                record_local_validation_failure()
    return None
