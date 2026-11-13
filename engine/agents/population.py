"""M05 depositor population and insurance — Wave B real implementation."""

from __future__ import annotations

import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from contracts.schemas import DepositorArrays, Scenario
from mocks.support import SEGMENT_WEIGHTS, bank_id_for, deposits_base, is_community, uninsured_share
from params.loader import RegistryError, get_parameter

# Segment → FDIC ownership categories (min(balance, limit × categories)).
_OWNERSHIP_CATEGORIES: dict[str, int] = {
    "startup": 1,
    "vc": 1,
    "founder": 1,
    "treasurer": 1,
    "retail": 1,
    "municipal": 2,
    "crypto": 1,
}

_LOGNORMAL_MU = 11.0
_LOGNORMAL_SIGMA = 1.35
_SECOND_BANK_RATE = 0.28
_BELIEF_BASE = 0.05
_THRESHOLD_BY_SEGMENT: dict[str, float] = {
    "startup": 0.42,
    "vc": 0.38,
    "founder": 0.40,
    "treasurer": 0.45,
    "retail": 0.62,
    "municipal": 0.55,
    "crypto": 0.36,
}


def parse_registry_number(name: str) -> float:
    """Parse a numeric value from the parameter registry (§8.1)."""
    try:
        row = get_parameter(name)
    except RegistryError as exc:
        raise KeyError(str(exc)) from exc
    parsed = _parse_human(row.value)
    if parsed is None:
        raise ValueError(f"cannot parse registry value for {name}: {row.value!r}")
    return parsed


def _parse_human(text: str) -> float | None:
    lowered = text.strip().lower().replace(",", "").replace("$", "").replace("±", "")
    match = re.match(r"[><=\s]*([0-9]*\.?[0-9]+)\s*([a-z%]*)", lowered)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "k" or unit.startswith("k"):
        return number * 1_000.0
    if unit.startswith("b"):
        return number * 1_000_000_000.0
    if unit.startswith("m"):
        return number * 1_000_000.0
    if unit.startswith("%") or "percent" in lowered or "point" in lowered:
        return number / 100.0
    if "hour" in lowered:
        return number
    if "per depositor" in lowered or "depositor" in lowered:
        return number
    return number


def segment_weights_for_preset(preset: str) -> dict[str, float]:
    """Return normalized segment shares for ``preset`` (crypto only on Signature-like presets)."""
    weights = dict(SEGMENT_WEIGHTS)
    if "crypto" not in preset.lower():
        crypto = weights.pop("crypto", 0.0)
        weights["startup"] = weights.get("startup", 0.0) + crypto
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def insured_amount(balance: float, segment: str, insurance_limit: float) -> float:
    categories = _OWNERSHIP_CATEGORIES.get(segment, 1)
    return min(balance, insurance_limit * categories)


def _allocate_segments(
    rng: random.Random, count: int, weights: dict[str, float]
) -> list[str]:
    names = list(weights)
    probs = [weights[name] for name in names]
    draws = [rng.random() for _ in range(count)]
    cumulative = 0.0
    edges = []
    for prob in probs:
        cumulative += prob
        edges.append(cumulative)
    segments: list[str] = []
    for draw in draws:
        for name, edge in zip(names, edges, strict=True):
            if draw <= edge:
                segments.append(name)
                break
        else:
            segments.append(names[-1])
    return segments


def _lognormal_draws(z_scores: list[float]) -> list[float]:
    return [math.exp(_LOGNORMAL_MU + _LOGNORMAL_SIGMA * z) for z in z_scores]


def _shape_balances(raw: list[float], spread: float) -> list[float]:
    """Raise balances to ``spread`` to steer insured share without changing the count."""
    return [max(value, 1.0) ** spread for value in raw]


def _apply_tail_boost(balances: list[float], tail_log: float) -> list[float]:
    """Boost the ten largest balances; ``tail_log`` is bisected in log-space."""
    if not balances:
        return []
    multiplier = math.exp(tail_log)
    indexed = sorted(enumerate(balances), key=lambda item: item[1], reverse=True)
    boosted = list(balances)
    for rank, (index, _value) in enumerate(indexed[:10]):
        weight = 1.0 + (10 - rank) / 10.0
        boosted[index] *= 1.0 + (multiplier - 1.0) * weight
    return boosted


def _scale_to_total_array(balances: np.ndarray, total: float) -> np.ndarray:
    current = float(balances.sum())
    if current <= 0.0:
        return balances
    return balances * (total / current)


def _caps_for_segments_array(segments: list[str], insurance_limit: float) -> np.ndarray:
    return np.asarray(
        [insurance_limit * _OWNERSHIP_CATEGORIES.get(segment, 1) for segment in segments],
        dtype=np.float64,
    )


def _insured_total_array(balances: np.ndarray, caps: np.ndarray) -> float:
    return float(np.minimum(balances, caps).sum())


# One-time setup: host refine matches the CPU book. Device scalar
# reads inside the donor loop stall at 100k depositors (ADR-059).
_HOST_REFINE_MIN = 64


def _as_host_array(array: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Copy a device or host array to NumPy. Prefer ``.get()`` over ``np.asarray``."""
    getter = getattr(array, "get", None)
    if callable(getter):
        host = np.asarray(getter())
    else:
        host = np.asarray(array)
    if dtype is not None:
        return np.asarray(host, dtype=dtype)
    return host


def _array_module(array: Any) -> Any:
    """NumPy or CuPy already backing ``array``; never copies."""
    module_name = type(array).__module__.split(".", 1)[0]
    if module_name == "cupy":
        loaded = sys.modules.get("cupy")
        if loaded is not None:
            return loaded
        import cupy as cp

        return cp
    return np


def _as_xp_array(values: Any, xp: Any, *, dtype: Any | None = None) -> Any:
    if dtype is not None:
        return xp.asarray(values, dtype=dtype)
    if getattr(values, "ndim", None) is not None:
        value_mod = type(values).__module__.split(".", 1)[0]
        xp_name = getattr(xp, "__name__", "")
        if (value_mod == "cupy" and xp_name == "cupy") or (
            value_mod != "cupy" and xp_name != "cupy"
        ):
            return values
    return xp.asarray(values)


def _remaining_before_donors(
    room: Any,
    amount: Any,
    donor_count: int,
    constrained: Any,
    xp: Any,
) -> Any:
    """Remaining just before each donor; same recurrence as the sequential loop."""
    amount_f = xp.asarray(amount, dtype=xp.float64)
    if donor_count <= 1:
        return xp.full((max(donor_count, 0),), amount_f, dtype=xp.float64)
    n_f = xp.asarray(donor_count, dtype=xp.float64)
    one = xp.asarray(1.0, dtype=xp.float64)
    zero = xp.asarray(0.0, dtype=xp.float64)
    alpha = (n_f - one) / n_f
    scale = xp.where(constrained, one, alpha)
    inject = xp.where(constrained, -room, zero)
    prefix = xp.empty(donor_count, dtype=xp.float64)
    prefix[0] = one
    prefix[1:] = xp.cumprod(scale[:-1])
    contrib = inject / (prefix * scale)
    extra = xp.cumsum(contrib)
    extra_before = xp.empty(donor_count, dtype=xp.float64)
    extra_before[0] = zero
    extra_before[1:] = extra[:-1]
    return prefix * (amount_f + extra_before)


def _scatter_sub(target: Any, indices: Any, values: Any, xp: Any) -> None:
    """Deterministic sort + segmented cumsum; one write per index, no atomics."""
    count = int(indices.size)
    if count == 0:
        return
    positions = xp.arange(count, dtype=xp.int64)
    packed = indices.astype(xp.int64) * xp.int64(count) + positions
    order = xp.argsort(packed)
    ordered_idx = indices[order]
    ordered_val = values[order]
    totals = xp.cumsum(ordered_val)
    last = xp.ones(count, dtype=bool)
    if count > 1:
        last[:-1] = ordered_idx[1:] != ordered_idx[:-1]
    ended = totals[last]
    prev = xp.empty(ended.size, dtype=ended.dtype)
    prev[0] = xp.asarray(0.0, dtype=ended.dtype)
    if int(ended.size) > 1:
        prev[1:] = ended[:-1]
    target[ordered_idx[last]] = target[ordered_idx[last]] - (ended - prev)


def _transfer_mass_array(
    balances: np.ndarray,
    donors: np.ndarray,
    receivers: np.ndarray,
    amount: float,
) -> None:
    xp = _array_module(balances)
    donors = _as_xp_array(donors, xp)
    receivers = _as_xp_array(receivers, xp)
    if amount <= 0.0 or donors.size == 0 or receivers.size == 0:
        return
    donor_count = int(donors.size)
    n_f = xp.asarray(max(donor_count, 1), dtype=xp.float64)
    amount_f = xp.asarray(amount, dtype=xp.float64)
    room = xp.maximum(
        xp.asarray(balances[donors], dtype=xp.float64) - xp.asarray(1.0, dtype=xp.float64),
        xp.asarray(0.0, dtype=xp.float64),
    )
    constrained = xp.zeros(donor_count, dtype=bool)
    remaining = _remaining_before_donors(room, amount_f, donor_count, constrained, xp)
    for _ in range(donor_count):
        nxt = room < (remaining / n_f)
        if bool(xp.all(nxt == constrained)):
            break
        constrained = nxt
        remaining = _remaining_before_donors(room, amount_f, donor_count, constrained, xp)
    takes = xp.minimum(room, remaining / n_f)
    _scatter_sub(balances, donors, takes, xp)
    leftover = amount_f - takes.sum()
    balances[receivers] = balances[receivers] + leftover / xp.asarray(
        int(receivers.size), dtype=xp.float64
    )


def _transfer_mass_cupy(
    balances: Any,
    donors: Any,
    receivers: Any,
    amount: float,
) -> None:
    """Same sequential donor loop as ``_transfer_mass_array``; one host round-trip."""
    if amount <= 0.0 or donors.size == 0 or receivers.size == 0:
        return
    host = np.array(_as_host_array(balances, dtype=np.float64), copy=True)
    donor_idx = _as_host_array(donors, dtype=np.intp)
    receiver_idx = _as_host_array(receivers, dtype=np.intp)
    _transfer_mass_array(host, donor_idx, receiver_idx, amount)
    balances[:] = host


def prefer_population_gpu() -> bool:
    """CuPy balance solve unless ``STAMPEDE_BACKEND=cpu`` (CPY-1: on by default; same switch as the server)."""
    from engine.xp import backend_preference

    return backend_preference() == "gpu"


def _cupy_runtime() -> Any | None:
    try:
        import cupy as cp
    except ImportError:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
    except Exception:
        return None
    return cp


def population_balance_backend() -> str:
    """``gpu`` when CuPy balance solve is active, else ``cpu`` (NumPy reference path)."""
    if prefer_population_gpu() and _cupy_runtime() is not None:
        return "gpu"
    return "cpu"


def population_device_description() -> str:
    """Human-readable device for calibration logs."""
    cp = _cupy_runtime()
    if not prefer_population_gpu() or cp is None:
        return "cpu"
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return f"cuda:0 {name}"
    except Exception:
        return "cuda:0"


def _refine_balances_numpy(
    balances: list[float],
    segments: list[str],
    total_deposits: float,
    insurance_limit: float,
    target_insured_share: float,
    target_top10: float,
) -> list[float]:
    """Adjust balances toward preset insured share and top-10 concentration (NumPy)."""
    caps = _caps_for_segments_array(segments, insurance_limit)
    current = np.maximum(np.asarray(balances, dtype=np.float64), 1.0)
    count = int(current.size)
    if count < 10:
        return _scale_to_total_array(current, total_deposits).tolist()

    for _ in range(96):
        current = _scale_to_total_array(current, total_deposits)
        ranked = np.argsort(current, kind="mergesort")
        top_idx = ranked[-10:]
        top_sum = float(current[top_idx].sum())
        if top_sum > 0.0:
            current[top_idx] *= target_top10 / top_sum
        current = _scale_to_total_array(current, total_deposits)

        insured_share = _insured_total_array(current, caps) / total_deposits
        gap = (target_insured_share - insured_share) * total_deposits
        if abs(gap) < total_deposits * 0.0005:
            break
        shift = min(abs(gap) * 0.5, total_deposits * 0.03)
        if gap > 0.0:
            receivers = ranked[current[ranked] < caps[ranked]]
            donors = ranked[count // 3 :]
            _transfer_mass_array(current, donors, receivers, shift)
        else:
            donors = ranked[current[ranked] <= caps[ranked] * 0.98]
            receivers = ranked[count // 2 :]
            _transfer_mass_array(current, donors, receivers, shift)

    current = _scale_to_total_array(current, total_deposits)
    return np.maximum(current, 1.0).tolist()


def _refine_balances_cupy(
    balances: list[float],
    segments: list[str],
    total_deposits: float,
    insurance_limit: float,
    target_insured_share: float,
    target_top10: float,
) -> list[float]:
    """GPU balance refinement; large n and missing CuPy use the host loop."""
    if len(balances) >= _HOST_REFINE_MIN:
        return _refine_balances_numpy(
            balances,
            segments,
            total_deposits,
            insurance_limit,
            target_insured_share,
            target_top10,
        )
    cp = _cupy_runtime()
    if cp is None:
        return _refine_balances_numpy(
            balances,
            segments,
            total_deposits,
            insurance_limit,
            target_insured_share,
            target_top10,
        )
    caps_host = _caps_for_segments_array(segments, insurance_limit)
    caps = cp.asarray(caps_host)
    current = cp.maximum(cp.asarray(balances, dtype=cp.float64), 1.0)
    count = int(current.size)
    if count < 10:
        scaled = current * (total_deposits / float(current.sum()))
        return cp.maximum(scaled, 1.0).get().tolist()

    for _ in range(96):
        total = float(current.sum())
        if total > 0.0:
            current *= total_deposits / total
        ranked = cp.argsort(current)
        top_idx = ranked[-10:]
        top_sum = float(current[top_idx].sum())
        if top_sum > 0.0:
            current[top_idx] *= target_top10 / top_sum
        total = float(current.sum())
        if total > 0.0:
            current *= total_deposits / total

        insured_share = float(cp.minimum(current, caps).sum()) / total_deposits
        gap = (target_insured_share - insured_share) * total_deposits
        if abs(gap) < total_deposits * 0.0005:
            break
        shift = min(abs(gap) * 0.5, total_deposits * 0.03)
        if gap > 0.0:
            receivers = ranked[current[ranked] < caps[ranked]]
            donors = ranked[count // 3 :]
        else:
            donors = ranked[current[ranked] <= caps[ranked] * 0.98]
            receivers = ranked[count // 2 :]
        _transfer_mass_cupy(current, donors, receivers, shift)

    total = float(current.sum())
    if total > 0.0:
        current *= total_deposits / total
    return cp.maximum(current, 1.0).get().tolist()


def _refine_balances(
    balances: list[float],
    segments: list[str],
    total_deposits: float,
    insurance_limit: float,
    target_insured_share: float,
    target_top10: float,
) -> list[float]:
    if population_balance_backend() == "gpu":
        return _refine_balances_cupy(
            balances,
            segments,
            total_deposits,
            insurance_limit,
            target_insured_share,
            target_top10,
        )
    return _refine_balances_numpy(
        balances,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )


def _solve_balances(
    z_scores: list[float],
    segments: list[str],
    total_deposits: float,
    insurance_limit: float,
    target_insured_share: float,
    target_top10: float,
) -> list[float]:
    raw = _shape_balances(_lognormal_draws(z_scores), 1.15)
    boosted = _apply_tail_boost(raw, math.log(12.0))
    seed = _scale_to_total_array(np.asarray(boosted, dtype=np.float64), total_deposits).tolist()
    return _refine_balances(
        seed,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )


@dataclass(frozen=True, slots=True)
class PopulationSummary:
    depositor_count: int
    deposits: float
    insurance_limit: float
    insured_share: float
    uninsured_share: float
    top10_balance: float
    segments: dict[str, float]


_POPULATION_CACHE: dict[tuple[str, int, int, str], tuple[DepositorArrays, PopulationSummary]] = {}


def clear_sample_population_cache() -> None:
    """Drop cached populations (tests only)."""
    _POPULATION_CACHE.clear()


def _population_cache_key(scenario: Scenario, tier: int) -> tuple[str, int, int, str, str]:
    override = os.environ.get("STAMPEDE_CALIB_DEPOSITORS", "").strip()
    return (scenario.preset, int(scenario.seed), tier, override, population_balance_backend())


def _depositor_count_for_tier(tier: int) -> int:
    override = os.environ.get("STAMPEDE_CALIB_DEPOSITORS", "").strip()
    if override:
        return max(1, int(override))
    if tier == 0:
        return int(parse_registry_number("Depositor count (Tier 0)"))
    return int(parse_registry_number("Depositor count (Tier 2)"))


def sample_population(
    scenario: Scenario,
    *,
    tier: int = 0,
) -> tuple[DepositorArrays, PopulationSummary]:
    """Sample segments and balances; bisect the Pareto tail to match preset targets."""
    cache_key = _population_cache_key(scenario, tier)
    cached = _POPULATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    count = _depositor_count_for_tier(tier)

    insurance_limit = parse_registry_number("Insurance limit")
    target_top10 = parse_registry_number("SVB-like: top-10 depositor balance")
    target_uninsured = uninsured_share(scenario.preset)
    target_insured_share = max(0.0, min(1.0, 1.0 - target_uninsured))
    total_deposits = deposits_base()

    weights = segment_weights_for_preset(scenario.preset)
    rng = random.Random(scenario.seed)
    segments = _allocate_segments(rng, count, weights)
    z_scores = [rng.gauss(0.0, 1.0) for _ in range(count)]
    balances = _solve_balances(
        z_scores,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )

    insured = [
        insured_amount(balance, segment, insurance_limit)
        for balance, segment in zip(balances, segments, strict=True)
    ]
    bank_idx = [0] * count
    belief = [_BELIEF_BASE] * count
    threshold = [_THRESHOLD_BY_SEGMENT.get(segment, 0.45) for segment in segments]
    attention = [0.0] * count
    needs_flag = [segment == "startup" for segment in segments]
    has_second_bank = [rng.random() < _SECOND_BANK_RATE for _ in range(count)]
    state = ["active"] * count

    arrays = DepositorArrays(
        bank_idx=bank_idx,
        segment=segments,
        balance=balances,
        insured=insured,
        belief=belief,
        threshold=threshold,
        attention=attention,
        needs_flag=needs_flag,
        has_second_bank=has_second_bank,
        state=state,
    )

    deposit_total = sum(balances)
    insured_total = sum(insured)
    insured_share = insured_total / deposit_total if deposit_total > 0.0 else 0.0
    top10_balance = sum(sorted(balances, reverse=True)[:10])
    segment_totals: dict[str, float] = {name: 0.0 for name in weights}
    for segment, balance in zip(segments, balances, strict=True):
        segment_totals[segment] = segment_totals.get(segment, 0.0) + balance
    for name in segment_totals:
        segment_totals[name] /= deposit_total if deposit_total > 0.0 else 1.0

    summary = PopulationSummary(
        depositor_count=count,
        deposits=deposit_total,
        insurance_limit=insurance_limit,
        insured_share=insured_share,
        uninsured_share=1.0 - insured_share,
        top10_balance=top10_balance,
        segments=segment_totals,
    )
    result = (arrays, summary)
    _POPULATION_CACHE[cache_key] = result
    return result


class PopulationModule:
    """M05 scheduler module: builds population on the first tick; publishes nothing."""

    module_id: str = "M05"
    publishes: ClassVar[tuple[str, ...]] = ()
    consumes: ClassVar[tuple[str, ...]] = ("Tick",)

    def __init__(self) -> None:
        self.scenario: Scenario | None = None
        self.arrays: DepositorArrays | None = None
        self.summary: PopulationSummary | None = None
        self._ready = False
        self.depositor_count = 0
        self.deposits = 0.0
        self.insurance_limit = 0.0
        self.insured_share = 0.0
        self.uninsured_share = 0.0
        self.top10_balance = 0.0
        self.example_top_insured = 0.0
        self.example_top_uninsured = 0.0
        self.segments: dict[str, float] = {}

    def attach(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def step(self, t: float) -> list[BaseModel]:
        del t
        if self.scenario is None or self._ready:
            return []
        self._ready = True
        if is_community(self.scenario.preset):
            self._apply_community_placebo()
            return []
        self.arrays, self.summary = sample_population(self.scenario)
        self._mirror_summary()
        return []

    def _apply_community_placebo(self) -> None:
        insurance_limit = parse_registry_number("Insurance limit")
        count = int(parse_registry_number("Depositor count (Tier 0)"))
        total = deposits_base()
        per = total / count
        segments = ["retail"] * count
        insured = [insured_amount(per, "retail", insurance_limit)] * count
        self.arrays = DepositorArrays(
            bank_idx=[0] * count,
            segment=segments,
            balance=[per] * count,
            insured=insured,
            belief=[_BELIEF_BASE] * count,
            threshold=[_THRESHOLD_BY_SEGMENT["retail"]] * count,
            attention=[0.0] * count,
            needs_flag=[False] * count,
            has_second_bank=[False] * count,
            state=["active"] * count,
        )
        insured_total = sum(insured)
        self.summary = PopulationSummary(
            depositor_count=count,
            deposits=total,
            insurance_limit=insurance_limit,
            insured_share=insured_total / total,
            uninsured_share=1.0 - insured_total / total,
            top10_balance=min(total, per * 10),
            segments={"retail": 1.0},
        )
        self._mirror_summary()

    def _mirror_summary(self) -> None:
        if self.summary is None:
            return
        self.depositor_count = self.summary.depositor_count
        self.deposits = self.summary.deposits
        self.insurance_limit = self.summary.insurance_limit
        self.insured_share = self.summary.insured_share
        self.uninsured_share = self.summary.uninsured_share
        self.top10_balance = self.summary.top10_balance
        self.segments = dict(self.summary.segments)
        each = self.top10_balance / 10.0
        self.example_top_insured = min(each, self.insurance_limit)
        self.example_top_uninsured = max(0.0, each - self.insurance_limit)

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError("M05 is not attached")
        return bank_id_for(self.scenario.preset)
