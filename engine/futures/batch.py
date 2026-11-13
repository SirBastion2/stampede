"""M26 ensemble and futures engine (§5, §18.6, §19, ADR-026).

Array math goes through a backend switch: CuPy when a CUDA device is visible,
NumPy otherwise. CPU results for a fixed seed do not depend on batch width.
The only CUDA calls live in the GPU section below; the CPU path never enters it.

The batched kernel is the §18.2 depositor rule (worry sigmoid, lagged complex
contagion, uninsured withdrawal). ``run_serial_reference`` is the main-engine
path for one future: the same formulas, one depositor at a time. M06's
RawKernel file is still a stub, so this module owns the formulas it batches.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from contracts.events import (
    AttentionTick,
    DirectiveIssued,
    FutureCompleted,
    PolicyEnacted,
    PriceTick,
    StatementIssued,
    VIPDecision,
)
from contracts.schemas import FutureResult, Scenario
from mocks.bus import EventBus
from mocks.support import bank_id_for, canonical_lever, is_community
from params.loader import RegistryError, get_parameter, list_by_module

# §18.2 feature order. The length is the formula's arity, not a calibrated value.
_FEATURE_COUNT = 7

_COUNT = "Futures count (typical batch)"
_AGENTS = "Futures depositor count per future"
_TICK_MINUTES = "Futures tick interval"
_INSURANCE = "Insurance limit"
_DEPOSITS = "SVB-like preset: deposits"
_TICKS = "M26 sim-day tick count"
_K = "M26 complex contagion minimum worried contacts (k)"
_OFFSETS = "M26 contact offsets"
_HUB_FRACTION = "M26 hub balance fraction"
_HUB_WEIGHT = "M26 hub contact weight"
_BIAS = "M26 worry logit bias"
_WEIGHTS = "M26 worry weights"
_LOGIT_CLIP = "M26 logit clip"
_THRESHOLD_LOW = "M26 threshold low"
_THRESHOLD_HIGH = "M26 threshold high"
_BELIEF_LOW = "M26 belief low"
_BELIEF_HIGH = "M26 belief high"
_EVIDENCE = "M26 belief evidence scale"
_ATTENTION_SIGNAL = "M26 attention signal weight"
_LOSS_SIGNAL = "M26 loss signal weight"
_PAYROLL = "M26 payroll need rate"
_SIGMA = "M26 balance lognormal sigma"
_SEGMENTS = "M26 segment names"
_SHARES = "M26 segment shares"
_WITHDRAW_FRACTIONS = "M26 segment withdrawal fractions"
_CATEGORIES = "M26 ownership categories per depositor"
_CASH = "M26 opening cash fraction of deposits"
_OPENING_PRICE = "M26 opening price"
_PRICE_OUTFLOW = "M26 price outflow beta"
_PRICE_ATTENTION = "M26 price attention beta"
_SHOCK = "M26 price shock scale"
_AMPLITUDE = "M26 weight perturbation amplitude"
_WORRY_SHIFT = "M26 statement worry shift"
_THRESHOLD_SHIFT = "M26 statement threshold shift"
_SHIFT_SPREAD = "M26 statement shift spread"
_CAP_SCALE = "M26 cap-withdrawals amount scale"
_TOLERANCE = "M26 single-future match tolerance"
_DIRECTIVE = "M26 directive withdraw level"
_FLOOR = "M26 probability floor"
_STREAMS = "M26 philox stream ids"

_NUMBER = re.compile(
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(?:\s*([kmb])\b)?",
    re.IGNORECASE,
)


def parse_registry_number(name: str, text: str) -> float:
    """Parse a registry cell. A trailing k/m/b suffix scales; words do not."""
    cleaned = text.strip().lower().replace(",", "").replace("$", "")
    match = _NUMBER.search(cleaned)
    if match is None:
        raise RegistryError(f"{name} is not a number: {text!r}")
    number = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        number *= 1_000.0
    elif suffix == "m":
        number *= 1_000_000.0
    elif suffix == "b":
        number *= 1_000_000_000.0
    if "%" in cleaned or "percent" in cleaned:
        number /= 100.0
    if not math.isfinite(number):
        raise RegistryError(f"{name} is not finite: {text!r}")
    return number


def _as_int(name: str, text: str) -> int:
    value = parse_registry_number(name, text)
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise RegistryError(f"{name} is not an integer: {text!r}")
    return int(rounded)


def _split(text: str, sep: str) -> list[str]:
    return [part.strip() for part in text.split(sep) if part.strip()]


def _require(rows: dict[str, str], name: str) -> str:
    try:
        return rows[name]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {name}") from exc


@dataclass(frozen=True, slots=True)
class FuturesParams:
    """M26 inputs loaded from ``params/registry.csv``."""

    n_futures: int
    n_agents: int
    n_ticks: int
    tick_minutes: float
    deposits: float
    insurance_cap: float
    contagion_k: float
    contact_offsets: tuple[int, ...]
    hub_fraction: float
    hub_weight: float
    worry_bias: float
    worry_weights: tuple[float, ...]
    logit_clip: float
    threshold_low: float
    threshold_high: float
    belief_low: float
    belief_high: float
    evidence_scale: float
    attention_signal_weight: float
    loss_signal_weight: float
    payroll_rate: float
    balance_sigma: float
    segment_names: tuple[str, ...]
    segment_shares: tuple[float, ...]
    segment_withdraw_fraction: tuple[float, ...]
    cash_fraction: float
    opening_price: float
    price_outflow_beta: float
    price_attention_beta: float
    price_shock_scale: float
    perturbation_amplitude: float
    statement_worry_shift: float
    statement_threshold_shift: float
    statement_shift_spread: float
    cap_scale: float
    match_tolerance: float
    directive_level: float
    probability_floor: float
    philox_streams: tuple[int, int, int]


def load_futures_params() -> FuturesParams:
    """Load M26 rows through ``params.loader``. Missing names raise."""
    rows = {row.parameter: row.value for row in list_by_module("M26")}
    rows[_INSURANCE] = get_parameter(_INSURANCE).value
    rows[_DEPOSITS] = get_parameter(_DEPOSITS).value
    weights = tuple(
        parse_registry_number(_WEIGHTS, part) for part in _split(_require(rows, _WEIGHTS), ",")
    )
    if len(weights) != _FEATURE_COUNT:
        raise RegistryError(
            f"{_WEIGHTS} must list {_FEATURE_COUNT} features "
            "(attention, contacts, belief, stock return, directive, payroll, uninsured gap)"
        )
    names = tuple(_split(_require(rows, _SEGMENTS), ";"))
    shares = tuple(
        parse_registry_number(_SHARES, part) for part in _split(_require(rows, _SHARES), ",")
    )
    fractions = tuple(
        parse_registry_number(_WITHDRAW_FRACTIONS, part)
        for part in _split(_require(rows, _WITHDRAW_FRACTIONS), ",")
    )
    offsets = tuple(_as_int(_OFFSETS, part) for part in _split(_require(rows, _OFFSETS), ";"))
    streams = tuple(_as_int(_STREAMS, part) for part in _split(_require(rows, _STREAMS), ";"))
    if len(streams) != 3:
        raise RegistryError(f"{_STREAMS} must list population, shock, and perturb ids")
    categories = _as_int(_CATEGORIES, _require(rows, _CATEGORIES))
    if categories < 1:
        raise RegistryError(f"{_CATEGORIES} must be at least 1")
    insurance = parse_registry_number(_INSURANCE, rows[_INSURANCE])
    params = FuturesParams(
        n_futures=_as_int(_COUNT, _require(rows, _COUNT)),
        n_agents=_as_int(_AGENTS, _require(rows, _AGENTS)),
        n_ticks=_as_int(_TICKS, _require(rows, _TICKS)),
        tick_minutes=parse_registry_number(_TICK_MINUTES, _require(rows, _TICK_MINUTES)),
        deposits=parse_registry_number(_DEPOSITS, rows[_DEPOSITS]),
        insurance_cap=insurance * float(categories),
        contagion_k=float(_as_int(_K, _require(rows, _K))),
        contact_offsets=offsets,
        hub_fraction=parse_registry_number(_HUB_FRACTION, _require(rows, _HUB_FRACTION)),
        hub_weight=parse_registry_number(_HUB_WEIGHT, _require(rows, _HUB_WEIGHT)),
        worry_bias=parse_registry_number(_BIAS, _require(rows, _BIAS)),
        worry_weights=weights,
        logit_clip=parse_registry_number(_LOGIT_CLIP, _require(rows, _LOGIT_CLIP)),
        threshold_low=parse_registry_number(_THRESHOLD_LOW, _require(rows, _THRESHOLD_LOW)),
        threshold_high=parse_registry_number(_THRESHOLD_HIGH, _require(rows, _THRESHOLD_HIGH)),
        belief_low=parse_registry_number(_BELIEF_LOW, _require(rows, _BELIEF_LOW)),
        belief_high=parse_registry_number(_BELIEF_HIGH, _require(rows, _BELIEF_HIGH)),
        evidence_scale=parse_registry_number(_EVIDENCE, _require(rows, _EVIDENCE)),
        attention_signal_weight=parse_registry_number(
            _ATTENTION_SIGNAL, _require(rows, _ATTENTION_SIGNAL)
        ),
        loss_signal_weight=parse_registry_number(_LOSS_SIGNAL, _require(rows, _LOSS_SIGNAL)),
        payroll_rate=parse_registry_number(_PAYROLL, _require(rows, _PAYROLL)),
        balance_sigma=parse_registry_number(_SIGMA, _require(rows, _SIGMA)),
        segment_names=names,
        segment_shares=shares,
        segment_withdraw_fraction=fractions,
        cash_fraction=parse_registry_number(_CASH, _require(rows, _CASH)),
        opening_price=parse_registry_number(_OPENING_PRICE, _require(rows, _OPENING_PRICE)),
        price_outflow_beta=parse_registry_number(_PRICE_OUTFLOW, _require(rows, _PRICE_OUTFLOW)),
        price_attention_beta=parse_registry_number(
            _PRICE_ATTENTION, _require(rows, _PRICE_ATTENTION)
        ),
        price_shock_scale=parse_registry_number(_SHOCK, _require(rows, _SHOCK)),
        perturbation_amplitude=parse_registry_number(_AMPLITUDE, _require(rows, _AMPLITUDE)),
        statement_worry_shift=parse_registry_number(_WORRY_SHIFT, _require(rows, _WORRY_SHIFT)),
        statement_threshold_shift=parse_registry_number(
            _THRESHOLD_SHIFT, _require(rows, _THRESHOLD_SHIFT)
        ),
        statement_shift_spread=parse_registry_number(_SHIFT_SPREAD, _require(rows, _SHIFT_SPREAD)),
        cap_scale=parse_registry_number(_CAP_SCALE, _require(rows, _CAP_SCALE)),
        match_tolerance=parse_registry_number(_TOLERANCE, _require(rows, _TOLERANCE)),
        directive_level=parse_registry_number(_DIRECTIVE, _require(rows, _DIRECTIVE)),
        probability_floor=parse_registry_number(_FLOOR, _require(rows, _FLOOR)),
        philox_streams=(streams[0], streams[1], streams[2]),
    )
    _validate_params(params)
    return params


def _validate_params(params: FuturesParams) -> None:
    if params.n_futures < 1 or params.n_agents < 1 or params.n_ticks < 1:
        raise RegistryError("M26 batch counts must be positive")
    if params.tick_minutes <= 0.0 or params.deposits <= 0.0 or params.insurance_cap <= 0.0:
        raise RegistryError("M26 tick, deposits, and insurance cap must be positive")
    if not params.contact_offsets:
        raise RegistryError("M26 contact offsets must be non-empty")
    if not 0.0 < params.hub_fraction < 1.0 or params.hub_weight <= 0.0:
        raise RegistryError("M26 hub fraction must be in (0, 1) and hub weight positive")
    if params.threshold_low >= params.threshold_high:
        raise RegistryError("M26 threshold low must be below threshold high")
    if params.belief_low >= params.belief_high:
        raise RegistryError("M26 belief low must be below belief high")
    if params.balance_sigma <= 0.0 or params.logit_clip <= 0.0:
        raise RegistryError("M26 balance sigma and logit clip must be positive")
    if not 0.0 < params.probability_floor < 0.5:
        raise RegistryError("M26 probability floor must be in (0, 0.5)")
    if not 0.0 < params.cash_fraction <= 1.0:
        raise RegistryError("M26 opening cash fraction must be in (0, 1]")
    if params.opening_price <= 0.0 or params.match_tolerance <= 0.0:
        raise RegistryError("M26 opening price and match tolerance must be positive")
    if params.perturbation_amplitude < 0.0 or not 0.0 <= params.cap_scale <= 1.0:
        raise RegistryError("M26 perturbation amplitude and cap scale are out of range")
    n = len(params.segment_names)
    if n < 1 or len(params.segment_shares) != n or len(params.segment_withdraw_fraction) != n:
        raise RegistryError("M26 segment names, shares, and withdrawal fractions differ in length")
    if any(share < 0.0 for share in params.segment_shares):
        raise RegistryError("M26 segment shares must be non-negative")
    if abs(sum(params.segment_shares) - 1.0) > 1e-9:
        raise RegistryError("M26 segment shares must sum to 1")
    if any(not 0.0 <= frac <= 1.0 for frac in params.segment_withdraw_fraction):
        raise RegistryError("M26 segment withdrawal fractions must be in [0, 1]")
    if not 0.0 <= params.payroll_rate <= 1.0:
        raise RegistryError("M26 payroll need rate must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Intervention:
    """Reduced rule applied inside a future. No new Grok call."""

    worry_shift: float = 0.0
    threshold_shift: float = 0.0
    directive: float = 0.0
    attention_shift: float = 0.0
    guarantee: bool = False
    withdraw_scale: float = 1.0
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class FutureSpec:
    """One future: its Philox seed, frozen public signals, and intervention."""

    future_id: str
    seed: int
    attention: float
    stock_return: float
    intervention: Intervention = field(default_factory=Intervention)
    weight_multipliers: tuple[float, ...] | None = None
    statement_id: str | None = None


@dataclass(frozen=True, slots=True)
class MarketSignals:
    """Public signals frozen when the batch launches."""

    attention: float
    stock_return: float
    opening_price: float


@dataclass(frozen=True, slots=True)
class FutureMetrics:
    """Per-future reduction. ``price_path_band`` is (min, max) along the path."""

    future_id: str
    seed: int
    failed: bool
    t_fail: float | None
    deposits_kept: float
    price_path_band: tuple[float, float]
    cost: float
    outflow: float
    outflow_fraction: float
    price_drawdown: float
    statement_id: str | None = None

    def to_future_result(self) -> FutureResult:
        return FutureResult(
            future_id=self.future_id,
            failed=self.failed,
            t_fail=self.t_fail,
            deposits_kept=self.deposits_kept,
            price_path_band=self.price_path_band,
            cost=self.cost,
        )


@dataclass(frozen=True, slots=True)
class EnsembleSummary:
    """Cross-future ranges. Outflow and price bands are min and max."""

    backend: str
    failure_probability: float
    outflow_band: tuple[float, float]
    price_band: tuple[float, float]
    time_to_failure: tuple[float, ...]
    futures: tuple[FutureMetrics, ...]


@dataclass(frozen=True, slots=True)
class Population:
    """One future's depositor arrays, drawn on the host with Philox."""

    segment: Any
    balance: Any
    insured: Any
    threshold: Any
    belief: Any
    needs: Any
    hub: Any


def probe_cuda() -> bool:
    """GPU path probe. True only when CuPy can see a CUDA device.

    Import failures and driver errors stay here so CPU CI never needs CuPy.
    """
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        if not bool(cp.cuda.is_available()):
            return False
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def resolve_backend() -> tuple[Any, str]:
    """Return ``(xp, "gpu")`` when CUDA is visible, else ``(numpy, "cpu")``."""
    if probe_cuda():
        import cupy as cp

        return cp, "gpu"
    return np, "cpu"


# --- GPU path (isolated) ----------------------------------------------------
# CPU runs never call these. ``backend`` must be "gpu"; the host path uses
# ``_to_device`` / ``_to_host``, which return the NumPy array unchanged.


def _gpu_to_device(xp: Any, host: Any, backend: str) -> Any:
    """GPU path only. Copy a host array onto the CUDA device."""
    if backend != "gpu":
        raise RuntimeError("GPU path entered without a CUDA backend")
    return xp.asarray(host)


def _gpu_to_host(xp: Any, device_array: Any, backend: str) -> Any:
    """GPU path only. Synchronize the null stream and copy back to the host."""
    if backend != "gpu":
        raise RuntimeError("GPU path entered without a CUDA backend")
    xp.cuda.Stream.null.synchronize()
    return xp.asnumpy(device_array)


def _to_device(host: Any, xp: Any, backend: str) -> Any:
    if backend == "cpu":
        return host
    return _gpu_to_device(xp, host, backend)


def _to_host(array: Any, xp: Any, backend: str) -> np.ndarray[Any, Any]:
    if backend == "cpu":
        return np.asarray(array)
    return np.asarray(_gpu_to_host(xp, array, backend))


def _rng(seed: int, stream: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), int(stream)])
    return np.random.Generator(np.random.Philox(sequence))


def contact_index(n_agents: int, offsets: Sequence[int]) -> np.ndarray[Any, Any]:
    """Fixed ring neighbors. The same index is shared by every future."""
    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    offs = np.asarray(tuple(offsets), dtype=np.int64)
    base = np.arange(n_agents, dtype=np.int64)[:, None]
    return (base + offs[None, :]) % n_agents


def draw_population(
    seed: int,
    n_agents: int,
    params: FuturesParams,
    *,
    fully_insured: bool,
) -> Population:
    """Seeded lognormal book scaled to ``params.deposits``. Draw order is fixed."""
    rng = _rng(seed, params.philox_streams[0])
    shares = np.asarray(params.segment_shares, dtype=np.float64)
    segment = rng.choice(len(shares), size=n_agents, p=shares).astype(np.int64)
    balance = rng.lognormal(0.0, params.balance_sigma, size=n_agents).astype(np.float64)
    total = float(balance.sum())
    if total <= 0.0:
        raise RuntimeError("population balances must be positive")
    balance *= params.deposits / total
    balance[-1] += params.deposits - float(balance.sum())
    if float(balance[-1]) <= 0.0:
        raise RuntimeError("deposit residual drove a balance non-positive")
    threshold = rng.uniform(params.threshold_low, params.threshold_high, size=n_agents)
    belief = rng.uniform(params.belief_low, params.belief_high, size=n_agents)
    needs = (rng.random(n_agents) < params.payroll_rate).astype(np.float64)
    insured = np.minimum(balance, params.insurance_cap)
    if fully_insured:
        insured = balance.copy()
    cutoff = float(np.quantile(balance, 1.0 - params.hub_fraction, method="linear"))
    hub = balance >= cutoff
    return Population(
        segment=segment,
        balance=balance.astype(np.float64),
        insured=insured.astype(np.float64),
        threshold=threshold.astype(np.float64),
        belief=belief.astype(np.float64),
        needs=needs.astype(np.float64),
        hub=hub,
    )


def draw_shocks(seed: int, n_ticks: int, params: FuturesParams) -> np.ndarray[Any, Any]:
    """Philox uniform shocks in ``[-scale, scale]``, one per 5-minute tick."""
    rng = _rng(seed, params.philox_streams[1])
    unit = rng.uniform(-1.0, 1.0, size=n_ticks).astype(np.float64)
    return unit * params.price_shock_scale


def weight_multipliers(seed: int, params: FuturesParams) -> tuple[float, ...]:
    """Multipliers in ``[1-amplitude, 1+amplitude]`` for a perturbed future."""
    rng = _rng(seed, params.philox_streams[2])
    unit = rng.random(_FEATURE_COUNT)
    span = params.perturbation_amplitude
    return tuple(float(1.0 + span * (2.0 * value - 1.0)) for value in unit)


def _scaled_weights(
    params: FuturesParams,
    multipliers: Sequence[float] | None,
) -> np.ndarray[Any, Any]:
    base = np.asarray(params.worry_weights, dtype=np.float64)
    if multipliers is None:
        return base
    scale = np.asarray(tuple(multipliers), dtype=np.float64)
    if scale.shape != base.shape:
        raise ValueError("weight multipliers must match the worry-weight length")
    return base * scale


def _sigmoid(logit: float, clip: float) -> float:
    clipped = min(clip, max(-clip, logit))
    return 1.0 / (1.0 + math.exp(-clipped))


def _logit(probability: float, floor: float) -> float:
    clipped = min(1.0 - floor, max(floor, probability))
    return math.log(clipped / (1.0 - clipped))


def _public_signal(
    attention: float,
    stock_return: float,
    attention_weight: float,
    loss_weight: float,
) -> float:
    raw = attention_weight * attention + loss_weight * max(-stock_return, 0.0)
    return min(1.0, max(0.0, raw))


def _scalar_contacts(
    index: int,
    withdrew: np.ndarray[Any, Any],
    worried: np.ndarray[Any, Any],
    hub: np.ndarray[Any, Any],
    neighbors: np.ndarray[Any, Any],
    hub_weight: float,
) -> tuple[float, float]:
    weighted = 0.0
    withdrew_w = 0.0
    worried_w = 0.0
    for neighbor in neighbors[index]:
        slot = int(neighbor)
        weight = hub_weight if bool(hub[slot]) else 1.0
        weighted += weight
        withdrew_w += weight * float(withdrew[slot])
        worried_w += weight * float(worried[slot])
    return withdrew_w / weighted, worried_w


def _scalar_decide(
    *,
    belief: float,
    balance: float,
    insured: float,
    already: float,
    threshold: float,
    needs: float,
    segment: int,
    contact_share: float,
    worried_contacts: float,
    attention: float,
    stock_return: float,
    directive: float,
    weights: np.ndarray[Any, Any],
    intervention: Intervention,
    params: FuturesParams,
) -> tuple[float, float, float]:
    """Main-engine step for one depositor. Returns belief, amount, worried flag."""
    signal = _public_signal(
        attention,
        stock_return,
        params.attention_signal_weight,
        params.loss_signal_weight,
    )
    logit_belief = _logit(belief, params.probability_floor) + params.evidence_scale * (signal - 0.5)
    new_belief = _sigmoid(logit_belief, params.logit_clip)
    gap_dollars = max(balance - insured, 0.0)
    gap = 0.0 if balance <= 0.0 else gap_dollars / balance
    if intervention.guarantee:
        gap = 0.0
        gap_dollars = 0.0
    logit = params.worry_bias + intervention.worry_shift
    logit += float(weights[0]) * attention
    logit += float(weights[1]) * contact_share
    logit += float(weights[2]) * new_belief
    logit += float(weights[3]) * stock_return
    logit += float(weights[4]) * directive
    logit += float(weights[5]) * needs
    logit += float(weights[6]) * gap
    worry = _sigmoid(logit, params.logit_clip)
    thresh = threshold + intervention.threshold_shift
    worried = 1.0 if worry > thresh else 0.0
    amount = 0.0
    eligible = already == 0.0 and worry > thresh and worried_contacts >= params.contagion_k
    if eligible and not intervention.guarantee:
        fraction = params.segment_withdraw_fraction[segment]
        amount = min(balance, gap_dollars * fraction) * intervention.withdraw_scale
    return new_belief, amount, worried


def run_serial_reference(
    spec: FutureSpec,
    params: FuturesParams,
    *,
    n_agents: int,
    n_ticks: int,
    fully_insured: bool,
    opening_price: float | None = None,
) -> FutureMetrics:
    """One future, one depositor at a time. This is the main-engine reference."""
    if n_ticks < 1 or n_agents < 1:
        raise ValueError("n_ticks and n_agents must be positive")
    launch = params.opening_price if opening_price is None else opening_price
    pop = draw_population(spec.seed, n_agents, params, fully_insured=fully_insured)
    shocks = draw_shocks(spec.seed, n_ticks, params)
    neighbors = contact_index(n_agents, params.contact_offsets)
    weights = _scaled_weights(params, spec.weight_multipliers)
    attention = spec.attention + spec.intervention.attention_shift
    directive = spec.intervention.directive
    stock_return = spec.stock_return
    price = launch
    prices = [price]
    belief = pop.belief.copy()
    balance = pop.balance.copy()
    insured = pop.insured.copy()
    withdrew = np.zeros(n_agents, dtype=np.float64)
    worried = np.zeros(n_agents, dtype=np.float64)
    cumulative = 0.0
    failed = False
    t_fail: float | None = None
    cash_limit = params.cash_fraction * params.deposits
    for tick in range(n_ticks):
        amount = np.zeros(n_agents, dtype=np.float64)
        new_belief = np.zeros(n_agents, dtype=np.float64)
        new_worried = np.zeros(n_agents, dtype=np.float64)
        for index in range(n_agents):
            share, count = _scalar_contacts(
                index, withdrew, worried, pop.hub, neighbors, params.hub_weight
            )
            decided = _scalar_decide(
                belief=float(belief[index]),
                balance=float(balance[index]),
                insured=float(insured[index]),
                already=float(withdrew[index]),
                threshold=float(pop.threshold[index]),
                needs=float(pop.needs[index]),
                segment=int(pop.segment[index]),
                contact_share=share,
                worried_contacts=count,
                attention=attention,
                stock_return=stock_return,
                directive=directive,
                weights=weights,
                intervention=spec.intervention,
                params=params,
            )
            new_belief[index] = decided[0]
            amount[index] = decided[1]
            new_worried[index] = decided[2]
        belief = new_belief
        worried = new_worried
        balance = balance - amount
        insured = np.minimum(insured, balance)
        withdrew = np.maximum(withdrew, (amount > 0.0).astype(np.float64))
        outflow = float(np.sum(amount))
        cumulative += outflow
        if not failed and cumulative >= cash_limit:
            failed = True
            t_fail = (tick + 1) * params.tick_minutes
        ret = (
            params.price_outflow_beta * (outflow / params.deposits)
            + params.price_attention_beta * attention
            + float(shocks[tick])
        )
        price = price * math.exp(ret)
        prices.append(price)
        if launch > 0.0:
            stock_return = price / launch - 1.0
    return _metrics_from_path(
        spec,
        params,
        failed=failed,
        t_fail=t_fail,
        deposits_kept=float(np.sum(balance)),
        outflow=cumulative,
        prices=tuple(prices),
        opening_price=launch,
    )


def _sigmoid_xp(xp: Any, logit: Any, clip: float) -> Any:
    clipped = xp.clip(logit, -clip, clip)
    return 1.0 / (1.0 + xp.exp(-clipped))


def lagged_contact_stats(
    xp: Any,
    withdrew: Any,
    worried: Any,
    hub: Any,
    neighbor_idx: Any,
    hub_weight: float,
) -> tuple[Any, Any]:
    """Hub-weighted share who withdrew and weighted worried-contact count."""
    neigh_withdrew = withdrew[:, neighbor_idx]
    neigh_worried = worried[:, neighbor_idx]
    neigh_hub = hub[:, neighbor_idx]
    weight = xp.where(neigh_hub, hub_weight, 1.0)
    denom = weight.sum(axis=-1)
    share = (neigh_withdrew * weight).sum(axis=-1) / denom
    count = (neigh_worried * weight).sum(axis=-1)
    return share, count


def step_m06_kernel(
    xp: Any,
    *,
    belief: Any,
    balance: Any,
    insured: Any,
    withdrew: Any,
    threshold: Any,
    needs: Any,
    segment: Any,
    contact_share: Any,
    worried_contacts: Any,
    attention: Any,
    stock_return: Any,
    directive: Any,
    weights: Any,
    worry_shift: Any,
    threshold_shift: Any,
    guarantee: Any,
    withdraw_scale: Any,
    segment_fraction: Any,
    params: FuturesParams,
) -> tuple[Any, Any, Any]:
    """One 5-minute tick of the batched M06 rule. Shape ``[futures, agents]``.

    Returns withdrawal amount, updated insolvency belief, and the worried flag.
    Contact inputs are lagged. A guarantee zeros the uninsured gap and the amount.
    """
    floor = params.probability_floor
    signal = params.attention_signal_weight * attention + params.loss_signal_weight * xp.maximum(
        -stock_return, 0.0
    )
    signal = xp.clip(signal, 0.0, 1.0)
    prior = xp.clip(belief, floor, 1.0 - floor)
    logit_belief = xp.log(prior / (1.0 - prior)) + params.evidence_scale * (signal[:, None] - 0.5)
    new_belief = _sigmoid_xp(xp, logit_belief, params.logit_clip)
    gap_dollars = xp.maximum(balance - insured, 0.0)
    safe_balance = xp.where(balance > 0.0, balance, 1.0)
    gap = xp.where(balance > 0.0, gap_dollars / safe_balance, 0.0)
    blocked = guarantee[:, None]
    gap = xp.where(blocked, 0.0, gap)
    gap_dollars = xp.where(blocked, 0.0, gap_dollars)
    logit = params.worry_bias + worry_shift[:, None]
    logit = logit + weights[:, 0:1] * attention[:, None]
    logit = logit + weights[:, 1:2] * contact_share
    logit = logit + weights[:, 2:3] * new_belief
    logit = logit + weights[:, 3:4] * stock_return[:, None]
    logit = logit + weights[:, 4:5] * directive[:, None]
    logit = logit + weights[:, 5:6] * needs
    logit = logit + weights[:, 6:7] * gap
    worry = _sigmoid_xp(xp, logit, params.logit_clip)
    thresh = threshold + threshold_shift[:, None]
    worried = (worry > thresh).astype(xp.float64)
    fraction = segment_fraction[segment]
    raw = xp.minimum(balance, gap_dollars * fraction) * withdraw_scale[:, None]
    raw = xp.where(blocked, 0.0, raw)
    eligible = (withdrew == 0.0) & (worry > thresh) & (worried_contacts >= params.contagion_k)
    amount = xp.where(eligible, raw, 0.0)
    return amount, new_belief, worried


def _metrics_from_path(
    spec: FutureSpec,
    params: FuturesParams,
    *,
    failed: bool,
    t_fail: float | None,
    deposits_kept: float,
    outflow: float,
    prices: tuple[float, ...],
    opening_price: float,
) -> FutureMetrics:
    low = min(prices)
    high = max(prices)
    drawdown = 0.0 if opening_price <= 0.0 else max(0.0, 1.0 - low / opening_price)
    return FutureMetrics(
        future_id=spec.future_id,
        seed=spec.seed,
        failed=failed,
        t_fail=t_fail,
        deposits_kept=deposits_kept,
        price_path_band=(low, high),
        cost=spec.intervention.cost,
        outflow=outflow,
        outflow_fraction=outflow / params.deposits,
        price_drawdown=drawdown,
        statement_id=spec.statement_id,
    )


def run_batch(
    specs: Sequence[FutureSpec],
    params: FuturesParams,
    *,
    n_agents: int,
    n_ticks: int,
    fully_insured: bool,
    opening_price: float | None = None,
    backend: tuple[Any, str] | None = None,
) -> EnsembleSummary:
    """Run futures as tensors shaped ``[futures × agents]``."""
    if not specs:
        raise ValueError("run_batch requires at least one future")
    if n_ticks < 1 or n_agents < 1:
        raise ValueError("n_ticks and n_agents must be positive")
    xp, backend_name = resolve_backend() if backend is None else backend
    launch = params.opening_price if opening_price is None else opening_price
    pops = [
        draw_population(spec.seed, n_agents, params, fully_insured=fully_insured) for spec in specs
    ]
    shocks = np.stack([draw_shocks(spec.seed, n_ticks, params) for spec in specs])
    neighbors = contact_index(n_agents, params.contact_offsets)
    balance = _to_device(np.stack([pop.balance for pop in pops]), xp, backend_name)
    insured = _to_device(np.stack([pop.insured for pop in pops]), xp, backend_name)
    belief = _to_device(np.stack([pop.belief for pop in pops]), xp, backend_name)
    threshold = _to_device(np.stack([pop.threshold for pop in pops]), xp, backend_name)
    needs = _to_device(np.stack([pop.needs for pop in pops]), xp, backend_name)
    segment = _to_device(np.stack([pop.segment for pop in pops]), xp, backend_name)
    hub = _to_device(np.stack([pop.hub for pop in pops]), xp, backend_name)
    neighbor_idx = _to_device(neighbors, xp, backend_name)
    fractions = _to_device(
        np.asarray(params.segment_withdraw_fraction, dtype=np.float64),
        xp,
        backend_name,
    )
    weights = _to_device(
        np.stack([_scaled_weights(params, spec.weight_multipliers) for spec in specs]),
        xp,
        backend_name,
    )
    attention_h = np.asarray(
        [spec.attention + spec.intervention.attention_shift for spec in specs],
        dtype=np.float64,
    )
    directive_h = np.asarray(
        [spec.intervention.directive for spec in specs],
        dtype=np.float64,
    )
    stock_h = np.asarray([spec.stock_return for spec in specs], dtype=np.float64)
    worry_shift = _to_device(
        np.asarray([spec.intervention.worry_shift for spec in specs], dtype=np.float64),
        xp,
        backend_name,
    )
    threshold_shift = _to_device(
        np.asarray([spec.intervention.threshold_shift for spec in specs], dtype=np.float64),
        xp,
        backend_name,
    )
    guarantee = _to_device(
        np.asarray([spec.intervention.guarantee for spec in specs], dtype=np.bool_),
        xp,
        backend_name,
    )
    withdraw_scale = _to_device(
        np.asarray([spec.intervention.withdraw_scale for spec in specs], dtype=np.float64),
        xp,
        backend_name,
    )
    n_futures = len(specs)
    withdrew = _to_device(np.zeros((n_futures, n_agents), dtype=np.float64), xp, backend_name)
    worried = _to_device(np.zeros((n_futures, n_agents), dtype=np.float64), xp, backend_name)
    price = np.full(n_futures, launch, dtype=np.float64)
    price_hist = [price.copy()]
    cumulative = np.zeros(n_futures, dtype=np.float64)
    failed = np.zeros(n_futures, dtype=np.bool_)
    t_fail = np.full(n_futures, np.nan, dtype=np.float64)
    cash_limit = params.cash_fraction * params.deposits
    for tick in range(n_ticks):
        share, count = lagged_contact_stats(
            xp, withdrew, worried, hub, neighbor_idx, params.hub_weight
        )
        amount, belief, worried = step_m06_kernel(
            xp,
            belief=belief,
            balance=balance,
            insured=insured,
            withdrew=withdrew,
            threshold=threshold,
            needs=needs,
            segment=segment,
            contact_share=share,
            worried_contacts=count,
            attention=_to_device(attention_h, xp, backend_name),
            stock_return=_to_device(stock_h, xp, backend_name),
            directive=_to_device(directive_h, xp, backend_name),
            weights=weights,
            worry_shift=worry_shift,
            threshold_shift=threshold_shift,
            guarantee=guarantee,
            withdraw_scale=withdraw_scale,
            segment_fraction=fractions,
            params=params,
        )
        balance = balance - amount
        insured = xp.minimum(insured, balance)
        withdrew = xp.maximum(withdrew, (amount > 0.0).astype(xp.float64))
        outflow = _to_host(amount.sum(axis=1), xp, backend_name).astype(np.float64)
        cumulative += outflow
        newly = (~failed) & (cumulative >= cash_limit)
        t_fail = np.where(newly, (tick + 1) * params.tick_minutes, t_fail)
        failed = failed | newly
        ret = (
            params.price_outflow_beta * (outflow / params.deposits)
            + params.price_attention_beta * attention_h
            + shocks[:, tick]
        )
        price = price * np.exp(ret)
        price_hist.append(price.copy())
        if launch > 0.0:
            stock_h = price / launch - 1.0
    kept = _to_host(balance.sum(axis=1), xp, backend_name).astype(np.float64)
    history = np.stack(price_hist)
    metrics: list[FutureMetrics] = []
    for index, spec in enumerate(specs):
        when = None if not bool(failed[index]) else float(t_fail[index])
        path = tuple(float(value) for value in history[:, index])
        metrics.append(
            _metrics_from_path(
                spec,
                params,
                failed=bool(failed[index]),
                t_fail=when,
                deposits_kept=float(kept[index]),
                outflow=float(cumulative[index]),
                prices=path,
                opening_price=launch,
            )
        )
    return _summarize(backend_name, tuple(metrics))


def _summarize(backend_name: str, futures: tuple[FutureMetrics, ...]) -> EnsembleSummary:
    failure = sum(1.0 for item in futures if item.failed) / float(len(futures))
    outflow = tuple(item.outflow_fraction for item in futures)
    lows = tuple(item.price_path_band[0] for item in futures)
    highs = tuple(item.price_path_band[1] for item in futures)
    times = tuple(sorted(item.t_fail for item in futures if item.t_fail is not None))
    return EnsembleSummary(
        backend=backend_name,
        failure_probability=failure,
        outflow_band=(min(outflow), max(outflow)),
        price_band=(min(lows), max(highs)),
        time_to_failure=times,
        futures=futures,
    )


def world_intervention(
    levers: Sequence[str],
    directive: float,
    params: FuturesParams,
) -> Intervention:
    """Map policy levers onto the reduced rule. Unknown levers are ignored."""
    canonical = {canonical_lever(name) for name in levers if name}
    scale = params.cap_scale if "cap_withdrawals" in canonical else 1.0
    return Intervention(
        guarantee="guarantee_deposits" in canonical,
        withdraw_scale=scale,
        directive=directive,
    )


def build_future_specs(
    *,
    base_seed: int,
    n_futures: int,
    signals: MarketSignals,
    statement_ids: Sequence[str],
    world: Intervention,
    params: FuturesParams,
) -> list[FutureSpec]:
    """Baseline plus one future per statement, or a perturbed seed ensemble."""
    if n_futures < 1:
        raise ValueError("n_futures must be positive")
    if statement_ids:
        specs = [
            FutureSpec(
                future_id="f00",
                seed=base_seed,
                attention=signals.attention,
                stock_return=signals.stock_return,
                intervention=world,
            )
        ]
        for index, stmt_id in enumerate(statement_ids):
            if len(specs) >= n_futures:
                break
            factor = 1.0 + index * params.statement_shift_spread
            specs.append(
                FutureSpec(
                    future_id=f"f{len(specs):02d}",
                    seed=base_seed,
                    attention=signals.attention,
                    stock_return=signals.stock_return,
                    intervention=Intervention(
                        worry_shift=world.worry_shift + params.statement_worry_shift * factor,
                        threshold_shift=(
                            world.threshold_shift + params.statement_threshold_shift * factor
                        ),
                        directive=world.directive,
                        attention_shift=world.attention_shift,
                        guarantee=world.guarantee,
                        withdraw_scale=world.withdraw_scale,
                        cost=world.cost,
                    ),
                    statement_id=stmt_id,
                )
            )
        return specs
    specs = []
    for index in range(n_futures):
        seed = base_seed + index
        multipliers = None if index == 0 else weight_multipliers(seed, params)
        specs.append(
            FutureSpec(
                future_id=f"f{index:02d}",
                seed=seed,
                attention=signals.attention,
                stock_return=signals.stock_return,
                intervention=world,
                weight_multipliers=multipliers,
            )
        )
    return specs


def future_completed_events(
    summary: EnsembleSummary,
    *,
    t: float,
    seed: int,
    bank_id: str | None,
) -> list[BaseModel]:
    """One ``FutureCompleted`` per future. Ensemble ranges are copied onto each."""
    band_out = [summary.outflow_band[0], summary.outflow_band[1]]
    band_px = [summary.price_band[0], summary.price_band[1]]
    times = list(summary.time_to_failure)
    events: list[BaseModel] = []
    for item in summary.futures:
        events.append(
            FutureCompleted(
                t=t,
                seed=seed,
                bank_id=bank_id,
                cause_id=item.future_id,
                future_id=item.future_id,
                metrics={
                    "failed": item.failed,
                    "t_fail": item.t_fail,
                    "deposits_kept": item.deposits_kept,
                    "price_low": item.price_path_band[0],
                    "price_high": item.price_path_band[1],
                    "cost": item.cost,
                    "outflow": item.outflow,
                    "outflow_fraction": item.outflow_fraction,
                    "price_drawdown": item.price_drawdown,
                    "statement_id": item.statement_id,
                    "future_seed": item.seed,
                    "failure_probability": summary.failure_probability,
                    "outflow_band": band_out,
                    "price_band": band_px,
                    "time_to_failure": times,
                    "backend": summary.backend,
                },
            )
        )
    return events


def _lever_name(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        lever = item.get("lever")
        if isinstance(lever, str):
            return lever
    return ""


class FuturesRunner:
    """Batched futures runner (§19 M26). ``step(t)`` publishes ``FutureCompleted``."""

    publishes: ClassVar[tuple[str, ...]] = ("FutureCompleted",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick",)

    def __init__(self, module_id: str = "M26") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.params = load_futures_params()
        self.agent_count = self.params.n_agents
        self.future_count = self.params.n_futures
        self.tick_count = self.params.n_ticks
        self.summary: EnsembleSummary | None = None
        self._launched = False

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        bus.subscribe("Tick", self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None or self._launched:
            return []
        self._launched = True
        signals, statement_ids, world = self._snapshot()
        specs = build_future_specs(
            base_seed=self.scenario.seed,
            n_futures=self.future_count,
            signals=signals,
            statement_ids=statement_ids,
            world=world,
            params=self.params,
        )
        summary = run_batch(
            specs,
            self.params,
            n_agents=self.agent_count,
            n_ticks=self.tick_count,
            fully_insured=is_community(self.scenario.preset),
            opening_price=signals.opening_price,
        )
        self.summary = summary
        return future_completed_events(
            summary,
            t=t,
            seed=self.scenario.seed,
            bank_id=bank_id_for(self.scenario.preset),
        )

    def _snapshot(self) -> tuple[MarketSignals, list[str], Intervention]:
        assert self.scenario is not None
        assert self.bus is not None
        attention = 0.0
        first_price: float | None = None
        latest_price: float | None = None
        directive = 0.0
        statement_ids: list[str] = []
        levers = [_lever_name(item) for item in self.scenario.policies]
        for event in self.bus.log:
            if isinstance(event, AttentionTick):
                attention = event.level
            elif isinstance(event, PriceTick):
                if first_price is None:
                    first_price = event.price
                latest_price = event.price
            elif isinstance(event, StatementIssued):
                statement_ids.append(event.stmt_id)
            elif isinstance(event, PolicyEnacted):
                levers.append(event.lever)
            elif isinstance(event, DirectiveIssued) and event.action == "withdraw":
                directive = self.params.directive_level
            elif isinstance(event, VIPDecision) and event.action == "withdraw":
                directive = self.params.directive_level
        opening = self.params.opening_price
        stock_return = 0.0
        if latest_price is not None and latest_price > 0.0:
            opening = latest_price
            if first_price is not None and first_price > 0.0:
                stock_return = latest_price / first_price - 1.0
        signals = MarketSignals(
            attention=attention,
            stock_return=stock_return,
            opening_price=opening,
        )
        world = world_intervention(levers, directive, self.params)
        return signals, statement_ids, world
