"""M06 depositor decision model (§5, §18.2, §19, ADR-029).

Array kernel with a backend switch: CuPy when a CUDA device is available,
NumPy otherwise. CPU CI uses the NumPy path only. The GPU entry point is
``run_gpu_decision`` and is not called unless CUDA is visible.

Worry is a sigmoid of registry weights on
[attention, contact withdraw share, insolvency belief, stock return,
directive, payroll urgency, uninsured gap]. A depositor withdraws when worry
exceeds their threshold and the weighted count of worried contacts is at
least k. k is the ASSUMED registry row, not the TBD cell.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from contracts.events import (
    AttentionTick,
    DepositRateChanged,
    DirectiveIssued,
    PolicyEnacted,
    PriceTick,
    RumorPosted,
    StatementIssued,
    VIPDecision,
    WithdrawalRequested,
)
from contracts.schemas import DepositorArrays, Scenario
from engine.agents.population import PopulationModule
from engine.depositor_rows import resolve as _resolve_row
from engine.live_overrides import flag_enabled, refuse_under_gpu_calibration
from mocks.bus import EventBus
from mocks.support import bank_id_for, is_community
from params.loader import RegistryError, get_parameter, list_by_module

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Feature order matches §18.2. The count is the spec, not a tuned parameter.
_FEATURES: tuple[str, ...] = (
    "attention",
    "contact_withdraw_share",
    "insolvency_belief",
    "stock_return",
    "directive",
    "payroll_urgency",
    "uninsured_gap",
)

# Numerical floor so a Bayes update never divides by zero. Not a behavior parameter.
_BAYES_EPS = 1e-6

_K = "M06 complex contagion k (ASSUMED)"
_BIAS = "M06 worry bias (ASSUMED)"
_IGNORE_PRICE = "M06 depositors ignore stock price"
_CRED_CEILING = "M06 attention ceiling by credibility enabled"
_CRED_CEILING_REF = "M06 attention ceiling reference credibility"
_CURE_WORRY = "M06 cure rumor worry shift enabled"
_CURE_FACTOR = "M06 cure rumor worry factor"
_WEIGHTS = "M06 worry weights (ASSUMED)"
_HUB_WEIGHT = "M06 hub contact weight (ASSUMED)"
_HUB_QUANTILE = "M06 hub balance quantile (ASSUMED)"
_DEGREE = "M06 contact degree (ASSUMED)"
_BELIEF_STRENGTH = "M06 belief update strength (ASSUMED)"
_SIGNAL_ATTENTION = "M06 public signal attention weight (ASSUMED)"
_SIGNAL_STOCK = "M06 public signal stock weight (ASSUMED)"
_SIGNAL_BASE = "M06 uninformative signal (ASSUMED)"
_BASELINE_BELIEF = "M06 baseline insolvency belief (ASSUMED)"
_CEILING = "M06 zero-attention worry ceiling (ASSUMED)"
_STATEMENT_LOGIT = "M06 statement logit shift (ASSUMED)"
_STATEMENT_THRESHOLD = "M06 statement threshold shift (ASSUMED)"
_RATE_RELIEF = "M06 deposit-rate relief weight (ASSUMED)"
_GUARANTEE_SCALE = "M06 guarantee uninsured-gap scale (ASSUMED)"
_CAP_SCALE = "M06 cap withdrawal scale (ASSUMED)"
_MIN_AMOUNT = "M06 minimum withdrawal amount (ASSUMED)"
_SEGMENT_FRACTIONS = "M06 segment withdraw fractions (ASSUMED)"
_DEFAULT_FRACTION = "M06 default segment withdraw fraction (ASSUMED)"
_DECISION_MODE = "M06 withdrawal decision mode (ASSUMED)"
_SEGMENT_CHANNELS = "M06 segment channels (ASSUMED)"
_DEFAULT_CHANNEL = "M06 default channel (ASSUMED)"
_COMMUNITY_PRESET = "Preset: insured community bank"
_MOBILE_SHARE = "Depositor mobile share of uninsured deposits (FIT)"
_MOBILE_SHARE_MIN = "Depositor mobile share of uninsured deposits (FIT) min"
_MOBILE_SHARE_MAX = "Depositor mobile share of uninsured deposits (FIT) max"
_MOBILE_SHARE_INDEX = 15
_AWARE_MEDIA = "M06 awareness media rate (FIT)"
_AWARE_MEDIA_MIN = "M06 awareness media rate (FIT) min"
_AWARE_MEDIA_MAX = "M06 awareness media rate (FIT) max"
_AWARE_WOM = "M06 awareness word-of-mouth rate (ASSUMED)"
_AWARE_NEWS = "M06 awareness news rate (ASSUMED)"
_AWARE_MEDIA_PRIOR = 0.001
_AWARE_WOM_PRIOR = 0.05
_AWARE_NEWS_PRIOR = 0.05
_AWARE_W1_PRIOR = 1.5
_DECISION_RATE = "M07 decision-to-request rate (FIT)"
_DECISION_RATE_MIN = "M07 decision-to-request rate (FIT) min"
_DECISION_RATE_MAX = "M07 decision-to-request rate (FIT) max"
_DECISION_RATE_INDEX = 16
_DECISION_RATE_PRIOR = 1.0 / 600.0


@dataclass(frozen=True, slots=True)
class DecisionParams:
    """ASSUMED M06 inputs loaded from the parameter registry."""

    bias: float
    weights: tuple[float, ...]
    k: float
    hub_weight: float
    hub_quantile: float
    contact_degree: int
    belief_strength: float
    signal_attention: float
    signal_stock: float
    signal_base: float
    baseline_belief: float
    worry_ceiling: float
    statement_logit_shift: float
    statement_threshold_shift: float
    rate_relief_weight: float
    guarantee_gap_scale: float
    cap_amount_scale: float
    minimum_amount: float
    segment_fraction: dict[str, float]
    default_fraction: float
    segment_channel: dict[str, str]
    default_channel: str


@dataclass(frozen=True, slots=True)
class DecisionInputs:
    """One row per depositor. Contacts are indexes into the same rows."""

    attention: FloatArray
    belief: FloatArray
    threshold: FloatArray
    balance: FloatArray
    insured: FloatArray
    stock_return: float
    directive: FloatArray
    payroll: FloatArray
    contacts: IntArray
    hub_weight: FloatArray
    already_withdrew: BoolArray
    statement_shift: FloatArray
    deposit_rate: FloatArray
    segment_fraction: FloatArray
    vip_amount: FloatArray
    decided: FloatArray | None = None  # SVB-28: soft once-mode decided probability
    aware: FloatArray | None = None  # SVB-33: per-depositor awareness in [0, 1]
    news: float = 0.0  # SVB-33: current feedback-news level (M03)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Kernel outputs. Arrays are NumPy even when the math ran on CuPy."""

    worry: FloatArray
    belief: FloatArray
    withdraw: BoolArray
    amount: FloatArray
    worried_contacts: FloatArray
    backend: str
    decided: FloatArray | None = None
    aware: FloatArray | None = None


def cuda_device_available() -> bool:
    """True only when CuPy can see a CUDA device. CPU CI stays false."""
    try:
        cupy: Any = importlib.import_module("cupy")
    except ModuleNotFoundError:
        return False
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return False
    return count > 0


def active_backend_name() -> str:
    """``gpu`` when CUDA is visible and ``STAMPEDE_BACKEND`` isn't ``cpu`` (CPY-1), otherwise ``cpu``."""
    from engine.xp import backend_preference

    if backend_preference() == "gpu" and cuda_device_available():
        return "gpu"
    return "cpu"


def _as_float(name: str, text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise RegistryError(f"{name} is not a float: {text!r}") from exc


def _assumed_rows() -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for row in list_by_module("M06"):
        found[row.parameter] = (row.value, row.status)
    return found


def _assumed(rows: dict[str, tuple[str, str]], name: str) -> str:
    try:
        value, status = rows[name]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {name}") from exc
    # persist_fit writes FITTED on the stage-1 worry-bias row; stage 2 must still load it.
    if name == _BIAS:
        if status not in {"ASSUMED", "FITTED"}:
            raise RegistryError(f"{name} must be ASSUMED or FITTED, got {status}")
        return value
    if status != "ASSUMED":
        raise RegistryError(f"{name} must be ASSUMED, got {status}")
    return value


def _parse_number_map(name: str, text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        key, sep, raw = item.partition(":")
        if sep != ":" or not key.strip() or not raw.strip():
            raise RegistryError(f"{name} entries must be name:value")
        parsed[key.strip()] = _as_float(name, raw.strip())
    if not parsed:
        raise RegistryError(f"{name} is empty")
    return parsed


def _parse_text_map(name: str, text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        key, sep, raw = item.partition(":")
        if sep != ":" or not key.strip() or not raw.strip():
            raise RegistryError(f"{name} entries must be name:value")
        parsed[key.strip()] = raw.strip()
    if not parsed:
        raise RegistryError(f"{name} is empty")
    return parsed


def load_decision_params() -> DecisionParams:
    """Load M06 rows from ``params/registry.csv`` via ``params.loader``.

    Worry bias may be ASSUMED or FITTED (persisted stage-1 fit). Every other
    M06 row must still be ASSUMED.
    """
    rows = _assumed_rows()
    weight_text = _assumed(rows, _WEIGHTS)
    weights = tuple(
        _as_float(_WEIGHTS, part.strip())
        for part in weight_text.split(",")
        if part.strip()
    )
    if len(weights) != len(_FEATURES):
        raise RegistryError(
            f"{_WEIGHTS} must list {len(_FEATURES)} weights in order {_FEATURES}"
        )
    degree = int(_as_float(_DEGREE, _assumed(rows, _DEGREE)))
    params = DecisionParams(
        bias=_as_float(_BIAS, _assumed(rows, _resolve_row(_BIAS))),
        weights=weights,
        k=_as_float(_K, _assumed(rows, _K)),
        hub_weight=_as_float(_HUB_WEIGHT, _assumed(rows, _HUB_WEIGHT)),
        hub_quantile=_as_float(_HUB_QUANTILE, _assumed(rows, _HUB_QUANTILE)),
        contact_degree=degree,
        belief_strength=_as_float(_BELIEF_STRENGTH, _assumed(rows, _BELIEF_STRENGTH)),
        signal_attention=_as_float(_SIGNAL_ATTENTION, _assumed(rows, _SIGNAL_ATTENTION)),
        signal_stock=_as_float(_SIGNAL_STOCK, _assumed(rows, _SIGNAL_STOCK)),
        signal_base=_as_float(_SIGNAL_BASE, _assumed(rows, _SIGNAL_BASE)),
        baseline_belief=_as_float(_BASELINE_BELIEF, _assumed(rows, _BASELINE_BELIEF)),
        worry_ceiling=_as_float(_CEILING, _assumed(rows, _CEILING)),
        statement_logit_shift=_as_float(_STATEMENT_LOGIT, _assumed(rows, _STATEMENT_LOGIT)),
        statement_threshold_shift=_as_float(
            _STATEMENT_THRESHOLD, _assumed(rows, _STATEMENT_THRESHOLD)
        ),
        rate_relief_weight=_as_float(_RATE_RELIEF, _assumed(rows, _RATE_RELIEF)),
        guarantee_gap_scale=_as_float(_GUARANTEE_SCALE, _assumed(rows, _GUARANTEE_SCALE)),
        cap_amount_scale=_as_float(_CAP_SCALE, _assumed(rows, _CAP_SCALE)),
        minimum_amount=_as_float(_MIN_AMOUNT, _assumed(rows, _MIN_AMOUNT)),
        segment_fraction=_parse_number_map(
            _SEGMENT_FRACTIONS, _assumed(rows, _SEGMENT_FRACTIONS)
        ),
        default_fraction=_as_float(_DEFAULT_FRACTION, _assumed(rows, _DEFAULT_FRACTION)),
        segment_channel=_parse_text_map(_SEGMENT_CHANNELS, _assumed(rows, _SEGMENT_CHANNELS)),
        default_channel=_assumed(rows, _DEFAULT_CHANNEL),
    )
    _validate_params(params)
    return params


def _validate_params(params: DecisionParams) -> None:
    if params.k <= 0.0:
        raise RegistryError("M06 complex contagion k must be positive")
    if params.hub_weight < 1.0:
        raise RegistryError("M06 hub contact weight must be at least 1")
    if not 0.0 <= params.hub_quantile <= 1.0:
        raise RegistryError("M06 hub balance quantile must be in [0, 1]")
    if params.contact_degree < 1:
        raise RegistryError("M06 contact degree must be at least 1")
    if not 0.0 <= params.belief_strength <= 1.0:
        raise RegistryError("M06 belief update strength must be in [0, 1]")
    if not 0.0 < params.worry_ceiling < 1.0:
        raise RegistryError("M06 zero-attention worry ceiling must be in (0, 1)")
    if params.minimum_amount < 0.0:
        raise RegistryError("M06 minimum withdrawal amount cannot be negative")
    if not params.default_channel:
        raise RegistryError("M06 default channel is empty")
    for name, fraction in params.segment_fraction.items():
        if not 0.0 <= fraction <= 1.0:
            raise RegistryError(f"M06 segment fraction for {name} must be in [0, 1]")
    if not 0.0 <= params.default_fraction <= 1.0:
        raise RegistryError("M06 default segment withdraw fraction must be in [0, 1]")


def _calibration_scale(component: float) -> float:
    """Map a unit hypercube coordinate to a positive multiplier (M27 harness)."""
    return math.exp(0.8 * (float(component) - 0.5))


def decision_params_from_vector(
    base: DecisionParams,
    vector: NDArray[np.float64],
    *,
    clamp_inputs: bool = True,
) -> DecisionParams:
    """Apply the 12-dimensional M27 behavioral vector onto ASSUMED M06 rows.

    ``clamp_inputs=True`` is the user-facing path: the unit cube is clipped
    and k / belief / signal stay inside their registry-valid ranges.
    The M27 fitter passes ``clamp_inputs=False`` and flags a parameter that
    sits on its own search bound instead of silently collapsing it.
    """
    values = np.asarray(vector, dtype=np.float64)
    if clamp_inputs:
        values = np.clip(values, 0.0, 1.0)
    if values.shape[0] < 12:
        msg = f"M27 behavioral vector needs 12 components, got {values.shape[0]}"
        raise ValueError(msg)
    weights = tuple(base.weights[index] * _calibration_scale(values[index]) for index in range(7))
    bias = base.bias + 4.0 * (float(values[7]) - 0.5)
    k_raw = base.k * _calibration_scale(values[8])
    k = max(1.0, k_raw) if clamp_inputs else max(1e-6, k_raw)
    belief_raw = base.belief_strength * _calibration_scale(values[10])
    signal_raw = base.signal_attention * _calibration_scale(values[11])
    if clamp_inputs:
        belief_strength = min(1.0, max(0.0, belief_raw))
        signal_attention = min(1.0, max(0.0, signal_raw))
    else:
        belief_strength = min(1.0 - 1e-12, max(1e-12, belief_raw))
        signal_attention = min(1.0 - 1e-12, max(1e-12, signal_raw))
    fitted = DecisionParams(
        bias=bias,
        weights=weights,
        k=k,
        hub_weight=base.hub_weight,
        hub_quantile=base.hub_quantile,
        contact_degree=base.contact_degree,
        belief_strength=belief_strength,
        signal_attention=signal_attention,
        signal_stock=base.signal_stock,
        signal_base=base.signal_base,
        baseline_belief=base.baseline_belief,
        worry_ceiling=base.worry_ceiling,
        statement_logit_shift=base.statement_logit_shift,
        statement_threshold_shift=base.statement_threshold_shift,
        rate_relief_weight=base.rate_relief_weight,
        guarantee_gap_scale=base.guarantee_gap_scale,
        cap_amount_scale=base.cap_amount_scale,
        minimum_amount=base.minimum_amount,
        segment_fraction=base.segment_fraction,
        default_fraction=base.default_fraction,
        segment_channel=base.segment_channel,
        default_channel=base.default_channel,
    )
    _validate_params(fitted)
    return fitted


def calibration_amount_scale(
    vector: NDArray[np.float64],
    *,
    clamp_inputs: bool = True,
) -> float:
    """Withdrawal amount multiplier from behavioral vector index 9."""
    values = np.asarray(vector, dtype=np.float64)
    if clamp_inputs:
        values = np.clip(values, 0.0, 1.0)
    if values.shape[0] < 10:
        raise ValueError("M27 behavioral vector needs at least 10 components")
    return _calibration_scale(values[9])


def mobile_share_bounds(path=None) -> tuple[float, float]:
    """Registry ``[min, max]`` for the mobile-share fit coordinate."""
    lo = float(get_parameter(_MOBILE_SHARE_MIN, path).value)
    hi = float(get_parameter(_MOBILE_SHARE_MAX, path).value)
    return lo, hi


def withdrawal_decision_once(path=None) -> bool:
    """SVB-28: True when each depositor decides at most once (registry ``once``; ``repeat`` is legacy).

    A registry without the row (trimmed test registries) uses ``once``.
    """
    try:
        value = get_parameter(_DECISION_MODE, path).value.strip().lower()
    except RegistryError:
        return True
    if value not in {"once", "repeat"}:
        raise RegistryError(f"{_DECISION_MODE} must be 'once' or 'repeat', got {value!r}")
    return value == "once"


def mobile_share_default(path=None) -> float:
    """Registry default (0.88). A depositor may request this share of uninsured."""
    return float(get_parameter(_resolve_row(_MOBILE_SHARE), path).value)


def _registry_float_or(name: str, fallback: float, path=None) -> float:
    try:
        return float(get_parameter(name, path).value)
    except (RegistryError, ValueError, TypeError):
        return float(fallback)


def awareness_media_rate_default(path=None) -> float:
    """Registry prior for the per-minute media hazard at attention 1 (not fitted)."""
    return _registry_float_or(_AWARE_MEDIA, _AWARE_MEDIA_PRIOR, path)


def awareness_media_rate_bounds(path=None) -> tuple[float, float]:
    """Registry ``[min, max]`` for the media-awareness-rate fit coordinate."""
    lo = _registry_float_or(_AWARE_MEDIA_MIN, 0.0001, path)
    hi = _registry_float_or(_AWARE_MEDIA_MAX, 0.01, path)
    return lo, hi


def awareness_wom_rate_default(path=None) -> float:
    """ASSUMED per-minute WOM hazard per unit decided-contact share at the prior w1."""
    return _registry_float_or(_AWARE_WOM, _AWARE_WOM_PRIOR, path)


def awareness_news_rate_default(path=None) -> float:
    """ASSUMED per-minute hazard per unit feedback-news level."""
    return _registry_float_or(_AWARE_NEWS, _AWARE_NEWS_PRIOR, path)


def awareness_w1_prior_default(path=None) -> float:
    """Registry contact-weight prior (second M06 worry weight; 1.5)."""
    try:
        text = get_parameter(_WEIGHTS, path).value
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) > 1:
            return float(parts[1])
    except (RegistryError, ValueError, TypeError, IndexError):
        pass
    return float(_AWARE_W1_PRIOR)


def awareness_rate_defaults(path=None) -> tuple[float, float, float, float]:
    """Media, WOM, and news rates plus the contact-weight prior.

    Registry first; module-level ``_AWARE_*_PRIOR`` fallbacks if a row is missing.
    """
    return (
        awareness_media_rate_default(path),
        awareness_wom_rate_default(path),
        awareness_news_rate_default(path),
        awareness_w1_prior_default(path),
    )


def decision_to_request_rate_default(path=None) -> float:
    """Registry prior for the decide-once request hazard (1/600 per minute)."""
    return _registry_float_or(_resolve_row(_DECISION_RATE), _DECISION_RATE_PRIOR, path)


def ratchet_alpha(dt: float, h: float) -> float:
    """Closed-form once-mode fill over a window of length ``dt`` (ADR-084).

    ``decided + (1-exp(-dt·h))·max(0, p_act-decided)`` composes exactly across
    sub-steps when ``p_act`` is held constant, so a coarse tick of length T
    matches the one-minute expansion of the same ``p_act``.
    """
    return 1.0 - math.exp(-float(dt) * float(h))


def decision_to_request_rate_bounds(path=None) -> tuple[float, float]:
    """Registry ``[min, max]`` for the decision-to-request-rate fit coordinate."""
    lo = _registry_float_or(_DECISION_RATE_MIN, 1.0 / 5000.0, path)
    hi = _registry_float_or(_DECISION_RATE_MAX, 1.0 / 20.0, path)
    return lo, hi


def calibration_mobile_share(
    vector: NDArray[np.float64],
    *,
    clamp_inputs: bool = True,
) -> float:
    """Mobile share of uninsured deposits from behavioral vector index 15.

    A short (pre-SVB-24) vector keeps the old uncapped path (1.0). A 16-vector
    at 0.5 is the registry default. The unit-cube slot is the same exp map as
    the other FIT coordinates, then clamped to the registry bounds.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if clamp_inputs:
        values = np.clip(values, 0.0, 1.0)
    if values.size <= _MOBILE_SHARE_INDEX:
        return 1.0
    lo, hi = mobile_share_bounds()
    physical = mobile_share_default() * _calibration_scale(float(values[_MOBILE_SHARE_INDEX]))
    return float(min(hi, max(lo, physical)))


def calibration_awareness_media_rate(
    vector: NDArray[np.float64],
    *,
    clamp_inputs: bool = True,
) -> float:
    """Media awareness rate. SVB-35: not fitted; always the registry prior."""
    del vector, clamp_inputs
    return awareness_media_rate_default()


def calibration_decision_to_request_rate(
    vector: NDArray[np.float64],
    *,
    clamp_inputs: bool = True,
) -> float:
    """Decision-to-request rate from behavioral vector index 16.

    A short (pre-SVB-35) vector keeps the registry prior. A 17-vector at 0.5
    is the registry default. The unit-cube slot is the same exp map as the
    other FIT coordinates, then clamped to the registry bounds.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if clamp_inputs:
        values = np.clip(values, 0.0, 1.0)
    if values.size <= _DECISION_RATE_INDEX:
        return decision_to_request_rate_default()
    lo, hi = decision_to_request_rate_bounds()
    physical = decision_to_request_rate_default() * _calibration_scale(
        float(values[_DECISION_RATE_INDEX])
    )
    return float(min(hi, max(lo, physical)))


def apply_calibration_vector(
    module: DecisionModule,
    vector: NDArray[np.float64],
    *,
    clamp_inputs: bool = True,
) -> None:
    """Install fitted behavioral parameters on a live ``DecisionModule``."""
    module.params = decision_params_from_vector(module.params, vector, clamp_inputs=clamp_inputs)
    module.amount_scale = calibration_amount_scale(vector, clamp_inputs=clamp_inputs)
    module.mobile_share = calibration_mobile_share(vector, clamp_inputs=clamp_inputs)
    module.awareness_media_rate = calibration_awareness_media_rate(
        vector, clamp_inputs=clamp_inputs
    )
    module.decision_to_request_rate = calibration_decision_to_request_rate(
        vector, clamp_inputs=clamp_inputs
    )


def community_placebo(preset: str) -> bool:
    """Insured community banks are the placebo that does not run (§12.1)."""
    if not is_community(preset):
        return False
    for row in list_by_module("M28"):
        if row.parameter == _COMMUNITY_PRESET and "no run" in row.value.lower():
            return True
    return False


def build_contact_index(n: int, degree: int, seed: int) -> IntArray:
    """Deterministic ring of ``degree`` other depositors, rotated by ``seed``.

    The same ``n``, degree, and seed always yield the same index. Built on
    NumPy so the CPU and GPU kernels share one graph.
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=np.int64)
    if n == 1 or degree <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    deg = min(degree, n - 1)
    shift = int(seed) % n
    offsets = np.arange(1, deg + 1, dtype=np.int64)
    base = np.arange(n, dtype=np.int64)[:, None]
    contacts = (base + offsets[None, :] + shift) % n
    self_hit = contacts == base
    contacts = np.where(self_hit, (contacts + 1) % n, contacts)
    return np.asarray(contacts, dtype=np.int64)


def hub_weights(balance: FloatArray, params: DecisionParams) -> FloatArray:
    """Hubs (balance at or above the registry quantile) count as extra contacts."""
    if balance.size == 0:
        return np.zeros(0, dtype=np.float64)
    cutoff = float(np.quantile(balance, params.hub_quantile))
    weights = np.ones(balance.shape[0], dtype=np.float64)
    weights[balance >= cutoff] = params.hub_weight
    return weights


def public_signal(
    attention: FloatArray,
    stock_return: float,
    params: DecisionParams,
) -> FloatArray:
    """Map public news to a likelihood in [0, 1]. The registry base is uninformative."""
    drawdown = max(0.0, -float(stock_return))
    raw = (
        params.signal_base
        + params.signal_attention * np.asarray(attention, dtype=np.float64)
        + params.signal_stock * drawdown
    )
    return np.asarray(np.clip(raw, 0.0, 1.0), dtype=np.float64)


def update_belief(
    belief: FloatArray,
    signal: FloatArray,
    strength: float,
) -> FloatArray:
    """Damped Bayes update. A signal equal to the prior likelihood leaves belief put."""
    updated = _update_belief(np, belief, signal, strength)
    return np.asarray(updated, dtype=np.float64)


def _update_belief(xp: Any, belief: Any, signal: Any, strength: float) -> Any:
    prior = xp.clip(belief, _BAYES_EPS, 1.0 - _BAYES_EPS)
    likelihood = xp.clip(signal, _BAYES_EPS, 1.0 - _BAYES_EPS)
    posterior = (likelihood * prior) / (
        likelihood * prior + (1.0 - likelihood) * (1.0 - prior)
    )
    return (1.0 - strength) * prior + strength * posterior


def _sigmoid(xp: Any, logits: Any) -> Any:
    exp_neg = xp.exp(-xp.abs(logits))
    denom = 1.0 + exp_neg
    return xp.where(logits >= 0.0, 1.0 / denom, exp_neg / denom)


def _length(array: FloatArray | IntArray | BoolArray) -> int:
    return int(array.shape[0])


def _check_inputs(inputs: DecisionInputs) -> int:
    n = _length(inputs.attention)
    rows: tuple[FloatArray | IntArray | BoolArray, ...] = (
        inputs.belief,
        inputs.threshold,
        inputs.balance,
        inputs.insured,
        inputs.directive,
        inputs.payroll,
        inputs.hub_weight,
        inputs.already_withdrew,
        inputs.statement_shift,
        inputs.deposit_rate,
        inputs.segment_fraction,
        inputs.vip_amount,
    )
    for array in rows:
        if _length(array) != n:
            raise ValueError("M06 depositor arrays must share one length")
    if inputs.contacts.ndim != 2 or int(inputs.contacts.shape[0]) != n:
        raise ValueError("M06 contacts must have shape (n, degree)")
    return n


def _to_float(value: Any) -> FloatArray:
    getter = getattr(value, "get", None)
    array = getter() if callable(getter) else value
    return np.asarray(array, dtype=np.float64)


def _to_bool(value: Any) -> BoolArray:
    getter = getattr(value, "get", None)
    array = getter() if callable(getter) else value
    return np.asarray(array, dtype=np.bool_)


def _evaluate(
    xp: Any,
    inputs: DecisionInputs,
    params: DecisionParams,
    *,
    gap_scale: float,
    amount_scale: float,
    backend: str,
    softness: float = 0.0,
    mobile_share: float = 1.0,
    initial_uninsured: FloatArray | None = None,
    decide_once: bool = False,
    news: float | None = None,
    aware: FloatArray | None = None,
    aware0: float = 0.0,
    awareness_media_rate: float | None = None,
    awareness_wom_rate: float | None = None,
    awareness_news_rate: float | None = None,
    awareness_w1_prior: float | None = None,
    decision_to_request_rate: float | None = None,
    dt: float = 1.0,
) -> DecisionResult:
    n = _check_inputs(inputs)
    if n == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_b = np.zeros(0, dtype=np.bool_)
        return DecisionResult(
            worry=empty_f,
            belief=empty_f,
            withdraw=empty_b,
            amount=empty_f,
            worried_contacts=empty_f,
            backend=backend,
            aware=empty_f,
        )
    attention = xp.asarray(inputs.attention, dtype=xp.float64)
    belief = xp.asarray(inputs.belief, dtype=xp.float64)
    threshold = xp.asarray(inputs.threshold, dtype=xp.float64)
    balance = xp.asarray(inputs.balance, dtype=xp.float64)
    insured = xp.asarray(inputs.insured, dtype=xp.float64)
    directive = xp.asarray(inputs.directive, dtype=xp.float64)
    payroll = xp.asarray(inputs.payroll, dtype=xp.float64)
    contacts = xp.asarray(inputs.contacts, dtype=xp.int64)
    hub = xp.asarray(inputs.hub_weight, dtype=xp.float64)
    already = xp.asarray(inputs.already_withdrew.astype(np.float64), dtype=xp.float64)
    statement_shift = xp.asarray(inputs.statement_shift, dtype=xp.float64)
    deposit_rate = xp.asarray(inputs.deposit_rate, dtype=xp.float64)
    segment_fraction = xp.asarray(inputs.segment_fraction, dtype=xp.float64)
    vip_amount = xp.asarray(inputs.vip_amount, dtype=xp.float64)
    weights = xp.asarray(params.weights, dtype=xp.float64)

    drawdown = max(0.0, -float(inputs.stock_return))
    signal = xp.clip(
        params.signal_base + params.signal_attention * attention + params.signal_stock * drawdown,
        0.0,
        1.0,
    )
    belief = _update_belief(xp, belief, signal, params.belief_strength)

    temp = float(softness)
    soft_once = bool(decide_once) and temp > 0.0
    if decide_once:
        decided = (
            xp.asarray(inputs.decided, dtype=xp.float64)
            if inputs.decided is not None
            else already.copy()
        )
    degree = int(inputs.contacts.shape[1])
    if degree == 0:
        share = xp.zeros(n, dtype=xp.float64)
    else:
        share = (decided if soft_once else already)[contacts].mean(axis=1)
    if aware is not None:
        aware_state = xp.asarray(aware, dtype=xp.float64)
    elif inputs.aware is not None:
        aware_state = xp.asarray(inputs.aware, dtype=xp.float64)
    else:
        aware_state = xp.full(n, float(aware0), dtype=xp.float64)
    news_level = float(inputs.news if news is None else news)
    r_media, r_wom, r_news, w1_prior = awareness_rate_defaults()
    if awareness_media_rate is not None:
        r_media = float(awareness_media_rate)
    if awareness_wom_rate is not None:
        r_wom = float(awareness_wom_rate)
    if awareness_news_rate is not None:
        r_news = float(awareness_news_rate)
    if awareness_w1_prior is not None:
        w1_prior = float(awareness_w1_prior)
    if w1_prior == 0.0:
        w1_prior = float(_AWARE_W1_PRIOR)
    w1 = float(params.weights[1]) if len(params.weights) > 1 else 0.0
    hazard = (
        r_media * attention
        + r_wom * (w1 / w1_prior) * share
        + r_news * max(0.0, news_level)
    )
    aware_state = aware_state + (1.0 - aware_state) * (1.0 - xp.exp(-float(dt) * hazard))
    aware_state = xp.clip(aware_state, 0.0, 1.0)
    feat0 = attention * aware_state
    safe_balance = xp.maximum(balance, 0.0)
    gap_balance = xp.maximum(0.0, safe_balance - insured) * gap_scale
    gap_fraction = xp.where(
        safe_balance > 0.0,
        gap_balance / xp.maximum(safe_balance, _BAYES_EPS),
        0.0,
    )
    gap_fraction = xp.clip(gap_fraction, 0.0, 1.0)
    stock = xp.full(n, float(inputs.stock_return), dtype=xp.float64)
    features = xp.stack(
        [feat0, share, belief, stock, directive, payroll, gap_fraction],
        axis=1,
    )
    logits = features @ weights
    logits = (
        logits
        + params.bias
        + statement_shift
        - params.rate_relief_weight * deposit_rate
    )
    worry = _sigmoid(xp, logits)
    vip_force = vip_amount >= params.minimum_amount
    base_amount = segment_fraction * gap_balance * amount_scale
    vip_take = xp.minimum(vip_amount, safe_balance) * amount_scale
    amount = xp.where(vip_force, vip_take, base_amount)
    if initial_uninsured is None:
        init_u_all = xp.maximum(0.0, safe_balance - insured)
    else:
        init_u_all = xp.asarray(initial_uninsured, dtype=xp.float64)
    decided_out = None
    if decide_once and temp <= 0.0:
        worried = worry > threshold
        if degree == 0:
            worried_contacts = xp.zeros(n, dtype=xp.float64)
        else:
            worried_contacts = (worried.astype(xp.float64)[contacts] * hub[contacts]).sum(axis=1)
        first = worried & (worried_contacts >= params.k) & ~(already > 0.5)
        amount = xp.where(first, base_amount, 0.0)
        amount = xp.where(amount >= params.minimum_amount, amount, 0.0)
        amount = xp.where(vip_force, vip_take, amount)
    elif decide_once:
        p_worried = _sigmoid(xp, (worry - threshold) / temp)
        if degree == 0:
            worried_contacts = xp.zeros(n, dtype=xp.float64)
        else:
            worried_contacts = (p_worried[contacts] * hub[contacts]).sum(axis=1)
        p_act = p_worried * _sigmoid(xp, (worried_contacts - params.k) / temp)
        h = decision_to_request_rate_default()
        if decision_to_request_rate is not None:
            h = float(decision_to_request_rate)
        alpha = ratchet_alpha(dt, h)
        d_new = decided + alpha * xp.maximum(0.0, p_act - decided)
        base_once = float(mobile_share) * init_u_all * gap_scale * amount_scale
        amount = xp.where(base_once >= params.minimum_amount, (d_new - decided) * base_once, 0.0)
        amount = xp.where(vip_force, vip_take, amount)
        decided_out = d_new
    elif temp <= 0.0:
        worried = worry > threshold
        if degree == 0:
            worried_contacts = xp.zeros(n, dtype=xp.float64)
        else:
            worried_f = worried.astype(xp.float64)
            worried_contacts = (worried_f[contacts] * hub[contacts]).sum(axis=1)
        rule = worried & (worried_contacts >= params.k)
        act = rule | vip_force
        amount = xp.where(act, amount, 0.0)
        amount = xp.where(amount >= params.minimum_amount, amount, 0.0)
        withdraw = amount >= params.minimum_amount
    else:
        p_worried = _sigmoid(xp, (worry - threshold) / temp)
        if degree == 0:
            worried_contacts = xp.zeros(n, dtype=xp.float64)
        else:
            worried_contacts = (p_worried[contacts] * hub[contacts]).sum(axis=1)
        p_k = _sigmoid(xp, (worried_contacts - params.k) / temp)
        p_act = p_worried * p_k
        p_act = xp.where(vip_force, xp.ones_like(p_act), p_act)
        amount = amount * p_act
        p_min = _sigmoid(xp, (amount - params.minimum_amount) / temp)
        amount = amount * p_min
        withdraw = amount > 0.0
    share = float(mobile_share)
    if share < 1.0:
        if initial_uninsured is None:
            init_u = xp.maximum(0.0, safe_balance - insured)
        else:
            init_u = xp.asarray(initial_uninsured, dtype=xp.float64)
        floor = (1.0 - share) * init_u
        room = xp.maximum(0.0, gap_balance - floor)
        amount = xp.minimum(amount, room)
    if decide_once and temp <= 0.0:
        withdraw = amount >= params.minimum_amount
        decided_out = xp.where(withdraw, 1.0, decided)
    elif decide_once:
        withdraw = (decided_out >= 0.5) | (vip_force & (amount > 0.0))
    return DecisionResult(
        worry=_to_float(worry),
        belief=_to_float(belief),
        withdraw=_to_bool(withdraw),
        amount=_to_float(amount),
        worried_contacts=_to_float(worried_contacts),
        backend=backend,
        decided=None if decided_out is None else _to_float(decided_out),
        aware=_to_float(aware_state),
    )


def evaluate_decisions(
    inputs: DecisionInputs,
    params: DecisionParams,
    *,
    gap_scale: float = 1.0,
    amount_scale: float = 1.0,
    backend: str = "auto",
    softness: float = 0.0,
    mobile_share: float = 1.0,
    initial_uninsured: FloatArray | None = None,
    decide_once: bool = False,
    news: float | None = None,
    aware: FloatArray | None = None,
    aware0: float = 0.0,
    awareness_media_rate: float | None = None,
    awareness_wom_rate: float | None = None,
    awareness_news_rate: float | None = None,
    awareness_w1_prior: float | None = None,
    decision_to_request_rate: float | None = None,
    dt: float = 1.0,
) -> DecisionResult:
    """Run one decision tick. ``backend`` is ``auto``, ``cpu``, or ``gpu``."""
    chosen = active_backend_name() if backend == "auto" else backend
    extra = dict(
        news=news,
        aware=aware,
        aware0=aware0,
        awareness_media_rate=awareness_media_rate,
        awareness_wom_rate=awareness_wom_rate,
        awareness_news_rate=awareness_news_rate,
        awareness_w1_prior=awareness_w1_prior,
        decision_to_request_rate=decision_to_request_rate,
        dt=dt,
    )
    if chosen == "cpu":
        return _evaluate(
            np,
            inputs,
            params,
            gap_scale=gap_scale,
            amount_scale=amount_scale,
            backend="cpu",
            softness=softness,
            mobile_share=mobile_share,
            initial_uninsured=initial_uninsured,
            decide_once=decide_once,
            **extra,
        )
    if chosen == "gpu":
        return run_gpu_decision(
            inputs,
            params,
            gap_scale=gap_scale,
            amount_scale=amount_scale,
            softness=softness,
            mobile_share=mobile_share,
            initial_uninsured=initial_uninsured,
            decide_once=decide_once,
            **extra,
        )
    raise ValueError(f"unknown M06 backend: {backend}")


def run_gpu_decision(
    inputs: DecisionInputs,
    params: DecisionParams,
    *,
    gap_scale: float = 1.0,
    amount_scale: float = 1.0,
    softness: float = 0.0,
    mobile_share: float = 1.0,
    initial_uninsured: FloatArray | None = None,
    decide_once: bool = False,
    news: float | None = None,
    aware: FloatArray | None = None,
    aware0: float = 0.0,
    awareness_media_rate: float | None = None,
    awareness_wom_rate: float | None = None,
    awareness_news_rate: float | None = None,
    awareness_w1_prior: float | None = None,
    decision_to_request_rate: float | None = None,
    dt: float = 1.0,
) -> DecisionResult:
    """GPU path. Isolated: the only function that runs the kernel on CuPy.

    CPU CI does not enter this function. It raises when no CUDA device is visible.
    """
    if not cuda_device_available():
        raise RuntimeError("CUDA device is not available for the M06 GPU path")
    cupy: Any = importlib.import_module("cupy")
    return _evaluate(
        cupy,
        inputs,
        params,
        gap_scale=gap_scale,
        amount_scale=amount_scale,
        backend="gpu",
        softness=softness,
        mobile_share=mobile_share,
        initial_uninsured=initial_uninsured,
        decide_once=decide_once,
        news=news,
        aware=aware,
        aware0=aware0,
        awareness_media_rate=awareness_media_rate,
        awareness_wom_rate=awareness_wom_rate,
        awareness_news_rate=awareness_news_rate,
        awareness_w1_prior=awareness_w1_prior,
        decision_to_request_rate=decision_to_request_rate,
        dt=dt,
    )


def _f64(values: list[float] | FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _population_arrays(scenario: Scenario) -> DepositorArrays:
    """Copy a population from M05. M06 does not write back into that module."""
    population = PopulationModule()
    population.attach(scenario)
    population.step(0.0)
    if population.arrays is None:
        raise RuntimeError("M05 did not build depositor arrays")
    return population.arrays


class DecisionModule:
    """Depositor decision kernel (§19 M06). Publishes WithdrawalRequested."""

    module_id: str
    aware0: float = 0.0
    publishes: ClassVar[tuple[str, ...]] = ("WithdrawalRequested",)
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "AttentionTick",
        "PriceTick",
        "StatementIssued",
        "PolicyEnacted",
        "DepositRateChanged",
        "DirectiveIssued",
        "VIPDecision",
    )

    def __init__(self, module_id: str = "M06") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.params = load_decision_params()
        self._cursor = 0
        self._seq = 0
        self._loaded = False
        self._placebo = False
        self.attention_level = 0.0
        self.stock_return = 0.0
        self._open_price: float | None = None
        self.gap_scale = 1.0
        self.amount_scale = 1.0
        self.mobile_share = mobile_share_default()
        self.decide_once = withdrawal_decision_once()
        self.initial_uninsured = np.zeros(0, dtype=np.float64)
        self.news_level = 0.0
        self.awareness_media_rate = awareness_media_rate_default()
        self.awareness_wom_rate = awareness_wom_rate_default()
        self.awareness_news_rate = awareness_news_rate_default()
        self.awareness_w1_prior = awareness_w1_prior_default()
        self.decision_to_request_rate = decision_to_request_rate_default()
        self.softness = 0.0
        self.backend = active_backend_name()
        self.worry: FloatArray | None = None
        self.belief: FloatArray | None = None
        self.segments: list[str] = []
        self.ids: list[str] = []
        self._index: dict[str, int] = {}
        self.balance = np.zeros(0, dtype=np.float64)
        self.insured = np.zeros(0, dtype=np.float64)
        self.base_threshold = np.zeros(0, dtype=np.float64)
        self.threshold_shift = np.zeros(0, dtype=np.float64)
        self.attention = np.zeros(0, dtype=np.float64)
        self.directive = np.zeros(0, dtype=np.float64)
        self.payroll = np.zeros(0, dtype=np.float64)
        self.statement_shift = np.zeros(0, dtype=np.float64)
        self.deposit_rate = np.zeros(0, dtype=np.float64)
        self.segment_fraction = np.zeros(0, dtype=np.float64)
        self.vip_amount = np.zeros(0, dtype=np.float64)
        self.already_withdrew = np.zeros(0, dtype=np.bool_)
        self.decided = np.zeros(0, dtype=np.float64)
        self.aware0 = float(type(self).aware0)
        self.aware = np.zeros(0, dtype=np.float64)
        self.contacts = np.zeros((0, 0), dtype=np.int64)
        self.hub_weight = np.zeros(0, dtype=np.float64)
        # SVB-58: off unless the registry row or the live server turns it on.
        refuse_under_gpu_calibration(_IGNORE_PRICE)
        self.ignore_price = flag_enabled(_IGNORE_PRICE)
        # SVB-62: cap attention at min(1, credibility / reference) (live opt-in).
        refuse_under_gpu_calibration(_CRED_CEILING)
        self.credibility_ceiling = flag_enabled(_CRED_CEILING)
        self._max_rumor_credibility: float | None = None
        self._ceiling_cursor = 0
        # SVB-59: a positive-tone rumor (lexicon) shifts worry down like a statement.
        refuse_under_gpu_calibration(_CURE_WORRY)
        self.cure_worry = flag_enabled(_CURE_WORRY)
        self._log_cursor = 0

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def load_depositor_arrays(self, arrays: DepositorArrays, *, seed: int) -> None:
        """Install a private copy of depositor rows and a seeded contact ring."""
        n = len(arrays.balance)
        if not (
            len(arrays.segment)
            == len(arrays.insured)
            == len(arrays.belief)
            == len(arrays.threshold)
            == len(arrays.needs_flag)
            == n
        ):
            raise ValueError("M06 depositor arrays must share one length")
        self.segments = list(arrays.segment)
        self.ids = [f"dep-{index}" for index in range(n)]
        self._index = {dep_id: index for index, dep_id in enumerate(self.ids)}
        self.balance = _f64(arrays.balance)
        self.insured = _f64(arrays.insured)
        belief = _f64(arrays.belief)
        if n and float(np.max(np.abs(belief))) == 0.0:
            belief = np.full(n, self.params.baseline_belief, dtype=np.float64)
        self.belief = belief
        self.base_threshold = _f64(arrays.threshold)
        self.threshold_shift = np.zeros(n, dtype=np.float64)
        self.attention = np.full(n, self.attention_level, dtype=np.float64)
        self.directive = np.zeros(n, dtype=np.float64)
        self.payroll = np.asarray(arrays.needs_flag, dtype=np.float64)
        self.statement_shift = np.zeros(n, dtype=np.float64)
        self.deposit_rate = np.zeros(n, dtype=np.float64)
        self.segment_fraction = np.asarray(
            [
                self.params.segment_fraction.get(segment, self.params.default_fraction)
                for segment in self.segments
            ],
            dtype=np.float64,
        )
        self.vip_amount = np.zeros(n, dtype=np.float64)
        self.already_withdrew = np.zeros(n, dtype=np.bool_)
        self.decided = np.zeros(n, dtype=np.float64)
        self.aware = np.full(n, float(getattr(self, "aware0", 0.0)), dtype=np.float64)
        self.initial_uninsured = np.maximum(0.0, self.balance - self.insured)
        self.contacts = build_contact_index(n, self.params.contact_degree, seed)
        self.hub_weight = hub_weights(self.balance, self.params)
        self.worry = None
        self._loaded = True

    def step(self, t: float, dt: float = 1.0) -> list[BaseModel]:
        """Advance one tick. No scenario attached → no events."""
        if self.scenario is None or self.bus is None:
            return []
        self._ensure_population()
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        if self._placebo or not self._loaded:
            return []
        self._apply_events(new_events)
        if self.cure_worry:
            self._apply_cures()
        result = self._decide(dt=float(dt))
        return self._withdrawals(t, result)

    @property
    def seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return self.scenario.seed

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_id_for(self.scenario.preset)

    def _ensure_population(self) -> None:
        if self._loaded or self._placebo or self.scenario is None:
            return
        if community_placebo(self.scenario.preset):
            self._placebo = True
            return
        self.load_depositor_arrays(
            _population_arrays(self.scenario),
            seed=self.scenario.seed,
        )

    def _applies(self, bank_id: str | None) -> bool:
        return bank_id is None or bank_id == self.bank_id

    def _attention_ceiling(self) -> float:
        """SVB-62: ``min(1, c / ref)`` for the bank's strongest live panic rumor.

        ``ref`` is the SVB fit rumor's credibility (``M06 attention ceiling reference
        credibility`` = 0.8), so any rumor at or above it keeps the clip at 1, as fitted.
        Positive-tone (cure) posts don't count. No rumor yet: 1 (no cap). Read from the
        bus log with its own cursor; ``consumes`` is unchanged.
        """
        if self.bus is not None:
            from agents.rumor_lexicon import lexicon_tone_and_strength, normalize

            log = self.bus.log
            for event in log[self._ceiling_cursor :]:
                if isinstance(event, RumorPosted) and self._applies(event.bank_id):
                    if lexicon_tone_and_strength(normalize(event.claim.strip()))[0] == "positive":
                        continue
                    credibility = float(event.credibility)
                    if self._max_rumor_credibility is None or credibility > self._max_rumor_credibility:
                        self._max_rumor_credibility = credibility
            self._ceiling_cursor = len(log)
        if self._max_rumor_credibility is None:
            return 1.0
        reference = float(get_parameter(_CRED_CEILING_REF).value)
        if reference <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, self._max_rumor_credibility / reference)))

    def _apply_events(self, events: list[BaseModel]) -> None:
        ceiling = self._attention_ceiling() if self.credibility_ceiling else 1.0
        for event in events:
            if isinstance(event, AttentionTick) and self._applies(event.bank_id):
                self.attention_level = float(np.clip(event.level, 0.0, ceiling))
            elif isinstance(event, PriceTick) and self._applies(event.bank_id):
                if not self.ignore_price:
                    self._apply_price(event.price)
            elif isinstance(event, StatementIssued) and self._applies(event.bank_id):
                self._apply_statement(event.segments)
            elif isinstance(event, PolicyEnacted) and self._applies(event.bank_id):
                self._apply_policy(event.lever)
            elif isinstance(event, DepositRateChanged) and self._applies(event.bank_id):
                self._apply_rate(event.segment, event.rate)
            elif isinstance(event, DirectiveIssued) and self._applies(event.bank_id):
                if event.action == "withdraw":
                    self._apply_directive(event.portfolio_ids)
            elif isinstance(event, VIPDecision) and self._applies(event.bank_id):
                self._apply_vip(event.agent_id, event.action, event.amount)

    def _apply_cures(self) -> None:
        """SVB-59: read cure rumors from the bus log (``consumes`` unchanged) and calm worry.

        Shift = the statement shifts times ``M06 cure rumor worry factor`` times the
        rumor's credibility, kept if calmer than the current shift. Depositors who
        already withdrew stay withdrawn (no redeposit).
        """
        if self.bus is None:
            return
        from agents.rumor_lexicon import lexicon_tone_and_strength, normalize

        log = self.bus.log
        factor = float(get_parameter(_CURE_FACTOR).value)
        for event in log[self._log_cursor :]:
            if not isinstance(event, RumorPosted) or not self._applies(event.bank_id):
                continue
            if lexicon_tone_and_strength(normalize(event.claim.strip()))[0] != "positive":
                continue
            weight = factor * float(event.credibility)
            logit = self.params.statement_logit_shift * weight
            threshold = self.params.statement_threshold_shift * weight
            if self.statement_shift.shape[0]:
                self.statement_shift = np.minimum(self.statement_shift, logit)
                self.threshold_shift = np.maximum(self.threshold_shift, threshold)
        self._log_cursor = len(log)

    def _apply_price(self, price: float) -> None:
        if self._open_price is None:
            self._open_price = price
        if self._open_price > 0.0:
            self.stock_return = price / self._open_price - 1.0

    def _apply_statement(self, segments: list[str]) -> None:
        targets = set(segments)
        for index, segment in enumerate(self.segments):
            if targets and segment not in targets:
                continue
            self.statement_shift[index] = self.params.statement_logit_shift
            self.threshold_shift[index] = self.params.statement_threshold_shift

    def _apply_policy(self, lever: str) -> None:
        if lever == "guarantee_deposits":
            self.gap_scale = self.params.guarantee_gap_scale
        elif lever == "cap_withdrawals":
            self.amount_scale = self.params.cap_amount_scale

    def _apply_rate(self, segment: str, rate: float) -> None:
        for index, name in enumerate(self.segments):
            if name == segment:
                self.deposit_rate[index] = rate

    def _apply_directive(self, portfolio_ids: list[str]) -> None:
        targets = set(portfolio_ids)
        if not targets:
            return
        for index, dep_id in enumerate(self.ids):
            if dep_id in targets or self.segments[index] in targets:
                self.directive[index] = 1.0

    def _apply_vip(self, agent_id: str, action: str, amount: float) -> None:
        if action != "withdraw":
            return
        index = self._index.get(agent_id)
        if index is None:
            return
        self.directive[index] = 1.0
        if amount > 0.0:
            self.vip_amount[index] = amount

    def _decide(self, dt: float = 1.0) -> DecisionResult:
        if self.belief is None:
            raise RuntimeError("M06 has no depositor belief")
        self.attention = np.full(self.balance.shape[0], self.attention_level, dtype=np.float64)
        inputs = DecisionInputs(
            attention=self.attention,
            belief=self.belief,
            threshold=self.base_threshold + self.threshold_shift,
            balance=self.balance,
            insured=self.insured,
            stock_return=self.stock_return,
            directive=self.directive,
            payroll=self.payroll,
            contacts=self.contacts,
            hub_weight=self.hub_weight,
            already_withdrew=self.already_withdrew,
            statement_shift=self.statement_shift,
            deposit_rate=self.deposit_rate,
            segment_fraction=self.segment_fraction,
            vip_amount=self.vip_amount,
            decided=getattr(self, "decided", None),
            aware=getattr(self, "aware", None),
            news=float(getattr(self, "news_level", 0.0) or 0.0),
        )
        result = evaluate_decisions(
            inputs,
            self.params,
            gap_scale=self.gap_scale,
            amount_scale=self.amount_scale,
            backend="auto",
            softness=self.softness,
            mobile_share=self.mobile_share,
            initial_uninsured=self.initial_uninsured,
            decide_once=self.decide_once,
            news=float(getattr(self, "news_level", 0.0) or 0.0),
            aware=getattr(self, "aware", None),
            awareness_media_rate=getattr(self, "awareness_media_rate", None),
            awareness_wom_rate=getattr(self, "awareness_wom_rate", None),
            awareness_news_rate=getattr(self, "awareness_news_rate", None),
            awareness_w1_prior=getattr(self, "awareness_w1_prior", None),
            decision_to_request_rate=getattr(self, "decision_to_request_rate", None),
            dt=float(dt),
        )
        self.backend = result.backend
        if result.decided is not None:
            self.decided = result.decided
        if result.aware is not None:
            self.aware = result.aware
        self.worry = result.worry
        self.belief = result.belief
        self.balance = np.maximum(0.0, self.balance - result.amount)
        self.already_withdrew = self.already_withdrew | result.withdraw
        self.vip_amount = np.zeros(self.balance.shape[0], dtype=np.float64)
        return result

    def _withdrawals(self, t: float, result: DecisionResult) -> list[BaseModel]:
        if self.scenario is None:
            return []
        if self.decide_once and float(self.softness) > 0.0:
            return self._soft_withdrawals(t, result)
        out: list[BaseModel] = []
        for index in np.flatnonzero(result.withdraw):
            row = int(index)
            amount = float(result.amount[row])
            if amount < self.params.minimum_amount:
                continue
            segment = self.segments[row]
            channel = self.params.segment_channel.get(segment, self.params.default_channel)
            self._seq += 1
            out.append(
                WithdrawalRequested(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=f"{self.module_id}-wd-{self._seq}",
                    depositor_id=self.ids[row],
                    amount=amount,
                    channel=channel,
                )
            )
        return out

    def _soft_withdrawals(self, t: float, result: DecisionResult) -> list[BaseModel]:
        """SVB-61: soft decide-once requests, aggregated per (segment, channel) per tick.

        In soft mode (softness > 0, the SVB fit's behavioral noise) each depositor
        requests an increment ``(d_new - decided) * amount`` every tick. The batched
        GPU kernel counts every increment; one event per depositor per tick would be
        ~N events a tick. So each tick publishes one request per (segment, channel)
        with the sum of its increments (depositor id ``agg:<segment>``): totals equal
        the kernel's, with no carry bias. M07 limits are bank-wide and M08 does not
        read depositor ids. ``withdraw`` (decided >= 0.5) still marks who has
        withdrawn for the crowd. Before SVB-61 only depositors with decided >= 0.5
        published, so soft mode requested ~0 (diag-08).
        """
        amount = np.asarray(result.amount, dtype=np.float64)
        totals: dict[tuple[str, str], float] = {}
        for index in np.flatnonzero(amount > 0.0):
            row = int(index)
            segment = self.segments[row]
            channel = self.params.segment_channel.get(segment, self.params.default_channel)
            key = (segment, channel)
            totals[key] = totals.get(key, 0.0) + float(amount[row])
        out: list[BaseModel] = []
        for (segment, channel), total in sorted(totals.items()):
            if total < self.params.minimum_amount:
                continue
            self._seq += 1
            out.append(
                WithdrawalRequested(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=f"{self.module_id}-wd-{self._seq}",
                    depositor_id=f"agg:{segment}",
                    amount=total,
                    channel=channel,
                )
            )
        return out
