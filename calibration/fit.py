"""M27 calibration and validation harness (§5, §18.6, §19)."""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from contracts.events import BankClosed, PaymentSettled, PriceTick, WithdrawalRequested
from contracts.schemas import Rumor, Scenario
from params.loader import RegistryError, get_parameter

_PARAM_BEHAVIORAL_DIM = "Calibration behavioral parameter dimension"
_PARAM_CMA_GENERATIONS = "Calibration CMA-ES max generations"
_PARAM_CMA_POPULATION = "Calibration CMA-ES population size"
_PARAM_EVAL_MINUTES = "Calibration eval simulation minutes"
_PARAM_E1_DAY1_WEIGHT = "Calibration E1 day-1 objective weight"
_PARAM_E1_DAY2_WEIGHT = "Calibration E1 two-day objective weight"
_PARAM_E1_REQUESTED_DAY1 = "Calibration E1 requested day-one target"
_PARAM_E1_STOCK_WEIGHT = "Calibration E1 day-1 stock objective weight"
_PARAM_E1_STOCK_TARGET = "Calibration E1 day-1 stock target"
_PARAM_MINUTES_PER_DAY = "Non-bank sim minutes per day"
_PARAM_IDENTIFIABLE = "Calibration identifiable parameter indices"
_PARAM_SHARED_SEEDS = "Calibration shared eval seed count"
_PARAM_SOFT_INITIAL = "Calibration softness initial"
_PARAM_SOFT_FINAL = "Calibration softness final"
_PARAM_BEHAVIORAL_NOISE = "Calibration behavioral noise temperature"
_PARAM_PRIOR_STRENGTH = "Calibration log-space prior strength"
_PARAM_BRACKET_GRID = "Calibration bracket grid size"
_PARAM_LOG_BOUND = "Calibration log-param bound"
_PARAM_BIAS_BOUND = "Calibration bias-param bound"
_PARAM_RESTARTS = "Calibration CMA-ES restarts"
_PARAM_INSURANCE = "Insurance limit"
_PARAM_DEPOSITS = "SVB-like preset: deposits"
_PARAM_SVB_DAY1 = "SVB-like: March 9 withdrawn"
_PARAM_SVB_DAY2 = "SVB-like: March 10 scheduled withdrawals"
_PARAM_SVB_CUMULATIVE = "SVB-like: two-day cumulative outflow share"
_PARAM_SIGNATURE_DAY1 = "Signature-like: first-day outflow"
# SVB-27d: requested (not settled) day-one fraction for the E5 holdout. Settled day one is bound by the
# shared M07 capacity multiplier, so every bank above capacity prints the same settled number.
_PARAM_SIGNATURE_REQUESTED = "Validation target value: E5 Signature first-day requested fraction"
_TOL_E1_DAY1 = "Validation target: E1 SVB day 1 outflow %"
_TOL_E1_DAY2 = "Validation target: E1 SVB 2-day cumulative outflow"
_TOL_E5_DAY1 = "Validation target: E5 Signature first-day outflow %"
_TOL_PLACEBO = "Validation target: placebo insured community bank survives"

_CALIB_REAL_MODULES: tuple[str, ...] = ("M02", "M03", "M05", "M06", "M07", "M08", "M09")
_DAY1_CLOSE = 390
_NEXT_OPEN = 390 + 1050
_OVERNIGHT_DT = 15
_PARAM_PAYMENT_CAPACITY = "M07 capacity multiplier (FIT)"
_STAGE2_GRID = 16
_STAGE2_LO = 0.05
_STAGE2_HI = 20.0
_STAGE2_TOL = 0.002
_STAGE2_REFINE_RATIO = 1.2
_TOL_NEXT_OPEN = "Validation target: requests by next open"
_TOL_SETTLED_DAY1 = "Validation target: settled day one"

_CANDIDATE_FAILURE_PENALTY = 1e9
_WORKER_LOGGER = logging.getLogger("m27.calibration.worker")

_FITTED_M06_ROWS: tuple[str, ...] = (
    "M06 worry weights (ASSUMED)",
    "M06 worry bias (ASSUMED)",
    "M06 complex contagion k (ASSUMED)",
    "M06 belief update strength (ASSUMED)",
    "M06 public signal attention weight (ASSUMED)",
    "Depositor mobile share of uninsured deposits (FIT)",
)
_FITTED_ROW_FOR_COORD: dict[str, str] = {
    "attention_weight": "M06 worry weights (ASSUMED)",
    "contact_weight": "M06 worry weights (ASSUMED)",
    "belief_weight": "M06 worry weights (ASSUMED)",
    "stock_weight": "M06 worry weights (ASSUMED)",
    "directive_weight": "M06 worry weights (ASSUMED)",
    "payroll_weight": "M06 worry weights (ASSUMED)",
    "gap_weight": "M06 worry weights (ASSUMED)",
    "worry_bias": "M06 worry bias (ASSUMED)",
    "contagion_k": "M06 complex contagion k (ASSUMED)",
    "amount_scale": "Calibration M06 amount scale (ASSUMED)",
    "belief_strength": "M06 belief update strength (ASSUMED)",
    "signal_attention": "M06 public signal attention weight (ASSUMED)",
    "feedback_gain": "M03 feedback gain (ASSUMED)",
    "breaking_point_dispersion": "M06 breaking-point dispersion (ASSUMED)",
    "price_impact": "M14 price impact (ASSUMED)",
    "mobile_share": "Depositor mobile share of uninsured deposits (FIT)",
    "awareness_media_rate": "M06 awareness media rate (FIT)",
    "decision_to_request_rate": "M07 decision-to-request rate (FIT)",
}


@dataclass(frozen=True, slots=True)
class CalibrationTargets:
    svb_day1_fraction: float
    svb_day2_increment_fraction: float
    svb_two_day_fraction: float
    svb_requested_day1_fraction: float
    signature_day1_fraction: float
    signature_day1_requested_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class ToleranceBand:
    relative: float | None
    absolute: float | None


@dataclass(frozen=True, slots=True)
class ValidationBoardRow:
    target_id: str
    label: str
    observed: float
    expected: float
    tolerance: ToleranceBand
    passed: bool
    detail: str


@dataclass
class FitResult:
    seed: int
    vector: tuple[float, ...]
    objective: float
    generations: int
    log_params: tuple[float, ...] = ()
    bound_flags: tuple[str, ...] = ()
    identifiable: tuple[str, ...] = ()


@dataclass
class HoldoutResult:
    signature_day1_fraction: float
    placebo_survives: bool
    signature_day1_requested: float | None = None


@dataclass(frozen=True, slots=True)
class CapsOnValidation:
    """E1 check with payment caps and closure ON (hard decisions)."""

    day1_settled: float
    two_day_settled: float
    day1_requested: float
    two_day_requested: float
    closed: bool


_VECTOR_PARAM_NAMES: tuple[str, ...] = (
    "attention_weight",
    "contact_weight",
    "belief_weight",
    "stock_weight",
    "directive_weight",
    "payroll_weight",
    "gap_weight",
    "worry_bias",
    "contagion_k",
    "amount_scale",
    "belief_strength",
    "signal_attention",
    "feedback_gain",
    "breaking_point_dispersion",
    "price_impact",
    "mobile_share",
    "decision_to_request_rate",
)
_BIAS_INDEX = 7
_IMPACT_INDEX = 14
_MOBILE_SHARE_INDEX = 15
_DECISION_RATE_INDEX = 16
_LIGHT_BOOK_SIZE = 48
_LIGHT_STEPS = 10


def _parse_money(value: str) -> float:
    lowered = value.strip().lower().replace(",", "").replace("$", "")
    match = re.search(r"([0-9]*\.?[0-9]+)\s*([a-z]*)", lowered)
    if match is None:
        raise RegistryError(f"cannot parse money value: {value!r}")
    number = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("b"):
        return number * 1_000_000_000.0
    if unit.startswith("m"):
        return number * 1_000_000.0
    return number


def _parse_fraction_cell(value: str) -> float:
    lowered = value.strip().lower()
    if "%" in lowered:
        match = re.search(r"([0-9]*\.?[0-9]+)\s*%", lowered)
        if match is None:
            raise RegistryError(f"cannot parse fraction: {value!r}")
        return float(match.group(1)) / 100.0
    match = re.search(r"([0-9]*\.?[0-9]+)", lowered)
    if match is None:
        raise RegistryError(f"cannot parse fraction: {value!r}")
    number = float(match.group(1))
    return number / 100.0 if number > 1.0 else number


def _parse_tolerance(value: str) -> ToleranceBand:
    text = value.strip()
    rel = re.search(r"±\s*([0-9]*\.?[0-9]+)\s*%\s*relative", text, flags=re.IGNORECASE)
    if rel is not None:
        return ToleranceBand(relative=float(rel.group(1)) / 100.0, absolute=None)
    plain = re.search(r"±\s*([0-9]*\.?[0-9]+)\s*%", text)
    if plain is not None:
        return ToleranceBand(relative=float(plain.group(1)) / 100.0, absolute=None)
    return ToleranceBand(relative=None, absolute=None)


def _registry_float(name: str, path: Path | None = None) -> float:
    raw = get_parameter(name, path).value
    try:
        return float(raw)
    except ValueError:
        parsed = re.search(r"([0-9]*\.?[0-9]+)", raw)
        if parsed is None:
            msg = f"cannot parse numeric registry value for {name}: {raw!r}"
            raise RegistryError(msg) from None
        return float(parsed.group(1))


def load_calibration_targets(path: Path | None = None) -> CalibrationTargets:
    """E1 and E5 fractions used for fit and holdout (§8.2)."""
    deposits = _parse_money(get_parameter(_PARAM_DEPOSITS, path).value)
    day1 = _parse_money(get_parameter(_PARAM_SVB_DAY1, path).value) / deposits
    day2_inc = _parse_money(get_parameter(_PARAM_SVB_DAY2, path).value) / deposits
    cumulative = _parse_fraction_cell(get_parameter(_PARAM_SVB_CUMULATIVE, path).value)
    requested_day1 = _registry_float(_PARAM_E1_REQUESTED_DAY1, path)
    signature = _parse_fraction_cell(get_parameter(_PARAM_SIGNATURE_DAY1, path).value)
    try:
        signature_requested: float | None = _parse_fraction_cell(
            get_parameter(_PARAM_SIGNATURE_REQUESTED, path).value
        )
    except RegistryError:
        signature_requested = None  # trimmed registries: E5 falls back to the settled comparison
    return CalibrationTargets(
        svb_day1_fraction=day1,
        svb_day2_increment_fraction=day2_inc,
        svb_two_day_fraction=cumulative,
        svb_requested_day1_fraction=requested_day1,
        signature_day1_fraction=signature,
        signature_day1_requested_fraction=signature_requested,
    )


def day1_close_minute(path: Path | None = None) -> float:
    """Minute of day-one close on the SVB-29 trading-day clock (one simulated session)."""
    return _registry_float(_PARAM_MINUTES_PER_DAY, path)


def e1_stock_ratio_summary(
    observed: float | None,
    *,
    path: Path | None = None,
) -> dict[str, float | None]:
    """Observed vs target for the E1 day-one stock ratio (1 + return)."""
    return {
        "observed": None if observed is None else float(observed),
        "expected": _registry_float(_PARAM_E1_STOCK_TARGET, path),
    }


def use_real_calibration_engine() -> bool:
    """Real M02–M09 evals when ``STAMPEDE_CALIBRATION_ENGINE=real``."""
    return os.environ.get("STAMPEDE_CALIBRATION_ENGINE", "fake").strip().lower() == "real"


def configure_calibration_gpu(enable: bool) -> str:
    """Set ``STAMPEDE_BACKEND=gpu`` for calibration; return the M06 backend label.

    With ``enable=True``, M05 uses the CuPy balance solve, and M02/M03/M06 use CuPy when a
    device is visible. CPU CI keeps the NumPy reference path when CUDA is absent.
    """
    if enable:
        os.environ["STAMPEDE_BACKEND"] = "gpu"
    from engine.agents.decide import active_backend_name

    return active_backend_name()


def eval_simulation_minutes(path: Path | None = None) -> int:
    """Simulated minutes per candidate eval (env override for GPU runs)."""
    override = os.environ.get("STAMPEDE_CALIB_EVAL_MINUTES", "").strip()
    if override:
        return max(1, int(override))
    return int(_registry_float(_PARAM_EVAL_MINUTES, path))


def default_tick_schedule() -> list[tuple[float, float]]:
    """390 one-minute ticks, then 70 fifteen-minute overnight ticks (about 1.2×)."""
    day = [(float(t), 1.0) for t in range(int(_DAY1_CLOSE))]
    overnight = [
        (float(_DAY1_CLOSE + index * _OVERNIGHT_DT), float(_OVERNIGHT_DT))
        for index in range(int((_NEXT_OPEN - _DAY1_CLOSE) / _OVERNIGHT_DT))
    ]
    return day + overnight


def day1_tick_schedule() -> list[tuple[float, float]]:
    """One-minute ticks through day-one close."""
    return [(float(t), 1.0) for t in range(int(_DAY1_CLOSE))]


def stage1_schedule() -> list[tuple[float, float]]:
    """Stage 1 always uses the overnight clock through next open."""
    return default_tick_schedule()


def _assert_reaches_next_open(ticks: list[tuple[float, float]]) -> None:
    if ticks[-1][0] + ticks[-1][1] < _NEXT_OPEN:
        raise ValueError("stage-1 schedule must reach next open")


def resolve_tick_schedule(
    steps: int,
    schedule: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Explicit schedule, overnight clock when ``steps`` covers a trading day, else 1-min ticks."""
    if schedule is not None:
        return [(float(t), float(dt)) for t, dt in schedule]
    if steps >= int(_NEXT_OPEN):
        return default_tick_schedule()
    if steps >= int(_DAY1_CLOSE):
        return day1_tick_schedule()
    return [(float(t), 1.0) for t in range(max(0, int(steps)))]


def real_calibration_modes() -> dict[str, Any]:
    from engine.modes import ModuleMode

    return {module_id: ModuleMode.REAL for module_id in _CALIB_REAL_MODULES}


def identifiable_indices(path: Path | None = None) -> tuple[int, ...]:
    """Stage-1 coordinates from the registry (worry bias, mobile share, h)."""
    raw = get_parameter(_PARAM_IDENTIFIABLE, path).value
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def identifiable_names(path: Path | None = None) -> tuple[str, ...]:
    return tuple(_VECTOR_PARAM_NAMES[index] for index in identifiable_indices(path))


def shared_eval_seeds(base_seed: int, path: Path | None = None) -> tuple[int, ...]:
    """Same seeds for every candidate. Derived only from the fit seed."""
    count = max(1, int(_registry_float(_PARAM_SHARED_SEEDS, path)))
    seq = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF, 0x4D273001])
    return tuple(int(x) for x in seq.generate_state(count, dtype=np.uint32))


def softness_schedule(generation: int, max_generations: int, path: Path | None = None) -> float:
    """Deprecated: annealing reintroduces the cliff (design v7.0)."""
    initial = _registry_float(_PARAM_SOFT_INITIAL, path)
    final = _registry_float(_PARAM_SOFT_FINAL, path)
    if max_generations <= 1:
        return float(final)
    frac = min(1.0, max(0.0, generation / (max_generations - 1)))
    if initial <= 0.0 or final <= 0.0:
        return float(final)
    return float(initial * (final / initial) ** frac)


def fixed_softness(path: Path | None = None) -> float:
    """Fixed behavioral-noise temperature for fit, validation, and holdouts."""
    return _registry_float(_PARAM_BEHAVIORAL_NOISE, path)


def _theta_bounds(index: int, path: Path | None = None) -> tuple[float, float]:
    if index == _MOBILE_SHARE_INDEX:
        from engine.agents.decide import mobile_share_bounds, mobile_share_default

        lo, hi = mobile_share_bounds(path)
        default = mobile_share_default(path)
        return math.log(lo / default), math.log(hi / default)
    if index == _DECISION_RATE_INDEX:
        from engine.agents.decide import (
            decision_to_request_rate_bounds,
            decision_to_request_rate_default,
        )

        lo, hi = decision_to_request_rate_bounds(path)
        default = decision_to_request_rate_default(path)
        return math.log(lo / default), math.log(hi / default)
    if index == _IMPACT_INDEX:
        from engine.gpu_full_io import price_impact_bounds, price_impact_default

        lo, hi = price_impact_bounds(path)
        default = price_impact_default(path)
        return math.log(lo / default), math.log(hi / default)
    width = _registry_float(_PARAM_BIAS_BOUND if index == _BIAS_INDEX else _PARAM_LOG_BOUND, path)
    return -width, width


def unit_from_theta(index: int, theta: float) -> float:
    """Map a log-space coordinate onto the behavioral-vector unit cube (unclamped)."""
    if index == _BIAS_INDEX:
        return float(theta) / 4.0 + 0.5
    return float(theta) / 0.8 + 0.5


def theta_from_unit(index: int, unit: float) -> float:
    if index == _BIAS_INDEX:
        return 4.0 * (float(unit) - 0.5)
    return 0.8 * (float(unit) - 0.5)


def vector_from_log_params(
    theta: NDArray[np.float64],
    indices: tuple[int, ...],
    *,
    dim: int | None = None,
    path: Path | None = None,
) -> NDArray[np.float64]:
    width = dim if dim is not None else int(_registry_float(_PARAM_BEHAVIORAL_DIM, path))
    vector = np.full(width, 0.5, dtype=np.float64)
    for value, index in zip(theta, indices, strict=True):
        if 0 <= index < width:
            vector[index] = unit_from_theta(index, float(value))
    return vector


def log_params_from_vector(
    vector: NDArray[np.float64],
    indices: tuple[int, ...],
) -> NDArray[np.float64]:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    mapped = [
        theta_from_unit(index, float(values[index]) if index < values.size else 0.5)
        for index in indices
    ]
    return np.asarray(mapped, dtype=np.float64)


def bound_flags_for(
    theta: NDArray[np.float64],
    indices: tuple[int, ...],
    path: Path | None = None,
) -> tuple[str, ...]:
    flags: list[str] = []
    for value, index in zip(theta, indices, strict=True):
        lo, hi = _theta_bounds(index, path)
        if float(value) <= lo + 1e-12 or float(value) >= hi - 1e-12:
            flags.append(_VECTOR_PARAM_NAMES[index])
    return tuple(flags)


def _prior_penalty(theta: NDArray[np.float64], path: Path | None = None) -> float:
    strength = _registry_float(_PARAM_PRIOR_STRENGTH, path)
    return float(strength * np.dot(theta, theta))


def _project_theta(
    theta: NDArray[np.float64],
    indices: tuple[int, ...],
    *,
    path: Path | None = None,
) -> NDArray[np.float64]:
    projected = np.asarray(theta, dtype=np.float64).copy()
    for i, index in enumerate(indices):
        lo, hi = _theta_bounds(index, path)
        projected[i] = min(hi, max(lo, float(projected[i])))
    return projected


def _light_requested_fractions(
    vector: NDArray[np.float64],
    *,
    seed: int,
    softness: float,
    preset: str = "tech-concentrated",
    rumor_strength: float = 1.0,
) -> tuple[float, float]:
    """Requested day-1 / two-day fractions on a small book (no payments, no closure)."""
    from engine.agents.decide import (
        DecisionInputs,
        build_contact_index,
        calibration_amount_scale,
        calibration_awareness_media_rate,
        calibration_decision_to_request_rate,
        calibration_mobile_share,
        decision_params_from_vector,
        evaluate_decisions,
        hub_weights,
        load_decision_params,
        withdrawal_decision_once,
    )
    from engine.agents.population import segment_weights_for_preset

    if "community" in preset.lower():
        return 0.0, 0.0
    params = decision_params_from_vector(load_decision_params(), vector, clamp_inputs=False)
    amount_scale = calibration_amount_scale(vector, clamp_inputs=False)
    mobile_share = calibration_mobile_share(vector, clamp_inputs=False)
    media_rate = calibration_awareness_media_rate(vector, clamp_inputs=False)
    request_rate = calibration_decision_to_request_rate(vector, clamp_inputs=False)
    decide_once = withdrawal_decision_once()
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    n = _LIGHT_BOOK_SIZE
    balance = np.exp(rng.normal(14.8, 1.1, size=n))
    insured = np.minimum(balance, _parse_money(get_parameter(_PARAM_INSURANCE).value))
    threshold = rng.uniform(0.32, 0.62, size=n)
    payroll = (rng.random(n) < 0.25).astype(np.float64)
    contacts = build_contact_index(n, params.contact_degree, seed)
    hub = hub_weights(balance, params)
    belief = np.full(n, params.baseline_belief, dtype=np.float64)
    already = np.zeros(n, dtype=np.bool_)
    weights = segment_weights_for_preset(preset)
    names = list(weights)
    probs = np.asarray([weights[name] for name in names], dtype=np.float64)
    probs = probs / probs.sum()
    segments = rng.choice(names, size=n, p=probs)
    segment_fraction = np.asarray(
        [params.segment_fraction.get(seg, params.default_fraction) for seg in segments],
        dtype=np.float64,
    )
    zeros = np.zeros(n, dtype=np.float64)
    book = float(balance.sum())
    initial_uninsured = np.maximum(0.0, balance - insured)
    decided = already.astype(np.float64)
    aware = np.zeros(n, dtype=np.float64)
    day1 = 0.0
    two_day = 0.0
    split = _LIGHT_STEPS // 2
    impulse = min(1.0, max(0.0, float(rumor_strength)))
    for tick in range(_LIGHT_STEPS):
        attention = min(0.95, (0.12 + 0.09 * tick) * impulse)
        stock = -0.03 * tick
        result = evaluate_decisions(
            DecisionInputs(
                attention=np.full(n, attention, dtype=np.float64),
                belief=belief,
                threshold=threshold,
                balance=balance,
                insured=insured,
                stock_return=stock,
                directive=zeros,
                payroll=payroll,
                contacts=contacts,
                hub_weight=hub,
                already_withdrew=already,
                statement_shift=zeros,
                deposit_rate=zeros,
                segment_fraction=segment_fraction,
                vip_amount=zeros,
                decided=decided,
                aware=aware,
            ),
            params,
            amount_scale=amount_scale,
            backend="cpu",
            softness=softness,
            mobile_share=mobile_share,
            initial_uninsured=initial_uninsured,
            decide_once=decide_once,
            awareness_media_rate=media_rate,
            decision_to_request_rate=request_rate,
        )
        paid = float(result.amount.sum())
        if tick < split:
            day1 += paid
        two_day += paid
        already = already | result.withdraw
        belief = result.belief
        balance = np.maximum(0.0, balance - result.amount)
        if result.decided is not None:
            decided = result.decided
        if result.aware is not None:
            aware = result.aware
    if book <= 0.0:
        return 0.0, 0.0
    return day1 / book, two_day / book


def _disable_caps_and_closure(engine: Any) -> None:
    """Fit-only: payment capacities go unused and the closure rule does not fire."""
    from dataclasses import replace

    payments = engine.get("M07")
    if payments is not None and hasattr(payments, "limits"):
        payments.limits = replace(
            payments.limits,
            app_daily_limit=1e30,
            wire_capacity_per_tick=1e30,
            ach_batch_capacity=1e30,
            branch_capacity_per_tick=1e30,
        )
    regulator = engine.get("M20")
    if regulator is not None:
        regulator.closed = True


def diagnostics_dir() -> Path:
    return Path(__file__).resolve().parent / "diagnostics"


def _log_sq_error(observed: float, target: float) -> float:
    floor = 1e-4
    return (math.log(max(observed, floor)) - math.log(max(target, floor))) ** 2


def _score_day1_stock_ratio(log: list[BaseModel], *, path: Path | None = None) -> float | None:
    """``1 + day-one price change`` = last PriceTick before close / opening print."""
    close = day1_close_minute(path)
    last: float | None = None
    first: float | None = None
    for event in log:
        if isinstance(event, PriceTick) and float(event.t) < close:
            if first is None:
                first = float(event.price)
            last = float(event.price)
    if last is None or first is None or first <= 0.0:
        return None
    return last / first


def _score_outflow_log(
    log: list[BaseModel],
    *,
    amount_scale: float = 1.0,
    requested: bool = False,
    path: Path | None = None,
) -> tuple[float, float, bool]:
    """Day-one (t < trading-session close) fraction, next-open (t < 1440) fraction, closed?"""
    deposits = _parse_money(get_parameter(_PARAM_DEPOSITS, path).value)
    close = day1_close_minute(path)
    day1_paid = 0.0
    next_open_paid = 0.0
    closed = False
    event_cls = WithdrawalRequested if requested else PaymentSettled
    for event in log:
        if isinstance(event, BankClosed):
            closed = True
        if isinstance(event, event_cls):
            amount = event.amount * amount_scale
            if event.t < close:
                day1_paid += amount
            if event.t < _NEXT_OPEN:
                next_open_paid += amount
    day1_fraction = day1_paid / deposits if deposits else 0.0
    next_open_fraction = next_open_paid / deposits if deposits else 0.0
    return day1_fraction, next_open_fraction, closed


def _gpu_sim_requested() -> bool:
    """Batched path only for real-engine evals with both GPU flags set."""
    from engine.gpu_sim import gpu_sim_enabled

    return use_real_calibration_engine() and gpu_sim_enabled()


def measure_outflow(
    scenario: Scenario,
    *,
    steps: int,
    param_vector: NDArray[np.float64] | None = None,
    requested: bool = False,
    disable_caps: bool = False,
    softness: float = 0.0,
    clamp_inputs: bool = False,
    schedule: list[tuple[float, float]] | None = None,
) -> tuple[float, float, bool]:
    """Return day-one fraction, next-open fraction, and whether the bank closed.

    Default (validation / gpu-sim parity): settled outflow, caps and closure on,
    hard decisions, unclamped inputs. The fit path passes ``requested=True``,
    ``disable_caps=True``, and a softness temperature. ``schedule`` is ``(t, dt)``
    pairs on the simulation clock; the CPU ``engine.run`` path still uses 1-minute
    ticks and cannot take ``dt``.
    """
    from engine.agents.decide import DecisionModule, apply_calibration_vector
    from engine.modes import ModuleMode
    from engine.registry import build_engine, run_scenario

    ticks = resolve_tick_schedule(steps, schedule)
    if _gpu_sim_requested() and param_vector is not None:
        from engine.gpu_sim import simulate_logs

        logs = simulate_logs(
            scenario,
            steps,
            [param_vector],
            disable_caps=disable_caps,
            softness=softness,
            clamp_inputs=clamp_inputs,
            schedule=ticks,
        )
        return _score_outflow_log(logs[0], requested=requested)
    if use_real_calibration_engine():
        engine = build_engine(
            modes=real_calibration_modes(),
            default_mode=ModuleMode.FAKE,
            scenario=scenario,
        )
        if param_vector is not None:
            module = engine["M06"]
            if isinstance(module, DecisionModule):
                apply_calibration_vector(module, param_vector, clamp_inputs=clamp_inputs)
                from engine.gpu_full_io import apply_lane_feedback

                apply_lane_feedback(module, param_vector)  # SVB-65: gain, dispersion, impact
                module.softness = float(softness)
        if disable_caps:
            _disable_caps_and_closure(engine)
        log = engine.run(steps)
    else:
        if param_vector is not None:
            strength = 0.0
            for rumor in scenario.rumors:
                strength = max(strength, max(0.0, min(1.0, float(rumor.credibility))))
            if strength <= 0.0:
                strength = 1.0
            day1, two_day = _light_requested_fractions(
                param_vector,
                seed=scenario.seed,
                softness=softness,
                preset=scenario.preset,
                rumor_strength=strength,
            )
            return day1, two_day, False
        log = run_scenario(scenario, steps)
    return _score_outflow_log(log, requested=requested)


def objective_e1(
    vector: NDArray[np.float64],
    *,
    scenario: Scenario,
    steps: int,
    targets: CalibrationTargets,
    w_day1: float,
    w_day2: float,
    softness: float | None = None,
    path: Path | None = None,
    average_seeds: bool = True,
) -> float:
    """Stage-1 requested-outflow loss (day-one close vs 0.249, next open vs 0.81)."""
    temp = fixed_softness(path) if softness is None else float(softness)
    seeds = shared_eval_seeds(scenario.seed, path) if average_seeds else (scenario.seed,)
    ticks = stage1_schedule()
    _assert_reaches_next_open(ticks)
    losses: list[float] = []
    if _gpu_sim_requested():
        from engine.gpu_sim import simulate_requested_fractions

        fractions = simulate_requested_fractions(
            scenario,
            ticks,
            [vector],
            seeds,
            softness=temp,
            clamp_inputs=False,
        )
        for row in fractions:
            day1, two_day, stock = (float(value) for value in row)
            losses.append(
                _objective_from_fractions(
                    day1,
                    two_day,
                    targets=targets,
                    w_day1=w_day1,
                    w_day2=w_day2,
                    stock_ratio=stock,
                    path=path,
                )
            )
    else:
        for seed in seeds:
            seeded = scenario.model_copy(update={"seed": int(seed)})
            day1, two_day, _ = measure_outflow(
                seeded,
                steps=steps,
                param_vector=vector,
                requested=True,
                disable_caps=True,
                softness=temp,
                clamp_inputs=False,
            )
            losses.append(
                _objective_from_fractions(
                    day1,
                    two_day,
                    targets=targets,
                    w_day1=w_day1,
                    w_day2=w_day2,
                )
            )
    loss = float(np.mean(losses))
    indices = identifiable_indices(path)
    theta = log_params_from_vector(np.asarray(vector, dtype=np.float64), indices)
    return loss + _prior_penalty(theta, path)


def candidate_eval_seed(fit_seed: int, generation: int, index: int) -> int:
    """Stable per-candidate seed for parallel eval (fit seed + generation + index only)."""
    seq = np.random.SeedSequence([int(fit_seed) & 0xFFFFFFFF, int(generation), int(index)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # Torch allows this once per process; later calls (in-process
                # workers, repeated fits in one pytest run) keep the first value.
                pass
    except ImportError:
        pass


@dataclass(frozen=True, slots=True)
class _CandidateEvalTask:
    index: int
    vector: tuple[float, ...]
    fit_seed: int
    generation: int
    scenario: Scenario
    steps: int
    targets: CalibrationTargets
    w_day1: float
    w_day2: float
    use_fake_objective: bool
    softness: float = 0.15


def _evaluate_candidate_task(task: _CandidateEvalTask) -> tuple[int, float, float, str | None]:
    """Worker entry: returns (index, objective, seconds, error_message)."""
    _limit_worker_threads()
    candidate_seed = candidate_eval_seed(task.fit_seed, task.generation, task.index)
    started = time.perf_counter()
    try:
        vector = np.asarray(task.vector, dtype=np.float64)
        if task.use_fake_objective:
            rng = np.random.default_rng(candidate_seed)
            objective = float(np.sum(vector**2) + rng.integers(0, 7) * 1e-9)
        else:
            objective = objective_e1(
                vector,
                scenario=task.scenario,
                steps=task.steps,
                targets=task.targets,
                w_day1=task.w_day1,
                w_day2=task.w_day2,
                softness=task.softness,
            )
        return task.index, objective, time.perf_counter() - started, None
    except Exception as exc:
        _WORKER_LOGGER.exception(
            "candidate eval failed seed=%s gen=%s index=%s",
            task.fit_seed,
            task.generation,
            task.index,
        )
        return task.index, _CANDIDATE_FAILURE_PENALTY, time.perf_counter() - started, str(exc)


def _evaluate_one_candidate(
    *,
    index: int,
    vector: NDArray[np.float64],
    fit_seed: int,
    generation: int,
    scenario: Scenario,
    steps: int,
    targets: CalibrationTargets,
    w_day1: float,
    w_day2: float,
    use_fake_objective: bool,
    softness: float,
) -> tuple[float, float, str | None]:
    task = _CandidateEvalTask(
        index=index,
        vector=tuple(float(x) for x in vector),
        fit_seed=fit_seed,
        generation=generation,
        scenario=scenario,
        steps=steps,
        targets=targets,
        w_day1=w_day1,
        w_day2=w_day2,
        use_fake_objective=use_fake_objective,
        softness=softness,
    )
    _, objective, elapsed, error = _evaluate_candidate_task(task)
    return objective, elapsed, error


def _objective_from_fractions(
    day1: float,
    two_day: float,
    *,
    targets: CalibrationTargets,
    w_day1: float,
    w_day2: float,
    stock_ratio: float | None = None,
    path: Path | None = None,
) -> float:
    """Stage-1 loss: requests by day-one close vs 0.249, next open vs 0.81, stock vs 0.40.

    Signature (E5) is a holdout and is not a term here. SVB-36: the stock term is
    skipped when its registry weight is 0 (the default), so the E1 stock ratio is
    reported on the board and in summary.json but not fitted.
    """
    loss = w_day1 * _log_sq_error(day1, targets.svb_requested_day1_fraction)
    loss += w_day2 * _log_sq_error(two_day, targets.svb_two_day_fraction)
    if stock_ratio is not None:
        weight = _registry_float(_PARAM_E1_STOCK_WEIGHT, path)
        if weight > 0.0:
            target = _registry_float(_PARAM_E1_STOCK_TARGET, path)
            loss += weight * _log_sq_error(stock_ratio, target)
    return loss


def _evaluate_population_gpu_sim(
    samples: list[NDArray[np.float64]],
    *,
    fit_seed: int,
    scenario: Scenario,
    steps: int,
    targets: CalibrationTargets,
    w_day1: float,
    w_day2: float,
    softness: float,
    on_candidate: Callable[[int, float, float], None] | None,
    on_candidate_error: Callable[[int, str], None] | None,
    path: Path | None = None,
) -> list[float]:
    """Score a generation as batched requested-outflow runs over shared seeds."""
    from engine.gpu_sim import simulate_requested_fractions

    softness = fixed_softness(path)
    logger = logging.getLogger("m27.calibration")
    started = time.perf_counter()
    seeds = shared_eval_seeds(fit_seed, path)
    totals = [0.0 for _ in samples]
    errors: list[str | None] = [None for _ in samples]
    n_samples = len(samples)

    ticks = stage1_schedule()
    _assert_reaches_next_open(ticks)

    def _once() -> Any:
        return simulate_requested_fractions(
            scenario,
            ticks,
            samples,
            seeds,
            softness=softness,
            clamp_inputs=False,
        )

    try:
        fractions: Any = _once()
    except Exception as first_exc:
        logger.exception("gpu-sim batch failed; retrying once")
        try:
            fractions = _once()
        except Exception:
            rows: list[Any] = []
            for seed in seeds:
                seeded = scenario.model_copy(update={"seed": int(seed)})
                for index, sample in enumerate(samples):
                    try:
                        rows.extend(
                            simulate_requested_fractions(
                                seeded,
                                ticks,
                                [sample],
                                (int(seed),),
                                softness=softness,
                                clamp_inputs=False,
                            )
                        )
                    except Exception as exc:
                        logger.exception("gpu-sim candidate %s seed %s failed", index, seed)
                        rows.append(exc)
            fractions = rows if rows else [first_exc for _ in range(n_samples * len(seeds))]

    for lane, item in enumerate(fractions):
        index = lane % n_samples if n_samples else 0
        if isinstance(item, BaseException):
            errors[index] = str(item)
            totals[index] = _CANDIDATE_FAILURE_PENALTY
            continue
        if totals[index] >= _CANDIDATE_FAILURE_PENALTY:
            continue
        day1, two_day, stock = (float(value) for value in item)
        totals[index] += _objective_from_fractions(
            day1,
            two_day,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
            stock_ratio=stock,
            path=path,
        )
    elapsed = time.perf_counter() - started
    share = elapsed / max(len(samples), 1)
    indices = identifiable_indices(path)
    objectives: list[float] = []
    for index, total in enumerate(totals):
        if errors[index] is not None and on_candidate_error is not None:
            on_candidate_error(index, errors[index] or "")
        if total >= _CANDIDATE_FAILURE_PENALTY:
            objective = _CANDIDATE_FAILURE_PENALTY
        else:
            theta = log_params_from_vector(samples[index], indices)
            objective = total / len(seeds) + _prior_penalty(theta, path)
        objectives.append(objective)
        if on_candidate is not None:
            on_candidate(index, objective, share)
    return objectives


def evaluate_population_objectives(
    samples: list[NDArray[np.float64]],
    *,
    fit_seed: int,
    generation: int,
    workers: int,
    scenario: Scenario,
    steps: int,
    targets: CalibrationTargets,
    w_day1: float,
    w_day2: float,
    use_fake_objective: bool = False,
    on_candidate: Callable[[int, float, float], None] | None = None,
    on_candidate_error: Callable[[int, str], None] | None = None,
    softness: float = 0.15,
    path: Path | None = None,
) -> list[float]:
    """Evaluate each population member; collect objectives in candidate index order."""
    if not samples:
        return []
    if _gpu_sim_requested() and not use_fake_objective:
        return _evaluate_population_gpu_sim(
            samples,
            fit_seed=fit_seed,
            scenario=scenario,
            steps=steps,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
            softness=softness,
            on_candidate=on_candidate,
            on_candidate_error=on_candidate_error,
            path=path,
        )
    worker_count = max(1, workers)

    def run_index(
        index: int,
        sample: NDArray[np.float64],
        retry: bool,
    ) -> tuple[int, float, float, str | None]:
        objective, elapsed, error = _evaluate_one_candidate(
            index=index,
            vector=sample,
            fit_seed=fit_seed,
            generation=generation,
            scenario=scenario,
            steps=steps,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
            use_fake_objective=use_fake_objective,
            softness=softness,
        )
        if error is not None and not retry:
            objective, elapsed, error = _evaluate_one_candidate(
                index=index,
                vector=sample,
                fit_seed=fit_seed,
                generation=generation,
                scenario=scenario,
                steps=steps,
                targets=targets,
                w_day1=w_day1,
                w_day2=w_day2,
                use_fake_objective=use_fake_objective,
                softness=softness,
            )
        return index, objective, elapsed, error

    objectives: list[float | None] = [None] * len(samples)

    if worker_count == 1:
        for index, sample in enumerate(samples):
            idx, objective, elapsed, error = run_index(index, sample, retry=False)
            if error is not None and on_candidate_error is not None:
                on_candidate_error(idx, error)
            objectives[idx] = objective
            if on_candidate is not None:
                on_candidate(idx, objective, elapsed)
        return [float(x) for x in objectives]

    ctx = get_context("spawn")
    tasks = [
        _CandidateEvalTask(
            index=index,
            vector=tuple(float(x) for x in sample),
            fit_seed=fit_seed,
            generation=generation,
            scenario=scenario,
            steps=steps,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
            use_fake_objective=use_fake_objective,
            softness=softness,
        )
        for index, sample in enumerate(samples)
    ]
    task_by_index = {task.index: task for task in tasks}
    retried: set[int] = set()
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
        futures = {pool.submit(_evaluate_candidate_task, task): task.index for task in tasks}
        while futures:
            for future in as_completed(list(futures)):
                index = futures.pop(future)
                idx, objective, elapsed, error = future.result()
                if error is not None:
                    if on_candidate_error is not None:
                        on_candidate_error(idx, error)
                    if idx not in retried:
                        retried.add(idx)
                        retry_future = pool.submit(_evaluate_candidate_task, task_by_index[idx])
                        futures[retry_future] = idx
                        continue
                    objective = _CANDIDATE_FAILURE_PENALTY
                objectives[idx] = objective
                if on_candidate is not None:
                    on_candidate(idx, objective, elapsed)
    return [float(x if x is not None else _CANDIDATE_FAILURE_PENALTY) for x in objectives]


def _default_svb_scenario(seed: int) -> Scenario:
    return Scenario(
        preset="tech-concentrated",
        seed=seed,
        rumors=[
            Rumor(
                id="fit-rumor",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                credibility=0.8,
                reach=5000,
            )
        ],
    )


def _signature_scenario(seed: int) -> Scenario:
    return Scenario(
        preset="crypto-heavy",
        seed=seed,
        rumors=[
            Rumor(
                id="sig-rumor",
                claim="Depositors question Fictional Harbor Bank reserves",
                credibility=0.7,
                reach=4000,
            )
        ],
    )


def _community_scenario(seed: int) -> Scenario:
    return Scenario(
        preset="insured-community",
        seed=seed,
        rumors=[
            Rumor(
                id="placebo",
                claim="Fictional Community Bank is well capitalized",
                credibility=0.2,
                reach=100,
            )
        ],
    )


@dataclass(frozen=True, slots=True)
class GenerationCheckpoint:
    """Serializable optimizer state for resume after crash."""

    run_id: str
    seed: int
    generation: int
    mean: tuple[float, ...]
    sigma: float
    best_vector: tuple[float, ...]
    best_objective: float
    steps: int
    population: int
    real_engine: bool


def fitted_m06_values(vector: NDArray[np.float64]) -> dict[str, str]:
    """Map a behavioral vector to registry cell strings for M06 FITTED rows."""
    from engine.agents.decide import (
        calibration_amount_scale,
        calibration_awareness_media_rate,
        calibration_decision_to_request_rate,
        calibration_mobile_share,
        decision_params_from_vector,
        load_decision_params,
    )

    base = load_decision_params()
    params = decision_params_from_vector(base, vector, clamp_inputs=False)
    amount_scale = calibration_amount_scale(vector, clamp_inputs=False)
    mobile_share = calibration_mobile_share(vector, clamp_inputs=False)
    media_rate = calibration_awareness_media_rate(vector, clamp_inputs=False)
    request_rate = calibration_decision_to_request_rate(vector, clamp_inputs=False)
    from engine.gpu_full_io import feedback_params_from_vector

    gain, dispersion, impact = feedback_params_from_vector(vector)
    return {
        "M06 worry weights (ASSUMED)": ",".join(f"{weight:.6g}" for weight in params.weights),
        "M06 worry bias (ASSUMED)": f"{params.bias:.6g}",
        "M06 complex contagion k (ASSUMED)": f"{params.k:.6g}",
        "M06 belief update strength (ASSUMED)": f"{params.belief_strength:.6g}",
        "M06 public signal attention weight (ASSUMED)": f"{params.signal_attention:.6g}",
        "Calibration M06 amount scale (ASSUMED)": f"{amount_scale:.6g}",
        "M03 feedback gain (ASSUMED)": f"{gain:.6g}",
        "M06 breaking-point dispersion (ASSUMED)": f"{dispersion:.6g}",
        "M14 price impact (ASSUMED)": f"{impact:.6g}",
        "Depositor mobile share of uninsured deposits (FIT)": f"{mobile_share:.6g}",
        "M06 awareness media rate (FIT)": f"{media_rate:.6g}",
        "M07 decision-to-request rate (FIT)": f"{request_rate:.6g}",
    }


def _registry_path(path: Path | None) -> Path:
    return path or Path(__file__).resolve().parents[1] / "params" / "registry.csv"


def fitted_registry_updates(vector: tuple[float, ...], *, path: Path | None = None) -> dict[str, str]:
    """The exact cell strings ``write_fitted_registry`` would write for ``vector``, without writing.

    The normalized vector is relative to the registry at fit time, so compute this *before*
    writing: afterwards the fitted values are the new centers and the same vector maps elsewhere.
    """
    all_values = fitted_m06_values(np.asarray(vector, dtype=np.float64))
    try:
        fitted_names = identifiable_names(path)
    except RegistryError:
        # A trimmed registry without the identifiable-index row: use the repo default set.
        fitted_names = identifiable_names(None)
    keep = {
        _FITTED_ROW_FOR_COORD[name]
        for name in fitted_names
        if name in _FITTED_ROW_FOR_COORD
    }
    return {key: value for key, value in all_values.items() if key in keep}


def write_fitted_registry(
    vector: tuple[float, ...],
    *,
    run_id: str,
    seed: int,
    path: Path | None = None,
    values: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write FITTED rows for the identifiable coordinates only; every other byte stays.

    Uses :func:`calibration.registry_patch.patch_registry_rows`, so ``git diff`` shows
    only the fitted rows (the old DictWriter rewrite changed line endings and quoting
    across the whole file). Returns the per-row status from the patcher. Pass ``values``
    (from :func:`fitted_registry_updates`) to write exactly what was computed earlier.
    """
    from calibration.registry_patch import patch_registry_rows

    registry_path = _registry_path(path)
    if values is None:
        values = fitted_registry_updates(vector, path=path)
    source = f"{run_id} seed={seed}"
    updates = {
        name: {"Value": value, "Status": "FITTED", "Source": source}
        for name, value in values.items()
    }
    amount_row = "Calibration M06 amount scale (ASSUMED)"
    append = (
        {amount_row: {"Module": "M27", "Unit": "multiplier"}} if amount_row in updates else None
    )
    # Rows a trimmed registry doesn't have are skipped, as before; only the amount row is appended.
    return patch_registry_rows(registry_path, updates, append_missing=append, skip_missing=True)


def write_payment_capacity(
    capacity: float,
    *,
    run_id: str,
    seed: int,
    path: Path | None = None,
    note: str | None = None,
) -> dict[str, str]:
    """Write the stage-2 capacity multiplier into its row; every other byte stays.

    ``note`` is appended to the Source text (SVB-40: an unreachable target is recorded there).
    """
    from calibration.registry_patch import patch_registry_rows

    source = f"{run_id} seed={seed}" + (f"; {note}" if note else "")
    value = f"{float(capacity):.6g}"
    return patch_registry_rows(
        _registry_path(path),
        {_PARAM_PAYMENT_CAPACITY: {"Value": value, "Status": "FITTED", "Source": source, "Unit": "multiplier"}},
        append_missing={_PARAM_PAYMENT_CAPACITY: {"Module": "M07"}},
    )


def _brentq(
    func: Callable[[float], float],
    low: float,
    high: float,
    *,
    xtol: float = 2e-12,
    max_iter: int = 200,
) -> float:
    """1-D Brent-style root finder (bisection + secant; scipy is not a dependency)."""
    a = float(low)
    b = float(high)
    fa = float(func(a))
    fb = float(func(b))
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("root is not bracketed")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    for _ in range(max_iter):
        if fb == 0.0 or abs(b - a) <= xtol * max(1.0, abs(b)):
            return b
        # Inverse-quadratic / secant step when it stays inside the bracket.
        if fa != fc and fb != fc:
            d = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            d = b - fb * (b - a) / (fb - fa)
        bisect = 0.5 * (a + b)
        if not (min(a, b) < d < max(a, b)) or abs(d - b) >= 0.5 * abs(b - c):
            d = bisect
        c, fc = b, fb
        fd = float(func(d))
        if fa * fd < 0.0:
            b, fb = d, fd
        else:
            a, fa = d, fd
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    return b


def _log_spaced_multipliers(low: float, high: float, n: int) -> list[float]:
    return [float(x) for x in np.logspace(math.log10(low), math.log10(high), int(n))]


def _monotone_log_interpolate(
    multipliers: Sequence[float],
    settled: Sequence[float],
    target: float,
) -> tuple[float, float, float]:
    """Interpolate monotone settled vs log-multiplier. Returns (m_star, m_lo, m_hi)."""
    order = np.argsort(np.asarray(multipliers, dtype=np.float64))
    xs = np.asarray(multipliers, dtype=np.float64)[order]
    ys = np.maximum.accumulate(np.asarray(settled, dtype=np.float64)[order])
    if target <= float(ys[0]):
        hi = float(xs[min(1, len(xs) - 1)])
        return float(xs[0]), float(xs[0]), hi
    if target >= float(ys[-1]):
        lo = float(xs[max(0, len(xs) - 2)])
        return float(xs[-1]), lo, float(xs[-1])
    hi_idx = int(np.searchsorted(ys, target, side="left"))
    lo_idx = max(0, hi_idx - 1)
    hi_idx = min(hi_idx, len(xs) - 1)
    x0, x1 = float(xs[lo_idx]), float(xs[hi_idx])
    y0, y1 = float(ys[lo_idx]), float(ys[hi_idx])
    if y1 <= y0 or x1 <= x0:
        return x0, x0, x1
    t = (target - y0) / (y1 - y0)
    log_m = math.log(x0) + t * (math.log(x1) - math.log(x0))
    return math.exp(log_m), x0, x1


def _as_batch_measure(
    measure: Callable[..., Any],
) -> Callable[[Sequence[float]], list[float]]:
    def batched(xs: Sequence[float]) -> list[float]:
        try:
            out = measure(xs)
        except TypeError:
            return [float(measure(float(x))) for x in xs]
        if np.isscalar(out):
            return [float(measure(float(x))) for x in xs]
        return [float(v) for v in out]

    return batched


def _settled_day1_for_multipliers(
    multipliers: Sequence[float],
    *,
    vector: Sequence[float],
    seed: int,
    path: Path | None = None,
) -> list[float]:
    """One caps-on ``simulate_logs`` call: multipliers × shared seeds at ``vector``."""
    from engine.gpu_sim import simulate_logs

    arr = np.asarray(vector, dtype=np.float64)
    scenario = _default_svb_scenario(int(seed))
    seeds = shared_eval_seeds(int(seed), path)
    rows = [arr] * len(multipliers)
    logs = simulate_logs(
        scenario,
        int(_DAY1_CLOSE),
        rows,
        disable_caps=False,
        softness=fixed_softness(path),
        clamp_inputs=False,
        seeds=seeds,
        capacity_multipliers=[float(m) for m in multipliers],
        schedule=day1_tick_schedule(),
    )
    n_m = len(multipliers)
    means: list[float] = []
    for index in range(n_m):
        day1s = [
            _score_outflow_log(logs[seed_i * n_m + index], requested=False)[0]
            for seed_i in range(len(seeds))
        ]
        means.append(float(np.mean(day1s)))
    return means


def _solve_capacity_grid(
    measure: Callable[[Sequence[float]], Sequence[float]],
    target: float,
    *,
    low: float = _STAGE2_LO,
    high: float = _STAGE2_HI,
    report: dict[str, Any] | None = None,
) -> float:
    """16-point log grid, monotone log interpolation, one refine if bracket > 1.2×.

    SVB-40: when the whole grid stays below ``target - _STAGE2_TOL`` the target is unreachable.
    The solver then returns the smallest grid multiplier already on the plateau (settled within
    ``_STAGE2_TOL`` of the grid maximum) and marks the result unreachable in ``report`` instead
    of extrapolating.
    """
    grid = _log_spaced_multipliers(low, high, _STAGE2_GRID)
    settled = [float(v) for v in measure(grid)]
    plateau = max(settled)
    info: dict[str, Any] = {"target": float(target), "plateau": float(plateau)}
    if plateau < target - _STAGE2_TOL:
        start = next(
            float(mult)
            for mult, value in zip(grid, settled, strict=True)
            if value >= plateau - _STAGE2_TOL
        )
        info.update(reachable=False, multiplier=start)
        if report is not None:
            report.update(info)
        return start
    m_star = _interpolate_capacity(measure, grid, settled, target)
    info.update(reachable=True, multiplier=float(m_star))
    if report is not None:
        report.update(info)
    return float(m_star)


def _interpolate_capacity(
    measure: Callable[[Sequence[float]], Sequence[float]],
    grid: Sequence[float],
    settled: Sequence[float],
    target: float,
) -> float:
    for mult, value in zip(grid, settled, strict=True):
        if abs(value - target) <= _STAGE2_TOL:
            return float(mult)
    m_star, lo, hi = _monotone_log_interpolate(grid, settled, target)
    if lo > 0.0 and hi / lo > _STAGE2_REFINE_RATIO:
        grid2 = _log_spaced_multipliers(lo, hi, _STAGE2_GRID)
        settled2 = [float(v) for v in measure(grid2)]
        for mult, value in zip(grid2, settled2, strict=True):
            if abs(value - target) <= _STAGE2_TOL:
                return float(mult)
        m_star, _lo, _hi = _monotone_log_interpolate(grid2, settled2, target)
    return float(m_star)


def _assert_vector_not_on_its_own_fit(
    vector: Sequence[float],
    *,
    run_id: str,
    seed: int,
    path: Path | None = None,
) -> None:
    """Refuse a stage-2 solve that applies a fit vector on top of its own saved rows (SVB-40).

    The normalized vector is relative to the registry at fit time. Once the run's identifiable
    rows are saved, those values are the new centers, so the matching vector is 0.5 at those
    coordinates. Passing the fit vector again moves them a second time and stage 2 then solves
    against a different model than the board measures.
    """
    try:
        indices = identifiable_indices(path)
    except RegistryError:
        indices = identifiable_indices(None)
    values = [float(x) for x in vector]
    moved = [i for i in indices if 0 <= i < len(values) and abs(values[i] - 0.5) > 1e-12]
    if not moved:
        return
    source = f"{run_id} seed={seed}"
    saved: list[str] = []
    for index in moved:
        row_name = _FITTED_ROW_FOR_COORD.get(_VECTOR_PARAM_NAMES[index])
        if row_name is None:
            continue
        try:
            row = get_parameter(row_name, path)
        except RegistryError:
            continue
        is_this_run = row.status.strip().upper() == "FITTED" and row.source.strip() == source
        if is_this_run and row_name not in saved:
            saved.append(row_name)
    if saved:
        raise RuntimeError(
            "stage 2 would apply the fit vector on top of its own saved rows "
            f"({', '.join(saved)}; source {source!r}). Solve stage 2 before saving, "
            "or pass the all-0.5 vector against the saved registry."
        )


def fit_payment_capacity(
    *,
    measure: Callable[..., Any] | None = None,
    vector: Sequence[float] | None = None,
    target: float | None = None,
    bracket: tuple[float, float] | None = None,
    path: Path | None = None,
    write: bool = False,
    run_id: str = "svb-05",
    seed: int = 0,
    report: dict[str, Any] | None = None,
) -> float:
    """Stage 2: batched log-grid on the FIT multiplier so settled day-one equals 0.24.

    ``report`` (optional) is filled with ``target``, ``reachable``, ``plateau`` and
    ``multiplier``. An unreachable target returns the start of the plateau, and a written row
    says so in its Source text (SVB-40).
    """
    goal = load_calibration_targets(path).svb_day1_fraction if target is None else float(target)
    if measure is None:
        if vector is None:
            raise ValueError("fit_payment_capacity needs measure or vector")
        _assert_vector_not_on_its_own_fit(vector, run_id=run_id, seed=seed, path=path)

        def measure_batch(xs: Sequence[float]) -> list[float]:
            return _settled_day1_for_multipliers(
                xs, vector=vector, seed=seed, path=path
            )

        batch = measure_batch
    else:
        batch = _as_batch_measure(measure)
    if bracket is None:
        low, high = _STAGE2_LO, _STAGE2_HI
    else:
        low, high = float(bracket[0]), float(bracket[1])
    info: dict[str, Any] = {}
    capacity = _solve_capacity_grid(batch, goal, low=low, high=high, report=info)
    if report is not None:
        report.update(info)
    if write:
        note = None
        if not info.get("reachable", True):
            note = (
                f"target {goal:.4g} unreachable; plateau {float(info['plateau']):.4g}; "
                "capacity non-binding at or above this multiplier"
            )
        write_payment_capacity(capacity, run_id=run_id, seed=seed, path=path, note=note)
    return float(capacity)


def write_best_vector_json(
    directory: Path,
    *,
    run_id: str,
    seed: int,
    vector: Sequence[float],
    objective: float,
    force: bool = False,
) -> Path:
    """Write ``best_vector.json`` from the reported global best.

    A worse objective does not overwrite a better file unless ``force``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "best_vector.json"
    if path.is_file() and not force:
        prior = json.loads(path.read_text(encoding="utf-8"))
        if float(prior.get("objective", math.inf)) <= float(objective):
            return path
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": int(seed),
                "vector": [float(x) for x in vector],
                "objective": float(objective),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def save_generation_checkpoint(
    checkpoint: GenerationCheckpoint,
    directory: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint.json"
    path.write_text(
        json.dumps(asdict(checkpoint), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_best_vector_json(
        directory,
        run_id=checkpoint.run_id,
        seed=checkpoint.seed,
        vector=checkpoint.best_vector,
        objective=checkpoint.best_objective,
    )
    return path


def load_generation_checkpoint(directory: Path) -> GenerationCheckpoint | None:
    path = directory / "checkpoint.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GenerationCheckpoint(**payload)


def save_cma_es(es: Any, directory: Path) -> Path:
    """Pickle the CMA-ES object so resume restores RNG and covariance."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "es.pkl"
    path.write_bytes(pickle.dumps(es))
    return path


def load_cma_es(directory: Path) -> Any | None:
    path = directory / "es.pkl"
    if not path.is_file():
        return None
    return pickle.loads(path.read_bytes())


def _cma_bound_lists(
    indices: tuple[int, ...],
    path: Path | None = None,
) -> tuple[list[float], list[float]]:
    lows: list[float] = []
    highs: list[float] = []
    for index in indices:
        lo, hi = _theta_bounds(index, path)
        lows.append(float(lo))
        highs.append(float(hi))
    return lows, highs


def _cma_stagnation(stop: dict[str, Any]) -> bool:
    if not stop:
        return False
    ignore = {"maxiter", "maxfevals", "callback", "ftarget", "manual"}
    return bool(set(stop) - ignore)


def _make_cma_es(
    x0: NDArray[np.float64],
    sigma0: float,
    *,
    pop_size: int,
    seed: int,
    max_generations: int,
    lows: list[float],
    highs: list[float],
) -> Any:
    import cma

    return cma.CMAEvolutionStrategy(
        np.asarray(x0, dtype=np.float64),
        float(sigma0),
        {
            "bounds": [lows, highs],
            "popsize": int(pop_size),
            "seed": int(seed),
            "verbose": -9,
            "tolfun": 1e-4,
            "maxiter": int(max_generations),
        },
    )


def _prepare_resumed_es(es: Any, max_generations: int) -> Any:
    es.opts.set({"maxiter": int(max_generations)})
    if es.stop(check=False):
        es.stop().clear()
    return es


def run_identifiability_sweep(
    *,
    seed: int = 0,
    path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Perturb every vector slot; report the ones that move requested outflow most.

    Selects as many slots as the registry's identifiable set (SVB-26: two). The
    sweep's metric is day-one requested outflow on a light book with no market, so
    it can't see price impact; the registry's fit set is chosen on the stage-1
    targets (next-open requests and the stock ratio), so the two needn't agree.
    """
    softness = _registry_float(_PARAM_SOFT_INITIAL, path)
    base = np.full(int(_registry_float(_PARAM_BEHAVIORAL_DIM, path)), 0.5, dtype=np.float64)
    base_day1, _ = _light_requested_fractions(base, seed=seed, softness=softness)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(_VECTOR_PARAM_NAMES):
        high = base.copy()
        low = base.copy()
        high[index] = 0.85
        low[index] = 0.15
        day1_high, _ = _light_requested_fractions(high, seed=seed, softness=softness)
        day1_low, _ = _light_requested_fractions(low, seed=seed, softness=softness)
        rows.append(
            {
                "index": index,
                "name": name,
                "day1_low": day1_low,
                "day1_high": day1_high,
                "delta_day1": day1_high - day1_low,
                "abs_delta": abs(day1_high - day1_low),
            }
        )
    ranked = sorted(rows, key=lambda row: row["abs_delta"], reverse=True)
    selected = [int(row["index"]) for row in ranked[: len(identifiable_indices(path))]]
    payload = {
        "seed": seed,
        "softness": softness,
        "baseline_day1": base_day1,
        "params": rows,
        "selected": selected,
        "selected_names": [_VECTOR_PARAM_NAMES[index] for index in selected],
        "reason": (
            "largest |Δ day-1 requested outflow| on a one-at-a-time sweep of the "
            f"{len(_VECTOR_PARAM_NAMES)}-vector with payment caps and closure off; "
            "count matches the registry's identifiable set"
        ),
    }
    if write:
        directory = diagnostics_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "identifiability-sweep.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# M27 identifiability sweep",
            "",
            f"Seed `{seed}`, softness `{softness}`.",
            "",
            "Selected: " + ", ".join(payload["selected_names"]) + ".",
            "",
            "| Index | Name | Day-1 low | Day-1 high | |Δ| |",
            "| :---- | :---- | :---- | :---- | :---- |",
        ]
        for row in ranked:
            lines.append(
                f"| {row['index']} | {row['name']} | {row['day1_low']:.6g} | "
                f"{row['day1_high']:.6g} | {row['abs_delta']:.6g} |"
            )
        (directory / "identifiability-sweep.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    return payload


def sweep_identifiable_objectives(
    *,
    seed: int,
    path: Path | None = None,
    grid: int | None = None,
    softness: float | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    """Sweep each identifiable θ; return (theta, day1, objective) curves."""
    targets = load_calibration_targets(path)
    indices = identifiable_indices(path)
    steps = eval_simulation_minutes(path)
    w_day1 = _registry_float(_PARAM_E1_DAY1_WEIGHT, path)
    w_day2 = _registry_float(_PARAM_E1_DAY2_WEIGHT, path)
    temp = _registry_float(_PARAM_SOFT_INITIAL, path) if softness is None else float(softness)
    count = grid or int(_registry_float(_PARAM_BRACKET_GRID, path))
    scenario = _default_svb_scenario(seed)
    curves: dict[str, list[tuple[float, float, float]]] = {}
    for index in indices:
        lo, hi = _theta_bounds(index, path)
        thetas = np.linspace(lo, hi, count)
        points: list[tuple[float, float, float]] = []
        for value in thetas:
            theta = np.zeros(len(indices), dtype=np.float64)
            theta[indices.index(index)] = float(value)
            vector = vector_from_log_params(theta, indices, path=path)
            day1, two_day, _ = measure_outflow(
                scenario,
                steps=steps,
                param_vector=vector,
                requested=True,
                disable_caps=True,
                softness=temp,
                clamp_inputs=False,
            )
            objective = _objective_from_fractions(
                day1,
                two_day,
                targets=targets,
                w_day1=w_day1,
                w_day2=w_day2,
            ) + _prior_penalty(theta, path)
            points.append((float(value), float(day1), float(objective)))
        curves[_VECTOR_PARAM_NAMES[index]] = points
    return curves


def bracket_tipping_point(
    *,
    seed: int,
    path: Path | None = None,
    scenario: Scenario | None = None,
    steps: int | None = None,
    softness: float | None = None,
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """Joint log-space sweep that brackets the E1 day-1 crossing."""
    targets = load_calibration_targets(path)
    indices = identifiable_indices(path)
    eval_steps = steps or eval_simulation_minutes(path)
    temp = fixed_softness(path) if softness is None else float(softness)
    grid_n = int(_registry_float(_PARAM_BRACKET_GRID, path))
    fit_scenario = scenario or _default_svb_scenario(seed)
    lo = min(_theta_bounds(index, path)[0] for index in indices)
    hi = max(_theta_bounds(index, path)[1] for index in indices)
    grid = np.linspace(lo, hi, grid_n)
    signs = np.asarray(
        [-1.0 if _VECTOR_PARAM_NAMES[index] == "contagion_k" else 1.0 for index in indices],
        dtype=np.float64,
    )
    vectors = [
        vector_from_log_params(
            _project_theta(signs * float(value), indices, path=path),
            indices,
            path=path,
        )
        for value in grid
    ]
    ticks = stage1_schedule()
    _assert_reaches_next_open(ticks)
    if _gpu_sim_requested():
        from engine.gpu_sim import simulate_requested_fractions

        fractions = simulate_requested_fractions(
            fit_scenario,
            ticks,
            vectors,
            (int(fit_scenario.seed),),
            softness=temp,
            clamp_inputs=False,
        )
        day1s = [float(row[0]) for row in fractions]
    else:
        day1s = []
        for vector in vectors:
            day1, _two, _ = measure_outflow(
                fit_scenario,
                steps=eval_steps,
                param_vector=vector,
                requested=True,
                disable_caps=True,
                softness=temp,
                clamp_inputs=False,
            )
            day1s.append(day1)
    target = targets.svb_day1_fraction
    box = (float(lo), float(hi))
    for i in range(len(grid) - 1):
        if (day1s[i] - target) * (day1s[i + 1] - target) <= 0.0:
            left = max(0, i - 1)
            right = min(len(grid) - 1, i + 2)
            box = (float(grid[left]), float(grid[right]))
            break
    else:
        if all(value < target for value in day1s):
            box = (float(grid[-2]), float(grid[-1])) if grid_n > 1 else (float(lo), float(hi))
        elif all(value > target for value in day1s):
            box = (float(grid[0]), float(grid[1])) if grid_n > 1 else (float(lo), float(hi))
    return box, list(zip((float(x) for x in grid), day1s, strict=True))


def fit_behavioral_parameters(
    seed: int,
    *,
    path: Path | None = None,
    scenario: Scenario | None = None,
    generations: int | None = None,
    population: int | None = None,
    steps: int | None = None,
    run_id: str | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 1,
    on_generation: Callable[[int, float, tuple[float, ...]], None] | None = None,
    start_generation: int = 0,
    initial_mean: NDArray[np.float64] | None = None,
    initial_sigma: float | None = None,
    initial_best_vector: NDArray[np.float64] | None = None,
    initial_best_objective: float | None = None,
    workers: int = 1,
    on_candidate: Callable[[int, int, float, float], None] | None = None,
    on_candidate_error: Callable[[int, str], None] | None = None,
    restarts: int | None = None,
    use_fake_objective: bool = False,
) -> FitResult:
    """Two-stage E1 fit: bracket the tipping point, then real CMA-ES in log space."""
    targets = load_calibration_targets(path)
    dim = int(_registry_float(_PARAM_BEHAVIORAL_DIM, path))
    indices = identifiable_indices(path)
    names = identifiable_names(path)
    max_generations = generations or int(_registry_float(_PARAM_CMA_GENERATIONS, path))
    pop_size = max(4, population or int(_registry_float(_PARAM_CMA_POPULATION, path)))
    eval_steps = steps or eval_simulation_minutes(path)
    w_day1 = _registry_float(_PARAM_E1_DAY1_WEIGHT, path)
    w_day2 = _registry_float(_PARAM_E1_DAY2_WEIGHT, path)
    if restarts is None:
        restart_budget = max(0, int(_registry_float(_PARAM_RESTARTS, path)))
    else:
        restart_budget = max(0, int(restarts))
    fit_scenario = scenario or _default_svb_scenario(seed)
    resolved_run_id = run_id or f"m27-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    lows, highs = _cma_bound_lists(indices, path)
    softness = fixed_softness(path)

    resuming = start_generation > 0 or initial_mean is not None
    es: Any = None
    if resuming and checkpoint_dir is not None:
        es = load_cma_es(checkpoint_dir)
        if es is not None:
            es = _prepare_resumed_es(es, max_generations)
        saved = load_generation_checkpoint(checkpoint_dir)
        if saved is not None:
            if initial_best_vector is None:
                initial_best_vector = np.asarray(saved.best_vector, dtype=np.float64)
            if initial_best_objective is None:
                initial_best_objective = float(saved.best_objective)

    box: tuple[float, float] | None = None
    if es is None:
        box, _curve = bracket_tipping_point(
            seed=seed,
            path=path,
            scenario=fit_scenario,
            steps=eval_steps,
            softness=softness,
        )
        if initial_mean is not None:
            resume_mean = np.asarray(initial_mean, dtype=np.float64)
            if resume_mean.shape[0] == dim:
                resume_mean = log_params_from_vector(resume_mean, indices)
            x0 = _project_theta(resume_mean, indices, path=path)
        else:
            center = 0.5 * (box[0] + box[1])
            start = np.asarray(
                [
                    (-center if _VECTOR_PARAM_NAMES[index] == "contagion_k" else center)
                    for index in indices
                ],
                dtype=np.float64,
            )
            x0 = _project_theta(start, indices, path=path)
        if initial_sigma is not None:
            sigma0 = float(initial_sigma)
        else:
            sigma0 = max(0.05, 0.25 * (box[1] - box[0]))
        es = _make_cma_es(
            x0,
            sigma0,
            pop_size=pop_size,
            seed=seed,
            max_generations=max_generations,
            lows=lows,
            highs=highs,
        )
        start_x0 = np.asarray(x0, dtype=np.float64).copy()
        start_sigma0 = float(sigma0)
    else:
        start_x0 = np.asarray(es.mean, dtype=np.float64).copy()
        start_sigma0 = float(es.sigma)

    if initial_best_vector is not None:
        best_vector = np.asarray(initial_best_vector, dtype=np.float64).copy()
        if best_vector.shape[0] != dim:
            best_vector = vector_from_log_params(best_vector, indices, dim=dim, path=path)
    else:
        best_vector = vector_from_log_params(
            np.asarray(es.mean, dtype=np.float64),
            indices,
            dim=dim,
            path=path,
        )
    best_theta = log_params_from_vector(best_vector, indices)
    if initial_best_objective is not None:
        best_obj = float(initial_best_objective)
    elif use_fake_objective:
        best_obj = float(np.dot(best_theta, best_theta))
    else:
        best_obj = objective_e1(
            best_vector,
            scenario=fit_scenario,
            steps=eval_steps,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
            softness=softness,
            path=path,
        )

    def _save(generation: int) -> None:
        if checkpoint_dir is None or generation % max(1, checkpoint_every) != 0:
            return
        save_generation_checkpoint(
            GenerationCheckpoint(
                run_id=resolved_run_id,
                seed=seed,
                generation=generation,
                mean=tuple(float(x) for x in np.asarray(es.mean, dtype=np.float64)),
                sigma=float(es.sigma),
                best_vector=tuple(float(x) for x in best_vector),
                best_objective=float(best_obj),
                steps=eval_steps,
                population=int(es.popsize),
                real_engine=use_real_calibration_engine(),
            ),
            checkpoint_dir,
        )
        save_cma_es(es, checkpoint_dir)

    if on_generation is not None and start_generation == 0:
        on_generation(0, best_obj, tuple(float(x) for x in best_vector))
    _save(max(start_generation, 0))

    def _score_vectors(
        generation: int,
        vectors: list[NDArray[np.float64]],
        softness_value: float,
    ) -> list[float]:
        def _on_candidate(index: int, objective: float, elapsed: float) -> None:
            if on_candidate is not None:
                on_candidate(generation, index, objective, elapsed)

        return evaluate_population_objectives(
            vectors,
            fit_seed=seed,
            generation=generation,
            workers=workers,
            scenario=fit_scenario,
            steps=eval_steps,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
            use_fake_objective=use_fake_objective,
            on_candidate=_on_candidate,
            on_candidate_error=on_candidate_error,
            softness=softness_value,
            path=path,
        )

    generation_cursor = start_generation
    ipop_used = 0
    while generation_cursor < max_generations:
        stop = es.stop()
        if stop:
            if _cma_stagnation(stop) and ipop_used < restart_budget:
                ipop_used += 1
                pop_size = max(4, int(pop_size) * 2)
                es = _make_cma_es(
                    start_x0,
                    start_sigma0,
                    pop_size=pop_size,
                    seed=seed + ipop_used,
                    max_generations=max_generations,
                    lows=lows,
                    highs=highs,
                )
                continue
            break
        xs = es.ask()
        thetas = [
            _project_theta(np.asarray(sample, dtype=np.float64), indices, path=path)
            for sample in xs
        ]
        vectors = [
            vector_from_log_params(theta, indices, dim=dim, path=path) for theta in thetas
        ]
        objectives = _score_vectors(generation_cursor, vectors, softness)
        es.tell(xs, objectives)
        for value, theta, vector in zip(objectives, thetas, vectors, strict=True):
            if value < best_obj:
                best_obj = float(value)
                best_theta = np.asarray(theta, dtype=np.float64).copy()
                best_vector = np.asarray(vector, dtype=np.float64).copy()
        generation_cursor += 1
        if on_generation is not None:
            on_generation(generation_cursor, best_obj, tuple(float(x) for x in best_vector))
        _save(generation_cursor)

    fit_behavioral_parameters.last_es = es  # type: ignore[attr-defined]
    if checkpoint_dir is not None:
        write_best_vector_json(
            checkpoint_dir,
            run_id=resolved_run_id,
            seed=seed,
            vector=best_vector,
            objective=best_obj,
            force=True,
        )
    return FitResult(
        seed=seed,
        vector=tuple(float(x) for x in best_vector),
        objective=float(best_obj),
        generations=generation_cursor,
        log_params=tuple(float(x) for x in best_theta),
        bound_flags=bound_flags_for(best_theta, indices, path),
        identifiable=names,
    )


def run_holdout(
    vector: tuple[float, ...],
    seed: int,
    *,
    path: Path | None = None,
    steps: int | None = None,
) -> HoldoutResult:
    """Evaluate Signature (E5) and the insured-community placebo without refitting.

    Uses the crypto-heavy follower-bank preset and the FITTED vector. The
    SVB-fitted ``M07 capacity multiplier (FIT)`` is read and held fixed
    (never re-solved on Signature).
    """
    eval_steps = steps if steps is not None else eval_simulation_minutes(path)
    arr = np.asarray(vector, dtype=np.float64)
    temp = fixed_softness(path)
    # Held fixed: stage 2 already wrote this on SVB. Re-solving here
    # would make the holdout the same settled-day-one number as E1.
    _ = _registry_float(_PARAM_PAYMENT_CAPACITY, path)
    signature = _signature_scenario(seed)
    if signature.preset != "crypto-heavy":
        raise RuntimeError("Signature holdout must use the crypto-heavy preset")
    sig_day1, _, _ = measure_outflow(
        signature,
        steps=eval_steps,
        param_vector=arr,
        softness=temp,
        clamp_inputs=False,
    )
    # Requested by day-one close, caps on: behavior, not the shared capacity ceiling.
    sig_requested, _, _ = measure_outflow(
        signature,
        steps=eval_steps,
        param_vector=arr,
        requested=True,
        disable_caps=False,
        softness=temp,
        clamp_inputs=False,
    )
    _, _, closed = measure_outflow(
        _community_scenario(seed),
        steps=eval_steps,
        param_vector=arr,
        softness=temp,
        clamp_inputs=False,
    )
    return HoldoutResult(
        signature_day1_fraction=sig_day1,
        placebo_survives=not closed,
        signature_day1_requested=sig_requested,
    )


def validate_fitted_parameters(
    vector: tuple[float, ...],
    seed: int,
    *,
    path: Path | None = None,
    steps: int | None = None,
) -> CapsOnValidation:
    """SVB replay with payment caps and closure ON (hard decisions)."""
    eval_steps = steps if steps is not None else eval_simulation_minutes(path)
    arr = np.asarray(vector, dtype=np.float64)
    scenario = _default_svb_scenario(seed)
    temp = fixed_softness(path)
    day1_settled, two_settled, closed = measure_outflow(
        scenario,
        steps=eval_steps,
        param_vector=arr,
        requested=False,
        disable_caps=False,
        softness=temp,
        clamp_inputs=False,
    )
    day1_req, two_req, _ = measure_outflow(
        scenario,
        steps=eval_steps,
        param_vector=arr,
        requested=True,
        disable_caps=False,
        softness=temp,
        clamp_inputs=False,
    )
    return CapsOnValidation(
        day1_settled=day1_settled,
        two_day_settled=two_settled,
        day1_requested=day1_req,
        two_day_requested=two_req,
        closed=closed,
    )


def measure_e1_stock_ratio(
    vector: tuple[float, ...] | NDArray[np.float64],
    seed: int,
    *,
    path: Path | None = None,
) -> float | None:
    """Stage-1 day-one last/first price (caps off). None on the fake engine."""
    if not use_real_calibration_engine():
        return None
    arr = np.asarray(vector, dtype=np.float64)
    scenario = _default_svb_scenario(seed)
    temp = fixed_softness(path)
    if _gpu_sim_requested():
        from engine.gpu_sim import simulate_requested_fractions

        fractions = simulate_requested_fractions(
            scenario,
            stage1_schedule(),
            [arr],
            (int(seed),),
            softness=temp,
            clamp_inputs=False,
        )
        if len(fractions) == 0:
            return None
        return float(fractions[0][2])
    from engine.agents.decide import DecisionModule, apply_calibration_vector
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    engine = build_engine(
        modes=real_calibration_modes(),
        default_mode=ModuleMode.FAKE,
        scenario=scenario,
    )
    module = engine["M06"]
    if isinstance(module, DecisionModule):
        apply_calibration_vector(module, arr, clamp_inputs=False)
        from engine.gpu_full_io import apply_lane_feedback

        apply_lane_feedback(module, arr)  # SVB-65: gain, dispersion, impact
        module.softness = float(temp)
    _disable_caps_and_closure(engine)
    log = engine.run(max(eval_simulation_minutes(path), int(day1_close_minute(path))))
    return _score_day1_stock_ratio(log, path=path)


def _within_band(observed: float, expected: float, band: ToleranceBand) -> bool:
    if band.relative is not None and expected > 0:
        return abs(observed - expected) <= band.relative * expected
    if band.absolute is not None:
        return abs(observed - expected) <= band.absolute
    return True


def build_validation_board(
    fit: FitResult,
    holdout: HoldoutResult,
    *,
    path: Path | None = None,
) -> list[ValidationBoardRow]:
    """Fill the §13 board from fit and holdout metrics."""
    targets = load_calibration_targets(path)
    steps = eval_simulation_minutes(path)
    vector = np.asarray(fit.vector, dtype=np.float64)
    scenario = _default_svb_scenario(fit.seed)
    temp = fixed_softness(path)
    day1_req, requests_next_open, _ = measure_outflow(
        scenario,
        steps=max(steps, int(_NEXT_OPEN)),
        param_vector=vector,
        requested=True,
        disable_caps=True,
        softness=temp,
        clamp_inputs=False,
    )
    settled_day1, _settled_next, _ = measure_outflow(
        scenario,
        steps=max(steps, int(day1_close_minute(path))),
        param_vector=vector,
        requested=False,
        disable_caps=False,
        softness=temp,
        clamp_inputs=False,
    )
    tol_next = _parse_tolerance(get_parameter(_TOL_NEXT_OPEN, path).value)
    tol_settled = _parse_tolerance(get_parameter(_TOL_SETTLED_DAY1, path).value)
    tol_e5 = _parse_tolerance(get_parameter(_TOL_E5_DAY1, path).value)
    use_requested = (
        targets.signature_day1_requested_fraction is not None
        and holdout.signature_day1_requested is not None
    )
    e5_observed = holdout.signature_day1_requested if use_requested else holdout.signature_day1_fraction
    e5_expected = (
        targets.signature_day1_requested_fraction if use_requested else targets.signature_day1_fraction
    )
    e5_detail = (
        "Signature-like holdout: requested by day-one close (caps on); settled is capacity-bound"
        if use_requested
        else "Signature-like holdout day-1 settled fraction (no requested target row)"
    )
    rows = [
        ValidationBoardRow(
            target_id="E1-requested-day1",
            label="requests by day-one close",
            observed=day1_req,
            expected=targets.svb_requested_day1_fraction,
            tolerance=tol_next,
            passed=_within_band(day1_req, targets.svb_requested_day1_fraction, tol_next),
            detail="Requested outflow by day-one close (caps off)",
        ),
        ValidationBoardRow(
            target_id="E1-next-open",
            label="requests by next open",
            observed=requests_next_open,
            expected=targets.svb_two_day_fraction,
            tolerance=tol_next,
            passed=_within_band(requests_next_open, targets.svb_two_day_fraction, tol_next),
            detail="Requested outflow by next open (caps off)",
        ),
        ValidationBoardRow(
            target_id="E1-settled-day1",
            label="settled day one",
            observed=settled_day1,
            expected=targets.svb_day1_fraction,
            tolerance=tol_settled,
            passed=_within_band(settled_day1, targets.svb_day1_fraction, tol_settled),
            detail="Settled outflow by day-one close (caps on)",
        ),
        ValidationBoardRow(
            target_id="E5-day1",
            label=_TOL_E5_DAY1,
            observed=float(e5_observed),
            expected=float(e5_expected),
            tolerance=tol_e5,
            passed=_within_band(float(e5_observed), float(e5_expected), tol_e5),
            detail=e5_detail,
        ),
        ValidationBoardRow(
            target_id="placebo",
            label=_TOL_PLACEBO,
            observed=1.0 if holdout.placebo_survives else 0.0,
            expected=1.0,
            tolerance=ToleranceBand(relative=None, absolute=0.0),
            passed=holdout.placebo_survives,
            detail="Insured community bank survives",
        ),
    ]
    return rows


def format_validation_board_markdown(
    board: list[ValidationBoardRow],
    *,
    run_id: str,
    seed: int,
    fit_objective: float,
    stock_ratio: float | None = None,
    stock_target: float | None = None,
) -> str:
    stock_obs = "n/a" if stock_ratio is None else f"{stock_ratio:.6g}"
    stock_exp = "n/a" if stock_target is None else f"{stock_target:.6g}"
    lines = [
        f"# M27 validation board (§13) — {run_id}",
        "",
        f"- Seed: `{seed}`",
        f"- E1 objective (weighted squared log-error): `{fit_objective:.6g}`",
        f"- E1 stock ratio (observed vs target): `{stock_obs}` vs `{stock_exp}`",
        "",
        "| Target | Observed | Expected | Passed | Detail |",
        "| :---- | :---- | :---- | :---- | :---- |",
    ]
    for row in board:
        lines.append(
            f"| {row.target_id} | {row.observed:.6g} | {row.expected:.6g} | "
            f"{'yes' if row.passed else 'no'} | {row.detail} |"
        )
    return "\n".join(lines) + "\n"


def run_calibration_harness(
    seed: int,
    *,
    path: Path | None = None,
) -> tuple[FitResult, HoldoutResult, list[ValidationBoardRow]]:
    """Fit on SVB, hold out Signature, and return the §13 board."""
    fit = fit_behavioral_parameters(seed, path=path)
    holdout = run_holdout(fit.vector, seed, path=path)
    board = build_validation_board(fit, holdout, path=path)
    return fit, holdout, board


class CalibrationModule:
    """M27 scheduler module: observes outflows each tick; fills the board once (§19)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, module_id: str = "M27") -> None:
        self.module_id = module_id
        self.scenario: Scenario | None = None
        self.bus: Any = None
        self.inbox: list[BaseModel] = []
        self._cursor = 0
        self.objective = 0.0
        self.placebo_survives = True
        self.outflow_fraction = 0.0
        self.target_outflow = 0.0
        self.fit: FitResult | None = None
        self.holdout: HoldoutResult | None = None
        self.validation_board: list[ValidationBoardRow] = []
        self._finalized = False

    def attach(self, bus: Any, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        return self.on_tick(t, new_events)

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.scenario is None:
            return []
        self._refresh_observed_metrics()
        return []

    def populate_validation_board(self) -> list[ValidationBoardRow]:
        """Run SVB fit, Signature holdout, and return the §13 board."""
        if self.scenario is None:
            return []
        if self._finalized:
            return self.validation_board
        self._finalize()
        return self.validation_board

    def _refresh_observed_metrics(self) -> None:
        if self.scenario is None:
            return
        deposits = _parse_money(get_parameter(_PARAM_DEPOSITS).value)
        target = _parse_fraction_cell(get_parameter(_PARAM_SIGNATURE_DAY1).value)
        paid = 0.0
        if self.bus is not None:
            for event in self.bus.log:
                if isinstance(event, PaymentSettled):
                    paid += event.amount
        fraction = paid / deposits if deposits else 0.0
        self.outflow_fraction = fraction
        self.target_outflow = target
        if fraction > 0 and target > 0:
            self.objective = _log_sq_error(fraction, target)
        else:
            self.objective = (fraction - target) ** 2
        preset = self.scenario.preset.lower()
        self.placebo_survives = "community" in preset or fraction < target

    def _finalize(self) -> None:
        self._finalized = True
        seed = self.scenario.seed if self.scenario is not None else 0
        fit, holdout, board = run_calibration_harness(seed)
        self.fit = fit
        self.holdout = holdout
        self.validation_board = board
        self.objective = fit.objective
        self.placebo_survives = holdout.placebo_survives
