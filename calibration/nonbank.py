"""Non-bank price-path calibration (M27, §8.4).

The SVB fit in ``calibration/fit.py`` is unchanged. This harness builds its own
scenarios, reads ``PriceTick`` paths, and does not write FITTED rows. With the
real engine and ``--gpu-sim``, equity candidates and the crypto and rates
holdouts share the batched depositor kernel in ``engine/gpu_sim.py``.
``--gpu-full`` runs attention, market, and payments on that same array device.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from agents.rumor_input import RumorMarketClassification, classify_rumor_market
from calibration.fit import (
    _CANDIDATE_FAILURE_PENALTY,
    FitResult,
    GenerationCheckpoint,
    _gpu_sim_requested,
    _limit_worker_threads,
    _log_sq_error,
    _parse_money,
    real_calibration_modes,
    save_generation_checkpoint,
    use_real_calibration_engine,
)
from contracts.events import PaymentSettled, PriceTick, WithdrawalRequested
from contracts.schemas import Rumor, Scenario, ScenarioShock
from params.loader import RegistryError, get_parameter

EventType = Literal["equity", "crypto", "rates"]

_EVENT_TYPES: tuple[EventType, ...] = ("equity", "crypto", "rates")

_EQUITY_CLAIM = "Fictional Retail Co stock is not doing well and shares are falling"
_CRYPTO_CLAIM = "A fictional chain token is crashing and crypto markets are in a panic"
_RATES_CLAIM = "Interest rates are rising and the bond market is in trouble"

_CLAIMS: dict[EventType, str] = {
    "equity": _EQUITY_CLAIM,
    "crypto": _CRYPTO_CLAIM,
    "rates": _RATES_CLAIM,
}

_CREDIBILITY_ROW: dict[EventType, str] = {
    "equity": "Non-bank equity rumor credibility",
    "crypto": "Non-bank crypto rumor credibility",
    "rates": "Non-bank rates rumor credibility",
}
_REACH_ROW: dict[EventType, str] = {
    "equity": "Non-bank equity rumor reach",
    "crypto": "Non-bank crypto rumor reach",
    "rates": "Non-bank rates rumor reach",
}

_DRAWDOWN_ROW: dict[str, str] = {
    "E13": "Non-bank E13 Meta peak-to-trough fraction",
    "E14": "Non-bank E14 GameStop peak-to-trough fraction",
    "E15": "Non-bank E15 LUNA peak-to-trough fraction",
    "E16": "Non-bank E16 SPX peak-to-trough fraction",
}
_DAYS_ROW: dict[str, str] = {
    "E13": "Non-bank E13 Meta days to bottom",
    "E14": "Non-bank E14 GameStop days to bottom",
    "E15": "Non-bank E15 LUNA days to bottom",
    # SVB-29: trading days, so one simulated session (390 minutes) is one day for every event.
    "E16": "Non-bank E16 SPX trading days to bottom",
}
_VOLUME_ROW: dict[str, str] = {
    "E13": "Non-bank E13 Meta volume multiple",
    "E14": "Non-bank E14 GameStop volume multiple",
    "E15": "Non-bank E15 LUNA volume multiple",
    "E16": "Non-bank E16 SPX volume multiple",
}

_PARAM_DIM = "Non-bank calibration parameter dimension"
_PARAM_POPULATION = "Non-bank CMA-ES population size"
_PARAM_GENERATIONS = "Non-bank CMA-ES max generations"
_PARAM_EVAL_MINUTES = "Non-bank eval simulation minutes"
_PARAM_W_DRAWDOWN = "Non-bank drawdown objective weight"
_PARAM_W_DAYS = "Non-bank days objective weight"
_PARAM_W_VOLUME = "Non-bank volume objective weight"
_PARAM_SCALE_COEF = "Non-bank price scale coefficient"
_PARAM_SCALE_CENTER = "Non-bank price scale center"
_PARAM_MINUTES_PER_DAY = "Non-bank sim minutes per day"
_PARAM_RATE_KNOTS = "Non-bank rates path knots"
_PARAM_UNEMPLOYMENT = "Non-bank rates unemployment shock"
_PARAM_SHOCK_MINUTE = "Non-bank rates unemployment shock minute"
_PARAM_SIGMA = "Non-bank CMA-ES initial sigma"
_PARAM_SIGMA_FLOOR = "Non-bank CMA-ES sigma floor"
_PARAM_SIGMA_DECAY = "Non-bank CMA-ES sigma decay"
_PARAM_ELITE = "Non-bank CMA-ES elite minimum"
_PARAM_BEHAVIORAL_DIM = "Calibration behavioral parameter dimension"
_PARAM_DEPOSITS = "SVB-like preset: deposits"
_PARAM_REQUEST_BOUND = "Non-bank daily deposit-request bound"
_PARAM_REQUEST_PENALTY = "Non-bank daily deposit-request bound penalty weight"

_LOG = logging.getLogger("m27.nonbank")


@dataclass(frozen=True, slots=True)
class PricePathStats:
    """Shape of one simulated price path."""

    peak_to_trough: float
    days_to_bottom: float
    volume_multiple: float
    n_prints: int


@dataclass(frozen=True, slots=True)
class ShapeTargets:
    """Point targets. ``None`` means the registry row is UNSET and drops out of the loss."""

    event_id: str
    peak_to_trough: float | None
    days_to_bottom: float | None
    volume_multiple: float | None

    def has_targets(self) -> bool:
        """True when at least one point target is set, so a loss means something."""
        return any(
            value is not None
            for value in (self.peak_to_trough, self.days_to_bottom, self.volume_multiple)
        )

    def unset_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.peak_to_trough is None:
            missing.append("peak-to-trough")
        if self.days_to_bottom is None:
            missing.append("days-to-bottom")
        if self.volume_multiple is None:
            missing.append("volume-multiple")
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class NonbankTargets:
    equity: ShapeTargets
    gamestop: ShapeTargets
    crypto: ShapeTargets
    rates: ShapeTargets


@dataclass(frozen=True, slots=True)
class NonbankHoldoutResult:
    crypto_loss: float
    rates_loss: float
    crypto: PricePathStats
    rates: PricePathStats
    crypto_excluded: tuple[str, ...] = ()
    rates_excluded: tuple[str, ...] = ()
    # SVB-38c: per-day profile of each holdout path (``price_path_profile``) and the wall
    # time of each holdout run. Read from the same log as the stats; no extra engine run.
    crypto_profile: dict[str, object] | None = None
    rates_profile: dict[str, object] | None = None
    crypto_seconds: float | None = None
    rates_seconds: float | None = None


def _registry_float(name: str, path: Path | None = None) -> float:
    raw = get_parameter(name, path).value.strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise RegistryError(f"cannot parse numeric registry value for {name}: {raw!r}") from exc


def _point_target(name: str, path: Path | None = None) -> float | None:
    """Return a point number, or None when the row is UNSET or not a point value."""
    row = get_parameter(name, path)
    if row.status.strip().upper() == "UNSET":
        return None
    raw = row.value.strip()
    if raw.upper() == "UNSET":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _shape_targets(event_id: str, path: Path | None) -> ShapeTargets:
    return ShapeTargets(
        event_id=event_id,
        peak_to_trough=_point_target(_DRAWDOWN_ROW[event_id], path),
        days_to_bottom=_point_target(_DAYS_ROW[event_id], path),
        volume_multiple=_point_target(_VOLUME_ROW[event_id], path),
    )


def reachable_targets(
    target: ShapeTargets, *, steps: int, minutes_per_day: float
) -> tuple[ShapeTargets, tuple[str, ...]]:
    """Drop a days-to-bottom target the run is too short to reach, and say why.

    A path of ``steps`` minutes can't show more than ``steps / minutes_per_day``
    days to bottom, so a longer target is a constant penalty that no vector can
    reduce. It's excluded from the loss and reported instead.
    """
    days = target.days_to_bottom
    if days is None or minutes_per_day <= 0.0:
        return target, ()
    horizon = float(steps) / float(minutes_per_day)
    if days <= horizon + 1e-9:
        return target, ()
    needed = int(math.ceil(days * minutes_per_day))
    reason = (
        f"{target.event_id} days-to-bottom {days:g} needs {needed} sim minutes; ran {int(steps)}"
    )
    return replace(target, days_to_bottom=None), (reason,)


def term_status(
    raw: ShapeTargets,
    used: ShapeTargets,
    excluded: tuple[str, ...],
    *,
    weights: tuple[float, float, float],
) -> dict[str, str]:
    """Why each shape term is or is not in the loss (SVB-38c).

    ``raw`` is the target as loaded from the registry, ``used`` is the target after
    ``reachable_targets``, and ``excluded`` is its reason tuple. Statuses are
    ``evaluated``, ``unset: ...`` (no point value in the registry), ``excluded: ...``
    (set, but the horizon can't reach it) and ``weight 0``.
    """
    rows = {
        "peak_to_trough": _DRAWDOWN_ROW.get(raw.event_id, "?"),
        "days_to_bottom": _DAYS_ROW.get(raw.event_id, "?"),
        "volume_multiple": _VOLUME_ROW.get(raw.event_id, "?"),
    }
    triples = (
        ("peak_to_trough", raw.peak_to_trough, used.peak_to_trough, weights[0]),
        ("days_to_bottom", raw.days_to_bottom, used.days_to_bottom, weights[1]),
        ("volume_multiple", raw.volume_multiple, used.volume_multiple, weights[2]),
    )
    out: dict[str, str] = {}
    for key, before, after, weight in triples:
        if before is None:
            out[key] = f"unset: registry row {rows[key]!r} has no point value"
        elif after is None:
            reason = "; ".join(excluded) if excluded else "dropped by reachable_targets"
            out[key] = f"excluded: {reason}"
        elif weight == 0.0:
            out[key] = "weight 0"
        else:
            out[key] = "evaluated"
    return out


def load_nonbank_targets(path: Path | None = None) -> NonbankTargets:
    """E13 is the equity fit. E15 and E16 are holdouts. E14 is documented only."""
    return NonbankTargets(
        equity=_shape_targets("E13", path),
        gamestop=_shape_targets("E14", path),
        crypto=_shape_targets("E15", path),
        rates=_shape_targets("E16", path),
    )


def _parse_knots(raw: str) -> list[tuple[float, float]]:
    knots: list[tuple[float, float]] = []
    for part in raw.split(";"):
        piece = part.strip()
        if not piece:
            continue
        left, right = piece.split(":", 1)
        knots.append((float(left), float(right)))
    return knots


def build_nonbank_scenario(
    event_type: EventType,
    seed: int,
    *,
    path: Path | None = None,
) -> Scenario:
    """Explicit credibility, reach, and event-type tag. The classifier does not set them."""
    if event_type not in _EVENT_TYPES:
        raise RegistryError(f"unknown non-bank event type: {event_type}")
    credibility = _registry_float(_CREDIBILITY_ROW[event_type], path)
    reach = int(_registry_float(_REACH_ROW[event_type], path))
    claim = _CLAIMS[event_type]
    rumor = Rumor(
        id=f"nonbank-{event_type}",
        claim=claim,
        credibility=credibility,
        reach=reach,
        notes=[f"event-type:{event_type}"],
    )
    rate_path: list[tuple[float, float]] = []
    shocks: list[ScenarioShock] = []
    if event_type == "rates":
        rate_path = _parse_knots(get_parameter(_PARAM_RATE_KNOTS, path).value)
        shocks.append(
            ScenarioShock(
                kind="unemployment",
                at=_registry_float(_PARAM_SHOCK_MINUTE, path),
                params={"rate": _registry_float(_PARAM_UNEMPLOYMENT, path)},
            )
        )
    return Scenario(
        preset=f"nonbank-{event_type}",
        seed=seed,
        rate_path=rate_path,
        shocks=shocks,
        rumors=[rumor],
    )


def label_nonbank_claim(event_type: EventType) -> RumorMarketClassification:
    """Same lexicon as the rumor box. Does not change the scenario's credibility."""
    return classify_rumor_market(_CLAIMS[event_type])


def summarize_price_path(
    times: list[float],
    prices: list[float],
    volumes: list[float],
    *,
    minutes_per_day: float,
) -> PricePathStats:
    """Peak-to-trough after the first peak, days from peak to trough, volume vs first print.

    Volume multiple is peak volume divided by the first positive print. That
    definition is the sim-side comparison (ASSUMED); the historical multiples
    in the registry are separate verified or UNSET rows.
    """
    n = min(len(times), len(prices), len(volumes))
    if n == 0 or minutes_per_day <= 0.0:
        return PricePathStats(0.0, 0.0, 0.0, 0)
    peak_i = 0
    for i in range(1, n):
        if prices[i] > prices[peak_i]:
            peak_i = i
    trough_i = peak_i
    for i in range(peak_i + 1, n):
        if prices[i] < prices[trough_i]:
            trough_i = i
    peak = prices[peak_i]
    trough = prices[trough_i]
    drawdown = 0.0 if peak <= 0.0 else max(0.0, (peak - trough) / peak)
    days = max(0.0, (times[trough_i] - times[peak_i]) / minutes_per_day)
    baseline = 0.0
    for volume in volumes[:n]:
        if volume > 0.0:
            baseline = volume
            break
    peak_volume = max(volumes[:n])
    multiple = 0.0 if baseline <= 0.0 else max(0.0, peak_volume / baseline)
    return PricePathStats(drawdown, days, multiple, n)


def _stats_from_events(log: Iterable[object], minutes_per_day: float) -> PricePathStats:
    times: list[float] = []
    prices: list[float] = []
    volumes: list[float] = []
    for event in log:
        if isinstance(event, PriceTick):
            times.append(float(event.t))
            prices.append(float(event.price))
            volumes.append(float(event.volume))
    return summarize_price_path(
        times,
        prices,
        volumes,
        minutes_per_day=minutes_per_day,
    )


def _stats_from_log(log: Iterable[object], path: Path | None) -> PricePathStats:
    return _stats_from_events(log, _registry_float(_PARAM_MINUTES_PER_DAY, path))


def price_path_profile(
    times: list[float],
    prices: list[float],
    volumes: list[float],
    *,
    minutes_per_day: float,
    steps: int,
) -> dict[str, object]:
    """Per-day shape of one price path, to see whether a shock keeps acting after day one (SVB-38c).

    Days are ``int(t // minutes_per_day)`` buckets of the sim clock. ``peak_to_trough_day1``
    and ``peak_to_trough_full`` use ``summarize_price_path``, the same statistic the loss
    scores. ``covers_horizon`` is False when the last print is more than one tick gap short
    of ``steps``, which is how a schedule that stops early shows up. ``max_gap_minutes`` above
    1 means the clock is not uniform one-minute ticks (for example a 15-minute overnight).
    """
    n = min(len(times), len(prices), len(volumes))
    base: dict[str, object] = {"steps": int(steps), "minutes_per_day": float(minutes_per_day)}
    if n == 0 or minutes_per_day <= 0.0:
        base.update({"n_prints": 0, "covers_horizon": False, "days": []})
        return base
    t = [float(x) for x in times[:n]]
    p = [float(x) for x in prices[:n]]
    v = [float(x) for x in volumes[:n]]
    gaps = [t[i] - t[i - 1] for i in range(1, n)]
    max_gap = max(gaps) if gaps else 0.0
    full = summarize_price_path(t, p, v, minutes_per_day=minutes_per_day)
    day1_n = sum(1 for x in t if x < minutes_per_day)
    day1 = summarize_price_path(t[:day1_n], p[:day1_n], v[:day1_n], minutes_per_day=minutes_per_day)
    peak_i = 0
    for i in range(1, n):
        if p[i] > p[peak_i]:
            peak_i = i
    trough_i = peak_i
    for i in range(peak_i + 1, n):
        if p[i] < p[trough_i]:
            trough_i = i
    running_low = p[0]
    last_new_low_t: float | None = None
    for i in range(1, n):
        if p[i] < running_low:
            running_low = p[i]
            last_new_low_t = t[i]
    days: list[dict[str, object]] = []
    prior_low = math.inf
    current: dict[str, object] | None = None
    for i in range(n):
        day = int(t[i] // minutes_per_day)
        if current is None or current["day"] != day:
            if current is not None:
                prior_low = min(prior_low, float(current["low"]))  # type: ignore[arg-type]
            current = {
                "day": day,
                "n_prints": 0,
                "t_first": t[i],
                "t_last": t[i],
                "open": p[i],
                "close": p[i],
                "low": p[i],
                "high": p[i],
                "volume": 0.0,
            }
            days.append(current)
        current["n_prints"] = int(current["n_prints"]) + 1  # type: ignore[arg-type]
        current["t_last"] = t[i]
        current["close"] = p[i]
        current["low"] = min(float(current["low"]), p[i])  # type: ignore[arg-type]
        current["high"] = max(float(current["high"]), p[i])  # type: ignore[arg-type]
        current["volume"] = float(current["volume"]) + v[i]  # type: ignore[arg-type]
        current["new_low"] = float(current["low"]) < prior_low  # type: ignore[arg-type]
    for row in days:
        opened = float(row["open"])  # type: ignore[arg-type]
        closed = float(row["close"])  # type: ignore[arg-type]
        row["close_return"] = 0.0 if opened == 0.0 else closed / opened - 1.0
    first_day = int(t[0] // minutes_per_day)
    later_new_lows = [
        int(row["day"])  # type: ignore[call-overload]
        for row in days
        if row["day"] != first_day and row["new_low"]
    ]
    base.update(
        {
            "n_prints": n,
            "t_first": t[0],
            "t_last": t[-1],
            "max_gap_minutes": max_gap,
            "clock": (
                "uniform 1-minute"
                if max_gap <= 1.0 + 1e-9
                else f"non-uniform (max gap {max_gap:g} min)"
            ),
            "covers_horizon": t[-1] >= float(steps) - 1.0 - max(1.0, max_gap),
            "peak_minute": t[peak_i],
            "trough_minute": t[trough_i],
            "peak_to_trough_minutes": t[trough_i] - t[peak_i],
            "last_new_low_minute": last_new_low_t,
            "peak_to_trough_day1": day1.peak_to_trough,
            "peak_to_trough_full": full.peak_to_trough,
            "peak_to_trough_added_after_day1": full.peak_to_trough - day1.peak_to_trough,
            "share_of_drawdown_after_day1": (
                0.0
                if full.peak_to_trough <= 0.0
                else (full.peak_to_trough - day1.peak_to_trough) / full.peak_to_trough
            ),
            "days_with_new_low_after_day1": later_new_lows,
            # Any new low counts, however small; read the share above for how much.
            "new_low_after_day1": bool(later_new_lows),
            "days": days,
        }
    )
    return base


_PRICE_SOURCE_FIELDS = ("symbol", "ticker", "instrument", "bank_id", "bank")


def _price_sources(log: Iterable[object]) -> list[str]:
    """Distinct instrument labels on ``PriceTick`` events, if the event carries one.

    The harness pools every ``PriceTick``. More than one label means two instruments are
    interleaved in one path.
    """
    seen: set[str] = set()
    for event in log:
        if not isinstance(event, PriceTick):
            continue
        for field in _PRICE_SOURCE_FIELDS:
            value = getattr(event, field, None)
            if value is not None:
                seen.add(f"{field}={value}")
                break
    return sorted(seen)


def event_activity_by_day(
    log: Iterable[object], *, minutes_per_day: float
) -> dict[str, dict[str, int]]:
    """Count of each event type per sim day, so a path that goes quiet after day one shows."""
    out: dict[str, dict[str, int]] = {}
    if minutes_per_day <= 0.0:
        return out
    for event in log:
        t = getattr(event, "t", None)
        if t is None:
            continue
        try:
            day = str(int(float(t) // minutes_per_day))
        except (TypeError, ValueError):
            continue
        bucket = out.setdefault(day, {})
        name = type(event).__name__
        bucket[name] = bucket.get(name, 0) + 1
    return out


def _measure_holdout_path(
    scenario: Scenario,
    *,
    steps: int,
    path: Path | None,
    param_vector: NDArray[np.float64] | None,
) -> tuple[PricePathStats, dict[str, object], float]:
    """One engine run: the scored stats, the per-day profile and the wall time."""
    started = time.perf_counter()
    log = _run_engine_log(scenario, steps=steps, path=path, param_vector=param_vector)
    seconds = time.perf_counter() - started
    minutes_per_day = _registry_float(_PARAM_MINUTES_PER_DAY, path)
    times: list[float] = []
    prices: list[float] = []
    volumes: list[float] = []
    for event in log:
        if isinstance(event, PriceTick):
            times.append(float(event.t))
            prices.append(float(event.price))
            volumes.append(float(event.volume))
    stats = summarize_price_path(times, prices, volumes, minutes_per_day=minutes_per_day)
    profile = price_path_profile(
        times, prices, volumes, minutes_per_day=minutes_per_day, steps=int(steps)
    )
    profile["price_sources"] = _price_sources(log)
    profile["activity_by_day"] = event_activity_by_day(log, minutes_per_day=minutes_per_day)
    profile["seconds"] = seconds
    return stats, profile, seconds


def daily_deposit_request_fraction(
    log: Iterable[object],
    *,
    path: Path | None = None,
) -> float:
    """Max one-day WithdrawalRequested (else PaymentSettled) over deposits."""
    deposits = _parse_money(get_parameter(_PARAM_DEPOSITS, path).value)
    minutes_per_day = _registry_float(_PARAM_MINUTES_PER_DAY, path)
    if deposits <= 0.0 or minutes_per_day <= 0.0:
        return 0.0
    requested: dict[int, float] = {}
    settled: dict[int, float] = {}
    saw_request = False
    for event in log:
        if isinstance(event, WithdrawalRequested):
            saw_request = True
            day = int(float(event.t) // minutes_per_day)
            requested[day] = requested.get(day, 0.0) + float(event.amount)
        elif isinstance(event, PaymentSettled):
            day = int(float(event.t) // minutes_per_day)
            settled[day] = settled.get(day, 0.0) + float(event.amount)
    by_day = requested if saw_request else settled
    if not by_day:
        return 0.0
    return max(amount / deposits for amount in by_day.values())


def nonbank_request_bound(path: Path | None = None) -> float:
    """ASSUMED 0.1% of deposits per day."""
    return _registry_float(_PARAM_REQUEST_BOUND, path)


def nonbank_request_bound_penalty(
    fraction: float,
    *,
    path: Path | None = None,
) -> float:
    """Squared relative excess over the daily deposit-request bound."""
    bound = nonbank_request_bound(path)
    weight = _registry_float(_PARAM_REQUEST_PENALTY, path)
    excess = max(0.0, float(fraction) - bound)
    if excess == 0.0 or bound <= 0.0 or weight == 0.0:
        return 0.0
    return weight * (excess / bound) ** 2


def validate_nonbank_bound(
    fraction: float,
    *,
    path: Path | None = None,
) -> bool:
    """True when daily deposit requests stay at or under the ASSUMED bound."""
    return float(fraction) <= nonbank_request_bound(path)


def _run_engine_log(
    scenario: Scenario,
    *,
    steps: int,
    path: Path | None,
    param_vector: NDArray[np.float64] | None,
) -> list[object]:
    from engine.depositor_rows import nonbank_depositor_rows

    # SVB-52 (D3): non-bank presets read the dedicated non-bank depositor rows, not
    # the SVB-fitted ones (worry bias, mobile share, request rate, capacity).
    with nonbank_depositor_rows():
        return _run_engine_log_inner(scenario, steps=steps, path=path, param_vector=param_vector)


def _run_engine_log_inner(
    scenario: Scenario,
    *,
    steps: int,
    path: Path | None,
    param_vector: NDArray[np.float64] | None,
) -> list[object]:
    from engine.modes import ModuleMode
    from engine.registry import build_engine, run_scenario

    if _gpu_sim_requested() and param_vector is not None:
        from engine.gpu_sim import simulate_logs

        return list(simulate_logs(scenario, steps, [np.asarray(param_vector, dtype=np.float64)])[0])
    if use_real_calibration_engine():
        engine = build_engine(
            modes=real_calibration_modes(),
            default_mode=ModuleMode.FAKE,
            scenario=scenario,
        )
        if param_vector is not None:
            from engine.agents.decide import DecisionModule, apply_calibration_vector

            module = engine["M06"]
            if isinstance(module, DecisionModule):
                apply_calibration_vector(module, np.asarray(param_vector, dtype=np.float64))
                from engine.gpu_full_io import apply_lane_feedback

                apply_lane_feedback(module, np.asarray(param_vector, dtype=np.float64))  # SVB-65: gain, dispersion, impact
        return list(engine.run(steps))
    return list(run_scenario(scenario, steps))


def measure_price_path(
    scenario: Scenario,
    *,
    steps: int,
    path: Path | None = None,
    param_vector: NDArray[np.float64] | None = None,
) -> PricePathStats:
    """Run the current engine and read ``PriceTick`` only.

    ``param_vector`` is the behavioral vector (length ``Calibration behavioral
    parameter dimension``). The fake-mode 3-vector does not pass one. With both
    GPU flags set, a behavioral vector uses the batched depositor kernel (NumPy
    when no CUDA device is visible).
    """
    return _stats_from_log(
        _run_engine_log(scenario, steps=steps, path=path, param_vector=param_vector),
        path,
    )


def measure_nonbank_observables(
    scenario: Scenario,
    *,
    steps: int,
    path: Path | None = None,
    param_vector: NDArray[np.float64] | None = None,
) -> tuple[PricePathStats, float]:
    """Price-path stats plus the daily deposit-request fraction from one log."""
    log = _run_engine_log(scenario, steps=steps, path=path, param_vector=param_vector)
    return _stats_from_log(log, path), daily_deposit_request_fraction(log, path=path)


def scale_price_stats(
    stats: PricePathStats,
    vector: NDArray[np.float64],
    *,
    coefficient: float,
    center: float,
) -> PricePathStats:
    """Fake-mode stand-in: scale the read path, mirroring the E1 amount scale.

    The engine run is not repeated for the scale. Drawdown stays in ``[0, 1]``.
    """
    if vector.shape[0] < 3:
        raise RegistryError("non-bank scale vector needs 3 components")

    def _scale(component: float) -> float:
        return math.exp(coefficient * (component - center))

    drawdown_s, days_s, volume_s = (
        _scale(float(vector[0])),
        _scale(float(vector[1])),
        _scale(float(vector[2])),
    )
    return PricePathStats(
        peak_to_trough=min(1.0, max(0.0, stats.peak_to_trough * drawdown_s)),
        days_to_bottom=max(0.0, stats.days_to_bottom * days_s),
        volume_multiple=max(0.0, stats.volume_multiple * volume_s),
        n_prints=stats.n_prints,
    )


def shape_loss(
    stats: PricePathStats,
    target: ShapeTargets,
    *,
    weights: tuple[float, float, float],
) -> float:
    """Weighted squared log-error. UNSET targets contribute nothing."""
    pairs = (
        (stats.peak_to_trough, target.peak_to_trough, weights[0]),
        (stats.days_to_bottom, target.days_to_bottom, weights[1]),
        (stats.volume_multiple, target.volume_multiple, weights[2]),
    )
    total = 0.0
    for observed, expected, weight in pairs:
        if expected is None or weight == 0.0:
            continue
        total += weight * _log_sq_error(observed, expected)
    return total


def shape_loss_terms(
    stats: PricePathStats,
    target: ShapeTargets,
    *,
    weights: tuple[float, float, float],
) -> dict[str, float | None]:
    """Per-statistic weighted squared log-error. UNSET targets are None.

    The non-None terms sum to ``shape_loss`` for the same arguments.
    """
    pairs = (
        ("peak_to_trough", stats.peak_to_trough, target.peak_to_trough, weights[0]),
        ("days_to_bottom", stats.days_to_bottom, target.days_to_bottom, weights[1]),
        ("volume_multiple", stats.volume_multiple, target.volume_multiple, weights[2]),
    )
    terms: dict[str, float | None] = {}
    for key, observed, expected, weight in pairs:
        if expected is None:
            terms[key] = None
        elif weight == 0.0:
            terms[key] = 0.0
        else:
            terms[key] = weight * _log_sq_error(observed, expected)
    return terms


def _weights(path: Path | None) -> tuple[float, float, float]:
    return (
        _registry_float(_PARAM_W_DRAWDOWN, path),
        _registry_float(_PARAM_W_DAYS, path),
        _registry_float(_PARAM_W_VOLUME, path),
    )


def objective_nonbank(
    vector: NDArray[np.float64],
    *,
    scenario: Scenario,
    steps: int,
    target: ShapeTargets,
    path: Path | None = None,
    raw: PricePathStats | None = None,
    request_fraction: float | None = None,
) -> float:
    if raw is None:
        observed, measured_fraction = measure_nonbank_observables(
            scenario,
            steps=steps,
            path=path,
        )
        fraction = measured_fraction if request_fraction is None else request_fraction
    else:
        observed = raw
        fraction = 0.0 if request_fraction is None else request_fraction
    scaled = scale_price_stats(
        observed,
        vector,
        coefficient=_registry_float(_PARAM_SCALE_COEF, path),
        center=_registry_float(_PARAM_SCALE_CENTER, path),
    )
    return shape_loss(scaled, target, weights=_weights(path)) + nonbank_request_bound_penalty(
        fraction,
        path=path,
    )


@dataclass(frozen=True, slots=True)
class _NonbankEvalTask:
    index: int
    vector: tuple[float, ...]
    scenario: Scenario
    steps: int
    target: ShapeTargets
    weights: tuple[float, float, float]
    minutes_per_day: float


def _objective_from_stats(
    stats: PricePathStats,
    target: ShapeTargets,
    weights: tuple[float, float, float],
    *,
    request_fraction: float = 0.0,
    path: Path | None = None,
) -> float:
    return shape_loss(stats, target, weights=weights) + nonbank_request_bound_penalty(
        request_fraction,
        path=path,
    )


def _evaluate_nonbank_task(task: _NonbankEvalTask) -> tuple[int, float, float, str | None]:
    """Worker entry: returns (index, objective, seconds, error_message)."""
    _limit_worker_threads()
    started = time.perf_counter()
    try:
        if task.minutes_per_day <= 0.0:
            raise RegistryError("non-bank sim minutes per day must be positive")
        vector = np.asarray(task.vector, dtype=np.float64)
        stats, request_fraction = measure_nonbank_observables(
            task.scenario,
            steps=task.steps,
            param_vector=vector,
        )
        objective = _objective_from_stats(
            stats,
            task.target,
            task.weights,
            request_fraction=request_fraction,
        )
        return task.index, objective, time.perf_counter() - started, None
    except Exception as exc:
        _LOG.exception("non-bank candidate %s failed", task.index)
        return task.index, _CANDIDATE_FAILURE_PENALTY, time.perf_counter() - started, str(exc)


def _score_population_logs(
    logs: list[object],
    *,
    target: ShapeTargets,
    weights: tuple[float, float, float],
    minutes_per_day: float,
    on_candidate: Callable[[int, float, float], None] | None,
    on_candidate_error: Callable[[int, str], None] | None,
    elapsed: float,
) -> list[float]:
    share = elapsed / len(logs) if logs else 0.0
    objectives: list[float] = []
    for index, item in enumerate(logs):
        if isinstance(item, BaseException):
            message = str(item)
            if on_candidate_error is not None:
                on_candidate_error(index, message)
            objective = _CANDIDATE_FAILURE_PENALTY
        else:
            stats = _stats_from_events(item, minutes_per_day)  # type: ignore[arg-type]
            request_fraction = daily_deposit_request_fraction(item)  # type: ignore[arg-type]
            objective = _objective_from_stats(
                stats,
                target,
                weights,
                request_fraction=request_fraction,
            )
        objectives.append(objective)
        if on_candidate is not None:
            on_candidate(index, objective, share)
    return objectives


def _evaluate_nonbank_gpu_sim(
    samples: list[NDArray[np.float64]],
    *,
    scenario: Scenario,
    steps: int,
    target: ShapeTargets,
    weights: tuple[float, float, float],
    minutes_per_day: float,
    on_candidate: Callable[[int, float, float], None] | None,
    on_candidate_error: Callable[[int, str], None] | None,
) -> list[float]:
    """Score one generation as a single batched real-engine run (ADR-051, ADR-052)."""
    from engine.depositor_rows import nonbank_depositor_rows
    from engine.gpu_sim import simulate_logs as _simulate_logs

    started = time.perf_counter()

    def simulate_logs(*args: object, **kwargs: object) -> list[list[object]]:
        # SVB-52 (D3): the batched kernel reads the non-bank depositor rows too.
        with nonbank_depositor_rows():
            return _simulate_logs(*args, **kwargs)

    def _run() -> list[list[object]]:
        return simulate_logs(scenario, steps, samples)

    logs: list[object]
    try:
        logs = list(_run())
    except Exception as first_exc:
        _LOG.exception("gpu-sim non-bank batch failed; retrying once")
        try:
            logs = list(_run())
        except Exception:
            logs = []
            for index, sample in enumerate(samples):
                try:
                    logs.extend(simulate_logs(scenario, steps, [sample]))
                except Exception as exc:
                    _LOG.exception("gpu-sim non-bank candidate %s failed", index)
                    logs.append(exc)
            if not logs:
                logs = [first_exc for _ in samples]
    elapsed = time.perf_counter() - started
    return _score_population_logs(
        logs,
        target=target,
        weights=weights,
        minutes_per_day=minutes_per_day,
        on_candidate=on_candidate,
        on_candidate_error=on_candidate_error,
        elapsed=elapsed,
    )


def _evaluate_nonbank_population(
    samples: list[NDArray[np.float64]],
    *,
    scenario: Scenario,
    steps: int,
    target: ShapeTargets,
    path: Path | None,
    workers: int,
    on_candidate: Callable[[int, float, float], None] | None,
    on_candidate_error: Callable[[int, str], None] | None,
) -> list[float]:
    """Price-path loss for each behavioral vector, in candidate index order."""
    if not samples:
        return []
    weights = _weights(path)
    minutes_per_day = _registry_float(_PARAM_MINUTES_PER_DAY, path)
    if _gpu_sim_requested():
        return _evaluate_nonbank_gpu_sim(
            samples,
            scenario=scenario,
            steps=steps,
            target=target,
            weights=weights,
            minutes_per_day=minutes_per_day,
            on_candidate=on_candidate,
            on_candidate_error=on_candidate_error,
        )
    worker_count = max(1, workers)

    def run_index(
        index: int,
        sample: NDArray[np.float64],
        retry: bool,
    ) -> tuple[int, float, float, str | None]:
        task = _NonbankEvalTask(
            index=index,
            vector=tuple(float(x) for x in sample),
            scenario=scenario,
            steps=steps,
            target=target,
            weights=weights,
            minutes_per_day=minutes_per_day,
        )
        idx, objective, elapsed, error = _evaluate_nonbank_task(task)
        if error is not None and not retry:
            idx, objective, elapsed, error = _evaluate_nonbank_task(task)
        return idx, objective, elapsed, error

    objectives: list[float | None] = [None] * len(samples)
    if worker_count == 1:
        for index, sample in enumerate(samples):
            idx, objective, elapsed, error = run_index(index, sample, retry=False)
            if error is not None and on_candidate_error is not None:
                on_candidate_error(idx, error)
            objectives[idx] = objective
            if on_candidate is not None:
                on_candidate(idx, objective, elapsed)
        return [float(item) for item in objectives]

    ctx = get_context("spawn")
    tasks = [
        _NonbankEvalTask(
            index=index,
            vector=tuple(float(x) for x in sample),
            scenario=scenario,
            steps=steps,
            target=target,
            weights=weights,
            minutes_per_day=minutes_per_day,
        )
        for index, sample in enumerate(samples)
    ]
    task_by_index = {task.index: task for task in tasks}
    retried: set[int] = set()
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
        futures = {pool.submit(_evaluate_nonbank_task, task): task.index for task in tasks}
        while futures:
            for future in as_completed(list(futures)):
                index = futures.pop(future)
                idx, objective, elapsed, error = future.result()
                if error is not None:
                    if on_candidate_error is not None:
                        on_candidate_error(idx, error)
                    if idx not in retried:
                        retried.add(idx)
                        retry_future = pool.submit(_evaluate_nonbank_task, task_by_index[idx])
                        futures[retry_future] = idx
                        continue
                    objective = _CANDIDATE_FAILURE_PENALTY
                objectives[idx] = objective
                if on_candidate is not None:
                    on_candidate(idx, objective, elapsed)
    return [float(item if item is not None else _CANDIDATE_FAILURE_PENALTY) for item in objectives]


def _record_generation(
    *,
    generation: int,
    best_obj: float,
    best_vector: NDArray[np.float64],
    mean: NDArray[np.float64],
    sigma: float,
    seed: int,
    eval_steps: int,
    pop_size: int,
    run_id: str,
    checkpoint_dir: Path | None,
    checkpoint_every: int,
    on_generation: Callable[[int, float, tuple[float, ...]], None] | None,
    diagnostics_dir: Path | None = None,
) -> None:
    vector = tuple(float(x) for x in best_vector)
    if on_generation is not None:
        on_generation(generation + 1, float(best_obj), vector)
    if diagnostics_dir is not None:
        from calibration.diagnostics import record_generation

        record_generation(
            diagnostics_dir,
            generation + 1,
            float(best_obj),
            vector,
            sigma=float(sigma),
            mean=tuple(float(x) for x in mean),
            seed=seed,
        )
    if checkpoint_dir is not None and (generation + 1) % max(1, checkpoint_every) == 0:
        save_generation_checkpoint(
            GenerationCheckpoint(
                run_id=run_id,
                seed=seed,
                generation=generation + 1,
                mean=tuple(float(x) for x in mean),
                sigma=float(sigma),
                best_vector=vector,
                best_objective=float(best_obj),
                steps=eval_steps,
                population=pop_size,
                real_engine=use_real_calibration_engine(),
            ),
            checkpoint_dir,
        )


def _search_limits(
    *,
    path: Path | None,
    generations: int | None,
    population: int | None,
    steps: int | None,
) -> tuple[int, int, int]:
    max_generations = generations if generations is not None else int(
        _registry_float(_PARAM_GENERATIONS, path)
    )
    if population is None:
        pop_size = int(_registry_float(_PARAM_POPULATION, path))
    else:
        pop_size = population
    if max_generations < 1 or pop_size < 2:
        raise RegistryError("non-bank fit needs at least 1 generation and population 2")
    eval_steps = steps if steps is not None else int(_registry_float(_PARAM_EVAL_MINUTES, path))
    if eval_steps < 1:
        raise RegistryError("non-bank eval minutes must be at least 1")
    return max_generations, pop_size, eval_steps


def _fit_nonbank_on_engine(
    seed: int,
    *,
    path: Path | None,
    generations: int | None,
    population: int | None,
    steps: int | None,
    workers: int,
    run_id: str | None,
    checkpoint_dir: Path | None,
    checkpoint_every: int,
    on_generation: Callable[[int, float, tuple[float, ...]], None] | None,
    start_generation: int,
    initial_mean: NDArray[np.float64] | None,
    initial_sigma: float | None,
    initial_best_vector: NDArray[np.float64] | None,
    initial_best_objective: float | None,
    on_candidate: Callable[[int, int, float, float], None] | None,
    on_candidate_error: Callable[[int, str], None] | None,
    diagnostics_dir: Path | None = None,
) -> FitResult:
    """CMA-ES on the equity preset. Each candidate is a real-engine price path.

    The search vector is the behavioral vector the depositor kernel applies.
    The loss is the E13 shape loss of that path. Crypto and rates are not fit.
    """
    targets = load_nonbank_targets(path)
    dim = int(_registry_float(_PARAM_BEHAVIORAL_DIM, path))
    max_generations, pop_size, eval_steps = _search_limits(
        path=path,
        generations=generations,
        population=population,
        steps=steps,
    )
    equity_target, equity_excluded = reachable_targets(
        targets.equity, steps=eval_steps, minutes_per_day=_registry_float(_PARAM_MINUTES_PER_DAY, path)
    )
    if equity_excluded:
        _LOG.info("non-bank fit excludes unreachable targets: %s", "; ".join(equity_excluded))
    targets = replace(targets, equity=equity_target)
    scenario = build_nonbank_scenario("equity", seed, path=path)
    rng = np.random.default_rng(seed)
    center = _registry_float(_PARAM_SCALE_CENTER, path)
    if initial_mean is None:
        mean = np.full(dim, center, dtype=np.float64)
    else:
        mean = np.clip(np.asarray(initial_mean, dtype=np.float64), 0.0, 1.0).copy()
    sigma = _registry_float(_PARAM_SIGMA, path) if initial_sigma is None else float(initial_sigma)
    sigma_floor = _registry_float(_PARAM_SIGMA_FLOOR, path)
    sigma_decay = _registry_float(_PARAM_SIGMA_DECAY, path)
    elite_min = int(_registry_float(_PARAM_ELITE, path))
    if initial_best_vector is None:
        best_vector = mean.copy()
    else:
        best_vector = np.clip(np.asarray(initial_best_vector, dtype=np.float64), 0.0, 1.0).copy()
    if initial_best_objective is None:
        best_stats, best_fraction = measure_nonbank_observables(
            scenario,
            steps=eval_steps,
            path=path,
            param_vector=best_vector,
        )
        best_obj = _objective_from_stats(
            best_stats,
            targets.equity,
            _weights(path),
            request_fraction=best_fraction,
            path=path,
        )
    else:
        best_obj = float(initial_best_objective)
    resolved_run_id = run_id or f"m27-nonbank-{seed}"

    def _on_candidate(index: int, objective: float, elapsed: float) -> None:
        if on_candidate is not None:
            on_candidate(generation, index, objective, elapsed)

    for generation in range(start_generation, max_generations):
        samples = [
            np.clip(mean + rng.normal(0.0, sigma, size=dim), 0.0, 1.0) for _ in range(pop_size)
        ]
        losses = _evaluate_nonbank_population(
            samples,
            scenario=scenario,
            steps=eval_steps,
            target=targets.equity,
            path=path,
            workers=workers,
            on_candidate=_on_candidate,
            on_candidate_error=on_candidate_error,
        )
        candidates = [(losses[index], samples[index]) for index in range(pop_size)]
        candidates.sort(key=lambda item: item[0])
        top = [vec for _, vec in candidates[: max(elite_min, pop_size // 2)]]
        mean = np.mean(top, axis=0)
        sigma = max(sigma_floor, sigma * sigma_decay)
        if candidates[0][0] < best_obj:
            best_obj = candidates[0][0]
            best_vector = candidates[0][1].copy()
        _record_generation(
            generation=generation,
            best_obj=best_obj,
            best_vector=best_vector,
            mean=mean,
            sigma=sigma,
            seed=seed,
            eval_steps=eval_steps,
            pop_size=pop_size,
            run_id=resolved_run_id,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            on_generation=on_generation,
            diagnostics_dir=diagnostics_dir,
        )
    return FitResult(
        seed=seed,
        vector=tuple(float(x) for x in best_vector),
        objective=float(best_obj),
        generations=max_generations,
    )


def fit_nonbank_parameters(
    seed: int,
    *,
    path: Path | None = None,
    generations: int | None = None,
    population: int | None = None,
    steps: int | None = None,
    engine_candidates: bool = False,
    workers: int = 1,
    run_id: str | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 1,
    on_generation: Callable[[int, float, tuple[float, ...]], None] | None = None,
    start_generation: int = 0,
    initial_mean: NDArray[np.float64] | None = None,
    initial_sigma: float | None = None,
    initial_best_vector: NDArray[np.float64] | None = None,
    initial_best_objective: float | None = None,
    on_candidate: Callable[[int, int, float, float], None] | None = None,
    on_candidate_error: Callable[[int, str], None] | None = None,
    diagnostics_dir: Path | None = None,
) -> FitResult:
    """CMA-ES-style search on the equity preset (E13). Does not touch the SVB vector.

    The default path measures one price path and scales it with a 3-vector.
    ``engine_candidates`` with the real calibration engine scores each sample
    on its own path. ``--gpu-sim`` batches those samples in one kernel call.
    """
    if engine_candidates and use_real_calibration_engine():
        return _fit_nonbank_on_engine(
            seed,
            path=path,
            generations=generations,
            population=population,
            steps=steps,
            workers=workers,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            on_generation=on_generation,
            start_generation=start_generation,
            initial_mean=initial_mean,
            initial_sigma=initial_sigma,
            initial_best_vector=initial_best_vector,
            initial_best_objective=initial_best_objective,
            on_candidate=on_candidate,
            on_candidate_error=on_candidate_error,
            diagnostics_dir=diagnostics_dir,
        )
    if engine_candidates:
        _LOG.info(
            "non-bank candidate engine eval needs STAMPEDE_CALIBRATION_ENGINE=real; "
            "using the single-path scale fit"
        )
    targets = load_nonbank_targets(path)
    dim = int(_registry_float(_PARAM_DIM, path))
    if dim != 3:
        raise RegistryError("non-bank calibration dimension must be 3")
    max_generations, pop_size, eval_steps = _search_limits(
        path=path,
        generations=generations,
        population=population,
        steps=steps,
    )
    equity_target, equity_excluded = reachable_targets(
        targets.equity, steps=eval_steps, minutes_per_day=_registry_float(_PARAM_MINUTES_PER_DAY, path)
    )
    if equity_excluded:
        _LOG.info("non-bank fit excludes unreachable targets: %s", "; ".join(equity_excluded))
    targets = replace(targets, equity=equity_target)
    scenario = build_nonbank_scenario("equity", seed, path=path)
    raw, request_fraction = measure_nonbank_observables(scenario, steps=eval_steps, path=path)
    rng = np.random.default_rng(seed)
    center = _registry_float(_PARAM_SCALE_CENTER, path)
    if initial_mean is None:
        mean = np.full(dim, center, dtype=np.float64)
    else:
        mean = np.clip(np.asarray(initial_mean, dtype=np.float64), 0.0, 1.0).copy()
    sigma = _registry_float(_PARAM_SIGMA, path) if initial_sigma is None else float(initial_sigma)
    sigma_floor = _registry_float(_PARAM_SIGMA_FLOOR, path)
    sigma_decay = _registry_float(_PARAM_SIGMA_DECAY, path)
    elite_min = int(_registry_float(_PARAM_ELITE, path))
    if initial_best_vector is None:
        best_vector = mean.copy()
    else:
        best_vector = np.clip(np.asarray(initial_best_vector, dtype=np.float64), 0.0, 1.0).copy()
    if initial_best_objective is None:
        best_obj = objective_nonbank(
            best_vector,
            scenario=scenario,
            steps=eval_steps,
            target=targets.equity,
            path=path,
            raw=raw,
            request_fraction=request_fraction,
        )
    else:
        best_obj = float(initial_best_objective)
    resolved_run_id = run_id or f"m27-nonbank-{seed}"
    for generation in range(start_generation, max_generations):
        candidates: list[tuple[float, NDArray[np.float64]]] = []
        for _ in range(pop_size):
            sample = np.clip(mean + rng.normal(0.0, sigma, size=dim), 0.0, 1.0)
            obj = objective_nonbank(
                sample,
                scenario=scenario,
                steps=eval_steps,
                target=targets.equity,
                path=path,
                raw=raw,
                request_fraction=request_fraction,
            )
            candidates.append((obj, sample))
        candidates.sort(key=lambda item: item[0])
        top = [vec for _, vec in candidates[: max(elite_min, pop_size // 2)]]
        mean = np.mean(top, axis=0)
        sigma = max(sigma_floor, sigma * sigma_decay)
        if candidates[0][0] < best_obj:
            best_obj = candidates[0][0]
            best_vector = candidates[0][1].copy()
        _record_generation(
            generation=generation,
            best_obj=best_obj,
            best_vector=best_vector,
            mean=mean,
            sigma=sigma,
            seed=seed,
            eval_steps=eval_steps,
            pop_size=pop_size,
            run_id=resolved_run_id,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            on_generation=on_generation,
            diagnostics_dir=diagnostics_dir,
        )
    return FitResult(
        seed=seed,
        vector=tuple(float(x) for x in best_vector),
        objective=float(best_obj),
        generations=max_generations,
    )


def _scale_dimension(path: Path | None) -> int:
    return int(_registry_float(_PARAM_DIM, path))


def run_nonbank_holdout(
    vector: tuple[float, ...],
    seed: int,
    *,
    path: Path | None = None,
    steps: int | None = None,
    rates_steps: int | None = None,
    crypto_steps: int | None = None,
) -> NonbankHoldoutResult:
    """Score crypto and rates with the equity vector. Does not refit.

    A 3-vector scales one unread path, which is the fake-mode stand-in. A
    behavioral vector is applied on the crypto and rates scenarios.
    ``rates_steps`` lets the E16 path run long enough for its days-to-bottom
    target (23 trading days = 8,970 minutes); otherwise that term is excluded
    and listed in ``rates_excluded`` (SVB-29). ``crypto_steps`` does the same
    for E15 so a multi-day crypto holdout is not dropped by ``reachable_targets``.
    """
    targets = load_nonbank_targets(path)
    eval_steps = steps if steps is not None else int(_registry_float(_PARAM_EVAL_MINUTES, path))
    rates_eval_steps = int(rates_steps) if rates_steps is not None else eval_steps
    crypto_eval_steps = int(crypto_steps) if crypto_steps is not None else eval_steps
    minutes_per_day = _registry_float(_PARAM_MINUTES_PER_DAY, path)
    crypto_target, crypto_excluded = reachable_targets(
        targets.crypto, steps=crypto_eval_steps, minutes_per_day=minutes_per_day
    )
    rates_target, rates_excluded = reachable_targets(
        targets.rates, steps=rates_eval_steps, minutes_per_day=minutes_per_day
    )
    targets = replace(targets, crypto=crypto_target, rates=rates_target)
    arr = np.asarray(vector, dtype=np.float64)
    weights = _weights(path)
    if arr.shape[0] != _scale_dimension(path):
        crypto, crypto_profile, crypto_seconds = _measure_holdout_path(
            build_nonbank_scenario("crypto", seed, path=path),
            steps=crypto_eval_steps,
            path=path,
            param_vector=arr,
        )
        rates, rates_profile, rates_seconds = _measure_holdout_path(
            build_nonbank_scenario("rates", seed, path=path),
            steps=rates_eval_steps,
            path=path,
            param_vector=arr,
        )
        return NonbankHoldoutResult(
            crypto_loss=_objective_from_stats(crypto, targets.crypto, weights),
            rates_loss=_objective_from_stats(rates, targets.rates, weights),
            crypto=crypto,
            rates=rates,
            crypto_excluded=crypto_excluded,
            rates_excluded=rates_excluded,
            crypto_profile=crypto_profile,
            rates_profile=rates_profile,
            crypto_seconds=crypto_seconds,
            rates_seconds=rates_seconds,
        )
    coef = _registry_float(_PARAM_SCALE_COEF, path)
    center = _registry_float(_PARAM_SCALE_CENTER, path)
    # The 3-vector scales statistics, not the path, so the profile is the unscaled engine path.
    crypto_raw, crypto_profile, crypto_seconds = _measure_holdout_path(
        build_nonbank_scenario("crypto", seed, path=path),
        steps=crypto_eval_steps,
        path=path,
        param_vector=None,
    )
    rates_raw, rates_profile, rates_seconds = _measure_holdout_path(
        build_nonbank_scenario("rates", seed, path=path),
        steps=rates_eval_steps,
        path=path,
        param_vector=None,
    )
    crypto_profile["note"] = "unscaled engine path; the 3-vector scales the statistics only"
    rates_profile["note"] = "unscaled engine path; the 3-vector scales the statistics only"
    crypto_scaled = scale_price_stats(crypto_raw, arr, coefficient=coef, center=center)
    rates_scaled = scale_price_stats(rates_raw, arr, coefficient=coef, center=center)
    return NonbankHoldoutResult(
        crypto_loss=shape_loss(crypto_scaled, targets.crypto, weights=weights),
        rates_loss=shape_loss(rates_scaled, targets.rates, weights=weights),
        crypto=crypto_scaled,
        rates=rates_scaled,
        crypto_excluded=crypto_excluded,
        rates_excluded=rates_excluded,
        crypto_profile=crypto_profile,
        rates_profile=rates_profile,
        crypto_seconds=crypto_seconds,
        rates_seconds=rates_seconds,
    )


def run_nonbank_harness(
    seed: int,
    *,
    path: Path | None = None,
    generations: int | None = None,
    population: int | None = None,
    steps: int | None = None,
    diagnostics_dir: Path | None = None,
) -> tuple[FitResult, NonbankHoldoutResult]:
    """Fit the equity shock, then hold out crypto and rates."""
    fit = fit_nonbank_parameters(
        seed,
        path=path,
        generations=generations,
        population=population,
        steps=steps,
        diagnostics_dir=diagnostics_dir,
    )
    holdout = run_nonbank_holdout(fit.vector, seed, path=path, steps=steps)
    return fit, holdout
