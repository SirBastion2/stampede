"""Market and payment array kernels for ``--gpu-full`` (ADR-055)."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
from pydantic import BaseModel

from contracts.events import (
    AttentionTick,
    DirectiveIssued,
    HeadlinePublished,
    PaymentSettled,
    PriceTick,
    RumorPosted,
    ThesisPublished,
    Tick,
    TradingHalt,
    VIPDecision,
    WithdrawalRequested,
)
from engine.gpu_full import _to_host
from mocks.demo_arc import (
    _PARAM_ATTN_PX,
    _PARAM_DRAWDOWN,
    _PARAM_STRONG,
    progress,
    rumor_strength,
    shaped,
)
from mocks.params_access import get_param
from mocks.support import (
    ASSUMED_HALT_DRAWDOWN,
    ASSUMED_HALT_MINUTES,
    OPENING_PRICE,
    levers_in,
    seed_scale,
)

_FEEDBACK_GAIN_INDEX = 12
_DISPERSION_INDEX = 13
_IMPACT_INDEX = 14
_PARAM_FEEDBACK_GAIN = "M03 feedback gain (ASSUMED)"
_PARAM_NEWS_HALF_LIFE = "M03 news half-life minutes"
_PARAM_NEWS_SCALE = "M03 news request scale"
_PARAM_PRICE_IMPACT = "M14 price impact (ASSUMED)"
_PARAM_PRICE_IMPACT_MIN = "M14 price impact (ASSUMED) min"
_PARAM_PRICE_IMPACT_MAX = "M14 price impact (ASSUMED) max"
_PARAM_IMPACT_CAP = "M14 price impact cap per minute"
_PARAM_IMPACT_NOISE = "M14 price impact noise"
_PARAM_ATTENTION_REF = "M14 attention reference level"
_PARAM_BREAKING_MEDIAN = "M06 breaking-point median (ASSUMED)"
_PARAM_BREAKING_DISP = "M06 breaking-point dispersion (ASSUMED)"

_CHANNELS: tuple[str, ...] = ("app", "wire", "ach", "branch")


def _unit_scale(component: float) -> float:
    """Map a unit-cube slot onto a positive multiplier (same shape as M06)."""
    return math.exp(0.8 * (float(component) - 0.5))


def feedback_params_from_vector(vector: Any) -> tuple[float, float, float]:
    """``(feedback_gain, breaking_point_dispersion, price_impact)`` from a vector.

    Slots 12–14 are the SVB-06 coordinates. A 12-vector leaves gain and impact
    at 0 and dispersion at 1 so existing lanes stay bit-identical.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    gain_prior = float(get_param(_PARAM_FEEDBACK_GAIN))
    disp_prior = float(get_param(_PARAM_BREAKING_DISP))
    impact_prior = float(get_param(_PARAM_PRICE_IMPACT))
    gain = (
        gain_prior * _unit_scale(float(values[_FEEDBACK_GAIN_INDEX]))
        if values.size > _FEEDBACK_GAIN_INDEX
        else 0.0
    )
    dispersion = (
        disp_prior * _unit_scale(float(values[_DISPERSION_INDEX]))
        if values.size > _DISPERSION_INDEX
        else 1.0
    )
    if values.size > _IMPACT_INDEX:
        impact = impact_prior * _unit_scale(float(values[_IMPACT_INDEX]))
        lo, hi = price_impact_bounds()
        impact = min(hi, max(lo, impact))
    else:
        impact = 0.0
    return float(gain), float(dispersion), float(impact)


def price_impact_default(path=None) -> float:
    """Registry prior for the fitted ``M14 price impact (ASSUMED)`` coordinate."""
    from params.loader import get_parameter

    return float(get_parameter(_PARAM_PRICE_IMPACT, path).value)


def price_impact_bounds(path=None) -> tuple[float, float]:
    """Registry ``[min, max]`` for the price-impact fit coordinate (SVB-25)."""
    from params.loader import get_parameter

    lo = float(get_parameter(_PARAM_PRICE_IMPACT_MIN, path).value)
    hi = float(get_parameter(_PARAM_PRICE_IMPACT_MAX, path).value)
    return lo, hi


def attention_factor(xp: Any, attention_level: Any, *, attention_ref: float | None = None) -> Any:
    """``clip(attention_level / attention_ref, 0, 1)``; ref defaults to the registry row."""
    ref = float(get_param(_PARAM_ATTENTION_REF) if attention_ref is None else attention_ref)
    if ref <= 0.0:
        ref = 1.0
    return xp.clip(attention_level / ref, 0.0, 1.0)


def apply_lane_feedback(module: Any, vector: Any) -> None:
    """Install per-lane SVB-06 params and stretch thresholds around the fixed median."""
    gain, dispersion, impact = feedback_params_from_vector(vector)
    module.feedback_gain = gain
    module.breaking_point_dispersion = dispersion
    module.price_impact = impact
    module.calibration_vector = np.asarray(vector, dtype=np.float64)
    thresh = getattr(module, "base_threshold", None)
    if thresh is None or int(getattr(thresh, "size", 0)) == 0:
        return
    raw = getattr(module, "_threshold_raw", None)
    if raw is None:
        raw = np.asarray(thresh, dtype=np.float64).copy()
        module._threshold_raw = raw
    median = float(get_param(_PARAM_BREAKING_MEDIAN))
    module.base_threshold = median + float(dispersion) * (raw - median)


def price_impact_move(
    xp: Any,
    price: Any,
    cum_share: Any,
    impact: Any,
    dt: float,
    *,
    open_price: Any = None,
    attention_level: Any = 1.0,
    attention_ref: float | None = None,
    noise: Any = 0.0,
    cap: float | None = None,
) -> Any:
    """Step toward ``impact * cum_share * attention_factor``, clipped to ``±cap * dt``.

    ``current_drop = -log(price / open_price)``. Recovery (negative step) is
    allowed at the same cap. Noise is added to the remaining gap before the clip.
    """
    bound = float(get_param(_PARAM_IMPACT_CAP) if cap is None else cap) * float(dt)
    open_px = float(OPENING_PRICE) if open_price is None else open_price
    target = impact * cum_share * attention_factor(
        xp, attention_level, attention_ref=attention_ref
    )
    current_drop = -xp.log(xp.maximum(price, 1e-30) / xp.maximum(open_px, 1e-30))
    raw = target - current_drop + noise
    return xp.clip(raw, -bound, bound)


def apply_price_impact(
    xp: Any,
    price: Any,
    cum_share: Any,
    impact: Any,
    dt: float,
    *,
    open_price: Any = None,
    attention_level: Any = 1.0,
    attention_ref: float | None = None,
    noise: Any = 0.0,
    cap: float | None = None,
) -> Any:
    """``price *= exp(-step)`` with ``step`` from ``price_impact_move``."""
    step = price_impact_move(
        xp,
        price,
        cum_share,
        impact,
        dt,
        open_price=open_price,
        attention_level=attention_level,
        attention_ref=attention_ref,
        noise=noise,
        cap=cap,
    )
    return price * xp.exp(-step)


def step_news(
    xp: Any,
    news: Any,
    d_requested_share: Any,
    feedback_gain: Any,
    dt: float,
    *,
    half_life: float | None = None,
    request_scale: float | None = None,
) -> Any:
    """Bank-specific news: half-life decay plus withdrawal impulse."""
    life = float(get_param(_PARAM_NEWS_HALF_LIFE) if half_life is None else half_life)
    scale = float(get_param(_PARAM_NEWS_SCALE) if request_scale is None else request_scale)
    decay = 0.5 ** (float(dt) / life)
    impulse = feedback_gain * xp.maximum(scale * d_requested_share, 0.0)
    return news * decay + impulse


def compose_attention(xp: Any, rumor_attention: Any, news: Any) -> Any:
    """Saturating attention ``tanh(rumor_attention + news)``."""
    return xp.tanh(rumor_attention + news)


def apply_composed_attention(engines: list[Any], t: float) -> None:
    """Overwrite consumer attention with ``tanh(rumor + news)``."""
    for engine in engines:
        m03 = engine.get("M03")
        m06 = engine.get("M06")
        if m03 is None:
            continue
        news = float(getattr(m03, "_gpu_news", 0.0) or 0.0)
        attention = getattr(m03, "attention", None)
        rumor = 0.0
        if attention is not None and getattr(attention, "size", 0):
            rumor = float(np.mean(attention))
        level = float(np.tanh(rumor + news))
        if m06 is not None:
            m06.attention_level = level
            m06.news_level = news
        market = engine.get("M14")
        if market is not None:
            market.attention = level
        scenario = getattr(m03, "scenario", None)
        seed = int(getattr(scenario, "seed", 0) or 0)
        bank_id = None
        if m06 is not None:
            try:
                bank_id = m06.bank_id
            except RuntimeError:
                bank_id = getattr(m03, "bank_id", None)
        engine.bus.publish(
            AttentionTick(
                t=float(t),
                seed=seed,
                bank_id=bank_id,
                cause_id="svb06-news",
                level=level,
                delta_4h=0.0,
            )
        )


def price_quote_batch(
    xp: Any, t: float, strength: Any, attention: Any, *, scripted: bool = True
) -> Any:
    """Vectorized ``mocks.demo_arc.price_quote``."""
    strong = float(get_param(_PARAM_STRONG))
    scale = 1.0 if strong <= 0.0 else 1.0 / strong
    depth = xp.where(strength <= 0.0, 0.0, xp.clip(strength * scale, 0.0, 1.0))
    gap = (
        float(get_param(_PARAM_DRAWDOWN)) * float(shaped(progress(t))) * depth
        if scripted
        else 0.0
    )
    gap = gap + float(get_param(_PARAM_ATTN_PX)) * xp.clip(attention, 0.0, 1.0)
    gap = xp.clip(gap, 0.0, 0.95)
    return xp.maximum(0.5, float(OPENING_PRICE) * (1.0 - gap))


def settle_payments_batch(
    xp: Any,
    incoming: Any,
    *,
    app_used: Any,
    wire_q: Any,
    ach_q: Any,
    branch_q: Any,
    app_limit: float,
    wire_cap: float,
    ach_cap: float,
    branch_cap: float,
    wire_window: bool,
    business_hours: bool,
    ach_open: bool,
) -> tuple[Any, Any, Any, Any, Any]:
    """Settle one aggregated request per channel. Returns settled ``(C, 4)`` plus queues."""
    zeros = xp.zeros_like(incoming[:, 0])
    app_now = xp.minimum(incoming[:, 0], xp.maximum(0.0, float(app_limit) - app_used))
    leftover = incoming[:, 0] - app_now
    app_used = app_used + app_now
    wire_q = wire_q + incoming[:, 1] + leftover
    ach_q = ach_q + incoming[:, 2]
    branch_q = branch_q + incoming[:, 3]
    wire_pay = xp.minimum(wire_q, float(wire_cap)) if wire_window else zeros
    wire_q = wire_q - wire_pay
    branch_pay = xp.minimum(branch_q, float(branch_cap)) if business_hours else zeros
    branch_q = branch_q - branch_pay
    ach_pay = xp.minimum(ach_q, float(ach_cap)) if ach_open else zeros
    ach_q = ach_q - ach_pay
    settled = xp.stack([app_now, wire_pay, ach_pay, branch_pay], axis=1)
    return settled, app_used, wire_q, ach_q, branch_q


def step_market_group(engines: list[Any], t: float, xp: Any, dt: float = 1.0) -> None:
    """Advance fake M14 quotes as one array; real books keep ``step``.

    ``dt`` scales the price-impact cap and the news half-life. Other market
    kernels still treat a tick as one minute when this is omitted.
    """
    from mocks.l3_markets import MarketFake

    fakes: list[tuple[Any, Any]] = []
    for engine in engines:
        module = engine.get("M14")
        if isinstance(module, MarketFake):
            fakes.append((engine, module))
        elif module is not None:
            if "dt" in inspect.signature(module.step).parameters:
                events = module.step(float(t), dt=float(dt))
            else:
                events = module.step(float(t))
            for event in events:
                engine.bus.publish(event)
            _step_book_news(engine, module, t, dt)
    if not fakes:
        return
    attentions: list[float] = []
    strengths: list[float] = []
    tick_shares: list[float] = []
    cum_shares: list[float] = []
    opens: list[float] = []
    impacts: list[float] = []
    gains: list[float] = []
    news_vals: list[float] = []
    factors: list[float] = []
    halted: list[bool] = []
    for engine, module in fakes:
        _absorb_market(module)
        attentions.append(float(module.attention))
        strengths.append(_cached_rumor_strength(engine, module))
        m06 = engine.get("M06")
        m03 = engine.get("M03")
        tick_share = _requested_share(engine, t, module, m06)
        cum_share = float(getattr(module, "_gpu_cum_share", 0.0)) + tick_share
        module._gpu_cum_share = cum_share
        tick_shares.append(tick_share)
        cum_shares.append(cum_share)
        if getattr(module, "_gpu_open_price", None) is None:
            module._gpu_open_price = float(module.price or OPENING_PRICE)
        opens.append(float(module._gpu_open_price))
        impacts.append(float(getattr(m06, "price_impact", 0.0) or 0.0) if m06 is not None else 0.0)
        gains.append(float(getattr(m06, "feedback_gain", 0.0) or 0.0) if m06 is not None else 0.0)
        news_vals.append(float(getattr(m03, "_gpu_news", 0.0) or 0.0) if m03 is not None else 0.0)
        factors.append(float(getattr(module, "_gpu_impact_factor", 1.0) or 1.0))
        halted.append(bool(getattr(module, "halted", False)))
    from engine.gpu_sim import calibration_uses_scripted_drawdown

    quoted = price_quote_batch(
        xp,
        t,
        xp.asarray(strengths),
        xp.asarray(attentions),
        scripted=calibration_uses_scripted_drawdown(),
    )
    tick_x = xp.asarray(tick_shares, dtype=xp.float64)
    share_x = xp.asarray(cum_shares, dtype=xp.float64)
    open_x = xp.asarray(opens, dtype=xp.float64)
    attn_x = xp.asarray(attentions, dtype=xp.float64)
    impact_x = xp.asarray(impacts, dtype=xp.float64)
    gain_x = xp.asarray(gains, dtype=xp.float64)
    news_x = xp.asarray(news_vals, dtype=xp.float64)
    factor_x = xp.asarray(factors, dtype=xp.float64)
    halt_x = xp.asarray(halted, dtype=xp.float64)
    noise = float(get_param(_PARAM_IMPACT_NOISE))
    current = quoted * factor_x
    move = price_impact_move(
        xp,
        current,
        share_x,
        impact_x,
        float(dt),
        open_price=open_x,
        attention_level=attn_x,
        noise=noise,
    )
    live = 1.0 - halt_x
    new_price = current * xp.exp(-move * live)
    new_factor = new_price / xp.maximum(quoted, 1e-30)
    prices = _to_host(new_price)
    news_host = _to_host(step_news(xp, news_x, tick_x, gain_x, float(dt)))
    factor_host = _to_host(new_factor)
    for (engine, module), price, news, factor in zip(
        fakes, prices, news_host, factor_host, strict=True
    ):
        module._gpu_impact_factor = float(factor)
        m03 = engine.get("M03")
        if m03 is not None:
            m03._gpu_news = float(news)
        _publish_market_tick(engine, module, t, float(price))


def _step_book_news(engine: Any, market: Any, t: float, dt: float) -> None:
    """SVB-65: news for a real M14 book (the live engine), as ``step_market_group`` does for fakes."""
    m03 = engine.get("M03")
    m06 = engine.get("M06")
    if m03 is None or m06 is None:
        return
    gain = float(getattr(m06, "feedback_gain", 0.0) or 0.0)
    news = float(getattr(m03, "_gpu_news", 0.0) or 0.0)
    tick_share = _requested_share(engine, t, market, m06)
    if gain == 0.0 and news == 0.0:
        return
    m03._gpu_news = float(step_news(np, news, tick_share, gain, float(dt)))


def _requested_share(engine: Any, t: float, market: Any, m06: Any) -> float:
    """This bank's requested outflow this tick, as a share of the opening book.

    Reads only events published since the previous call (SVB-65): this tick's requests all
    come after it, so the sum and its order match a scan of the whole log.
    """
    bank_id = getattr(market, "bank_id", None)
    total = 0.0
    log = engine.bus.log
    start = int(getattr(market, "_gpu_request_cursor", 0) or 0)
    market._gpu_request_cursor = len(log)
    for event in log[start:]:
        if not isinstance(event, WithdrawalRequested):
            continue
        if float(event.t) != float(t):
            continue
        event_bank = getattr(event, "bank_id", None)
        if event_bank not in (None, bank_id):
            continue
        total += float(event.amount)
    book = 0.0
    if m06 is not None:
        book = float(getattr(m06, "_book_deposits", 0.0) or 0.0)
        if book <= 0.0:
            balance = getattr(m06, "balance", None)
            if balance is not None and int(getattr(balance, "size", 0)) > 0:
                book = float(np.sum(balance))
                m06._book_deposits = book
    if book <= 0.0:
        book = float(get_param("SVB-like preset: deposits"))
    return total / book if book > 0.0 else 0.0


def step_payments_group(engines: list[Any], t: float, xp: Any) -> None:
    """Settle aggregated channel requests on the array device."""
    from engine.bank.payments import calendar_flags

    modules = [engine.get("M07") for engine in engines]
    if any(module is None for module in modules):
        return
    first = modules[0]
    incoming = np.zeros((len(modules), len(_CHANNELS)), dtype=np.float64)
    tick: Tick | None = None
    for index, module in enumerate(modules):
        new_events = module.inbox[module._cursor :]
        module._cursor = len(module.inbox)
        for event in new_events:
            if isinstance(event, Tick):
                tick = event
            elif isinstance(event, WithdrawalRequested) and event.amount > 0.0:
                name = event.channel if event.channel in _CHANNELS else "wire"
                incoming[index, _CHANNELS.index(name)] += event.amount
            elif (
                isinstance(event, VIPDecision)
                and event.action == "withdraw"
                and event.amount > 0.0
            ):
                incoming[index, _CHANNELS.index(first.limits.vip_default_channel)] += event.amount
    flags = calendar_flags(first._sim_minute(t, tick), first.calendar)
    state = _payment_state(first, xp, len(modules), flags.day_index)
    settled, app_used, wire_q, ach_q, branch_q = settle_payments_batch(
        xp,
        xp.asarray(incoming, dtype=xp.float64),
        app_used=state["app_used"],
        wire_q=state["wire"],
        ach_q=state["ach"],
        branch_q=state["branch"],
        app_limit=float(first.limits.app_daily_limit),
        wire_cap=float(first.limits.wire_capacity_per_tick),
        ach_cap=float(first.limits.ach_batch_capacity),
        branch_cap=float(first.limits.branch_capacity_per_tick),
        wire_window=bool(flags.wire_window),
        business_hours=bool(flags.business_hours),
        ach_open=bool(
            not flags.weekend
            and flags.minute_of_day >= first.limits.ach_batch_close_minute
        ),
    )
    state["app_used"] = app_used
    state["wire"] = wire_q
    state["ach"] = ach_q
    state["branch"] = branch_q
    host = _to_host(settled)
    queued = _to_host(wire_q)
    for engine, module, row, left in zip(engines, modules, host, queued, strict=True):
        for channel, amount in zip(_CHANNELS, row, strict=True):
            if float(amount) <= 0.0:
                continue
            engine.bus.publish(
                PaymentSettled(
                    t=t,
                    seed=module.seed,
                    bank_id=module.bank_id,
                    cause_id=module._cause(channel),
                    amount=float(amount),
                    channel=channel,
                    queued_remaining=int(max(0.0, float(left))),
                )
            )


def _payment_state(first: Any, xp: Any, width: int, day_index: int) -> dict[str, Any]:
    state = getattr(first, "_gpu_full_state", None)
    if state is None:
        state = {
            "app_used": xp.zeros(width, dtype=xp.float64),
            "wire": xp.zeros(width, dtype=xp.float64),
            "ach": xp.zeros(width, dtype=xp.float64),
            "branch": xp.zeros(width, dtype=xp.float64),
            "day": day_index,
        }
        first._gpu_full_state = state
    if day_index != state["day"]:
        state["app_used"] = xp.zeros_like(state["app_used"])
        state["day"] = day_index
    return state


def _cached_rumor_strength(engine: Any, module: Any) -> float:
    value = getattr(module, "_gpu_rumor_strength", None)
    if value is None:
        value = float(rumor_strength(engine.bus.log))
        module._gpu_rumor_strength = value
    return float(value)


def _absorb_market(module: Any) -> None:
    new_events = module.inbox[module._cursor :]
    module._cursor = len(module.inbox)
    for event in new_events:
        if isinstance(event, AttentionTick):
            module.attention = event.level
        elif hasattr(event, "cet1_ratio"):
            module.cet1 = event.cet1_ratio
        elif isinstance(event, HeadlinePublished):
            module.headline_pressure = min(1.0, event.reach / 50_000.0)
        elif isinstance(event, ThesisPublished):
            module.short_interest += 0.05 * seed_scale(module.seed, "short")
        elif isinstance(event, RumorPosted):
            prev = float(getattr(module, "_gpu_rumor_strength", 0.0) or 0.0)
            cred = max(0.0, min(1.0, float(event.credibility)))
            module._gpu_rumor_strength = max(prev, cred)
        elif isinstance(event, DirectiveIssued) and event.action == "withdraw":
            if float(getattr(module, "_gpu_rumor_strength", 0.0) or 0.0) <= 0.0:
                module._gpu_rumor_strength = float(rumor_strength([event]))


def _publish_market_tick(engine: Any, module: Any, t: float, price: float) -> None:
    from mocks.l3_markets import _drawdown

    module.price = price
    volume = 1_000.0 + 80_000.0 * max(module.attention, 0.05) * seed_scale(module.seed, "volume")
    module.spread = 0.02 * (1.0 + 6.0 * module.attention + 4.0 * module.short_interest)
    module.history.append(module.price)
    events: list[BaseModel] = [
        PriceTick(
            t=t,
            seed=module.seed,
            bank_id=module.bank_id,
            cause_id=module.cause("px"),
            price=module.price,
            volume=volume,
        )
    ]
    if not module.halted and _drawdown(module.history) <= -ASSUMED_HALT_DRAWDOWN:
        module.halted = True
        events.append(
            TradingHalt(
                t=t,
                seed=module.seed,
                bank_id=module.bank_id,
                cause_id=module.cause("halt"),
                reason="volatility",
                duration=ASSUMED_HALT_MINUTES,
            )
        )
    elif not module.halted and "suspend_trading" in levers_in(module.log()):
        module.halted = True
        events.append(
            TradingHalt(
                t=t,
                seed=module.seed,
                bank_id=module.bank_id,
                cause_id=module.cause("halt-policy"),
                reason="policy",
                duration=ASSUMED_HALT_MINUTES,
            )
        )
    module.published.extend(events)
    for event in events:
        engine.bus.publish(event)
