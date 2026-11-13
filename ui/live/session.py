"""Single-process live engine session (one bus, crowd + market frames)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agents.guardrails import moderate_rumor
from agents.rumor_inbox import (
    RumorInbox,
    rate_limit_reason,
    rumor_too_long,
    too_long_reason,
)
from agents.rumor_input import classify_rumor_market
from agents.rumor_judge import RumorJudgment
from contracts.schemas import CrowdFrame
from engine import build_engine
from engine.registry import Engine
from mocks.l6_crowd_ui import CrowdFake
from ui.crowd.frames import build_crowd_frame
from ui.crowd.layout import CrowdLayout, build_layout
from engine.bank.ledger import svb_like_opening_book
from engine.fitted_feedback import install_persisted_feedback, step_module
from engine.live_overrides import live_flags, registry_flag
from engine.modes import ModuleMode
from ui.live.config import default_module_mode
from ui.live.engine_crowd import (
    apply_engine_crowd,
    ledger_cash,
    live_engine_modes,
    read_engine_crowd,
)
from ui.live.pacing import (
    PacingClock,
    base_sim_minutes_per_real_second,
    crisis_sim_minutes,
    normalize_speed,
    playback_head_start_seconds,
)
from ui.live.labels import run_label

TRIGGER_KINDS = ("requested", "cash")  # UI-3d: share of deposits requested >= level; cash left (of opening) <= level
from ui.live.playback import BufferedFrame, PlaybackBuffer
from ui.live.presets import INSURED_CHIP, LIVE_PRESETS, RESPONSES, STEPS, PresetPost, response_post
from ui.live.scenario import live_scenario
from ui.market.snapshot import build_market_frame


def step_engine(engine: Engine, t: float) -> None:
    """Advance one simulated minute without ``engine.run`` (for live streaming)."""
    for module_id in engine.order:
        step_module(engine, module_id, t)


# Live-only opt-ins: the live row (M29) switches on the engine row for the live engine only.
_LIVE_OPT_INS: dict[str, tuple[str, ...]] = {
    "M29 live depositors ignore stock price": ("M06 depositors ignore stock price",),
    "M29 live cure rumor": ("M03 cure rumor enabled", "M06 cure rumor worry shift enabled"),
    "M29 live attention ceiling by credibility": ("M06 attention ceiling by credibility enabled",),
    "M29 live emergency line on BTFP terms": ("M12 emergency line on BTFP terms",),  # CM-2
    "M29 live regulator closes on what the bank can pay": ("M20 closes on what the bank can pay",),  # UF-1
}


def _live_engine_flags() -> set[str]:
    """SVB-58/59: engine rows the live real-mode engine turns on (calibration never does)."""
    rows: set[str] = set()
    for live_row, engine_rows in _LIVE_OPT_INS.items():
        if registry_flag(live_row):
            rows.update(engine_rows)
    return rows


def _apply_fit_noise(engine: Any) -> None:
    """SVB-61: run the live M06 at the SVB fit's behavioral noise temperature.

    The fit uses ``Calibration behavioral noise temperature`` (soft decide-once:
    depositors convert at rate h). M06's default softness is 0, the hard branch,
    where everyone whose gates pass requests at once: not the calibrated model,
    and the reason the live run drained the bank within hours (diag-08).
    """
    from params.loader import RegistryError, get_parameter

    try:
        temperature = float(get_parameter("Calibration behavioral noise temperature").value)
    except (RegistryError, ValueError):
        return
    module = engine.get("M06")
    if module is not None and hasattr(module, "softness"):
        module.softness = temperature


def _hold_closure(engine: Any) -> None:
    """SVB-64: the live regulator closes no earlier than ``M29 live earliest closure minute``."""
    from params.loader import RegistryError, get_parameter

    try:
        minute = float(get_parameter("M29 live earliest closure minute").value)
    except (RegistryError, ValueError):
        return
    regulator = engine.get("M20")
    if regulator is not None and hasattr(regulator, "earliest_closure_minute"):
        regulator.earliest_closure_minute = minute


def _live_opening_book():
    """SVB-56: the live real-mode bank opens with the SVB-like book when the live row says so."""
    from contextlib import nullcontext

    from params.loader import RegistryError, get_parameter

    try:
        on = get_parameter("M29 live SVB-like opening book").value.strip().lower() in {"true", "1", "yes", "on"}
    except RegistryError:
        on = False
    return svb_like_opening_book() if on else nullcontext()


@dataclass
class RumorSubmitResult:
    ok: bool
    reasons: tuple[str, ...] = ()
    rumor_id: str | None = None
    read_as: str | None = None
    affects_simulation: bool = False
    effect_size: float = 0.0
    credibility: float | None = None
    credibility_breakdown: dict[str, Any] | None = None
    judgment: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok}
        if self.reasons:
            out["reasons"] = list(self.reasons)
        if self.rumor_id is not None:
            out["rumor_id"] = self.rumor_id
        if self.read_as is not None:
            out["read_as"] = self.read_as
        out["affects_simulation"] = self.affects_simulation
        out["effect_size"] = self.effect_size
        if self.credibility is not None:
            out["credibility"] = self.credibility
        if self.credibility_breakdown is not None:
            out["credibility_breakdown"] = self.credibility_breakdown
        if self.judgment is not None:
            out["judgment"] = self.judgment
        return out


@dataclass
class LiveSession:
    seed: int = 1
    paused: bool = True
    tick: float = 0.0
    speed: float = 1.0
    pacing: PacingClock = field(default_factory=PacingClock)
    engine: Engine = field(init=False)
    _layout: CrowdLayout | None = field(default=None, init=False)
    _rumor_seq: int = field(default=0, init=False)
    _inbox: RumorInbox = field(default_factory=RumorInbox, init=False)
    follows_engine: bool = field(default=False, init=False)
    _opening_cash: float | None = field(default=None, init=False)
    # SVB-63: live price paced to SVB's day-one drop (real mode, live row).
    price_paced: bool = field(default=False, init=False)
    _requested_usd: float = field(default=0.0, init=False)
    _requested_cursor: int = field(default=0, init=False)
    _paced_prices: list[tuple[float, float]] = field(default_factory=list, init=False)
    # P-A: preset posts by minute, the guarantee chip, and the trading halt at the price floor.
    preset: str | None = field(default=None, init=False)
    _scheduled: list[PresetPost] = field(default_factory=list, init=False)
    _post_seq: int = field(default=0, init=False)
    guarantee_minute: float | None = field(default=None, init=False)
    halt_minute: float | None = field(default=None, init=False)
    halt_at_floor: bool = field(default=False, init=False)
    # PA-C: what the run is labelled (fitted only for the 2023 replay).
    _responses: list[str] = field(default_factory=list, init=False)
    _free_text: bool = field(default=False, init=False)
    response_minutes: dict[str, float] = field(default_factory=dict, init=False)  # PA-A
    # UI-3d: trigger rules, a response that fires on a condition (kind, level), checked every minute
    _triggers: list[tuple[str, str, float]] = field(default_factory=list, init=False)
    trigger_minutes: dict[str, float] = field(default_factory=dict, init=False)
    trigger_requested: dict[str, float] = field(default_factory=dict, init=False)  # UI-3e: the share requested when it fired
    # UI-1: what the screens show, scanned from each step's new events (never the whole log).
    _timeline: list[dict[str, Any]] = field(default_factory=list, init=False)
    _ui_cursor: int = field(default=0, init=False)
    _ui_requested: float = field(default=0.0, init=False)
    _ui_paid: float = field(default=0.0, init=False)
    _ui_closure: float | None = field(default=None, init=False)
    _ui_cash_out: float | None = field(default=None, init=False)
    _ui_halt_logged: bool = field(default=False, init=False)
    # UI-2: the vault's depositors (a fixed sample of real M06 depositors) and the bank's borrowing.
    _ui_sample: Any = field(default=None, init=False)
    _ui_borrowed: float = field(default=0.0, init=False)
    _ui_emergency: float = field(default=0.0, init=False)
    # P-A2: buffered playback. Off until the server starts the producer (start_producer).
    buffered: bool = field(default=False, init=False)
    _buffer: PlaybackBuffer | None = field(default=None, init=False)
    _buffering: bool = field(default=False, init=False)
    _engine_lock: Any = field(default_factory=threading.RLock, init=False)
    _producer: threading.Thread | None = field(default=None, init=False)
    _stop_producer: threading.Event = field(default_factory=threading.Event, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        with self._engine_lock:
            self._reset_unlocked()
            self.buffered = self.follows_engine and registry_flag("M29 live buffered playback")
            head = playback_head_start_seconds() * base_sim_minutes_per_real_second()
            self._buffer = PlaybackBuffer(head, head + 2.0)
            if self._buffering and not self.buffered:
                self._buffering = False
            if self._buffering:
                self._buffer.push(self._capture())

    # ---- P-A2: producer and playback -------------------------------------------------

    @property
    def buffering(self) -> bool:
        return self._buffering

    @property
    def playback_tick(self) -> float:
        return self._buffer.position if self._buffering and self._buffer is not None else self.tick

    def start_producer(self, *, thread: bool = True) -> bool:
        """Buffer frames from here on (real mode with ``M29 live buffered playback``); start the thread."""
        with self._engine_lock:
            if not self.buffered or self._buffer is None:
                return False
            if not self._buffering:
                self._buffering = True
                self._buffer.clear(self.tick)
                self._buffer.push(self._capture())
        if thread and (self._producer is None or not self._producer.is_alive()):
            self._stop_producer.clear()
            self._producer = threading.Thread(target=self._produce_loop, name="stampede-producer", daemon=True)
            self._producer.start()
        return True

    def stop_producer(self) -> None:
        self._stop_producer.set()
        if self._producer is not None:
            self._producer.join(timeout=5.0)
        self._producer = None
        with self._engine_lock:
            self._buffering = False

    def produce_once(self) -> bool:
        """Step the engine one minute into the buffer if the run is playing and the lead allows."""
        with self._engine_lock:
            buffer = self._buffer
            if not self._buffering or buffer is None or self.paused:
                return False
            if self.tick >= crisis_sim_minutes() or buffer.lead(self.tick) >= buffer.max_lead_ticks:
                return False
            self.step_once()
            buffer.push(self._capture())
            return True

    def _produce_loop(self) -> None:
        log = logging.getLogger(__name__)
        while not self._stop_producer.is_set():
            try:
                stepped = self.produce_once()
            except Exception:  # keep serving: the screens hold the newest frame
                log.exception("live producer step failed")
                self._stop_producer.wait(0.5)
                continue
            if not stepped:
                self._stop_producer.wait(0.005)

    def _capture(self) -> BufferedFrame:
        return BufferedFrame(self.tick, self._engine_crowd_frame(), self._engine_market_frame(), self._ui_info())

    # ---- UI-1: the screen's state ------------------------------------------------------

    _LEVER_NAMES = {"guarantee_deposits": "Deposit guarantee", "emergency_lending": "Emergency credit line",
                    "cap_withdrawals": "Withdrawal limit"}

    @classmethod
    def _lever_words(cls, lever: str) -> tuple[str, str, str]:
        """VF-3: (label, who acts, note) for a lever, from the steps the plans use, so the bank's own tools
        (pledged bonds, big-bank deposits, statement, rate rise) never reach the screen by their code names."""
        from ui.live.presets import RESPONSES, STEPS

        step = next((r for r in STEPS.values() if r.lever == lever), None)
        regulator = {r.lever for r in RESPONSES.values()}
        label = cls._LEVER_NAMES.get(lever) or (step.label if step else lever.replace("_", " ").capitalize())
        return label, ("regulator" if lever in regulator or step is None else "bank"), (step.note if step else "")

    def _ui_scan(self) -> None:
        """Fold this step's new events into the UI totals and timeline."""
        from contracts.events import (
            BankClosed, LiquidityDrawn, PaymentSettled, PolicyEnacted, RumorPosted, WithdrawalRequested,
        )

        log = self.engine.bus.log
        for event in log[self._ui_cursor:]:
            if isinstance(event, WithdrawalRequested):
                self._ui_requested += float(event.amount)
            elif isinstance(event, PaymentSettled):
                self._ui_paid += float(event.amount)
            elif isinstance(event, LiquidityDrawn):
                self._ui_borrowed += float(event.amount)
                if event.source == "emergency":  # the credit line's pool (M12)
                    self._ui_emergency += float(event.amount)
            elif isinstance(event, RumorPosted) and str(event.cause_id or "").startswith("M29"):
                self._timeline.append({"t": float(event.t), "type": "rumor", "text": event.claim,
                                       "poster": event.source_node})
            elif isinstance(event, PolicyEnacted):
                label, who, note = self._lever_words(event.lever)
                self._timeline.append({"t": float(event.t), "type": "response", "text": label, "lever": event.lever,
                                       "by": who, "note": note})
            elif isinstance(event, BankClosed) and self._ui_closure is None:
                self._ui_closure = float(event.t)
                self._timeline.append({"t": float(event.t), "type": "closed", "text": "Regulators closed the bank",
                                       "reason": getattr(event, "reason", "")})
        self._ui_cursor = len(log)
        if self.halt_minute is not None and not self._ui_halt_logged:
            self._ui_halt_logged = True
            self._timeline.append({"t": self.halt_minute, "type": "halt",
                                   "text": "Trading halted: the share price hit its floor"})
        crowd = self.engine.get("M29")
        if self._ui_cash_out is None and isinstance(crowd, CrowdFake) and crowd.cash_gauge <= 0.0:
            self._ui_cash_out = self.tick
            self._timeline.append({"t": self.tick, "type": "cash_out", "text": "The bank ran out of cash"})

    # UI-2 sample: the 400 biggest depositors with uninsured money, 6,600 more of them and 3,000 fully insured.
    SAMPLE_TOP, SAMPLE_REST, SAMPLE_INSURED = 400, 6_600, 3_000

    def _vault_sample(self) -> Any:
        """Fixed for the run, from the seed; None until M06 has filled its arrays (its first step)."""
        import numpy as np

        if self._ui_sample is not None:
            return self._ui_sample
        m = self.engine.get("M06") if self.follows_engine else None
        unins_all = getattr(m, "initial_uninsured", None)
        if unins_all is None or np.asarray(unins_all).size == 0:
            return None
        deposits = np.asarray(m.insured, dtype=float) + np.asarray(unins_all, dtype=float)
        rng = np.random.default_rng(1000 + int(self.seed))
        unins = np.flatnonzero(np.asarray(unins_all) > 0)
        insured = np.flatnonzero(np.asarray(unins_all) <= 0)
        top = unins[np.argsort(-deposits[unins], kind="stable")[: self.SAMPLE_TOP]]
        rest_pool = np.setdiff1d(unins, top)
        rest = rng.choice(rest_pool, min(self.SAMPLE_REST, rest_pool.size), replace=False) if rest_pool.size else rest_pool
        ins = rng.choice(insured, min(self.SAMPLE_INSURED, insured.size), replace=False) if insured.size else insured
        idx = np.concatenate([top, np.sort(rest), np.sort(ins)]).astype(int)
        self._ui_sample = {"idx": idx, "n_unins": int(top.size + rest.size), "deposits": deposits[idx],
                           "uninsured_total": int(unins.size), "insured_total": int(insured.size),
                           "top_share": float(deposits[top].sum() / max(1.0, deposits.sum()))}
        return self._ui_sample

    def vault_layout(self) -> dict[str, Any]:
        """UI-2: who the vault's dots are. Each dot is one real engine depositor."""
        sample = self._vault_sample()
        if sample is None:
            return {"ready": False}
        return {"ready": True, "n": int(sample["idx"].size), "n_unins": sample["n_unins"],
                "deposits": [round(float(d)) for d in sample["deposits"]],
                "uninsured_total": sample["uninsured_total"], "insured_total": sample["insured_total"],
                "top_share": sample["top_share"], "top": self.SAMPLE_TOP}

    def _vault_fracs(self) -> str | None:
        """Each sampled depositor's share of its own deposits requested, 0-255, base64."""
        import base64

        import numpy as np

        sample = self._vault_sample()
        if sample is None:
            return None
        m = self.engine["M06"]
        idx = sample["idx"]
        asked = np.asarray(getattr(m, "mobile_share", 1.0)) * np.asarray(m.initial_uninsured)[idx] * np.asarray(m.decided)[idx]
        frac = np.clip(np.round(255.0 * asked / np.maximum(sample["deposits"], 1.0)), 0, 255).astype(np.uint8)
        return base64.b64encode(frac.tobytes()).decode("ascii")

    def _ui_info(self) -> dict[str, Any]:
        crowd = self.engine.get("M29")
        counts = (int(crowd.withdrew), int(crowd.wavering), int(crowd.calm)) if isinstance(crowd, CrowdFake) else (0, 0, 0)
        cash = ledger_cash(self.engine) if self.follows_engine else None
        return {"requested_usd": self._ui_requested, "paid_usd": self._ui_paid,
                "cash_usd": cash, "cash_fraction": float(getattr(crowd, "cash_gauge", 1.0)),
                "withdrew": counts[0], "wavering": counts[1], "calm": counts[2],
                "closure_minute": self._ui_closure, "cash_out_minute": self._ui_cash_out,
                "halt_minute": self.halt_minute, "timeline_len": len(self._timeline),
                "borrowed_usd": self._ui_borrowed, "emergency_usd": self._ui_emergency,
                "fracs": self._vault_fracs() if self.follows_engine else None}

    def ui_state(self, price_since: int = 0, event_since: int = 0) -> dict[str, Any]:
        """UI-1: one JSON snapshot of what the screens show, at the playback minute while buffering."""
        from engine.bank.ledger import registry_deposits_total

        current = self._buffer.current() if self._buffering and self._buffer is not None else None
        if current is not None and current.info is not None:
            tick, info, market = current.tick, current.info, current.market
        else:
            with self._engine_lock:
                tick, info, market = self.tick, self._ui_info(), self._engine_market_frame()
        prices = [[float(t), float(p)] for t, p in market.prices]
        horizon = float(crisis_sim_minutes())
        book = self.engine.get("M14")
        return {
            "tick": tick, "engine_tick": self.tick, "horizon": horizon, "paused": self.paused,
            "finished": tick >= horizon or info["closure_minute"] is not None and tick >= info["closure_minute"],
            "buffered": self._buffering, "speed": self.speed, "seed": self.seed, "follows_engine": self.follows_engine,
            "run": self.run_label(),
            "crowd": {k: info[k] for k in ("withdrew", "wavering", "calm", "cash_fraction")},
            "money": {"requested_usd": info["requested_usd"], "paid_usd": info["paid_usd"], "cash_usd": info["cash_usd"],
                      "deposits_usd": registry_deposits_total(), "opening_cash_usd": self._opening_cash,
                      "borrowed_usd": info.get("borrowed_usd", 0.0), "emergency_usd": info.get("emergency_usd", 0.0)},
            "fracs": info.get("fracs"),
            "opening_price": float(getattr(getattr(book, "params", None), "opening_price", 100.0)),
            "price_count": len(prices), "prices": prices[max(0, int(price_since)):],
            "halt": market.halt, "statements": [r.stmt_id for r in market.statements],
            "closure_minute": info["closure_minute"], "cash_out_minute": info["cash_out_minute"],
            "halt_minute": info["halt_minute"],
            "timeline_count": info["timeline_len"],
            "timeline": self._timeline[max(0, int(event_since)):info["timeline_len"]],
        }

    def _reset_unlocked(self) -> None:
        self.tick = 0.0
        self.paused = True
        self.speed = 1.0
        self.pacing.reset()
        self.pacing.set_speed(1.0)
        self._rumor_seq = 0
        self._inbox.reset()
        self._layout = None
        self._opening_cash = None
        self.price_paced = False
        self._requested_usd = 0.0
        self._requested_cursor = 0
        self._paced_prices = []
        self.preset = None
        self._responses = []
        self._free_text = False
        self.response_minutes = {}
        self._triggers = []
        self.trigger_minutes = {}
        self.trigger_requested = {}
        self._timeline = []
        self._ui_cursor = 0
        self._ui_requested = self._ui_paid = 0.0
        self._ui_closure = self._ui_cash_out = None
        self._ui_halt_logged = False
        self._ui_sample = None
        self._ui_borrowed = self._ui_emergency = 0.0
        self._scheduled = []
        self._post_seq = 0
        self.guarantee_minute = None
        self.halt_minute = None
        self.halt_at_floor = False
        if default_module_mode() is ModuleMode.REAL:
            # SVB-50: the calibrated module set, not default_mode=REAL (the stream
            # loader rejects real M09, so that path never built). Crowd follows M06/M08.
            with _live_opening_book(), live_flags(_live_engine_flags()):
                self.engine = build_engine(
                    modes=live_engine_modes(),
                    default_mode=ModuleMode.FAKE,
                    scenario=live_scenario(self.seed),
                )
            self.follows_engine = True
            _apply_fit_noise(self.engine)
            # SVB-65: the fitted attention feedback (news from requests, tanh composition).
            self.engine.fitted_feedback = registry_flag("M29 live fitted feedback")
            if self.engine.fitted_feedback:
                install_persisted_feedback(self.engine)
            self.price_paced = registry_flag("M29 live price paced to SVB day-one drop")
            self.halt_at_floor = self.price_paced and registry_flag("M29 live trading halt at price floor")
            _hold_closure(self.engine)
            crowd = self.engine.get("M29")
            if isinstance(crowd, CrowdFake):
                crowd.follow_engine = True
        else:
            self.engine = build_engine(
                scenario=live_scenario(self.seed),
                default_mode=default_module_mode(),
            )
            self.engine.fitted_feedback = False
            self.follows_engine = False

    def start(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def set_speed(self, speed: float) -> None:
        self.speed = normalize_speed(speed)
        self.pacing.set_speed(self.speed)

    def load_preset(self, name: str, response: str | None = None, response_minute: int | None = None,
                    plan: list[tuple[str, int]] | None = None, credibility: float | None = None) -> list[dict[str, Any]]:
        """P-A: schedule a preset's posts by minute (from the current tick) and start the run.

        PA-A: ``response`` adds a countermeasure at its minute (240), for the with/without comparison.
        """
        with self._engine_lock:
            return self._load_preset_unlocked(name, response, response_minute, plan, credibility)

    def _load_preset_unlocked(self, name: str, response: str | None = None,
                              response_minute: int | None = None,
                              plan: list[tuple[str, int]] | None = None,
                              credibility: float | None = None) -> list[dict[str, Any]]:
        preset = LIVE_PRESETS.get(name)
        if preset is None:
            raise KeyError(f"unknown preset: {name}")
        if response is not None and response not in RESPONSES:
            raise KeyError(f"unknown response: {response}")
        steps = list(plan or [])  # UI-3: a plan is several responses, each at its own minute or on a trigger
        if response is not None:
            steps.append((response, RESPONSES[response].minute if response_minute is None else int(response_minute)))
        if any(name_ not in STEPS for name_, _ in steps) or len({n for n, _ in steps}) != len(steps):   # AU-1: + TOOLS
            raise KeyError(f"bad plan: {steps}")
        for name_, when in steps:
            if not isinstance(when, (int, float)) and (not isinstance(when, (tuple, list)) or len(when) != 2
                                                       or when[0] not in TRIGGER_KINDS):
                raise KeyError(f"bad trigger: {when}")
        self.preset = name
        base = int(self.tick)
        self._responses.extend("guarantee" for post in preset.posts if post.guarantee)
        for post in preset.posts:  # UI-3e: ``credibility`` sets the rumor's strength (how believable), for the search
            cred = credibility if (credibility is not None and not post.guarantee and not post.lever) else post.credibility
            self._scheduled.append(PresetPost(base + post.minute, post.claim, post.poster, cred,
                                              post.guarantee, post.lever, post.response))
        for name_, at in steps:  # UI-3: when each response comes is the question; 240 (1:30 PM) by default
            self._responses.append(name_)
            if isinstance(at, (int, float)):
                self._scheduled.append(response_post(name_, base + int(at)))
            else:  # UI-3d: fires the minute after its condition is crossed
                self._triggers.append((name_, str(at[0]), float(at[1])))
        self.start()
        return [{"minute": p.minute, "guarantee": p.guarantee, "response": p.response or None}
                for p in self._scheduled]

    def enact_response(self, name: str) -> float:
        """PA-A: a countermeasure mid-run, at the engine's minute (ahead of the screen by the lead)."""
        if name not in RESPONSES:
            raise KeyError(f"unknown response: {name}")
        if name == "guarantee":
            return self.enact_guarantee()
        with self._engine_lock:
            minute = int(self.tick)
            self._responses.append(name)
            self._scheduled.append(response_post(name, minute))
            return float(minute)

    def enact_guarantee(self) -> float:
        """P-A: the insured chip mid-run. It takes effect at the engine's next minute; returns it.

        While buffering, the engine is ahead of the screen by at most the head start plus 2 minutes.
        """
        with self._engine_lock:
            return self._enact_guarantee_unlocked()

    def _enact_guarantee_unlocked(self) -> float:
        minute = int(self.tick)
        self._responses.append("guarantee")
        self._scheduled.append(PresetPost(minute, INSURED_CHIP.claim, INSURED_CHIP.poster,
                                          INSURED_CHIP.credibility, guarantee=True))
        return float(minute)

    def run_label(self) -> dict:
        """PA-C: ``fitted`` for the 2023 replay only; any response or typed rumor makes it a scenario."""
        label = LIVE_PRESETS[self.preset].label if self.preset in LIVE_PRESETS else None
        return run_label(self.preset, label, tuple(self._responses), self._free_text)

    def _check_triggers(self) -> None:
        """UI-3d: a trigger fires when what the bank reported up to the previous minute crosses its level."""
        if not self._triggers:
            return
        from engine.bank.ledger import registry_deposits_total

        requested = self._ui_requested / max(1.0, registry_deposits_total())
        crowd = self.engine.get("M29")
        cash = float(getattr(crowd, "cash_gauge", 1.0))
        left = []
        for name_, kind, level in self._triggers:
            hit = requested >= level if kind == "requested" else cash <= level
            if hit:
                self._scheduled.append(response_post(name_, int(self.tick)))
                self.trigger_minutes[name_] = self.tick
                self.trigger_requested[name_] = requested
            else:
                left.append((name_, kind, level))
        self._triggers = left

    def _publish_due_posts(self) -> None:
        due = [p for p in self._scheduled if p.minute <= int(self.tick)]
        if not due:
            return
        self._scheduled = [p for p in self._scheduled if p.minute > int(self.tick)]
        crowd = self.engine.get("M29")
        bank_id = crowd.bank_id if isinstance(crowd, CrowdFake) else "bank-fvb"
        for post in due:
            self._post_seq += 1
            if post.guarantee and self.follows_engine and registry_flag("M29 live insured chip guarantee"):
                self._enact_guarantee_now(bank_id)
            elif post.lever:
                if self.follows_engine:  # PA-A: live real mode only, as the guarantee
                    self._enact_lever_now(post.lever, post.response, bank_id)
            elif isinstance(crowd, CrowdFake):
                crowd.queue_live_rumor(claim=post.claim, source_node=post.poster, bank_id=bank_id,
                                       rumor_id=f"preset-{self.preset or 'chip'}-{self._post_seq}",
                                       credibility=post.credibility)

    def _enact_lever_now(self, lever: str, name: str, bank_id: str) -> None:
        """PA-A: publish the response's policy before this minute's step (M06 / M12 read it this minute)."""
        from contracts.events import PolicyEnacted

        if name in self.response_minutes:
            return
        self.response_minutes[name] = self.tick
        if lever == "statement":   # AU-1: the bank's message reaches depositors as a statement (M03 attention, M06 worry)
            from contracts.events import StatementIssued

            self.engine.bus.publish(StatementIssued(t=self.tick, seed=self.seed, bank_id=bank_id,
                                                    stmt_id=f"plan-{name}", channel="press", segments=[]))
            return
        if lever == "deposit_rate":   # AU-1: the bank raises its deposit rate for every kind of depositor
            from contracts.events import DepositRateChanged
            from params.loader import get_parameter

            rate = float(get_parameter("M42 deposit rate cap (ASSUMED)").value)
            m06 = self.engine["M06"]
            if hasattr(m06, "_ensure_population"):
                m06._ensure_population()
            for segment in sorted(set(getattr(m06, "segments", []))):
                self.engine.bus.publish(DepositRateChanged(t=self.tick, seed=self.seed, bank_id=bank_id,
                                                           segment=segment, rate=rate))
            return
        self.engine.bus.publish(
            PolicyEnacted(t=self.tick, seed=self.seed, bank_id=bank_id, cause_id=f"response-{name}",
                          lever=lever, scope=bank_id, cost_model={})
        )

    def _enact_guarantee_now(self, bank_id: str) -> None:
        """Publish the deposit guarantee before this minute's step (M06 reads it this minute)."""
        from contracts.events import PolicyEnacted

        if self.guarantee_minute is not None:
            return
        self.guarantee_minute = self.tick
        self.response_minutes["guarantee"] = self.tick
        self.engine.bus.publish(
            PolicyEnacted(t=self.tick, seed=self.seed, bank_id=bank_id, cause_id="insured-chip",
                          lever="guarantee_deposits", scope=bank_id, cost_model={})
        )

    def step_once(self) -> None:
        with self._engine_lock:
            self._check_triggers()
            self._publish_due_posts()
            step_engine(self.engine, self.tick)
            self.tick += 1.0
            if self.follows_engine:
                self._sync_engine_crowd()
                self._ui_scan()

    def _pace_price(self) -> None:
        """SVB-63: displayed price = opening x (1 - (1 - ratio) x min(1, requested / r1)).

        ``r1`` is the E1 day-one requested target (0.249) and ``ratio`` the E1 day-one
        stock target (0.40, SVB's close/open on Mar 9): when the simulated depositors
        have requested what SVB's did on day one, the price shows SVB's -60%. Paced to
        the engine's run, not a calibrated market model; the order book still runs.
        """
        from engine.bank.ledger import registry_deposits_total
        from params.loader import get_parameter

        log = self.engine.bus.log
        for event in log[self._requested_cursor :]:
            if type(event).__name__ == "WithdrawalRequested":
                self._requested_usd += float(event.amount)
        self._requested_cursor = len(log)
        r1 = float(get_parameter("Calibration E1 requested day-one target").value)
        ratio = float(get_parameter("Calibration E1 day-1 stock target").value)
        book = self.engine.get("M14")
        opening = float(getattr(getattr(book, "params", None), "opening_price", 100.0))
        fraction = self._requested_usd / registry_deposits_total()
        price = opening * (1.0 - (1.0 - ratio) * min(1.0, fraction / r1 if r1 > 0 else 1.0))
        if self.halt_minute is not None:
            return  # P-A: trading halted at the floor; no more prints for the rest of the run
        self._paced_prices.append((self.tick - 1.0, price))
        if self.halt_at_floor and price <= opening * ratio * (1.0 + 1e-12):
            self.halt_minute = self.tick - 1.0

    def _sync_engine_crowd(self) -> None:
        """Copy M06/M08 state onto the crowd panel (real mode only)."""
        if self.price_paced:
            self._pace_price()
        if self._opening_cash is None:
            self._opening_cash = ledger_cash(self.engine)
        crowd = self.engine.get("M29")
        if isinstance(crowd, CrowdFake):
            snapshot = read_engine_crowd(self.engine, opening_cash=self._opening_cash)
            if snapshot is not None and self.guarantee_minute is not None:
                # P-A: under the guarantee no deposit is at risk; whoever hasn't requested is calm.
                from dataclasses import replace

                snapshot = replace(snapshot, calm=snapshot.calm + snapshot.wavering, wavering=0)
            apply_engine_crowd(crowd, snapshot)

    def advance_wall(self, real_seconds: float) -> int:
        """Advance from wall time; returns whole minutes advanced.

        Buffering (P-A2): moves playback through computed frames, never past the newest one.
        Otherwise: steps the engine inline, as before.
        """
        if self._buffering and self._buffer is not None:
            rate = base_sim_minutes_per_real_second() * self.speed
            return self._buffer.advance(real_seconds * rate, finished=self.tick >= crisis_sim_minutes())
        steps = self.pacing.advance(real_seconds)
        for _ in range(steps):
            self.step_once()
        return steps

    def _outcome_reached(self) -> bool:
        crowd = self.engine.get("M29")
        if not isinstance(crowd, CrowdFake):
            return False
        if crowd.time_to_failure_ticks == 0:
            return True
        if crowd.cash_gauge < 0.15:
            return True
        return False

    def skip_to_outcome(self) -> int:
        """Fast-forward to the registry crisis horizon (the demo's outcome)."""
        target = float(crisis_sim_minutes())
        max_steps = int(target) + 32
        stepped = 0
        with self._engine_lock:
            while self.tick < target and stepped < max_steps:
                self.step_once()
                stepped += 1
            if self._buffering and self._buffer is not None:
                self._buffer.push(self._capture())
                self._buffer.jump_to_newest()
        return stepped

    def crowd_frame(self) -> CrowdFrame:
        """The frame on screen: the buffered one at the playback minute while buffering."""
        if self._buffering and self._buffer is not None:
            current = self._buffer.current()
            if current is not None:
                return current.crowd
        with self._engine_lock:
            return self._engine_crowd_frame()

    def market_frame(self):
        if self._buffering and self._buffer is not None:
            current = self._buffer.current()
            if current is not None:
                return current.market
        with self._engine_lock:
            return self._engine_market_frame()

    def _engine_crowd_frame(self) -> CrowdFrame:
        crowd = self.engine.get("M29")
        if not isinstance(crowd, CrowdFake):
            raise TypeError("engine has no CrowdFake at M29")
        if self._layout is None:
            from ui.crowd.frames import layout_community_count, render_point_count

            self._layout = build_layout(
                crowd.seed,
                render_point_count(),
                layout_community_count(),
            )
        return build_crowd_frame(crowd, self.tick, self._layout)

    def _engine_market_frame(self):
        frame = build_market_frame(self.engine, t=self.tick)
        if self.follows_engine and self.price_paced:
            # The order book's LULD halts belong to its own price, not the paced one on screen, and the
            # screen never cleared them (a minute-3 halt stayed up all run). Only the floor halt shows.
            frame = frame.model_copy(update={"prices": list(self._paced_prices), "halt": None,
                                             "markers": [m for m in frame.markers if m.kind != "halt"]})
        if self.halt_minute is not None:
            # P-A: SVB's stock was halted before the Mar 10 open; here at the paced -60% floor.
            from contracts.schemas import ChartMarker

            frame = frame.model_copy(update={
                "halt": {"reason": "price floor: down 60% from the open (rest of the run)", "t": self.halt_minute},
                "markers": list(frame.markers) + [ChartMarker(t=self.halt_minute, kind="halt", label="halt: price floor")],
            })
        if self.guarantee_minute is not None:
            from contracts.schemas import CrisisStatementRow

            frame = frame.model_copy(update={"statements": list(frame.statements) + [
                CrisisStatementRow(stmt_id="all-deposits-guaranteed", channel="regulator", verdict="accepted", reason="")
            ]})
        extra = [n for n in ("credit-line", "withdrawal-limit") if n in self.response_minutes]
        if extra:  # PA-A: unknown ids show as stored on the market screen
            from contracts.schemas import CrisisStatementRow

            frame = frame.model_copy(update={"statements": list(frame.statements) + [
                CrisisStatementRow(stmt_id=n, channel="regulator", verdict="accepted", reason="") for n in extra]})
        return frame

    def submit_rumor(self, text: str, poster: str, bank_id: str | None, *, now: float | None = None,
                     judgment: RumorJudgment | None = None) -> RumorSubmitResult:
        with self._engine_lock:
            return self._submit_rumor_unlocked(text, poster, bank_id, now=now, judgment=judgment)

    def _submit_rumor_unlocked(
        self,
        text: str,
        poster: str,
        bank_id: str | None,
        *,
        now: float | None = None,
        judgment: RumorJudgment | None = None,
    ) -> RumorSubmitResult:
        """Moderate, classify, and queue a rumor.

        ``judgment`` is a judge result computed earlier (the server computes it
        off the event loop and outside the session lock). With none, the judge
        runs here: the lexicon offline.
        """
        if rumor_too_long(text):
            return RumorSubmitResult(ok=False, reasons=(too_long_reason(),))
        moderation = moderate_rumor(text)
        if not moderation.allowed:
            return RumorSubmitResult(ok=False, reasons=moderation.reasons)
        classification = classify_rumor_market(text, poster, judgment=judgment)
        breakdown = (
            classification.breakdown.as_dict() if classification.breakdown is not None else None
        )
        judged = (
            classification.judgment.as_dict() if classification.judgment is not None else None
        )
        if not classification.affects_simulation:
            return RumorSubmitResult(
                ok=True,
                read_as=classification.read_as,
                affects_simulation=False,
                effect_size=0.0,
            )
        crowd = self.engine.get("M29")
        if not isinstance(crowd, CrowdFake):
            return RumorSubmitResult(ok=False, reasons=("crowd module unavailable",))
        resolved_bank = bank_id or crowd.bank_id
        self._rumor_seq += 1
        rumor_id = f"rumor-live-{self._rumor_seq}"
        decision = self._inbox.decide(
            classification.input_class,
            rumor_id,
            now=time.monotonic() if now is None else now,
        )
        if decision.kind == "rate_limited":
            return RumorSubmitResult(ok=False, reasons=(rate_limit_reason(),))
        if decision.kind == "merge":
            return RumorSubmitResult(
                ok=True,
                rumor_id=decision.rumor_id,
                read_as=classification.read_as,
                affects_simulation=True,
                effect_size=classification.effect_size,
                credibility=classification.credibility,
                credibility_breakdown=breakdown,
                judgment=judged,
            )
        self._free_text = True  # PA-C: a typed rumor is never the fitted replay
        crowd.queue_live_rumor(
            claim=text.strip(),
            source_node=poster,
            bank_id=resolved_bank,
            rumor_id=rumor_id,
            credibility=classification.credibility,
        )
        self.start()
        return RumorSubmitResult(
            ok=True,
            rumor_id=rumor_id,
            read_as=classification.read_as,
            affects_simulation=True,
            effect_size=classification.effect_size,
            credibility=classification.credibility,
            credibility_breakdown=breakdown,
            judgment=judged,
        )
