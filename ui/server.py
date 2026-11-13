"""Unified FastAPI server: crowd + market WebSockets and rumor/control API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.guardrails import moderate_rumor
from agents.rumor_inbox import rumor_too_long, too_long_reason
from agents.rumor_input import (
    classify_rumor_market,
    default_role_credibility,
    prejudge_rumor,
    rumor_roles,
)
from ui.crowd.frames import encode_crowd_frame
from ui.live.config import server_port
from ui.live.pacing import client_stream_fps, speed_steps
from engine.live_overrides import registry_flag
from ui.live.labels import FREE_TEXT_OFF, limits_screen
from ui.live.presets import preset_table, response_table
from ui.live.session import LiveSession
from ui.market.codec import frame_to_msgpack

_session = LiveSession(seed=1)
_session_lock = asyncio.Lock()
_clients: dict[str, set[WebSocket]] = {"crowd": set(), "market": set()}
_pending_send: dict[WebSocket, asyncio.Task[None]] = {}


class RumorBody(BaseModel):
    text: str = Field(min_length=1)
    poster: str = "ui-user"
    bank_id: str | None = None


class ControlBody(BaseModel):
    action: str
    speed: float | None = None


class PresetBody(BaseModel):
    name: str
    response: str | None = None  # PA-A
    response_minute: int | None = None  # UI-3: when the response comes
    plan: list[PlanStep] | None = None  # UI-3b: a searched plan, replayed live


class PlanStep(BaseModel):
    name: str
    # a clock minute, or (UI-3d) a trigger such as ["requested", 0.08] or ["cash", 0.75]: the search's plans are
    # triggers, so the wall's click and the engine's pick (UI-4) post them as two-item lists
    minute: int | list[str | float]


class ResponseBody(BaseModel):
    name: str


async def _send_bytes_drop_if_busy(socket: WebSocket, payload: bytes) -> None:
    """Backpressure: skip a frame when the client has not finished the prior send."""
    prior = _pending_send.get(socket)
    if prior is not None and not prior.done():
        return

    async def _do_send() -> None:
        try:
            await socket.send_bytes(payload)
        except Exception:
            pass

    task = asyncio.create_task(_do_send())
    _pending_send[socket] = task


async def _broadcast(channel: str, crowd_blob: bytes, market_blob: bytes) -> None:
    dead: list[WebSocket] = []
    for socket in list(_clients[channel]):
        payload = crowd_blob if channel == "crowd" else market_blob
        try:
            await _send_bytes_drop_if_busy(socket, payload)
        except Exception:
            dead.append(socket)
    for socket in dead:
        _clients[channel].discard(socket)
        _pending_send.pop(socket, None)


async def _publish_frames() -> None:
    crowd_blob = encode_crowd_frame(_session.crowd_frame())
    market_blob = frame_to_msgpack(_session.market_frame())
    await _broadcast("crowd", crowd_blob, market_blob)
    await _broadcast("market", crowd_blob, market_blob)


async def _pacing_task() -> None:
    fps = client_stream_fps()
    interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        real_dt = now - last
        last = now
        async with _session_lock:
            if not _session.paused:
                _session.advance_wall(real_dt)
            try:
                crowd_blob = encode_crowd_frame(_session.crowd_frame())
                market_blob = frame_to_msgpack(_session.market_frame())
            except Exception:
                continue
        await _broadcast("crowd", crowd_blob, market_blob)
        await _broadcast("market", crowd_blob, market_blob)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # P-A2: in real mode with M29 live buffered playback, a producer thread steps the engine
    # into a frame buffer and the pacing task only plays frames back (no engine work on the loop).
    _session.start_producer()
    task = asyncio.create_task(_pacing_task())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        _session.stop_producer()


app = FastAPI(title="STAMPEDE live", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _demo_population() -> int:
    from params.loader import RegistryError, get_parameter

    try:
        raw = get_parameter("M29 demo crowd population").value
    except RegistryError:
        return 1_000_000
    digits = "".join(ch for ch in raw if ch.isdigit())
    return int(digits) if digits else 1_000_000


def _backend_report() -> dict[str, Any]:
    try:
        from engine.xp import backend_report

        return backend_report()
    except Exception as exc:  # the health check must not fail on a GPU probe
        return {"error": str(exc)}


@app.get("/health")
async def health() -> dict[str, Any]:
    import os

    from ui.live.config import default_module_mode, gpu_available, prefer_gpu
    from ui.live.pacing import crisis_sim_minutes

    offline = os.environ.get("STAMPEDE_OFFLINE", "1") != "0"
    return {
        "status": "ok",
        "tick": _session.tick,
        "playback_tick": _session.playback_tick,
        "buffered": _session.buffering,
        "run": _session.run_label(),
        "paused": _session.paused,
        "speed": _session.speed,
        "speed_steps": list(speed_steps()),
        "crisis_sim_minutes": crisis_sim_minutes(),
        "demo_population": _demo_population(),
        "offline": offline,
        "mode": default_module_mode().value,
        "crowd_source": "engine" if _session.follows_engine else "scripted",
        "price_source": (
            "paced to SVB's day-one drop (not a calibrated market model)"
            if _session.follows_engine and _session.price_paced
            else ("order book" if _session.follows_engine else "scripted")
        ),
        "backend": "gpu" if prefer_gpu() and gpu_available() else "cpu",
        "backends": _backend_report(),   # CPY-1: what each array path uses
    }


def _apply_control(action: str, speed: float | None = None) -> dict[str, Any] | None:
    act = action.strip().lower()
    if act in {"start", "resume"}:
        _session.start()
    elif act == "pause":
        _session.pause()
    elif act == "reset":
        _session.reset()
    elif act in {"skip", "skip_to_outcome", "skip-outcome"}:
        _session.skip_to_outcome()
        _session.pause()
    elif act in {"set_speed", "speed"}:
        if speed is None:
            return {"ok": False, "error": "speed required"}
        _session.set_speed(speed)
    else:
        return {"ok": False, "error": f"unknown action: {action}"}
    return None


@app.post("/api/rumor/interpret")
async def interpret_rumor(body: RumorBody) -> dict[str, Any]:
    moderation = moderate_rumor(body.text)
    if not moderation.allowed:
        return {"ok": False, "reasons": list(moderation.reasons)}
    if rumor_too_long(body.text):
        return {"ok": False, "reasons": [too_long_reason()]}
    # The judge may call a live model: run it in a worker thread, not on the loop.
    classification = await asyncio.to_thread(classify_rumor_market, body.text, body.poster)
    breakdown = classification.breakdown
    judgment = classification.judgment
    return {
        "ok": True,
        "read_as": classification.read_as,
        "affects_simulation": classification.affects_simulation,
        "effect_size": classification.effect_size,
        "market_related": classification.market_related,
        "tone": classification.tone,
        "strength": classification.strength,
        "input_class": classification.input_class,
        "credibility": classification.credibility,
        "role": body.poster,
        "credibility_breakdown": breakdown.as_dict() if breakdown is not None else None,
        "judgment": judgment.as_dict() if judgment is not None else None,
        "strength_lexicon": classification.strength_lexicon,
    }


@app.get("/api/rumor/roles")
async def rumor_role_table() -> dict[str, Any]:
    """Registry role table so the UI and viewers see each role's ASSUMED base."""
    return {
        "roles": [{"id": key, "base": value} for key, value in rumor_roles().items()],
        "unknown_role_base": default_role_credibility(),
    }


def _free_text_allowed() -> bool:
    """PA-C: viewers pick preset rumors; typed rumors only when an operator turns the row on."""
    return registry_flag("M29 live free-text rumors")


def _free_text_refusal(text: str) -> dict[str, Any] | None:
    """None if typed rumors are allowed. Blocked names keep their moderation reasons either way."""
    if _free_text_allowed():
        return None
    moderation = moderate_rumor(text)
    if not moderation.allowed:
        return {"ok": False, "reasons": list(moderation.reasons)}
    return {"ok": False, "reasons": [FREE_TEXT_OFF]}


@app.get("/api/ui/state")
async def ui_state(price_since: int = 0, event_since: int = 0) -> dict[str, Any]:
    """UI-1: what the screens show, at the playback minute; prices and events since the given cursors."""
    async with _session_lock:
        return _session.ui_state(price_since, event_since)


@app.get("/ui")
async def ui_page():
    """The three-beat UI (a rumor hits, the engine tries the plans, what helps), served from the same origin."""
    from pathlib import Path

    from fastapi.responses import HTMLResponse

    return HTMLResponse((Path(__file__).parent / "live" / "stampede_ui.html").read_text(encoding="utf-8"))


@app.get("/api/labels")
async def labels() -> dict[str, Any]:
    """PA-C: is the current run the fitted 2023 replay or a scenario."""
    return {"run": _session.run_label()}


@app.get("/api/limits")
async def limits() -> dict[str, Any]:
    """PA-C: the limits screen's approved copy."""
    return limits_screen()


@app.post("/api/rumor")
async def post_rumor(body: RumorBody) -> dict[str, Any]:
    refusal = _free_text_refusal(body.text)
    if refusal is not None:
        return refusal
    # Judge before taking the lock so a slow model never stalls the stream.
    judgment = await asyncio.to_thread(prejudge_rumor, body.text)
    async with _session_lock:
        result = _session.submit_rumor(
            body.text, body.poster, body.bank_id, judgment=judgment
        )
        if result.ok and result.affects_simulation and not _session.buffering:
            _session.step_once()
        out = _with_timing(result.as_dict())
    if result.ok and result.affects_simulation:
        await _publish_frames()
    return out


@app.get("/api/presets")
async def presets() -> dict[str, Any]:
    """P-A: the live preset table (chips for the UI)."""
    from ui.live.batch import load_batch
    from ui.live.cards import response_cards

    cards = response_cards(load_batch())  # UI-2: the hover cards (measured from the batch file, assumed costs)
    return {"presets": preset_table(), "responses": [{**r, "card": cards.get(r["name"])} for r in response_table()]}


@app.get("/api/ui/depositors")
async def ui_depositors() -> dict[str, Any]:
    """UI-2: the vault's dots, a fixed sample of the engine's real depositors (ready after the first step)."""
    async with _session_lock:
        return _session.vault_layout()


def _load_preset(name: str, response: str | None = None, response_minute: int | None = None,
                 plan: list[tuple[str, int]] | None = None) -> dict[str, Any]:
    try:
        scheduled = _session.load_preset(name, response, response_minute, plan)
    except KeyError as exc:
        return {"ok": False, "error": str(exc).strip("'\"")}
    return {"ok": True, "preset": name, "response": response, "scheduled": scheduled, "tick": _session.tick,
            "run": _session.run_label()}


def _response(name: str) -> dict[str, Any]:
    """PA-A: a countermeasure mid-run, at the engine's minute."""
    try:
        _session.enact_response(name)
    except KeyError as exc:
        return {"ok": False, "error": str(exc).strip("'\"")}
    return _with_timing({"ok": True, "response": name, "tick": _session.tick})


def _with_timing(out: dict[str, Any]) -> dict[str, Any]:
    """P-A2: while buffering the engine runs ahead of the screens; say when a post takes effect."""
    out["takes_effect_at_minute"] = _session.tick
    out["screen_minute"] = _session.playback_tick
    return out


def _guarantee() -> dict[str, Any]:
    _session.enact_guarantee()
    return _with_timing({"ok": True, "tick": _session.tick})


@app.post("/api/preset")
async def post_preset(body: PresetBody) -> dict[str, Any]:
    async with _session_lock:
        result = _load_preset(body.name, body.response, body.response_minute,
                              [(p.name, p.minute) for p in body.plan] if body.plan else None)
    if result["ok"]:
        await _publish_frames()
    return result


@app.get("/api/batch")
async def batch() -> dict[str, Any]:
    """PA-B: the multi-seed table's summary (the file scripts/batch_table.py writes)."""
    from ui.live.batch import batch_file, load_batch

    data = load_batch()
    if data is None:
        return {"ok": False, "error": f"no batch file at {batch_file()}; run scripts/batch_table.py"}
    return {"ok": True, **{k: v for k, v in data.items() if k != "rows"}}


@app.get("/api/batch/run")
async def batch_run(preset: str, seed: int, response: str | None = None) -> dict[str, Any]:
    """PA-B: one run's series, e.g. the no-response run behind PA-A's with/without comparison."""
    from ui.live.batch import load_batch

    data = load_batch()
    if data is None:
        return {"ok": False, "error": "no batch file"}
    for row in data["rows"]:
        if row["preset"] == preset and row["seed"] == seed and row["response"] == (response or None):
            return {"ok": True, "run": row}
    return {"ok": False, "error": "no such run in the batch"}


def _pipeline_status_path() -> Path:
    from ui.live.search import search_file

    return Path(search_file()).parent / "pipeline_status.json"


def _pipeline_status() -> dict[str, Any]:
    path = _pipeline_status_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"stage": None}


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


@app.post("/api/pipeline/run")
async def pipeline_run(force: bool = False) -> dict[str, Any]:
    """AU-1: start the whole search as one background job (scripts/pipeline.py); the UI polls the status."""
    import subprocess
    import sys

    now = _pipeline_status()
    if now.get("stage") not in (None, "done", "failed") and _pid_alive(now.get("pid")):
        return {"ok": False, "error": "the search is already running", **now}
    root = Path(__file__).resolve().parents[1]
    log = open(_pipeline_status_path().with_name("pipeline.log"), "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "scripts/pipeline.py"] + (["--force"] if force else []), cwd=root,
                     env={**os.environ, "PYTHONPATH": str(root)}, stdout=log, stderr=log, start_new_session=True)
    return {"ok": True, "stage": "starting"}


@app.get("/api/pipeline/status")
async def pipeline_status() -> dict[str, Any]:
    return {"ok": True, **_pipeline_status()}


@app.get("/api/search")
async def search(runs: int | None = None) -> dict[str, Any]:
    """UI-3: the response search (rounds, eliminations, held-out check, deadlines), without the recorded series.

    UI-4: ``?runs=N`` serves the option with N runs per plan (the first N seeds of the pool); without it, the
    largest. ``run_options`` lists the options whose files are on the server, for the page's selector.
    """
    from ui.live.search import available_run_options, load_search, option_file, search_file

    options = available_run_options()
    if runs is not None and runs not in options:
        return {"ok": False, "error": f"no search with {runs} runs per plan; options: {options}"}
    data = load_search(runs=runs)
    if data is None:
        return {"ok": False, "error": f"no search file at {option_file(runs) if runs else search_file()}; run scripts/search_responses.py"}
    from engine.bank.ledger import registry_deposits_total

    # UI-8: the bank's deposits, so the page can state margins in dollars without the live engine
    return {"ok": True, **{k: v for k, v in data.items() if k != "series"}, "run_options": options,
            "deposits_usd": float(registry_deposits_total())}


@app.get("/api/search/run")
async def search_run(plan: str, seed: int, runs: int | None = None) -> dict[str, Any]:
    """UI-3: one searched run's curve (every 10 minutes: requested share, cash), for the tournament replay."""
    from ui.live.search import load_search

    data = load_search(runs=runs) or {}
    series = data.get("series", {}).get(f"{plan}|{seed}")
    return {"ok": series is not None, "plan": plan, "seed": seed, "series": series,
            "outcome": data.get("outcomes", {}).get(f"{plan}|{seed}")}


@app.post("/api/response")
async def post_response(body: ResponseBody) -> dict[str, Any]:
    """PA-A: deposit guarantee, emergency credit line or withdrawal limit, mid-run."""
    async with _session_lock:
        return _response(body.name)


@app.post("/api/guarantee")
async def post_guarantee() -> dict[str, Any]:
    """P-A: the insured chip mid-run (the regulator's full deposit guarantee)."""
    async with _session_lock:
        return _guarantee()


@app.post("/api/control")
async def post_control(body: ControlBody) -> dict[str, Any]:
    async with _session_lock:
        err = _apply_control(body.action, body.speed)
        if err is not None:
            return err
    await _publish_frames()
    return {
        "ok": True,
        "action": body.action,
        "paused": _session.paused,
        "tick": _session.tick,
        "speed": _session.speed,
    }


async def _handle_socket_command(message: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid json"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "expected object"}
    cmd = str(payload.get("cmd", "")).lower()
    if cmd == "rumor":
        text = str(payload.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "text required"}
        refusal = _free_text_refusal(text)
        if refusal is not None:
            return refusal
        poster = str(payload.get("poster", "ui-user"))
        bank_id = payload.get("bank_id")
        bank = str(bank_id) if bank_id is not None else None
        judgment = await asyncio.to_thread(prejudge_rumor, text)
        async with _session_lock:
            result = _session.submit_rumor(text, poster, bank, judgment=judgment)
            if result.ok and result.affects_simulation and not _session.buffering:
                _session.step_once()
            out = _with_timing(result.as_dict())
        if result.ok and result.affects_simulation:
            await _publish_frames()
        return out
    if cmd == "preset":
        async with _session_lock:
            response_raw = payload.get("response")
            result = _load_preset(str(payload.get("name", "")), str(response_raw) if response_raw else None)
        if result["ok"]:
            await _publish_frames()
        return result
    if cmd == "guarantee":
        async with _session_lock:
            return _guarantee()
    if cmd == "response":
        async with _session_lock:
            return _response(str(payload.get("name", "")))
    if cmd in {"start", "resume", "pause", "reset", "skip", "skip_to_outcome"}:
        speed_raw = payload.get("speed")
        speed = float(speed_raw) if speed_raw is not None else None
        async with _session_lock:
            if cmd in {"set_speed", "speed"}:
                err = _apply_control("set_speed", speed)
            else:
                err = _apply_control(cmd, speed)
            if err is not None:
                return err
        await _publish_frames()
        return {
            "ok": True,
            "cmd": cmd,
            "paused": _session.paused,
            "tick": _session.tick,
            "speed": _session.speed,
        }
    if cmd in {"set_speed", "speed"}:
        speed_raw = payload.get("speed")
        if speed_raw is None:
            return {"ok": False, "error": "speed required"}
        async with _session_lock:
            err = _apply_control("set_speed", float(speed_raw))
            if err is not None:
                return err
        await _publish_frames()
        return {"ok": True, "cmd": cmd, "speed": _session.speed}
    return {"ok": False, "error": f"unknown cmd: {cmd}"}


@app.websocket("/ws/crowd")
async def crowd_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients["crowd"].add(websocket)
    try:
        async with _session_lock:
            await _send_bytes_drop_if_busy(
                websocket, encode_crowd_frame(_session.crowd_frame())
            )
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            reply = await _handle_socket_command(message)
            if reply is not None:
                await websocket.send_text(json.dumps(reply))
    finally:
        _clients["crowd"].discard(websocket)
        _pending_send.pop(websocket, None)


@app.websocket("/ws/market")
async def market_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients["market"].add(websocket)
    try:
        async with _session_lock:
            await _send_bytes_drop_if_busy(
                websocket, frame_to_msgpack(_session.market_frame())
            )
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            reply = await _handle_socket_command(message)
            if reply is not None:
                await websocket.send_text(json.dumps(reply))
    finally:
        _clients["market"].discard(websocket)
        _pending_send.pop(websocket, None)


def main() -> None:
    import uvicorn

    uvicorn.run("ui.server:app", host="0.0.0.0", port=server_port(), reload=False)


if __name__ == "__main__":
    main()
