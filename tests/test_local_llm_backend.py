"""Local OpenAI-compatible LLM backend (§7.1). Stub HTTP only; no real model."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from agents import local_llm as local_mod
from agents.llm_cache import (
    LLMCache,
    RemoteJsonBridge,
    cached_json_completion,
    complete_json_llm,
    effective_cache_model,
    live_json_completion_enabled,
    record_local_validation_failure,
)
from agents.personas import ArchetypeCard, RuleBasedPersonaProvider, validate_persona_batch


class _GoodScore(BaseModel):
    worry_delta: float = Field(ge=-1.0, le=1.0)
    threshold_delta: float = Field(ge=-1.0, le=1.0)
    act_now_share: float = Field(ge=0.0, le=1.0)


class _GoodBatch(BaseModel):
    archetype_version: str
    scores: dict[str, _GoodScore]


def _reset_failure_counter() -> None:
    local_mod.local_llm_validation_failures = 0


class _StubHandler(BaseHTTPRequestHandler):
    calls = 0
    mode: str = "ok"
    hang_seconds: float = 0.0

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_POST(self) -> None:
        _StubHandler.calls += 1
        if self.path.endswith("/chat/completions"):
            if _StubHandler.hang_seconds > 0:
                import time

                time.sleep(_StubHandler.hang_seconds)
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode())
            assert isinstance(body.get("messages"), list)
            if _StubHandler.mode == "invalid":
                payload = {"choices": [{"message": {"content": '{"not": "schema"}'}}]}
            elif _StubHandler.mode == "ok":
                payload = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "archetype_version": "v1",
                                        "scores": {
                                            "a1": {
                                                "worry_delta": 0.1,
                                                "threshold_delta": 0.0,
                                                "act_now_share": 0.2,
                                            }
                                        },
                                    }
                                )
                            }
                        }
                    ]
                }
            else:
                payload = {"choices": [{"message": {"content": "{}"}}]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)


@pytest.fixture
def local_stub_server(monkeypatch: pytest.MonkeyPatch) -> str:
    _StubHandler.calls = 0
    _StubHandler.mode = "ok"
    _StubHandler.hang_seconds = 0.0
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setenv("STAMPEDE_LLM", "local")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_URL", base)
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_MODEL", "Meta-Llama-3.1-8B-Instruct-Q4_K_M")
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_TIMEOUT", "2")
    yield base
    server.shutdown()


def test_live_json_enabled_in_local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_LLM", "local")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    assert live_json_completion_enabled() is True


def test_local_success(local_stub_server: str) -> None:
    _reset_failure_counter()
    prompt = json.dumps({"task": "persona_panel"}, sort_keys=True)
    raw = complete_json_llm(prompt, "grok-4.1-fast")
    batch = validate_persona_batch(raw, archetype_version="v1")
    assert batch.scores["a1"].worry_delta == 0.1
    assert _StubHandler.calls == 1


def test_invalid_json_then_rule_fallback(local_stub_server: str) -> None:
    _reset_failure_counter()
    _StubHandler.mode = "invalid"
    prompt = json.dumps({"task": "persona_panel"}, sort_keys=True)
    rule = RuleBasedPersonaProvider()
    failures_before = local_mod.local_llm_validation_failures
    archetypes = (ArchetypeCard(archetype_id="a1", segment="startup", label="x"),)
    fixed: dict[str, object] | None = None
    for attempt in range(2):
        raw = complete_json_llm(prompt, "grok-4.1-fast")
        try:
            validate_persona_batch(raw, archetype_version="v1")
            fixed = raw
            break
        except ValueError:
            if attempt == 1:
                record_local_validation_failure()
                fixed = rule.score(
                    preset="tech-concentrated",
                    seed=1,
                    stmt_id="s1",
                    channel="press_release",
                    segments=["startup"],
                    text="We are well capitalized.",
                    archetypes=archetypes,
                    archetype_version="v1",
                ).model_dump()
                break
    assert fixed is not None
    _ = validate_persona_batch(fixed, archetype_version="v1")
    assert _StubHandler.calls == 2
    assert local_mod.local_llm_validation_failures == failures_before + 1


def test_timeout_then_fallback(local_stub_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _StubHandler.hang_seconds = 5.0
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_TIMEOUT", "0.2")
    prompt = json.dumps({"task": "persona_panel"}, sort_keys=True)
    rule = RuleBasedPersonaProvider()

    import urllib.error

    try:
        complete_json_llm(prompt, "grok-4.1-fast")
        raised = False
    except (OSError, TimeoutError, urllib.error.URLError):
        raised = True
    assert raised
    fallback = rule.score(
        preset="tech-concentrated",
        seed=1,
        stmt_id="s1",
        channel="press_release",
        segments=["startup"],
        text="We are well capitalized.",
        archetypes=(ArchetypeCard(archetype_id="a1", segment="startup", label="x"),),
        archetype_version="v1",
    ).model_dump()
    assert "scores" in fallback


def test_cache_hit_no_second_http_call(local_stub_server: str, tmp_path: Path) -> None:
    _StubHandler.calls = 0
    cache = LLMCache(tmp_path / "local.sqlite3")
    prompt = json.dumps({"task": "persona_panel"}, sort_keys=True)
    model = effective_cache_model("grok-4.1-fast")
    bridge = RemoteJsonBridge(prompt, "grok-4.1-fast")
    first = cached_json_completion(prompt=prompt, model=model, provider=bridge, cache=cache)
    second = cached_json_completion(prompt=prompt, model=model, provider=bridge, cache=cache)
    assert first == second
    assert _StubHandler.calls == 1
    cache.close()
