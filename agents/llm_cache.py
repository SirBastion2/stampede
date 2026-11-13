"""SQLite-backed LLM response cache (§7.1). Used by Grok agents; offline-safe.

M25 keys rows by prompt hash and model (`LLMCache` JSON table). M24 keys a
statement cache by namespace payload (`LlmCache`). M44 short sellers use the
text table on the same `LLMCache` connection via `cached_text`.
Local HTTP, fake backend, and settings live in `/agents/local_llm.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from agents import local_llm as _local
from agents.local_llm import (
    complete_local_chat_json,
    complete_local_chat_text,
    effective_cache_model,
    live_json_completion_enabled,
    live_text_completion_enabled,
    local_llm_model,
    offline_enabled,
    offline_mode,
    record_local_validation_failure,
    stampede_llm_mode,
)

__all__ = (
    "effective_cache_model",
    "live_json_completion_enabled",
    "live_text_completion_enabled",
    "offline_enabled",
    "offline_mode",
    "record_local_validation_failure",
    "stampede_llm_mode",
)

T = TypeVar("T")


class JsonCompletionProvider(Protocol):
    """Return a JSON object for a prompt; may perform network I/O."""

    def complete_json(self, prompt: str) -> dict[str, Any]: ...


def prompt_hash(prompt: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()
    return digest


def text_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _call_grok_chat_json(prompt: str, model: str) -> dict[str, Any]:
    try:
        from xai_sdk import Client
    except ImportError as exc:
        raise RuntimeError("xai-sdk is required for live Grok calls") from exc
    client = Client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 429:
            raise
        raise RuntimeError(f"Grok chat completion failed: {exc}") from exc
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("empty Grok response")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Grok response was not a JSON object")
    return cast(dict[str, Any], parsed)


def _call_grok_chat_text(prompt: str, model: str) -> str:
    try:
        from xai_sdk import Client
    except ImportError as exc:
        raise RuntimeError("xai-sdk is required for live Grok calls") from exc
    client = Client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("empty Grok response")
    return content


def complete_json_llm(
    prompt: str,
    model: str,
    *,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remote JSON completion via grok or local HTTP (§7.1)."""
    mode = stampede_llm_mode()
    if mode == "local":
        return complete_local_chat_json(
            local_llm_model(),
            prompt,
            json_schema=json_schema,
        )
    return _call_grok_chat_json(prompt, model)


def complete_text_llm(prompt: str, model: str) -> str:
    """Remote plain-text completion via grok or local HTTP."""
    mode = stampede_llm_mode()
    if mode == "local":
        return complete_local_chat_text(local_llm_model(), prompt)
    return _call_grok_chat_text(prompt, model)


def complete_json_or_fallback(
    prompt: str,
    model: str,
    *,
    validate: Callable[[dict[str, Any]], T],
    fallback: Callable[[], T],
    json_schema: dict[str, Any] | None = None,
) -> T:
    """Call the live backend twice, then the rule path. Counts validation misses."""
    if not live_json_completion_enabled():
        return fallback()
    for attempt in range(2):
        try:
            raw = complete_json_llm(prompt, model, json_schema=json_schema)
            return validate(raw)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ):
            if attempt == 1:
                record_local_validation_failure()
    return fallback()


class RemoteJsonBridge:
    """Cache bridge that routes ``complete_json_llm`` for a fixed prompt."""

    def __init__(
        self,
        prompt: str,
        model: str,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        self._prompt = prompt
        self._model = model
        self._json_schema = json_schema

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if prompt != self._prompt:
            raise RuntimeError("prompt mismatch in remote JSON cache bridge")
        return complete_json_llm(
            self._prompt,
            self._model,
            json_schema=self._json_schema,
        )


def live_grok_text(model: str, prompt: str) -> str | None:
    """Return a live Grok completion, or None so the caller uses its rule."""
    if not live_text_completion_enabled():
        return None
    if stampede_llm_mode() == "local":
        try:
            return complete_text_llm(prompt, model)
        except (OSError, RuntimeError, json.JSONDecodeError, urllib.error.URLError):
            return None
    if offline_enabled():
        return None
    if not os.environ.get("XAI_API_KEY", "").strip():
        return None
    try:
        return _call_grok_chat_text(prompt, model)
    except (ImportError, RuntimeError, OSError):
        return None


class LLMCache:
    """JSON and plain-text prompt cache keyed by model and prompt hash."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            self.path: Path | None = None
            self._conn = sqlite3.connect(":memory:")
        else:
            self.path = Path(path)
            self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                response_json TEXT NOT NULL,
                PRIMARY KEY (prompt_hash, model)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_text_cache (
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                PRIMARY KEY (model, prompt_hash)
            )
            """
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str, model: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT response_json FROM llm_cache WHERE prompt_hash = ? AND model = ?",
            (key, model),
        ).fetchone()
        if row is None:
            return None
        return cast(dict[str, Any], json.loads(row[0]))

    def put(self, key: str, model: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (prompt_hash, model, response_json) VALUES (?, ?, ?)",
            (key, model, json.dumps(payload, sort_keys=True)),
        )
        self._conn.commit()

    def get_text(self, model: str, prompt: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_text_cache WHERE model = ? AND prompt_hash = ?",
            (model, text_prompt_hash(prompt)),
        ).fetchone()
        if row is None:
            return None
        self.hits += 1
        return str(row[0])

    def put_text(self, model: str, prompt: str, response: str) -> None:
        self._conn.execute(
            """
            INSERT INTO llm_text_cache (model, prompt_hash, response)
            VALUES (?, ?, ?)
            ON CONFLICT(model, prompt_hash) DO UPDATE SET response = excluded.response
            """,
            (model, text_prompt_hash(prompt), response),
        )
        self._conn.commit()


def cached_json_completion(
    *,
    prompt: str,
    model: str,
    provider: JsonCompletionProvider,
    cache: LLMCache | None = None,
) -> dict[str, Any]:
    """Call ``provider`` at most once per (prompt, model); always validate downstream."""
    key = prompt_hash(prompt, model)
    store = cache or LLMCache()
    hit = store.get(key, model)
    if hit is not None:
        return hit
    payload = provider.complete_json(prompt)
    store.put(key, model, payload)
    return payload


def cached_text(
    *,
    model: str,
    prompt: str,
    fallback: Callable[[], str],
    validate: Callable[[str], str],
    cache: LLMCache,
) -> str:
    """Return validated text from the cache, then a live reply, then ``fallback``."""
    cached = cache.get_text(model, prompt)
    if cached is not None:
        try:
            return validate(cached)
        except ValueError:
            pass
    cache.misses += 1
    if live_text_completion_enabled():
        live = live_grok_text(model, prompt)
        if live is not None:
            try:
                checked = validate(live)
            except ValueError:
                checked = None
            if checked is not None:
                cache.put_text(model, prompt, checked)
                return checked
    checked = validate(fallback())
    cache.put_text(model, prompt, checked)
    return checked


def resolve_provider(
    live_factory: Callable[[], JsonCompletionProvider],
    offline_factory: Callable[[], JsonCompletionProvider],
) -> JsonCompletionProvider:
    if stampede_llm_mode() == "local":
        return live_factory()
    if stampede_llm_mode() == "grok":
        if offline_mode() or not os.environ.get("XAI_API_KEY"):
            return offline_factory()
        return live_factory()
    if offline_mode() or not os.environ.get("XAI_API_KEY"):
        return offline_factory()
    return live_factory()


def _default_cache_path() -> Path:
    root = os.environ.get("STAMPEDE_LLM_CACHE")
    if root:
        return Path(root)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "stampede_llm_cache.sqlite"


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"namespace": namespace, "payload": payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class LlmCache:
    """In-process SQLite cache keyed by prompt hash (M24 vague-claim review)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_cache_path()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS m24_llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM m24_llm_cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO m24_llm_cache (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def cached_json(
    namespace: str,
    payload: dict[str, Any],
    *,
    producer: Callable[[], dict[str, Any]] | None = None,
    cache: LlmCache | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable output; never calls the network when offline."""
    key = cache_key(namespace, payload)
    store = cache or LlmCache()
    hit = store.get(key)
    if hit is not None:
        return cast(dict[str, Any], json.loads(hit))
    if not live_json_completion_enabled() or producer is None:
        raise RuntimeError(f"offline: no cached LLM response for {namespace}")
    result = producer()
    store.set(key, json.dumps(result, sort_keys=True))
    return result


def cached_or_offline(
    namespace: str,
    payload: dict[str, Any],
    offline: Callable[[dict[str, Any]], T],
    *,
    producer: Callable[[], T] | None = None,
    cache: LlmCache | None = None,
) -> T:
    """Use cache, else live producer, else the rule path."""
    key = cache_key(namespace, payload)
    store = cache or LlmCache()
    hit = store.get(key)
    if hit is not None:
        return cast(T, json.loads(hit))
    if not live_json_completion_enabled() or producer is None:
        return offline(payload)
    try:
        value = producer()
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        record_local_validation_failure()
        return offline(payload)
    store.set(key, json.dumps(value, sort_keys=True))
    return value


def __getattr__(name: str) -> Any:
    if name == "local_llm_validation_failures":
        return _local.local_llm_validation_failures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
