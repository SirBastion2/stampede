"""Local LLM settings, fake backend, and OpenAI-compatible HTTP (§7.1)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from params.loader import RegistryError, get_parameter

T = TypeVar("T")

_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_ON_VALUES = frozenset({"1", "true", "yes", "on"})
_LLM_MODES = frozenset({"offline", "grok", "local"})
_PARAM_LOCAL_MODEL = "Local LLM default model (ASSUMED)"
_PARAM_LOCAL_URL = "Local LLM OpenAI base URL (ASSUMED)"
_PARAM_LOCAL_CONCURRENCY = "Local LLM concurrency semaphore (ASSUMED)"
_PARAM_LOCAL_TIMEOUT = "Local LLM request timeout seconds (ASSUMED)"
_PARAM_LOCAL_WEIGHTS = "Local LLM weights directory (ASSUMED)"

local_llm_validation_failures = 0
_local_semaphore: threading.Semaphore | None = None
_fake_backend: FakeLocalBackend | None = None


class FakeLocalBackend:
    """CPU in-process backend for unit tests. Never loads weights or CUDA."""

    def __init__(
        self,
        json_handler: Callable[[str], dict[str, Any]] | None = None,
        text_handler: Callable[[str], str] | None = None,
    ) -> None:
        self.json_handler = json_handler
        self.text_handler = text_handler
        self.json_calls = 0
        self.text_calls = 0

    def complete_json(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del json_schema
        self.json_calls += 1
        if self.json_handler is None:
            raise RuntimeError("fake JSON handler is not installed")
        parsed = self.json_handler(prompt)
        if not isinstance(parsed, dict):
            raise RuntimeError("fake JSON handler must return an object")
        return parsed

    def complete_text(self, prompt: str) -> str:
        self.text_calls += 1
        if self.text_handler is None:
            raise RuntimeError("fake text handler is not installed")
        return self.text_handler(prompt)


def install_fake_local_backend(backend: FakeLocalBackend | None) -> None:
    global _fake_backend
    _fake_backend = backend


def fake_backend_enabled() -> bool:
    if _fake_backend is not None:
        return True
    return os.environ.get("STAMPEDE_LOCAL_LLM_FAKE", "").strip().lower() in _ON_VALUES


def record_local_validation_failure() -> None:
    global local_llm_validation_failures
    local_llm_validation_failures += 1


def offline_mode() -> bool:
    value = os.environ.get("STAMPEDE_OFFLINE", "1").strip().lower()
    return value in _ON_VALUES


def offline_enabled() -> bool:
    raw = os.environ.get("STAMPEDE_OFFLINE", "1")
    return raw.strip().lower() not in _OFF_VALUES


def stampede_llm_mode() -> Literal["offline", "grok", "local"]:
    raw = os.environ.get("STAMPEDE_LLM", "offline").strip().lower()
    if raw in _LLM_MODES:
        return cast(Literal["offline", "grok", "local"], raw)
    return "offline"


@lru_cache(maxsize=8)
def _registry_value(name: str, default: str) -> str:
    try:
        return get_parameter(name).value.strip() or default
    except RegistryError:
        return default


def local_llm_model() -> str:
    override = os.environ.get("STAMPEDE_LOCAL_LLM_MODEL", "").strip()
    if override:
        return override
    return _registry_value(_PARAM_LOCAL_MODEL, "Meta-Llama-3.1-8B-Instruct-Q4_K_M")


def local_llm_base_url() -> str:
    override = os.environ.get("STAMPEDE_LOCAL_LLM_URL", "").strip()
    if override:
        return override.rstrip("/")
    return _registry_value(_PARAM_LOCAL_URL, "http://127.0.0.1:8080/v1").rstrip("/")


def local_llm_timeout_seconds() -> float:
    override = os.environ.get("STAMPEDE_LOCAL_LLM_TIMEOUT", "").strip()
    if override:
        return float(override)
    return float(_registry_value(_PARAM_LOCAL_TIMEOUT, "120"))


def local_llm_concurrency() -> int:
    raw = _registry_value(_PARAM_LOCAL_CONCURRENCY, "4")
    try:
        return max(1, int(raw.split()[0]))
    except (ValueError, IndexError):
        return 4


def local_llm_weights_dir() -> str:
    override = os.environ.get("STAMPEDE_LOCAL_LLM_WEIGHTS", "").strip()
    if override:
        return override
    return _registry_value(_PARAM_LOCAL_WEIGHTS, "")


def local_llm_gguf_path() -> str:
    override = os.environ.get("STAMPEDE_LOCAL_LLM_GGUF", "").strip()
    if override:
        return override
    root = local_llm_weights_dir()
    if not root:
        return ""
    return str(Path(root) / f"{local_llm_model()}.gguf")


def nvidia_gpu_visible(*, probe: Callable[[], bool] | None = None) -> bool:
    if probe is not None:
        return probe()
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def require_gpu_for_inference(*, probe: Callable[[], bool] | None = None) -> None:
    if fake_backend_enabled():
        return
    if not nvidia_gpu_visible(probe=probe):
        raise RuntimeError(
            "real local inference requires an NVIDIA GPU; unit tests use FakeLocalBackend"
        )


def effective_cache_model(agent_model: str) -> str:
    if stampede_llm_mode() == "local":
        return local_llm_model()
    return agent_model


def live_json_completion_enabled() -> bool:
    mode = stampede_llm_mode()
    if mode == "local":
        return True
    return not offline_mode() and bool(os.environ.get("XAI_API_KEY", "").strip())


def live_text_completion_enabled() -> bool:
    return live_json_completion_enabled()


def _local_gate() -> threading.Semaphore:
    global _local_semaphore
    if _local_semaphore is None:
        _local_semaphore = threading.Semaphore(local_llm_concurrency())
    return _local_semaphore


def _chat_completion_content(
    *,
    model: str,
    messages: list[dict[str, str]],
    json_object: bool = False,
    json_schema: dict[str, Any] | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> str:
    url = f"{(base_url or local_llm_base_url()).rstrip('/')}/chat/completions"
    body: dict[str, Any] = {"model": model, "messages": messages}
    if json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "stampede_response",
                "strict": True,
                "schema": json_schema,
            },
        }
    elif json_object:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    wait = timeout if timeout is not None else local_llm_timeout_seconds()
    try:
        with urllib.request.urlopen(request, timeout=wait) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if json_schema is not None and exc.code == 400:
            return _chat_completion_content(
                model=model,
                messages=messages,
                json_object=True,
                json_schema=None,
                timeout=wait,
                base_url=base_url,
            )
        raise
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("empty LLM response")
    return content


def complete_local_chat_text(model: str, prompt: str) -> str:
    if _fake_backend is not None:
        return _fake_backend.complete_text(prompt)
    if fake_backend_enabled():
        raise RuntimeError("STAMPEDE_LOCAL_LLM_FAKE is set but no fake backend is installed")
    with _local_gate():
        return _chat_completion_content(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )


def complete_local_chat_json(
    model: str,
    prompt: str,
    *,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _fake_backend is not None:
        return _fake_backend.complete_json(prompt, json_schema)
    if fake_backend_enabled():
        raise RuntimeError("STAMPEDE_LOCAL_LLM_FAKE is set but no fake backend is installed")
    with _local_gate():
        content = _chat_completion_content(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            json_object=True,
            json_schema=json_schema,
        )
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("local LLM response was not a JSON object")
    return cast(dict[str, Any], parsed)
