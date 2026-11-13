"""M33 observability, determinism, and demo mode — Wave B (§5, §19)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from contracts.schemas import Scenario
from ops.event_log import event_log_hash, write_event_log_parquet
from ops.llm_cache import CacheEntry, LlmCache
from ops.rng import philox_uniform
from params.loader import RegistryError, get_parameter

_PARAM_DEMO_MODE = "M33 demo mode default"
_PARAM_OFFLINE = "M33 offline flag default"
_PARAM_PHILOX_SAMPLES = "M33 Philox samples per tick"
_PARAM_CACHE_FILE = "M33 LLM cache filename"
_PARAM_DEMO_PROMPT_HASH = "M33 demo script prompt hash"
_PARAM_DEMO_MODEL = "M33 demo script model name"
_PARAM_DEMO_RESPONSE = "M33 demo script cached response"


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    demo_mode_default: bool
    offline_default: bool
    philox_samples_per_tick: int
    cache_filename: str
    demo_prompt_hash: str
    demo_model: str
    demo_response: str


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    tick_time_sec: float
    gpu_memory_mb: float
    llm_latency_ms: float
    philox_first: float


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "1", "on"}:
        return True
    if lowered in {"false", "no", "0", "off"}:
        return False
    raise RegistryError(f"cannot parse boolean registry value: {value!r}")


def _parse_int(value: str) -> int:
    text = value.strip().replace(",", "")
    for token in text.split():
        if token.isdigit():
            return int(token)
    raise RegistryError(f"cannot parse integer registry value: {value!r}")


def load_observability_config(path: Path | None = None) -> ObservabilityConfig:
    """Load M33 parameters from the registry (§8.1)."""
    return ObservabilityConfig(
        demo_mode_default=_parse_bool(get_parameter(_PARAM_DEMO_MODE, path).value),
        offline_default=_parse_bool(get_parameter(_PARAM_OFFLINE, path).value),
        philox_samples_per_tick=max(
            0, _parse_int(get_parameter(_PARAM_PHILOX_SAMPLES, path).value)
        ),
        cache_filename=get_parameter(_PARAM_CACHE_FILE, path).value.strip(),
        demo_prompt_hash=get_parameter(_PARAM_DEMO_PROMPT_HASH, path).value.strip(),
        demo_model=get_parameter(_PARAM_DEMO_MODEL, path).value.strip(),
        demo_response=get_parameter(_PARAM_DEMO_RESPONSE, path).value.strip(),
    )


def _config(config: ObservabilityConfig | None) -> ObservabilityConfig:
    return config if config is not None else load_observability_config()


def demo_mode_enabled(*, config: ObservabilityConfig | None = None) -> bool:
    return _config(config).demo_mode_default


def offline_flag_default(*, config: ObservabilityConfig | None = None) -> bool:
    return _config(config).offline_default


def offline_ready(cache: LlmCache, *, config: ObservabilityConfig | None = None) -> bool:
    """True when offline/demo mode can run without network (acceptance smoke)."""
    cfg = _config(config)
    if not cfg.offline_default and not cfg.demo_mode_default:
        return False
    if cache.get(cfg.demo_prompt_hash, cfg.demo_model) is not None:
        return True
    return cache.has_entries()


def _gpu_memory_mb() -> float:
    try:
        import cupy as cp

        mempool = cp.get_default_memory_pool()
        used = float(mempool.used_bytes())
        return used / (1024.0 * 1024.0)
    except ImportError:
        return 0.0


def _default_cache_path(cfg: ObservabilityConfig) -> Path:
    return Path(".stampede") / cfg.cache_filename


class ObservabilityModule:
    """M33 scheduler module: seeded Philox, Parquet log, offline cache, metrics (§19)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        module_id: str = "M33",
        *,
        config: ObservabilityConfig | None = None,
        cache: LlmCache | None = None,
        log_path: Path | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        self.module_id = module_id
        self._config = config
        self._cache = cache
        self._log_path = log_path
        self._prefer_gpu = prefer_gpu
        self.scenario: Scenario | None = None
        self.bus: Any = None
        self.inbox: list[BaseModel] = []
        self._cursor = 0
        self.event_count = 0
        self.log_hash = ""
        self.offline = False
        self.metrics: MetricsSnapshot | None = None
        self.philox_samples: tuple[float, ...] = ()
        self._philox_offset = 0
        self._attached = False

    def attach(self, bus: Any, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        cfg = _config(self._config)
        if self._cache is None:
            self._cache = LlmCache(_default_cache_path(cfg))
        if cfg.demo_mode_default:
            self._cache.seed_demo_entries(
                (
                    CacheEntry(
                        prompt_hash=cfg.demo_prompt_hash,
                        model=cfg.demo_model,
                        response=cfg.demo_response,
                    ),
                )
            )
        self.offline = offline_ready(self._cache, config=cfg)
        for event_type in self.consumes:
            bus.subscribe(event_type, self)
        self._attached = True

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        return self.on_tick(t, new_events)

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.bus is None:
            return []
        cfg = _config(self._config)
        started = time.perf_counter()
        seed = self.scenario.seed if self.scenario is not None else 0
        count = cfg.philox_samples_per_tick
        self.philox_samples = philox_uniform(
            seed,
            count,
            offset=self._philox_offset,
            prefer_gpu=self._prefer_gpu,
        )
        self._philox_offset += count
        llm_started = time.perf_counter()
        cached = None
        if self._cache is not None:
            cached = self._cache.get(cfg.demo_prompt_hash, cfg.demo_model)
        llm_latency_ms = (time.perf_counter() - llm_started) * 1000.0
        if cached is None and cfg.demo_mode_default:
            llm_latency_ms = 0.0
        tick_time = time.perf_counter() - started
        first = self.philox_samples[0] if self.philox_samples else 0.0
        self.metrics = MetricsSnapshot(
            tick_time_sec=tick_time,
            gpu_memory_mb=_gpu_memory_mb(),
            llm_latency_ms=llm_latency_ms,
            philox_first=first,
        )
        events = list(self.bus.log)
        self.event_count = len(events)
        self.log_hash = event_log_hash(events)
        if self._log_path is not None and events:
            write_event_log_parquet(events, self._log_path)
        self.offline = offline_ready(self._cache, config=cfg) if self._cache else False
        return []


def scenario_run_fingerprint(scenario: Scenario, event_hash: str) -> str:
    """Stable id combining scenario seed and event-log hash (M28 replay compare)."""
    material = f"{scenario.seed}:{scenario.preset}:{event_hash}"
    return hashlib.sha256(material.encode()).hexdigest()
