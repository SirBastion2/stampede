"""M34 safety and content guardrails — Wave B (§5, §19)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from contracts.schemas import Scenario
from params.loader import RegistryError, get_parameter

_PARAM_BANK_NAMES = "M34 blocklist bank names"
_PARAM_TICKERS = "M34 blocklist tickers"
_PARAM_FUZZY_DISTANCE = "M34 fuzzy match max edit distance"
_PARAM_LIVE_ATTENTION = "M34 live attention mode default"
_PARAM_SLUR_TOKENS = "M34 rumor moderation slur tokens"
_PARAM_REAL_PEOPLE = "M34 rumor moderation real-person names"


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    bank_names: tuple[str, ...]
    tickers: frozenset[str]
    fuzzy_max_distance: int
    live_attention_default: bool
    slur_tokens: frozenset[str]
    real_person_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModerationResult:
    allowed: bool
    reasons: tuple[str, ...]


def _split_registry_list(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.replace("|", ";").split(";")]
    return tuple(part for part in parts if part)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "1", "on"}:
        return True
    if lowered in {"false", "no", "0", "off"}:
        return False
    raise RegistryError(f"cannot parse boolean registry value: {value!r}")


def _parse_int(value: str) -> int:
    match = re.search(r"-?\d+", value.strip())
    if match is None:
        raise RegistryError(f"cannot parse integer registry value: {value!r}")
    return int(match.group(0))


def load_guardrail_config(path: Path | None = None) -> GuardrailConfig:
    """Load M34 parameters from the registry (§8.1)."""
    bank_names = _split_registry_list(get_parameter(_PARAM_BANK_NAMES, path).value)
    tickers = frozenset(
        token.lower() for token in _split_registry_list(get_parameter(_PARAM_TICKERS, path).value)
    )
    fuzzy = _parse_int(get_parameter(_PARAM_FUZZY_DISTANCE, path).value)
    live = _parse_bool(get_parameter(_PARAM_LIVE_ATTENTION, path).value)
    slurs = frozenset(
        token.lower()
        for token in _split_registry_list(get_parameter(_PARAM_SLUR_TOKENS, path).value)
    )
    people = _split_registry_list(get_parameter(_PARAM_REAL_PEOPLE, path).value)
    return GuardrailConfig(
        bank_names=bank_names,
        tickers=tickers,
        fuzzy_max_distance=max(0, fuzzy),
        live_attention_default=live,
        slur_tokens=slurs,
        real_person_names=people,
    )


def _config(config: GuardrailConfig | None) -> GuardrailConfig:
    return config if config is not None else load_guardrail_config()


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _fuzzy_hits_phrase(text: str, phrase: str, max_distance: int) -> bool:
    normalized = _normalize(text)
    target = _normalize(phrase)
    if not target:
        return False
    words = normalized.split()
    target_words = target.split()
    # A short single token ("chase") must be a whole word. Edit distance 2
    # would also block ordinary words such as "share".
    if len(target_words) == 1 and len(target) <= 6:
        return target in words
    if target in normalized:
        return True
    if max_distance <= 0:
        return False
    width = len(target_words)
    if width == 0:
        return False
    for index in range(0, max(1, len(words) - width + 1)):
        window = " ".join(words[index : index + width])
        if _levenshtein(window, target) <= max_distance:
            return True
    return False


def _blocked_bank_reference(text: str, config: GuardrailConfig) -> str | None:
    for name in config.bank_names:
        if _fuzzy_hits_phrase(text, name, config.fuzzy_max_distance):
            return name
    return None


def _blocked_ticker_reference(text: str, config: GuardrailConfig) -> str | None:
    tokens = set(_normalize(text).split())
    for ticker in config.tickers:
        if ticker in tokens:
            return ticker
    return None


def bank_name_allowed(name: str, *, config: GuardrailConfig | None = None) -> bool:
    """Return whether a bank display name is fictional (§19)."""
    cfg = _config(config)
    return _blocked_bank_reference(name, cfg) is None


def ticker_allowed(ticker: str, *, config: GuardrailConfig | None = None) -> bool:
    """Return whether a ticker is not on the real-institution blocklist (§19)."""
    cfg = _config(config)
    token = ticker.strip().lower()
    return token not in cfg.tickers


def claim_allowed(claim: str, *, config: GuardrailConfig | None = None) -> bool:
    """Rumor-box filter: fictional institutions only (§19)."""
    return moderate_rumor(claim, config=config).allowed


def statement_allowed(text: str, *, config: GuardrailConfig | None = None) -> bool:
    """Statement text must stay fictional and moderated (§19)."""
    cfg = _config(config)
    if _blocked_bank_reference(text, cfg) is not None:
        return False
    if _blocked_ticker_reference(text, cfg) is not None:
        return False
    return moderate_rumor(text, config=cfg).allowed


def moderate_rumor(text: str, *, config: GuardrailConfig | None = None) -> ModerationResult:
    """Moderate rumor text: blocklist, slurs, and named real people (§19)."""
    cfg = _config(config)
    reasons: list[str] = []
    bank = _blocked_bank_reference(text, cfg)
    if bank is not None:
        reasons.append(f"blocked bank name: {bank}")
    ticker = _blocked_ticker_reference(text, cfg)
    if ticker is not None:
        reasons.append(f"blocked ticker: {ticker}")
    lowered = _normalize(text)
    for slur in cfg.slur_tokens:
        if slur and slur in lowered:
            reasons.append("blocked slur token")
            break
    for person in cfg.real_person_names:
        if person and _fuzzy_hits_phrase(text, person, cfg.fuzzy_max_distance):
            reasons.append(f"blocked real-person name: {person}")
            break
    return ModerationResult(allowed=not reasons, reasons=tuple(reasons))


def live_attention_enabled(*, config: GuardrailConfig | None = None) -> bool:
    """Optional live-attention UI mode; off by default (§5 M34)."""
    return _config(config).live_attention_default


class GuardrailModule:
    """M34 scheduler module: scans scenario rumors once; publishes nothing (§19)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick",)
    publishes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, module_id: str = "M34", *, config: GuardrailConfig | None = None) -> None:
        self.module_id = module_id
        self._config = config
        self.scenario: Scenario | None = None
        self.bus: Any = None
        self.blocked: list[str] = []
        self._scanned = False
        self.inbox: list[BaseModel] = []
        self._cursor = 0

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
        del t, new_events
        if self._scanned or self.scenario is None:
            return []
        self._scanned = True
        for rumor in self.scenario.rumors:
            if not claim_allowed(rumor.claim, config=self._config):
                self.blocked.append(rumor.id)
        return []
