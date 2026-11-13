"""Live rumor-box caps: length, merge-by-claim-type, and new-rumor cooldown."""

from __future__ import annotations

from dataclasses import dataclass, field

from params.loader import get_parameter

_PARAM_MAX_CHARS = "M34 rumor max characters"
_PARAM_COOLDOWN = "M34 rumor new cooldown seconds"

TOO_LONG_REASON = "rumor longer than 280 characters"
RATE_LIMIT_REASON = "at most one new rumor per 30 seconds"


def _reg_float(name: str) -> float:
    return float(get_parameter(name).value.split()[0])


def rumor_max_characters() -> int:
    return int(_reg_float(_PARAM_MAX_CHARS))


def rumor_cooldown_seconds() -> float:
    return _reg_float(_PARAM_COOLDOWN)


def rumor_too_long(text: str) -> bool:
    return len(text) > rumor_max_characters()


def too_long_reason() -> str:
    return TOO_LONG_REASON


def rate_limit_reason() -> str:
    return RATE_LIMIT_REASON


@dataclass(frozen=True, slots=True)
class InboxDecision:
    kind: str
    rumor_id: str | None = None
    reason: str | None = None


@dataclass
class RumorInbox:
    """One new rumor per cooldown; repeats of a claim type reuse the first id."""

    _seen: dict[str, str] = field(default_factory=dict)
    _last_new: float | None = None

    def reset(self) -> None:
        self._seen.clear()
        self._last_new = None

    def decide(self, claim_type: str, rumor_id: str, *, now: float) -> InboxDecision:
        existing = self._seen.get(claim_type)
        if existing is not None:
            return InboxDecision(kind="merge", rumor_id=existing)
        cooldown = rumor_cooldown_seconds()
        if self._last_new is not None and now - self._last_new < cooldown:
            return InboxDecision(kind="rate_limited", reason=RATE_LIMIT_REASON)
        self._seen[claim_type] = rumor_id
        self._last_new = now
        return InboxDecision(kind="accept", rumor_id=rumor_id)
