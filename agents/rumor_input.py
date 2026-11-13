"""Shared rumor classification for UI and calibration (§5 M29 / M34)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from agents.guardrails import moderate_rumor
from agents.rumor_judge import RUN_CLAIM_TYPES, RumorJudgment, judge_rumor
from agents.rumor_lexicon import (
    BANK_HINTS,
    MACRO_HINTS,
    NAMED_BANK_HINTS,
    count_hints,
    is_market_related,
    lexicon_tone_and_strength,
    normalize,
    specificity_hits,
    strength_from_fear,
)
from params.loader import get_parameter

Tone = Literal["none", "fear", "neutral", "positive"]
Strength = Literal["none", "low", "medium", "high"]
Category = Literal["irrelevant", "market"]
InputClass = Literal[
    "bank_rumor",
    "market_event",
    "macro_event",
    "irrelevant",
    "unsafe",
]

IRRELEVANT_READ_AS = "Not market-related: no effect on the simulation."

_PARAM_POSTERS = "M29 fake poster credibility (ASSUMED)"
_PARAM_DEFAULT = "M29 fake default rumor credibility (ASSUMED)"
_PARAM_FLOOR = "M34 rumor credibility floor"
_PARAM_CAP = "M34 rumor credibility cap"
_PARAM_SHIFT = "M34 rumor model shift max"
_PARAM_SPEC = "M34 rumor specificity weight"
_PARAM_CTA_LEVELS = "M34 rumor judge call-to-action strength levels"
_PARAM_RUN_FLOOR = "M34 rumor judge run-claim minimum strength"
_LEVELS: tuple[Strength, ...] = ("low", "medium", "high")


@dataclass(frozen=True, slots=True)
class CredibilityBreakdown:
    """How the code reached a rumor's credibility (shown to viewers in the UI).

    ``raw = role_base + specificity_bonus + model_shift``; ``credibility`` is
    ``raw`` clamped to ``[floor, cap]``. The model never writes ``credibility``.
    ``specificity`` is the judged 0..1 score (lexicon hits / 4 offline) and
    ``model_shift`` comes from judged plausibility (0 offline).
    """

    role: str | None
    role_known: bool
    role_base: float
    specificity_hits: int
    specificity_bonus: float
    model_shift: float
    raw: float
    floor: float
    cap: float
    credibility: float
    specificity: float = 0.0
    judge_source: str = "lexicon"

    @property
    def clamped(self) -> bool:
        return self.credibility != self.raw

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["clamped"] = self.clamped
        return out


@dataclass(frozen=True, slots=True)
class RumorMarketClassification:
    """Deterministic market class, tone, and code-computed credibility."""

    market_related: bool
    category: Category
    input_class: InputClass
    tone: Tone
    strength: Strength
    credibility: float
    effect_size: float
    affects_simulation: bool
    read_as: str
    breakdown: CredibilityBreakdown | None = None
    judgment: RumorJudgment | None = None
    strength_lexicon: Strength | None = None


def _reg_float(name: str) -> float:
    return float(get_parameter(name).value.split()[0])


def rumor_roles() -> dict[str, float]:
    """Poster roles and their base credibility, in registry order.

    The registry cell is ``id:value,id:value,...``. Every value is ASSUMED: it is
    an ordinal judgement from documented cases, not a measured credibility.
    """
    table: dict[str, float] = {}
    raw = get_parameter(_PARAM_POSTERS).value
    for part in raw.split(","):
        piece = part.strip()
        if ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        table[key.strip()] = float(value.strip())
    return table


def default_role_credibility() -> float:
    return _reg_float(_PARAM_DEFAULT)


def poster_base_credibility(poster: str | None) -> float:
    """Credibility from the who-posts-it selector (registry map)."""
    default = default_role_credibility()
    if not poster:
        return default
    return rumor_roles().get(poster.strip(), default)


def _effect_size_for(tone: Tone, strength: Strength) -> float:
    if tone == "none":
        return 0.0
    if tone == "positive":
        return 0.12
    if tone == "neutral":
        return 0.18
    if strength == "high":
        return 1.0
    if strength == "medium":
        return 0.65
    return 0.35


def _strength_from_fear(score: int) -> Strength:
    return cast(Strength, strength_from_fear(score))


def _input_class(normalized: str, *, unsafe: bool, market_related: bool) -> InputClass:
    if unsafe:
        return "unsafe"
    if not market_related:
        return "irrelevant"
    if count_hints(normalized, BANK_HINTS) > 0:
        return "bank_rumor"
    if count_hints(normalized, MACRO_HINTS) > 0:
        return "macro_event"
    return "market_event"


def _read_as_sentence(
    normalized: str,
    *,
    input_class: InputClass,
    tone: Tone,
    strength: Strength,
) -> str:
    if input_class in {"irrelevant", "unsafe"}:
        return IRRELEVANT_READ_AS
    if tone == "positive":
        return "Read as: market-related, tone calm, strength low."
    if tone == "neutral":
        return "Read as: market-related, tone neutral, strength low."
    bankish = count_hints(normalized, NAMED_BANK_HINTS) > 0 or bool(
        re.search(r"\bbanks?\b", normalized)
    )
    if bankish or input_class == "bank_rumor":
        return f"Read as: bank solvency fear, strength {strength}."
    return f"Read as: general market fear, strength {strength}."


def _clamp_shift(raw: float) -> float:
    limit = _reg_float(_PARAM_SHIFT)
    return max(-limit, min(limit, raw))


def _plausibility_shift(judgment: RumorJudgment, explicit: float | None) -> float:
    """Explicit shift (tests) or judged plausibility mapped to ±shift max; 0 offline."""
    if explicit is not None:
        return _clamp_shift(explicit)
    if judgment.source != "model":
        return 0.0
    return _clamp_shift((judgment.plausibility - 0.5) * 2.0 * _reg_float(_PARAM_SHIFT))


def credibility_breakdown(
    text: str,
    poster: str | None,
    *,
    model_shift: float | None = None,
    judgment: RumorJudgment | None = None,
) -> CredibilityBreakdown:
    """Role base plus judged specificity plus judged plausibility shift, clamped 0.05–0.9.

    With no ``judgment`` the judge runs (lexicon offline). The bonus is written
    ``specificity·4·w/4`` so a lexicon score (hits/4) gives exactly the old
    ``hits·w/4`` for any registry weight.
    """
    normalized = normalize(text.strip())
    role = poster.strip() if poster else None
    roles = rumor_roles()
    base = poster_base_credibility(poster)
    hits = specificity_hits(normalized)
    judged = judgment if judgment is not None else judge_rumor(text)
    bonus = judged.specificity * 4.0 * _reg_float(_PARAM_SPEC) / 4.0
    shift = _plausibility_shift(judged, model_shift)
    raw = base + bonus + shift
    floor = _reg_float(_PARAM_FLOOR)
    cap = _reg_float(_PARAM_CAP)
    return CredibilityBreakdown(
        role=role,
        role_known=role is not None and role in roles,
        role_base=base,
        specificity_hits=hits,
        specificity_bonus=bonus,
        model_shift=shift,
        raw=raw,
        floor=floor,
        cap=cap,
        credibility=max(floor, min(cap, raw)),
        specificity=judged.specificity,
        judge_source=judged.source,
    )


def code_credibility(
    text: str,
    poster: str | None,
    *,
    model_shift: float | None = None,
    judgment: RumorJudgment | None = None,
) -> float:
    """Poster base plus specificity; optional model shift; clamp 0.05–0.9."""
    return credibility_breakdown(
        text, poster, model_shift=model_shift, judgment=judgment
    ).credibility


def judged_strength(lexicon: Strength, tone: Tone, judgment: RumorJudgment) -> Strength:
    """Raise a fear rumor's strength from a *model* judgment; lexicon judgments change nothing.

    A judged call to act adds the registry number of levels. A judged solvency,
    liquidity, or regulatory claim is at least the registry minimum. Never lowers.
    """
    if judgment.source != "model" or tone != "fear" or lexicon not in _LEVELS:
        return lexicon
    level = _LEVELS.index(lexicon)
    if judgment.claim_type in RUN_CLAIM_TYPES:
        floor_name = get_parameter(_PARAM_RUN_FLOOR).value.strip().lower()
        if floor_name in _LEVELS:
            level = max(level, _LEVELS.index(floor_name))
    if judgment.call_to_action:
        level += int(_reg_float(_PARAM_CTA_LEVELS))
    return _LEVELS[max(0, min(len(_LEVELS) - 1, level))]


def prejudge_rumor(text: str) -> RumorJudgment | None:
    """Judge text that would affect the sim; None otherwise. Safe to call off the event loop."""
    stripped = text.strip()
    normalized = normalize(stripped)
    if not stripped or not moderate_rumor(stripped).allowed:
        return None
    if not is_market_related(normalized):
        return None
    return judge_rumor(stripped)


def classify_rumor_market(
    text: str,
    poster: str | None = None,
    *,
    model_shift: float | None = None,
    judgment: RumorJudgment | None = None,
) -> RumorMarketClassification:
    """Shared UI + calibration rule. The model never sets credibility directly."""
    stripped = text.strip()
    normalized = normalize(stripped)
    unsafe = bool(stripped) and not moderate_rumor(stripped).allowed
    market_related = bool(normalized) and not unsafe and is_market_related(normalized)
    input_class = _input_class(normalized, unsafe=unsafe, market_related=market_related)
    affects = input_class in {"bank_rumor", "market_event", "macro_event"}
    if not affects:
        return RumorMarketClassification(
            market_related=False,
            category="irrelevant",
            input_class=input_class,
            tone="none",
            strength="none",
            credibility=0.0,
            effect_size=0.0,
            affects_simulation=False,
            read_as=IRRELEVANT_READ_AS,
        )

    tone_text, strength_text = lexicon_tone_and_strength(normalized)
    tone = cast(Tone, tone_text)
    strength = cast(Strength, strength_text)

    judged = judgment if judgment is not None else judge_rumor(stripped)
    lexicon_strength = strength
    strength = judged_strength(strength, tone, judged)
    breakdown = credibility_breakdown(
        stripped, poster, model_shift=model_shift, judgment=judged
    )
    return RumorMarketClassification(
        market_related=True,
        category="market",
        input_class=input_class,
        tone=tone,
        strength=strength,
        credibility=breakdown.credibility,
        effect_size=_effect_size_for(tone, strength),
        affects_simulation=True,
        read_as=_read_as_sentence(
            normalized,
            input_class=input_class,
            tone=tone,
            strength=strength,
        ),
        breakdown=breakdown,
        judgment=judged,
        strength_lexicon=lexicon_strength,
    )
