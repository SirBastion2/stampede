"""Offline rumor hint lists and scoring (shared by classify_rumor_market)."""

from __future__ import annotations

import re

MARKET_HINTS: tuple[str, ...] = (
    "stock",
    "stocks",
    "share price",
    "shares",
    "equity",
    "equities",
    "market",
    "trading",
    "nasdaq",
    "dow",
    "s&p",
    "sp500",
    "bond",
    "bonds",
    "yield",
    "interest rate",
    "interest rates",
    "fed",
    "federal reserve",
    "treasury",
    "liquidity",
    "solvency",
    "capital",
    "capitalized",
    "reserve",
    "reserves",
    "deposit",
    "deposits",
    "depositor",
    "depositors",
    "uninsured",
    "withdraw",
    "withdrawal",
    "withdrawals",
    "wire",
    "bank run",
    "run on",
    "outflow",
    "insolvent",
    "bankrupt",
    "bankruptcy",
    "default",
    "collapse",
    "crashed",
    "crash",
    "meltdown",
    "panic",
    "recession",
    "inflation",
    "hedge",
    "crypto",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "commodity",
    "commodities",
    "oil",
    "gold",
    "silver",
    "gamestop",
    "gme",
    "tesla",
    "portfolio",
    "margin",
    "forex",
    "currency",
    "cet1",
    "balance sheet",
    "etf",
    "ipo",
    "earnings",
    "dividend",
    "kestrel",
    "fictional valley",
    "fictional harbor",
    "fictional community",
    "pulling their money",
    "out of cash",
    "lost billions",
    "bank",
    "banks",
    "money",
    "funds",
    "accounts",
    "payroll",
    "venture",
    "startups",
    "startup",
    "vcs",
    "banco",
    "bancos",
    "dinero",
    "fondos",
    "cuentas",
    "nomina",
    "nómina",
    "depositos",
    "depósitos",
    "retirar",
    "retiro",
    "quebrar",
    "quiebra",
    "insolvente",
    "liquidez",
    "corrida",
    "pánico",
    "panico",
)

FEAR_HINTS: tuple[str, ...] = (
    "crash",
    "crashing",
    "drop",
    "dropped",
    "falling",
    "fell",
    "plunge",
    "plunged",
    "bankrupt",
    "insolvency",
    "insolvent",
    "panic",
    "run",
    "out of cash",
    "collaps",
    "crisis",
    "fear",
    "worry",
    "doom",
    "fail",
    "failing",
    "trouble",
    "bleeding",
    "tank",
    "tanking",
    "meltdown",
    "leaving",
    "question",
    "delay",
    "may delay",
    "hears",
    "unsure",
    "concern",
    "pulling",
    "pull",
    "quebrar",
    "quiebra",
    "insolvente",
    "pánico",
    "panico",
    "corrida",
    "retirar",
)

POSITIVE_HINTS: tuple[str, ...] = (
    "well capitalized",
    "capitalized",
    "strong",
    "safe",
    "stable",
    "rallied",
    "rally",
    "surge",
    "record high",
    "healthy",
    "sound",
    "resilient",
    "recovered",
    "solid",
)

STRONG_FEAR_HINTS: frozenset[str] = frozenset(
    {
        "bankrupt",
        "insolvent",
        "bankruptcy",
        "panic",
        "crash",
        "crashing",
        "meltdown",
        "out of cash",
        "run on",
        "bank run",
        "quebrar",
        "quiebra",
        "insolvente",
    }
)

LEISURE_ONLY_HINTS: tuple[str, ...] = (
    "weather",
    "lakers",
    "nba",
    "nfl",
    "touchdown",
    "world cup",
    "celebrity",
    "gossip",
    "hello",
    "good morning",
    "nice day",
    "won last night",
    "soccer",
    "baseball",
    "llover",
)

BANK_HINTS: tuple[str, ...] = (
    "bank",
    "banks",
    "banco",
    "bancos",
    "deposit",
    "deposits",
    "depositor",
    "depositors",
    "uninsured",
    "withdraw",
    "withdrawal",
    "wire",
    "wires",
    "bank run",
    "run on",
    "kestrel",
    "fictional valley",
    "fictional harbor",
    "fictional community",
    "solvency",
    "payroll",
    "startup",
    "startups",
    "vcs",
    "venture",
    "pulling their money",
    "out of cash",
    "quebrar",
    "quiebra",
    "dinero",
    "fondos",
    "cuentas",
    "nomina",
    "nómina",
    "depositos",
    "depósitos",
    "retirar",
    "corrida",
    "move their money",
)

MACRO_HINTS: tuple[str, ...] = (
    "fed",
    "federal reserve",
    "inflation",
    "recession",
    "interest rate",
    "interest rates",
    "unemployment",
    "policy rate",
    "rate hike",
    "cut rates",
)

NAMED_BANK_HINTS: tuple[str, ...] = (
    "kestrel",
    "fictional valley",
    "fictional harbor",
    "fictional community",
)

SPECIFICITY_TIME_HINTS: tuple[str, ...] = (
    "this week",
    "this morning",
    "today",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "hours",
)

SPECIFICITY_CHANNEL_HINTS: tuple[str, ...] = (
    "uninsured",
    "wire",
    "wires",
    "payroll",
    "vcs",
    "startup",
    "startups",
    "venture",
)

_INJECTION_HINTS: tuple[str, ...] = (
    "credibl",
    "rate it",
    "ignore your instruction",
    "set credibility",
)


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def count_hints(normalized: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in hints if hint in normalized)


def is_market_related(normalized: str) -> bool:
    if count_hints(normalized, MARKET_HINTS) > 0:
        return True
    if count_hints(normalized, LEISURE_ONLY_HINTS) > 0:
        return False
    if re.fullmatch(r"[a-z?¿!.\s]{0,40}", normalized) and len(normalized) <= 24:
        tokens = normalized.split()
        if len(tokens) <= 3 and not any(ch.isdigit() for ch in normalized):
            return False
    return False


def fear_score(normalized: str) -> int:
    score = count_hints(normalized, FEAR_HINTS)
    for strong in STRONG_FEAR_HINTS:
        if strong in normalized:
            score += 2
    return score


def strength_from_fear(score: int) -> str:
    """Fear-word score to strength: 5+ high, 2+ medium, else low."""
    if score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def lexicon_tone_and_strength(normalized: str) -> tuple[str, str]:
    """The offline tone and strength rule, shared by M34 classification and M03.

    More positive than fear hints: ("positive", "low"). Any fear: ("fear",
    strength_from_fear). Otherwise ("neutral", "low").
    """
    fear = fear_score(normalized)
    positive = count_hints(normalized, POSITIVE_HINTS)
    if positive > fear and positive > 0:
        return "positive", "low"
    if fear > 0:
        return "fear", strength_from_fear(fear)
    return "neutral", "low"


def looks_like_injection(normalized: str) -> bool:
    return count_hints(normalized, _INJECTION_HINTS) > 0


def specificity_hits(normalized: str) -> int:
    hits = 0
    if count_hints(normalized, NAMED_BANK_HINTS) > 0:
        hits += 1
    if count_hints(normalized, SPECIFICITY_TIME_HINTS) > 0:
        hits += 1
    if count_hints(normalized, SPECIFICITY_CHANNEL_HINTS) > 0:
        hits += 1
    if not looks_like_injection(normalized) and re.search(
        r"\$|\b\d+(?:\.\d+)?\s*(?:billion|million|percent|%)\b",
        normalized,
    ):
        hits += 1
    return hits


# M34 judge (SVB-40): offline claim-type and call-to-action hints. These only
# label the lexicon judgment; offline credibility and strength do not use them.
CALL_TO_ACTION_HINTS: tuple[str, ...] = (
    "pull your money",
    "pull their money",
    "pull out now",
    "withdraw now",
    "withdraw your",
    "withdraw everything",
    "move your money",
    "move their money",
    "get your money out",
    "get out now",
    "take your money out",
    "sell now",
    "saquen",
    "retiren",
    "saca tu dinero",
)

LIQUIDITY_CLAIM_HINTS: tuple[str, ...] = (
    "out of cash",
    "cover withdrawals",
    "withdrawal",
    "withdrawals",
    "wire",
    "wires",
    "liquidity",
    "halt",
    "frozen",
    "freeze",
    "delay",
    "liquidez",
)

SOLVENCY_CLAIM_HINTS: tuple[str, ...] = (
    "insolvent",
    "insolvency",
    "bankrupt",
    "bankruptcy",
    "solvency",
    "lost billions",
    "losses",
    "capital",
    "cet1",
    "underwater",
    "collaps",
    "fail",
    "quiebra",
    "quebrar",
    "insolvente",
)

REGULATORY_CLAIM_HINTS: tuple[str, ...] = (
    "fdic",
    "regulator",
    "regulators",
    "receivership",
    "seized",
    "seize",
    "shut down",
    "supervisor",
)


def has_call_to_action(normalized: str) -> bool:
    return count_hints(normalized, CALL_TO_ACTION_HINTS) > 0


def claim_type_hint(normalized: str) -> str:
    """Lexicon claim type: regulatory, liquidity, solvency, market, or irrelevant."""
    if not is_market_related(normalized):
        return "irrelevant"
    if count_hints(normalized, REGULATORY_CLAIM_HINTS) > 0:
        return "regulatory"
    if count_hints(normalized, LIQUIDITY_CLAIM_HINTS) > 0:
        return "liquidity"
    if count_hints(normalized, SOLVENCY_CLAIM_HINTS) > 0:
        return "solvency"
    return "market"
