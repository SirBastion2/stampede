"""M28 preset catalog: fictional Bank objects and preset metadata (§12.1)."""

from __future__ import annotations

import re
from functools import lru_cache

from contracts.schemas import (
    Bank,
    BankAssets,
    BankCapital,
    BankLiabilities,
    LadderStep,
    Security,
)
from engine.agents.population import segment_weights_for_preset
from engine.bank.ledger import initial_ledger_state, registry_deposits_total
from engine.bank.securities import BondBucket, initial_buckets
from params.loader import get_parameter

_PARAM_OPENING_CET1 = "M28 opening CET1 ratio"

_PRESET_REGISTRY_NAMES: dict[str, str] = {
    "tech-concentrated": "Preset: tech-concentrated bank (SVB-like)",
    "crypto-heavy": "Preset: crypto-heavy follower bank (Signature-like)",
    "insured-community": "Preset: insured community bank",
}

_PRESET_ALIASES: dict[str, str] = {
    "tech": "tech-concentrated",
    "svb-like": "tech-concentrated",
    "crypto": "crypto-heavy",
    "signature-like": "crypto-heavy",
    "community": "insured-community",
    "placebo": "insured-community",
}

_BANK_META: dict[str, tuple[str, str, str]] = {
    "tech-concentrated": ("bank-fvb", "Fictional Valley Bank", "FVB"),
    "crypto-heavy": ("bank-crypto-fictional", "Fictional Harbor Bank", "HRB"),
    "insured-community": ("bank-community", "Fictional Community Bank", "FCB"),
}


def normalize_preset_key(preset: str) -> str:
    """Map user-facing preset strings to canonical keys."""
    key = preset.strip().lower().replace(" ", "-").replace("_", "-")
    return _PRESET_ALIASES.get(key, key)


def list_preset_keys() -> tuple[str, ...]:
    return tuple(_PRESET_REGISTRY_NAMES)


@lru_cache(maxsize=1)
def _opening_cet1() -> float:
    row = get_parameter(_PARAM_OPENING_CET1)
    match = re.search(r"0?\.\d+", row.value.strip())
    if match is None:
        raise ValueError(f"cannot parse opening CET1 from registry: {row.value!r}")
    return float(match.group(0))


def _securities_to_schema(buckets: list[BondBucket]) -> list[Security]:
    out: list[Security] = []
    for bucket in buckets:
        out.append(
            Security.model_validate(
                {
                    "class": bucket.asset_class,
                    "maturity_bucket": bucket.maturity_bucket,
                    "duration": bucket.duration,
                    "book_value": bucket.book_value,
                    "market_value": bucket.market_value,
                    "htm": bucket.htm,
                }
            )
        )
    return out


def preset_description(preset_key: str) -> str:
    """Human-readable preset line from the parameter registry."""
    canonical = normalize_preset_key(preset_key)
    name = _PRESET_REGISTRY_NAMES.get(canonical)
    if name is None:
        raise KeyError(f"unknown preset: {preset_key!r}")
    return get_parameter(name).value


def bank_for_preset(preset_key: str) -> Bank:
    """Build the opening ``Bank`` object for a preset (§5 M28, ADR-026)."""
    canonical = normalize_preset_key(preset_key)
    if canonical not in _PRESET_REGISTRY_NAMES:
        raise KeyError(f"unknown preset: {preset_key!r}")
    bank_id, display_name, ticker = _BANK_META[canonical]
    deposits = registry_deposits_total()
    ledger = initial_ledger_state(deposits=deposits, opening_cet1_ratio=_opening_cet1())
    weights = segment_weights_for_preset(canonical)
    deposits_by_segment = {segment: deposits * share for segment, share in weights.items()}
    securities = _securities_to_schema(initial_buckets(canonical))
    return Bank(
        id=bank_id,
        name=display_name,
        ticker=ticker,
        preset=canonical,
        assets=BankAssets(
            cash=ledger.cash,
            reserves=0.0,
            securities=securities,
            loans=[{"book": ledger.loans}],
            other=ledger.other_assets,
        ),
        liabilities=BankLiabilities(
            deposits_by_segment=deposits_by_segment,
            fhlb=0.0,
            brokered=0.0,
            other=0.0,
        ),
        capital=BankCapital(cet1=ledger.equity / deposits if deposits else 0.0, aoci_toggle=False),
        ladder=[
            LadderStep(source="cash", capacity=ledger.cash, eta_hours=0.0),
            LadderStep(source="fhlb", capacity=deposits * 0.01, eta_hours=4.0),
        ],
        stock_ref=ticker,
    )
