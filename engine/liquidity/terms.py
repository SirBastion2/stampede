"""CM-2: the emergency credit line's terms (ADR-122).

PA-A's line is assumed: a pool of 5% of deposits, 1% a minute, covering 80% of each minute's cash
shortfall as M12's old 0.1% cash slice measures it. With ``M12 emergency line on BTFP terms`` on (the
live server opts in through ``M29 live emergency line on BTFP terms``), the line takes the 2023 Bank
Term Funding Program's terms: it lends up to the par value of the bank's bonds with no haircut, so
the pool is the opening securities (M08's SVB-like row: AFS at fair value plus HTM at amortized
cost, a slight underestimate of par) and it covers the whole shortfall; once it is on, the shortfall
is measured against M08's own opening cash. Read the terms while the engine is built, inside the
live context, as M08 reads its opening book.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.bank.ledger import opening_book_fractions
from engine.live_overrides import flag_enabled
from params.loader import RegistryError, get_parameter

BTFP_FLAG = "M12 emergency line on BTFP terms"
BTFP_SHARE_ROW = "M12 BTFP line shortfall share"
BTFP_RAMP_ROW = "M12 BTFP line ramp fraction of deposits per minute"


@dataclass(frozen=True, slots=True)
class EmergencyTerms:
    """Pool and per-minute ramp as fractions of deposits, the share of a shortfall covered, and the cash
    the shortfall is measured from (None: M12's old slice, as before CM-2)."""

    total_fraction: float
    tick_fraction: float
    shortfall_share: float
    opening_cash_fraction: float | None = None

    @property
    def btfp(self) -> bool:
        return self.opening_cash_fraction is not None


ASSUMED_TERMS = EmergencyTerms(0.05, 0.01, 0.8)


def emergency_terms() -> EmergencyTerms:
    if not flag_enabled(BTFP_FLAG):
        return ASSUMED_TERMS
    try:
        share = float(get_parameter(BTFP_SHARE_ROW).value)
        ramp = float(get_parameter(BTFP_RAMP_ROW).value)
    except (RegistryError, ValueError):
        return ASSUMED_TERMS
    cash_fraction, securities_fraction = opening_book_fractions()
    return EmergencyTerms(securities_fraction, ramp, share, cash_fraction)
