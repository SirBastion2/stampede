"""UF-1: what the bank can pay its depositors by the next open, from the SVB-like balance sheet (10-K FY2022).

One rule for the live regulator (M20, behind ``M20 closes on what the bank can pay``) and the usefulness ranking
(``ui/live/useful.py``), so the wall and the live run agree. A bank fails at the next open when depositors have
asked for more than it can pay without selling bonds at a loss that wipes out its capital:

  with nothing done : opening cash + what it could borrow in 2023 (the old repo/FHLB/discount rungs, 2.6% of
                      deposits, about $4.6B; SVB borrowed $5.3B on Mar 9) + bonds it can sell near book value
                      (AFS at fair value, $26.1B). About 25.5% of deposits. Selling the HTM bonds would lock in their
                      $15.1B loss against $16.3B of equity.
  with the Fed's 2023 program on before that point: opening cash + the line (every bond at face value, the CM-2
                      pool) + repo and FHLB. About 77.6%.
  UF-2, tools that change what the bank can pay but not what depositors do (so the usefulness ranking computes them
  from the no-response runs; the live engine doesn't simulate them yet):
  bonds pledged in advance: opening cash + every bond at market value times the Fed's margin + repo and FHLB.
                      About 65.3%.
  deposits from big banks : $30B more, as 11 banks gave First Republic on Mar 16 2023, if it arrives before the bank
                      runs short. About 17.1 points.
"""

from __future__ import annotations

from dataclasses import dataclass

from params.loader import get_parameter

CASH_ROW = "M08 SVB-like opening cash fraction of deposits"
LINE_ROW = "M08 SVB-like opening securities fraction of deposits"
AFS_ROW = "M20 bonds sellable near book value fraction of deposits"
HTM_ROW = "M20 HTM bonds fair value fraction of deposits"
MARGIN_ROW = "M20 discount window margin on pledged bonds"
BIG_BANKS_ROW = "M20 deposits from big banks fraction of deposits"


@dataclass(frozen=True, slots=True)
class CanPay:
    none: float   # nothing done
    fed: float    # with the Fed's 2023 program on in time
    prepared: float = 0.0   # UF-2: bonds pledged at the discount window in advance
    big_banks: float = 0.0  # UF-2: deposits from big banks, added when they arrive in time


def can_pay_fractions() -> CanPay:
    from mocks.support import DISCOUNT_TOTAL_FRACTION, FHLB_TOTAL_FRACTION, REPO_TOTAL_FRACTION

    cash = float(get_parameter(CASH_ROW).value)
    afs = float(get_parameter(AFS_ROW).value)
    line = float(get_parameter(LINE_ROW).value)
    ladder = REPO_TOTAL_FRACTION + FHLB_TOTAL_FRACTION + DISCOUNT_TOTAL_FRACTION
    other = REPO_TOTAL_FRACTION + FHLB_TOTAL_FRACTION
    pledged = (afs + float(get_parameter(HTM_ROW).value)) * float(get_parameter(MARGIN_ROW).value)
    return CanPay(none=cash + ladder + afs, fed=cash + line + other, prepared=cash + pledged + other,
                  big_banks=float(get_parameter(BIG_BANKS_ROW).value))


def capacity(line_minute: float | None, past_none_minute: float | None, fr: CanPay | None = None) -> float:
    """The line helps only if it is on before demand passes what the bank could pay without it."""
    fr = fr or can_pay_fractions()
    if line_minute is not None and (past_none_minute is None or line_minute <= past_none_minute):
        return fr.fed
    return fr.none
