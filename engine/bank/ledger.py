"""M08 balance sheet and accounting — double-entry ledger (Wave B)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import (
    AssetSold,
    CapitalChanged,
    DepositRateChanged,
    LiquidityDrawn,
    LossRealized,
    PaymentSettled,
)
from contracts.schemas import Scenario
from mocks.bus import EventBus
from params.loader import RegistryError, get_parameter

# Opening balance-sheet splits are not registry rows; they mirror the fake fold_book
# bootstrap until M28 ships preset Bank objects (ADR: opening structure).
_OPENING_CASH_FRACTION = 0.001
_OPENING_SECURITIES_FRACTION = 0.004
_DEFAULT_OPENING_CET1 = 0.12
# SVB-56: SVB-like opening book (SOURCED rows), off unless the flag or the live context is on.
_SVB_BOOK_FLAG = "M08 SVB-like opening book enabled"
_SVB_CASH_ROW = "M08 SVB-like opening cash fraction of deposits"
_SVB_SECURITIES_ROW = "M08 SVB-like opening securities fraction of deposits"
_SVB_BOOK_CONTEXT: ContextVar[bool] = ContextVar("stampede_svb_opening_book", default=False)


@contextmanager
def svb_like_opening_book() -> Iterator[None]:
    """Build ledgers with the SVB-like opening book for the duration of the block (live opt-in)."""
    token = _SVB_BOOK_CONTEXT.set(True)
    try:
        yield
    finally:
        _SVB_BOOK_CONTEXT.reset(token)


def svb_opening_book_active() -> bool:
    if _SVB_BOOK_CONTEXT.get():
        return True
    try:
        value = get_parameter(_SVB_BOOK_FLAG).value
    except RegistryError:
        return False
    return str(value).strip().lower() in {"true", "yes", "on", "1"}


def opening_book_fractions() -> tuple[float, float]:
    """(cash, securities) as fractions of deposits: the SVB-like rows when active, else the old slices."""
    if not svb_opening_book_active():
        return _OPENING_CASH_FRACTION, _OPENING_SECURITIES_FRACTION
    return (
        float(get_parameter(_SVB_CASH_ROW).value),
        float(get_parameter(_SVB_SECURITIES_ROW).value),
    )


@dataclass(frozen=True, slots=True)
class LedgerState:
    """Balance sheet snapshot used for CET1 and the accounting invariant."""

    cash: float
    securities: float
    loans: float
    other_assets: float
    deposits: float
    borrowings: float
    other_liabilities: float
    equity: float
    deposits_remaining: float
    aoci_toggle: bool = False
    unrealized_afs_loss: float = 0.0
    pending_realized_loss: float = 0.0

    @property
    def assets(self) -> float:
        return self.cash + self.securities + self.loans + self.other_assets

    @property
    def liabilities(self) -> float:
        return self.deposits + self.borrowings + self.other_liabilities


@dataclass(frozen=True, slots=True)
class JournalEntry:
    debit: str
    credit: str
    amount: float


def _parse_money_cell(text: str) -> float:
    lowered = text.strip().lower().replace(",", "").replace("$", "")
    match = re.match(r"[><=\s]*([0-9]*\.?[0-9]+)\s*([a-z%]*)", lowered)
    if match is None:
        raise ValueError(f"cannot parse registry money value: {text!r}")
    number = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("b"):
        return number * 1_000_000_000.0
    if unit.startswith("m"):
        return number * 1_000_000.0
    return number


def registry_deposits_total() -> float:
    """SVB-like deposit total from §8.1 / params/registry.csv."""
    row = get_parameter("SVB-like preset: deposits")
    return _parse_money_cell(row.value)


def initial_ledger_state(
    *,
    deposits: float,
    opening_cet1_ratio: float = _DEFAULT_OPENING_CET1,
    aoci_toggle: bool = False,
    cash_fraction: float = _OPENING_CASH_FRACTION,
    securities_fraction: float = _OPENING_SECURITIES_FRACTION,
) -> LedgerState:
    """Opening book: cash and securities slices; loans plug assets = liabilities + equity."""
    if deposits <= 0.0:
        raise ValueError("deposits must be positive")
    cash = deposits * cash_fraction
    securities = deposits * securities_fraction
    equity = deposits * opening_cet1_ratio
    loans = deposits + equity - cash - securities
    if loans < 0.0:
        raise ValueError("opening balance sheet would have negative loans")
    return LedgerState(
        cash=cash,
        securities=securities,
        loans=loans,
        other_assets=0.0,
        deposits=deposits,
        borrowings=0.0,
        other_liabilities=0.0,
        equity=equity,
        deposits_remaining=deposits,
        aoci_toggle=aoci_toggle,
        unrealized_afs_loss=0.0,
        pending_realized_loss=0.0,
    )


def invariant_holds(state: LedgerState, *, tolerance: float = 1.0) -> bool:
    """assets + pending loss ≈ liabilities + equity (§5 M08, §10.3 gate 4)."""
    gap = (state.assets + state.pending_realized_loss) - (
        state.liabilities + state.equity
    )
    return abs(gap) <= tolerance


def cet1_ratio(state: LedgerState) -> float:
    """CET1 with the AOCI toggle (§5 M08)."""
    rwa = state.deposits if state.deposits > 0.0 else 1.0
    equity = state.equity
    if state.aoci_toggle:
        equity -= state.unrealized_afs_loss
    return max(0.0, equity / rwa)


def _apply(
    state: LedgerState,
    *,
    debit: str,
    credit: str,
    amount: float,
) -> LedgerState:
    if amount < 0.0:
        raise ValueError("posting amount must be non-negative")
    if amount == 0.0:
        return state
    data = {
        "cash": state.cash,
        "securities": state.securities,
        "loans": state.loans,
        "other_assets": state.other_assets,
        "deposits": state.deposits,
        "borrowings": state.borrowings,
        "other_liabilities": state.other_liabilities,
        "equity": state.equity,
        "deposits_remaining": state.deposits_remaining,
        "aoci_toggle": state.aoci_toggle,
        "unrealized_afs_loss": state.unrealized_afs_loss,
        "pending_realized_loss": state.pending_realized_loss,
    }

    def dec(account: str) -> None:
        data[account] -= amount

    def inc(account: str) -> None:
        data[account] += amount

    asset_accounts = {"cash", "securities", "loans", "other_assets"}
    liability_accounts = {"deposits", "borrowings", "other_liabilities"}

    if debit in asset_accounts:
        inc(debit)
    elif debit == "equity":
        inc(debit)
    elif debit in liability_accounts:
        dec(debit)
    else:
        raise KeyError(debit)

    if credit in asset_accounts:
        dec(credit)
    elif credit == "equity":
        dec(credit)
    elif credit in liability_accounts:
        inc(credit)
    else:
        raise KeyError(credit)

    new_state = LedgerState(
        cash=float(data["cash"]),
        securities=float(data["securities"]),
        loans=float(data["loans"]),
        other_assets=float(data["other_assets"]),
        deposits=float(data["deposits"]),
        borrowings=float(data["borrowings"]),
        other_liabilities=float(data["other_liabilities"]),
        equity=float(data["equity"]),
        deposits_remaining=float(data["deposits_remaining"]),
        aoci_toggle=bool(data["aoci_toggle"]),
        unrealized_afs_loss=float(data["unrealized_afs_loss"]),
        pending_realized_loss=float(data["pending_realized_loss"]),
    )
    if not invariant_holds(new_state):
        raise ValueError(
            f"posting {debit}/{credit} {amount} broke invariant "
            f"(assets={new_state.assets}, liab+eq={new_state.liabilities + new_state.equity})"
        )
    return new_state


def post_payment_settled(state: LedgerState, amount: float) -> LedgerState:
    """Withdrawal settlement: cash out, deposit liability down."""
    state = _apply(state, debit="deposits", credit="cash", amount=amount)
    remaining = max(0.0, state.deposits_remaining - amount)
    return LedgerState(
        cash=state.cash,
        securities=state.securities,
        loans=state.loans,
        other_assets=state.other_assets,
        deposits=state.deposits,
        borrowings=state.borrowings,
        other_liabilities=state.other_liabilities,
        equity=state.equity,
        deposits_remaining=remaining,
        aoci_toggle=state.aoci_toggle,
        unrealized_afs_loss=state.unrealized_afs_loss,
        pending_realized_loss=state.pending_realized_loss,
    )


def post_liquidity_drawn(state: LedgerState, amount: float) -> LedgerState:
    """Funding draw: cash in, borrowings up."""
    return _apply(state, debit="cash", credit="borrowings", amount=amount)


def post_loss_realized(state: LedgerState, amount: float) -> LedgerState:
    """Hit equity for realized loss once M09 publishes LossRealized."""
    if amount <= 0.0:
        return state
    from_pending = min(amount, state.pending_realized_loss)
    remainder = amount - from_pending
    if remainder > state.loans:
        raise ValueError("realized loss exceeds loan assets available to absorb it")
    new_state = LedgerState(
        cash=state.cash,
        securities=state.securities,
        loans=state.loans - remainder,
        other_assets=state.other_assets,
        deposits=state.deposits,
        borrowings=state.borrowings,
        other_liabilities=state.other_liabilities,
        equity=state.equity - amount,
        deposits_remaining=state.deposits_remaining,
        aoci_toggle=state.aoci_toggle,
        unrealized_afs_loss=state.unrealized_afs_loss,
        pending_realized_loss=state.pending_realized_loss - from_pending,
    )
    if not invariant_holds(new_state):
        raise ValueError("loss posting broke accounting invariant")
    return new_state


def post_asset_sale(
    state: LedgerState,
    *,
    book_amount: float,
    price_impact: float,
) -> LedgerState:
    """Sell AFS; loss stays pending until M09's LossRealized (same run, later tick)."""
    impact = min(max(price_impact, 0.0), 1.0)
    cash_in = book_amount * (1.0 - impact)
    loss = book_amount * impact
    sold = min(book_amount, state.securities)
    new_state = LedgerState(
        cash=state.cash + cash_in,
        securities=max(0.0, state.securities - sold),
        loans=state.loans,
        other_assets=state.other_assets,
        deposits=state.deposits,
        borrowings=state.borrowings,
        other_liabilities=state.other_liabilities,
        equity=state.equity,
        deposits_remaining=state.deposits_remaining,
        aoci_toggle=state.aoci_toggle,
        unrealized_afs_loss=state.unrealized_afs_loss,
        pending_realized_loss=state.pending_realized_loss + loss,
    )
    if not invariant_holds(new_state):
        raise ValueError("asset sale broke accounting invariant")
    return new_state


def policy_rate_at(path: list[tuple[float, float]], t: float) -> float:
    if not path:
        return 0.045
    if t <= path[0][0]:
        return path[0][1]
    for (t0, r0), (t1, _r1) in zip(path, path[1:], strict=False):
        if t0 <= t <= t1 and t1 > t0:
            weight = (t - t0) / (t1 - t0)
            return r0 + (_r1 - r0) * weight
    return path[-1][1]


def securities_sale_impact(policy_rate: float) -> float:
    """Monotone markdown used when the ledger must sell AFS (aligned with M09 fake)."""
    return min(max((policy_rate - 0.01) * 4.0, 0.0), 0.45)


def bank_id_for(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "bank-community"
    if "crypto" in lowered:
        return "bank-crypto-fictional"
    return "bank-fvb"


class LedgerModule:
    """M08 double-entry ledger (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("AssetSold", "CapitalChanged")
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "PaymentSettled",
        "LiquidityDrawn",
        "LossRealized",
        "DepositRateChanged",
    )

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0
        self.state: LedgerState | None = None
        self.deposit_rates: dict[str, float] = {}
        self.last_cet1 = _DEFAULT_OPENING_CET1
        self.journal: list[JournalEntry] = []

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        cash_fraction, securities_fraction = opening_book_fractions()
        self.state = initial_ledger_state(
            deposits=registry_deposits_total(),
            cash_fraction=cash_fraction,
            securities_fraction=securities_fraction,
        )
        self.last_cet1 = cet1_ratio(self.state)
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None or self.state is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        out = self._on_tick(t, new_events)
        self.published.extend(out)
        return out

    def _on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        state = self.state
        assert state is not None
        for event in new_events:
            if isinstance(event, DepositRateChanged):
                self.deposit_rates[event.segment] = event.rate
            elif isinstance(event, PaymentSettled) and event.amount > 0.0:
                state = post_payment_settled(state, event.amount)
                self.journal.append(
                    JournalEntry("deposits", "cash", event.amount),
                )
            elif isinstance(event, LiquidityDrawn) and event.amount > 0.0:
                state = post_liquidity_drawn(state, event.amount)
                self.journal.append(
                    JournalEntry("cash", "borrowings", event.amount),
                )
            elif isinstance(event, LossRealized) and event.amount > 0.0:
                state = post_loss_realized(state, event.amount)
                self.journal.append(JournalEntry("equity", "pending_loss", event.amount))
        self.state = state
        if not invariant_holds(state):
            raise RuntimeError(f"{self.module_id}: accounting invariant failed at t={t}")

        out: list[BaseModel] = []
        scenario = self.scenario
        assert scenario is not None
        rate = policy_rate_at(scenario.rate_path, t)
        cet1 = cet1_ratio(state)
        if cet1 + 1e-12 < self.last_cet1:
            out.append(
                CapitalChanged(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause("cet1"),
                    cet1_ratio=cet1,
                )
            )
            self.last_cet1 = cet1

        if state.cash < -1.0 and state.securities > 1.0 and not self._bonds_pledged():
            impact = securities_sale_impact(rate)
            proceed = max(0.05, 1.0 - impact)
            book_sold = min(state.securities, (-state.cash) / proceed)
            if book_sold > 1.0:
                self.state = post_asset_sale(
                    state,
                    book_amount=book_sold,
                    price_impact=impact,
                )
                out.append(
                    AssetSold.model_validate(
                        {
                            "t": t,
                            "seed": self.seed,
                            "bank_id": self.bank_id,
                            "cause_id": self.cause("sale"),
                            "class": "afs",
                            "amount": book_sold,
                            "price_impact": impact,
                        }
                    )
                )
        return out

    def _bonds_pledged(self) -> bool:
        """AU-1: bonds pledged at the discount window back the bank's borrowing, so the ledger can't also sell them."""
        from mocks.support import levers_in

        return self.bus is not None and "pledge_collateral" in levers_in(self.bus.log)

    @property
    def seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return self.scenario.seed

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_id_for(self.scenario.preset)

    def cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"
