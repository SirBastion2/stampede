"""L4 fakes: closure rule and policy levers."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from contracts.events import BankClosed, PolicyEnacted, SupervisoryAction
from engine.system.regulators import (
    closure_reason,
    load_regulator_config,
    watch_recommended,
)
from mocks.base import FakeModule
from mocks.support import (
    canonical_lever,
    clamp,
    deposits_base,
    fold_book,
)


class RegulatorFake(FakeModule):
    """M20. Closes a bank that cannot pay, or whose CET1 falls through an assumed floor.

    """

    publishes: ClassVar[tuple[str, ...]] = ("BankClosed", "SupervisoryAction")
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "CapitalChanged", "PolicyEnacted")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.status = "open"
        self.closed = False
        self._cfg = load_regulator_config()
        # SVB-64: the live server holds a closure until this minute (SVB closed the
        # next morning). None (default) closes as soon as the rule says so.
        self.earliest_closure_minute: float | None = None
        self.held_reason: str | None = None
        # UF-1: with ``M20 closes on what the bank can pay`` (live opt-in), the liquidity test is the one the
        # usefulness ranking uses (engine/system/can_pay.py), not negative cash on the 0.1% slice
        self._can_pay = None
        self._cp_cursor, self._requested, self._line_at, self._past_none_at = 0, 0.0, None, None
        self._prepared, self._big_at, self._past = False, None, {}   # AU-1: the bank's own tools

    def attach(self, bus, scenario) -> None:
        super().attach(bus, scenario)
        from engine.live_overrides import flag_enabled

        if flag_enabled("M20 closes on what the bank can pay"):
            from engine.system.can_pay import can_pay_fractions

            self._can_pay = can_pay_fractions()

    def _can_pay_reason(self, t: float, book) -> str | None:
        """Closed when depositors have asked for more than the bank can pay (cash, what it can borrow, bonds it
        can sell near book value; with the Fed's program on in time, every bond at face value)."""
        from contracts.events import PolicyEnacted, WithdrawalRequested
        from engine.system.can_pay import capacity

        log = self.log()
        for event in log[self._cp_cursor:]:   # its own cursor: FakeModule.step owns ``_cursor``
            if isinstance(event, WithdrawalRequested):
                self._requested += float(event.amount)
            elif isinstance(event, PolicyEnacted) and event.lever == "emergency_lending" and self._line_at is None:
                self._line_at = float(event.t)
            elif isinstance(event, PolicyEnacted) and event.lever == "pledge_collateral":
                self._prepared = True
            elif isinstance(event, PolicyEnacted) and event.lever == "big_bank_deposits" and self._big_at is None:
                self._big_at = float(event.t)
        self._cp_cursor = len(log)
        asked = self._requested / max(1.0, deposits_base())
        fr = self._can_pay
        if self._past_none_at is None and asked > fr.none:
            self._past_none_at = float(t)
        for level in (fr.prepared, fr.fed, fr.none + fr.big_banks, fr.prepared + fr.big_banks, fr.fed + fr.big_banks):
            if level not in self._past and asked > level:
                self._past[level] = float(t)
        # AU-1: the same order as the usefulness ranking: pledged bonds, then the Fed's line in time, then big banks in time
        cap = fr.prepared if self._prepared else fr.none
        past = self._past.get(cap) if self._prepared else self._past_none_at
        if self._line_at is not None and (past is None or self._line_at <= past):
            cap = max(cap, fr.fed)
            past = self._past.get(cap)
        if self._big_at is not None and (past is None or self._big_at <= past):
            cap += fr.big_banks
        if asked > cap:
            return "unable to pay obligations"
        return "capital below threshold" if book.cet1 < self._cfg.cet1_floor else None

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.closed:
            return []
        book = fold_book(
            self.log(),
            deposits=deposits_base(),
            opening_cet1=self._cfg.opening_cet1,
        )
        reason = self._can_pay_reason(t, book) if self._can_pay is not None else closure_reason(book, self._cfg)
        if reason is None:
            if watch_recommended(book, self._cfg):
                if self.status != "watch":
                    self.status = "watch"
                    return [
                        SupervisoryAction(
                            t=t,
                            seed=self.seed,
                            bank_id=self.bank_id,
                            cause_id=self.cause("watch"),
                            action="watch",
                        )
                    ]
            return []
        if self.earliest_closure_minute is not None and t < self.earliest_closure_minute:
            first_hold = self.held_reason is None
            self.held_reason = reason
            if first_hold and self.status != "watch":
                self.status = "watch"
                return [
                    SupervisoryAction(
                        t=t,
                        seed=self.seed,
                        bank_id=self.bank_id,
                        cause_id=self.cause("watch"),
                        action="watch",
                    )
                ]
            return []
        self.closed = True
        self.status = "closed"
        return [
            BankClosed(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("close"),
                reason=reason,
            )
        ]


class PolicyFake(FakeModule):
    """M22. Turns scenario policies into PolicyEnacted. Dollar costs are not specified."""

    publishes: ClassVar[tuple[str, ...]] = ("PolicyEnacted",)

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.enacted: list[str] = []
        self._done: list[int] = []

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.scenario is None:
            return []
        out: list[BaseModel] = []
        for index, policy in enumerate(self.scenario.policies):
            if index in self._done:
                continue
            at_raw = policy.get("at", 0.0)
            if isinstance(at_raw, (int, float)) and not isinstance(at_raw, bool):
                at = float(at_raw)
            else:
                at = 0.0
            if t < at:
                continue
            lever_raw = policy.get("lever", "joint_statement")
            lever = canonical_lever(lever_raw if isinstance(lever_raw, str) else "joint_statement")
            scope_raw = policy.get("scope", self.bank_id)
            scope = scope_raw if isinstance(scope_raw, str) else self.bank_id
            self._done.append(index)
            self.enacted.append(lever)
            out.append(
                PolicyEnacted(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause(lever),
                    lever=lever,
                    scope=scope,
                    cost_model=_cost_model(lever),
                )
            )
        return out


def _cost_model(lever: str) -> dict[str, Any]:
    # DIF and Fed dollar formulas are not specified. The meter is an ordinal stub in [0, 1].
    moral = {
        "guarantee_deposits": 0.55,
        "emergency_lending": 0.35,
        "raise_insurance_limit": 0.40,
        "cap_withdrawals": 0.15,
        "big_bank_deposits": 0.22,
        "suspend_trading": 0.10,
        "joint_statement": 0.05,
    }.get(lever, 0.25)
    return {"dif_loss": 0.0, "fed_exposure": 0.0, "moral_hazard": clamp(moral, 0.0, 1.0)}


def l4_modules() -> dict[str, FakeModule]:
    return {"M20": RegulatorFake("M20"), "M22": PolicyFake("M22")}
