"""M23 bank management: determinism, policy model, real-vs-fake contract."""

from __future__ import annotations

from contracts.events import (
    CEOAction,
    LossRealized,
    PaymentSettled,
    PriceTick,
    StatementIssued,
    Tick,
)
from contracts.schemas import Scenario
from engine.bank.management import (
    ManagementModule,
    ManagementPolicy,
    capital_raise_success_probability,
    load_management_policy,
    roll_capital_raise_success,
    statement_template,
)
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.bus import EventBus
from mocks.l5_grok import ManagementFake
from mocks.support import OPENING_PRICE, deposits_base


def _scenario(seed: int = 11) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed, rate_path=[(0.0, 0.05)])


def test_load_management_policy_uses_assumed_when_registry_tbd() -> None:
    policy = load_management_policy()
    assert policy.cash_threshold_assumed or policy.cash_borrow_threshold is not None
    assert policy.capital_threshold_assumed or policy.capital_raise_cet1_threshold > 0


def test_capital_raise_probability_and_roll_are_deterministic() -> None:
    p = capital_raise_success_probability(-0.08, 0.4, 0.03)
    assert 0.05 <= p <= 0.95
    first = roll_capital_raise_success(42, "bank-fvb", "march-8", p)
    second = roll_capital_raise_success(42, "bank-fvb", "march-8", p)
    assert first == second


def test_statement_template_is_deterministic() -> None:
    kwargs = {"preset": "tech-concentrated", "action": "announce_capital_raise", "succeeded": False}
    a = statement_template(**kwargs)
    b = statement_template(**kwargs)
    assert a == b
    assert a[0] == "failed_capital_raise"


def test_real_module_matches_fake_on_ceo_action() -> None:
    scenario = _scenario(7)
    bus = EventBus()
    ceo = CEOAction(
        t=1.0,
        seed=scenario.seed,
        bank_id="bank-fvb",
        cause_id="ceo-1",
        action="announce_capital_raise",
        params={"segment": "startup"},
        rationale="attention is elevated",
    )
    fake = ManagementFake("M23")
    fake.attach(bus, scenario)
    fake_out = fake.on_tick(1.0, [ceo])

    real = ManagementModule("M23")
    real.attach(bus, scenario)
    real_out = real._on_tick(1.0, [ceo])

    assert len(fake_out) == len(real_out) == 1
    fake_stmt = fake_out[0]
    real_stmt = real_out[0]
    assert isinstance(fake_stmt, StatementIssued)
    assert isinstance(real_stmt, StatementIssued)
    assert fake_stmt.stmt_id == real_stmt.stmt_id == "stmt-management-1"
    assert fake_stmt.channel == real_stmt.channel == "press_release"
    assert fake_stmt.segments == real_stmt.segments
    assert any(entry.action == "announce_capital_raise" for entry in real.actions)


def test_real_module_matches_fake_on_stock_slide() -> None:
    scenario = _scenario(3)
    bus = EventBus()
    ticks = [
        PriceTick(
            t=0.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="px-0",
            price=OPENING_PRICE,
            volume=1.0,
        ),
        PriceTick(
            t=1.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="px-1",
            price=OPENING_PRICE * 0.94,
            volume=2.0,
        ),
    ]
    fake = ManagementFake("M23")
    fake.attach(bus, scenario)
    fake_out = fake.on_tick(1.0, ticks)

    real = ManagementModule("M23")
    real.attach(bus, scenario)
    real_out = real._on_tick(1.0, ticks)

    assert len(fake_out) == len(real_out) == 1
    assert fake_out[0].stmt_id == real_out[0].stmt_id == "stmt-management-1"


def test_svb_march8_sequence_logs_capital_raise_after_loss() -> None:
    scenario = _scenario(99)
    bus = EventBus()
    module = ManagementModule("M23")
    module.attach(bus, scenario)
    module.open_price = OPENING_PRICE
    loss = LossRealized(
        t=8.0,
        seed=scenario.seed,
        bank_id="bank-fvb",
        cause_id="loss-march8",
        amount=1.8e9,
        kind="securities",
    )
    bus.publish(loss)
    bus.publish(
        PriceTick(
            t=8.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="px-8",
            price=OPENING_PRICE * 0.92,
            volume=5e6,
        )
    )
    bus.publish(
        CEOAction(
            t=8.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="ceo-march8",
            action="announce_capital_raise",
            params={},
            rationale="March 8 capital plan after securities loss",
        )
    )
    out = module.step(8.0)
    assert any(isinstance(event, StatementIssued) for event in out)
    assert any(entry.action == "announce_capital_raise" for entry in module.actions)


def test_management_step_is_deterministic() -> None:
    scenario = _scenario(5)

    def run_once() -> list[str]:
        bus = EventBus()
        module = ManagementModule("M23")
        module.attach(bus, scenario)
        bus.publish(Tick(t=0.0, seed=scenario.seed, bank_id=None, cause_id="tick-0"))
        bus.publish(
            CEOAction(
                t=0.0,
                seed=scenario.seed,
                bank_id="bank-fvb",
                cause_id="ceo",
                action="issue_statement",
                params={},
                rationale="test",
            )
        )
        events = module.step(0.0)
        return [e.model_dump_json() for e in events]

    assert run_once() == run_once()


def test_build_engine_m23_real_mode() -> None:
    scenario = _scenario(1)
    engine = build_engine(modes={"M23": ModuleMode.REAL}, scenario=scenario)
    module = engine["M23"]
    assert isinstance(module, ManagementModule)
    engine.run(3)
    assert isinstance(module, ManagementModule)


def test_policy_rules_log_borrow_when_cash_negative() -> None:
    policy = ManagementPolicy(
        cash_borrow_threshold=0.0,
        capital_raise_cet1_threshold=0.05,
        dividend_cut_cet1_threshold=0.05,
        cash_threshold_assumed=True,
        capital_threshold_assumed=True,
    )
    scenario = _scenario(2)
    bus = EventBus()
    module = ManagementModule("M23", policy=policy)
    module.attach(bus, scenario)
    module._policy_flags = {k: False for k in module._policy_flags}
    bus.publish(
        PaymentSettled(
            t=0.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="pay",
            amount=deposits_base() * 0.01,
            channel="wire",
            queued_remaining=0,
        )
    )
    out = module._apply_policy_rules(1.0, bus.log)
    assert any(entry.action == "borrow_facility" for entry in module.actions)
    assert out == [] or all(isinstance(event, StatementIssued) for event in out)
