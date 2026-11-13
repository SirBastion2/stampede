"""M09 securities valuer: duration/convexity marks, sales, determinism, contracts."""

from __future__ import annotations

from typing import Any

import pytest

from contracts import parse_event
from contracts.events import AssetSold, LossRealized, RatesChanged
from contracts.schemas import Scenario
from engine.bank.securities import (
    BondBucket,
    SecuritiesValuer,
    expected_six_year_hundred_bp_move,
    initial_buckets,
    mark_buckets,
    price_change_fraction,
    realize_sale,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.bus import EventBus
from mocks.l2_banks import SecuritiesFake
from ui.live.pacing import crisis_sim_minutes


def _minimal_scenario(seed: int = 11) -> Scenario:
    return Scenario(
        preset="tech-concentrated",
        seed=seed,
        rate_path=[(0.0, 0.04), (120.0, 0.055)],
    )


def test_hundred_bp_six_year_duration_matches_registry() -> None:
    target = expected_six_year_hundred_bp_move()
    change = price_change_fraction(6.0, 0.0, 0.01)
    assert abs(change - target) < 0.005


def test_mark_buckets_deterministic() -> None:
    buckets = initial_buckets("tech-concentrated")
    first = mark_buckets(buckets, 0.01)
    second = mark_buckets(buckets, 0.01)
    assert first == second
    assert sum(b.market_value for b in first) < sum(b.market_value for b in buckets)


def test_realize_sale_underwater_loss() -> None:
    bucket = BondBucket(
        asset_class="afs",
        maturity_bucket="5y",
        book_value=100.0,
        market_value=90.0,
        coupon=0.03,
        duration=5.0,
        convexity=0.25,
        htm=False,
    )
    updated, loss = realize_sale([bucket], "afs", 50.0, htm_reclassify=False)
    assert loss == pytest.approx(5.0)
    assert updated[0].book_value == pytest.approx(50.0)
    assert updated[0].market_value == pytest.approx(45.0)


def test_htm_sale_reclassifies_remaining() -> None:
    buckets = [
        BondBucket("htm", "2y", 100.0, 95.0, 0.03, 2.0, 0.04, True),
        BondBucket("htm", "5y", 100.0, 95.0, 0.03, 5.0, 0.25, True),
    ]
    updated, _ = realize_sale(buckets, "htm", 10.0, htm_reclassify=True)
    assert all(not b.htm for b in updated)
    assert all(b.asset_class == "afs" for b in updated)


def test_securities_valuer_step_publishes_parseable_loss() -> None:
    bus = EventBus()
    scenario = _minimal_scenario()
    valuer = SecuritiesValuer("M09")
    valuer.attach(bus, scenario)
    valuer.inbox.append(
        RatesChanged(
            t=0.0,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="m01-rates-1",
            curve=[0.04, 0.042, 0.044],
            policy_rate=0.04,
        )
    )
    valuer.inbox.append(
        RatesChanged(
            t=0.5,
            seed=scenario.seed,
            bank_id="bank-fvb",
            cause_id="m01-rates-2",
            curve=[0.06, 0.062, 0.064],
            policy_rate=0.06,
        )
    )
    valuer.inbox.append(
        AssetSold.model_validate(
            {
                "t": 1.0,
                "seed": scenario.seed,
                "bank_id": "bank-fvb",
                "cause_id": "m08-sale-1",
                "class": "afs",
                "amount": 1_000_000.0,
                "price_impact": 0.1,
            }
        )
    )
    events = valuer.step(1.0)
    assert len(events) == 1
    loss = events[0]
    assert isinstance(loss, LossRealized)
    roundtrip = parse_event(loss.model_dump(mode="json", by_alias=True))
    assert isinstance(roundtrip, LossRealized)
    assert roundtrip.kind == "securities"
    assert roundtrip.amount > 0.0


def test_real_m09_engine_is_deterministic() -> None:
    scenario = _minimal_scenario(42)
    modes = {"M09": ModuleMode.REAL}
    a = run_scenario(scenario, 8, modes=modes)
    b = run_scenario(scenario, 8, modes=modes)
    assert _dump(a) == _dump(b)


def test_real_vs_fake_loss_events_share_contract() -> None:
    from tests.test_fakes import panic

    scenario = panic(7)
    steps = int(crisis_sim_minutes())
    fake_log = run_scenario(scenario, steps, modes={"M09": ModuleMode.FAKE})
    real_log = run_scenario(scenario, steps, modes={"M09": ModuleMode.REAL})
    fake_losses = [e for e in fake_log if isinstance(e, LossRealized)]
    real_losses = [e for e in real_log if isinstance(e, LossRealized)]
    for event in fake_losses + real_losses:
        parsed = parse_event(event.model_dump(mode="json", by_alias=True))
        assert isinstance(parsed, LossRealized)
        assert parsed.kind == "securities"
        assert parsed.amount > 0.0
    assert fake_losses or real_losses


def test_build_engine_real_m09_not_not_implemented() -> None:
    engine = build_engine(modes={"M09": ModuleMode.REAL}, scenario=_minimal_scenario())
    assert isinstance(engine["M09"], SecuritiesValuer)
    assert not isinstance(engine["M09"], SecuritiesFake)


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json", by_alias=True) for event in events]
