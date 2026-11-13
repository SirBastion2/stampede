"""M28 scenario builder, presets, replay hash, and real module wiring."""

from __future__ import annotations

from typing import Any

from contracts.schemas import Rumor, Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.bus import EventBus
from mocks.l8_calibration import ScenarioFake
from scenarios import (
    ScenarioControls,
    branch_scenario,
    build_scenario,
    compare_branches,
    event_log_hash,
    list_presets,
    snapshot_at,
)
from scenarios.module import ScenarioModule
from scenarios.presets import bank_for_preset, normalize_preset_key


def _panic(seed: int) -> Scenario:
    return Scenario(
        preset="tech-concentrated",
        seed=seed,
        rate_path=[(0.0, 0.04), (3.0, 0.06)],
        rumors=[
            Rumor(
                id="rumor-1",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                credibility=0.85,
                reach=12000,
            )
        ],
    )


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json", by_alias=True) for event in events]


def test_list_presets_covers_registry_rows() -> None:
    menu = list_presets()
    keys = {row["key"] for row in menu}
    assert keys == {"tech-concentrated", "crypto-heavy", "insured-community"}
    tech = next(row for row in menu if row["key"] == "tech-concentrated")
    assert tech["bank_id"] == "bank-fvb"
    assert tech["ticker"] == "FVB"


def test_bank_for_preset_matches_ledger_bootstrap() -> None:
    bank = bank_for_preset("tech-concentrated")
    assert bank.id == "bank-fvb"
    assert bank.assets.cash > 0.0
    assert sum(bank.liabilities.deposits_by_segment.values()) > bank.assets.cash
    assert bank.capital.cet1 > 0.0


def test_build_scenario_uses_default_rate_path() -> None:
    scenario = build_scenario(ScenarioControls(preset="tech-concentrated", seed=3))
    assert scenario.rate_path[0] == (0.0, 0.04)
    assert normalize_preset_key("svb-like") == "tech-concentrated"


def test_event_log_hash_is_stable() -> None:
    log = run_scenario(_panic(5), 6)
    assert event_log_hash(log) == event_log_hash(log)
    other = run_scenario(_panic(6), 6)
    assert event_log_hash(log) != event_log_hash(other)


def test_branch_compare_finds_divergence() -> None:
    left_log = run_scenario(_panic(1), 4)
    right_log = run_scenario(_panic(2), 4)
    left = snapshot_at(_panic(1), left_log, branch_index=2, branch_id="main")
    right = snapshot_at(_panic(2), right_log, branch_index=2, branch_id="alt")
    summary = compare_branches(left, right)
    assert summary.left_event_count == len(left_log)
    assert summary.first_divergence_index is not None


def test_branch_compare_sees_payload_changes_not_only_types() -> None:
    """SVB-48: same event types, different payload, is a divergence at that index."""
    from contracts.events import RumorPosted, Tick
    from scenarios.replay import RunSnapshot, event_log_hash

    def _log(credibility: float) -> list:
        return [
            Tick(t=0.0, seed=1),
            RumorPosted(
                t=0.0,
                seed=1,
                bank_id="bank-fvb",
                cause_id="c",
                rumor_id="r",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                source_node="hub-public-1",
                credibility=credibility,
            ),
            Tick(t=1.0, seed=1),
        ]

    base = _panic(1)
    same_a = snapshot_at(base, _log(0.8), branch_index=0, branch_id="a")
    same_b = snapshot_at(base, _log(0.8), branch_index=0, branch_id="b")
    other = snapshot_at(base, _log(0.5), branch_index=0, branch_id="c")
    longer = snapshot_at(base, [*_log(0.8), Tick(t=2.0, seed=1)], branch_index=0, branch_id="d")
    assert isinstance(same_a, RunSnapshot)

    identical = compare_branches(same_a, same_b)
    assert identical.first_divergence_index is None
    assert identical.left_hash == identical.right_hash

    changed = compare_branches(same_a, other)
    assert changed.first_divergence_index == 1
    assert changed.left_only_types == changed.right_only_types == ()
    assert changed.left_hash != changed.right_hash

    extended = compare_branches(same_a, longer)
    assert extended.first_divergence_index == 3
    for pair in ((same_a, same_b), (same_a, other), (same_a, longer)):
        summary = compare_branches(*pair)
        hashes_equal = event_log_hash(pair[0].event_log) == event_log_hash(pair[1].event_log)
        assert (summary.first_divergence_index is None) is hashes_equal


def test_branch_scenario_carries_parent_marker() -> None:
    base = _panic(9)
    log = run_scenario(base, 3)
    snap = snapshot_at(base, log, branch_index=1)
    fork = branch_scenario(
        snap,
        ScenarioControls(preset="tech-concentrated", seed=9, policies=[{"lever": "guarantee"}]),
    )
    assert any(shock.kind == "branch" for shock in fork.shocks)


def test_real_module_matches_fake_rumor_posts() -> None:
    scenario = _panic(1)
    bus = EventBus()
    fake = ScenarioFake("M28")
    fake.attach(bus, scenario)
    fake_published = fake.step(0.0)
    real = ScenarioModule("M28")
    real.attach(bus, scenario)
    real_published = real.step(0.0)
    assert fake.dropped == real.dropped == []
    assert len(fake_published) == len(real_published) == 1
    assert _dump(fake_published) == _dump(real_published)


def test_build_engine_m28_real_mode() -> None:
    engine = build_engine(modes={"M28": ModuleMode.REAL}, scenario=_panic(2))
    assert isinstance(engine["M28"], ScenarioModule)
    log = engine.run(4)
    rumors = [event for event in log if event.__class__.__name__ == "RumorPosted"]
    assert len(rumors) == 1


def test_same_scenario_seed_same_hash_with_real_m28() -> None:
    modes = {"M28": ModuleMode.REAL}
    first = event_log_hash(run_scenario(_panic(11), 8, modes=modes))
    second = event_log_hash(run_scenario(_panic(11), 8, modes=modes))
    assert first == second
