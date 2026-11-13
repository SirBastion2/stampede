"""Deterministic demo scenario for the M30 shell (fake engine, no GPU)."""

from __future__ import annotations

from contracts.schemas import Rumor, Scenario, ScenarioShock


def demo_scenario(seed: int = 1) -> Scenario:
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
        policies=[{"lever": "guarantee_deposits", "scope": "bank-fvb", "at": 10}],
        shocks=[
            ScenarioShock(
                kind="ui_rumor",
                at=3.0,
                params={
                    "rumor_id": "rumor-ui",
                    "claim": "Group chat says Fictional Valley Bank is raising cash",
                    "source_node": "ui-user",
                    "credibility": 0.45,
                },
            )
        ],
    )
