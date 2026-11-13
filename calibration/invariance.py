"""Exact-zero invariance: unrelated text must not change the event log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.rumor_input import IRRELEVANT_READ_AS, classify_rumor_market
from contracts.schemas import Rumor, Scenario
from ops.event_log import event_log_hash

IRRELEVANT_PROBES: tuple[str, ...] = (
    "I blinked",
    "It's raining in Princeton",
    "ajjfalk",
)


@dataclass(frozen=True, slots=True)
class InvarianceCheck:
    """One exact-zero comparison."""

    probe: str
    category: str
    read_as: str
    affects_simulation: bool
    baseline_hash: str
    probe_hash: str
    passed: bool
    detail: str


def is_irrelevant_input(text: str) -> bool:
    """Shared rumor rule: non-market text is category irrelevant and has no effect."""
    label = classify_rumor_market(text)
    return (
        label.category == "irrelevant"
        and not label.market_related
        and not label.affects_simulation
        and label.read_as == IRRELEVANT_READ_AS
    )


def gated_rumors_for_text(text: str, *, rumor_id: str = "invariance-probe") -> list[Rumor]:
    """Attach a rumor only when the shared classifier says it affects the simulation."""
    label = classify_rumor_market(text)
    if label.category == "irrelevant" or not label.affects_simulation:
        return []
    return [
        Rumor(
            id=rumor_id,
            claim=text,
            credibility=label.credibility,
            reach=1,
            notes=["category:market"],
        )
    ]


def _with_gated_text(scenario: Scenario, text: str) -> Scenario:
    extra = gated_rumors_for_text(text, rumor_id=f"invariance-{scenario.seed}")
    if not extra:
        return scenario
    return scenario.model_copy(update={"rumors": list(scenario.rumors) + extra})


def validate_exact_zero_invariance(
    scenario: Scenario,
    *,
    steps: int,
    probes: tuple[str, ...] = IRRELEVANT_PROBES,
    path: Path | None = None,
) -> list[InvarianceCheck]:
    """Event logs must match the baseline when unrelated probes are gated by the classifier.

    ``path`` is accepted for registry-style callers and is unused: the check
    reads the shared lexicon only.
    """
    del path
    from engine.registry import run_scenario

    baseline_log = run_scenario(scenario, steps)
    baseline_hash = event_log_hash(baseline_log)
    rows: list[InvarianceCheck] = []
    for probe in probes:
        label = classify_rumor_market(probe)
        gated = _with_gated_text(scenario, probe)
        probe_hash = event_log_hash(run_scenario(gated, steps))
        passed = (
            label.category == "irrelevant"
            and not label.affects_simulation
            and label.read_as == IRRELEVANT_READ_AS
            and probe_hash == baseline_hash
        )
        rows.append(
            InvarianceCheck(
                probe=probe,
                category=label.category,
                read_as=label.read_as,
                affects_simulation=label.affects_simulation,
                baseline_hash=baseline_hash,
                probe_hash=probe_hash,
                passed=passed,
                detail=(
                    "Not market-related: no effect on the simulation."
                    if passed
                    else "unrelated input changed the event log or classifier label"
                ),
            )
        )
    return rows


def invariance_passed(checks: list[InvarianceCheck]) -> bool:
    return bool(checks) and all(row.passed for row in checks)
