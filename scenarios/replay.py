"""M28 replay helpers: event-log hash, snapshots, and branch comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from contracts.schemas import Scenario, ScenarioShock
from scenarios.builder import ScenarioControls, build_scenario


def event_log_hash(events: Sequence[BaseModel]) -> str:
    """Deterministic SHA-256 of the serialized event log (same shape as M33 fake)."""
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(event.model_dump(mode="json", by_alias=True))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical_event(event: BaseModel) -> str:
    """One event as the canonical JSON that ``event_log_hash`` hashes (SVB-48)."""
    return json.dumps(
        event.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Captured run state for branching (§5 M28)."""

    branch_id: str
    scenario: Scenario
    event_log: tuple[BaseModel, ...]
    branch_index: int


def snapshot_at(
    scenario: Scenario,
    event_log: Sequence[BaseModel],
    *,
    branch_index: int,
    branch_id: str | None = None,
) -> RunSnapshot:
    if branch_index < 0 or branch_index > len(event_log):
        raise IndexError("branch_index out of range for event log")
    bid = branch_id or f"branch-{branch_index}"
    return RunSnapshot(
        branch_id=bid,
        scenario=scenario,
        event_log=tuple(event_log),
        branch_index=branch_index,
    )


def branch_scenario(
    snapshot: RunSnapshot,
    overrides: ScenarioControls,
) -> Scenario:
    """Fork a scenario at ``snapshot.branch_index`` with new controls."""
    base = snapshot.scenario
    merged = ScenarioControls(
        preset=overrides.preset or base.preset,
        seed=overrides.seed if overrides.seed is not None else base.seed,
        bank_id=overrides.bank_id,
        rumors=overrides.rumors if overrides.rumors is not None else list(base.rumors),
        rumor_poster=overrides.rumor_poster,
        rate_path=overrides.rate_path if overrides.rate_path is not None else list(base.rate_path),
        shocks=overrides.shocks if overrides.shocks is not None else list(base.shocks),
        policies=overrides.policies if overrides.policies is not None else list(base.policies),
        start_minute_of_day=overrides.start_minute_of_day,
        start_weekday=overrides.start_weekday,
        fidelity=overrides.fidelity or base.fidelity,
    )
    scenario = build_scenario(merged)
    scenario.shocks.append(
        ScenarioShock(
            kind="branch",
            at=float(snapshot.branch_index),
            params={"parent": snapshot.branch_id},
        )
    )
    return scenario


@dataclass(frozen=True, slots=True)
class BranchCompare:
    """Side-by-side summary for two branches."""

    left_id: str
    right_id: str
    left_hash: str
    right_hash: str
    left_event_count: int
    right_event_count: int
    first_divergence_index: int | None
    left_only_types: tuple[str, ...]
    right_only_types: tuple[str, ...]


def _event_types(events: Sequence[BaseModel]) -> list[str]:
    return [event.__class__.__name__ for event in events]


def compare_branches(left: RunSnapshot, right: RunSnapshot) -> BranchCompare:
    """Compare two snapshots event-by-event and by log hash.

    The first divergence is the first index whose canonical event JSON differs (type
    *and* payload), or the shorter length when one log extends the other. So
    ``first_divergence_index is None`` exactly when the two log hashes are equal.
    Comparing types alone (before SVB-48) reported no divergence for logs with the
    same event sequence but different amounts.
    """
    left_types = _event_types(left.event_log)
    right_types = _event_types(right.event_log)
    left_rows = [_canonical_event(event) for event in left.event_log]
    right_rows = [_canonical_event(event) for event in right.event_log]
    divergence: int | None = None
    limit = min(len(left_rows), len(right_rows))
    for index in range(limit):
        if left_rows[index] != right_rows[index]:
            divergence = index
            break
    if divergence is None and len(left_rows) != len(right_rows):
        divergence = limit
    left_set = set(left_types)
    right_set = set(right_types)
    return BranchCompare(
        left_id=left.branch_id,
        right_id=right.branch_id,
        left_hash=event_log_hash(left.event_log),
        right_hash=event_log_hash(right.event_log),
        left_event_count=len(left.event_log),
        right_event_count=len(right.event_log),
        first_divergence_index=divergence,
        left_only_types=tuple(sorted(left_set - right_set)),
        right_only_types=tuple(sorted(right_set - left_set)),
    )
