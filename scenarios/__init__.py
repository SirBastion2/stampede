"""M28 scenario builder, presets, and replay (§18.6)."""

from scenarios.builder import ScenarioControls, build_scenario, list_presets
from scenarios.module import ScenarioModule
from scenarios.presets import bank_for_preset, list_preset_keys, normalize_preset_key
from scenarios.replay import (
    BranchCompare,
    RunSnapshot,
    branch_scenario,
    compare_branches,
    event_log_hash,
    snapshot_at,
)

__all__ = [
    "BranchCompare",
    "RunSnapshot",
    "ScenarioControls",
    "ScenarioModule",
    "bank_for_preset",
    "branch_scenario",
    "build_scenario",
    "compare_branches",
    "event_log_hash",
    "list_preset_keys",
    "list_presets",
    "normalize_preset_key",
    "snapshot_at",
]
