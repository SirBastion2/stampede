"""Bundled run-phase crowd frame for the fake UI (registry population, orb budget)."""

from __future__ import annotations

from contracts.schemas import CrowdFrame, CrowdPanels
from ui.crowd.frames import layout_community_count, render_point_count, state_ribbon
from ui.crowd.layout import build_layout
from ui.crowd.orbs import aggregate_orb_states, demo_crowd_counts, demo_gauges

_DEMO_RUMOR = "I heard they lost billions on bonds"
_DEMO_BANK = "bank-fvb"
_DEMO_SEED = 1
_DEMO_T = 5.0


def build_demo_frame() -> CrowdFrame:
    """Run phase, rumor posted, 1M people drawn as the registry orb budget."""
    calm, wavering, withdrew = demo_crowd_counts()
    cash, ticks, tipping = demo_gauges()
    orb_count = render_point_count()
    communities = layout_community_count()
    layout = build_layout(_DEMO_SEED, orb_count, communities)
    states = aggregate_orb_states(
        calm + wavering + withdrew,
        layout.point_count,
        calm,
        wavering,
        withdrew,
        layout.community_count,
        _DEMO_SEED,
    )
    panels = CrowdPanels(
        cash_gauge=cash,
        time_to_failure_ticks=ticks,
        tipping_point=tipping,
        rumor=_DEMO_RUMOR,
        state_ribbon=state_ribbon(calm, wavering, withdrew),
        calm_count=calm,
        wavering_count=wavering,
        withdrew_count=withdrew,
    )
    return CrowdFrame(
        t=_DEMO_T,
        seed=_DEMO_SEED,
        bank_id=_DEMO_BANK,
        point_count=layout.point_count,
        positions_xy=list(layout.positions_xy),
        state=states,
        balance_norm=list(layout.balance_norm),
        uninsured_mask=list(layout.uninsured_mask),
        panels=panels,
    )
