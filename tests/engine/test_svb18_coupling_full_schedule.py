"""SVB-18b decision input: does one candidate's vector move another's results on the full schedule?

The 390-minute diagnostic in test_svb18_rumor_timeline.py XPASSes. This reruns it on the real
stage-1 schedule (390 one-minute ticks plus 70 fifteen-minute overnight ticks, 460 ticks).
Opt-in, because it's slow on CPU: run on the 3060 with

    STAMPEDE_RUN_COUPLING=1 pytest tests/engine/test_svb18_coupling_full_schedule.py -q -rxX

XFAIL: coupling is present -> build SVB-18b. XPASS: no coupling at the full schedule -> skip SVB-18b.
Record the result and the max difference printed by the test in the SVB-10 verification doc.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from calibration.fit import _default_svb_scenario, default_tick_schedule, fixed_softness
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, simulate_requested_fractions

pytestmark = pytest.mark.skipif(
    os.environ.get("STAMPEDE_RUN_COUPLING") != "1", reason="opt-in: set STAMPEDE_RUN_COUPLING=1"
)
_SEEDS = (17, 23)


def _vectors() -> list[np.ndarray]:
    prior = np.full(16, 0.5)
    hot = np.full(16, 0.5)
    hot[14] = 0.75
    hot[15] = 0.60
    return [prior, hot, np.full(16, 0.45)]


def _run(vectors) -> np.ndarray:
    clear_sample_population_cache()
    clear_static_pack_cache()
    return simulate_requested_fractions(
        _default_svb_scenario(_SEEDS[0]), default_tick_schedule(), vectors, list(_SEEDS),
        softness=fixed_softness(),
    )


@pytest.mark.xfail(reason="lead-lane reposts couple a seed's candidates (SVB-18b)", strict=False)
def test_candidate_zero_does_not_move_candidate_one_on_the_full_schedule(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)
    for name, value in {
        "STAMPEDE_CALIBRATION_ENGINE": "real", "STAMPEDE_BACKEND": "gpu", "STAMPEDE_CALIB_GPU_SIM": "1",
        "STAMPEDE_CALIB_GPU_FULL": "1", "STAMPEDE_CALIB_DEPOSITORS": "2000", "STAMPEDE_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    assert len(default_tick_schedule()) == 460
    vectors = _vectors()
    base = _run(vectors)
    changed = [v.copy() for v in vectors]
    changed[0][14] = 0.95                     # candidate 0 only: much stronger price impact
    changed[0][15] = 0.95                     # and a much larger mobile share
    moved = _run(changed)
    n = len(vectors)
    worst = 0.0
    for seed_i in range(len(_SEEDS)):
        lane1 = seed_i * n + 1
        worst = max(worst, float(np.max(np.abs(base[lane1] - moved[lane1]))))
    print(f"max |change| in candidate 1 = {worst:.3e}")
    assert worst <= 1e-9
