"""SVB-65 (ADR-110): the CPU engine runs the fitted feedback the GPU calibration kernels run.

The GPU calibration paths (the stage-1 lean path and ``--gpu-full``) do two things the CPU
modules never did (SVB-06, SVB-25):

* depositors read ``tanh(mean M03 attention + news)``, where news decays with
  ``M03 news half-life minutes`` and grows with each tick's requested share times
  ``M03 feedback gain`` (``apply_composed_attention``, ``step_news``);
* the fake M14 quote moves toward ``price impact x cumulative requested share``
  (``step_market_group``).

So the live demo and the per-lane gpu-sim path ran a different model from the one the fit
calibrated (diag-11 and the sandbox at the persisted fit: day-one 0.189 on the CPU engine and
gpu-sim, 0.245 on gpu-full). ``step_module`` advances one module of one engine through the same
functions the GPU kernels call, so every path runs one model.

On for calibration real engines (``STAMPEDE_CALIBRATION_ENGINE=real``) and for the live
real-mode engine (``M29 live fitted feedback``). Off for every other engine (``engine.fitted_feedback``
unset and no calibration env), so seeded and default runs are unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def fitted_feedback_enabled(engine: Any) -> bool:
    """True where the fitted model runs: calibration real engines, and engines that opt in."""
    flag = getattr(engine, "fitted_feedback", None)
    if flag is not None:
        return bool(flag)
    return os.environ.get("STAMPEDE_CALIBRATION_ENGINE", "fake").strip().lower() == "real"


def install_persisted_feedback(engine: Any) -> None:
    """Give M06 the persisted feedback gain and price impact (the fit vector at its centre)."""
    from calibration.fit import _VECTOR_PARAM_NAMES
    from engine.gpu_full_io import apply_lane_feedback

    module = engine.get("M06")
    if module is not None:
        apply_lane_feedback(module, np.full(len(_VECTOR_PARAM_NAMES), 0.5))


def step_module(engine: Any, module_id: str, t: float) -> None:
    """One module's one-minute tick, with the fitted feedback where it is enabled."""
    feedback = fitted_feedback_enabled(engine)
    if feedback and module_id == "M14":
        from engine.gpu_full_io import step_market_group

        step_market_group([engine], float(t), np, 1.0)
        return
    for event in engine[module_id].step(t):
        engine.bus.publish(event)
    if feedback and module_id == "M03":
        from engine.gpu_full_io import apply_composed_attention

        apply_composed_attention([engine], float(t))
