"""Load fake module maps by stream and mode (used by ``engine.registry``)."""

from __future__ import annotations

from collections.abc import Mapping

from engine.interfaces import SimModule
from engine.modes import ModuleMode
from mocks.l1_engine import l1_modules
from mocks.l2_banks import l2_modules
from mocks.l3_markets import l3_modules
from mocks.l4_system import l4_modules
from mocks.l5_grok import l5_modules
from mocks.l6_crowd_ui import l6_modules
from mocks.l7_market_ui import l7_modules
from mocks.l8_calibration import l8_modules

_STREAM_BUILDERS = {
    "L1": l1_modules,
    "L2": l2_modules,
    "L3": l3_modules,
    "L4": l4_modules,
    "L5": l5_modules,
    "L6": l6_modules,
    "L7": l7_modules,
    "L8": l8_modules,
}


def stream_modules_for_mode(
    stream: str,
    modes: Mapping[str, ModuleMode] | None,
    default_mode: ModuleMode,
) -> dict[str, SimModule]:
    builder = _STREAM_BUILDERS[stream]
    raw = builder()
    out: dict[str, SimModule] = {}
    for module_id, module in raw.items():
        mode = (modes or {}).get(module_id, default_mode)
        if mode is ModuleMode.OFF:
            continue
        if mode is ModuleMode.FAKE:
            out[module_id] = module
        elif mode is ModuleMode.REAL and module_id == "M00":
            from engine.core.clock import Clock

            out[module_id] = Clock()
        elif mode is ModuleMode.REAL and module_id == "M05":
            continue
        elif mode is ModuleMode.REAL and module_id in {"M07", "M08"}:
            continue
        elif mode is ModuleMode.REAL and module_id == "M12":
            continue
        elif mode is ModuleMode.REAL and module_id in {"M28", "M33"}:
            continue
        elif mode is ModuleMode.REAL and module_id == "M20":
            continue
        elif mode is ModuleMode.REAL and module_id in {"M04", "M41", "M42", "M44"}:
            continue
        else:
            raise NotImplementedError(
                f"real mode is not implemented for {module_id}; "
                "only off, fake, and M00/M05/M07/M08/M12/M20/M28/M33 real are available in this wave"
            )
    return out
