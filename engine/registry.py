"""Compose fake modules and run them on the in-memory event bus."""

from __future__ import annotations

from engine.fitted_feedback import step_module
from collections.abc import Mapping

from pydantic import BaseModel

from contracts.schemas import Scenario
from engine.agents.attention import AttentionEngine
from engine.agents.decide import DecisionModule
from engine.agents.network import NetworkModule
from engine.agents.population import PopulationModule
from engine.bank.ledger import LedgerModule
from engine.bank.payments import PaymentsModule
from engine.bank.securities import SecuritiesValuer
from engine.interfaces import SimModule
from engine.liquidity.ladder import LiquidityLadderModule
from engine.macro.rates import MacroRates
from engine.modes import ModuleMode
from mocks.stream_loader import stream_modules_for_mode
from mocks.bus import EventBus

# §5 M00 phase order, with modules placed so a consumer can see same-tick events
# from publishers that §19 lists ahead of it. Headlines and prices feed back next tick.
TICK_ORDER: tuple[str, ...] = (
    "M00",
    "M01",
    "M22",
    "M28",
    "M34",
    "M02",
    "M05",
    "M41",
    "M42",
    "M44",
    "M03",
    "M04",
    "M25",
    "M06",
    "M07",
    "M12",
    "M08",
    "M09",
    "M23",
    "M14",
    "M20",
    "M26",
    "M24",
    "M27",
    "M29",
    "M30",
    "M33",
)


class Engine(dict[str, SimModule]):
    """Module map plus the bus those modules publish on."""

    bus: EventBus
    scenario: Scenario | None
    order: tuple[str, ...]

    def __init__(
        self,
        modules: dict[str, SimModule],
        bus: EventBus,
        scenario: Scenario | None,
    ) -> None:
        super().__init__(modules)
        self.bus = bus
        self.scenario = scenario
        known = tuple(module_id for module_id in TICK_ORDER if module_id in self)
        extra = tuple(module_id for module_id in self if module_id not in known)
        self.order = known + extra
        self._ran = False

    def run(self, steps: int) -> list[BaseModel]:
        if self.scenario is None:
            raise RuntimeError("build_engine was given no scenario")
        if self._ran:
            raise RuntimeError("engine.run() has already been called")
        if steps < 0:
            raise ValueError("steps must be >= 0")
        self._ran = True
        for index in range(steps):
            t = float(index)
            for module_id in self.order:
                step_module(self, module_id, t)
        return list(self.bus.log)


# Modules with a real implementation that ``build_engine`` can swap in (E1, ADR-101).
REAL_MODULE_IDS: tuple[str, ...] = (
    "M00", "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M12", "M14",
    "M20", "M22", "M23", "M24", "M25", "M26", "M27", "M28", "M33", "M34", "M41", "M42",
    "M44",
)


def build_engine(
    modes: Mapping[str, ModuleMode] | None = None,
    default_mode: ModuleMode = ModuleMode.FAKE,
    scenario: Scenario | None = None,
) -> Engine:
    """Return module map on the ordered bus.

    The default mode stays fake. Real mode is implemented for the modules in
    ``REAL_MODULE_IDS``. ``default_mode=REAL`` means real for every module in
    ``REAL_MODULE_IDS`` and fake for the rest (E1): before, it reached the stream
    loader as REAL for M09 and the UI modules, which the loader rejects. Asking
    for a module without a real implementation explicitly, e.g.
    ``modes={"M29": ModuleMode.REAL}``, still raises NotImplementedError.
    """
    if default_mode is ModuleMode.REAL:
        explicit = dict(modes or {})
        modes = {**{mid: ModuleMode.REAL for mid in REAL_MODULE_IDS}, **explicit}
        default_mode = ModuleMode.FAKE
    requested = dict(modes or {})
    m01_mode = requested.get("M01", default_mode)
    m03_mode = requested.get("M03", default_mode)
    m09_mode = requested.get("M09", default_mode)
    m23_mode = requested.get("M23", default_mode)
    m25_mode = requested.get("M25", default_mode)
    m41_mode = requested.get("M41", default_mode)
    m33_mode = requested.get("M33", default_mode)
    m34_mode = requested.get("M34", default_mode)
    m14_mode = requested.get("M14", default_mode)
    m02_mode = requested.get("M02", default_mode)
    m06_mode = requested.get("M06", default_mode)
    m24_mode = requested.get("M24", default_mode)
    m26_mode = requested.get("M26", default_mode)
    m27_mode = requested.get("M27", default_mode)
    m04_mode = requested.get("M04", default_mode)
    m20_mode = requested.get("M20", default_mode)
    m22_mode = requested.get("M22", default_mode)
    m42_mode = requested.get("M42", default_mode)
    m44_mode = requested.get("M44", default_mode)
    # The mock loader builds M00 real and skips M05 and M08 real. It rejects
    # M01, M09, and M34 real. M09 is omitted from the loader's map, so the loader
    # builds it with ``default_mode`` (always FAKE here, see the REAL rewrite at
    # the top) and this function then swaps REAL or drops OFF. M01, M06, M14, M34, M41, M42,
    # and M44 real are requested as fake and swapped for their real modules. M02
    # and M03 real are swapped the same way.
    loader_requested = {mid: mode for mid, mode in requested.items() if mid != "M09"}
    loader_modes: Mapping[str, ModuleMode] | None
    if m01_mode is ModuleMode.REAL:
        loader_modes = {**loader_requested, "M01": ModuleMode.FAKE}
    elif "M09" in requested:
        loader_modes = loader_requested
    else:
        loader_modes = modes
    if m23_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M23": ModuleMode.FAKE}
    if m25_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M25": ModuleMode.FAKE}
    if m41_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M41": ModuleMode.FAKE}
    if m33_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M33": ModuleMode.FAKE}
    if m34_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M34": ModuleMode.FAKE}
    if m14_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M14": ModuleMode.FAKE}
    if m02_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M02": ModuleMode.FAKE}
    if m03_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M03": ModuleMode.FAKE}
    if m06_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M06": ModuleMode.FAKE}
    if m24_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M24": ModuleMode.FAKE}
    if m26_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M26": ModuleMode.FAKE}
    if m27_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M27": ModuleMode.FAKE}
    if m04_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M04": ModuleMode.FAKE}
    if m20_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M20": ModuleMode.FAKE}
    if m22_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M22": ModuleMode.FAKE}
    if m42_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M42": ModuleMode.FAKE}
    if m44_mode is ModuleMode.REAL:
        base = {} if loader_modes is None else dict(loader_modes)
        loader_modes = {**base, "M44": ModuleMode.FAKE}
    bus = EventBus()
    resolved = modes or {}
    merged: dict[str, SimModule] = {}
    for stream in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"):
        merged.update(stream_modules_for_mode(stream, loader_modes, default_mode))
    if m01_mode is ModuleMode.REAL:
        merged["M01"] = MacroRates("M01")
    if resolved.get("M05", default_mode) is ModuleMode.REAL:
        population = PopulationModule()
        if scenario is not None:
            population.attach(scenario)
        merged["M05"] = population
    if resolved.get("M07", default_mode) is ModuleMode.REAL:
        merged["M07"] = PaymentsModule("M07")
    if resolved.get("M08", default_mode) is ModuleMode.REAL:
        merged["M08"] = LedgerModule("M08")
    if m09_mode is ModuleMode.REAL:
        merged["M09"] = SecuritiesValuer("M09")
    elif m09_mode is ModuleMode.OFF:
        merged.pop("M09", None)
    if resolved.get("M12", default_mode) is ModuleMode.REAL:
        merged["M12"] = LiquidityLadderModule("M12")
    if m23_mode is ModuleMode.REAL:
        from engine.bank.management import ManagementModule

        merged["M23"] = ManagementModule("M23")
    if m25_mode is ModuleMode.REAL:
        from agents.personas import PersonaPanelModule

        merged["M25"] = PersonaPanelModule("M25")
    if m41_mode is ModuleMode.REAL:
        from agents.vip import VIPModule

        merged["M41"] = VIPModule("M41")
    if m33_mode is ModuleMode.REAL:
        from ops.observability import ObservabilityModule

        merged["M33"] = ObservabilityModule("M33")
    if m34_mode is ModuleMode.REAL:
        from agents.guardrails import GuardrailModule

        merged["M34"] = GuardrailModule("M34")
    if m14_mode is ModuleMode.REAL:
        from engine.markets.book import OrderBook

        merged["M14"] = OrderBook("M14")
    if m02_mode is ModuleMode.REAL:
        merged["M02"] = NetworkModule("M02")
    if m03_mode is ModuleMode.REAL:
        merged["M03"] = AttentionEngine("M03")
    if isinstance(merged.get("M02"), NetworkModule) and isinstance(merged.get("M03"), AttentionEngine):
        # SVB-57: M03 reads M02's network only when ``M03 uses M02 network`` is on.
        merged["M03"].network_source = merged["M02"]
    if m06_mode is ModuleMode.REAL:
        merged["M06"] = DecisionModule("M06")
    if m24_mode is ModuleMode.REAL:
        from agents.crisis_room import CrisisRoomModule

        merged["M24"] = CrisisRoomModule("M24")
    if m26_mode is ModuleMode.REAL:
        from engine.futures.batch import FuturesRunner

        merged["M26"] = FuturesRunner("M26")
    if m27_mode is ModuleMode.REAL:
        from calibration.fit import CalibrationModule

        merged["M27"] = CalibrationModule("M27")
    if m04_mode is ModuleMode.REAL:
        from agents.media import MediaModule

        merged["M04"] = MediaModule("M04")
    if m20_mode is ModuleMode.REAL:
        from engine.system.regulators import RegulatorModule

        merged["M20"] = RegulatorModule("M20")
    if m22_mode is ModuleMode.REAL:
        from engine.policy.levers import PolicyLabModule

        merged["M22"] = PolicyLabModule("M22")
    if resolved.get("M28", default_mode) is ModuleMode.REAL:
        from scenarios.module import ScenarioModule

        merged["M28"] = ScenarioModule("M28")
    if m42_mode is ModuleMode.REAL:
        from agents.ceo import CEOModule

        merged["M42"] = CEOModule("M42")
    if m44_mode is ModuleMode.REAL:
        from agents.shorts import ShortSellerModule

        merged["M44"] = ShortSellerModule("M44")
    engine = Engine(merged, bus, scenario)
    if scenario is not None:
        for module in engine.values():
            # PopulationModule.attach takes the scenario only. Every other
            # module with attach, including M34, takes (bus, scenario).
            if isinstance(module, PopulationModule):
                if module.scenario is None:
                    module.attach(scenario)
            else:
                attach = getattr(module, "attach", None)
                if callable(attach):
                    attach(bus, scenario)
    return engine


def run_scenario(
    scenario: Scenario,
    steps: int,
    modes: Mapping[str, ModuleMode] | None = None,
    default_mode: ModuleMode = ModuleMode.FAKE,
) -> list[BaseModel]:
    """Run ``steps`` minute-ticks and return the ordered event log."""
    return build_engine(modes=modes, default_mode=default_mode, scenario=scenario).run(steps)
