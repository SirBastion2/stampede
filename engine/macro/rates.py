"""M01 yield curve, policy rate, and property-price paths (§5, §18, §19)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from contracts.events import RatesChanged
from contracts.schemas import Scenario, ScenarioShock
from mocks.bus import EventBus
from params.loader import RegistryError, list_by_module

_SECTORS: tuple[str, ...] = ("office", "multifamily", "retail")

# Registry names. Values live in params/registry.csv (§8.1); nothing numeric is inlined.
_TENORS = "M01 maturity buckets (years)"
_LAMBDA = "M01 Nelson-Siegel lambda"
_BETA0 = "M01 Nelson-Siegel beta0"
_BETA2 = "M01 Nelson-Siegel beta2"
_DEFAULT_POLICY = "M01 default policy rate"
_POLICY_FLOOR = "M01 policy rate floor"
_POLICY_CAP = "M01 policy rate cap"
_UNEMPLOYMENT = "M01 unemployment baseline"
_UNEMPLOYMENT_FLOOR = "M01 unemployment floor"
_UNEMPLOYMENT_CAP = "M01 unemployment cap"
_CRE_OFFICE = "M01 CRE office baseline"
_CRE_MULTIFAMILY = "M01 CRE multifamily baseline"
_CRE_RETAIL = "M01 CRE retail baseline"
_CRE_FLOOR = "M01 CRE index floor"


@dataclass(frozen=True, slots=True)
class MacroParams:
    """Assumed M01 inputs loaded from the parameter registry."""

    tenors: tuple[float, ...]
    ns_lambda: float
    beta0: float
    beta2: float
    default_policy_rate: float
    policy_floor: float
    policy_cap: float
    unemployment_baseline: float
    unemployment_floor: float
    unemployment_cap: float
    cre_office: float
    cre_multifamily: float
    cre_retail: float
    cre_floor: float


def piecewise_linear(
    path: Sequence[tuple[float, float]],
    t: float,
    default: float,
) -> float:
    """Interpolate a piecewise-linear schedule. Flat outside the knots.

    An empty path returns ``default``. Knots are read in order; a segment
    with zero length is skipped.
    """
    if not path:
        return default
    if t <= path[0][0]:
        return path[0][1]
    for (t0, level0), (t1, level1) in zip(path, path[1:], strict=False):
        if t0 <= t <= t1 and t1 > t0:
            weight = (t - t0) / (t1 - t0)
            return level0 + (level1 - level0) * weight
    return path[-1][1]


def nelson_siegel(
    tenor_years: float,
    beta0: float,
    beta1: float,
    beta2: float,
    ns_lambda: float,
) -> float:
    """Nelson–Siegel yield at ``tenor_years``.

    The zero-tenor limit is beta0 + beta1. With beta1 = policy rate − beta0
    that limit is the policy rate (ADR-019).
    """
    if tenor_years <= 0.0 or ns_lambda <= 0.0:
        return beta0 + beta1
    x = ns_lambda * tenor_years
    if x < 1e-12:
        return beta0 + beta1
    exp_neg = math.exp(-x)
    slope_loading = (1.0 - exp_neg) / x
    curvature_loading = slope_loading - exp_neg
    return beta0 + beta1 * slope_loading + beta2 * curvature_loading


def _as_float(name: str, text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise RegistryError(f"{name} is not a float: {text!r}") from exc


def _required(rows: dict[str, str], name: str) -> float:
    try:
        text = rows[name]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {name}") from exc
    return _as_float(name, text)


def load_macro_params() -> MacroParams:
    """Load M01 rows from ``params/registry.csv`` via ``params.loader``."""
    rows = {row.parameter: row.value for row in list_by_module("M01")}
    try:
        tenor_text = rows[_TENORS]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {_TENORS}") from exc
    tenors = tuple(
        _as_float(_TENORS, part.strip())
        for part in tenor_text.split(",")
        if part.strip()
    )
    params = MacroParams(
        tenors=tenors,
        ns_lambda=_required(rows, _LAMBDA),
        beta0=_required(rows, _BETA0),
        beta2=_required(rows, _BETA2),
        default_policy_rate=_required(rows, _DEFAULT_POLICY),
        policy_floor=_required(rows, _POLICY_FLOOR),
        policy_cap=_required(rows, _POLICY_CAP),
        unemployment_baseline=_required(rows, _UNEMPLOYMENT),
        unemployment_floor=_required(rows, _UNEMPLOYMENT_FLOOR),
        unemployment_cap=_required(rows, _UNEMPLOYMENT_CAP),
        cre_office=_required(rows, _CRE_OFFICE),
        cre_multifamily=_required(rows, _CRE_MULTIFAMILY),
        cre_retail=_required(rows, _CRE_RETAIL),
        cre_floor=_required(rows, _CRE_FLOOR),
    )
    pairs = zip(params.tenors, params.tenors[1:], strict=False)
    if len(params.tenors) < 2 or any(right <= left for left, right in pairs):
        raise RegistryError("M01 maturity buckets must be strictly increasing")
    if any(tenor <= 0.0 for tenor in params.tenors):
        raise RegistryError("M01 maturity buckets must be positive")
    if params.ns_lambda <= 0.0:
        raise RegistryError("M01 Nelson-Siegel lambda must be positive")
    if params.policy_floor > params.policy_cap:
        raise RegistryError("M01 policy rate floor exceeds cap")
    if params.unemployment_floor > params.unemployment_cap:
        raise RegistryError("M01 unemployment floor exceeds cap")
    return params


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MacroRates:
    """Yield-curve and property-path generator (§19 M01).

    ``step(t)`` consumes Tick (the scheduler passes ``t``) and publishes
    one ``RatesChanged``. Scenario paths and shocks are the inputs; the
    curve, policy rate, three CRE indexes, and unemployment are the state.
    """

    def __init__(self, module_id: str = "M01") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.params = load_macro_params()
        self.policy_rate = self.params.default_policy_rate
        self.unemployment = self.params.unemployment_baseline
        self.cre: dict[str, float] = {}
        self.curve: list[float] = []
        self._applied: set[int] = set()
        self._seq = 0
        self._restore_baselines()

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        """Bind the scenario and subscribe to Tick. One attach per run."""
        self.scenario = scenario
        self._applied = set()
        self._seq = 0
        self._restore_baselines()
        bus.subscribe("Tick", self)

    def step(self, t: float) -> list[BaseModel]:
        """Advance one tick. No scenario attached → no events."""
        if self.scenario is None:
            return []
        self._apply_due_shocks(t)
        rate = piecewise_linear(
            self.scenario.rate_path,
            t,
            self.params.default_policy_rate,
        )
        rate = _clamp(rate, self.params.policy_floor, self.params.policy_cap)
        self.policy_rate = rate
        self.curve = self.yield_curve(rate)
        self._seq += 1
        event = RatesChanged(
            t=t,
            seed=self.scenario.seed,
            bank_id=None,
            cause_id=f"{self.module_id}-rates-{self._seq}",
            curve=list(self.curve),
            policy_rate=rate,
        )
        out: list[BaseModel] = [event]
        return out

    def yield_curve(self, policy_rate: float) -> list[float]:
        """Map Nelson–Siegel onto the registry maturity buckets.

        Slope is beta1 = policy rate − beta0 (ADR-019).
        """
        beta1 = policy_rate - self.params.beta0
        return [
            nelson_siegel(
                tenor,
                self.params.beta0,
                beta1,
                self.params.beta2,
                self.params.ns_lambda,
            )
            for tenor in self.params.tenors
        ]

    def yield_at(self, tenor_years: float) -> float:
        """Linear interpolation of the published bucket curve."""
        tenors = self.params.tenors
        curve = self.curve
        if not tenors or not curve:
            raise RuntimeError("yield curve is empty")
        if tenor_years <= tenors[0]:
            return curve[0]
        if tenor_years >= tenors[-1]:
            return curve[-1]
        for index, (left, right) in enumerate(zip(tenors, tenors[1:], strict=False)):
            if left <= tenor_years <= right:
                weight = (tenor_years - left) / (right - left)
                return curve[index] + (curve[index + 1] - curve[index]) * weight
        return curve[-1]

    def _restore_baselines(self) -> None:
        self.policy_rate = self.params.default_policy_rate
        self.unemployment = _clamp(
            self.params.unemployment_baseline,
            self.params.unemployment_floor,
            self.params.unemployment_cap,
        )
        self.cre = {
            "office": max(self.params.cre_floor, self.params.cre_office),
            "multifamily": max(self.params.cre_floor, self.params.cre_multifamily),
            "retail": max(self.params.cre_floor, self.params.cre_retail),
        }
        self.curve = self.yield_curve(self.policy_rate)

    def _apply_due_shocks(self, t: float) -> None:
        if self.scenario is None:
            return
        for index, shock in enumerate(self.scenario.shocks):
            if index in self._applied or t < shock.at:
                continue
            self._applied.add(index)
            self._apply_shock(shock)

    def _apply_shock(self, shock: ScenarioShock) -> None:
        if shock.kind == "unemployment":
            raw = shock.params.get("rate")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return
            self.unemployment = _clamp(
                float(raw),
                self.params.unemployment_floor,
                self.params.unemployment_cap,
            )
            return
        if shock.kind != "property":
            return
        factor = shock.params.get("factor")
        sector = shock.params.get("sector", "office")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            return
        if not isinstance(sector, str):
            return
        targets = _SECTORS if sector == "all" else (sector,)
        for name in targets:
            if name not in self.cre:
                continue
            self.cre[name] = max(self.params.cre_floor, self.cre[name] * float(factor))
