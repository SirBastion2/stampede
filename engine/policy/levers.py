"""M22 policy lab — scheduled levers, cost models, moral-hazard meter (§5, §18, §19)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, Field, ValidationError

from agents.llm_cache import (
    LLMCache,
    RemoteJsonBridge,
    cached_json_completion,
    effective_cache_model,
    live_json_completion_enabled,
    offline_mode,
)
from contracts.events import PolicyEnacted
from contracts.schemas import Scenario
from mocks.bus import EventBus
from mocks.support import bank_id_for, canonical_lever, clamp
from params.loader import RegistryError, get_parameter, list_by_module

_PARAM_BIG_BANK_DEPOSIT = "Policy: First Republic big-bank deposit analog"
_PARAM_MORAL_BASES = "M22 moral hazard base by lever (ASSUMED)"
_PARAM_MORAL_FREQ = "M22 moral hazard frequency weight (ASSUMED)"
_PARAM_MORAL_BREADTH = "M22 moral hazard breadth weight (ASSUMED)"
_PARAM_DIF_UNINSURED = "M22 DIF loss uninsured fraction (ASSUMED)"
_PARAM_FED_EMERGENCY = "M22 Fed emergency lending exposure fraction (ASSUMED)"
_PARAM_INSURED_SHARE = "M22 assumed uninsured deposit share for DIF (ASSUMED)"
_PARAM_ADVISOR_MODEL = "M22 regulator advisor Grok model (ASSUMED)"
_PARAM_DEPOSITS = "SVB-like preset: deposits"

_LEVER_MENU: tuple[str, ...] = (
    "guarantee_deposits",
    "emergency_lending",
    "raise_insurance_limit",
    "cap_withdrawals",
    "big_bank_deposits",
    "suspend_trading",
    "joint_statement",
)

_DEFAULT_MORAL_BASES: dict[str, float] = {
    "guarantee_deposits": 0.55,
    "emergency_lending": 0.35,
    "raise_insurance_limit": 0.40,
    "cap_withdrawals": 0.15,
    "big_bank_deposits": 0.22,
    "suspend_trading": 0.10,
    "joint_statement": 0.05,
}


def _parse_money_cell(text: str) -> float:
    lowered = text.strip().lower().replace(",", "").replace("$", "")
    match = re.match(r"[><=\s]*([0-9]*\.?[0-9]+)\s*([a-z%]*)", lowered)
    if match is None:
        raise RegistryError(f"cannot parse registry money value: {text!r}")
    number = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("b"):
        return number * 1_000_000_000.0
    if unit.startswith("m"):
        return number * 1_000_000.0
    return number


def _as_float(name: str, text: str) -> float:
    try:
        return float(text.strip())
    except ValueError as exc:
        raise RegistryError(f"{name}: cannot parse float from {text!r}") from exc


def _parse_lever_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise RegistryError(f"expected lever:value pairs, got {piece!r}")
        lever, raw = piece.split(":", 1)
        out[canonical_lever(lever.strip())] = _as_float("moral hazard base", raw.strip())
    return out


@dataclass(frozen=True, slots=True)
class PolicyParams:
    deposits_total: float
    big_bank_deposit_usd: float
    moral_bases: dict[str, float]
    moral_frequency_weight: float
    moral_breadth_weight: float
    dif_uninsured_fraction: float
    fed_emergency_fraction: float
    uninsured_share: float
    advisor_model: str


def load_policy_params(path: Any = None) -> PolicyParams:
    """Load M22 ASSUMED rows from the registry (§8.1)."""
    rows = {row.parameter: row.value for row in list_by_module("M22", path)}
    moral_raw = rows.get(_PARAM_MORAL_BASES)
    if moral_raw:
        moral_bases = _parse_lever_map(moral_raw)
    else:
        moral_bases = dict(_DEFAULT_MORAL_BASES)
    for lever, base in _DEFAULT_MORAL_BASES.items():
        moral_bases.setdefault(lever, base)
    deposits = _parse_money_cell(get_parameter(_PARAM_DEPOSITS, path).value)
    big_bank = _parse_money_cell(get_parameter(_PARAM_BIG_BANK_DEPOSIT, path).value)
    return PolicyParams(
        deposits_total=deposits,
        big_bank_deposit_usd=big_bank,
        moral_bases=moral_bases,
        moral_frequency_weight=_as_float(
            _PARAM_MORAL_FREQ,
            rows.get(_PARAM_MORAL_FREQ, "0.08"),
        ),
        moral_breadth_weight=_as_float(
            _PARAM_MORAL_BREADTH,
            rows.get(_PARAM_MORAL_BREADTH, "0.12"),
        ),
        dif_uninsured_fraction=_as_float(
            _PARAM_DIF_UNINSURED,
            rows.get(_PARAM_DIF_UNINSURED, "0.15"),
        ),
        fed_emergency_fraction=_as_float(
            _PARAM_FED_EMERGENCY,
            rows.get(_PARAM_FED_EMERGENCY, "0.25"),
        ),
        uninsured_share=_as_float(
            _PARAM_INSURED_SHARE,
            rows.get(_PARAM_INSURED_SHARE, "0.93"),
        ),
        advisor_model=rows.get(_PARAM_ADVISOR_MODEL, "grok-4.1-fast").strip(),
    )


@dataclass
class MoralHazardMeter:
    """Tracks how broad and frequent guarantees are (§5 M22)."""

    guarantee_count: int = 0
    scopes: set[str] = field(default_factory=set)

    def record(self, lever: str, scope: str) -> None:
        if lever in {"guarantee_deposits", "raise_insurance_limit"}:
            self.guarantee_count += 1
            self.scopes.add(scope)

    def score(
        self,
        lever: str,
        params: PolicyParams,
        *,
        base: float,
    ) -> float:
        breadth = len(self.scopes) * params.moral_breadth_weight
        frequency = self.guarantee_count * params.moral_frequency_weight
        return clamp(base + breadth + frequency, 0.0, 1.0)


def dollar_costs(lever: str, params: PolicyParams) -> tuple[float, float]:
    """Return (dif_loss_usd, fed_exposure_usd) for a lever."""
    deposits = params.deposits_total
    uninsured = deposits * params.uninsured_share
    if lever == "guarantee_deposits":
        return uninsured * params.dif_uninsured_fraction, 0.0
    if lever == "raise_insurance_limit":
        return uninsured * params.dif_uninsured_fraction * 0.5, 0.0
    if lever == "emergency_lending":
        return 0.0, deposits * params.fed_emergency_fraction
    if lever == "big_bank_deposits":
        return 0.0, params.big_bank_deposit_usd
    return 0.0, 0.0


def build_cost_model(
    lever: str,
    params: PolicyParams,
    meter: MoralHazardMeter,
) -> dict[str, Any]:
    base = params.moral_bases.get(lever, 0.25)
    dif_loss, fed_exposure = dollar_costs(lever, params)
    moral = meter.score(lever, params, base=base)
    return {
        "dif_loss": dif_loss,
        "fed_exposure": fed_exposure,
        "moral_hazard": moral,
    }


def scope_applies(scope: str, bank_id: str) -> bool:
    normalized = scope.strip().lower()
    if normalized in {"", "all", "*"}:
        return True
    return normalized == bank_id.strip().lower()


class PolicyExplanation(BaseModel):
    lever: str
    summary: str
    moral_hazard: float = Field(ge=0.0, le=1.0)


class PolicyAdvisorBatch(BaseModel):
    explanations: list[PolicyExplanation]


class JsonCompletionProvider(Protocol):
    def complete_json(self, prompt: str) -> dict[str, Any]: ...


class _OfflineAdvisorProvider:
    def complete_json(self, prompt: str) -> dict[str, Any]:
        del prompt
        params = load_policy_params()
        meter = MoralHazardMeter()
        explanations = []
        for lever in _LEVER_MENU:
            cost = build_cost_model(lever, params, meter)
            meter.record(lever, "offline")
            explanations.append(
                {
                    "lever": lever,
                    "summary": _offline_summary(lever, cost),
                    "moral_hazard": cost["moral_hazard"],
                }
            )
        return {"explanations": explanations}


def _offline_summary(lever: str, cost: dict[str, Any]) -> str:
    dif = cost["dif_loss"]
    fed = cost["fed_exposure"]
    moral = cost["moral_hazard"]
    if lever == "guarantee_deposits":
        return (
            f"Backstops uninsured balances; modeled DIF stress about ${dif:,.0f} "
            f"with moral-hazard meter {moral:.2f}."
        )
    if lever == "emergency_lending":
        return (
            f"Lends against bonds at face value; Fed exposure about ${fed:,.0f}, "
            f"moral hazard {moral:.2f}."
        )
    if lever == "big_bank_deposits":
        return f"Peer banks place about ${fed:,.0f}; limited moral hazard ({moral:.2f})."
    return f"Policy lever {lever}; moral hazard {moral:.2f}."


def explain_policy_options(
    *,
    bank_id: str,
    levers: tuple[str, ...] = _LEVER_MENU,
    provider: JsonCompletionProvider | None = None,
    cache: LLMCache | None = None,
) -> PolicyAdvisorBatch:
    """Regulator advisor (§7): explain costs; never decides."""
    params = load_policy_params()
    prompt = json.dumps(
        {
            "bank_id": bank_id,
            "levers": list(levers),
            "deposits": params.deposits_total,
        },
        sort_keys=True,
    )
    if provider is None:
        provider = (
            RemoteJsonBridge(prompt, params.advisor_model)
            if live_json_completion_enabled()
            else _OfflineAdvisorProvider()
        )
    raw = cached_json_completion(
        prompt=prompt,
        model=effective_cache_model(params.advisor_model),
        provider=provider,
        cache=cache,
    )
    try:
        return PolicyAdvisorBatch.model_validate(raw)
    except ValidationError:
        return PolicyAdvisorBatch(
            explanations=[
                PolicyExplanation(
                    lever=lever,
                    summary=_offline_summary(
                        lever,
                        build_cost_model(lever, params, MoralHazardMeter()),
                    ),
                    moral_hazard=build_cost_model(
                        lever, params, MoralHazardMeter()
                    )["moral_hazard"],
                )
                for lever in levers
            ]
        )


class PolicyLabModule:
    """M22 policy lever generators (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("PolicyEnacted",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick",)

    def __init__(self, module_id: str, params: PolicyParams | None = None) -> None:
        self.module_id = module_id
        self.params = params or load_policy_params()
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self.meter = MoralHazardMeter()
        self.enacted: list[str] = []
        self._done: list[int] = []
        self._seq = 0

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None:
            return []
        out: list[BaseModel] = []
        bank_id = bank_id_for(self.scenario.preset)
        for index, policy in enumerate(self.scenario.policies):
            if index in self._done:
                continue
            at_raw = policy.get("at", 0.0)
            if isinstance(at_raw, (int, float)) and not isinstance(at_raw, bool):
                at = float(at_raw)
            else:
                at = 0.0
            if t < at:
                continue
            lever_raw = policy.get("lever", "joint_statement")
            lever = canonical_lever(lever_raw if isinstance(lever_raw, str) else "joint_statement")
            scope_raw = policy.get("scope", bank_id)
            scope = scope_raw if isinstance(scope_raw, str) else bank_id
            self._done.append(index)
            self.enacted.append(lever)
            self.meter.record(lever, scope)
            cost_model = build_cost_model(lever, self.params, self.meter)
            self._seq += 1
            out.append(
                PolicyEnacted(
                    t=t,
                    seed=self.scenario.seed,
                    bank_id=bank_id,
                    cause_id=f"{self.module_id}-{lever}-{self._seq}",
                    lever=lever,
                    scope=scope,
                    cost_model=cost_model,
                )
            )
        return out
