"""SVB-52 (D3): non-bank presets read dedicated depositor rows, not the SVB fit (ADR-099)."""

from __future__ import annotations

import pytest

import calibration.nonbank as nonbank
from engine.agents.decide import (
    DecisionModule,
    decision_to_request_rate_default,
    load_decision_params,
    mobile_share_default,
)
from engine.bank.payments import load_payment_params
from engine.depositor_rows import (
    SVB_TO_NONBANK,
    nonbank_depositor_rows,
    nonbank_rows_active,
    resolve,
)
from engine.gpu_sim import _registry_capacity_multiplier
from params.loader import get_parameter

NONBANK_VALUES = {
    "Non-bank M06 worry bias (ASSUMED)": -2.4,
    "Non-bank depositor mobile share of uninsured deposits (ASSUMED)": 0.878938,
    "Non-bank M07 decision-to-request rate (ASSUMED)": 0.0016667,
    "Non-bank M07 capacity multiplier (ASSUMED)": 4.20575,
}


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _snapshot() -> tuple[float, float, float, float]:
    return (
        load_decision_params().bias,
        mobile_share_default(),
        decision_to_request_rate_default(),
        _registry_capacity_multiplier(),
    )


def _svb_values() -> tuple[float, float, float, float]:
    return tuple(float(get_parameter(name).value) for name in SVB_TO_NONBANK)  # type: ignore[return-value]


def test_nonbank_rows_are_assumed_with_the_pre_fit_values() -> None:
    assert set(SVB_TO_NONBANK.values()) == set(NONBANK_VALUES)
    for name, value in NONBANK_VALUES.items():
        row = get_parameter(name)
        assert row.status == "ASSUMED"
        assert float(row.value) == pytest.approx(value)
        assert "4ed6ad9" in row.source and "non-bank" in row.source


def test_svb_rows_are_untouched_and_still_fitted() -> None:
    for name in SVB_TO_NONBANK:
        assert get_parameter(name).status == "FITTED"


def test_loaders_read_svb_rows_outside_the_context() -> None:
    assert not nonbank_rows_active()
    assert resolve("M06 worry bias (ASSUMED)") == "M06 worry bias (ASSUMED)"
    assert _snapshot() == pytest.approx(_svb_values())


def test_loaders_read_nonbank_rows_inside_the_context() -> None:
    with nonbank_depositor_rows():
        assert nonbank_rows_active()
        assert _snapshot() == pytest.approx(tuple(NONBANK_VALUES.values()))
    assert _snapshot() == pytest.approx(_svb_values())


def test_capacity_rescales_every_rail_by_the_multiplier_ratio() -> None:
    svb = load_payment_params()
    with nonbank_depositor_rows():
        nb = load_payment_params()
    ratio = NONBANK_VALUES["Non-bank M07 capacity multiplier (ASSUMED)"] / float(
        get_parameter("M07 capacity multiplier (FIT)").value
    )
    assert nb.app_daily_limit == pytest.approx(svb.app_daily_limit * ratio)


def test_context_resets_after_an_exception() -> None:
    with pytest.raises(RuntimeError):
        with nonbank_depositor_rows():
            raise RuntimeError("boom")
    assert not nonbank_rows_active()
    assert _snapshot() == pytest.approx(_svb_values())


def test_real_nonbank_engine_is_built_with_nonbank_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.registry as registry

    built: list[object] = []
    real_build = registry.build_engine

    def spy(*args, **kwargs):
        engine = real_build(*args, **kwargs)
        built.append(engine)
        assert nonbank_rows_active()
        return engine

    monkeypatch.setattr(registry, "build_engine", spy)
    monkeypatch.setattr(nonbank, "use_real_calibration_engine", lambda: True)
    scenario = nonbank.build_nonbank_scenario("equity", 11)
    nonbank._run_engine_log(scenario, steps=2, path=None, param_vector=None)
    assert built
    module = built[0]["M06"]
    assert isinstance(module, DecisionModule)
    assert module.params.bias == pytest.approx(-2.4)
    assert module.mobile_share == pytest.approx(0.878938)
    assert not nonbank_rows_active()


def test_gpu_sim_nonbank_batch_runs_inside_the_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine.gpu_sim as gpu_sim

    seen: list[tuple[float, float]] = []

    def fake_simulate(scenario, steps, samples):
        seen.append((mobile_share_default(), _registry_capacity_multiplier()))
        return [[] for _ in samples]

    monkeypatch.setattr(gpu_sim, "simulate_logs", fake_simulate)
    monkeypatch.setattr(nonbank, "_score_population_logs", lambda logs, **kw: [0.0 for _ in logs])
    nonbank._evaluate_nonbank_gpu_sim(
        [object(), object()],
        scenario=nonbank.build_nonbank_scenario("equity", 11),
        steps=2,
        target=None,
        weights=(1.0, 1.0, 1.0),
        minutes_per_day=390.0,
        on_candidate=None,
        on_candidate_error=None,
    )
    assert seen == [(pytest.approx(0.878938), pytest.approx(4.20575))]
    assert not nonbank_rows_active()


def test_svb_calibration_path_does_not_activate_nonbank_rows() -> None:
    from calibration.fit import real_calibration_modes
    from engine.modes import ModuleMode
    from engine.registry import build_engine
    from calibration.fit import _default_svb_scenario

    engine = build_engine(
        modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(11)
    )
    module = engine["M06"]
    assert module.params.bias == pytest.approx(float(get_parameter("M06 worry bias (ASSUMED)").value))
    assert module.mobile_share == pytest.approx(
        float(get_parameter("Depositor mobile share of uninsured deposits (FIT)").value)
    )
