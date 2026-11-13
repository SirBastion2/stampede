"""M27 calibration harness: reproducibility, board, real module wiring."""

from __future__ import annotations

import csv

import numpy as np

from calibration.fit import (
    CalibrationModule,
    _default_svb_scenario,
    _theta_bounds,
    bound_flags_for,
    bracket_tipping_point,
    configure_calibration_gpu,
    evaluate_population_objectives,
    fit_behavioral_parameters,
    identifiable_indices,
    identifiable_names,
    load_calibration_targets,
    measure_outflow,
    objective_e1,
    run_calibration_harness,
    run_holdout,
    run_identifiability_sweep,
    shared_eval_seeds,
    sweep_identifiable_objectives,
    validate_fitted_parameters,
    vector_from_log_params,
    write_fitted_registry,
)
from contracts.schemas import Rumor, Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l8_calibration import CalibrationFake


def test_load_calibration_targets_match_e1_e5() -> None:
    targets = load_calibration_targets()
    assert 0.0 < targets.svb_day1_fraction < 1.0
    assert 0.0 < targets.svb_two_day_fraction <= 1.0
    assert targets.signature_day1_fraction == 0.20


def test_fit_is_reproducible_for_seed() -> None:
    first = fit_behavioral_parameters(42)
    second = fit_behavioral_parameters(42)
    assert first.vector == second.vector
    assert first.objective == second.objective


def test_validation_board_has_five_rows() -> None:
    _, _, board = run_calibration_harness(7)
    assert len(board) == 5
    ids = {row.target_id for row in board}
    assert ids == {
        "E1-requested-day1",
        "E1-next-open",
        "E1-settled-day1",
        "E5-day1",
        "placebo",
    }


def test_holdout_does_not_mutate_fit_vector() -> None:
    fit = fit_behavioral_parameters(3)
    before = fit.vector
    run_holdout(fit.vector, 3)
    assert fit.vector == before


def test_real_module_matches_fake_tick_metrics() -> None:
    scenario = Scenario(
        preset="tech-concentrated",
        seed=11,
        rumors=[
            Rumor(
                id="r1",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                credibility=0.7,
                reach=100,
            )
        ],
    )
    fake = CalibrationFake("M27")
    fake.scenario = scenario
    fake.on_tick(0.0, [])
    real = CalibrationModule("M27")
    real.scenario = scenario
    real.on_tick(0.0, [])
    assert fake.outflow_fraction == real.outflow_fraction
    assert fake.target_outflow == real.target_outflow


def test_parallel_workers_match_sequential_fake_objective() -> None:
    """Offline: population eval is identical for workers=1 and workers=3."""
    from params.loader import get_parameter

    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    rng = np.random.default_rng(99)
    samples = [rng.random(dim) for _ in range(6)]
    scenario = _default_svb_scenario(42)
    targets = load_calibration_targets()
    kwargs = {
        "fit_seed": 42,
        "generation": 3,
        "scenario": scenario,
        "steps": 1,
        "targets": targets,
        "w_day1": 1.0,
        "w_day2": 1.0,
        "use_fake_objective": True,
    }
    one = evaluate_population_objectives(samples, workers=1, **kwargs)
    three = evaluate_population_objectives(samples, workers=3, **kwargs)
    assert one == three


def test_configure_calibration_gpu_cpu_fallback(monkeypatch) -> None:
    """--gpu is a no-op fallback when CUDA is unavailable (CPU CI)."""
    from engine.agents.decide import cuda_device_available
    from ui.live.config import gpu_available, prefer_gpu

    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    label = configure_calibration_gpu(True)
    assert prefer_gpu() is True
    if cuda_device_available():
        assert label == "gpu"
        assert gpu_available() is True
    else:
        assert label == "cpu"
        assert gpu_available() is False


def test_shared_eval_seeds_are_stable_and_shared() -> None:
    first = shared_eval_seeds(42)
    second = shared_eval_seeds(42)
    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8


def test_identifiability_sweep_writes_the_identifiable_count(tmp_path, monkeypatch) -> None:
    """The sweep covers every slot and selects as many as the registry's fit set.

    It doesn't have to pick the same slots: its light book has no market, so it
    can't see price impact (see run_identifiability_sweep).
    """
    monkeypatch.setattr("calibration.fit.diagnostics_dir", lambda: tmp_path)
    payload = run_identifiability_sweep(seed=0, write=True)
    assert len(payload["selected"]) == len(identifiable_indices())
    assert {row["index"] for row in payload["params"]} >= set(identifiable_indices())
    assert (tmp_path / "identifiability-sweep.json").is_file()


def test_requested_objective_moves_and_is_monotone_through_bracket() -> None:
    """The three identifiable params change requested outflow; the joint sweep is monotone-ish."""
    seed = 11
    softness = 0.15
    curves = sweep_identifiable_objectives(seed=seed, grid=9, softness=softness)
    names = identifiable_names()
    assert set(curves) == set(names)
    changed = False
    for name, points in curves.items():
        objectives = [row[2] for row in points]
        day1 = [row[1] for row in points]
        assert any(abs(a - b) > 1e-12 for a, b in zip(objectives, objectives[1:]))
        deltas = [later - earlier for earlier, later in zip(day1, day1[1:])]
        if name == "contagion_k":
            # Higher k blocks more requests.
            assert sum(1 for delta in deltas if delta <= 0.0) >= len(deltas) - 2
        else:
            assert sum(1 for delta in deltas if delta >= 0.0) >= len(deltas) - 2
        changed = True
    assert changed
    box, joint = bracket_tipping_point(seed=seed, softness=softness)
    assert box[0] <= box[1]
    joint_day1 = [row[1] for row in joint]
    joint_deltas = [later - earlier for earlier, later in zip(joint_day1, joint_day1[1:])]
    assert any(abs(a - b) > 1e-12 for a, b in zip(joint_day1, joint_day1[1:]))
    assert sum(1 for delta in joint_deltas if delta >= 0.0) >= len(joint_deltas) - 2


def test_objective_e1_is_deterministic_per_seed() -> None:
    indices = identifiable_indices()
    vector = vector_from_log_params(np.zeros(len(indices)), indices)
    scenario = _default_svb_scenario(21)
    targets = load_calibration_targets()
    kwargs = {
        "scenario": scenario,
        "steps": 8,
        "targets": targets,
        "w_day1": 1.0,
        "w_day2": 1.0,
        "softness": 0.15,
    }
    first = objective_e1(vector, **kwargs)
    second = objective_e1(vector, **kwargs)
    assert first == second


def test_validation_with_caps_on_runs(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "16")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    from engine.agents.population import clear_sample_population_cache

    clear_sample_population_cache()
    vector = tuple(float(x) for x in np.full(12, 0.5))
    result = validate_fitted_parameters(vector, seed=5, steps=3)
    assert result.day1_settled >= 0.0
    assert result.two_day_settled >= result.day1_settled
    assert result.day1_requested >= 0.0
    assert result.closed in {True, False}


def test_fitted_bound_is_flagged() -> None:
    indices = identifiable_indices()
    theta = np.zeros(len(indices), dtype=np.float64)
    theta[0] = _theta_bounds(indices[0])[1]
    flags = bound_flags_for(theta, indices)
    assert identifiable_names()[0] in flags


def test_measure_outflow_default_is_still_settled() -> None:
    scenario = _default_svb_scenario(4)
    vector = np.full(12, 0.5)
    day1, two_day, closed = measure_outflow(scenario, steps=2, param_vector=vector)
    assert day1 >= 0.0
    assert two_day >= day1
    assert closed in {True, False}


def test_write_fitted_registry_roundtrips_extra_columns(tmp_path) -> None:
    """Extra CSV cells (DictReader restkey None) must not crash the FITTED write."""
    path = tmp_path / "registry.csv"
    path.write_text(
        "Parameter,Value,Status,Source,Module,Unit\n"
        "Keep me,hello,ASSUMED,src,M00,u,orphan\n"
        'Broken quotes,"{\\"status\\":\\"ok\\",\\"offline\\":true}",ASSUMED,src,M33,\n'
        "M06 worry bias (ASSUMED),-2.4,ASSUMED,src,M06,logit\n",
        encoding="utf-8",
    )
    vector = tuple(float(x) for x in np.full(12, 0.5))
    write_fitted_registry(vector, run_id="roundtrip", seed=7, path=path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames or []) == [
            "Parameter",
            "Value",
            "Status",
            "Source",
            "Module",
            "Unit",
        ]
        rows = list(reader)
    assert rows
    by_name = {row["Parameter"]: row for row in rows}
    # Unrelated rows are kept byte for byte now, including the orphan extra cell.
    assert "Keep me,hello,ASSUMED,src,M00,u,orphan\n" in path.read_text(encoding="utf-8")
    assert by_name["Keep me"]["Value"] == "hello"
    assert by_name["Keep me"]["Unit"] == "u"
    assert by_name["M06 worry bias (ASSUMED)"]["Status"] == "FITTED"
    assert by_name["M06 worry bias (ASSUMED)"]["Source"] == "roundtrip seed=7"
    second = tmp_path / "registry-again.csv"
    second.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    write_fitted_registry(vector, run_id="roundtrip", seed=7, path=second)
    assert second.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_build_engine_m27_real_mode() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=5, rumors=[])
    engine = build_engine(modes={"M27": ModuleMode.REAL}, scenario=scenario)
    module = engine["M27"]
    assert isinstance(module, CalibrationModule)
    engine.run(4)
    board = module.populate_validation_board()
    assert len(board) == 5
