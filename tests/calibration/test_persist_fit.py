"""Persisting runs: the non-bank vector gets its own row; nothing else in the registry moves."""
from __future__ import annotations

import json
import logging
import os

from calibration.persist_fit import NONBANK_VECTOR_ROW, apply_stage2_engine, main, write_nonbank_fitted_vector

_ENGINE_ENV = (
    "STAMPEDE_OFFLINE",
    "STAMPEDE_CALIBRATION_ENGINE",
    "STAMPEDE_BACKEND",
    "STAMPEDE_CALIB_GPU_SIM",
    "STAMPEDE_CALIB_GPU_FULL",
)

_REG = (
    "Parameter,Value,Status,Source,Module,Unit\n"
    "M14 price impact (ASSUMED),1.13,ASSUMED,SVB-25 prior,M14,1/share\n"
    "Depositor mobile share of uninsured deposits (FIT),0.88,ASSUMED,SVB-24,M06,fraction\n"
)


def test_nonbank_vector_appends_its_own_row_and_leaves_svb_rows(tmp_path):
    reg = tmp_path / "registry.csv"
    reg.write_text(_REG, encoding="utf-8")
    status = write_nonbank_fitted_vector((0.25, 0.5, 0.0), run_id="nb", seed=3060, objective=7.47, path=reg)
    assert status == {NONBANK_VECTOR_ROW: "appended"}
    text = reg.read_text(encoding="utf-8")
    assert text.startswith(_REG)
    assert text.endswith(f"{NONBANK_VECTOR_ROW},0.25;0.5;0,FITTED,nb seed=3060 objective=7.47,M27,normalized vector\n")


def test_cli_nonbank_reads_best_vector_json(tmp_path, capsys=None):
    reg = tmp_path / "registry.csv"
    reg.write_text(_REG, encoding="utf-8")
    run = tmp_path / "nonbank-x"
    run.mkdir()
    (run / "best_vector.json").write_text(
        json.dumps({"vector": [0.1, 0.2], "seed": 7, "objective": 1.5, "run_id": "nonbank-x"}), encoding="utf-8"
    )
    assert main(["--run-dir", str(run), "--kind", "nonbank", "--registry", str(reg)]) == 0
    assert f"{NONBANK_VECTOR_ROW},0.1;0.2,FITTED,nonbank-x seed=7 objective=1.5" in reg.read_text(encoding="utf-8")


# --- SVB-31: the printout is what was written, and a run can't be persisted twice ---------------


_SVB_REG = (
    "Parameter,Value,Status,Source,Module,Unit\n"
    "M14 price impact (ASSUMED),1.13,ASSUMED,SVB-25 prior,M14,1/share\n"
    "Depositor mobile share of uninsured deposits (FIT),0.88,ASSUMED,SVB-24,M06,fraction\n"
)


def _run_dir(tmp_path):
    run = tmp_path / "m27-x"
    run.mkdir()
    (run / "best_vector.json").write_text(
        json.dumps({"vector": [0.5] * 14 + [0.559, 0.4985], "seed": 3060, "objective": 1e-4, "run_id": "m27-x"}),
        encoding="utf-8",
    )
    return run


def _fake_updates(registry):
    """Like the real mapping: physical = current registry value x a vector scale. After a write the
    current value changes, so recomputing afterwards gives a different number (the old bug)."""
    import csv

    def compute(vector, *, path=None):
        rows = {r["Parameter"]: r for r in csv.DictReader(registry.open(encoding="utf-8"))}
        impact = float(rows["M14 price impact (ASSUMED)"]["Value"]) * 1.0484
        share = float(rows["Depositor mobile share of uninsured deposits (FIT)"]["Value"]) * 0.9988
        return {"M14 price impact (ASSUMED)": f"{impact:.6g}",
                "Depositor mobile share of uninsured deposits (FIT)": f"{share:.6g}"}

    return compute


def test_printout_equals_written_values_and_second_persist_is_refused(tmp_path, monkeypatch, capsys):
    import csv

    import calibration.fit as fit

    registry = tmp_path / "registry.csv"
    registry.write_text(_SVB_REG, encoding="utf-8")
    monkeypatch.setattr(fit, "fitted_registry_updates", _fake_updates(registry))
    monkeypatch.setattr(fit, "fitted_m06_values", lambda vector: {})   # the old recompute path, unused now
    run = _run_dir(tmp_path)

    assert main(["--run-dir", str(run), "--registry", str(registry)]) == 0
    printed = json.loads(capsys.readouterr().out)["values"]
    written = {r["Parameter"]: r["Value"] for r in csv.DictReader(registry.open(encoding="utf-8"))}
    for name, value in printed.items():
        assert written[name] == value

    before = registry.read_bytes()
    assert main(["--run-dir", str(run), "--registry", str(registry)]) == 3   # guard
    assert registry.read_bytes() == before                                    # nothing compounded


def _clear_engine_env(monkeypatch) -> None:
    for key in _ENGINE_ENV:
        monkeypatch.setenv(key, "")   # registers a restore so env set by the code under test is undone
        monkeypatch.delenv(key)


def _engine_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _ENGINE_ENV}


def test_stage2_sets_the_same_engine_env_as_run_fit(tmp_path, monkeypatch, capsys):
    """persist_fit --stage2 uses apply_run_fit_engine (--gpu-full) before the solve."""
    import calibration.fit as fit
    from calibration.run_fit import apply_run_fit_engine
    from engine.gpu_sim import calibration_uses_scripted_drawdown, gpu_sim_enabled

    _clear_engine_env(monkeypatch)
    registry = tmp_path / "registry.csv"
    registry.write_text(_SVB_REG, encoding="utf-8")
    monkeypatch.setattr(fit, "fitted_registry_updates", _fake_updates(registry))
    monkeypatch.setattr(fit, "fitted_m06_values", lambda vector: {})
    captured: dict[str, object] = {}

    def _fake_capacity(**kwargs):
        captured["kwargs"] = dict(kwargs)
        captured["env"] = _engine_env()
        captured["gpu_sim"] = gpu_sim_enabled()
        captured["scripted_drawdown"] = calibration_uses_scripted_drawdown()
        captured["real_engine"] = os.environ.get("STAMPEDE_CALIBRATION_ENGINE")
        return 1.5

    monkeypatch.setattr(fit, "fit_payment_capacity", _fake_capacity)
    run = _run_dir(tmp_path)
    assert main(["--run-dir", str(run), "--registry", str(registry), "--stage2", "--force"]) == 0
    out = capsys.readouterr().out
    assert "engine_path " in out
    assert "capacity_multiplier=1.5" in out
    assert captured["kwargs"]["write"] is True
    assert captured["kwargs"]["vector"] == tuple(float(x) for x in json.loads(
        (run / "best_vector.json").read_text(encoding="utf-8")
    )["vector"])
    persist_env = captured["env"]
    assert persist_env["STAMPEDE_CALIBRATION_ENGINE"] == "real"
    assert persist_env["STAMPEDE_BACKEND"] == "gpu"
    assert persist_env["STAMPEDE_CALIB_GPU_SIM"] == "1"
    assert persist_env["STAMPEDE_CALIB_GPU_FULL"] == "1"
    assert persist_env["STAMPEDE_OFFLINE"] == "1"
    assert captured["gpu_sim"] is True
    assert captured["real_engine"] == "real"
    # Registry cell is `no`; MarketFake only consults it when real engine / gpu-sim is on.
    assert captured["scripted_drawdown"] is False

    _clear_engine_env(monkeypatch)
    logger = logging.getLogger("test_persist_stage2")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    fit_setup = apply_run_fit_engine(
        gpu=True, gpu_sim=True, gpu_full=True, logger=logger, set_process_env=True
    )
    assert persist_env == _engine_env()
    assert fit_setup["gpu_sim"] is True
    assert fit_setup["gpu_full"] is True
    assert fit_setup["scripted_drawdown"] is False
    assert apply_stage2_engine(logger)["STAMPEDE_CALIB_GPU_FULL"] == persist_env["STAMPEDE_CALIB_GPU_FULL"]


def test_stage2_solver_sees_scripted_drawdown_flag(tmp_path, monkeypatch):
    """The solve reads calibration_uses_scripted_drawdown() under the fit-path env."""
    import calibration.fit as fit

    _clear_engine_env(monkeypatch)
    monkeypatch.setattr(
        "engine.gpu_sim.calibration_uses_scripted_drawdown", lambda: True
    )
    registry = tmp_path / "registry.csv"
    registry.write_text(_SVB_REG, encoding="utf-8")
    monkeypatch.setattr(fit, "fitted_registry_updates", _fake_updates(registry))
    monkeypatch.setattr(fit, "fitted_m06_values", lambda vector: {})
    captured: dict[str, object] = {}

    def _fake_capacity(**kwargs):
        from engine.gpu_sim import calibration_uses_scripted_drawdown, gpu_sim_enabled

        captured["kwargs"] = dict(kwargs)
        captured["scripted_drawdown"] = calibration_uses_scripted_drawdown()
        captured["gpu_sim"] = gpu_sim_enabled()
        captured["real_engine"] = os.environ.get("STAMPEDE_CALIBRATION_ENGINE")
        captured["in_calibration"] = gpu_sim_enabled() or (
            os.environ.get("STAMPEDE_CALIBRATION_ENGINE", "fake").strip().lower() == "real"
        )
        return 2.0

    monkeypatch.setattr(fit, "fit_payment_capacity", _fake_capacity)
    run = _run_dir(tmp_path)
    assert main(["--run-dir", str(run), "--registry", str(registry), "--stage2", "--force"]) == 0
    assert captured["kwargs"]["write"] is True
    assert captured["gpu_sim"] is True
    assert captured["real_engine"] == "real"
    assert captured["in_calibration"] is True
    assert captured["scripted_drawdown"] is True
