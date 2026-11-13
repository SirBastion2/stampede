"""Offline tests for the non-bank price-path harness."""

from __future__ import annotations

import hashlib
import json
import math
import re
import os
from pathlib import Path

import numpy as np
import pytest

from agents.guardrails import claim_allowed
from agents.rumor_input import classify_rumor_market
from calibration.fit import (
    _community_scenario,
    _default_svb_scenario,
    _log_sq_error,
    _signature_scenario,
    measure_outflow,
)
from calibration.nonbank import (
    ShapeTargets,
    build_nonbank_scenario,
    fit_nonbank_parameters,
    label_nonbank_claim,
    load_nonbank_targets,
    run_nonbank_holdout,
    scale_price_stats,
    shape_loss,
    shape_loss_terms,
    summarize_price_path,
)
from calibration.run_nonbank_fit import main
from contracts.events import BankClosed, PaymentSettled, PriceTick
from engine.registry import run_scenario
from params.loader import get_parameter

_SEED = 20260309
_STEPS = 8
_SCENARIO_HASH = {
    "svb": "482ca4af093e52c1c717cdc68914ef012fe98e4eb04673eb3a38960df124e05f",
    "sig": "cd1bebc15f464ffa6ef2266624c2f7ac80ffa57d1fdad54018a85a83ce22fc0c",
    "com": "f394e651b44f2c579b09ebc9b0d78ea402db31b6f63b83971f847bba63ecb0cf",
}
# Fake-engine path pins as of ADR-048.
_PATH_HASH = {
    "svb": "f1e7f543b9b7cd5ef017cd9a1b4389f0e6e487cb6a3738329886719e6513aab0",
    "sig": "9a5ed8dc7f29cd6da35e0de2f5bb431d80836de2d937c39cfca84d82627332a8",
    "com": "175dc094465e8339cf95be13197b165987f957bfcb226b0e911c22f82029f0c1",
}
# Fake-engine outflow pins as of ADR-048.
_OUTFLOW = {
    "svb": (2.591976703622465e-05, 2.591976703622465e-05, False),
    "sig": (2.4981736309821542e-05, 2.4981736309821542e-05, False),
    "com": (0.0, 0.0, False),
}


def _repo_runs_snapshot() -> frozenset[tuple[str, int, int]]:
    """Every file under the repo's calibration/runs, with size and mtime (SVB-46).

    The dev checkout keeps real, untracked runs there, so "the directory does not
    exist" is the wrong check. A test must only leave it exactly as it found it.
    """
    runs = Path(__file__).resolve().parents[1] / "calibration" / "runs"
    if not runs.exists():
        return frozenset()
    return frozenset(
        (str(path.relative_to(runs)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in runs.rglob("*")
        if path.is_file()
    )


def _sha(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def test_nonbank_point_targets_match_citations() -> None:
    targets = load_nonbank_targets()
    assert targets.equity.event_id == "E13"
    assert targets.equity.peak_to_trough == 0.2639009287925697
    assert targets.equity.days_to_bottom == 1.0
    assert targets.equity.volume_multiple == 3.218018656033446
    assert targets.equity.unset_fields() == ()
    assert targets.rates.peak_to_trough == 0.339
    assert targets.rates.days_to_bottom == 23.0
    assert targets.rates.volume_multiple is None

    meta = get_parameter("Non-bank E13 Meta peak-to-trough fraction")
    assert meta.status == "VERIFIED"
    assert "finance.yahoo.com/quote/META/history" in meta.source
    volume = get_parameter("Non-bank E13 Meta volume multiple")
    assert "188119900" in volume.source
    assert "58458300" in volume.source
    days = get_parameter("Non-bank E13 Meta days to bottom")
    assert "cnbc.com/2022/02/03" in days.source
    spx = get_parameter("Non-bank E16 SPX peak-to-trough fraction")
    assert "yardenibook.com" in spx.source
    assert get_parameter("Non-bank E16 SPX peak close").value == "3386.15"
    assert get_parameter("Non-bank E16 SPX trough close").value == "2237.40"
    gme = get_parameter("Non-bank E14 GameStop intraday high USD")
    assert gme.status == "VERIFIED"
    assert gme.value == "483"
    assert "GameStop_short_squeeze" in gme.source
    luna = get_parameter("Non-bank E15 LUNA price 2022-05-05 USD")
    assert luna.value == "87"
    assert "arxiv.org" in luna.source
    peak = get_parameter("Non-bank E15 LUNA April 5 2022 peak USD")
    assert peak.value == "119.18"
    assert "w31160" in peak.source


def test_unset_shape_targets_stay_unset() -> None:
    targets = load_nonbank_targets()
    assert targets.gamestop.unset_fields() == (
        "peak-to-trough",
        "days-to-bottom",
        "volume-multiple",
    )
    assert targets.crypto.unset_fields() == (
        "days-to-bottom",
        "volume-multiple",
    )
    assert targets.rates.unset_fields() == ("volume-multiple",)
    for name in (
        "Non-bank E14 GameStop peak-to-trough fraction",
        "Non-bank E14 GameStop days to bottom",
        "Non-bank E14 GameStop volume multiple",
        "Non-bank E15 LUNA days to bottom",
        "Non-bank E15 LUNA volume multiple",
        "Non-bank E16 SPX volume multiple",
    ):
        row = get_parameter(name)
        assert row.status == "UNSET"
        assert row.value == "UNSET"


def test_shape_loss_on_a_peaked_path() -> None:
    stats = summarize_price_path(
        [0.0, 10.0, 20.0, 30.0],
        [100.0, 110.0, 80.0, 90.0],
        [10.0, 10.0, 40.0, 10.0],
        minutes_per_day=1440.0,
    )
    assert stats.peak_to_trough == (110.0 - 80.0) / 110.0
    assert stats.days_to_bottom == 10.0 / 1440.0
    assert stats.volume_multiple == 4.0
    assert stats.n_prints == 4
    target = ShapeTargets("unit", 0.5, 1.0, 4.0)
    loss = shape_loss(stats, target, weights=(1.0, 0.0, 0.0))
    assert loss == _log_sq_error(stats.peak_to_trough, 0.5)
    flat = scale_price_stats(stats, np.array([0.5, 0.5, 0.5]), coefficient=0.8, center=0.5)
    assert flat.peak_to_trough == stats.peak_to_trough
    assert flat.volume_multiple == stats.volume_multiple
    empty = summarize_price_path([], [], [], minutes_per_day=1440.0)
    assert empty.n_prints == 0
    assert empty.peak_to_trough == 0.0


def test_unset_targets_add_nothing_to_the_loss() -> None:
    stats = summarize_price_path([0.0, 1.0], [100.0, 40.0], [5.0, 20.0], minutes_per_day=1440.0)
    blank = ShapeTargets("E14", None, None, None)
    assert shape_loss(stats, blank, weights=(1.0, 1.0, 1.0)) == 0.0
    partial = ShapeTargets("E16", 0.339, 33.0, None)
    full = shape_loss(stats, partial, weights=(1.0, 1.0, 1.0))
    without_volume = shape_loss(stats, partial, weights=(1.0, 1.0, 0.0))
    assert full == without_volume
    assert full > 0.0


def test_presets_use_explicit_rumors_and_shared_classifier() -> None:
    example = classify_rumor_market("gamestop is not doing well")
    assert example.market_related
    assert example.affects_simulation
    for event_type in ("equity", "crypto", "rates"):
        scenario = build_nonbank_scenario(event_type, 3)
        rumor = scenario.rumors[0]
        assert rumor.notes == [f"event-type:{event_type}"]
        credibility_row = get_parameter(f"Non-bank {event_type} rumor credibility")
        reach_row = get_parameter(f"Non-bank {event_type} rumor reach")
        assert rumor.credibility == float(credibility_row.value)
        assert rumor.reach == int(float(reach_row.value))
        assert claim_allowed(rumor.claim)
        label = label_nonbank_claim(event_type)
        assert label.market_related
        assert label.tone == "fear"
    rates = build_nonbank_scenario("rates", 3)
    assert rates.rate_path
    assert rates.shocks[0].kind == "unemployment"
    equity = build_nonbank_scenario("equity", 3)
    assert equity.rate_path == []
    assert equity.shocks == []
    assert equity.preset == "nonbank-equity"


def test_fake_harness_is_deterministic() -> None:
    first = fit_nonbank_parameters(11, generations=1, population=4, steps=4)
    second = fit_nonbank_parameters(11, generations=1, population=4, steps=4)
    assert first.vector == second.vector
    assert first.vector == (
        0.5068385534506369,
        0.7719495080619924,
        0.7449442157171865,
    )
    assert first.objective == second.objective
    assert first.generations == 1
    assert len(first.vector) == 3
    assert math.isfinite(first.objective)


def test_holdout_keeps_the_fit_vector() -> None:
    fit = fit_nonbank_parameters(5, generations=1, population=4, steps=4)
    before = fit.vector
    holdout = run_nonbank_holdout(fit.vector, 5, steps=4)
    assert fit.vector == before
    assert holdout.crypto_loss is not None and math.isfinite(holdout.crypto_loss)
    assert math.isfinite(holdout.rates_loss)
    assert holdout.crypto.n_prints == 4
    assert holdout.rates.n_prints == 4


def test_cli_tiny_fake_run(capsys: pytest.CaptureFixture[str]) -> None:
    runs_before = _repo_runs_snapshot()
    code = main(["--seed", "11", "--generations", "1", "--population", "4", "--eval-minutes", "4"])
    assert code == 0
    out = capsys.readouterr().out
    assert "equity_loss=" in out
    assert "E14 UNSET: peak-to-trough, days-to-bottom, volume-multiple" in out
    assert "E15 UNSET: days-to-bottom, volume-multiple" in out
    assert "E16 UNSET: volume-multiple" in out
    assert "E13: all point targets set" in out
    assert _repo_runs_snapshot() == runs_before


def test_svb_signature_placebo_paths_stay_bit_identical() -> None:
    builders = {
        "svb": _default_svb_scenario,
        "sig": _signature_scenario,
        "com": _community_scenario,
    }
    for name, build in builders.items():
        scenario = build(_SEED)
        assert _sha(scenario.model_dump(mode="json")) == _SCENARIO_HASH[name]
        log = run_scenario(scenario, _STEPS)
        rows = [
            event.model_dump(mode="json")
            for event in log
            if isinstance(event, (PriceTick, PaymentSettled, BankClosed))
        ]
        assert _sha(rows) == _PATH_HASH[name]
        day1, two_day, closed = measure_outflow(scenario, steps=_STEPS, param_vector=None)
        assert (day1, two_day, closed) == _OUTFLOW[name]


_ENV_KEYS = (
    "STAMPEDE_CALIB_GPU_SIM",
    "STAMPEDE_BACKEND",
    "STAMPEDE_CALIBRATION_ENGINE",
    "STAMPEDE_OFFLINE",
    "STAMPEDE_CALIB_DEPOSITORS",
)


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_checkpointed_scale_run_stops_early_without_fitted_rows(tmp_path: Path) -> None:
    """Patience and checkpoints stay on the fake 3-vector path and skip FITTED rows."""
    registry = Path(__file__).resolve().parents[1] / "params" / "registry.csv"
    before = registry.read_bytes()
    runs_before = _repo_runs_snapshot()
    code = main(
        [
            "--seed",
            "11",
            "--generations",
            "4",
            "--population",
            "4",
            "--eval-minutes",
            "2",
            "--patience",
            "2",
            "--checkpoint-every",
            "1",
            "--improvement-epsilon",
            "1e9",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "nb-scale",
        ]
    )
    assert code == 0
    checkpoint = json.loads((tmp_path / "nb-scale" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["generation"] == 1
    assert len(checkpoint["best_vector"]) == 3
    converged = json.loads((tmp_path / "nb-scale" / "converged.json").read_text(encoding="utf-8"))
    assert converged["fitted_rows_written"] is False
    assert converged["vector_kind"] == "scale"
    assert len(converged["best_vector"]) == 3
    log = (tmp_path / "nb-scale" / "calibration.log").read_text(encoding="utf-8")
    assert "Early stop" in log
    assert registry.read_bytes() == before
    assert _repo_runs_snapshot() == runs_before
    board = Path(__file__).resolve().parents[1] / "docs" / "validation-board-nb-scale.md"
    assert not board.exists()


def test_default_runs_dir_never_writes_fitted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long fake run with no --runs-dir does not patch params/registry.csv."""
    registry = Path(__file__).resolve().parents[1] / "params" / "registry.csv"
    before = registry.read_bytes()
    runs_before = _repo_runs_snapshot()
    monkeypatch.setattr("calibration.run_nonbank_fit._DEFAULT_RUNS", tmp_path)
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "fake")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    code = main(
        [
            "--run-id",
            "t",
            "--generations",
            "1",
            "--population",
            "2",
            "--patience",
            "1",
        ]
    )
    assert code == 0
    assert registry.read_bytes() == before
    converged = json.loads((tmp_path / "t" / "converged.json").read_text(encoding="utf-8"))
    assert converged["fitted_rows_written"] is False
    assert converged["fitted_rows"] == {}
    assert "persist_command" in converged
    assert "--kind nonbank" in converged["persist_command"]
    assert str(tmp_path / "t") in converged["persist_command"]
    assert _repo_runs_snapshot() == runs_before


def test_gpu_sim_cli_batches_equity_and_holdouts(tmp_path: Path) -> None:
    """NumPy stands in for the GPU. The run writes checkpoints and no FITTED rows."""
    from engine.agents.population import clear_sample_population_cache

    registry = Path(__file__).resolve().parents[1] / "params" / "registry.csv"
    before = registry.read_bytes()
    runs_before = _repo_runs_snapshot()
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    os.environ["STAMPEDE_CALIB_DEPOSITORS"] = "8"
    os.environ["STAMPEDE_OFFLINE"] = "1"
    os.environ.pop("STAMPEDE_CALIB_GPU_SIM", None)
    os.environ.pop("STAMPEDE_BACKEND", None)
    os.environ.pop("STAMPEDE_CALIBRATION_ENGINE", None)
    clear_sample_population_cache()
    try:
        code = main(
            [
                "--gpu",
                "--gpu-sim",
                "--seed",
                "11",
                "--generations",
                "1",
                "--population",
                "2",
                "--eval-minutes",
                "3",
                "--patience",
                "8",
                "--checkpoint-every",
                "1",
                "--workers",
                "1",
                "--runs-dir",
                str(tmp_path),
                "--run-id",
                "nb-gpu",
            ]
        )
    finally:
        _restore_env(saved)
        clear_sample_population_cache()
    assert code == 0
    converged = json.loads((tmp_path / "nb-gpu" / "converged.json").read_text(encoding="utf-8"))
    assert converged["fitted_rows_written"] is False
    assert converged["vector_kind"] == "behavioral"
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    assert len(converged["best_vector"]) == dim
    assert math.isfinite(converged["holdout"]["crypto_loss"])
    assert math.isfinite(converged["holdout"]["rates_loss"])
    assert math.isfinite(converged["best_objective"])
    checkpoint = json.loads((tmp_path / "nb-gpu" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["generation"] == 1
    assert len(checkpoint["best_vector"]) == dim
    assert (tmp_path / "nb-gpu" / "best_vector.json").is_file()
    log = (tmp_path / "nb-gpu" / "calibration.log").read_text(encoding="utf-8")
    assert "gpu-sim on" in log
    # The batching is what's under test, not the backend: CuPy is used when the machine has it.
    assert re.search(r"array module=(cpu|gpu)\b", log), "gpu-sim log line names no array module"
    assert registry.read_bytes() == before
    assert _repo_runs_snapshot() == runs_before


def test_engine_workers_match_and_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy per-candidate eval is deterministic and agrees across worker counts."""
    from engine.agents.population import clear_sample_population_cache

    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "8")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    kwargs = {"generations": 1, "population": 2, "steps": 2, "engine_candidates": True}
    first = fit_nonbank_parameters(3, workers=1, **kwargs)
    second = fit_nonbank_parameters(3, workers=1, **kwargs)
    pooled = fit_nonbank_parameters(3, workers=2, **kwargs)
    assert first.vector == second.vector
    assert first.objective == second.objective
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    assert len(first.vector) == dim
    assert first.vector == pooled.vector
    assert first.objective == pooled.objective
    assert math.isfinite(first.objective)


# --- SVB-27c: crypto (E15) holdout is evaluated, or reported as not evaluated -------------------


def test_shape_targets_has_targets_only_when_a_point_is_set() -> None:
    from calibration.nonbank import ShapeTargets

    assert not ShapeTargets("E15", None, None, None).has_targets()
    assert ShapeTargets("E15", 0.9999996, None, None).has_targets()


def test_holdout_payload_reports_null_without_targets() -> None:
    from calibration.nonbank import NonbankHoldoutResult, NonbankTargets, PricePathStats, ShapeTargets
    from calibration.run_nonbank_fit import holdout_payload

    stats = PricePathStats(peak_to_trough=0.5, days_to_bottom=3.0, volume_multiple=2.0, n_prints=10)
    empty = ShapeTargets("E15", None, None, None)
    rates = ShapeTargets("E16", 0.339, 33.0, None)
    targets = NonbankTargets(equity=rates, gamestop=empty, crypto=empty, rates=rates)
    payload = holdout_payload(NonbankHoldoutResult(0.0, 43.5, stats, stats), targets)
    assert payload["crypto_loss"] is None
    assert payload["crypto_status"] == "no targets"
    assert payload["rates_loss"] == 43.5
    assert payload["rates_status"] == "evaluated"


def test_e15_has_a_set_peak_to_trough_target() -> None:
    """The registry row from calibration/registry_updates/svb27c_e15_luna_target.json is applied."""
    targets = load_nonbank_targets()
    assert targets.crypto.has_targets()
    assert targets.crypto.peak_to_trough is not None and 0.999 < targets.crypto.peak_to_trough <= 1.0


# --- SVB-29: days-to-bottom targets on a trading-day clock, unreachable ones excluded -----------


def test_reachable_targets_excludes_a_days_target_the_run_cannot_reach() -> None:
    from calibration.nonbank import ShapeTargets, reachable_targets

    e16 = ShapeTargets("E16", 0.339, 23.0, None)
    short, why = reachable_targets(e16, steps=390, minutes_per_day=390.0)
    assert short.days_to_bottom is None and short.peak_to_trough == 0.339
    assert why and "8970" in why[0]
    full, why_full = reachable_targets(e16, steps=8970, minutes_per_day=390.0)
    assert full.days_to_bottom == 23.0 and why_full == ()


def test_trading_day_registry_rows() -> None:
    from params.loader import get_parameter

    assert float(get_parameter("Non-bank sim minutes per day").value) == 390.0
    targets = load_nonbank_targets()
    assert targets.rates.days_to_bottom == 23.0          # trading days, not the 33 calendar days
    assert targets.equity.days_to_bottom is not None
    assert targets.equity.days_to_bottom <= 390.0 / 390.0  # E13's one session fits a 390-minute fit


# --- SVB-38b: per-statistic shape terms and split crypto/rates holdout horizons ---------------


def test_shape_loss_terms_sum_to_shape_loss_and_unset_is_none() -> None:
    stats = summarize_price_path(
        [0.0, 10.0, 20.0, 30.0],
        [100.0, 110.0, 80.0, 90.0],
        [10.0, 10.0, 40.0, 10.0],
        minutes_per_day=1440.0,
    )
    target = ShapeTargets("unit", 0.5, 1.0, 4.0)
    weights = (1.0, 1.0, 1.0)
    terms = shape_loss_terms(stats, target, weights=weights)
    assert set(terms) == {"peak_to_trough", "days_to_bottom", "volume_multiple"}
    assert all(value is not None for value in terms.values())
    assert sum(float(v) for v in terms.values() if v is not None) == shape_loss(
        stats, target, weights=weights
    )
    blank = ShapeTargets("E14", None, None, None)
    unset = shape_loss_terms(stats, blank, weights=weights)
    assert unset == {
        "peak_to_trough": None,
        "days_to_bottom": None,
        "volume_multiple": None,
    }
    partial = ShapeTargets("E16", 0.339, 33.0, None)
    partial_terms = shape_loss_terms(stats, partial, weights=weights)
    assert partial_terms["volume_multiple"] is None
    assert partial_terms["peak_to_trough"] is not None
    assert partial_terms["days_to_bottom"] is not None
    assert sum(v for v in partial_terms.values() if v is not None) == shape_loss(
        stats, partial, weights=weights
    )


def test_summary_splits_holdout_horizons(tmp_path: Path) -> None:
    code = main(
        [
            "--seed",
            "11",
            "--generations",
            "1",
            "--population",
            "4",
            "--eval-minutes",
            "12",
            "--rates-holdout-minutes",
            "30",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "nb-horizons",
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "nb-horizons" / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_minutes_holdout_rates"] == 30
    assert summary["eval_minutes_holdout_crypto"] == 12
    assert "eval_minutes_holdout" not in summary
    assert summary["equity_terms"] is None
    assert summary["crypto_terms"] is None
    assert summary["rates_terms"] is None


def test_crypto_holdout_minutes_in_summary(tmp_path: Path) -> None:
    code = main(
        [
            "--seed",
            "11",
            "--generations",
            "1",
            "--population",
            "4",
            "--crypto-holdout-minutes",
            "40",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "nb-crypto-h",
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "nb-crypto-h" / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_minutes_holdout_crypto"] == 40


def test_rates_holdout_minutes_takes_long_run_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_long(args: object) -> int:
        seen["args"] = args
        return 0

    monkeypatch.setattr("calibration.run_nonbank_fit._long_main", fake_long)
    monkeypatch.setattr("calibration.run_nonbank_fit._legacy_main", lambda args: 99)
    assert main(["--rates-holdout-minutes", "30"]) == 0
    assert seen["args"].rates_holdout_minutes == 30  # type: ignore[union-attr]
