"""SVB-38c: score-only non-bank re-score, per-term status, and the multi-day path profile.

Offline and CPU-only. The CLI tests use the fake engine and the fake 3-vector, so no
behavioral vector is applied and nothing touches the GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from calibration.nonbank import (
    NonbankHoldoutResult,
    PricePathStats,
    ShapeTargets,
    event_activity_by_day,
    fit_nonbank_parameters,
    load_nonbank_targets,
    price_path_profile,
    reachable_targets,
    run_nonbank_holdout,
    summarize_price_path,
    term_status,
)
from calibration.run_nonbank_fit import main
from params.loader import get_parameter

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "params" / "registry.csv"
_RUNS = _REPO / "calibration" / "runs"


def _repo_runs_snapshot() -> frozenset[tuple[str, int, int]]:
    """Files under the repo's calibration/runs with size and mtime (SVB-46).

    The dev checkout keeps real, untracked runs there; a test must leave it as it found it.
    """
    if not _RUNS.exists():
        return frozenset()
    return frozenset(
        (str(path.relative_to(_RUNS)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in _RUNS.rglob("*")
        if path.is_file()
    )


def _three_session_path() -> tuple[list[float], list[float], list[float]]:
    """Falls on day 0, falls further on day 1, flat on day 2 (390-minute days, 1-minute ticks)."""
    times = [float(t) for t in range(3 * 390)]
    prices: list[float] = []
    for t in times:
        if t < 390:
            prices.append(100.0 - 10.0 * t / 389.0)  # 100 -> 90
        elif t < 780:
            prices.append(90.0 - 30.0 * (t - 390.0) / 389.0)  # 90 -> 60
        else:
            prices.append(60.0)
    volumes = [10.0 if t < 390 else 20.0 for t in times]
    return times, prices, volumes


# --- price_path_profile ------------------------------------------------------------------------


def test_profile_sees_a_shock_that_keeps_falling_after_day_one() -> None:
    times, prices, volumes = _three_session_path()
    profile = price_path_profile(times, prices, volumes, minutes_per_day=390.0, steps=1170)
    assert profile["n_prints"] == 1170
    assert profile["covers_horizon"] is True
    assert profile["clock"] == "uniform 1-minute"
    assert [row["day"] for row in profile["days"]] == [0, 1, 2]
    assert profile["days_with_new_low_after_day1"] == [1]
    assert profile["new_low_after_day1"] is True
    assert math.isclose(profile["share_of_drawdown_after_day1"], 0.3 / 0.4)
    # Same statistic the loss scores, on the day-one prefix and on the whole path.
    day1 = summarize_price_path(times[:390], prices[:390], volumes[:390], minutes_per_day=390.0)
    full = summarize_price_path(times, prices, volumes, minutes_per_day=390.0)
    assert profile["peak_to_trough_day1"] == day1.peak_to_trough
    assert profile["peak_to_trough_full"] == full.peak_to_trough
    assert math.isclose(profile["peak_to_trough_full"], 0.4)
    assert math.isclose(profile["peak_to_trough_added_after_day1"], 0.4 - 0.1)
    assert profile["trough_minute"] == 779.0
    assert profile["days"][1]["volume"] == 390 * 20.0


def test_profile_reports_a_path_that_goes_flat_after_day_one() -> None:
    times = [float(t) for t in range(780)]
    prices = [100.0 - 5.0 * min(t, 389.0) / 389.0 for t in times]
    volumes = [1.0] * len(times)
    profile = price_path_profile(times, prices, volumes, minutes_per_day=390.0, steps=780)
    assert profile["new_low_after_day1"] is False
    assert profile["share_of_drawdown_after_day1"] == 0.0
    assert profile["days_with_new_low_after_day1"] == []
    assert profile["peak_to_trough_added_after_day1"] == 0.0


def test_profile_flags_a_truncated_or_non_uniform_clock() -> None:
    # Asked for 3120 minutes, got one session plus a 15-minute overnight to t=1425.
    times = [float(t) for t in range(390)] + [390.0 + 15.0 * k for k in range(70)]
    prices = [100.0] * len(times)
    volumes = [1.0] * len(times)
    profile = price_path_profile(times, prices, volumes, minutes_per_day=390.0, steps=3120)
    assert profile["covers_horizon"] is False
    assert profile["max_gap_minutes"] == 15.0
    assert profile["clock"].startswith("non-uniform")
    empty = price_path_profile([], [], [], minutes_per_day=390.0, steps=10)
    assert empty["n_prints"] == 0 and empty["covers_horizon"] is False


def test_event_activity_by_day_counts_types_per_bucket() -> None:
    class Tick:
        def __init__(self, t: float) -> None:
            self.t = t

    class Other:
        def __init__(self, t: float) -> None:
            self.t = t

    log = [Tick(0.0), Tick(389.0), Other(390.0), Tick(400.0), object()]
    counts = event_activity_by_day(log, minutes_per_day=390.0)
    assert counts == {"0": {"Tick": 2}, "1": {"Other": 1, "Tick": 1}}


# --- term_status -------------------------------------------------------------------------------


def test_term_status_names_unset_excluded_and_evaluated_terms() -> None:
    weights = (1.0, 1.0, 1.0)
    targets = load_nonbank_targets()
    crypto, crypto_why = reachable_targets(targets.crypto, steps=3120, minutes_per_day=390.0)
    status = term_status(targets.crypto, crypto, crypto_why, weights=weights)
    assert status["peak_to_trough"] == "evaluated"
    assert status["days_to_bottom"].startswith("unset:")
    assert "Non-bank E15 LUNA days to bottom" in status["days_to_bottom"]
    assert status["volume_multiple"].startswith("unset:")

    e16 = ShapeTargets("E16", 0.339, 23.0, None)
    short, why = reachable_targets(e16, steps=390, minutes_per_day=390.0)
    short_status = term_status(e16, short, why, weights=weights)
    assert short_status["days_to_bottom"].startswith("excluded:")
    assert "8970" in short_status["days_to_bottom"]
    full, why_full = reachable_targets(e16, steps=8970, minutes_per_day=390.0)
    assert term_status(e16, full, why_full, weights=weights)["days_to_bottom"] == "evaluated"
    assert term_status(e16, full, why_full, weights=(1.0, 0.0, 1.0))["days_to_bottom"] == "weight 0"


# --- holdout result ----------------------------------------------------------------------------


def test_holdout_carries_profiles_and_timings_without_changing_losses() -> None:
    fit = fit_nonbank_parameters(5, generations=1, population=4, steps=4)
    holdout = run_nonbank_holdout(fit.vector, 5, steps=4, crypto_steps=9, rates_steps=6)
    again = run_nonbank_holdout(fit.vector, 5, steps=4, crypto_steps=9, rates_steps=6)
    assert holdout.crypto_loss == again.crypto_loss
    assert holdout.rates_loss == again.rates_loss
    assert holdout.crypto_profile is not None and holdout.rates_profile is not None
    assert holdout.crypto_profile["n_prints"] == holdout.crypto.n_prints == 9
    assert holdout.rates_profile["n_prints"] == holdout.rates.n_prints == 6
    assert holdout.crypto_profile["steps"] == 9
    assert holdout.crypto_seconds is not None and holdout.crypto_seconds >= 0.0
    # The old positional constructor still works (the new fields default to None).
    stats = PricePathStats(0.1, 0.0, 1.0, 2)
    legacy = NonbankHoldoutResult(0.0, 1.0, stats, stats)
    assert legacy.crypto_profile is None and legacy.rates_seconds is None


# --- score-only CLI ----------------------------------------------------------------------------


def _source_run(tmp_path: Path) -> tuple[Path, float]:
    """A converged.json in the pre-SVB-38b shape of nonbank-svb37 (single holdout key)."""
    fit = fit_nonbank_parameters(11, generations=1, population=4, steps=4)
    holdout = run_nonbank_holdout(fit.vector, 11, steps=4)
    src_dir = tmp_path / "src-run"
    src_dir.mkdir()
    payload = {
        "run_id": "src-run",
        "best_seed": 11,
        "best_objective": fit.objective,
        "best_vector": list(fit.vector),
        "eval_minutes_fit": 4,
        "eval_minutes_holdout": 4,
        "fitted_rows_written": True,
        "holdout": {
            "crypto_loss": holdout.crypto_loss,
            "rates_loss": holdout.rates_loss,
            "crypto_excluded": [],
            "rates_excluded": [],
        },
    }
    path = src_dir / "converged.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, float(fit.objective)


def test_score_only_rescore_writes_summary_and_nothing_else(tmp_path: Path) -> None:
    before = _REGISTRY.read_bytes()
    runs_before = _repo_runs_snapshot()
    source, objective = _source_run(tmp_path)
    runs = tmp_path / "runs"
    code = main(
        [
            "--score-only",
            str(source),
            "--crypto-holdout-minutes",
            "20",
            "--rates-holdout-minutes",
            "12",
            "--runs-dir",
            str(runs),
            "--run-id",
            "rescore",
        ]
    )
    assert code == 0
    run_dir = runs / "rescore"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "score_only"
    assert summary["fitted_rows_written"] is False
    assert summary["fitted_rows"] == {}
    assert summary["persist_command"] is None
    assert summary["eval_minutes_holdout_crypto"] == 20
    assert summary["eval_minutes_holdout_rates"] == 12
    assert summary["eval_minutes_fit"] == 4
    assert summary["seed"] == 11
    # Same vector, seed and horizon as the source fit: the equity objective reproduces.
    assert summary["equity_objective"] == pytest.approx(objective, rel=0.0, abs=1e-12)
    assert summary["comparison"]["equity_objective"]["same_horizon"] is True
    # Baseline defaults to the source's own (pre-38b) holdout horizon and reproduces it.
    assert summary["baseline"]["eval_minutes_holdout_crypto"] == 4
    assert summary["comparison"]["crypto_loss_baseline"]["same_horizon"] is True
    assert summary["comparison"]["crypto_loss_baseline"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert summary["comparison"]["rates_loss_baseline"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert summary["comparison"]["crypto_loss_new_horizon"]["same_horizon"] is False
    assert set(summary["term_status"]) == {"E13", "E15", "E16"}
    assert summary["term_status"]["E15"]["peak_to_trough"] == "evaluated"
    assert summary["term_status"]["E16"]["days_to_bottom"].startswith("excluded:")
    assert summary["crypto_profile"]["n_prints"] == 20
    assert summary["rates_profile"]["n_prints"] == 12
    assert summary["timing_model"]["crypto"]["from_minutes"] == [4, 20]
    assert summary["source"]["fitted_rows_written"] is True  # recorded from the source, not acted on
    # No fit artifacts, nothing persist_fit can load, registry untouched, no repo runs dir.
    assert sorted(p.name for p in run_dir.iterdir()) == ["calibration.log", "summary.json"]
    assert "best_vector" not in summary
    from calibration.persist_fit import _load_best

    with pytest.raises(KeyError):
        _load_best(run_dir)
    assert _REGISTRY.read_bytes() == before
    assert _repo_runs_snapshot() == runs_before
    assert json.loads(source.read_text(encoding="utf-8"))["run_id"] == "src-run"


def test_score_only_baseline_zero_skips_the_baseline(tmp_path: Path) -> None:
    source, _ = _source_run(tmp_path)
    code = main(
        [
            "--score-only",
            str(source),
            "--crypto-holdout-minutes",
            "8",
            "--rates-holdout-minutes",
            "8",
            "--baseline-holdout-minutes",
            "0",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "nobase",
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "runs" / "nobase" / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"] is None
    assert summary["timing_model"]["crypto"] is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--generations", "3"],
        ["--population", "12"],
        ["--patience", "30"],
        ["--extra-seeds", "3061"],
        ["--resume"],
        ["--holdout-minutes", "390"],
    ],
)
def test_score_only_rejects_fit_flags(tmp_path: Path, extra: list[str]) -> None:
    source, _ = _source_run(tmp_path)
    argv = [
        "--score-only",
        str(source),
        "--crypto-holdout-minutes",
        "8",
        "--rates-holdout-minutes",
        "8",
        "--runs-dir",
        str(tmp_path / "runs"),
        *extra,
    ]
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_score_only_needs_both_holdout_horizons(tmp_path: Path) -> None:
    source, _ = _source_run(tmp_path)
    with pytest.raises(SystemExit) as info:
        main(["--score-only", str(source), "--crypto-holdout-minutes", "8"])
    assert info.value.code == 2


def test_score_only_refuses_the_source_dir_and_a_wrong_layout(tmp_path: Path) -> None:
    source, _ = _source_run(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--score-only",
                str(source),
                "--crypto-holdout-minutes",
                "8",
                "--rates-holdout-minutes",
                "8",
                "--runs-dir",
                str(source.parent.parent),
                "--run-id",
                source.parent.name,
            ]
        )
    assert sorted(p.name for p in source.parent.iterdir()) == ["converged.json"]
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"best_vector": [0.5] * (dim + 1), "best_seed": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--score-only",
                str(wrong),
                "--eval-minutes",
                "4",
                "--crypto-holdout-minutes",
                "8",
                "--rates-holdout-minutes",
                "8",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--run-id",
                "wrong",
            ]
        )


def test_long_run_summary_lists_term_status_and_profiles(tmp_path: Path) -> None:
    code = main(
        [
            "--seed",
            "11",
            "--generations",
            "1",
            "--population",
            "4",
            "--eval-minutes",
            "6",
            "--crypto-holdout-minutes",
            "10",
            "--rates-holdout-minutes",
            "8",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "nb-38c",
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "nb-38c" / "summary.json").read_text(encoding="utf-8"))
    assert summary["fitted_rows_written"] is False
    assert summary["term_status"]["E15"]["days_to_bottom"].startswith("unset:")
    assert summary["crypto_profile"]["n_prints"] == 10
    assert summary["rates_profile"]["n_prints"] == 8


def test_nonbank_launcher_has_no_registry_writer() -> None:
    """PA-A guard, kept static: the launcher never imports a registry writer."""
    source = (_REPO / "calibration" / "run_nonbank_fit.py").read_text(encoding="utf-8")
    for name in ("write_nonbank_fitted_vector", "patch_registry_rows", "write_fitted_registry"):
        assert name not in source
