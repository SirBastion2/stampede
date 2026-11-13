"""CLI for the non-bank price-path fit.

Tiny registry defaults keep an accidental launch short. The post-SVB command
is in ``docs/ARCHITECTURE.md`` §20.3. With no run-control flags this module
does not write under ``calibration/runs/`` and does not write FITTED rows.

``--gpu``, ``--gpu-sim``, ``--gpu-full``, and ``--workers`` match ``calibration.run_fit``.
The checkpointed long run writes ``checkpoint.json`` and ``converged.json`` under
``calibration/runs/<run-id>/``. It never writes FITTED rows; saving is
manual via ``python -m calibration.persist_fit --kind nonbank``.

``--score-only <converged.json>`` (SVB-38c) does not fit. It re-scores a finished
run's vector on new holdout horizons, writes only ``summary.json`` (plus the log) in a
new run dir, and leaves nothing that ``persist_fit`` can read as a best vector.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from calibration.diagnostics import diagnostics_dir, write_run_diagnostics
from calibration.fit import configure_calibration_gpu, load_generation_checkpoint
from calibration.nonbank import (
    NonbankHoldoutResult,
    NonbankTargets,
    PricePathStats,
    ShapeTargets,
    build_nonbank_scenario,
    fit_nonbank_parameters,
    load_nonbank_targets,
    measure_nonbank_observables,
    nonbank_request_bound_penalty,
    objective_nonbank,
    reachable_targets,
    run_nonbank_harness,
    run_nonbank_holdout,
    shape_loss_terms,
    term_status,
)
from engine.agents.population import population_balance_backend, population_device_description
from params.loader import get_parameter

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RUNS = _REPO_ROOT / "calibration" / "runs"


def _configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m27.nonbank")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _gpu_mem_mib() -> str:
    try:
        import cupy as cp

        free, total = cp.cuda.Device().mem_info
        used = total - free
        return f"{used / (1024**2):.0f} MiB used / {total / (1024**2):.0f} MiB total"
    except Exception:
        return "n/a"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M27 non-bank price-path calibration "
            "(fake engine unless STAMPEDE_CALIBRATION_ENGINE=real)."
        ),
    )
    parser.add_argument("--run-id", default="", help="Stable run identifier (default: timestamp).")
    parser.add_argument("--seed", type=int, default=3060, help="Fit seed and scenario seed.")
    parser.add_argument(
        "--extra-seeds",
        default="",
        help="Comma-separated restart seeds. Empty keeps a single seed.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="CMA-ES-style generations (default: registry tiny count).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Population size (default: registry tiny count).",
    )
    parser.add_argument(
        "--eval-minutes",
        type=int,
        default=None,
        help="Simulated minutes per path (default: registry tiny horizon).",
    )
    parser.add_argument(
        "--holdout-minutes",
        type=int,
        default=None,
        help="Simulated minutes for crypto and rates holdouts (default: the fit horizon).",
    )
    parser.add_argument(
        "--rates-holdout-minutes",
        type=int,
        default=None,
        help=(
            "Simulated minutes for the rates (E16) holdout only. Its 23-trading-day days-to-bottom "
            "target needs 8970 at 390 minutes per day; shorter runs exclude that term and report it."
        ),
    )
    parser.add_argument(
        "--crypto-holdout-minutes",
        type=int,
        default=None,
        help=(
            "Simulated minutes for the crypto (E15) holdout only. Default: the shared holdout "
            "horizon. A multi-day run such as 4320 or 8640 keeps days-to-bottom targets reachable."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Checkpoint interval in generations (default: 2 on a checkpointed run).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop a seed after this many generations without relative improvement (default: 6).",
    )
    parser.add_argument(
        "--improvement-epsilon",
        type=float,
        default=None,
        help="Relative objective improvement required to reset patience (default: 1e-3).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.json in the run dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel candidate evaluators (default: 1, or 2 with --gpu). "
            "Does not split a GPU batch."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require GPU path: STAMPEDE_BACKEND=gpu (M05 balance solve + M02/M03/M06 CuPy).",
    )
    parser.add_argument(
        "--gpu-sim",
        action="store_true",
        help=(
            "Batched depositor eval (STAMPEDE_CALIB_GPU_SIM=1). Requires --gpu. "
            "Equity, crypto, and rates use the same kernel. "
            "The legacy per-depositor engine stays in use unless both are set."
        ),
    )
    parser.add_argument(
        "--gpu-full",
        action="store_true",
        help=(
            "Whole-tick array path (STAMPEDE_CALIB_GPU_FULL=1): attention, "
            "contagion, market, and payments on CuPy (NumPy if no device). "
            "Implies --gpu and --gpu-sim."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Parent directory for run artifacts (default: calibration/runs).",
    )
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        help=(
            "SVB-38c: do not fit. Re-score the vector in this converged.json, summary.json or "
            "best_vector.json on the holdout horizons given. Needs --crypto-holdout-minutes and "
            "--rates-holdout-minutes. Writes summary.json only; never FITTED rows."
        ),
    )
    parser.add_argument(
        "--baseline-holdout-minutes",
        type=int,
        default=None,
        help=(
            "Score-only: also score crypto and rates at this horizon for a like-for-like "
            "comparison with the source run. Default: the source run's own holdout horizons. "
            "0 skips the baseline."
        ),
    )
    args = parser.parse_args(argv)
    if args.score_only is not None:
        _check_score_only_args(parser, args)
    return args


_FIT_ONLY_FLAGS: tuple[tuple[str, str], ...] = (
    ("generations", "--generations"),
    ("population", "--population"),
    ("patience", "--patience"),
    ("checkpoint_every", "--checkpoint-every"),
    ("improvement_epsilon", "--improvement-epsilon"),
    ("holdout_minutes", "--holdout-minutes"),
)


def _check_score_only_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """A re-score never fits, so fit-only flags are an error rather than silently ignored."""
    used = [flag for attr, flag in _FIT_ONLY_FLAGS if getattr(args, attr) is not None]
    if args.resume:
        used.append("--resume")
    if args.extra_seeds.strip():
        used.append("--extra-seeds")
    if used:
        parser.error(f"--score-only does not fit; remove {', '.join(used)}")
    missing = [
        flag
        for attr, flag in (
            ("crypto_holdout_minutes", "--crypto-holdout-minutes"),
            ("rates_holdout_minutes", "--rates-holdout-minutes"),
        )
        if getattr(args, attr) is None
    ]
    if missing:
        parser.error(f"--score-only needs {' and '.join(missing)}")
    for attr, flag in (
        ("crypto_holdout_minutes", "--crypto-holdout-minutes"),
        ("rates_holdout_minutes", "--rates-holdout-minutes"),
    ):
        if int(getattr(args, attr)) < 1:
            parser.error(f"{flag} must be at least 1")
    if args.baseline_holdout_minutes is not None and args.baseline_holdout_minutes < 0:
        parser.error("--baseline-holdout-minutes must be 0 (skip) or positive")


def _uses_run_controls(args: argparse.Namespace) -> bool:
    """True when the caller asked for the long-run controls, not the tiny fake fit."""
    return any(
        (
            args.gpu,
            args.gpu_sim,
            args.gpu_full,
            args.resume,
            bool(args.run_id),
            bool(args.extra_seeds.strip()),
            args.workers is not None,
            args.patience is not None,
            args.checkpoint_every is not None,
            args.runs_dir is not None,
            args.holdout_minutes is not None,
            args.improvement_epsilon is not None,
            args.rates_holdout_minutes is not None,
            args.crypto_holdout_minutes is not None,
            args.score_only is not None,
            args.baseline_holdout_minutes is not None,
        )
    )


def _format_unset(label: str, fields: tuple[str, ...]) -> str:
    if not fields:
        return f"{label}: all point targets set"
    return f"{label} UNSET: {', '.join(fields)}"


def _loss_or_none(loss: float, target: ShapeTargets) -> float | None:
    """A holdout with no set targets isn't evaluated: report None (JSON null), not 0.0."""
    return float(loss) if target.has_targets() else None


def _loss_text(loss: float, target: ShapeTargets) -> str:
    value = _loss_or_none(loss, target)
    return "not evaluated (no set targets)" if value is None else f"{value:.6g}"


def _holdout_horizons(args: argparse.Namespace) -> tuple[int | None, int | None, int | None]:
    """Shared, crypto, and rates holdout step counts as recorded on the summary."""
    shared = args.holdout_minutes if args.holdout_minutes is not None else args.eval_minutes
    crypto = args.crypto_holdout_minutes if args.crypto_holdout_minutes is not None else shared
    rates = args.rates_holdout_minutes if args.rates_holdout_minutes is not None else shared
    return shared, crypto, rates


def _registry_eval_minutes() -> int:
    return int(float(get_parameter("Non-bank eval simulation minutes").value))


def _resolved_steps(value: int | None) -> int:
    return int(value) if value is not None else _registry_eval_minutes()


def _json_terms(terms: dict[str, float | None]) -> dict[str, float | None]:
    return {key: (None if value is None else float(value)) for key, value in terms.items()}


def _target_used_payload(target: ShapeTargets, excluded: tuple[str, ...]) -> dict[str, object]:
    return {
        "peak_to_trough": target.peak_to_trough,
        "days_to_bottom": target.days_to_bottom,
        "volume_multiple": target.volume_multiple,
        "excluded": list(excluded),
    }


def _stats_payload(stats: PricePathStats) -> dict[str, float | int]:
    return {
        "peak_to_trough": float(stats.peak_to_trough),
        "days_to_bottom": float(stats.days_to_bottom),
        "volume_multiple": float(stats.volume_multiple),
        "n_prints": int(stats.n_prints),
    }


def _scale_term_fields() -> dict[str, object]:
    return {
        "equity_stats": None,
        "equity_terms": None,
        "equity_request_penalty": None,
        "crypto_terms": None,
        "rates_terms": None,
        "targets_used": None,
    }


def _registry_weights() -> tuple[float, float, float]:
    return (
        float(get_parameter("Non-bank drawdown objective weight").value),
        float(get_parameter("Non-bank days objective weight").value),
        float(get_parameter("Non-bank volume objective weight").value),
    )


def _term_status_payload(
    targets: NonbankTargets,
    *,
    equity_steps: int,
    crypto_steps: int,
    rates_steps: int,
) -> dict[str, dict[str, str]]:
    """SVB-38c: for each event, which shape terms are evaluated, unset, or excluded, and why."""
    minutes_per_day = float(get_parameter("Non-bank sim minutes per day").value)
    weights = _registry_weights()
    out: dict[str, dict[str, str]] = {}
    for label, raw, steps in (
        ("E13", targets.equity, equity_steps),
        ("E15", targets.crypto, crypto_steps),
        ("E16", targets.rates, rates_steps),
    ):
        used, excluded = reachable_targets(raw, steps=steps, minutes_per_day=minutes_per_day)
        out[label] = term_status(raw, used, excluded, weights=weights)
    return out


def _behavioral_term_fields(
    *,
    vector: tuple[float, ...],
    seed: int,
    equity_steps: int,
    crypto_steps: int,
    rates_steps: int,
    holdout: NonbankHoldoutResult,
    targets: NonbankTargets,
) -> dict[str, object]:
    import numpy as np

    minutes_per_day = float(get_parameter("Non-bank sim minutes per day").value)
    arr = np.asarray(vector, dtype=np.float64)
    stats, fraction = measure_nonbank_observables(
        build_nonbank_scenario("equity", seed),
        steps=equity_steps,
        param_vector=arr,
    )
    weights = _registry_weights()
    equity_target, equity_excluded = reachable_targets(
        targets.equity, steps=equity_steps, minutes_per_day=minutes_per_day
    )
    crypto_target, crypto_excluded = reachable_targets(
        targets.crypto, steps=crypto_steps, minutes_per_day=minutes_per_day
    )
    rates_target, rates_excluded = reachable_targets(
        targets.rates, steps=rates_steps, minutes_per_day=minutes_per_day
    )
    return {
        "equity_stats": _stats_payload(stats),
        "equity_terms": _json_terms(shape_loss_terms(stats, equity_target, weights=weights)),
        "equity_request_penalty": float(nonbank_request_bound_penalty(fraction)),
        "crypto_terms": _json_terms(
            shape_loss_terms(holdout.crypto, crypto_target, weights=weights)
        ),
        "rates_terms": _json_terms(shape_loss_terms(holdout.rates, rates_target, weights=weights)),
        "targets_used": {
            "E13": _target_used_payload(equity_target, equity_excluded),
            "E15": _target_used_payload(crypto_target, crypto_excluded),
            "E16": _target_used_payload(rates_target, rates_excluded),
        },
    }


def holdout_payload(holdout: NonbankHoldoutResult, targets: NonbankTargets) -> dict[str, object]:
    """JSON for the holdout block: null losses and an explicit status when a holdout has no targets."""
    return {
        "crypto_loss": _loss_or_none(holdout.crypto_loss, targets.crypto),
        "rates_loss": _loss_or_none(holdout.rates_loss, targets.rates),
        "crypto_status": "evaluated" if targets.crypto.has_targets() else "no targets",
        "rates_status": "evaluated" if targets.rates.has_targets() else "no targets",
        "crypto_excluded": list(getattr(holdout, "crypto_excluded", ())),
        "rates_excluded": list(getattr(holdout, "rates_excluded", ())),
    }


def _print_summary(
    fit_seed: int,
    generations: int,
    objective: float,
    vector: tuple[float, ...],
    holdout: NonbankHoldoutResult,
    targets: NonbankTargets,
) -> None:
    print(f"seed={fit_seed} generations={generations} equity_loss={objective:.6g}")
    print(f"vector={','.join(f'{item:.6g}' for item in vector)}")
    print(f"crypto_holdout_loss={_loss_text(holdout.crypto_loss, targets.crypto)}")
    print(f"rates_holdout_loss={_loss_text(holdout.rates_loss, targets.rates)}")
    print(_format_unset("E13", targets.equity.unset_fields()))
    print(_format_unset("E14", targets.gamestop.unset_fields()))
    print(_format_unset("E15", targets.crypto.unset_fields()))
    print(_format_unset("E16", targets.rates.unset_fields()))


def _legacy_main(args: argparse.Namespace) -> int:
    targets = load_nonbank_targets()
    run_id = f"m27-nonbank-{args.seed}"
    diag_dir = diagnostics_dir(run_id)
    fit, holdout = run_nonbank_harness(
        args.seed,
        generations=args.generations,
        population=args.population,
        steps=args.eval_minutes,
        diagnostics_dir=diag_dir,
    )
    write_run_diagnostics(
        diag_dir,
        run_id=run_id,
        extra={"objective": float(fit.objective), "seed": fit.seed},
    )
    _print_summary(fit.seed, fit.generations, fit.objective, fit.vector, holdout, targets)
    return 0


def _relative_gain(previous: float, current: float) -> float:
    if previous <= 0.0:
        return 1.0 if current < previous else 0.0
    return (previous - current) / previous


def _resolved_workers(args: argparse.Namespace) -> int:
    if args.workers is not None and args.workers >= 0:
        return max(1, args.workers)
    return 2 if args.gpu else 1


def _run_seed_pass(
    *,
    run_id: str,
    seed: int,
    args: argparse.Namespace,
    workers: int,
    patience: int,
    checkpoint_every: int,
    improvement_epsilon: float,
    engine_candidates: bool,
    run_dir: Path,
    logger: logging.Logger,
    global_best: dict[str, float | tuple[float, ...]],
) -> int:
    checkpoint = load_generation_checkpoint(run_dir) if args.resume else None
    start_generation = 0
    initial_mean = None
    initial_sigma = None
    initial_best_vector = None
    initial_best_objective = None
    if checkpoint is not None and checkpoint.seed == seed:
        start_generation = checkpoint.generation
        import numpy as np

        initial_mean = np.asarray(checkpoint.mean, dtype=np.float64)
        initial_sigma = checkpoint.sigma
        initial_best_vector = np.asarray(checkpoint.best_vector, dtype=np.float64)
        initial_best_objective = checkpoint.best_objective
        logger.info("Resuming seed %s from generation %s", seed, start_generation)

    stale = 0
    last_best = float(initial_best_objective if initial_best_objective is not None else 1e9)

    def on_generation(generation: int, objective: float, vector: tuple[float, ...]) -> None:
        nonlocal stale, last_best
        gain = _relative_gain(last_best, objective)
        if gain > improvement_epsilon:
            stale = 0
            last_best = objective
        else:
            stale += 1
        if objective < float(global_best["objective"]):
            global_best["objective"] = objective
            global_best["vector"] = vector
            global_best["seed"] = float(seed)
        logger.info(
            "seed=%s gen=%s objective=%.6g stale=%s gpu=%s",
            seed,
            generation,
            objective,
            stale,
            _gpu_mem_mib(),
        )
        if stale >= patience:
            raise StopIteration

    def on_candidate(
        generation_index: int,
        candidate_index: int,
        loss: float,
        elapsed: float,
    ) -> None:
        logger.info(
            "seed=%s gen=%s candidate=%s loss=%.6g elapsed=%.2fs",
            seed,
            generation_index + 1,
            candidate_index,
            loss,
            elapsed,
        )

    def on_candidate_error(candidate_index: int, message: str) -> None:
        logger.error("seed=%s candidate=%s eval failed: %s", seed, candidate_index, message)

    try:
        fit = fit_nonbank_parameters(
            seed,
            generations=args.generations,
            population=args.population,
            steps=args.eval_minutes,
            engine_candidates=engine_candidates,
            workers=workers,
            run_id=run_id,
            checkpoint_dir=run_dir,
            checkpoint_every=checkpoint_every,
            diagnostics_dir=diagnostics_dir(run_id),
            on_generation=on_generation,
            start_generation=start_generation,
            initial_mean=initial_mean,
            initial_sigma=initial_sigma,
            initial_best_vector=initial_best_vector,
            initial_best_objective=initial_best_objective,
            on_candidate=on_candidate if engine_candidates else None,
            on_candidate_error=on_candidate_error if engine_candidates else None,
        )
    except StopIteration:
        checkpoint = load_generation_checkpoint(run_dir)
        if checkpoint is None:
            raise
        fit_vector = tuple(checkpoint.best_vector)
        fit_objective = checkpoint.best_objective
        reported_generations = checkpoint.generation
        logger.info(
            "Early stop seed %s at generation %s objective=%.6g",
            seed,
            checkpoint.generation,
            fit_objective,
        )
    else:
        fit_vector = fit.vector
        fit_objective = fit.objective
        reported_generations = fit.generations

    if fit_objective < float(global_best["objective"]):
        global_best["objective"] = fit_objective
        global_best["vector"] = fit_vector
        global_best["seed"] = float(seed)
    return reported_generations


def _long_main(args: argparse.Namespace) -> int:
    if args.gpu or args.gpu_sim or args.gpu_full:
        os.environ.setdefault("STAMPEDE_OFFLINE", "1")
        os.environ.setdefault("STAMPEDE_CALIBRATION_ENGINE", "real")
    run_id = args.run_id or datetime.now(UTC).strftime(f"m27-nonbank-{args.seed}-%Y%m%d-%H%M%S")
    runs_root = args.runs_dir if args.runs_dir is not None else _DEFAULT_RUNS
    run_dir = runs_root / run_id
    if args.score_only is not None:
        _check_score_only_run_dir(Path(args.score_only), run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(run_dir / "calibration.log")
    try:
        if args.score_only is not None:
            return _run_score_only(args, run_id, run_dir, logger)
        return _run_long(args, run_id, run_dir, logger)
    except Exception:
        logger.exception("non-bank calibration run failed")
        raise


def _apply_gpu_flags(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Shared by the fit and the score-only run: set the GPU env exactly as before SVB-38c."""
    if args.gpu_full:
        from engine.gpu_full import configure_calibration_gpu_full

        configure_calibration_gpu_full(True)
        args.gpu = True
        args.gpu_sim = True
    if args.gpu_sim:
        os.environ["STAMPEDE_CALIB_GPU_SIM"] = "1"
    if args.gpu:
        configure_calibration_gpu(True)
    elif args.gpu_sim:
        logger.warning(
            "--gpu-sim is set but --gpu is not, so STAMPEDE_BACKEND is not gpu "
            "and the legacy engine stays in use."
        )


def _run_long(args: argparse.Namespace, run_id: str, run_dir: Path, logger: logging.Logger) -> int:
    _apply_gpu_flags(args, logger)
    from engine.agents.decide import active_backend_name
    from engine.gpu_sim import gpu_sim_enabled, resolve_array_module

    workers = _resolved_workers(args)
    patience = 6 if args.patience is None else args.patience
    checkpoint_every = 2 if args.checkpoint_every is None else args.checkpoint_every
    improvement_epsilon = 1e-3 if args.improvement_epsilon is None else args.improvement_epsilon
    engine_candidates = bool(
        args.gpu or args.gpu_sim or args.gpu_full or (args.workers is not None and args.workers > 1)
    )
    array_backend = active_backend_name()
    m05_backend = population_balance_backend()
    device = population_device_description()
    if args.gpu and m05_backend != "gpu":
        logger.warning(
            "Requested --gpu but M05 balance solve fell back to cpu "
            "(CuPy missing or no CUDA device)."
        )
    logger.info(
        "Starting non-bank run %s real_engine=%s eval_minutes=%s workers=%s "
        "m06_backend=%s m05_balance=%s device=%s engine_candidates=%s",
        run_id,
        os.environ.get("STAMPEDE_CALIBRATION_ENGINE"),
        args.eval_minutes,
        workers,
        array_backend,
        m05_backend,
        device,
        engine_candidates,
    )
    logger.info("GPU %s", _gpu_mem_mib())
    if gpu_sim_enabled():
        from engine.gpu_full import gpu_full_enabled

        _xp, xp_name = resolve_array_module("auto")
        logger.info(
            "gpu-sim on (array module=%s full=%s). Equity, crypto, and rates share that kernel. "
            "The generation is one in-process batch; --workers does not split it.",
            xp_name,
            gpu_full_enabled(),
        )
        del _xp

    summary_path = run_dir / "summary.json"
    converged_marker = run_dir / "converged.json"
    global_best: dict[str, float | tuple[float, ...]] = {
        "objective": 1e9,
        "vector": (),
        "seed": float(args.seed),
    }
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        global_best["objective"] = float(prior.get("best_objective", global_best["objective"]))
        if prior.get("best_vector"):
            global_best["vector"] = tuple(prior["best_vector"])
        global_best["seed"] = float(prior.get("best_seed", args.seed))

    extra = [int(item.strip()) for item in args.extra_seeds.split(",") if item.strip()]
    seeds = [args.seed, *extra]
    started = time.perf_counter()
    targets = load_nonbank_targets()
    holdout_steps, crypto_holdout_steps, rates_holdout_steps = _holdout_horizons(args)
    for seed in seeds:
        logger.info("=== seed pass %s ===", seed)
        reported_generations = _run_seed_pass(
            run_id=run_id,
            seed=seed,
            args=args,
            workers=workers,
            patience=patience,
            checkpoint_every=checkpoint_every,
            improvement_epsilon=improvement_epsilon,
            engine_candidates=engine_candidates,
            run_dir=run_dir,
            logger=logger,
            global_best=global_best,
        )
        vector = global_best["vector"]
        if not vector:
            continue
        holdout = run_nonbank_holdout(
            tuple(vector),
            int(global_best["seed"]),
            steps=holdout_steps,
            rates_steps=args.rates_holdout_minutes,
            crypto_steps=args.crypto_holdout_minutes,
        )
        summary: dict[str, object] = {
            "run_id": run_id,
            "best_seed": int(global_best["seed"]),
            "best_objective": float(global_best["objective"]),
            "best_vector": list(vector),
            "vector_kind": "scale" if len(vector) == 3 else "behavioral",
            "holdout": holdout_payload(holdout, targets),
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_memory": _gpu_mem_mib(),
            "eval_minutes_fit": args.eval_minutes,
            "eval_minutes_holdout_crypto": crypto_holdout_steps,
            "eval_minutes_holdout_rates": rates_holdout_steps,
        }
        if len(vector) == 3:
            summary.update(_scale_term_fields())
        else:
            summary.update(
                _behavioral_term_fields(
                    vector=tuple(vector),
                    seed=int(global_best["seed"]),
                    equity_steps=_resolved_steps(args.eval_minutes),
                    crypto_steps=_resolved_steps(crypto_holdout_steps),
                    rates_steps=_resolved_steps(rates_holdout_steps),
                    holdout=holdout,
                    targets=targets,
                )
            )
        summary["term_status"] = _term_status_payload(
            targets,
            equity_steps=_resolved_steps(args.eval_minutes),
            crypto_steps=_resolved_steps(crypto_holdout_steps),
            rates_steps=_resolved_steps(rates_holdout_steps),
        )
        summary["crypto_profile"] = holdout.crypto_profile
        summary["rates_profile"] = holdout.rates_profile
        summary["fitted_rows"] = {}
        summary["fitted_rows_written"] = False
        summary["persist_command"] = (
            f"python -m calibration.persist_fit --run-dir {run_dir} --kind nonbank"
        )
        payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        summary_path.write_text(payload, encoding="utf-8")
        if not converged_marker.is_file():
            converged_marker.write_text(payload, encoding="utf-8")
            logger.info("Wrote %s", converged_marker)
            logger.info("%s", summary["persist_command"])
        write_run_diagnostics(
            diagnostics_dir(run_id),
            run_id=run_id,
            extra={
                "seed": int(global_best["seed"]),
                "objective": float(global_best["objective"]),
            },
        )
        _print_summary(
            int(global_best["seed"]),
            reported_generations,
            float(global_best["objective"]),
            tuple(vector),
            holdout,
            targets,
        )

    logger.info(
        "Run %s finished in %.1f s; best objective=%.6g",
        run_id,
        time.perf_counter() - started,
        global_best["objective"],
    )
    return 0


# --- SVB-38c: score-only re-score of a finished run's vector ------------------------------------

_SCRIPTED_DRAWDOWN_ROW = "M27 calibration uses scripted drawdown"
_ENGINE_ENV_KEYS = (
    "STAMPEDE_CALIBRATION_ENGINE",
    "STAMPEDE_BACKEND",
    "STAMPEDE_CALIB_GPU_SIM",
    "STAMPEDE_CALIB_GPU_FULL",
    "STAMPEDE_OFFLINE",
    "STAMPEDE_CALIB_DEPOSITORS",
)
_PERSIST_NOTE = (
    "Score-only run: nothing was persisted and this directory has no best_vector.json, so "
    "persist_fit cannot read it. The vector belongs to the source run; persisting it needs "
    "explicit maintainer approval and is run by hand against the source run directory."
)


def _check_score_only_run_dir(source: Path, run_dir: Path) -> None:
    """Refuse to write into the source run or over an earlier summary (checked before logging)."""
    if not source.is_file():
        raise SystemExit(f"--score-only: {source} is not a file")
    if source.resolve().parent == run_dir.resolve():
        raise SystemExit(
            "--score-only: --run-id points at the source run directory; pick a new run id"
        )
    if (run_dir / "summary.json").exists():
        raise SystemExit(
            f"--score-only: {run_dir / 'summary.json'} already exists; pick a new run id"
        )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[call-overload]


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _load_score_source(path: Path) -> dict[str, object]:
    """Vector, seed and reported numbers from converged.json, summary.json or best_vector.json.

    Reads both the pre-SVB-38b key (``eval_minutes_holdout``) and the split keys.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    vector = data.get("best_vector") or data.get("vector")
    if not vector:
        raise SystemExit(f"--score-only: {path} has no best_vector or vector")
    holdout = data.get("holdout") or {}
    shared = data.get("eval_minutes_holdout")
    return {
        "path": str(path.resolve()),
        "run_id": data.get("run_id"),
        "vector": tuple(float(x) for x in vector),
        "seed": _optional_int(data.get("best_seed", data.get("seed"))),
        "objective": _optional_float(data.get("best_objective", data.get("objective"))),
        "crypto_loss": _optional_float(holdout.get("crypto_loss")),
        "rates_loss": _optional_float(holdout.get("rates_loss")),
        "crypto_excluded": holdout.get("crypto_excluded"),
        "rates_excluded": holdout.get("rates_excluded"),
        "eval_minutes_fit": _optional_int(data.get("eval_minutes_fit")),
        "eval_minutes_holdout_crypto": _optional_int(
            data.get("eval_minutes_holdout_crypto", shared)
        ),
        "eval_minutes_holdout_rates": _optional_int(
            data.get("eval_minutes_holdout_rates", shared)
        ),
        "fitted_rows_written": data.get("fitted_rows_written"),
    }


def _gpu_pool_mib() -> str:
    """Device memory held by this process's CuPy pool (mem_info also counts other processes)."""
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        return (
            f"{pool.used_bytes() / (1024**2):.0f} MiB used / "
            f"{pool.total_bytes() / (1024**2):.0f} MiB held by this process"
        )
    except Exception:
        return "n/a"


def _peak_rss_mib() -> float | None:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0  # Linux: KiB
    except Exception:
        return None


def _memory_snapshot() -> dict[str, object]:
    return {"device": _gpu_mem_mib(), "cupy_pool": _gpu_pool_mib(), "peak_rss_mib": _peak_rss_mib()}


def _registry_row_or_note(name: str) -> dict[str, str] | str:
    try:
        row = get_parameter(name)
    except Exception:
        return "row not found"
    return {"value": row.value, "status": row.status}


def _holdout_block(
    holdout: NonbankHoldoutResult,
    targets: NonbankTargets,
    *,
    crypto_steps: int,
    rates_steps: int,
) -> dict[str, object]:
    """Losses, per-term losses, term status and profiles for one pair of holdout horizons."""
    minutes_per_day = float(get_parameter("Non-bank sim minutes per day").value)
    weights = _registry_weights()
    crypto_target, _ = reachable_targets(
        targets.crypto, steps=crypto_steps, minutes_per_day=minutes_per_day
    )
    rates_target, _ = reachable_targets(
        targets.rates, steps=rates_steps, minutes_per_day=minutes_per_day
    )
    status = _term_status_payload(
        targets, equity_steps=1, crypto_steps=crypto_steps, rates_steps=rates_steps
    )
    return {
        "eval_minutes_holdout_crypto": int(crypto_steps),
        "eval_minutes_holdout_rates": int(rates_steps),
        "holdout": holdout_payload(holdout, targets),
        "crypto_stats": _stats_payload(holdout.crypto),
        "rates_stats": _stats_payload(holdout.rates),
        "crypto_terms": _json_terms(
            shape_loss_terms(holdout.crypto, crypto_target, weights=weights)
        ),
        "rates_terms": _json_terms(
            shape_loss_terms(holdout.rates, rates_target, weights=weights)
        ),
        "term_status": {"E15": status["E15"], "E16": status["E16"]},
        "crypto_profile": holdout.crypto_profile,
        "rates_profile": holdout.rates_profile,
        "seconds": {"crypto": holdout.crypto_seconds, "rates": holdout.rates_seconds},
    }


def _compare(
    source: float | None, rescored: float | None, *, same_horizon: bool
) -> dict[str, object]:
    delta = None
    if source is not None and rescored is not None:
        delta = float(rescored) - float(source)
    return {"source": source, "rescored": rescored, "delta": delta, "same_horizon": same_horizon}


def _timing_model(
    baseline: dict[str, object] | None, main: dict[str, object], key: str
) -> dict[str, object] | None:
    """Fixed cost plus seconds per sim minute from two horizons of the same holdout."""
    if baseline is None:
        return None
    minutes_key = f"eval_minutes_holdout_{key}"
    m0 = int(baseline[minutes_key])  # type: ignore[call-overload]
    m1 = int(main[minutes_key])  # type: ignore[call-overload]
    s0 = baseline["seconds"][key]  # type: ignore[index]
    s1 = main["seconds"][key]  # type: ignore[index]
    if m0 == m1 or s0 is None or s1 is None:
        return None
    slope = (float(s1) - float(s0)) / float(m1 - m0)
    return {
        "seconds_per_sim_minute": slope,
        "fixed_seconds": float(s0) - slope * float(m0),
        "from_minutes": [m0, m1],
        "note": "estimate = fixed_seconds + seconds_per_sim_minute * minutes, one single-lane path",
    }


def _run_score_only(
    args: argparse.Namespace, run_id: str, run_dir: Path, logger: logging.Logger
) -> int:
    """Re-score a finished run's vector on new holdout horizons. No fit, no FITTED rows."""
    import numpy as np

    from calibration.fit import use_real_calibration_engine

    started = time.perf_counter()
    _apply_gpu_flags(args, logger)
    source = _load_score_source(Path(args.score_only))
    vector: tuple[float, ...] = source["vector"]  # type: ignore[assignment]
    behavioral_dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    scale_dim = int(float(get_parameter("Non-bank calibration parameter dimension").value))
    if len(vector) not in (behavioral_dim, scale_dim):
        raise SystemExit(
            f"--score-only: vector has {len(vector)} components; the current layout is "
            f"{behavioral_dim} (behavioral) or {scale_dim} (fake scale). It was fit on another "
            "layout and can't be re-scored as is."
        )
    is_scale = len(vector) == scale_dim
    if not is_scale and not use_real_calibration_engine():
        raise SystemExit(
            "--score-only: a behavioral vector needs the real calibration engine (--gpu-full, "
            "or STAMPEDE_CALIBRATION_ENGINE=real); the fake engine ignores the vector."
        )
    source_seed = source["seed"]
    seed = int(source_seed) if source_seed is not None else int(args.seed)  # type: ignore[call-overload]
    equity_steps = source["eval_minutes_fit"] if args.eval_minutes is None else args.eval_minutes
    if equity_steps is None:
        raise SystemExit("--score-only: the source has no eval_minutes_fit; pass --eval-minutes")
    equity_steps = int(equity_steps)  # type: ignore[call-overload]
    crypto_steps = int(args.crypto_holdout_minutes)
    rates_steps = int(args.rates_holdout_minutes)
    if args.baseline_holdout_minutes is None:
        baseline_steps = (
            source["eval_minutes_holdout_crypto"],
            source["eval_minutes_holdout_rates"],
        )
    elif args.baseline_holdout_minutes == 0:
        baseline_steps = (None, None)
    else:
        baseline_steps = (args.baseline_holdout_minutes, args.baseline_holdout_minutes)
    logger.info(
        "Score-only run %s: source=%s run_id=%s seed=%s dim=%s equity=%s crypto=%s rates=%s "
        "baseline=%s",
        run_id,
        source["path"],
        source["run_id"],
        seed,
        len(vector),
        equity_steps,
        crypto_steps,
        rates_steps,
        baseline_steps,
    )
    memory: dict[str, object] = {"start": _memory_snapshot()}
    targets = load_nonbank_targets()
    minutes_per_day = float(get_parameter("Non-bank sim minutes per day").value)

    baseline: dict[str, object] | None = None
    if baseline_steps[0] is not None and baseline_steps[1] is not None:
        b_crypto = int(baseline_steps[0])  # type: ignore[call-overload]
        b_rates = int(baseline_steps[1])  # type: ignore[call-overload]
        logger.info("Baseline holdout at crypto=%s rates=%s", b_crypto, b_rates)
        b_holdout = run_nonbank_holdout(
            vector, seed, steps=equity_steps, crypto_steps=b_crypto, rates_steps=b_rates
        )
        baseline = _holdout_block(b_holdout, targets, crypto_steps=b_crypto, rates_steps=b_rates)
        memory["after_baseline"] = _memory_snapshot()
        logger.info(
            "Baseline crypto_loss=%s rates_loss=%s seconds=%s",
            baseline["holdout"]["crypto_loss"],  # type: ignore[index]
            baseline["holdout"]["rates_loss"],  # type: ignore[index]
            baseline["seconds"],
        )

    logger.info("Main holdout at crypto=%s rates=%s", crypto_steps, rates_steps)
    holdout = run_nonbank_holdout(
        vector, seed, steps=equity_steps, crypto_steps=crypto_steps, rates_steps=rates_steps
    )
    main_block = _holdout_block(
        holdout, targets, crypto_steps=crypto_steps, rates_steps=rates_steps
    )
    memory["after_main"] = _memory_snapshot()

    equity_started = time.perf_counter()
    if is_scale:
        equity_target, _ = reachable_targets(
            targets.equity, steps=equity_steps, minutes_per_day=minutes_per_day
        )
        equity_objective = float(
            objective_nonbank(
                np.asarray(vector, dtype=np.float64),
                scenario=build_nonbank_scenario("equity", seed),
                steps=equity_steps,
                target=equity_target,
            )
        )
        term_fields = _scale_term_fields()
    else:
        term_fields = _behavioral_term_fields(
            vector=vector,
            seed=seed,
            equity_steps=equity_steps,
            crypto_steps=crypto_steps,
            rates_steps=rates_steps,
            holdout=holdout,
            targets=targets,
        )
        equity_terms: dict[str, float | None] = term_fields["equity_terms"]  # type: ignore[assignment]
        equity_objective = float(
            sum(float(v) for v in equity_terms.values() if v is not None)
            + float(term_fields["equity_request_penalty"])  # type: ignore[arg-type]
        )
    equity_seconds = time.perf_counter() - equity_started

    source_fit_minutes = source["eval_minutes_fit"]
    comparison = {
        "equity_objective": _compare(
            source["objective"],  # type: ignore[arg-type]
            equity_objective,
            same_horizon=source_fit_minutes == equity_steps,
        ),
        "crypto_loss_baseline": _compare(
            source["crypto_loss"],  # type: ignore[arg-type]
            None if baseline is None else baseline["holdout"]["crypto_loss"],  # type: ignore[index]
            same_horizon=baseline is not None
            and baseline["eval_minutes_holdout_crypto"] == source["eval_minutes_holdout_crypto"],
        ),
        "rates_loss_baseline": _compare(
            source["rates_loss"],  # type: ignore[arg-type]
            None if baseline is None else baseline["holdout"]["rates_loss"],  # type: ignore[index]
            same_horizon=baseline is not None
            and baseline["eval_minutes_holdout_rates"] == source["eval_minutes_holdout_rates"],
        ),
        "crypto_loss_new_horizon": _compare(
            source["crypto_loss"],  # type: ignore[arg-type]
            main_block["holdout"]["crypto_loss"],  # type: ignore[index]
            same_horizon=crypto_steps == source["eval_minutes_holdout_crypto"],
        ),
        "rates_loss_new_horizon": _compare(
            source["rates_loss"],  # type: ignore[arg-type]
            main_block["holdout"]["rates_loss"],  # type: ignore[index]
            same_horizon=rates_steps == source["eval_minutes_holdout_rates"],
        ),
        "note": (
            "A loss is only like-for-like when same_horizon is true and term_status lists the same "
            "evaluated terms. At a longer rates horizon the days-to-bottom term joins the loss, so "
            "compare rates_terms.peak_to_trough instead."
        ),
    }
    summary: dict[str, object] = {
        "mode": "score_only",
        "run_id": run_id,
        "source": {key: value for key, value in source.items() if key != "vector"},
        "seed": seed,
        "vector": list(vector),
        "vector_kind": "scale" if is_scale else "behavioral",
        "equity_objective": equity_objective,
        "eval_minutes_fit": equity_steps,
        "eval_minutes_holdout_crypto": crypto_steps,
        "eval_minutes_holdout_rates": rates_steps,
        "minutes_per_day": minutes_per_day,
        **main_block,
        **term_fields,
        "term_status": _term_status_payload(
            targets, equity_steps=equity_steps, crypto_steps=crypto_steps, rates_steps=rates_steps
        ),
        "baseline": baseline,
        "comparison": comparison,
        "timings_seconds": {
            "baseline": None if baseline is None else baseline["seconds"],
            "main": main_block["seconds"],
            "equity": equity_seconds,
            "total": time.perf_counter() - started,
        },
        "timing_model": {
            "crypto": _timing_model(baseline, main_block, "crypto"),
            "rates": _timing_model(baseline, main_block, "rates"),
        },
        "memory": memory,
        "market_quote": {_SCRIPTED_DRAWDOWN_ROW: _registry_row_or_note(_SCRIPTED_DRAWDOWN_ROW)},
        "engine_env": {key: os.environ.get(key) for key in _ENGINE_ENV_KEYS},
        "fitted_rows": {},
        "fitted_rows_written": False,
        "persist_command": None,
        "persist_note": _PERSIST_NOTE,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    (run_dir / "summary.json").write_text(payload, encoding="utf-8")
    logger.info("Wrote %s", run_dir / "summary.json")
    print(f"mode=score_only run_id={run_id} source={source['run_id']} seed={seed}")
    print(f"equity_objective={equity_objective:.6g} (source {source['objective']})")
    for label, block in (("baseline", baseline), ("main", main_block)):
        if block is None:
            continue
        hold = block["holdout"]  # type: ignore[index]
        print(
            f"{label}: crypto[{block['eval_minutes_holdout_crypto']}]={hold['crypto_loss']} "
            f"rates[{block['eval_minutes_holdout_rates']}]={hold['rates_loss']} "
            f"seconds={block['seconds']}"
        )
    crypto_profile = holdout.crypto_profile or {}
    print(
        "crypto new_low_after_day1="
        f"{crypto_profile.get('new_low_after_day1')} "
        f"share_of_drawdown_after_day1={crypto_profile.get('share_of_drawdown_after_day1')} "
        f"covers_horizon={crypto_profile.get('covers_horizon')} clock={crypto_profile.get('clock')}"
    )
    print("fitted_rows_written=False persist_command=None")
    total = time.perf_counter() - started
    logger.info("Score-only run %s finished in %.1f s", run_id, total)
    if not math.isfinite(equity_objective):
        logger.warning("equity objective is not finite: %s", equity_objective)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not _uses_run_controls(args):
        return _legacy_main(args)
    return _long_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
