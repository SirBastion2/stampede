"""Long-running M27 GPU calibration with checkpoints and multi-seed restarts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from calibration.diagnostics import attach_on_generation, diagnostics_dir, write_run_diagnostics
from calibration.fit import (
    FitResult,
    build_validation_board,
    configure_calibration_gpu,
    e1_stock_ratio_summary,
    fit_behavioral_parameters,
    fit_payment_capacity,
    format_validation_board_markdown,
    identifiable_indices,
    identifiable_names,
    load_generation_checkpoint,
    measure_e1_stock_ratio,
    run_holdout,
    validate_fitted_parameters,
    write_best_vector_json,
    write_fitted_registry,
)
from engine.agents.population import population_balance_backend, population_device_description

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RUNS = _REPO_ROOT / "calibration" / "runs"


def _configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m27.calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M27 behavioral calibration (GPU real engine).")
    parser.add_argument("--run-id", default="", help="Stable run identifier (default: timestamp).")
    parser.add_argument("--seed", type=int, default=3060, help="Primary fit seed.")
    parser.add_argument(
        "--extra-seeds",
        default="",
        help="Comma-separated extra seed passes. Empty (default) runs the primary seed once; CMA-ES restarts on stagnation.",
    )
    parser.add_argument("--generations", type=int, default=40, help="Generations per seed pass.")
    parser.add_argument("--population", type=int, default=12, help="CMA-ES-shaped population size.")
    parser.add_argument(
        "--eval-minutes",
        type=int,
        default=1440,
        help="Stage 1 needs >= 1440 (next open).",
    )
    parser.add_argument(
        "--holdout-minutes",
        type=int,
        default=780,
        help="Simulated minutes for E5/placebo holdout after fit.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2, help="Checkpoint interval (generations).")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Optional early-stop after this many generations without improvement (0 = no cap).",
    )
    parser.add_argument(
        "--improvement-epsilon",
        type=float,
        default=1e-3,
        help="Relative objective improvement required to reset patience.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.json in the run dir.")
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Parallel candidate evaluators per generation (default: 1 CPU, 2 with --gpu).",
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
            "The legacy per-depositor engine stays in use unless both are set."
        ),
    )
    parser.add_argument(
        "--gpu-full",
        action="store_true",
        help=(
            "Whole-tick array path (STAMPEDE_CALIB_GPU_FULL=1): attention, "
            "contagion, market, and payments on CuPy (NumPy if no device). "
            "Implies --gpu and --gpu-sim. Default engine path stays off."
        ),
    )
    parser.add_argument("--runs-dir", type=Path, default=_DEFAULT_RUNS, help="Parent directory for run artifacts.")
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help=(
            "Also write the fitted rows into params/registry.csv at the end (SVB-49). Off by default: "
            "a run only proposes rows, and persisting needs explicit maintainer approval "
            "(python -m calibration.persist_fit --run-dir <run>)."
        ),
    )
    return parser.parse_args()


def _relative_gain(previous: float, current: float) -> float:
    if previous <= 0.0:
        return 1.0 if current < previous else 0.0
    return (previous - current) / previous


def _resolved_workers(args: argparse.Namespace) -> int:
    if args.workers >= 0:
        return max(1, args.workers)
    return 2 if args.gpu else 1


def apply_run_fit_engine(
    *,
    gpu: bool = False,
    gpu_sim: bool = False,
    gpu_full: bool = False,
    logger: logging.Logger | None = None,
    set_process_env: bool = True,
) -> dict[str, object]:
    """Set the same STAMPEDE_* env as ``run_fit._run`` before stage 2.

    ``gpu_full`` implies ``gpu`` and ``gpu_sim`` (``--gpu-full``). Missing CUDA
    falls back the same way ``run_fit`` does: ``STAMPEDE_BACKEND=gpu`` stays set
    and the kernels run on NumPy. Returns the resolved path for logging.
    """
    if set_process_env:
        os.environ.setdefault("STAMPEDE_OFFLINE", "1")
        os.environ.setdefault("STAMPEDE_CALIBRATION_ENGINE", "real")
    if gpu_full:
        from engine.gpu_full import configure_calibration_gpu_full

        configure_calibration_gpu_full(True)
        gpu = True
        gpu_sim = True
    if gpu_sim:
        os.environ["STAMPEDE_CALIB_GPU_SIM"] = "1"
    if gpu:
        configure_calibration_gpu(True)
    elif gpu_sim and logger is not None:
        logger.warning(
            "--gpu-sim is set but --gpu is not, so STAMPEDE_BACKEND is not gpu "
            "and the legacy engine stays in use."
        )
    from engine.agents.decide import active_backend_name
    from engine.gpu_sim import calibration_uses_scripted_drawdown, gpu_sim_enabled, resolve_array_module

    array_backend = active_backend_name()
    m05_backend = population_balance_backend()
    device = population_device_description()
    if gpu and m05_backend != "gpu" and logger is not None:
        logger.warning(
            "Requested --gpu but M05 balance solve fell back to cpu (CuPy missing or no CUDA device)."
        )
    sim_on = gpu_sim_enabled()
    xp_name: str | None = None
    full_on = False
    if sim_on:
        from engine.gpu_full import gpu_full_enabled

        full_on = gpu_full_enabled()
        _xp, xp_name = resolve_array_module("auto")
        del _xp
    snapshot: dict[str, object] = {
        "real_engine": os.environ.get("STAMPEDE_CALIBRATION_ENGINE"),
        "offline": os.environ.get("STAMPEDE_OFFLINE"),
        "backend": os.environ.get("STAMPEDE_BACKEND", "cpu"),
        "gpu": gpu,
        "gpu_sim": sim_on,
        "gpu_full": full_on,
        "m06_backend": array_backend,
        "m05_balance": m05_backend,
        "device": device,
        "array_module": xp_name,
        "scripted_drawdown": calibration_uses_scripted_drawdown(),
        "STAMPEDE_CALIBRATION_ENGINE": os.environ.get("STAMPEDE_CALIBRATION_ENGINE"),
        "STAMPEDE_BACKEND": os.environ.get("STAMPEDE_BACKEND"),
        "STAMPEDE_CALIB_GPU_SIM": os.environ.get("STAMPEDE_CALIB_GPU_SIM"),
        "STAMPEDE_CALIB_GPU_FULL": os.environ.get("STAMPEDE_CALIB_GPU_FULL"),
        "STAMPEDE_OFFLINE": os.environ.get("STAMPEDE_OFFLINE"),
    }
    if logger is not None:
        logger.info(
            "engine_path real_engine=%s backend=%s gpu_sim=%s gpu_full=%s "
            "m06_backend=%s m05_balance=%s device=%s array_module=%s scripted_drawdown=%s",
            snapshot["real_engine"],
            snapshot["backend"],
            snapshot["gpu_sim"],
            snapshot["gpu_full"],
            snapshot["m06_backend"],
            snapshot["m05_balance"],
            snapshot["device"],
            snapshot["array_module"],
            snapshot["scripted_drawdown"],
        )
    return snapshot


def _run_seed_pass(
    *,
    run_id: str,
    seed: int,
    args: argparse.Namespace,
    workers: int,
    run_dir: Path,
    logger: logging.Logger,
    global_best: dict[str, float | tuple[float, ...]],
) -> None:
    os.environ["STAMPEDE_CALIB_EVAL_MINUTES"] = str(args.eval_minutes)
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
        if gain > args.improvement_epsilon:
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
        if args.patience > 0 and stale >= args.patience:
            raise StopIteration

    def on_candidate(generation_index: int, candidate_index: int, loss: float, elapsed: float) -> None:
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

    diag_dir = diagnostics_dir(run_id)
    try:
        fit = fit_behavioral_parameters(
            seed,
            generations=args.generations,
            population=args.population,
            steps=args.eval_minutes,
            run_id=run_id,
            checkpoint_dir=run_dir,
            checkpoint_every=args.checkpoint_every,
            on_generation=attach_on_generation(diag_dir, on_generation, seed=seed),
            start_generation=start_generation,
            initial_mean=initial_mean,
            initial_sigma=initial_sigma,
            initial_best_vector=initial_best_vector,
            initial_best_objective=initial_best_objective,
            workers=workers,
            on_candidate=on_candidate,
            on_candidate_error=on_candidate_error,
        )
    except StopIteration:
        checkpoint = load_generation_checkpoint(run_dir)
        if checkpoint is None:
            raise
        fit_vector = checkpoint.best_vector
        fit_objective = checkpoint.best_objective
        logger.info("Early stop seed %s at generation %s objective=%.6g", seed, checkpoint.generation, fit_objective)
    else:
        fit_vector = fit.vector
        fit_objective = fit.objective

    if fit_objective < float(global_best["objective"]):
        global_best["objective"] = fit_objective
        global_best["vector"] = fit_vector
        global_best["seed"] = float(seed)
    if global_best["vector"]:
        write_best_vector_json(
            run_dir,
            run_id=run_id,
            seed=int(global_best["seed"]),
            vector=global_best["vector"],
            objective=float(global_best["objective"]),
            force=True,
        )
    write_run_diagnostics(
        diag_dir,
        run_id=run_id,
        extra={"seed": seed, "objective": float(fit_objective)},
    )


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    os.environ.setdefault("STAMPEDE_CALIBRATION_ENGINE", "real")

    run_id = args.run_id or datetime.now(UTC).strftime("m27-3060-%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(run_dir / "calibration.log")
    try:
        return _run(args, run_id, run_dir, logger)
    except Exception:
        logger.exception("calibration run failed")
        raise


def _run(args: argparse.Namespace, run_id: str, run_dir: Path, logger: logging.Logger) -> int:
    setup = apply_run_fit_engine(
        gpu=args.gpu,
        gpu_sim=args.gpu_sim,
        gpu_full=args.gpu_full,
        logger=logger,
        set_process_env=False,
    )
    args.gpu = bool(setup["gpu"])
    args.gpu_sim = bool(setup["gpu_sim"])
    workers = _resolved_workers(args)
    logger.info(
        "Starting run %s real_engine=%s eval_minutes=%s workers=%s m06_backend=%s m05_balance=%s device=%s",
        run_id,
        setup["real_engine"],
        args.eval_minutes,
        workers,
        setup["m06_backend"],
        setup["m05_balance"],
        setup["device"],
    )
    logger.info(
        "identifiable_indices=%s names=%s",
        ",".join(str(i) for i in identifiable_indices()),
        ",".join(identifiable_names()),
    )
    logger.info("Depositors=%s", os.environ.get("STAMPEDE_CALIB_DEPOSITORS", "registry tier default"))
    logger.info("GPU %s", _gpu_mem_mib())
    if setup["gpu_sim"]:
        logger.info(
            "gpu-sim on (array module=%s full=%s). One in-process batch; --workers does not split it.",
            setup["array_module"],
            setup["gpu_full"],
        )

    summary_path = run_dir / "summary.json"
    converged_marker = run_dir / "converged.json"
    global_best: dict[str, float | tuple[float, ...]] = {"objective": 1e9, "vector": (), "seed": float(args.seed)}

    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        global_best["objective"] = float(prior.get("best_objective", global_best["objective"]))
        if prior.get("best_vector"):
            global_best["vector"] = tuple(prior["best_vector"])
        global_best["seed"] = float(prior.get("best_seed", args.seed))

    seeds = [args.seed] + [int(item.strip()) for item in args.extra_seeds.split(",") if item.strip()]
    started = time.perf_counter()
    for seed in seeds:
        logger.info("=== seed pass %s ===", seed)
        _run_seed_pass(
            run_id=run_id,
            seed=seed,
            args=args,
            workers=workers,
            run_dir=run_dir,
            logger=logger,
            global_best=global_best,
        )

    vector = global_best["vector"]
    if vector:
        write_best_vector_json(
            run_dir,
            run_id=run_id,
            seed=int(global_best["seed"]),
            vector=vector,
            objective=float(global_best["objective"]),
            force=True,
        )
        capacity_multiplier = fit_payment_capacity(
            vector=tuple(vector),
            seed=int(global_best["seed"]),
            write=True,
            run_id=run_id,
        )
        holdout = run_holdout(tuple(vector), int(global_best["seed"]), steps=args.holdout_minutes)
        caps_on = validate_fitted_parameters(
            tuple(vector), int(global_best["seed"]), steps=args.holdout_minutes
        )
        fit_result = FitResult(
            seed=int(global_best["seed"]),
            vector=tuple(vector),
            objective=float(global_best["objective"]),
            generations=args.generations,
        )
        board = build_validation_board(fit_result, holdout)
        stock_observed = measure_e1_stock_ratio(tuple(vector), int(global_best["seed"]))
        stock_fields = e1_stock_ratio_summary(stock_observed)
        logger.info(
            "E1 stock ratio: observed=%s expected=%s",
            stock_fields["observed"],
            stock_fields["expected"],
        )
        board_md = format_validation_board_markdown(
            board,
            run_id=run_id,
            seed=int(global_best["seed"]),
            fit_objective=float(global_best["objective"]),
            stock_ratio=stock_fields["observed"],
            stock_target=stock_fields["expected"],
        )
        (run_dir / "validation_board.md").write_text(board_md, encoding="utf-8")
        summary = {
            "run_id": run_id,
            "best_seed": int(global_best["seed"]),
            "best_objective": float(global_best["objective"]),
            "best_vector": list(vector),
            "holdout": {
                "signature_day1_fraction": holdout.signature_day1_fraction,
                "signature_day1_requested": holdout.signature_day1_requested,
                "placebo_survives": holdout.placebo_survives,
            },
            "validation_caps_on": {
                "day1_settled": caps_on.day1_settled,
                "two_day_settled": caps_on.two_day_settled,
                "day1_requested": caps_on.day1_requested,
                "two_day_requested": caps_on.two_day_requested,
                "closed": caps_on.closed,
            },
            "e1_stock_ratio": stock_fields,
            "board": [
                {
                    "target_id": row.target_id,
                    "observed": row.observed,
                    "expected": row.expected,
                    "passed": row.passed,
                }
                for row in board
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_memory": _gpu_mem_mib(),
            "eval_minutes_fit": args.eval_minutes,
            "eval_minutes_holdout": args.holdout_minutes,
            "capacity_multiplier": float(capacity_multiplier),
        }
        # SVB-49: no registry write unless --write-registry. "unchanged" is not a write.
        write_registry = bool(getattr(args, "write_registry", False))
        fitted_rows: dict[str, str] = {}
        if write_registry:
            fitted_rows = (
                write_fitted_registry(tuple(vector), run_id=run_id, seed=int(global_best["seed"]))
                or {}
            )
        summary["fitted_rows"] = dict(fitted_rows)
        summary["fitted_rows_written"] = any(
            state in {"updated", "appended"} for state in fitted_rows.values()
        )
        summary["persist_command"] = (
            None if write_registry else f"python -m calibration.persist_fit --run-dir {run_dir}"
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        converged_marker.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # SVB-55 (E2): the board belongs to the run, not to the tracked docs/ tree.
        docs_board = run_dir / f"validation-board-{run_id}.md"
        docs_board.write_text(board_md, encoding="utf-8")
        if write_registry:
            logger.info("Wrote FITTED registry rows %s and %s", sorted(fitted_rows), docs_board)
        else:
            logger.info("Registry untouched; to persist (needs approval): %s", summary["persist_command"])
            logger.info("Wrote %s", docs_board)

    logger.info("Run %s finished in %.1f s; best objective=%.6g", run_id, time.perf_counter() - started, global_best["objective"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
