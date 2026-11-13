"""Persist a finished calibration run's results into params/registry.csv (row patches only).

Usage, from the repo root (in a branch, then open a PR):

    # SVB (E1) run: the identifiable rows (price impact, mobile share)
    python -m calibration.persist_fit --run-dir calibration/runs/m27-e1-svb26-20261101
    # ...and re-solve stage 2 with the same engine as run_fit --gpu-full
    python -m calibration.persist_fit --run-dir calibration/runs/m27-e1-svb26-20261101 --stage2

    # Non-bank run: its vector goes to its own row, never the SVB M06/M14 rows
    python -m calibration.persist_fit --run-dir calibration/runs/nonbank-svb26-c-20261101 --kind nonbank

Then ``git diff --stat params/registry.csv`` should show only the patched rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from calibration.registry_patch import patch_registry_rows

NONBANK_VECTOR_ROW = "Non-bank E13 fitted behavioral vector (normalized)"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _registry_path(path: Path | None) -> Path:
    return path or _REPO_ROOT / "params" / "registry.csv"


def write_nonbank_fitted_vector(
    vector: tuple[float, ...],
    *,
    run_id: str,
    seed: int,
    objective: float,
    path: Path | None = None,
) -> dict[str, str]:
    """Store the non-bank fit's normalized vector in its own row.

    The non-bank fit uses the same 16-dim behavioral vector layout as the SVB fit
    but fits it on equity data. Writing it into the shared M06/M14 rows would
    overwrite the SVB-fitted price impact and mobile share, so it gets one
    dedicated row (semicolon-separated normalized coordinates).
    """
    value = ";".join(f"{float(x):.6g}" for x in vector)
    source = f"{run_id} seed={seed} objective={objective:.6g}"
    return patch_registry_rows(
        _registry_path(path),
        {NONBANK_VECTOR_ROW: {"Value": value, "Status": "FITTED", "Source": source}},
        append_missing={NONBANK_VECTOR_ROW: {"Module": "M27", "Unit": "normalized vector"}},
    )


def _rows(path: Path | None, names) -> dict[str, dict[str, str]]:
    """Current registry rows for ``names`` (read-only)."""
    import csv

    wanted = set(names)
    out: dict[str, dict[str, str]] = {}
    with _registry_path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Parameter") or "").strip()
            if name in wanted:
                out[name] = {k: (v or "") for k, v in row.items() if k is not None}
    return out


def _load_best(run_dir: Path) -> dict:
    best = run_dir / "best_vector.json"
    if best.is_file():
        return json.loads(best.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "vector": summary["best_vector"],
        "seed": summary["best_seed"],
        "objective": summary["best_objective"],
        "run_id": summary["run_id"],
    }


def _stage2_logger() -> logging.Logger:
    logger = logging.getLogger("m27.persist_fit")
    logger.setLevel(logging.INFO)
    for old in list(logger.handlers):
        logger.removeHandler(old)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def apply_stage2_engine(logger: logging.Logger | None = None) -> dict[str, object]:
    """Same engine/GPU env as ``run_fit._run`` with ``--gpu-full`` (CPU NumPy fallback)."""
    from calibration.run_fit import apply_run_fit_engine

    return apply_run_fit_engine(
        gpu=True,
        gpu_sim=True,
        gpu_full=True,
        logger=logger,
        set_process_env=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("svb", "nonbank"), default="svb")
    parser.add_argument("--stage2", action="store_true", help="SVB only: re-solve and write the capacity multiplier")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="SVB only: persist even if these rows already came from this run (see the guard below)",
    )
    args = parser.parse_args(argv)

    best = _load_best(args.run_dir)
    vector = tuple(float(x) for x in best["vector"])
    seed = int(best["seed"])
    run_id = str(best.get("run_id") or args.run_dir.name)
    if args.kind == "nonbank":
        status = write_nonbank_fitted_vector(
            vector, run_id=run_id, seed=seed, objective=float(best["objective"]), path=args.registry
        )
        print(json.dumps(status, indent=2))
        return 0

    from calibration.fit import fit_payment_capacity, fitted_registry_updates, write_fitted_registry

    # Compute before writing: the normalized vector is relative to the registry at fit time.
    values = fitted_registry_updates(vector, path=args.registry)
    source = f"{run_id} seed={seed}"
    already = [name for name, row in _rows(args.registry, values).items()
               if row.get("Status", "").strip() == "FITTED" and row.get("Source", "").strip() == source]
    if already and not args.force:
        print(
            "refusing: these rows were already persisted from this run: " + ", ".join(already) + ". "
            "Persisting again would re-center the normalized vector on the fitted values and compound "
            "them. Use --force only after restoring the pre-persist registry values.",
            file=sys.stderr,
        )
        return 3
    if args.stage2:
        # SVB-40: solve stage 2 BEFORE writing the fitted rows. The fit vector is relative to the
        # registry at fit time; after the write it would be applied on top of its own values and
        # stage 2 would solve against a different model than the board. If the solve raises,
        # nothing is written.
        logger = _stage2_logger()
        setup = apply_stage2_engine(logger)
        print(json.dumps({"engine": setup}, indent=2, default=str))
        stage2_report: dict[str, object] = {}
        capacity = fit_payment_capacity(
            vector=vector,
            seed=seed,
            write=True,
            run_id=run_id,
            path=args.registry,
            report=stage2_report,
        )
        print(json.dumps({"stage2": stage2_report}, indent=2, default=str))
        print(f"capacity_multiplier={capacity:.6g}")
    status = write_fitted_registry(vector, run_id=run_id, seed=seed, path=args.registry, values=values)
    shown = {name: values[name] for name, state in status.items() if state != "absent" and name in values}
    print(json.dumps({"rows": status, "values": shown}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
