"""PA-B: run the multi-seed batch table offline and write the file the server serves at /api/batch.

    PYTHONPATH=. $PY scripts/batch_table.py --workers 6 --seeds 1-10            # about 12-15 min on the 3060 desktop
    PYTHONPATH=. $PY scripts/batch_table.py --out /tmp/batch.json --seeds 1,2 --minutes 60   # a quick look

Writes data/live_batch.json (or STAMPEDE_BATCH_FILE / --out) and prints the summary table.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())


def _seeds(text: str) -> tuple[int, ...]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(part) for part in text.split(",") if part)


def _backend_flags(parser) -> None:
    """CPY-1: GPU by default. --cpu is the NumPy reference; --require-gpu refuses to fall back."""
    parser.add_argument("--cpu", action="store_true", help="the NumPy reference: STAMPEDE_BACKEND=cpu, no batching")
    parser.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to NumPy")


def _apply_backend_flags(args) -> None:
    if args.cpu:
        os.environ["STAMPEDE_BACKEND"] = "cpu"
        os.environ["STAMPEDE_SEARCH_GPU"] = "0"
    if args.require_gpu:
        os.environ["STAMPEDE_REQUIRE_GPU"] = "1"
    if getattr(args, "gpu", False):
        os.environ["STAMPEDE_SEARCH_GPU"] = "1"
    from engine.xp import format_report, require_gpu

    require_gpu("this run")
    print(format_report(), flush=True)


def main() -> int:
    from ui.live.batch import run_grid, write_batch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--workers", type=int, default=None, help="default: 6 (STAMPEDE_WORKERS overrides); with the GPU, "
                        "every worker shares the one 12 GB RTX 3060")
    parser.add_argument("--minutes", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--gpu", action="store_true", help="force batching (GPU-1); CPY-1: already the default "
                        "when CuPy sees a device. Check scripts/gpu_parity.py first")
    _backend_flags(parser)
    args = parser.parse_args()
    _apply_backend_flags(args)
    if args.workers is None:
        from engine.xp import default_workers

        args.workers = default_workers()
    data = run_grid(_seeds(args.seeds), workers=args.workers, minutes=args.minutes, gpu=True if args.gpu else None)
    path = write_batch(data, args.out)
    print(f"wrote {path}: {len(data['rows'])} runs in {data['wall_seconds']} s with {data['workers']} workers")
    print(f"{'preset':18} {'response':17} {'runs':>4}  {'req@390 min/mean/max':26} {'req@1440 min/mean/max':26} halted closed")
    for s in data["summary"]:
        f = lambda d: "-" if d is None else f"{d['min']:.4f}/{d['mean']:.4f}/{d['max']:.4f}"
        print(f"{s['preset']:18} {str(s['response']):17} {s['runs']:>4}  {f(s['requested_390']):26} "
              f"{f(s['requested_1440']):26} {s['halted_runs']:>6} {s['closed_runs']:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
