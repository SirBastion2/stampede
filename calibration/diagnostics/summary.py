"""Print a short summary of a calibration diagnostics directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.diagnostics.recorder import DEFAULT_DIAGNOSTICS_ROOT


def _load(path: Path) -> dict[object, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def format_summary(run_dir: Path) -> str:
    """Human-readable block for one diagnostics directory."""
    run_dir = Path(run_dir)
    summary = _load(run_dir / "summary.json")
    sweeps = _load(run_dir / "parameter_sweeps.json")
    withdrawn = _load(run_dir / "withdrawn_share.json")
    binds = _load(run_dir / "bind_rates.json")
    variance = _load(run_dir / "seed_variance.json")
    bounds = _load(run_dir / "parameters_at_bounds.json")
    sigma = _load(run_dir / "sigma_history.json")
    generations = sweeps.get("generations") if isinstance(sweeps.get("generations"), list) else []
    withdrawn_rows = withdrawn.get("rows") if isinstance(withdrawn.get("rows"), list) else []
    bind_rows = binds.get("rows") if isinstance(binds.get("rows"), list) else []
    bound_rows = bounds.get("rows") if isinstance(bounds.get("rows"), list) else []
    sigma_hist = sigma.get("history") if isinstance(sigma.get("history"), list) else []
    last_sigma = None
    if sigma_hist:
        last = sigma_hist[-1]
        if isinstance(last, dict):
            last_sigma = last.get("sigma")
    lines = [
        f"run_id={summary.get('run_id', run_dir.name)}",
        f"generations={len(generations)}",
        f"last_objective={summary.get('last_objective')}",
        f"last_sigma={last_sigma}",
        f"withdrawn_share_rows={len(withdrawn_rows)}",
        f"bind_rate_rows={len(bind_rows)}",
        f"parameters_at_bounds={len(bound_rows)}",
        f"seed_variance={variance.get('variance')}",
        f"seed_mean_objective={variance.get('mean')}",
    ]
    return "\n".join(lines) + "\n"


def print_summary(run_dir: Path) -> str:
    text = format_summary(run_dir)
    print(text, end="")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print M27 calibration diagnostics.")
    parser.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run folder under calibration/diagnostics/ (default: newest).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_ROOT,
        help="Diagnostics root (default: calibration/diagnostics).",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.run_id:
        target = root / args.run_id if not Path(args.run_id).is_dir() else Path(args.run_id)
    else:
        candidates = [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "summary.json").is_file()
        ]
        if not candidates:
            print(f"no diagnostics under {root}")
            return 1
        target = max(candidates, key=lambda path: path.stat().st_mtime)
    if not target.is_dir():
        print(f"missing diagnostics directory: {target}")
        return 1
    print_summary(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
