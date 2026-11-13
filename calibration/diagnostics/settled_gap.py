"""SVB-40 S3: where the settled-vs-requested gap sits at day-one close, by payment channel.

Diagnostic only. It changes no engine, payment, or registry state.

``settled_gap_by_channel`` is pure: it reads ``WithdrawalRequested`` and ``PaymentSettled``
events with ``t`` before day-one close and reports each channel's requested, settled, and
gap (requested minus settled) as fractions of deposits, plus what was requested in the last
``late_window`` minutes before close. Requested is by the channel the depositor asked for;
settled is by the rail M07 paid on, so the two can disagree per row (SVB-41).

``settled_gap_by_origin`` (SVB-41) reconciles the rows. M07 moves money across channels in two
ways: an app request above the app daily limit goes to the wire queue and settles as ``wire``
(``engine/bank/payments.py::enqueue_withdrawal``, ``engine/gpu_full_io.py::settle_payments_batch``),
and a ``VIPDecision`` withdrawal settles on the registry VIP channel with no
``WithdrawalRequested``. Its ``unsettled`` column is what is still owed by close.

The CLI runs stage 2's own settled measure (same engine setup as ``persist_fit --stage2``,
the shared eval seeds, one-minute ticks to day-one close, caps on) at one capacity
multiplier and prints the seed-mean per channel. Run it on the GPU box:

    python -m calibration.diagnostics.settled_gap \\
        --run-dir calibration/runs/m27-e1-svb37-20261102 --multiplier 20 --vector fit

Pairing: ``--vector fit`` (default) uses the run's fit vector and is only correct while the
registry still holds the pre-save values. ``--vector half`` uses the all-0.5 vector and is
only correct once this run's rows have been saved.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

_CHANNELS = ("app", "wire", "ach", "branch")


def _empty_row() -> dict[str, float]:
    return {"requested": 0.0, "settled": 0.0, "gap": 0.0, "late_requested": 0.0}


def settled_gap_by_channel(
    log: Iterable[object],
    *,
    close_minute: float,
    deposits: float,
    late_window: float = 30.0,
) -> dict[str, dict[str, float]]:
    """Per-channel requested, settled, gap, and late-requested fractions before ``close_minute``.

    Channel names go through ``engine.bank.payments.normalize_channel`` (so "transfer" is ACH
    and "mobile" is app, as in M07). The ``total`` row sums the channels.
    """
    from contracts.events import PaymentSettled, WithdrawalRequested
    from engine.bank.payments import normalize_channel

    if deposits <= 0.0:
        raise ValueError("deposits must be positive")
    rows: dict[str, dict[str, float]] = {name: _empty_row() for name in _CHANNELS}
    late_start = float(close_minute) - float(late_window)
    for event in log:
        if isinstance(event, WithdrawalRequested):
            if float(event.t) >= close_minute:
                continue
            row = rows.setdefault(normalize_channel(event.channel), _empty_row())
            row["requested"] += float(event.amount)
            if float(event.t) >= late_start:
                row["late_requested"] += float(event.amount)
        elif isinstance(event, PaymentSettled):
            if float(event.t) >= close_minute:
                continue
            row = rows.setdefault(normalize_channel(event.channel), _empty_row())
            row["settled"] += float(event.amount)
    total = _empty_row()
    out: dict[str, dict[str, float]] = {}
    for name, row in rows.items():
        fractions = {
            "requested": row["requested"] / deposits,
            "settled": row["settled"] / deposits,
            "late_requested": row["late_requested"] / deposits,
        }
        fractions["gap"] = fractions["requested"] - fractions["settled"]
        out[name] = fractions
        for key in total:
            total[key] += fractions[key]
    out["total"] = total
    return out


_ORIGIN_KEYS = (
    "requested",
    "vip_requested",
    "rerouted_out",
    "rerouted_in",
    "settled",
    "unsettled",
)


def _empty_origin_row() -> dict[str, float]:
    return {key: 0.0 for key in _ORIGIN_KEYS}


def _registry_vip_channel() -> str:
    from params.loader import get_parameter

    return str(get_parameter("M07 VIP withdrawal channel").value)


def settled_gap_by_origin(
    log: Iterable[object],
    *,
    close_minute: float,
    deposits: float,
    vip_channel: str | None = None,
) -> dict[str, dict[str, float]]:
    """Per-channel settled vs owed before ``close_minute``, with M07's cross-channel moves.

    Columns (fractions of deposits): ``requested`` (``WithdrawalRequested``), ``vip_requested``
    (``VIPDecision`` withdrawals, on ``vip_channel``; default the registry
    ``M07 VIP withdrawal channel``), ``rerouted_out`` / ``rerouted_in`` (app over its daily
    limit, moved to the wire queue), ``settled`` (``PaymentSettled``, same as
    :func:`settled_gap_by_channel`), and ``unsettled`` = requested + vip_requested +
    rerouted_in - rerouted_out - settled. App settles at once or moves to wire in the same
    step, so before close the app reroute is app inflow minus app settled (never below zero).
    The ``total`` row sums the channels; reroutes cancel there.
    """
    from contracts.events import PaymentSettled, VIPDecision, WithdrawalRequested
    from engine.bank.payments import normalize_channel

    if deposits <= 0.0:
        raise ValueError("deposits must be positive")
    app = normalize_channel("app")
    wire = normalize_channel("wire")
    vip = normalize_channel(_registry_vip_channel() if vip_channel is None else vip_channel)
    rows: dict[str, dict[str, float]] = {name: _empty_origin_row() for name in _CHANNELS}
    for event in log:
        if float(getattr(event, "t", close_minute)) >= close_minute:
            continue
        if isinstance(event, WithdrawalRequested):
            name = normalize_channel(event.channel)
            rows.setdefault(name, _empty_origin_row())["requested"] += float(event.amount)
        elif isinstance(event, VIPDecision):
            if event.action == "withdraw" and float(event.amount) > 0.0:
                rows.setdefault(vip, _empty_origin_row())["vip_requested"] += float(event.amount)
        elif isinstance(event, PaymentSettled):
            name = normalize_channel(event.channel)
            rows.setdefault(name, _empty_origin_row())["settled"] += float(event.amount)
    app_row = rows[app]
    moved = max(0.0, app_row["requested"] + app_row["vip_requested"] - app_row["settled"])
    app_row["rerouted_out"] += moved
    rows[wire]["rerouted_in"] += moved
    out: dict[str, dict[str, float]] = {}
    total = _empty_origin_row()
    for name, row in rows.items():
        fractions = {key: row[key] / deposits for key in _ORIGIN_KEYS if key != "unsettled"}
        fractions["unsettled"] = (
            fractions["requested"]
            + fractions["vip_requested"]
            + fractions["rerouted_in"]
            - fractions["rerouted_out"]
            - fractions["settled"]
        )
        out[name] = fractions
        for key in _ORIGIN_KEYS:
            total[key] += fractions[key]
    out["total"] = total
    return out


def _mean_rows(
    per_log: Sequence[dict[str, dict[str, float]]],
    keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    names = sorted({name for item in per_log for name in item})
    empty = {key: 0.0 for key in keys}
    mean: dict[str, dict[str, float]] = {}
    for name in names:
        rows = [item.get(name, empty) for item in per_log]
        mean[name] = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    return mean


def mean_gap_by_channel(
    logs: Sequence[Iterable[object]],
    *,
    close_minute: float,
    deposits: float,
    late_window: float = 30.0,
) -> dict[str, dict[str, float]]:
    """Seed-mean of :func:`settled_gap_by_channel` over ``logs``."""
    if not logs:
        raise ValueError("no logs")
    per_log = [
        settled_gap_by_channel(
            log, close_minute=close_minute, deposits=deposits, late_window=late_window
        )
        for log in logs
    ]
    return _mean_rows(per_log, tuple(_empty_row()))


def mean_gap_by_origin(
    logs: Sequence[Iterable[object]],
    *,
    close_minute: float,
    deposits: float,
    vip_channel: str | None = None,
) -> dict[str, dict[str, float]]:
    """Seed-mean of :func:`settled_gap_by_origin` over ``logs``."""
    if not logs:
        raise ValueError("no logs")
    per_log = [
        settled_gap_by_origin(
            log, close_minute=close_minute, deposits=deposits, vip_channel=vip_channel
        )
        for log in logs
    ]
    return _mean_rows(per_log, _ORIGIN_KEYS)


def _load_vector(run_dir: Path | None, which: str) -> tuple[list[float], int]:
    from calibration.fit import _registry_float

    best: dict[str, object] = {}
    if run_dir is not None:
        best = json.loads((run_dir / "best_vector.json").read_text(encoding="utf-8"))
    seed = int(best.get("seed", 3060))  # type: ignore[arg-type]
    if which == "half":
        width = int(_registry_float("Calibration behavioral parameter dimension"))
        return [0.5] * width, seed
    if not best:
        raise SystemExit("--vector fit needs --run-dir with best_vector.json")
    return [float(x) for x in best["vector"]], seed  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--vector", choices=("fit", "half"), default="fit")
    parser.add_argument("--multiplier", type=float, default=20.0)
    parser.add_argument("--late-window", type=float, default=30.0)
    args = parser.parse_args(argv)

    import numpy as np

    from calibration import fit
    from calibration.persist_fit import apply_stage2_engine
    from engine import gpu_sim
    from params.loader import get_parameter

    setup = apply_stage2_engine()
    vector, seed = _load_vector(args.run_dir, args.vector)
    scenario = fit._default_svb_scenario(int(seed))
    seeds = fit.shared_eval_seeds(int(seed))
    logs = gpu_sim.simulate_logs(
        scenario,
        int(fit._DAY1_CLOSE),
        [np.asarray(vector, dtype=np.float64)],
        disable_caps=False,
        softness=fit.fixed_softness(),
        clamp_inputs=False,
        seeds=seeds,
        capacity_multipliers=[float(args.multiplier)],
        schedule=fit.day1_tick_schedule(),
    )
    deposits = fit._parse_money(get_parameter("SVB-like preset: deposits").value)
    result = mean_gap_by_channel(
        logs,
        close_minute=fit.day1_close_minute(),
        deposits=deposits,
        late_window=args.late_window,
    )
    by_origin = mean_gap_by_origin(
        logs,
        close_minute=fit.day1_close_minute(),
        deposits=deposits,
    )
    print(
        json.dumps(
            {
                "vector": args.vector,
                "seed": int(seed),
                "seeds": [int(s) for s in seeds],
                "multiplier": float(args.multiplier),
                "engine": setup,
                "by_channel": result,
                "by_origin": by_origin,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
