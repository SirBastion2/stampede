"""Write per-generation calibration diagnostics under /calibration/diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTICS_ROOT = _REPO_ROOT / "calibration" / "diagnostics"

_SWEEPS = "parameter_sweeps.json"
_WITHDRAWN = "withdrawn_share.json"
_BINDS = "bind_rates.json"
_VARIANCE = "seed_variance.json"
_BOUNDS = "parameters_at_bounds.json"
_SIGMA = "sigma_history.json"
_SUMMARY = "summary.json"
_STATE = "recorder_state.json"

_BOUND_LOW = 0.0
_BOUND_HIGH = 1.0
_BOUND_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class GenerationData:
    """Per-generation payload the fit can pass to ``record_generation``."""

    generation: int
    objective: float
    vector: tuple[float, ...]
    sigma: float | None = None
    mean: tuple[float, ...] | None = None
    samples: tuple[tuple[float, ...], ...] | None = None
    withdrawn_share: float | None = None
    payment_cap_bind_rate: float | None = None
    closure_bind_rate: float | None = None
    seed: int | None = None


def diagnostics_dir(run_id: str, *, root: Path | None = None) -> Path:
    """``calibration/diagnostics/<run-id>`` unless ``root`` is set."""
    base = DEFAULT_DIAGNOSTICS_ROOT if root is None else Path(root)
    return base / run_id


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _vector_at_bounds(vector: Iterable[float]) -> list[dict[str, float | int]]:
    hits: list[dict[str, float | int]] = []
    for index, raw in enumerate(vector):
        value = float(raw)
        if abs(value - _BOUND_LOW) <= _BOUND_TOL or abs(value - _BOUND_HIGH) <= _BOUND_TOL:
            hits.append({"index": index, "value": value})
    return hits


def _load_state(run_dir: Path) -> dict[str, Any]:
    return _load_json(
        run_dir / _STATE,
        {
            "run_id": run_dir.name,
            "generations": [],
            "seeds": [],
            "withdrawn_share": [],
            "bind_rates": [],
        },
    )


def record_generation(
    run_dir: Path,
    generation: int,
    objective: float,
    vector: tuple[float, ...],
    *,
    sigma: float | None = None,
    mean: tuple[float, ...] | None = None,
    samples: Iterable[Iterable[float]] | None = None,
    withdrawn_share: float | None = None,
    payment_cap_bind_rate: float | None = None,
    closure_bind_rate: float | None = None,
    seed: int | None = None,
) -> Path:
    """Append one generation and rewrite the diagnostics files. Fit-callable hook."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = None
    if samples is not None:
        sample_rows = [tuple(float(x) for x in row) for row in samples]
    data = GenerationData(
        generation=int(generation),
        objective=float(objective),
        vector=tuple(float(x) for x in vector),
        sigma=None if sigma is None else float(sigma),
        mean=None if mean is None else tuple(float(x) for x in mean),
        samples=None if sample_rows is None else tuple(sample_rows),
        withdrawn_share=withdrawn_share,
        payment_cap_bind_rate=payment_cap_bind_rate,
        closure_bind_rate=closure_bind_rate,
        seed=seed,
    )
    state = _load_state(run_dir)
    state["generations"].append(asdict(data))
    if seed is not None:
        state["seeds"].append(
            {
                "seed": int(seed),
                "generation": int(generation),
                "objective": float(objective),
                "vector": list(data.vector),
            }
        )
    if withdrawn_share is not None:
        state["withdrawn_share"].append(
            {"generation": int(generation), "share": float(withdrawn_share), "seed": seed}
        )
    if payment_cap_bind_rate is not None or closure_bind_rate is not None:
        state["bind_rates"].append(
            {
                "generation": int(generation),
                "payment_cap_bind_rate": payment_cap_bind_rate,
                "closure_bind_rate": closure_bind_rate,
                "seed": seed,
            }
        )
    _write_artifacts(run_dir, state)
    return run_dir


def attach_on_generation(
    run_dir: Path,
    inner: Callable[[int, float, tuple[float, ...]], None] | None = None,
    **static: Any,
) -> Callable[[int, float, tuple[float, ...]], None]:
    """Thin hook: record then call the existing ``on_generation`` callback."""

    def hook(generation: int, objective: float, vector: tuple[float, ...]) -> None:
        record_generation(run_dir, generation, objective, vector, **static)
        if inner is not None:
            inner(generation, objective, vector)

    return hook


def write_run_diagnostics(
    run_dir: Path,
    *,
    run_id: str | None = None,
    withdrawn_share: float | None = None,
    payment_cap_bind_rate: float | None = None,
    closure_bind_rate: float | None = None,
    seed_objectives: Iterable[tuple[int, float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write or refresh the run-level files. Safe to call after the last generation."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(run_dir)
    if run_id is not None:
        state["run_id"] = run_id
    if withdrawn_share is not None:
        state["withdrawn_share"].append(
            {"generation": None, "share": float(withdrawn_share), "seed": None}
        )
    if payment_cap_bind_rate is not None or closure_bind_rate is not None:
        state["bind_rates"].append(
            {
                "generation": None,
                "payment_cap_bind_rate": payment_cap_bind_rate,
                "closure_bind_rate": closure_bind_rate,
                "seed": None,
            }
        )
    if seed_objectives is not None:
        for seed, objective in seed_objectives:
            state["seeds"].append(
                {
                    "seed": int(seed),
                    "generation": None,
                    "objective": float(objective),
                    "vector": [],
                }
            )
    if extra:
        state["extra"] = extra
    _write_artifacts(run_dir, state)
    return run_dir


def _write_artifacts(run_dir: Path, state: dict[str, Any]) -> None:
    generations: list[dict[str, Any]] = list(state.get("generations") or [])
    seeds: list[dict[str, Any]] = list(state.get("seeds") or [])
    sweeps = {
        "run_id": state.get("run_id", run_dir.name),
        "generations": [
            {
                "generation": row.get("generation"),
                "best_vector": list(row.get("vector") or ()),
                "best_objective": row.get("objective"),
                "mean": row.get("mean"),
                "samples": [list(sample) for sample in row["samples"]]
                if row.get("samples")
                else None,
            }
            for row in generations
        ],
    }
    _dump(run_dir / _SWEEPS, sweeps)

    withdrawn = {
        "run_id": state.get("run_id", run_dir.name),
        "rows": list(state.get("withdrawn_share") or []),
    }
    _dump(run_dir / _WITHDRAWN, withdrawn)

    binds = {
        "run_id": state.get("run_id", run_dir.name),
        "rows": list(state.get("bind_rates") or []),
    }
    _dump(run_dir / _BINDS, binds)

    objectives = [float(row["objective"]) for row in seeds if row.get("objective") is not None]
    variance = None
    mean_obj = None
    if objectives:
        arr: NDArray[np.float64] = np.asarray(objectives, dtype=np.float64)
        mean_obj = float(np.mean(arr))
        variance = float(np.var(arr, ddof=0))
    _dump(
        run_dir / _VARIANCE,
        {
            "run_id": state.get("run_id", run_dir.name),
            "seeds": seeds,
            "objectives": objectives,
            "mean": mean_obj,
            "variance": variance,
        },
    )

    bound_rows = []
    for row in generations:
        hits = _vector_at_bounds(row.get("vector") or ())
        if hits:
            bound_rows.append({"generation": row.get("generation"), "parameters": hits})
    _dump(
        run_dir / _BOUNDS,
        {
            "run_id": state.get("run_id", run_dir.name),
            "bound_low": _BOUND_LOW,
            "bound_high": _BOUND_HIGH,
            "rows": bound_rows,
        },
    )

    sigma_rows = [
        {"generation": row.get("generation"), "sigma": row.get("sigma")}
        for row in generations
        if row.get("sigma") is not None
    ]
    _dump(
        run_dir / _SIGMA,
        {
            "run_id": state.get("run_id", run_dir.name),
            "history": sigma_rows,
        },
    )

    last = generations[-1] if generations else {}
    summary = {
        "run_id": state.get("run_id", run_dir.name),
        "generations": len(generations),
        "last_objective": last.get("objective"),
        "last_vector": list(last.get("vector") or ()),
        "last_sigma": last.get("sigma"),
        "parameters_at_bounds": len(bound_rows),
        "seed_count": len({row.get("seed") for row in seeds if row.get("seed") is not None}),
        "objective_variance": variance,
        "withdrawn_share_rows": len(withdrawn["rows"]),
        "bind_rate_rows": len(binds["rows"]),
        "finite": last.get("objective") is None or math.isfinite(float(last["objective"])),
    }
    if state.get("extra"):
        summary["extra"] = state["extra"]
    _dump(run_dir / _SUMMARY, summary)
    _dump(run_dir / _STATE, state)
