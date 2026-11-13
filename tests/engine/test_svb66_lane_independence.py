"""SVB-66: a lane's run does not depend on the other candidates in its batch.

M03 reposts read the lane's own attention (``AttentionEngine._draw_reposts``), so a
seed-shared rumor spread (SVB-02, SVB-18a) made every candidate follow its seed's first
lane. Each lane now keeps its own rumor state: batched and solo runs are identical.
"""

from __future__ import annotations

import numpy as np

from calibration.fit import _default_svb_scenario, fixed_softness
from contracts.events import AttentionTick, WithdrawalRequested
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, simulate_logs, simulate_requested_fractions

_N_DEPOSITORS = 2000
_MINUTES = 30
_SEEDS = (17, 23)


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(_N_DEPOSITORS))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _candidates() -> list[np.ndarray]:
    base = np.linspace(0.2, 0.8, 12)
    return [
        base,
        np.clip(base + 0.07, 0.0, 1.0),
        np.clip(base * 0.6 + 0.15, 0.0, 1.0),
    ]


def _dump(log: list[object]) -> list[dict]:
    return [event.model_dump() for event in log]


def _clear() -> None:
    clear_sample_population_cache()
    clear_static_pack_cache()


def test_gpu_full_batched_lane_equals_its_solo_run(monkeypatch) -> None:
    """At 94ef7aa lane (seed 23, vector 1) took seed 23's first lane's reposts and diverged."""
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    scenario = _default_svb_scenario(_SEEDS[0])
    vectors = _candidates()
    _clear()
    batched = simulate_logs(scenario, _MINUTES, vectors, array_module="numpy", seeds=_SEEDS)
    assert len(batched) == len(vectors) * len(_SEEDS)
    withdrew = 0
    for seed_index, seed in enumerate(_SEEDS):
        for vec_index, vector in enumerate(vectors):
            _clear()
            solo = simulate_logs(scenario, _MINUTES, [vector], array_module="numpy", seeds=(seed,))[0]
            lane = batched[seed_index * len(vectors) + vec_index]
            assert _dump(lane) == _dump(solo), (seed, vec_index)
            withdrew += any(isinstance(event, WithdrawalRequested) for event in lane)
    # The case that broke SVB-02's parity: lane-specific withdrawal shocks inside the window.
    assert withdrew >= 2


def test_gpu_full_lanes_differ_only_through_their_own_attention(monkeypatch) -> None:
    """Different candidates on one seed may repost differently; identical ones never do."""
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    scenario = _default_svb_scenario(_SEEDS[1])
    vector = _candidates()[1]
    _clear()
    twins = simulate_logs(scenario, _MINUTES, [vector, vector], array_module="numpy", seeds=(_SEEDS[1],))
    assert _dump(twins[0]) == _dump(twins[1])
    levels = [e.level for e in twins[0] if isinstance(e, AttentionTick)]
    assert levels and all(np.isfinite(levels))


def test_lean_batched_lane_equals_its_solo_run(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    scenario = _default_svb_scenario(_SEEDS[0])
    vectors = [np.full(17, 0.5), np.clip(np.full(17, 0.5) + 0.12, 0.0, 1.0), np.full(17, 0.42)]
    ticks = [(float(t), 1.0) for t in range(60)]
    _clear()
    batched = simulate_requested_fractions(
        scenario, ticks, vectors, list(_SEEDS), softness=fixed_softness()
    )
    for seed_index, seed in enumerate(_SEEDS):
        for vec_index, vector in enumerate(vectors):
            _clear()
            solo = simulate_requested_fractions(
                scenario, ticks, [vector], [seed], softness=fixed_softness()
            )
            # Equal up to summation order: the real NumPy/numba stack reduces a (lanes, depositors)
            # batch in a different order than a single lane (Grok's 3060 run: up to 2.8e-8 relative
            # on the request columns). Coupling showed up as whole reposts (0.03 of attention), far
            # above this bound; test_lean_spreads_each_lane_with_its_own_attention checks the
            # mechanism itself.
            np.testing.assert_allclose(
                batched[seed_index * len(vectors) + vec_index], solo[0], rtol=1e-6, atol=0.0,
                err_msg=f"seed {seed}, vector {vec_index}",
            )


def test_lean_spreads_each_lane_with_its_own_attention(monkeypatch) -> None:
    """One spread per lane per tick, each on its own engine with that lane's attention row."""
    import engine.gpu_full as gpu_full

    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    seen: dict[float, list[tuple[int, np.ndarray]]] = {}
    original = gpu_full._spread_posted_rumors

    def _recording(module, t, posted):
        seen.setdefault(float(t), []).append((id(module), np.array(module.attention, copy=True)))
        return original(module, t, posted)

    monkeypatch.setattr(gpu_full, "_spread_posted_rumors", _recording)
    vectors = [np.full(17, 0.5), np.where(np.isin(np.arange(17), [1, 7, 12, 14]), 0.95, 0.5)]
    _clear()
    simulate_requested_fractions(
        _default_svb_scenario(_SEEDS[0]),
        [(float(t), 1.0) for t in range(40)],
        vectors,
        list(_SEEDS),
        softness=fixed_softness(),
    )
    n_lanes = len(vectors) * len(_SEEDS)
    assert seen and all(len(calls) == n_lanes for calls in seen.values())
    assert all(len({engine_id for engine_id, _ in calls}) == n_lanes for calls in seen.values())
