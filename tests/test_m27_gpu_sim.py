"""M27 batched candidate eval: NumPy parity with the CPU engine (ADR-051)."""

from __future__ import annotations

import numpy as np

from calibration.fit import (
    _community_scenario,
    _default_svb_scenario,
    _objective_from_fractions,
    _score_outflow_log,
    _signature_scenario,
    load_calibration_targets,
    measure_outflow,
    objective_e1,
    real_calibration_modes,
)
from calibration.nonbank import build_nonbank_scenario, measure_price_path, summarize_price_path
from contracts.events import PriceTick
from engine.agents.decide import (
    DecisionInputs,
    apply_calibration_vector,
    build_contact_index,
    evaluate_decisions,
    hub_weights,
    load_decision_params,
)
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import (
    e1_relative_tolerance,
    gpu_sim_enabled,
    relative_error,
    simulate_logs,
    step_depositor_batch,
)
from engine.gpu_sim_kernels import _fuse, _gap_fraction, _signal_from_public
from engine.modes import ModuleMode
from engine.registry import build_engine
from params.loader import get_parameter


def _assert_close(left: float, right: float) -> None:
    assert relative_error(left, right) <= e1_relative_tolerance()


def test_gpu_sim_flags_are_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    assert gpu_sim_enabled() is False
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    assert gpu_sim_enabled() is False
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    assert gpu_sim_enabled() is True


def test_flag_off_objective_matches_itself(monkeypatch) -> None:
    """The legacy engine stays on the existing path and repeats for a fixed seed."""
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "24")
    clear_sample_population_cache()
    vector = np.linspace(0.15, 0.85, 12)
    scenario = _default_svb_scenario(11)
    first = objective_e1(
        vector,
        scenario=scenario,
        steps=3,
        targets=load_calibration_targets(),
        w_day1=float(get_parameter("Calibration E1 day-1 objective weight").value),
        w_day2=float(get_parameter("Calibration E1 two-day objective weight").value),
    )
    second = objective_e1(
        vector,
        scenario=scenario,
        steps=3,
        targets=load_calibration_targets(),
        w_day1=float(get_parameter("Calibration E1 day-1 objective weight").value),
        w_day2=float(get_parameter("Calibration E1 two-day objective weight").value),
    )
    assert first == second


class _FuseTracer:
    """cupy.fuse-like value: arithmetic only, no ``.clip`` / ``.astype`` / ``.sum``."""

    def __init__(self, data: np.ndarray) -> None:
        object.__setattr__(self, "_data", np.asarray(data, dtype=np.float64))

    def __getattr__(self, name: str):
        raise AttributeError(name)

    def __add__(self, other):
        rhs = other._data if isinstance(other, _FuseTracer) else other
        return _FuseTracer(self._data + rhs)

    __radd__ = __add__

    def __mul__(self, other):
        rhs = other._data if isinstance(other, _FuseTracer) else other
        return _FuseTracer(self._data * rhs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        rhs = other._data if isinstance(other, _FuseTracer) else other
        return _FuseTracer(self._data / rhs)

    def __gt__(self, other):
        rhs = other._data if isinstance(other, _FuseTracer) else other
        return _FuseTracer(self._data > rhs)


class _FuseXp:
    def clip(self, array, lo, hi):
        data = array._data if isinstance(array, _FuseTracer) else array
        return _FuseTracer(np.clip(data, lo, hi))

    def maximum(self, left, right):
        a = left._data if isinstance(left, _FuseTracer) else left
        b = right._data if isinstance(right, _FuseTracer) else right
        return _FuseTracer(np.maximum(a, b))


def test_fused_bodies_use_xp_functions_not_methods() -> None:
    """Fused helpers must not call ``.clip`` on tracer objects (ADR-058)."""
    wrap = _fuse(_FuseXp())
    signal = wrap(_signal_from_public)
    gap = wrap(_gap_fraction)
    raw = signal(
        _FuseTracer(0.2),
        _FuseTracer(0.5),
        _FuseTracer(0.8),
        _FuseTracer(0.3),
        _FuseTracer(2.0),
    )
    assert raw._data == np.clip(0.2 + 0.5 * 0.8 + 0.3 * 2.0, 0.0, 1.0)
    frac = gap(_FuseTracer(0.0), _FuseTracer(4.0))
    assert frac._data == 0.0
    frac = gap(_FuseTracer(10.0), _FuseTracer(4.0))
    assert frac._data == np.clip(4.0 / 10.0, 0.0, 1.0)


def test_step_depositor_batch_matches_m06_kernel() -> None:
    """One batched row uses the same amounts and belief as ``evaluate_decisions``."""
    rng = np.random.default_rng(0)
    n = 19
    params = load_decision_params()
    balance = rng.uniform(1.0e5, 5.0e6, size=n)
    insured = np.minimum(balance, 250_000.0)
    belief0 = np.full(n, params.baseline_belief)
    already = rng.random(n) < 0.25
    contacts = build_contact_index(n, params.contact_degree, seed=3)
    hub = hub_weights(balance, params)
    payroll = (rng.random(n) < 0.3).astype(np.float64)
    segment_fraction = np.full(n, params.default_fraction)
    directive = (rng.random(n) < 0.05).astype(np.float64)
    statement = np.zeros(n)
    threshold_shift = np.zeros(n)
    deposit_rate = np.zeros(n)
    vip = np.zeros(n)
    vip[2] = max(params.minimum_amount * 4.0, 10.0)
    base_threshold = np.full(n, 0.42)
    attention = 0.35
    stock = -0.08
    gap_scale = 1.0
    amount_scale = 1.15
    belief = belief0.reshape(1, -1).copy()
    bal = balance.reshape(1, -1).copy()
    alr = already.reshape(1, -1).copy()
    amounts = step_depositor_batch(
        np,
        belief=belief,
        balance=bal,
        already=alr,
        insured=insured,
        base_threshold=base_threshold,
        threshold_shift=threshold_shift.reshape(1, -1),
        statement_shift=statement.reshape(1, -1),
        deposit_rate=deposit_rate.reshape(1, -1),
        directive=directive.reshape(1, -1),
        payroll=payroll,
        segment_fraction=segment_fraction,
        vip_amount=vip.reshape(1, -1),
        hub=hub,
        contacts=contacts,
        attention_level=np.array([attention]),
        stock_return=np.array([stock]),
        gap_scale=np.array([gap_scale]),
        amount_scale=np.array([amount_scale]),
        weights=np.asarray(params.weights, dtype=np.float64).reshape(1, -1),
        bias=np.array([params.bias]),
        k=np.array([params.k]),
        belief_strength=np.array([params.belief_strength]),
        signal_attention=np.array([params.signal_attention]),
        signal_stock=np.array([params.signal_stock]),
        signal_base=np.array([params.signal_base]),
        rate_relief=np.array([params.rate_relief_weight]),
        minimum_amount=params.minimum_amount,
        chunk=7,
    )
    result = evaluate_decisions(
        DecisionInputs(
            attention=np.full(n, attention),
            belief=belief0,
            threshold=base_threshold + threshold_shift,
            balance=balance,
            insured=insured,
            stock_return=stock,
            directive=directive,
            payroll=payroll,
            contacts=contacts,
            hub_weight=hub,
            already_withdrew=already,
            statement_shift=statement,
            deposit_rate=deposit_rate,
            segment_fraction=segment_fraction,
            vip_amount=vip,
        ),
        params,
        gap_scale=gap_scale,
        amount_scale=amount_scale,
        backend="cpu",
    )
    # Since SVB-19b the batch kernel casts its inputs to float32, so it can't match the
    # float64 M06 reference bit for bit: allow float32 rounding (and a cent on amounts).
    assert np.allclose(amounts[0], result.amount, rtol=1e-5, atol=1e-2)
    assert np.array_equal(alr[0], np.logical_or(already, result.withdraw))
    assert np.allclose(belief[0], result.belief, rtol=1e-5, atol=1e-7)
    assert np.allclose(bal[0], np.maximum(0.0, balance - result.amount), rtol=1e-5, atol=1e-2)


def test_numpy_sim_matches_objective_e1(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "40")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = _default_svb_scenario(21)
    vector = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.45, 0.55, 0.35, 0.5, 0.4, 0.6])
    other = np.clip(vector + 0.1, 0.0, 1.0)
    steps = 5
    targets = load_calibration_targets()
    w_day1 = float(get_parameter("Calibration E1 day-1 objective weight").value)
    w_day2 = float(get_parameter("Calibration E1 two-day objective weight").value)

    def _legacy(vec: np.ndarray) -> tuple[float, float, bool, float]:
        day1, two_day, closed = measure_outflow(scenario, steps=steps, param_vector=vec)
        objective = _objective_from_fractions(
            day1,
            two_day,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
        )
        return day1, two_day, closed, objective

    left = _legacy(vector)
    right_components = _legacy(other)
    # Fit objective scores requested outflow; gpu-sim parity stays on settled.

    logs = simulate_logs(scenario, steps, [vector, other], array_module="numpy")
    assert len(logs) == 2
    for legacy, log in zip((left, right_components), logs, strict=True):
        day1, two_day, closed = _score_outflow_log(log)
        _assert_close(day1, legacy[0])
        _assert_close(two_day, legacy[1])
        assert closed is legacy[2]
        objective = _objective_from_fractions(
            day1,
            two_day,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
        )
        _assert_close(objective, legacy[3])

    alone = simulate_logs(scenario, steps, [other], array_module="numpy")
    day1, two_day, closed = _score_outflow_log(alone[0])
    batched_day1, batched_two, batched_closed = _score_outflow_log(logs[1])
    assert day1 == batched_day1
    assert two_day == batched_two
    assert closed is batched_closed


def test_signature_numpy_sim_matches_measure_outflow(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "28")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = _signature_scenario(9)
    vector = np.full(12, 0.5)
    steps = 4
    legacy = measure_outflow(scenario, steps=steps, param_vector=vector)
    log = simulate_logs(scenario, steps, [vector], array_module="numpy")[0]
    got = _score_outflow_log(log)
    _assert_close(got[0], legacy[0])
    _assert_close(got[1], legacy[1])
    assert got[2] is legacy[2]


def test_placebo_closed_flag_matches(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = _community_scenario(3)
    vector = np.full(12, 0.5)
    steps = 2
    legacy = measure_outflow(scenario, steps=steps, param_vector=vector)
    got = _score_outflow_log(simulate_logs(scenario, steps, [vector], array_module="numpy")[0])
    _assert_close(got[0], legacy[0])
    _assert_close(got[1], legacy[1])
    assert got[2] is legacy[2]
    assert legacy[2] is False


def _price_rows(log: list[object]) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for event in log:
        if isinstance(event, PriceTick):
            rows.append((float(event.t), float(event.price), float(event.volume)))
    return rows


def test_numpy_sim_matches_nonbank_price_paths(monkeypatch) -> None:
    """E13, E15, and E16 use the same kernel and the E1 relative tolerance."""
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "16")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    vector = np.linspace(0.2, 0.8, 12)
    other = np.clip(vector + 0.05, 0.0, 1.0)
    steps = 4
    minutes = float(get_parameter("Non-bank sim minutes per day").value)
    for event_type, seed in (("equity", 11), ("crypto", 12), ("rates", 13)):
        scenario = build_nonbank_scenario(event_type, seed)
        legacy_logs = []
        for sample in (vector, other):
            clear_sample_population_cache()
            engine = build_engine(
                modes=real_calibration_modes(),
                default_mode=ModuleMode.FAKE,
                scenario=scenario,
            )
            apply_calibration_vector(engine["M06"], sample)
            legacy_logs.append(engine.run(steps))
        clear_sample_population_cache()
        batched = simulate_logs(scenario, steps, [vector, other], array_module="numpy")
        assert len(batched) == 2
        for legacy, log in zip(legacy_logs, batched, strict=True):
            left = _price_rows(legacy)
            right = _price_rows(log)
            assert len(left) == len(right) == steps
            for (t_left, price_left, volume_left), (t_right, price_right, volume_right) in zip(
                left, right, strict=True
            ):
                assert t_left == t_right
                _assert_close(price_left, price_right)
                _assert_close(volume_left, volume_right)
            left_stats = summarize_price_path(
                [row[0] for row in left],
                [row[1] for row in left],
                [row[2] for row in left],
                minutes_per_day=minutes,
            )
            right_stats = summarize_price_path(
                [row[0] for row in right],
                [row[1] for row in right],
                [row[2] for row in right],
                minutes_per_day=minutes,
            )
            _assert_close(left_stats.peak_to_trough, right_stats.peak_to_trough)
            _assert_close(left_stats.days_to_bottom, right_stats.days_to_bottom)
            _assert_close(left_stats.volume_multiple, right_stats.volume_multiple)
            legacy_out = _score_outflow_log(legacy)
            batched_out = _score_outflow_log(log)
            _assert_close(batched_out[0], legacy_out[0])
            _assert_close(batched_out[1], legacy_out[1])
            assert batched_out[2] is legacy_out[2]
        clear_sample_population_cache()
        alone = simulate_logs(scenario, steps, [other], array_module="numpy")
        assert _price_rows(alone[0]) == _price_rows(batched[1])


def test_nonbank_flag_uses_numpy_fallback_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "12")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = build_nonbank_scenario("equity", 4)
    vector = np.linspace(0.1, 0.9, 12)
    legacy = measure_price_path(scenario, steps=3, param_vector=vector)
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    clear_sample_population_cache()
    wired = measure_price_path(scenario, steps=3, param_vector=vector)
    _assert_close(wired.peak_to_trough, legacy.peak_to_trough)
    _assert_close(wired.days_to_bottom, legacy.days_to_bottom)
    _assert_close(wired.volume_multiple, legacy.volume_multiple)
    assert wired.n_prints == legacy.n_prints


def test_enabled_flag_uses_numpy_fallback_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "20")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = _default_svb_scenario(4)
    vector = np.linspace(0.1, 0.9, 12)
    steps = 3
    legacy = measure_outflow(scenario, steps=steps, param_vector=vector)
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    clear_sample_population_cache()
    wired = measure_outflow(scenario, steps=steps, param_vector=vector)
    _assert_close(wired[0], legacy[0])
    _assert_close(wired[1], legacy[1])
    assert wired[2] is legacy[2]
