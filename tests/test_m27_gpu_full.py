"""M27 ``--gpu-full`` NumPy parity (ADR-055)."""

from __future__ import annotations

import numpy as np

from calibration.fit import _default_svb_scenario, _score_outflow_log
from calibration.nonbank import build_nonbank_scenario, summarize_price_path
from contracts.events import PriceTick
from engine.agents.attention import fallback_layers, load_attention_params, update_attention
from engine.agents.population import clear_sample_population_cache
from engine.gpu_full import csr_matvec_batch, gpu_full_enabled, update_attention_batch
from engine.gpu_full_io import price_quote_batch, settle_payments_batch
from engine.gpu_sim import e1_relative_tolerance, relative_error, simulate_logs
from mocks.demo_arc import price_quote
from params.loader import get_parameter


def _assert_close(left: float, right: float) -> None:
    assert relative_error(left, right) <= e1_relative_tolerance()


def _price_rows(log: list[object]) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for event in log:
        if isinstance(event, PriceTick):
            rows.append((float(event.t), float(event.price), float(event.volume)))
    return rows


def test_gpu_full_flags_are_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    assert gpu_full_enabled() is False
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    assert gpu_full_enabled() is False
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    assert gpu_full_enabled() is True


class _DeviceArray:
    """CuPy-like wrapper: ``np.asarray`` must not succeed (use ``.get()``)."""

    __array_ufunc__ = None

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)

    def __array__(self, *_args, **_kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. Please use .get()")

    def get(self) -> np.ndarray:
        return np.array(self._data, copy=True)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def T(self):
        return _DeviceArray(self._data.T)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(part._data if isinstance(part, _DeviceArray) else part for part in key)
        elif isinstance(key, _DeviceArray):
            key = key._data
        return _DeviceArray(self._data[key])

    def __mul__(self, other):
        rhs = other._data if isinstance(other, _DeviceArray) else other
        return _DeviceArray(self._data * rhs)

    __rmul__ = __mul__

    def __add__(self, other):
        rhs = other._data if isinstance(other, _DeviceArray) else other
        return _DeviceArray(self._data + rhs)

    __radd__ = __add__


class _FakeCupy:
    """Array module whose arrays raise on implicit NumPy conversion."""

    float64 = np.float64
    int64 = np.int64
    int32 = np.int32

    def asarray(self, array, dtype=None):
        data = array._data if isinstance(array, _DeviceArray) else np.asarray(array)
        return _DeviceArray(np.asarray(data, dtype=dtype) if dtype is not None else data)

    def zeros(self, shape, dtype=None):
        return _DeviceArray(np.zeros(shape, dtype=dtype or np.float64))

    def zeros_like(self, array):
        data = array._data if isinstance(array, _DeviceArray) else np.asarray(array)
        return _DeviceArray(np.zeros_like(data))

    class add:
        @staticmethod
        def at(target, indices, values) -> None:
            dest = target._data if isinstance(target, _DeviceArray) else target
            idx = indices._data if isinstance(indices, _DeviceArray) else indices
            src = values._data if isinstance(values, _DeviceArray) else values
            np.add.at(dest, idx, src)


def test_csr_matvec_batch_stays_on_fake_device() -> None:
    """Mimic CuPy: implicit np.asarray on the batch must not run (ADR-058)."""
    params = load_attention_params()
    layers, _ids, _banks = fallback_layers(params, "bank-svb")
    rng = np.random.default_rng(3)
    host = rng.random((4, params.fallback_nodes))
    xp = _FakeCupy()
    got = csr_matvec_batch(xp, layers[0], xp.asarray(host))
    ref = csr_matvec_batch(np, layers[0], host)
    assert np.allclose(got.get(), ref, rtol=0.0, atol=1e-12)
    decay = 0.5 ** (1.0 / 4.0)
    attention = rng.random((4, params.fallback_nodes))
    exposure = rng.random((4, params.fallback_nodes))
    device = update_attention_batch(xp, xp.asarray(attention), decay, layers, xp.asarray(exposure))
    numpy_out = update_attention_batch(np, attention, decay, layers, exposure)
    assert np.allclose(device.get(), numpy_out, rtol=0.0, atol=1e-12)


def test_attention_and_market_kernels_match_numpy() -> None:
    params = load_attention_params()
    layers, _ids, _banks = fallback_layers(params, "bank-svb")
    rng = np.random.default_rng(7)
    attention = rng.random((3, params.fallback_nodes))
    exposure = rng.random((3, params.fallback_nodes))
    decay = 0.5 ** (1.0 / 4.0)
    batched = update_attention_batch(np, attention.copy(), decay, layers, exposure)
    from engine.agents.attention import resolve_backend

    backend = resolve_backend()
    for row in range(3):
        left = update_attention(attention[row], decay, layers, exposure[row], backend=backend)
        # SVB-08's float32 SpMM against the float64 per-row update: SVB-08's own bound.
        assert np.allclose(batched[row], left, rtol=1e-5, atol=1e-7)
    t = 12.0
    strength = np.array([0.0, 0.4, 0.9])
    attn = np.array([0.1, 0.5, 0.8])
    quotes = price_quote_batch(np, t, strength, attn)
    for index in range(3):
        assert quotes[index] == price_quote(t, float(strength[index]), float(attn[index]))


def test_payments_kernel_matches_capacity_rule() -> None:
    incoming = np.array([[80.0, 50.0, 10.0, 5.0], [5.0, 0.0, 0.0, 0.0]])
    settled, app_used, wire_q, ach_q, branch_q = settle_payments_batch(
        np,
        incoming,
        app_used=np.zeros(2),
        wire_q=np.zeros(2),
        ach_q=np.zeros(2),
        branch_q=np.zeros(2),
        app_limit=30.0,
        wire_cap=20.0,
        ach_cap=8.0,
        branch_cap=4.0,
        wire_window=True,
        business_hours=True,
        ach_open=True,
    )
    assert settled[0, 0] == 30.0
    assert settled[0, 1] == 20.0
    assert settled[0, 2] == 8.0
    assert settled[0, 3] == 4.0
    assert app_used[0] == 30.0
    assert wire_q[0] == 80.0
    assert ach_q[0] == 2.0
    assert branch_q[0] == 1.0
    assert settled[1, 0] == 5.0
    assert wire_q[1] == 0.0


def test_numpy_gpu_full_matches_gpu_sim(monkeypatch) -> None:
    """Same seed: full-tick NumPy path matches the existing gpu-sim path."""
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "24")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = _default_svb_scenario(17)
    vector = np.linspace(0.2, 0.8, 12)
    other = np.clip(vector + 0.08, 0.0, 1.0)
    steps = 4
    baseline = simulate_logs(scenario, steps, [vector, other], array_module="numpy")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    clear_sample_population_cache()
    full = simulate_logs(scenario, steps, [vector, other], array_module="numpy")
    minutes = float(get_parameter("Non-bank sim minutes per day").value)
    assert len(full) == 2
    for left, right in zip(baseline, full, strict=True):
        out_left = _score_outflow_log(left, requested=True)
        out_right = _score_outflow_log(right, requested=True)
        _assert_close(out_left[0], out_right[0])
        _assert_close(out_left[1], out_right[1])
        assert out_left[2] is out_right[2]
        prices_left = _price_rows(left)
        prices_right = _price_rows(right)
        assert len(prices_left) == len(prices_right)
        for (t_l, p_l, v_l), (t_r, p_r, v_r) in zip(prices_left, prices_right, strict=True):
            assert t_l == t_r
            _assert_close(p_l, p_r)
            _assert_close(v_l, v_r)
        if prices_left:
            left_stats = summarize_price_path(
                [row[0] for row in prices_left],
                [row[1] for row in prices_left],
                [row[2] for row in prices_left],
                minutes_per_day=minutes,
            )
            right_stats = summarize_price_path(
                [row[0] for row in prices_right],
                [row[1] for row in prices_right],
                [row[2] for row in prices_right],
                minutes_per_day=minutes,
            )
            _assert_close(left_stats.peak_to_trough, right_stats.peak_to_trough)
    clear_sample_population_cache()
    again = simulate_logs(scenario, steps, [vector, other], array_module="numpy")
    left = _score_outflow_log(again[0], requested=True)
    right = _score_outflow_log(full[0], requested=True)
    assert left == right


def test_numpy_gpu_full_matches_nonbank_prices(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "16")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = build_nonbank_scenario("equity", 11)
    vector = np.linspace(0.15, 0.85, 12)
    steps = 3
    baseline = simulate_logs(scenario, steps, [vector], array_module="numpy")[0]
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    clear_sample_population_cache()
    full = simulate_logs(scenario, steps, [vector], array_module="numpy")[0]
    left = _price_rows(baseline)
    right = _price_rows(full)
    assert len(left) == len(right) == steps
    for (t_l, p_l, v_l), (t_r, p_r, v_r) in zip(left, right, strict=True):
        assert t_l == t_r
        _assert_close(p_l, p_r)
        _assert_close(v_l, v_r)


def test_numpy_batched_lanes_match_single_candidate(monkeypatch) -> None:
    """One (candidate, seed) lane matches the same seed run alone (ADR-058)."""
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "20")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    scenario = _default_svb_scenario(19)
    left_v = np.linspace(0.2, 0.7, 12)
    right_v = np.clip(left_v + 0.1, 0.0, 1.0)
    seeds = (19, 23)
    steps = 3
    batched = simulate_logs(
        scenario,
        steps,
        [left_v, right_v],
        array_module="numpy",
        seeds=seeds,
    )
    assert len(batched) == 4
    singles: list[list[object]] = []
    for seed in seeds:
        seeded = scenario.model_copy(update={"seed": int(seed)})
        for vector in (left_v, right_v):
            singles.extend(
                simulate_logs(seeded, steps, [vector], array_module="numpy")
            )
    for left, right in zip(singles, batched, strict=True):
        out_left = _score_outflow_log(left, requested=True)
        out_right = _score_outflow_log(right, requested=True)
        _assert_close(out_left[0], out_right[0])
        _assert_close(out_left[1], out_right[1])
        prices_left = _price_rows(left)
        prices_right = _price_rows(right)
        assert len(prices_left) == len(prices_right)
        for (t_l, p_l, v_l), (t_r, p_r, v_r) in zip(prices_left, prices_right, strict=True):
            assert t_l == t_r
            _assert_close(p_l, p_r)
            _assert_close(v_l, v_r)
    clear_sample_population_cache()


def test_simulate_logs_solves_population_once_per_seed(monkeypatch) -> None:
    import engine.agents.population as pop

    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "24")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    calls = 0
    original = pop._solve_balances

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pop, "_solve_balances", counting)
    scenario = _default_svb_scenario(31)
    vectors = [np.full(12, 0.3), np.full(12, 0.4)]
    timings: dict[str, float] = {}
    simulate_logs(
        scenario,
        2,
        vectors,
        array_module="numpy",
        seeds=(31, 37),
        timings=timings,
    )
    assert calls == 2
    assert timings["setup_s"] >= 0.0
    assert timings["step_s"] >= 0.0
    simulate_logs(scenario, 2, vectors, array_module="numpy", seeds=(31, 37))
    assert calls == 2
    clear_sample_population_cache()
