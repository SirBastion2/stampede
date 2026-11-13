"""M06 real decision kernel: zero attention, complex contagion k, CPU determinism."""

from __future__ import annotations

import numpy as np
import pytest

from contracts import parse_event
from contracts.events import StatementIssued, WithdrawalRequested
from contracts.schemas import DepositorArrays, Scenario
from engine.agents.decide import (
    DecisionInputs,
    DecisionModule,
    active_backend_name,
    build_contact_index,
    cuda_device_available,
    evaluate_decisions,
    load_decision_params,
    public_signal,
    run_gpu_decision,
    update_belief,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.bus import EventBus
from mocks.l1_engine import DecisionFake
from mocks.support import bank_id_for
from params.loader import RegistryError, get_parameter
from tests.test_fakes import panic


def _params():
    return load_decision_params()


def _inputs(
    *,
    attention: list[float],
    threshold: list[float],
    balance: list[float],
    insured: list[float],
    contacts: list[list[int]],
    hub_weight: list[float] | None = None,
    directive: list[float] | None = None,
    payroll: list[float] | None = None,
    belief: list[float] | None = None,
    statement_shift: list[float] | None = None,
    deposit_rate: list[float] | None = None,
    segment_fraction: list[float] | None = None,
    vip_amount: list[float] | None = None,
    stock_return: float = 0.0,
    already_withdrew: list[bool] | None = None,
) -> DecisionInputs:
    n = len(attention)
    params = _params()
    belief_row = [params.baseline_belief] * n if belief is None else belief
    return DecisionInputs(
        attention=np.asarray(attention, dtype=np.float64),
        belief=np.asarray(belief_row, dtype=np.float64),
        threshold=np.asarray(threshold, dtype=np.float64),
        balance=np.asarray(balance, dtype=np.float64),
        insured=np.asarray(insured, dtype=np.float64),
        stock_return=stock_return,
        directive=np.asarray([0.0] * n if directive is None else directive, dtype=np.float64),
        payroll=np.asarray([0.0] * n if payroll is None else payroll, dtype=np.float64),
        contacts=np.asarray(contacts, dtype=np.int64),
        hub_weight=np.asarray(
            [1.0] * n if hub_weight is None else hub_weight,
            dtype=np.float64,
        ),
        already_withdrew=np.asarray(
            [False] * n if already_withdrew is None else already_withdrew,
            dtype=np.bool_,
        ),
        statement_shift=np.asarray(
            [0.0] * n if statement_shift is None else statement_shift,
            dtype=np.float64,
        ),
        deposit_rate=np.asarray(
            [0.0] * n if deposit_rate is None else deposit_rate,
            dtype=np.float64,
        ),
        segment_fraction=np.asarray(
            [params.segment_fraction["startup"]] * n
            if segment_fraction is None
            else segment_fraction,
            dtype=np.float64,
        ),
        vip_amount=np.asarray([0.0] * n if vip_amount is None else vip_amount, dtype=np.float64),
    )


def _run(inputs: DecisionInputs, **kwargs: float):
    return evaluate_decisions(inputs, _params(), backend="cpu", **kwargs)


def test_load_decision_params_accepts_fitted_worry_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.agents import decide

    rows = decide._assumed_rows()
    rows[decide._BIAS] = ("-1.25", "FITTED")
    monkeypatch.setattr(decide, "_assumed_rows", lambda: rows)
    params = load_decision_params()
    assert params.bias == -1.25


def test_load_decision_params_rejects_wrong_status_on_other_m06_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.agents import decide

    rows = decide._assumed_rows()
    value, _status = rows[decide._K]
    rows[decide._K] = (value, "FITTED")
    monkeypatch.setattr(decide, "_assumed_rows", lambda: rows)
    with pytest.raises(RegistryError, match=r"must be ASSUMED, got FITTED"):
        load_decision_params()


def test_k_is_assumed_row_and_tbd_row_stays_unset() -> None:
    assumed = get_parameter("M06 complex contagion k (ASSUMED)")
    tbd = get_parameter("Depositor complex contagion minimum worried contacts (k)")
    params = _params()
    assert assumed.status == "ASSUMED"
    assert "ASSUMED" in assumed.parameter
    assert params.k == float(assumed.value)
    assert tbd.value == "TBD"
    assert tbd.status == "UNSET"


def test_zero_attention_produces_no_run() -> None:
    params = _params()
    n = 6
    contacts = [[(i + 1) % n, (i + 2) % n, (i + 3) % n] for i in range(n)]
    inputs = _inputs(
        attention=[0.0] * n,
        threshold=[params.worry_ceiling] * n,
        balance=[1_000_000.0] * n,
        insured=[0.0] * n,
        contacts=contacts,
        payroll=[1.0] * n,
        hub_weight=[params.hub_weight] * n,
    )
    result = _run(inputs)
    assert result.backend == "cpu"
    assert float(np.max(result.worry)) < params.worry_ceiling
    assert not bool(np.any(result.withdraw))
    assert float(np.max(result.amount)) == 0.0


def test_withdrawal_requires_at_least_k_worried_contacts() -> None:
    params = _params()
    k = int(params.k)
    # Focal depositor 0. Contacts 1..k are the neighbor slots; the last starts unworried.
    n = k + 2
    contacts = [[index for index in range(1, k + 1)] for _ in range(n)]
    low = params.worry_ceiling
    high = 0.99
    thresholds = [low] * n
    thresholds[k] = high  # one neighbor is not worried
    inputs = _inputs(
        attention=[1.0] * n,
        threshold=thresholds,
        balance=[1_000_000.0] * n,
        insured=[0.0] * n,
        contacts=contacts,
    )
    blocked = _run(inputs, aware=np.ones(n))
    assert float(blocked.worry[0]) > low
    assert float(blocked.worried_contacts[0]) == pytest.approx(k - 1)
    assert bool(blocked.withdraw[0]) is False

    thresholds[k] = low
    opened = _run(
        _inputs(
            attention=[1.0] * n,
            threshold=thresholds,
            balance=[1_000_000.0] * n,
            insured=[0.0] * n,
            contacts=contacts,
        ),
        aware=np.ones(n),
    )
    assert float(opened.worried_contacts[0]) == pytest.approx(k)
    assert bool(opened.withdraw[0]) is True
    fraction = params.segment_fraction["startup"]
    assert float(opened.amount[0]) == pytest.approx(fraction * 1_000_000.0)


def test_hub_contacts_carry_extra_weight() -> None:
    params = _params()
    # Two worried neighbors. Only the hub weight reaches k.
    contacts = [[1, 2], [0, 2], [0, 1]]
    base = dict(
        attention=[1.0, 1.0, 1.0],
        threshold=[params.worry_ceiling] * 3,
        balance=[1_000_000.0] * 3,
        insured=[0.0] * 3,
        contacts=contacts,
    )
    plain = _run(_inputs(**base, hub_weight=[1.0, 1.0, 1.0]), aware=np.ones(3))
    assert float(plain.worried_contacts[0]) == pytest.approx(2.0)
    assert bool(plain.withdraw[0]) is False

    hubs = _run(_inputs(**base, hub_weight=[1.0, params.hub_weight, 1.0]), aware=np.ones(3))
    assert float(hubs.worried_contacts[0]) == pytest.approx(1.0 + params.hub_weight)
    assert params.hub_weight + 1.0 >= params.k
    assert bool(hubs.withdraw[0]) is True


def test_insured_gap_sets_the_amount_and_fully_insured_do_not_withdraw() -> None:
    params = _params()
    k = int(params.k)
    n = k + 1
    contacts = [[index for index in range(n) if index != row][:k] for row in range(n)]
    inputs = _inputs(
        attention=[1.0] * n,
        threshold=[params.worry_ceiling] * n,
        balance=[1_000.0] * n,
        insured=[1_000.0] * n,
        contacts=contacts,
    )
    result = _run(inputs, aware=np.ones(n))
    assert float(np.max(result.worry)) > params.worry_ceiling
    assert not bool(np.any(result.withdraw))


def test_cpu_path_is_deterministic_for_a_fixed_seed() -> None:
    params = _params()
    first = build_contact_index(32, params.contact_degree, seed=11)
    second = build_contact_index(32, params.contact_degree, seed=11)
    other = build_contact_index(32, params.contact_degree, seed=12)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)

    n = 8
    contacts = build_contact_index(n, 3, seed=11)
    kwargs = dict(
        attention=[0.2 * (index + 1) for index in range(n)],
        threshold=[params.worry_ceiling] * n,
        balance=[500_000.0 + index for index in range(n)],
        insured=[100_000.0] * n,
        contacts=contacts.tolist(),
        payroll=[1.0 if index % 2 == 0 else 0.0 for index in range(n)],
        stock_return=-0.04,
    )
    left = _run(_inputs(**kwargs))
    right = _run(_inputs(**kwargs))
    assert left.backend == "cpu"
    assert np.array_equal(left.worry, right.worry)
    assert np.array_equal(left.amount, right.amount)
    assert np.array_equal(left.withdraw, right.withdraw)
    assert np.array_equal(left.belief, right.belief)


def test_belief_updates_from_public_signal_and_holds_when_uninformative() -> None:
    params = _params()
    prior = np.asarray([params.baseline_belief, params.baseline_belief], dtype=np.float64)
    quiet = public_signal(np.zeros(2, dtype=np.float64), 0.0, params)
    assert np.allclose(quiet, params.signal_base)
    held = update_belief(prior, quiet, params.belief_strength)
    assert np.allclose(held, prior)

    loud = public_signal(np.ones(2, dtype=np.float64), -0.2, params)
    assert float(loud[0]) > params.signal_base
    moved = update_belief(prior, loud, params.belief_strength)
    assert float(moved[0]) > params.baseline_belief
    assert np.allclose(moved, update_belief(prior, loud, params.belief_strength))


def test_gpu_path_is_isolated_on_cpu_ci() -> None:
    params = _params()
    inputs = _inputs(
        attention=[0.0],
        threshold=[params.worry_ceiling],
        balance=[1.0],
        insured=[0.0],
        contacts=np.zeros((1, 0), dtype=np.int64).tolist(),
    )
    if cuda_device_available():
        gpu = run_gpu_decision(inputs, params)
        assert gpu.backend == "gpu"
        cpu = _run(inputs)
        assert cpu.backend == "cpu"
        assert np.allclose(gpu.worry, cpu.worry)
        assert np.array_equal(gpu.withdraw, cpu.withdraw)
        assert np.allclose(gpu.amount, cpu.amount)
    else:
        assert active_backend_name() == "cpu"
        with pytest.raises(RuntimeError, match="CUDA"):
            run_gpu_decision(inputs, params)


def test_statement_shifts_worry_for_the_named_segment() -> None:
    params = _params()
    shared = dict(
        attention=[1.0, 1.0],
        threshold=[params.worry_ceiling, params.worry_ceiling],
        balance=[1_000_000.0, 1_000_000.0],
        insured=[0.0, 0.0],
        contacts=[[1], [0]],
    )
    plain = _run(_inputs(**shared))
    shifted = _run(
        _inputs(**shared, statement_shift=[params.statement_logit_shift, 0.0])
    )
    assert params.statement_logit_shift < 0.0
    assert float(shifted.worry[0]) < float(plain.worry[0])
    assert float(shifted.worry[1]) == pytest.approx(float(plain.worry[1]))

    arrays = DepositorArrays(
        bank_idx=[0, 0],
        segment=["startup", "retail"],
        balance=[1_000_000.0, 1_000_000.0],
        insured=[0.0, 0.0],
        belief=[params.baseline_belief, params.baseline_belief],
        threshold=[params.worry_ceiling, params.worry_ceiling],
        attention=[0.0, 0.0],
        needs_flag=[True, False],
        has_second_bank=[False, False],
        state=["active", "active"],
    )
    scenario = panic(4)
    module = DecisionModule("M06")
    bus = EventBus()
    module.attach(bus, scenario)
    module.load_depositor_arrays(arrays, seed=scenario.seed)
    bus.publish(
        StatementIssued(
            t=0.0,
            seed=scenario.seed,
            bank_id=bank_id_for(scenario.preset),
            stmt_id="stmt-1",
            channel="press",
            segments=["startup"],
        )
    )
    events = module.step(0.0)
    assert float(module.statement_shift[0]) == params.statement_logit_shift
    assert float(module.threshold_shift[0]) == params.statement_threshold_shift
    assert float(module.statement_shift[1]) == 0.0
    assert all(isinstance(event, WithdrawalRequested) for event in events)


def test_fake_stays_default_and_real_mode_is_wired() -> None:
    scenario = panic(3)
    fake_engine = build_engine(scenario=scenario)
    assert isinstance(fake_engine["M06"], DecisionFake)
    real_engine = build_engine(modes={"M06": ModuleMode.REAL}, scenario=scenario)
    assert isinstance(real_engine["M06"], DecisionModule)
    assert isinstance(fake_engine["M06"], DecisionFake)

    real_log = real_engine.run(1)
    again = run_scenario(scenario, 1, modes={"M06": ModuleMode.REAL})
    real_dump = [event.model_dump(mode="json", by_alias=True) for event in real_log]
    again_dump = [event.model_dump(mode="json", by_alias=True) for event in again]
    assert real_dump == again_dump
    for event in real_log:
        payload = event.model_dump(mode="json", by_alias=True)
        parsed = parse_event(payload)
        assert parsed.model_dump(mode="json", by_alias=True) == payload
        if isinstance(event, WithdrawalRequested):
            assert event.amount > 0.0
            assert event.channel
            assert event.depositor_id.startswith("dep-")


def test_community_placebo_publishes_no_withdrawals() -> None:
    scenario = Scenario(preset="insured-community", seed=1)
    log = run_scenario(scenario, 2, modes={"M06": ModuleMode.REAL})
    assert not any(isinstance(event, WithdrawalRequested) for event in log)


def test_other_real_mode_still_raises() -> None:
    with pytest.raises(NotImplementedError, match="M30"):
        build_engine(modes={"M30": ModuleMode.REAL})
