"""SVB-41: settled-gap diagnostic reconciles M07's app-to-wire overflow and VIP wires.

CPU-only. No engine run, no GPU, no hypothesis. The two M07 tests drive the real CPU and
array (NumPy) settlement functions so the diagnostic's reroute rule is pinned to M07.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.events import PaymentSettled, VIPDecision, WithdrawalRequested

_CLOSE = 390.0
_DEPOSITS = 1000.0


def _req(t, amount, channel):
    return WithdrawalRequested(t=t, seed=1, depositor_id="d", amount=amount, channel=channel)


def _paid(t, amount, channel):
    return PaymentSettled(t=t, seed=1, amount=amount, channel=channel, queued_remaining=0)


def _vip(t, amount, action="withdraw"):
    return VIPDecision(t=t, seed=1, agent_id="vip-1", action=action, amount=amount, rationale="")


def _s3_like_log():
    """Same shape as the SVB-41 S3 output (x1000): app over its limit, one VIP wire."""
    return [
        _vip(30.0, 0.02),
        _paid(30.0, 0.02, "wire"),
        _req(50.0, 44.7, "app"),
        _paid(50.0, 5.7, "app"),
        _paid(50.0, 39.0, "wire"),  # app overflow settles on the wire rail
        _req(60.0, 173.5, "wire"),
        _paid(60.0, 173.5, "wire"),
        _req(100.0, 14.3, "ach"),  # ACH batch is after day-one close
        _req(200.0, 13.5, "branch"),
        _paid(201.0, 2.0, "branch"),
        _vip(395.0, 5.0),  # after close: ignored
        _req(396.0, 9.0, "app"),  # after close: ignored
    ]


def test_by_channel_shows_the_rail_mismatch():
    from calibration.diagnostics.settled_gap import settled_gap_by_channel

    out = settled_gap_by_channel(_s3_like_log(), close_minute=_CLOSE, deposits=_DEPOSITS)
    assert out["wire"]["gap"] == pytest.approx(-0.03902)
    assert out["app"]["gap"] == pytest.approx(0.039)


def test_by_origin_reconciles_app_overflow_and_vip():
    from calibration.diagnostics.settled_gap import settled_gap_by_origin

    out = settled_gap_by_origin(
        _s3_like_log(), close_minute=_CLOSE, deposits=_DEPOSITS, vip_channel="wire"
    )
    assert out["app"]["requested"] == pytest.approx(0.0447)
    assert out["app"]["rerouted_out"] == pytest.approx(0.039)
    assert out["app"]["settled"] == pytest.approx(0.0057)
    assert out["app"]["unsettled"] == pytest.approx(0.0, abs=1e-12)
    assert out["wire"]["vip_requested"] == pytest.approx(0.00002)
    assert out["wire"]["rerouted_in"] == pytest.approx(0.039)
    assert out["wire"]["unsettled"] == pytest.approx(0.0, abs=1e-12)
    assert out["ach"]["unsettled"] == pytest.approx(0.0143)
    assert out["branch"]["unsettled"] == pytest.approx(0.0115)
    assert out["total"]["rerouted_in"] == pytest.approx(out["total"]["rerouted_out"])
    assert out["total"]["unsettled"] == pytest.approx(0.0258)
    total = out["total"]
    assert total["unsettled"] == pytest.approx(
        total["requested"] + total["vip_requested"] - total["settled"]
    )


def test_vip_channel_defaults_to_registry_row():
    from calibration.diagnostics.settled_gap import settled_gap_by_origin
    from engine.bank.payments import normalize_channel
    from params.loader import get_parameter

    channel = normalize_channel(get_parameter("M07 VIP withdrawal channel").value)
    out = settled_gap_by_origin(
        [_vip(5.0, 10.0), _vip(6.0, 99.0, action="stay")],
        close_minute=_CLOSE,
        deposits=_DEPOSITS,
    )
    assert out[channel]["vip_requested"] == pytest.approx(0.01)
    assert out["total"]["vip_requested"] == pytest.approx(0.01)


def _events_from_settlement(t, requests, settled):
    log = [_req(t, amount, name) for name, amount in requests]
    log += [_paid(t, amount, name) for name, amount in settled if amount > 0.0]
    return log


@pytest.mark.parametrize(("wire_cap", "wire_unsettled"), [(1000.0, 0.0), (12.0, 8.0)])
def test_matches_cpu_m07_overflow_rule(wire_cap, wire_unsettled):
    from calibration.diagnostics.settled_gap import settled_gap_by_origin
    from engine.bank.payments import ChannelLimits, PaymentState, enqueue_withdrawal, settle_tick
    from engine.core.clock import CalendarFlags

    limits = ChannelLimits(
        app_daily_limit=10.0,
        wire_capacity_per_tick=wire_cap,
        ach_batch_capacity=0.0,
        ach_batch_close_minute=1020,
        branch_capacity_per_tick=0.0,
        vip_default_channel="wire",
    )
    flags = CalendarFlags(
        business_hours=True,
        wire_window=True,
        facility_open=True,
        weekend=False,
        minute_of_day=600,
        day_index=0,
    )
    state = PaymentState()
    state, instant_app = enqueue_withdrawal(state, 25.0, "app", limits, flags)
    state, instant_wire = enqueue_withdrawal(state, 5.0, "wire", limits, flags)
    paid = settle_tick(state, flags, limits)
    settled = list(instant_app) + list(instant_wire) + sorted(paid.items())
    log = _events_from_settlement(60.0, [("app", 25.0), ("wire", 5.0)], settled)
    out = settled_gap_by_origin(log, close_minute=_CLOSE, deposits=1.0, vip_channel="wire")
    assert out["app"]["rerouted_out"] == pytest.approx(15.0)
    assert out["app"]["unsettled"] == pytest.approx(0.0, abs=1e-12)
    assert out["wire"]["unsettled"] == pytest.approx(wire_unsettled)


def test_matches_array_m07_overflow_rule():
    from calibration.diagnostics.settled_gap import settled_gap_by_origin
    from engine.gpu_full_io import settle_payments_batch

    zeros = np.zeros(1, dtype=np.float64)
    settled, *_ = settle_payments_batch(
        np,
        np.asarray([[25.0, 5.0, 0.0, 0.0]], dtype=np.float64),
        app_used=zeros.copy(),
        wire_q=zeros.copy(),
        ach_q=zeros.copy(),
        branch_q=zeros.copy(),
        app_limit=10.0,
        wire_cap=1000.0,
        ach_cap=0.0,
        branch_cap=0.0,
        wire_window=True,
        business_hours=True,
        ach_open=False,
    )
    row = [float(x) for x in np.asarray(settled)[0]]
    names = ("app", "wire", "ach", "branch")
    log = _events_from_settlement(60.0, [("app", 25.0), ("wire", 5.0)], list(zip(names, row)))
    out = settled_gap_by_origin(log, close_minute=_CLOSE, deposits=1.0, vip_channel="wire")
    assert out["app"]["rerouted_out"] == pytest.approx(15.0)
    assert out["wire"]["rerouted_in"] == pytest.approx(15.0)
    assert out["total"]["unsettled"] == pytest.approx(0.0, abs=1e-12)


def test_mean_by_origin_and_bad_inputs():
    from calibration.diagnostics.settled_gap import mean_gap_by_origin, settled_gap_by_origin

    first = [_req(1.0, 100.0, "app"), _paid(1.0, 40.0, "app"), _paid(1.0, 60.0, "wire")]
    second = [_req(1.0, 100.0, "app"), _paid(1.0, 100.0, "app")]
    out = mean_gap_by_origin(
        [first, second], close_minute=_CLOSE, deposits=_DEPOSITS, vip_channel="wire"
    )
    assert out["app"]["rerouted_out"] == pytest.approx(0.03)
    assert out["wire"]["rerouted_in"] == pytest.approx(0.03)
    assert out["total"]["unsettled"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        settled_gap_by_origin([], close_minute=_CLOSE, deposits=0.0, vip_channel="wire")
    with pytest.raises(ValueError):
        mean_gap_by_origin([], close_minute=_CLOSE, deposits=_DEPOSITS, vip_channel="wire")


def test_cli_prints_by_origin(tmp_path, monkeypatch, capsys):
    """Wiring only: GPU setup and simulate_logs are stubbed, so this stays on CPU."""
    import calibration.persist_fit as persist
    import engine.gpu_sim as gpu_sim
    from calibration.diagnostics import settled_gap

    monkeypatch.setattr(persist, "apply_stage2_engine", lambda logger=None: {"stub": True})
    monkeypatch.setattr(
        gpu_sim,
        "simulate_logs",
        lambda scenario, steps, vectors, **kwargs: [_s3_like_log() for _ in kwargs["seeds"]],
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_vector.json").write_text(
        json.dumps({"vector": [0.5] * 17, "seed": 3060, "objective": 0.0, "run_id": "run"}),
        encoding="utf-8",
    )
    assert settled_gap.main(["--run-dir", str(run), "--multiplier", "20"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["by_channel"]) >= {"app", "wire", "ach", "branch", "total"}
    origin = printed["by_origin"]
    assert set(origin) >= {"app", "wire", "ach", "branch", "total"}
    assert set(origin["total"]) == {
        "requested",
        "vip_requested",
        "rerouted_out",
        "rerouted_in",
        "settled",
        "unsettled",
    }
    assert origin["total"]["rerouted_in"] == pytest.approx(origin["total"]["rerouted_out"])
