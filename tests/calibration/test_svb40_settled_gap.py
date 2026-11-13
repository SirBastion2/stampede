"""SVB-40 S3: settled-vs-requested gap by channel at day-one close. CPU-only, no engine run."""
from __future__ import annotations

import json

import pytest

from contracts.events import PaymentSettled, WithdrawalRequested

_CLOSE = 390.0
_DEPOSITS = 1000.0


def _req(t, amount, channel):
    return WithdrawalRequested(t=t, seed=1, depositor_id="d", amount=amount, channel=channel)


def _paid(t, amount, channel):
    return PaymentSettled(t=t, seed=1, amount=amount, channel=channel, queued_remaining=0)


def _log():
    return [
        _req(10.0, 100.0, "wire"),
        _paid(11.0, 100.0, "wire"),
        _req(100.0, 50.0, "ach"),              # ACH requested on day one, never settled by close
        _req(380.0, 20.0, "branch"),           # late request
        _paid(385.0, 5.0, "branch"),
        _req(389.0, 30.0, "wire"),             # late request that settles after close
        _paid(395.0, 30.0, "wire"),            # after close: ignored
        _req(400.0, 70.0, "wire"),             # after close: ignored
    ]


def test_per_channel_fractions_before_close():
    from calibration.diagnostics.settled_gap import settled_gap_by_channel

    out = settled_gap_by_channel(_log(), close_minute=_CLOSE, deposits=_DEPOSITS)
    assert out["wire"]["requested"] == pytest.approx(0.13)
    assert out["wire"]["settled"] == pytest.approx(0.10)
    assert out["wire"]["gap"] == pytest.approx(0.03)
    assert out["wire"]["late_requested"] == pytest.approx(0.03)
    assert out["ach"]["requested"] == pytest.approx(0.05)
    assert out["ach"]["settled"] == pytest.approx(0.0)
    assert out["ach"]["gap"] == pytest.approx(0.05)
    assert out["ach"]["late_requested"] == pytest.approx(0.0)
    assert out["branch"]["gap"] == pytest.approx(0.015)
    assert out["branch"]["late_requested"] == pytest.approx(0.02)
    assert out["app"] == {"requested": 0.0, "settled": 0.0, "late_requested": 0.0, "gap": 0.0}
    assert out["total"]["requested"] == pytest.approx(0.20)
    assert out["total"]["settled"] == pytest.approx(0.105)
    assert out["total"]["gap"] == pytest.approx(0.095)


def test_channel_aliases_match_m07():
    from calibration.diagnostics.settled_gap import settled_gap_by_channel

    log = [_req(1.0, 10.0, "transfer"), _req(2.0, 20.0, "mobile"), _paid(3.0, 20.0, "app")]
    out = settled_gap_by_channel(log, close_minute=_CLOSE, deposits=_DEPOSITS)
    assert out["ach"]["requested"] == pytest.approx(0.01)
    assert out["app"]["requested"] == pytest.approx(0.02)
    assert out["app"]["gap"] == pytest.approx(0.0)


def test_mean_over_seeds():
    from calibration.diagnostics.settled_gap import mean_gap_by_channel

    first = [_req(1.0, 100.0, "wire"), _paid(2.0, 100.0, "wire")]
    second = [_req(1.0, 100.0, "wire")]
    out = mean_gap_by_channel([first, second], close_minute=_CLOSE, deposits=_DEPOSITS)
    assert out["wire"]["requested"] == pytest.approx(0.10)
    assert out["wire"]["settled"] == pytest.approx(0.05)
    assert out["wire"]["gap"] == pytest.approx(0.05)


def test_bad_inputs_raise():
    from calibration.diagnostics.settled_gap import mean_gap_by_channel, settled_gap_by_channel

    with pytest.raises(ValueError):
        settled_gap_by_channel([], close_minute=_CLOSE, deposits=0.0)
    with pytest.raises(ValueError):
        mean_gap_by_channel([], close_minute=_CLOSE, deposits=_DEPOSITS)


def test_cli_runs_with_a_stubbed_simulation(tmp_path, monkeypatch, capsys):
    """Wiring only: the GPU engine setup and simulate_logs are stubbed, so this stays on CPU."""
    import calibration.persist_fit as persist
    import engine.gpu_sim as gpu_sim
    from calibration.diagnostics import settled_gap

    calls: dict[str, object] = {}

    def fake_simulate_logs(scenario, steps, vectors, **kwargs):
        calls["steps"] = steps
        calls["kwargs"] = kwargs
        calls["n_vectors"] = len(vectors)
        return [_log() for _ in kwargs["seeds"]]

    monkeypatch.setattr(persist, "apply_stage2_engine", lambda logger=None: {"stub": True})
    monkeypatch.setattr(gpu_sim, "simulate_logs", fake_simulate_logs)
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_vector.json").write_text(
        json.dumps({"vector": [0.5] * 17, "seed": 3060, "objective": 0.0, "run_id": "run"}),
        encoding="utf-8",
    )
    assert settled_gap.main(["--run-dir", str(run), "--multiplier", "20"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["multiplier"] == 20.0
    assert printed["vector"] == "fit"
    assert calls["steps"] == 390
    assert calls["n_vectors"] == 1
    assert calls["kwargs"]["capacity_multipliers"] == [20.0]
    assert calls["kwargs"]["disable_caps"] is False
    assert set(printed["by_channel"]) >= {"app", "wire", "ach", "branch", "total"}
