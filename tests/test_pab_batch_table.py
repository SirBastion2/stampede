"""PA-B: the multi-seed batch table (grid, one run, summary, file, server)."""

from __future__ import annotations

import pytest

from ui.live import batch


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def test_grid_covers_every_preset_response_and_seed() -> None:
    cells = batch.grid((1, 2))
    # UI-3: the bond rumor gets each response at four minutes; frozen-branches at 1:30 PM; stadium none
    assert len(cells) == (1 + 3 * 4) * 2 + (1 + 3) * 2 + 2 and len(set(cells)) == len(cells)
    assert {r for p, r, m, s in cells if p in batch.PRESETS} == {None, "guarantee", "credit-line", "withdrawal-limit"}
    assert {m for p, r, m, s in cells if p == "lost-bonds" and r} == set(batch.TIMING_MINUTES)
    assert {m for p, r, m, s in cells if p == "frozen-branches" and r} == {240}
    assert {r for p, r, m, s in cells if p == "stadium"} == {None}
    assert len(batch.grid()) == 180 and batch.DEFAULT_SEEDS == tuple(range(1, 11))


def test_one_run_is_deterministic_and_complete() -> None:
    a = batch.run_one("lost-bonds", "withdrawal-limit", 3, minutes=8)
    b = batch.run_one("lost-bonds", "withdrawal-limit", 3, minutes=8)
    a.pop("seconds"), b.pop("seconds")
    assert a == b
    assert a["minutes"] == 8 and a["series"][-1]["minute"] == 8 and a["label"]["kind"] == "scenario"
    assert set(a["series"][0]) == {"minute", "requested", "price", "cash", "withdrew", "wavering", "calm"}
    assert batch.run_one("lost-bonds", None, 3, minutes=2)["label"]["kind"] == "fitted"


def test_summary_spreads() -> None:
    rows = [{"preset": "p", "response": None, "label": {"kind": "fitted"}, "halt_minute": h, "closure_minute": c,
             "cash_out_minute": None, "requested_at": {"390": v, "1440": 2 * v}}
            for h, c, v in ((380.0, 1440.0, 0.25), (None, None, 0.27), (390.0, 1440.0, 0.26))]
    s, = batch.summarize(rows)
    assert s["runs"] == 3 and s["requested_390"] == {"min": 0.25, "mean": 0.26, "max": 0.27}
    assert s["halted_runs"] == 2 and s["halt_minute"]["mean"] == 385.0 and s["closed_runs"] == 2
    assert s["cash_out_runs"] == 0 and s["cash_out_minute"] is None


def test_grid_file_round_trip(tmp_path) -> None:
    data = batch.run_grid((1,), workers=1, minutes=3)
    assert len(data["rows"]) == 18 and len(data["summary"]) == 18
    path = batch.write_batch(data, str(tmp_path / "b.json"))
    assert batch.load_batch(path)["summary"] == data["summary"]
    assert batch.load_batch(str(tmp_path / "missing.json")) is None


def test_server_serves_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    monkeypatch.setenv("STAMPEDE_BATCH_FILE", str(tmp_path / "none.json"))
    client = TestClient(server.app)
    assert client.get("/api/batch").json()["ok"] is False
    batch.write_batch(batch.run_grid((1,), workers=1, minutes=2), str(tmp_path / "none.json"))
    body = client.get("/api/batch").json()
    assert body["ok"] is True and "rows" not in body and len(body["summary"]) == 18
    run = client.get("/api/batch/run", params={"preset": "lost-bonds", "seed": 1}).json()
    assert run["ok"] is True and run["run"]["response"] is None
