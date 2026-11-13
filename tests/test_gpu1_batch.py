"""GPU-1 (sandbox test): the search's runs batched with the depositor kernel; NumPy stands in for CuPy here."""

from __future__ import annotations

import pytest

from ui.live import search


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def test_batched_runs_match_the_cpu_path() -> None:
    from ui.live.gpu_batch import run_plans_batched

    cells = [([], 2, 40), ([("guarantee", ("requested", 0.00002))], 1, 40), ([], 1, 25)]
    cpu = [search.run_plan(c) for c in cells]
    timings: dict = {}
    gpu = run_plans_batched(cells, array_module="numpy", timings=timings)
    assert [r["seed"] for r in gpu] == [2, 1, 1] and [r["minutes"] for r in gpu] == [40, 40, 25]  # input order kept
    for a, b in zip(cpu, gpu):
        assert a["plan"] == b["plan"] and a["open"] == b["open"] and a["fired"].keys() == b["fired"].keys()
        assert b["requested"] == pytest.approx(a["requested"], rel=1e-4)
        assert len(a["series"]) == len(b["series"])
        for x, y in zip(a["series"], b["series"]):
            assert x[0] == y[0] and y[1] == pytest.approx(x[1], rel=1e-3, abs=1e-6) and y[2] == pytest.approx(x[2], abs=1e-4)
            assert y[4] == pytest.approx(x[4], abs=2e-3)  # wavering share, from the recovered worry
    assert gpu[1]["fired"]["guarantee"] == pytest.approx(cpu[1]["fired"]["guarantee"], abs=1)
    assert timings["backend"] == "cpu" and timings["lane_minutes"] == 105 and timings["kernel_s"] > 0


def test_the_runner_batches_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(search, "_run_batch", lambda cells: seen.append(len(cells)) or
                        [{"plan": search.plan_id(p), "seed": s} for p, s, *_ in cells])
    monkeypatch.setattr(search, "run_plan", lambda c: pytest.fail("CPU path used"))
    monkeypatch.setenv("STAMPEDE_GPU_LANES", "2")
    runner = search.Runner(1, 10, batched=True)
    runner.run([([], 1), ([], 2), ([("guarantee", ("requested", 0.08))], 1)])
    assert seen == [2, 1] and len(runner.done) == 3 and sorted(runner.done) == sorted([("none", 1), ("none", 2), ("guarantee@req0.08", 1)])
    monkeypatch.setenv("STAMPEDE_SEARCH_GPU", "1")
    assert search.Runner(1, 10).batched and not search.Runner(1, 10, batched=False).batched
    monkeypatch.delenv("STAMPEDE_SEARCH_GPU")
    assert not search.Runner(1, 10).batched


def test_batch_table_cells_match_the_cpu_path() -> None:
    from ui.live import batch
    from ui.live.gpu_batch import run_batch_cells_batched

    cells = [("lost-bonds", "withdrawal-limit", 15, 1, 35), ("lost-bonds", None, None, 2, 35)]
    cpu = [batch.run_one(p, r, s, m, rm) for p, r, rm, s, m in cells]
    gpu = run_batch_cells_batched(cells, array_module="numpy")
    for a, b in zip(cpu, gpu):
        assert (a["preset"], a["response"], a["seed"], a["response_enacted_minute"]) == \
               (b["preset"], b["response"], b["seed"], b["response_enacted_minute"])
        assert [x["minute"] for x in a["series"]] == [x["minute"] for x in b["series"]]
        for x, y in zip(a["series"], b["series"]):
            assert y["requested"] == pytest.approx(x["requested"], rel=1e-3, abs=1e-6)
            assert abs(y["withdrew"] - x["withdrew"]) <= 5 and abs(y["wavering"] - x["wavering"]) <= 50
            assert y["price"] == pytest.approx(x["price"], rel=1e-4)
        assert a["label"] == b["label"] and a["closure_minute"] == b["closure_minute"]
