"""UI-4: viewers choose how many runs per plan to include (the first N of a pool run before the demo)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui.live import search


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _stand_in(monkeypatch: pytest.MonkeyPatch) -> list:
    from tools.ui4_preview import _fake_run

    calls: list = []

    def run(args):
        row = _fake_run(args)
        calls.append((row["plan"], row["seed"]))
        return row

    monkeypatch.setattr(search, "run_plan", run)
    return calls


def test_one_pool_serves_every_option(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stand_in(monkeypatch)
    results = search.run_trigger_options(options=(10, 20))
    assert sorted(results) == [10, 20]
    assert len(calls) == len(set(calls))  # each (plan, seed) is run once, whatever the options
    for n, data in results.items():
        assert data["seeds"] == list(range(1, n + 1))  # the first N, never a chosen subset
        assert data["runs_per_plan"] == n and data["run_options"] == [10, 20] and data["pool_seeds"] == 20
        assert all(len(p["open_by_seed"]) == n for p in data["plans"])
        assert data["holdout"]["seeds"] == list(range(1001, 1011))  # new days, outside the pool
        assert data["deadline_seeds"] == 10
        assert {int(k.split("|")[1]) for k in data["series"]} <= set(range(1, n + 1)) | set(range(1001, 1011))
    assert not set(range(1001, 1011)) & set(range(1, 21))
    with pytest.raises(ValueError):
        search.run_trigger_options(options=(5, 10))


def test_files_per_option_and_the_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stand_in(monkeypatch)
    path = str(tmp_path / "live_search.json")
    monkeypatch.setenv("STAMPEDE_SEARCH_FILE", path)
    written = search.write_search_options(search.run_trigger_options(options=(10, 20)))
    assert written == [str(tmp_path / "live_search_10.json"), str(tmp_path / "live_search_20.json"), path]
    assert search.available_run_options() == [10, 20]
    assert search.load_search()["runs_per_plan"] == 20 and search.load_search(runs=10)["runs_per_plan"] == 10
    assert search.load_search(runs=30) is None


def test_server_serves_the_chosen_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ui.server import app

    _stand_in(monkeypatch)
    monkeypatch.setenv("STAMPEDE_SEARCH_FILE", str(tmp_path / "live_search.json"))
    search.write_search_options(search.run_trigger_options(options=(10, 20)))
    client = TestClient(app)
    d = client.get("/api/search", params={"runs": 10}).json()
    assert d["ok"] and d["runs_per_plan"] == 10 and d["run_options"] == [10, 20] and "series" not in d
    assert client.get("/api/search").json()["runs_per_plan"] == 20
    assert not client.get("/api/search", params={"runs": 30}).json()["ok"]


def test_the_page_has_the_selector_the_sync_and_the_days_strip() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    for needle in ('class="runsPick"', "function chooseRuns", "BroadcastChannel('stampede-runs')", "function daysStrip",
                   "'/api/search' + (runs ? `?runs=${runs}` : '')", "never a chosen subset",
                   "Versions of the day</span>"):   # UF-2 counts groups; UI-5 calls them versions of the day
        assert needle in html, needle
    assert html.count('class="runsPick"') == 2  # the wall and the Why page


def test_the_preview_is_marked_as_placeholder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tools.ui4_preview as preview

    monkeypatch.setattr(search, "run_plan", search.run_plan)  # build() swaps in the stand-in; undo it after
    out = preview.build(str(tmp_path / "p.html"), options=(10, 20))
    page = Path(out).read_text(encoding="utf-8")
    assert "PLACEHOLDER RUNS" in page and "every number is fake" in page
    data = json.loads(page.split("const DATA=", 1)[1].split(";const PRESETS=", 1)[0])
    assert sorted(map(int, data)) == [10, 20] and all(v["placeholder"] for v in data.values())


def test_the_main_screen_runs_the_engines_pick_and_cards_are_read_only() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    assert "function loadPickChip" in html and "Run this plan live" in html   # UI-7: What helps runs the plan live too
    assert "Run the engine's pick" in html and "Use this response" not in html and "lab('Respond:')" not in html


def test_a_trigger_plan_posts_to_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ui.server import app

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    client = TestClient(app)
    r = client.post("/api/preset", json={"name": "lost-bonds", "plan": [
        {"name": "withdrawal-limit", "minute": ["requested", 0.02]}, {"name": "guarantee", "minute": ["requested", 0.08]}]})
    assert r.status_code == 200 and r.json()["ok"], r.text
