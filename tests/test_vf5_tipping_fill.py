"""VF-5: named tipping strengths that only simulate the missing (strength, version of the day) runs, and a
"Run the search" button that is operator-only and asks first."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _pipeline(name: str):
    spec = importlib.util.spec_from_file_location(name, "scripts/pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tipping_runs_only_simulate_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _pipeline("pipeline_vf5a")
    ten, fifty = tmp_path / "live_search_10.json", tmp_path / "live_search_50.json"
    have = {f"strength:{c:g}|{s}": [[10, 0.0]] for c in (0.26, 0.27) for s in range(1, 11)}
    ten.write_text(json.dumps({"seeds": list(range(1, 11)), "series": dict(have)}), encoding="utf-8")
    fifty.write_text(json.dumps({"seeds": list(range(1, 51)), "series": dict(have)}), encoding="utf-8")
    asked: list = []

    class FakeRunner:
        def __init__(self, workers, minutes):
            pass

        def _map(self, todo):
            asked.extend((c, s) for _, s, _, _, c in todo)
            return [{"seed": s, "series": [[10, c]]} for _, s, _, _, c in todo]

        def close(self):
            pass

    import ui.live.search as search
    monkeypatch.setattr(search, "Runner", FakeRunner)
    monkeypatch.setattr(pipe, "status", lambda *a, **k: None)
    pipe._tipping_runs([0.26, 0.27], 50, 1, None, [str(ten), str(fifty)])
    assert len(asked) == 80 and all(s > 10 for _, s in asked)          # 2 strengths x versions 11-50 only
    big = json.loads(fifty.read_text())["series"]
    small = json.loads(ten.read_text())["series"]
    assert all(f"strength:{c:g}|{s}" in big for c in (0.26, 0.27) for s in range(1, 51))
    assert not any(k.endswith(("|11", "|50")) for k in small)            # the 10-version file gets no seed past 10
    asked.clear()
    pipe._tipping_runs([0.26, 0.27], 50, 1, None, [str(ten), str(fifty)])
    assert asked == []                                                    # nothing left to run


def test_named_strengths_replace_the_suggestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_au_automatic import _search

    monkeypatch.setenv("STAMPEDE_SEARCH_FILE", str(tmp_path / "live_search.json"))
    pipe = _pipeline("pipeline_vf5b")
    data = _search(seeds=tuple(range(1, 11)))
    for name in ("live_search.json", "live_search_10.json"):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    import ui.live.search as search
    monkeypatch.setattr(search, "run_trigger_options", lambda **k: pytest.fail("resimulated with --reuse"))
    got: list = []
    monkeypatch.setattr(pipe, "_tipping_runs", lambda strengths, seeds, *a, **k: got.append((list(strengths), seeds)))
    assert pipe.main(["--runs", "10", "--reuse", "--tip-strengths", "0.26,0.27,0.28,0.29", "--tip-seeds", "50",
                      "--messaging-plans", "0"]) == 0
    assert got == [([0.26, 0.27, 0.28, 0.29], 50)]


def test_run_the_search_is_operator_only_and_asks_first() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    assert 'class="btn opx" id="wallRun"' in html
    assert "if (!confirm('Run the whole search again?" in html
    assert ".presenter .opx{display:none}" in html


def test_the_early_warning_is_gone_once_the_day_is_over() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    assert "!(minute > 0) || sim.over) return '';" in html
