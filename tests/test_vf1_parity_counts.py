"""VF-1: GPU parity judged on an absolute share of deposits (ADR-132), and every count of useful plans is the total."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(requested: float, *, open_: bool = True, fired: dict | None = None, series_shift: float = 0.0,
         samples: int = 144, mid_bump: float = 0.0) -> dict:
    series = [[10 * (i + 1), round(requested * (i + 1) / samples + series_shift + (mid_bump if i == samples // 2 else 0.0), 5),
               1.0, 0.0, 0.0] for i in range(samples)]
    return {"requested": requested, "open": open_, "fired": dict({"guarantee": 261.0} if fired is None else fired), "series": series}


def test_the_3060s_worst_guarantee_gap_passes_and_a_real_regression_does_not() -> None:
    gp = _load("gpu_parity_vf1", "scripts/gpu_parity.py")
    assert gp.TOL == pytest.approx(1e-4)
    cpu = _run(0.08010)
    # seed 3, guarantee once 8% asked, on the 3060 (Nov 6): 0.0036 points lower, outcome and trigger the same
    ok, g = gp.compare(cpu, _run(0.08010 - 0.0000364))
    assert ok and g["rel"] > 1e-4        # failed GPU-1's 1e-4 relative rule; passes the absolute one
    assert not gp.compare(cpu, _run(0.08010 + 0.0002))[0]                    # 0.02 points: fails
    assert not gp.compare(cpu, _run(0.08010, open_=False))[0]                # another outcome: fails
    assert not gp.compare(cpu, _run(0.08010, fired={"guarantee": 263.0}))[0]  # trigger 2 minutes off: fails
    assert not gp.compare(cpu, _run(0.08010, fired={}))[0]                    # trigger missing: fails
    assert not gp.compare(cpu, _run(0.08010, mid_bump=0.0003))[0]             # ends equal, diverges mid-day: fails
    assert not gp.compare(cpu, _run(0.08010, samples=100))[0]                 # a cut-short run: fails


def test_reuse_cpu_refuses_other_runs(tmp_path: Path) -> None:
    gp = _load("gpu_parity_vf1b", "scripts/gpu_parity.py")
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"cells": [["none", 9]], "cpu": [{"minutes": 1441}], "cpu_s": 1.0}), encoding="utf-8")
    with pytest.raises(SystemExit, match="other runs"):
        gp.main(["--seeds", "1", "--reuse-cpu", str(old), "--out", str(tmp_path / "out.json"), "--xp", "numpy"])


def test_the_summary_counts_every_useful_plan_and_says_how_many_are_on_the_wall() -> None:
    mod = _load("rescore_vf1", "scripts/rescore_useful.py")
    out = {"useful": [f"p{i}" for i in range(16)], "considered": {"useful": 18}, "plans": [{"label": "Top plan"}],
           "action_window": {"text": "Window."}}
    assert mod.summary_line("data/x.json", out) == "data/x.json: 18 useful plans (16 on the wall); #1 Top plan; Window."
    out["considered"]["useful"] = 16
    assert mod.summary_line("data/x.json", out) == "data/x.json: 16 useful plans; #1 Top plan; Window."
    assert "mod.summary_line(p, mod.rescore_file(p))" in Path("scripts/pipeline.py").read_text(encoding="utf-8")


def test_rescore_keeps_the_wall_list_and_the_total_apart() -> None:
    from tests.test_au_automatic import _search
    from ui.live.useful import rescore

    out = rescore(_search(seeds=tuple(range(1, 11))), checks=False, wall_useful=1)
    assert len(out["useful"]) <= 1
    assert out["considered"]["useful"] == len(out["useful_all"]) >= len(out["useful"])


def test_the_page_shows_the_total() -> None:
    html = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    assert "function usefulTotal(D)" in html
    assert "`${usefulTotal(D)} useful plans`" in html   # UI-7 shortened the title
    assert "and ${usefulTotal(D) - 3 - k} more on What helps" in html
    assert "${(D.useful || []).length} useful plans" not in html
    # the window's two labels grow away from their own marks (the final data's window is 39 minutes wide)
    assert "xOf(A.likely_minute) - 4" in html and "xOf(A.deadline_minute) + 4" in html
    assert "xOf(A.likely_minute) - 60" not in html
