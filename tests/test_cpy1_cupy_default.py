"""CPY-1: CuPy is the default for every array path; STAMPEDE_BACKEND=cpu is the NumPy reference."""

from __future__ import annotations

import types

import numpy as np
import pytest

import engine.xp as xp_mod


@pytest.fixture
def fake_device(monkeypatch: pytest.MonkeyPatch):
    """Pretend CuPy imports and sees a device (nothing here runs a CuPy array op)."""
    fake = types.SimpleNamespace(__name__="cupy", __version__="0-fake",
                                 cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(
                                     getDeviceCount=lambda: 1, getDeviceProperties=lambda i: {"name": b"Fake GPU"})))
    monkeypatch.setattr(xp_mod, "_cupy_with_device", lambda: fake)
    import engine.agents.attention as attention
    import engine.agents.decide as decide
    import engine.agents.network as network
    import engine.agents.population as population

    monkeypatch.setattr(decide, "cuda_device_available", lambda: True)
    monkeypatch.setattr(attention, "cuda_device_available", lambda: True)
    monkeypatch.setattr(network, "_cupy_module", lambda: fake)
    monkeypatch.setattr(population, "_cupy_runtime", lambda: fake)
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake)
    attention._resolve_backend.cache_clear()
    yield fake
    attention._resolve_backend.cache_clear()


def test_the_default_is_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    assert xp_mod.backend_preference() == "gpu"
    monkeypatch.setenv("STAMPEDE_BACKEND", "CPU ")
    assert xp_mod.backend_preference() == "cpu"
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    assert xp_mod.backend_preference() == "gpu"


def test_every_path_follows_the_switch_on_a_gpu_machine(monkeypatch: pytest.MonkeyPatch, fake_device) -> None:
    from engine.agents.attention import resolve_backend
    from engine.agents.decide import active_backend_name
    from engine.agents.network import select_backend
    from engine.agents.population import population_balance_backend
    from ui.live.config import prefer_gpu
    from ui.live.search import gpu_search_enabled

    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    monkeypatch.delenv("STAMPEDE_SEARCH_GPU", raising=False)
    assert (active_backend_name(), select_backend()[1], resolve_backend().name, population_balance_backend()) == \
        ("gpu", "gpu", "cupy", "gpu")
    assert prefer_gpu() and gpu_search_enabled() and xp_mod.array_module()[1] == "cupy"
    report = xp_mod.backend_report()
    assert report["device"] == "Fake GPU" and report["M06 depositors"] == "gpu" and report["batched runs"] == "cupy"

    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")   # the NumPy reference, even with a device
    assert (active_backend_name(), select_backend()[1], resolve_backend().name, population_balance_backend()) == \
        ("cpu", "cpu", "numpy", "cpu")
    assert select_backend()[0] is np and not gpu_search_enabled() and xp_mod.array_module()[1] == "numpy"
    monkeypatch.setenv("STAMPEDE_SEARCH_GPU", "1")
    assert gpu_search_enabled()   # forced batching still works (on NumPy)


def test_without_a_device_everything_falls_back_unless_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.search import gpu_search_enabled

    monkeypatch.setattr(xp_mod, "_cupy_with_device", lambda: None)
    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    monkeypatch.delenv("STAMPEDE_SEARCH_GPU", raising=False)
    monkeypatch.delenv("STAMPEDE_REQUIRE_GPU", raising=False)
    assert not xp_mod.gpu_active() and not gpu_search_enabled()
    xp_mod.require_gpu()   # no requirement: falls back quietly
    monkeypatch.setenv("STAMPEDE_REQUIRE_GPU", "1")
    with pytest.raises(RuntimeError, match="sees no CUDA device"):
        xp_mod.require_gpu("the search")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    with pytest.raises(RuntimeError, match="STAMPEDE_BACKEND=cpu"):
        xp_mod.require_gpu("the search")


def test_batched_runs_refuse_numpy_when_the_gpu_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.gpu_batch import _xp

    monkeypatch.setattr(xp_mod, "_cupy_with_device", lambda: None)
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_REQUIRE_GPU", raising=False)
    assert _xp("auto")[1] == "cpu"   # the reference: NumPy
    monkeypatch.setenv("STAMPEDE_REQUIRE_GPU", "1")
    with pytest.raises(RuntimeError, match="batched runs requires the GPU"):
        _xp("auto")


def test_the_scripts_and_the_server_say_what_they_use() -> None:
    from pathlib import Path

    for script in ("scripts/search_responses.py", "scripts/batch_table.py"):
        text = Path(script).read_text(encoding="utf-8")
        assert "--cpu" in text and "--require-gpu" in text and "format_report()" in text, script
    assert '"backends": _backend_report()' in Path("ui/server.py").read_text(encoding="utf-8")
    assert 'STAMPEDE_BACKEND:-gpu' in Path("scripts/run_live.sh").read_text(encoding="utf-8")
    assert 'setdefault("STAMPEDE_BACKEND", "cpu")' not in Path("ui/live/search.py").read_text(encoding="utf-8")


def test_six_workers_share_the_one_gpu(monkeypatch: pytest.MonkeyPatch, fake_device) -> None:
    monkeypatch.delenv("STAMPEDE_WORKERS", raising=False)
    monkeypatch.delenv("STAMPEDE_BACKEND", raising=False)
    monkeypatch.setattr(xp_mod.os, "cpu_count", lambda: 16)
    assert xp_mod.default_workers() == xp_mod.GPU_WORKERS == 6   # Sebastian: 6 are fine on the 3060
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    assert xp_mod.default_workers() == 6
    monkeypatch.setenv("STAMPEDE_WORKERS", "3")
    assert xp_mod.default_workers() == 3
    for script in ("scripts/search_responses.py", "scripts/batch_table.py"):
        assert "default_workers()" in __import__("pathlib").Path(script).read_text(encoding="utf-8")


def test_vram_is_none_on_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xp_mod, "_cupy_with_device", lambda: None)
    assert xp_mod.vram_mb() is None
