"""Local LLM settings, fake backend, GPU gate, and M24 fallback (ADR-055)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import local_llm as local_mod
from agents.crisis_room import build_claims_ledger, filter_statement
from agents.llm_cache import LlmCache, complete_json_or_fallback
from agents.local_llm import (
    FakeLocalBackend,
    install_fake_local_backend,
    local_llm_gguf_path,
    local_llm_model,
    local_llm_weights_dir,
    nvidia_gpu_visible,
    require_gpu_for_inference,
    stampede_llm_mode,
)
from contracts.schemas import Scenario, Statement
from params.loader import get_parameter


def _reset() -> None:
    install_fake_local_backend(None)
    local_mod.local_llm_validation_failures = 0


@pytest.fixture(autouse=True)
def _clean_backend() -> None:
    _reset()
    yield
    _reset()


def test_default_mode_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STAMPEDE_LLM", raising=False)
    assert stampede_llm_mode() == "offline"


def test_model_swap_via_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_MODEL", "Qwen2.5-7B-Instruct-Q4_K_M")
    assert local_llm_model() == "Qwen2.5-7B-Instruct-Q4_K_M"


def test_weights_dir_reads_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_WEIGHTS", str(tmp_path))
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_MODEL", "TinyTest-Q4")
    assert local_llm_weights_dir() == str(tmp_path)
    assert local_llm_gguf_path() == str(tmp_path / "TinyTest-Q4.gguf")


def test_test_code_does_not_hardcode_host_weights() -> None:
    roots = (
        Path("agents/local_llm.py"),
        Path("agents/llm_cache.py"),
        Path("agents/crisis_room.py"),
        Path("tests/test_local_llm_settings.py"),
        Path("tests/test_local_llm_backend.py"),
    )
    for path in roots:
        assert "mnt" + "/S" not in path.read_text(encoding="utf-8")


def test_registry_weights_row_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    row = get_parameter("Local LLM weights directory (ASSUMED)")
    assert row.value
    monkeypatch.setenv("STAMPEDE_LOCAL_LLM_WEIGHTS", "/tmp/stampede-weights-test")
    assert local_llm_weights_dir() == "/tmp/stampede-weights-test"


def test_fake_backend_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_LLM", "local")
    fake = FakeLocalBackend(json_handler=lambda _p: {"ok": True})
    install_fake_local_backend(fake)
    from agents.llm_cache import complete_json_llm

    assert complete_json_llm("prompt", "grok-4.1-fast") == {"ok": True}
    assert fake.json_calls == 1


def test_complete_json_or_fallback_uses_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_LLM", "local")
    install_fake_local_backend(FakeLocalBackend(json_handler=lambda _p: {"not": "schema"}))

    def validate(raw: dict[str, object]) -> str:
        if "verdict" not in raw:
            raise ValueError("missing verdict")
        return str(raw["verdict"])

    result = complete_json_or_fallback(
        "p",
        "m",
        validate=validate,
        fallback=lambda: "rule",
    )
    assert result == "rule"
    assert local_mod.local_llm_validation_failures == 1


def test_gpu_probe_is_injectable() -> None:
    assert nvidia_gpu_visible(probe=lambda: False) is False
    assert nvidia_gpu_visible(probe=lambda: True) is True


def test_require_gpu_skips_when_fake_installed() -> None:
    install_fake_local_backend(FakeLocalBackend(json_handler=lambda _p: {}))
    require_gpu_for_inference(probe=lambda: False)


def test_require_gpu_raises_without_device() -> None:
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        require_gpu_for_inference(probe=lambda: False)


def _scenario() -> Scenario:
    return Scenario(preset="tech-concentrated", seed=3, rate_path=[(0.0, 0.05)])


def test_m24_local_invalid_falls_back_to_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STAMPEDE_LLM", "local")
    install_fake_local_backend(FakeLocalBackend(json_handler=lambda _p: {"not": "schema"}))
    scenario = _scenario()
    ledger = build_claims_ledger(scenario, [], 0.0)
    stmt = Statement(
        id="vague-ok",
        text="We remain focused on serving clients.",
        claims=['{"type":"vague","text":"We remain focused on serving clients."}'],
        channel="press_release",
    )
    cache = LlmCache(tmp_path / "m24-invalid.sqlite")
    row = filter_statement(stmt, ledger, scenario, [], cache=cache)
    cache.close()
    assert row.verdict == "accepted"
    assert local_mod.local_llm_validation_failures == 1


def test_m24_local_valid_uses_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STAMPEDE_LLM", "local")
    install_fake_local_backend(
        FakeLocalBackend(
            json_handler=lambda _p: {
                "verdict": "rejected",
                "reason": "vague wording rejected",
            }
        )
    )
    scenario = _scenario()
    ledger = build_claims_ledger(scenario, [], 0.0)
    stmt = Statement(
        id="vague-live",
        text="We remain focused on serving clients.",
        claims=['{"type":"vague","text":"We remain focused on serving clients."}'],
        channel="press_release",
    )
    cache = LlmCache(tmp_path / "m24-valid.sqlite")
    row = filter_statement(stmt, ledger, scenario, [], cache=cache)
    cache.close()
    assert row.verdict == "rejected"
    assert "vague" in row.reason


def test_m25_invalid_local_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.personas import score_statement

    monkeypatch.setenv("STAMPEDE_LLM", "local")
    install_fake_local_backend(FakeLocalBackend(json_handler=lambda _p: {"not": "schema"}))
    batch = score_statement(
        preset="tech-concentrated",
        seed=1,
        stmt_id="s1",
        channel="press_release",
        segments=["startup"],
        text="We are well capitalized.",
    )
    assert batch.scores
    assert local_mod.local_llm_validation_failures == 1
