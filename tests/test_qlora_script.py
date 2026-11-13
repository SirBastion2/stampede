"""CPU-only checks for training/qlora_train.py: no weights, no GPU, no torch needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "qlora_train", Path(__file__).resolve().parents[1] / "training" / "qlora_train.py"
)
qlora = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qlora)


class _Tok:
    """Stand-in chat template: one token per word, a marker token before the assistant turn."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = []
        for message in messages:
            ids += [len(word) for word in message["content"].split()] + [999]
        if add_generation_prompt:
            ids.append(777)
        elif messages and messages[-1]["role"] == "assistant":
            # the assistant turn follows the generation marker
            content = [len(word) for word in messages[-1]["content"].split()] + [999]
            ids = ids[: -len(content)] + [777] + content
        return ids


def test_encode_masks_everything_but_the_final_assistant_turn():
    row = {"prompt": "one two three", "completion": "four five"}
    out = qlora.encode(_Tok(), row, max_len=64)
    n_prompt = 3 + 1 + 1  # three words, end marker, generation marker
    assert out["labels"][:n_prompt] == [-100] * n_prompt
    assert out["labels"][n_prompt:] == out["input_ids"][n_prompt:]


def test_smoke_set_is_valid_and_data_is_required_otherwise(tmp_path):
    assert len(qlora._load_examples(None, smoke=True)) == 8
    with pytest.raises(SystemExit):
        qlora._load_examples(None, smoke=False)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "no"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        qlora._load_examples(bad, smoke=False)


def test_offline_env_is_forced(monkeypatch):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    qlora._set_offline_env()
    import os

    assert all(os.environ[n] == "1" for n in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"))


# --- SVB-30: default model path, and the SQLite exporter -------------------------------------


def test_default_model_path_prefers_hf_then_models(tmp_path, monkeypatch):
    monkeypatch.delenv("STAMPEDE_QLORA_MODEL", raising=False)
    name = qlora.MODEL_NAME
    assert qlora.default_model_path(tmp_path) == tmp_path / "hf" / name          # nothing exists yet
    (tmp_path / "models" / name).mkdir(parents=True)
    assert qlora.default_model_path(tmp_path) == tmp_path / "models" / name
    (tmp_path / "hf" / name).mkdir(parents=True)
    assert qlora.default_model_path(tmp_path) == tmp_path / "hf" / name
    monkeypatch.setenv("STAMPEDE_QLORA_MODEL", str(tmp_path / "x"))
    assert qlora.default_model_path(tmp_path) == tmp_path / "x"


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "export_jsonl", Path(__file__).resolve().parents[1] / "training" / "export_jsonl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exporter_writes_pairs_dedupes_and_refuses_hashed_prompts(tmp_path):
    import json
    import sqlite3

    exp = _exporter()
    db = tmp_path / "c.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE cache (prompt TEXT, response TEXT, key TEXT)")
        rows = [(f"prompt {i}", f"answer {i}", f"{i:064x}") for i in range(40)] + [("prompt 1", "answer 1", "dup")]
        con.executemany("INSERT INTO cache VALUES (?, ?, ?)", rows)
    out = tmp_path / "train.jsonl"
    train, evals = exp.export(db, "cache", "prompt", "response", out)
    assert train + evals == 40                                    # the duplicate pair is dropped
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]
    with pytest.raises(SystemExit):
        exp.export(db, "cache", "key", "response", tmp_path / "bad.jsonl")
