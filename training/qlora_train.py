"""4-bit QLoRA adapter training for Llama 3.1 8B Instruct on one RTX 3060 (12 GB), fully offline.

Weights, caches, and the venv live under $STAMPEDE_TRAIN_ROOT (default ./training_runs):
    $STAMPEDE_TRAIN_ROOT/hf        HF_HOME (no network: HF_HUB_OFFLINE=1, local_files_only=True)
    $STAMPEDE_TRAIN_ROOT/models    base model dir (safetensors, HF format) and adapter outputs
    $STAMPEDE_TRAIN_ROOT/venvs     the training venv (see training/requirements-qlora.txt)

Data is a local JSONL file, one example per line, either
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}   (last turn = assistant)
or  {"prompt": "...", "completion": "..."}
Loss is on the final assistant turn only; prompt tokens are masked.

The script refuses to start if another process holds GPU memory (for example a calibration fit),
unless --force. Smoke run (a few steps on a built-in 8-example set, then saves an adapter):

    $STAMPEDE_TRAIN_ROOT/venvs/qlora/bin/python training/qlora_train.py --smoke

The served model is a GGUF in llama.cpp (scripts/serve_local_llm.sh). Using the adapter there needs a
conversion step (llama.cpp's convert_lora_to_gguf.py, then the server's --lora flag); not done here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("STAMPEDE_TRAIN_ROOT", "training_runs"))
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
DEFAULT_MODEL = ROOT / "hf" / MODEL_NAME  # default location of the HF-format base weights


def default_model_path(root: Path | None = None) -> Path:
    """First existing base-model dir: $STAMPEDE_QLORA_MODEL, <root>/hf/<model>, <root>/models/<model>.

    Returns the hf/ path when none exists, so the error message names the expected place.
    """
    override = os.environ.get("STAMPEDE_QLORA_MODEL")
    if override:
        return Path(override)
    base = root or ROOT
    candidates = (base / "hf" / MODEL_NAME, base / "models" / MODEL_NAME)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]
DEFAULT_OUT = ROOT / "models" / "adapters"
GPU_BUSY_MIB = 500  # another process using more than this on the card counts as "busy"

_SMOKE_EXAMPLES = [
    {"messages": [{"role": "user", "content": "Summarize: deposits fell 4% in an hour."},
                  {"role": "assistant", "content": "Withdrawals accelerated sharply within the hour."}]},
    {"messages": [{"role": "user", "content": "Is 'It's raining in Princeton' about a bank?"},
                  {"role": "assistant", "content": "No. It is not market-related."}]},
    {"prompt": "Classify: 'Fictional Harbor Bank lost billions on bonds'", "completion": "bank_rumor"},
    {"prompt": "Classify: 'The Fed just hiked rates'", "completion": "macro_event"},
    {"messages": [{"role": "system", "content": "You are a cautious bank communications officer."},
                  {"role": "user", "content": "Draft one sentence on liquidity."},
                  {"role": "assistant", "content": "Our liquidity position is strong and fully reported."}]},
    {"prompt": "Classify: 'ajjfalk'", "completion": "irrelevant"},
    {"messages": [{"role": "user", "content": "One word for a stock down 60%?"},
                  {"role": "assistant", "content": "Collapse."}]},
    {"prompt": "Classify: 'A fictional token is crashing'", "completion": "market_event"},
]


def _set_offline_env() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def gpu_busy_processes(min_mib: int = GPU_BUSY_MIB) -> list[str]:
    """Other compute processes holding more than ``min_mib`` of GPU memory, per nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout
    busy = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit() and int(parts[2]) > min_mib and int(parts[0]) != os.getpid():
            busy.append(f"pid {parts[0]} {parts[1]} {parts[2]} MiB")
    return busy


def _load_examples(path: Path | None, smoke: bool) -> list[dict]:
    if path is None:
        if not smoke:
            raise SystemExit("--data is required unless --smoke")
        return list(_SMOKE_EXAMPLES)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row and not {"prompt", "completion"} <= row.keys():
                raise SystemExit(f"{path}:{number}: needs 'messages' or 'prompt'+'completion'")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} has no examples")
    return rows


def _as_messages(row: dict) -> list[dict]:
    if "messages" in row:
        messages = row["messages"]
    else:
        messages = [{"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]}]
    if not messages or messages[-1]["role"] != "assistant":
        raise SystemExit("each example must end with an assistant turn")
    return messages


def _token_ids(out) -> list[int]:
    """transformers 5 returns a BatchEncoding from apply_chat_template(tokenize=True)."""
    if hasattr(out, "keys") and "input_ids" in out.keys():
        out = out["input_ids"]
    return list(out)


def encode(tokenizer, row: dict, max_len: int) -> dict:
    """Token ids with labels on the final assistant turn only (-100 elsewhere)."""
    messages = _as_messages(row)
    prompt_ids = _token_ids(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = _token_ids(tokenizer.apply_chat_template(messages, tokenize=True))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SystemExit("chat template prefix mismatch; can't mask the prompt reliably")
    full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels[: len(full_ids)]}


def collate(batch: list[dict], pad_id: int):
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        n = len(item["input_ids"])
        ids[row, :n] = torch.tensor(item["input_ids"])
        labels[row, :n] = torch.tensor(item["labels"])
        mask[row, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=None,
                        help="base model dir (default: hf/ or models/ under $STAMPEDE_TRAIN_ROOT)")
    parser.add_argument("--data", type=Path, default=None, help="local JSONL training file")
    parser.add_argument("--out", type=Path, default=None, help="adapter output dir (default: models/adapters/<run>)")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument("--smoke", action="store_true", help="5 steps at 256 tokens on the built-in set")
    parser.add_argument("--force", action="store_true", help="start even if the GPU looks busy")
    args = parser.parse_args(argv)

    if args.model is None:
        args.model = default_model_path()
    _set_offline_env()
    busy = gpu_busy_processes()
    if busy and not args.force:
        print("GPU is busy (a fit may be running); refusing to start:\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2
    if not args.model.is_dir():
        print(f"base model not found at {args.model} (HF-format directory with safetensors)", file=sys.stderr)
        return 2
    if args.smoke:
        args.max_steps, args.max_len, args.grad_accum = 5, 256, 1

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        print("CUDA is not available; QLoRA training is GPU-only", file=sys.stderr)
        return 2
    set_seed(args.seed)
    random.seed(args.seed)
    run = time.strftime("qlora-%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "")
    out = args.out or DEFAULT_OUT / run
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [encode(tokenizer, row, args.max_len) for row in _load_examples(args.data, args.smoke)]

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(round(0.03 * args.max_steps))),  # warmup_ratio was removed in transformers 5
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke else 10,
        save_strategy="no" if args.smoke else "epoch",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    result = trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    meta = {
        "run": run, "base_model": str(args.model), "data": str(args.data) if args.data else "built-in smoke set",
        "examples": len(examples), "steps": int(result.global_step), "train_loss": float(result.training_loss),
        "rank": args.rank, "alpha": args.alpha, "max_len": args.max_len, "seed": args.seed,
        "peak_gpu_mib": int(torch.cuda.max_memory_allocated() // (1024 * 1024)),
    }
    (out / "stampede_adapter.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if not (out / "adapter_model.safetensors").is_file() and not (out / "adapter_model.bin").is_file():
        print("adapter weights were not written", file=sys.stderr)
        return 1
    print(f"adapter saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
