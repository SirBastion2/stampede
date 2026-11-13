"""Export prompt/response pairs from a local SQLite file to QLoRA JSONL. Offline, read-only.

No schema guessing: list what's in the file first, then name the table and columns.

    python training/export_jsonl.py --db <cache.sqlite> --list
    python training/export_jsonl.py --db <cache.sqlite> --table <t> --prompt-col <p> --response-col <r> \\
        [--where "model LIKE 'Meta-Llama%'"] --out training_runs/data/train.jsonl --holdout 0.05

Writes {"messages": [user, assistant]} lines, skips empty rows, drops exact duplicate pairs, and puts a
deterministic 5% (by hash) in <out>.eval.jsonl. Rows whose prompt column holds only a hash can't be
used: the exporter refuses columns where most values look like 64-hex digests.

Before training on another provider's outputs (for example cached Grok replies), check that provider's
terms; many forbid using outputs to train other models. Local-model outputs and the repo's own rule-path
outputs don't have that problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

_HEX = re.compile(r"^[0-9a-f]{32,128}$")


def list_schema(db: Path) -> dict[str, list[str]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: [row[1] for row in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}


def _looks_hashed(values: list[str]) -> bool:
    sample = [v for v in values[:200] if v]
    return bool(sample) and sum(bool(_HEX.match(v.strip().lower())) for v in sample) > len(sample) / 2


def export(db: Path, table: str, prompt_col: str, response_col: str, out: Path,
           where: str | None = None, holdout: float = 0.05) -> tuple[int, int]:
    schema = list_schema(db)
    if table not in schema:
        raise SystemExit(f"no table {table!r}; tables: {sorted(schema)}")
    for col in (prompt_col, response_col):
        if col not in schema[table]:
            raise SystemExit(f"no column {col!r} in {table}; columns: {schema[table]}")
    query = f'SELECT "{prompt_col}", "{response_col}" FROM "{table}"' + (f" WHERE {where}" if where else "")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = [(str(p or ""), str(r or "")) for p, r in con.execute(query)]
    if _looks_hashed([p for p, _ in rows]):
        raise SystemExit(f"{prompt_col!r} holds hashes, not prompt text; this table can't train an adapter")
    seen: set[tuple[str, str]] = set()
    train = evals = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    eval_path = out.with_suffix(".eval.jsonl")
    with out.open("w", encoding="utf-8") as tr, eval_path.open("w", encoding="utf-8") as ev:
        for prompt, response in rows:
            prompt, response = prompt.strip(), response.strip()
            if not prompt or not response or (prompt, response) in seen:
                continue
            seen.add((prompt, response))
            line = json.dumps({"messages": [{"role": "user", "content": prompt},
                                            {"role": "assistant", "content": response}]}, ensure_ascii=False)
            bucket = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if bucket < holdout:
                ev.write(line + "\n"); evals += 1
            else:
                tr.write(line + "\n"); train += 1
    return train, evals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table"); ap.add_argument("--prompt-col"); ap.add_argument("--response-col")
    ap.add_argument("--where", default=None)
    train_root = Path(os.environ.get("STAMPEDE_TRAIN_ROOT", "training_runs"))
    ap.add_argument("--out", type=Path, default=train_root / "data" / "train.jsonl")
    ap.add_argument("--holdout", type=float, default=0.05)
    args = ap.parse_args(argv)
    if args.list:
        print(json.dumps(list_schema(args.db), indent=2))
        return 0
    if not (args.table and args.prompt_col and args.response_col):
        print("--table, --prompt-col and --response-col are required (run --list first)", file=sys.stderr)
        return 2
    train, evals = export(args.db, args.table, args.prompt_col, args.response_col, args.out,
                          where=args.where, holdout=args.holdout)
    print(f"wrote {train} train and {evals} eval examples to {args.out} and {args.out.with_suffix('.eval.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
