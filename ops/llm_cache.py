"""SQLite LLM response cache for offline demo mode (M33)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CacheEntry:
    prompt_hash: str
    model: str
    response: str


class LlmCache:
    """Keyed by prompt hash and model (§7.1)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                PRIMARY KEY (prompt_hash, model)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, prompt_hash: str, model: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE prompt_hash = ? AND model = ?",
            (prompt_hash, model),
        ).fetchone()
        return row[0] if row else None

    def put(self, prompt_hash: str, model: str, response: str) -> None:
        self._conn.execute(
            """
            INSERT INTO llm_cache (prompt_hash, model, response)
            VALUES (?, ?, ?)
            ON CONFLICT(prompt_hash, model) DO UPDATE SET response = excluded.response
            """,
            (prompt_hash, model, response),
        )
        self._conn.commit()

    def has_entries(self) -> bool:
        row = self._conn.execute("SELECT 1 FROM llm_cache LIMIT 1").fetchone()
        return row is not None

    def seed_demo_entries(self, entries: tuple[CacheEntry, ...]) -> None:
        for entry in entries:
            self.put(entry.prompt_hash, entry.model, entry.response)
