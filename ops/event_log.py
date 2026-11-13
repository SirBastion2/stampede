"""Event log hashing and Parquet persistence (M33)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


def events_to_rows(events: list[BaseModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        payload = event.model_dump(mode="json", by_alias=True)
        rows.append(
            {
                "event_type": type(event).__name__,
                "t": payload.get("t"),
                "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            }
        )
    return rows


def event_log_hash(events: list[BaseModel]) -> str:
    """SHA-256 of the canonical JSON event list (matches ObservabilityFake)."""
    rows: list[dict[str, object]] = []
    for event in events:
        payload = event.model_dump(mode="json", by_alias=True)
        rows.append(payload)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_event_log_parquet(events: list[BaseModel], path: Path) -> None:
    """Write events to Parquet; requires pyarrow (listed in §18.0 stack)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Parquet event logs") from exc
    rows = events_to_rows(events)
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_event_log_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Parquet event logs") from exc
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows
