"""Build M30 ``MarketScreenFrame`` snapshots from the fake engine (no GPU)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, cast

from contracts.schemas import (
    ChartMarker,
    CrisisStatementRow,
    DepthLevel,
    FutureFanBar,
    MarketScreenFrame,
)
from mocks.l7_market_ui import MarketScreenFake
from params.loader import RegistryError, get_parameter

if TYPE_CHECKING:
    from engine.registry import Engine


def stream_hz() -> float:
    """UI WebSocket stream rate from ``params/registry.csv`` (§20.3)."""
    try:
        row = get_parameter("UI WebSocket stream rate")
    except RegistryError:
        return 10.0
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", row.value)
    return float(match.group(1)) if match else 10.0


def build_market_frame(engine: Engine, t: float | None = None) -> MarketScreenFrame:
    """Compose the latest M30 frame from fake modules on ``engine``."""
    screen = engine.get("M30")
    if not isinstance(screen, MarketScreenFake):
        raise TypeError("engine has no MarketScreenFake at M30")
    tick = float(t if t is not None else max((p[0] for p in screen.prices), default=0.0))
    bank_id = screen.bank_id if screen.scenario is not None else None
    depth_levels = [DepthLevel(price=px, size=sz) for px, sz in screen.depth]
    markers = [ChartMarker(**marker) for marker in screen.markers]
    statements = _crisis_rows(engine, screen)
    futures = [
        FutureFanBar(
            future_id=str(row["future_id"]),
            failure_probability=float(row["failure_probability"]),
            price_low=_optional_float(row.get("price_low")),
            price_high=_optional_float(row.get("price_high")),
        )
        for row in screen.futures
    ]
    return MarketScreenFrame(
        t=tick,
        bank_id=bank_id,
        prices=list(screen.prices),
        volumes=list(screen.volumes),
        depth=depth_levels,
        halt=screen.halt,
        markers=markers,
        statements=statements,
        futures=futures,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _crisis_rows(engine: Engine, screen: MarketScreenFake) -> list[CrisisStatementRow]:
    verdict_map: dict[str, dict[str, str]] = {}
    crisis = engine.get("M24")
    if crisis is not None and hasattr(crisis, "verdicts"):
        for row in crisis.verdicts:
            verdict_map[row["stmt_id"]] = row
    rows: list[CrisisStatementRow] = []
    seen: set[str] = set()
    for stmt in screen.statements:
        stmt_id = str(stmt["stmt_id"])
        seen.add(stmt_id)
        verdict_row = verdict_map.get(stmt_id)
        verdict = cast(
            Literal["accepted", "rejected", "pending"],
            verdict_row["verdict"] if verdict_row else "accepted",
        )
        reason = verdict_row["reason"] if verdict_row else ""
        rows.append(
            CrisisStatementRow(
                stmt_id=stmt_id,
                channel=str(stmt["channel"]),
                verdict=verdict,
                reason=reason,
            )
        )
    for stmt_id, verdict_row in verdict_map.items():
        if stmt_id in seen:
            continue
        rows.append(
            CrisisStatementRow(
                stmt_id=stmt_id,
                channel="",
                verdict=cast(
                    Literal["accepted", "rejected", "pending"], verdict_row["verdict"]
                ),
                reason=verdict_row["reason"],
            )
        )
    return rows
