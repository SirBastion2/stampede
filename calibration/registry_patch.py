"""Patch individual rows of params/registry.csv without rewriting the rest of the file.

``csv.DictWriter`` rewrites every row, changing line endings and quoting across all
~890 rows, so a two-value change shows up as a whole-file diff. This module keeps
every byte outside the changed rows: it splits the file into logical CSV records
(a quoted field may span physical lines), rewrites only the records whose
``Parameter`` matches, and appends new records using the file's own line ending.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = ["patch_registry_rows", "apply_update_file", "RegistryPatchError"]


class RegistryPatchError(ValueError):
    """The registry can't be patched as asked (missing header column, unknown row)."""


def _line_ending(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _split_records(text: str) -> list[str]:
    """Split into logical CSV records, each keeping its own trailing line ending."""
    records: list[str] = []
    current: list[str] = []
    in_quotes = False
    for line in text.splitlines(keepends=True):
        current.append(line)
        # A doubled quote "" inside a quoted field toggles twice, so parity tracks state.
        in_quotes ^= line.count('"') % 2 == 1
        if not in_quotes:
            records.append("".join(current))
            current = []
    if current:
        records.append("".join(current))
    return records


def _parse(record: str) -> list[str]:
    rows = list(csv.reader(io.StringIO(record)))
    return rows[0] if rows else []


def _format(cells: Sequence[str], ending: str) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator=ending, quoting=csv.QUOTE_MINIMAL).writerow(list(cells))
    return buffer.getvalue()


def patch_registry_rows(
    path: Path,
    updates: Mapping[str, Mapping[str, str]],
    *,
    append_missing: Mapping[str, Mapping[str, str]] | None = None,
    skip_missing: bool = False,
) -> dict[str, str]:
    """Set fields on the rows named in ``updates``; append rows in ``append_missing`` if absent.

    ``updates`` maps a Parameter name to ``{column: new value}``. Every named row
    must exist unless it's also in ``append_missing``, which gives the full row to
    append (missing columns are left empty), or ``skip_missing`` is set, which
    leaves it out (status ``"absent"``). Returns
    ``{name: "updated"|"appended"|"unchanged"|"absent"}``.
    Untouched records are written back byte for byte.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    ending = _line_ending(text)
    records = _split_records(text)
    if not records:
        raise RegistryPatchError("registry is empty")
    header = _parse(records[0])
    if "Parameter" not in header:
        raise RegistryPatchError("registry header has no Parameter column")
    columns = {name: index for index, name in enumerate(header)}
    for changes in list(updates.values()) + list((append_missing or {}).values()):
        unknown = [column for column in changes if column not in columns]
        if unknown:
            raise RegistryPatchError(f"unknown registry columns: {unknown}")

    status: dict[str, str] = {}
    out = [records[0]]
    for record in records[1:]:
        cells = _parse(record)
        name = cells[0].strip() if cells else ""
        if name in updates and name not in status:
            new_cells = list(cells) + [""] * max(0, len(header) - len(cells))
            for column, value in updates[name].items():
                new_cells[columns[column]] = str(value)
            if new_cells == cells:
                status[name] = "unchanged"
                out.append(record)
                continue
            trailing = record[len(record.rstrip("\r\n")):] or ending
            out.append(_format(new_cells, trailing))
            status[name] = "updated"
        else:
            out.append(record)

    missing = [name for name in updates if name not in status]
    for name in list(missing):
        if append_missing and name in append_missing:
            continue
        if skip_missing:
            status[name] = "absent"
            missing.remove(name)
            continue
        raise RegistryPatchError(f"registry has no row named {name!r}")
    if out and not out[-1].endswith(("\n", "\r")):
        out[-1] = out[-1] + ending
    for name in missing:
        row = [""] * len(header)
        row[columns["Parameter"]] = name
        for column, value in {**append_missing[name], **updates.get(name, {})}.items():
            row[columns[column]] = str(value)
        out.append(_format(row, ending))
        status[name] = "appended"

    new_text = "".join(out)
    if new_text != text:
        path.write_bytes(new_text.encode("utf-8"))
    return status


def apply_update_file(spec_path: Path, registry: Path | None = None) -> dict[str, str]:
    """Apply a JSON spec ``{"updates": {name: {col: value}}, "append": {name: {col: value}}}``."""
    import json

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    target = registry or Path(__file__).resolve().parents[1] / "params" / "registry.csv"
    return patch_registry_rows(
        target,
        spec.get("updates", {}),
        append_missing=spec.get("append") or None,
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m calibration.registry_patch apply calibration/registry_updates/<file>.json``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Apply row-only registry updates from a JSON spec.")
    sub = parser.add_subparsers(dest="command", required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("spec", type=Path)
    apply.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(apply_update_file(args.spec, args.registry), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
