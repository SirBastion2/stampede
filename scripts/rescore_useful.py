"""UF-1: rescore the response search's files for usefulness to the bank, in place (no new simulations).

    PYTHONPATH=. python scripts/rescore_useful.py            # data/live_search.json and data/live_search_{N}.json

Each original is kept next to it as ``*.raw.json`` (and reused, so running this twice changes nothing). Run it
again after any new search run. The server, the wall and the Why page read the rescored files as before.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def rescore_file(path: str) -> dict:
    from ui.live.useful import rescore

    raw = path[:-5] + ".raw.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("mode") == "useful":
        if not os.path.exists(raw):
            raise SystemExit(f"{path} is already rescored and its original ({raw}) is missing")
        with open(raw, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        with open(raw, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    out = rescore(data)
    out["series"] = data.get("series", {})   # /api/search/run still replays one run's curve
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def summary_line(path: str, out: dict) -> str:
    """VF-1: the count is every useful plan (``considered.useful``); ``useful`` is the wall's share of them."""
    total = (out.get("considered") or {}).get("useful", len(out["useful"]))
    shown = f" ({len(out['useful'])} on the wall)" if total != len(out["useful"]) else ""
    return f"{path}: {total} useful plans{shown}; #1 {out['plans'][0]['label']}; {out['action_window']['text']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)
    from ui.live.search import search_file

    base = search_file()
    stem = base[:-5]
    paths = args.paths or sorted({base, *[p for p in glob.glob(stem + "_*.json") if not p.endswith(".raw.json")]})
    for path in paths:
        if not os.path.exists(path):
            continue
        out = rescore_file(path)
        print(summary_line(path, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
