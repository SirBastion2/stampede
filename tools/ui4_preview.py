"""UI-4: a standalone preview of the search screens with PLACEHOLDER runs, to check the layout before the pool run.

    PYTHONPATH=. python tools/ui4_preview.py            # writes data/ui4_preview.html (open it in a browser)

The runs come from a made-up stand-in, not the engine; the scoring, elimination and text are the real ones
(ui/live/search.py), at each option in RUN_OPTIONS. The page is the real ui/live/stampede_ui.html with fetch()
answered from the embedded data, opened on the search screen; a red banner says the numbers are placeholders.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, os.getcwd())

BANNER = ("PREVIEW WITH PLACEHOLDER RUNS: a made-up stand-in, not the engine. The layout, scoring and text are the "
          "real ones; every number is fake.")


def _fake_run(args):
    from ui.live import search

    plan, seed, minutes = args[:3]
    preset = args[3] if len(args) > 3 else search.PRESET
    cred = args[4] if len(args) > 4 else None
    rng = random.Random(zlib.crc32(repr((seed, cred, preset)).encode()))
    c = 0.9 if cred is None else cred
    # weak rumors fizzle, middling ones stall at 5-9% and the bank survives, strong ones sink it (about 25% by 4 PM)
    final = (0.0003 if preset != search.PRESET else 0.007 if c < 0.3 else 0.05 + (c - 0.3) * 0.25 if c < 0.45
             else 0.84 * min(1.0, (c / 0.9) ** 1.2)) * (1 + 0.04 * rng.gauss(0, 1))
    mid, width = 520 * (1 + 0.06 * rng.gauss(0, 1)), 200.0
    survive_at = rng.uniform(0.11, 0.2)       # a guarantee saves the bank if it lands before this share is asked for
    line_saves = rng.random() < 0.6           # the credit line keeps the bank open on some days
    lo = 1 / (1 + math.exp(mid / width))
    base = lambda m: final * max(0.0, (1 / (1 + math.exp(-(m - mid) / width)) - lo) / (1 - lo))  # noqa: E731
    steps = sorted(plan, key=search._when_key)
    fired, fired_req, req, series, scale, frozen = {}, {}, 0.0, [], 1.0, None
    prev = 0.0
    for m in range(1, 1442):
        inc = max(0.0, base(m) - base(m - 1)) * scale
        req = frozen if frozen is not None else req + inc
        cash = max(0.0, 1 - req / 0.62) if "credit-line" not in fired else 1.0
        for name, when in steps:
            if name in fired:
                continue
            hit = (m >= when) if isinstance(when, (int, float)) else (req >= when[1] if when[0] == "requested" else cash <= when[1])
            if hit:
                fired[name], fired_req[name] = m, req
                if name == "guarantee":
                    frozen = req
                elif name == "withdrawal-limit":
                    scale = 0.3
        if m % 10 == 0 or m == 1441:
            series.append([m, round(req, 5), round(cash, 4), round(min(1, req * 1.2), 4), round(min(0.3, req * 0.5), 4)])
        prev = req
    del prev
    g = fired_req.get("guarantee")
    open_ = preset != search.PRESET or (cred is not None and final < 0.1)
    if preset == search.PRESET and not open_:
        open_ = (g is not None and g < survive_at) or ("credit-line" in fired and line_saves)
    tag = search.plan_id(plan) if preset == search.PRESET else f"baseline:{preset}"
    if cred is not None:
        tag = f"strength:{cred:g}"
    return {"plan": tag, "seed": seed, "minutes": 1441, "credibility": cred, "open": bool(open_),
            "closure_minute": None if open_ else 1440.0, "fired": fired, "fired_requested": fired_req,
            "cash_out_minute": None, "requested": req, "requested_usd": req * 175e9, "emergency_usd": 0.0,
            "series": series, "seconds": 0.0}


def build(out: str = "data/ui4_preview.html", options: tuple[int, ...] | None = None) -> str:
    from ui.live import search

    search.run_plan = _fake_run
    results = search.run_trigger_options(workers=1, options=options or search.RUN_OPTIONS)
    payload = {}
    for n, data in results.items():
        data = {k: v for k, v in data.items() if k != "series"}
        payload[n] = {"ok": True, **data, "run_options": sorted(results), "generated": "", "placeholder": True}
    from ui.live.cards import response_cards
    from ui.live.presets import preset_table, response_table

    cards = response_cards(None)   # no batch file in the preview: the cards show only the assumed parts
    presets = {"presets": preset_table(), "responses": [{**r, "card": cards.get(r["name"])} for r in response_table()]}
    page = Path("ui/live/stampede_ui.html").read_text(encoding="utf-8")
    shim = ("<script>(function(){const DATA=" + json.dumps(payload, separators=(",", ":")) + ";"
            "const PRESETS=" + json.dumps(presets, separators=(",", ":")) + ";"
            "const BIG=Math.max(...Object.keys(DATA).map(Number));"
            "const real=window.fetch;window.fetch=async function(url,opts){const u=new URL(url,location.href);"
            "if(u.pathname.endsWith('/api/search')){const n=+(u.searchParams.get('runs')||BIG);"
            "const d=DATA[n]||{ok:false,error:'no such option'};return new Response(JSON.stringify(d),{status:200});}"
            "if(u.pathname.endsWith('/api/presets'))return new Response(JSON.stringify(PRESETS),{status:200});"
            "return new Response(JSON.stringify({ok:false,error:'preview: no live server'}),{status:200});};"
            "if(new URLSearchParams(location.search).get('screen')==='main'){history.replaceState(null,'',location.pathname);}"
            "else if(!new URLSearchParams(location.search).get('screen')){const q=new URLSearchParams(location.search);"
            "q.set('screen','search');location.replace(location.pathname+'?'+q.toString());}"
            "document.addEventListener('DOMContentLoaded',()=>{const b=document.createElement('div');"
            "b.style.cssText='position:fixed;left:0;right:0;bottom:0;z-index:99999;background:#7a1f27;color:#fff;"
            "font:600 12px system-ui;padding:6px 10px;display:flex;gap:14px;align-items:center';"
            "b.innerHTML='<span>" + BANNER + "</span><a style=\"color:#ffd\" href=\"?screen=search\">Wall</a>"
            "<a style=\"color:#ffd\" href=\"?screen=why\">Why page</a><a style=\"color:#ffd\" href=\"?screen=main\">Main screen</a>';document.body.appendChild(b);});})();</script>")
    page = page.replace("<head>", "<head>" + shim, 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    Path(out).write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print("wrote", build(*sys.argv[1:2]))
