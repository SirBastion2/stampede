"""Check the saved search replay's text layout without starting any simulation.

Requires Playwright and its Chromium browser. Preview local HTML against the running
server's saved data, blocking every non-GET request. Exit 1 for collisions or errors.

    python tools/check_wall_alignment.py --url http://localhost:8000 --out /tmp/alignment
    python tools/check_wall_alignment.py --html /path/to/baseline.html --out /tmp/before
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from urllib.parse import urlparse

# Measure rendered glyphs, including the canvas transform, rather than guessing from
# character counts. Retain only one frame so animation does not create false pairs.
MEASURE_TEXT = """(() => {
  const original = CanvasRenderingContext2D.prototype.fillText;
  window.alignmentText = []; window.alignmentFrame = 0;
  const clear = CanvasRenderingContext2D.prototype.clearRect;
  CanvasRenderingContext2D.prototype.clearRect = function(...args) {
    if (this.canvas.id === 'wallCv') window.alignmentFrame++;
    return clear.apply(this, args);
  };
  CanvasRenderingContext2D.prototype.fillText = function(text, x, y) {
    const rect = this.canvas.getBoundingClientRect();
    if (this.canvas.id === 'wallCv' && rect.width && this.globalAlpha > 0.9) {
      const m = this.measureText(text), tr = this.getTransform();
      const a = tr.transformPoint(new DOMPoint(x - m.actualBoundingBoxLeft,
                                               y - m.actualBoundingBoxAscent));
      const b = tr.transformPoint(new DOMPoint(x + m.actualBoundingBoxRight,
                                               y + m.actualBoundingBoxDescent));
      const sx = rect.width / this.canvas.width, sy = rect.height / this.canvas.height;
      window.alignmentText.push({text: String(text), x: a.x * sx, y: a.y * sy,
                                 w: (b.x-a.x)*sx, h: (b.y-a.y)*sy, time: performance.now(), frame: window.alignmentFrame});
      if (window.alignmentText.length > 20000) window.alignmentText.splice(0, 10000);
    }
    return original.apply(this, arguments);
  };
})();"""


def collisions(boxes: list[dict], finale: bool = False) -> list[dict]:
    """Check day headers and sidebar counts, the strings that collided in VF-3."""
    result = []
    for a, b in itertools.combinations(boxes, 2):
        count = re.match(r"^\d+/\d+(?: · #1 in \d+)?$", a["text"]) or re.match(
            r"^\d+/\d+(?: · #1 in \d+)?$", b["text"]
        )
        day = (a["text"].startswith("Day ") or b["text"].startswith("Day ")) and (
            "no response:" in a["text"] or "no response:" in b["text"]
        )
        if not (finale or count or day):
            continue
        width = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
        height = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
        if width > 0.5 and height > 2:
            result.append({"a": a["text"], "b": b["text"], "width": width, "height": height})
    return result


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--html", type=Path, default=Path(__file__).resolve().parents[1]
                        / "ui/live/stampede_ui.html")
    parser.add_argument("--out", type=Path, default=Path("wall-alignment"))
    parser.add_argument("--widths", default="1600,1280")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases, errors, blocked = [], [], []
    url = args.url.rstrip("/")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.add_init_script(MEASURE_TEXT)
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_request(route):
            request = route.request
            if request.method != "GET":
                blocked.append({"method": request.method, "url": request.url})
                route.abort()
            elif request.url == url + "/ui":
                route.fulfill(path=str(args.html.resolve()), content_type="text/html")
            else:
                route.continue_()

        page.route("**/*", route_request)
        page.goto(url + "/ui")
        page.click('.tabs [data-tab="search"]')
        page.wait_for_function("window.__stampede.wall().data !== null")
        for width in map(int, args.widths.split(",")):
            page.set_viewport_size({"width": width, "height": 900 if width >= 1600 else 800})
            for count in (10, 25, 50):
                page.click(f'#tab-search .runsPick button[data-runs="{count}"]')
                page.wait_for_function("n => window.__stampede.wall().data !== null && "
                                       "window.__stampede.wall().runs === n", arg=count)
                for minute, phase in ((0, "play"), (720, "play"), (1441, "play"), (1441, "final")):
                    page.evaluate("""({minute, phase}) => {
                      const wall = window.__stampede.wall();
                      wall.paused = true; wall.phase = phase; wall.t = minute;
                      wall.tiles.clear(); if (wall.gt) wall.gt.clear(); window.scrollTo(0, 0);
                    }""", {"minute": minute, "phase": phase})
                    page.wait_for_timeout(500)
                    measured = page.evaluate("""() => {
                      const all = window.alignmentText;
                      const last = all.length ? all[all.length-1].frame : 0;
                      const frame = new Map();
                      for (const b of all) if (b.frame === last)
                        frame.set([b.text, Math.round(b.x), Math.round(b.y)].join('|'), b);
                      const rect = selector => {
                        const el = document.querySelector(selector);
                        if (!el || el.hidden) return null;
                        const r = el.getBoundingClientRect();
                        return {x:r.x, y:r.y, w:r.width, h:r.height};
                      };
                      return {text:[...frame.values()], canvas:rect('#wallCv'),
                        summary:rect('#wallSummary'), selector:rect('#tab-search .runsPick'),
                        description:rect('#wallDescription'), legend:rect('#wallLegend'),
                        footer:rect('.wall-footer'), next:rect('#tab-search .next'),
                        actions:rect('.wall-actions'), horizontalOverflow:
                          document.documentElement.scrollWidth > innerWidth};
                    }""")
                    defects = collisions(measured["text"], finale=phase == "final")
                    # DOM rows must reserve their own space, including after text wraps.
                    for name in ("legend", "next", "actions"):
                        r, footer = measured.get(name), measured.get("footer")
                        if r and footer and (r["y"] < footer["y"] - 0.5 or
                                             r["y"] + r["h"] > footer["y"] + footer["h"] + 0.5):
                            defects.append({"control_outside_footer": name})
                    if measured.get("description"):
                        r = measured["description"]
                        if r["y"] + r["h"] > measured["canvas"]["y"] + 0.5:
                            defects.append({"description_over_canvas": True})
                    if measured["horizontalOverflow"]:
                        defects.append({"horizontal_overflow": True})
                    stem = f"wall_{width}_{count}_{minute}_{phase}"
                    page.screenshot(path=str(args.out / (stem + ".png")), full_page=True)
                    cases.append({"case": stem, "ok": not defects, "defects": defects,
                                  "measurements": measured})
                    print(f"{'PASS' if not defects else 'FAIL'} {stem}: "
                          f"{len(defects)} defects", flush=True)
        # Navigating from a scrolled beat must not leave the next headline behind the toolbar.
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.click('#tab-search [data-goto="why"]')
        page.wait_for_timeout(200)
        navigation_ok = page.evaluate("""() => {
          const head = document.querySelector('#headline').getBoundingClientRect();
          const top = document.querySelector('.top').getBoundingClientRect();
          return scrollY === 0 && head.top >= top.bottom - 0.5;
        }""")
        cases.append({"case": "headline_after_navigation", "ok": navigation_ok})
        browser.close()
    report = {"html": str(args.html.resolve()), "server": urlparse(url).netloc,
              "passed": sum(c["ok"] for c in cases), "failed": sum(not c["ok"] for c in cases),
              "errors": errors, "blocked_mutations": blocked, "cases": cases}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['passed']} passed, {report['failed']} failed, {len(errors)} script errors")
    return int(bool(report["failed"] or errors or blocked))


if __name__ == "__main__":
    raise SystemExit(main())
