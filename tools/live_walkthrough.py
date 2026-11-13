"""STAMPEDE live walkthrough: the three beats in a real browser against the running server.

    STAMPEDE_MODE=real python -m ui.server          # in another terminal, from the repo root
    python tools/live_walkthrough.py --url http://localhost:8000 --out walkthrough

Needs Playwright: ``pip install playwright && python -m playwright install chromium``.

It never starts the search. Requests to ``/api/pipeline/run`` are blocked in the browser, the script never clicks
"Run the search", and the report counts any attempt. A live run at 8x can take a while: ``--run-timeout`` caps
each of the two runs (the bond rumor, then the engine's pick). ``--skip-runs`` checks everything else in minutes.
The canvas text (beat 2's timeline labels) can't be read from the page: look at ``beat2_end.png``.

It resets the engine first (a run left going, e.g. by an earlier ``--skip-runs`` pass, hides the rumor chips)
and again at the end, so the server is left idle. The report is written even if a step raises.

Writes ``<out>/report.json`` and the screenshots; exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

NEUTRAL = "Can Copperleaf survive a bank run? Compare its responses across many versions of the day."
ANSWER = "Best tested plan:"
COUNT = re.compile(r"(\d+) useful plans(?: in all(?:; the details list the (\d+) on the wall)?)?")
LEVER_IDS = ("pledge_collateral", "big_bank_deposits", "deposit_rate", "guarantee_deposits", "emergency_lending",
             "cap_withdrawals")   # code names that must never reach the screen (VF-3)


def get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="walkthrough")
    ap.add_argument("--run-timeout", type=int, default=1800, help="seconds per live run")
    ap.add_argument("--speed", default="8")
    ap.add_argument("--skip-runs", action="store_true")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    checks: list[dict] = []
    errors: list[str] = []      # script errors: exceptions and console errors other than failed loads
    failed_loads: list[str] = []  # requests the server answered with an error (each is also a console line)
    blocked: list[str] = []

    def check(name: str, ok: bool, detail: object = "") -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail != "" else ""), flush=True)
        return ok

    def skip(name: str, why: str) -> None:
        checks.append({"check": name, "ok": None, "detail": f"skipped: {why}"})
        print(f"skip {name}: {why}", flush=True)

    health = get_json(base + "/health")
    engine = bool(health) and health.get("mode") == "real"
    check("server answers /health in real mode", engine, {k: health.get(k) for k in ("mode", "backend")} if health else "no answer")
    search = get_json(base + "/api/search?runs=50") or {}
    total = (search.get("considered") or {}).get("useful")
    wall_n = len(search.get("useful") or [])
    check("the search file is served", bool(search.get("ok")), {"useful": total, "on the wall": wall_n})

    def wait_finished(label: str, page) -> dict | None:
        t0 = time.time(); last = -1.0; captured = False
        while time.time() - t0 < args.run_timeout:
            st = get_json(base + "/api/ui/state") or {}
            if st.get("finished"):
                return st
            tick = float(st.get("tick") or 0)
            if tick >= 300 and not captured:
                page.wait_for_timeout(300)
                text = page.inner_text('#vaultHud')
                check(label + ': mid-run context says without help', 'Without help:' in text and 'similar simulated days' in text, text)
                page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / ('pick_mid.png' if label == "engine's pick" else 'bond_mid.png')), full_page=True)
                captured = True
            if tick - last >= 120:
                print(f"     {label}: minute {tick:.0f} of {st.get('horizon')}, {time.time() - t0:.0f} s", flush=True); last = tick
            time.sleep(5)
        return None

    def reset(page) -> None:
        if engine:
            page.evaluate("fetch('/api/control', {method: 'POST', headers: {'Content-Type': 'application/json'},"
                          " body: JSON.stringify({action: 'reset'})})")
            page.wait_for_timeout(2500)

    def screen_words(page, where: str) -> None:
        text = page.inner_text("body")
        bad = [lv for lv in LEVER_IDS if lv in text]
        check(f"{where}: no lever code names on screen", not bad, bad)
        if "at the end" in where:
            check(f"{where}: the early warning is gone after the run", "Without help:" not in text)
        if "Without help:" in text:
            line = next((ln for ln in text.splitlines() if "Without help:" in ln), "")
            check(f"{where}: the early warning says it compares with no response", "similar simulated days" in line, line[:160])

    aborted = None
    with sync_playwright() as p:
      browser = p.chromium.launch(headless=not args.headed)
      try:
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.error: {m.text}")
                    if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
            page.on("response", lambda r: failed_loads.append(f"{r.status} {r.request.method} {r.url}") if r.status >= 400 else None)

            def block(route):
                blocked.append(route.request.url); route.abort()
            page.route("**/api/pipeline/run*", block)
            head = lambda: page.inner_text("#headline").strip()   # noqa: E731

            page.goto(base + "/ui"); page.wait_for_timeout(2500)
            reset(page); page.reload(); page.wait_for_timeout(3500)
            tabs = page.eval_on_selector_all(".tabs button", "bs => bs.filter(b => getComputedStyle(b).display !== 'none').map(b => b.textContent.trim())")
            check("the nav shows the three beats only", tabs == ["1A rumor hits", "2The engine tries the plans", "3What helps"], tabs)
            check("beat 1 headline is the neutral sentence", head() == NEUTRAL, head()[:90])
            page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "beat1_start.png"))

            # beat 1: a live run of the bond rumor
            if args.skip_runs or not engine:
                skip("beat 1 live run ends with the Next button", "--skip-runs" if args.skip_runs else "engine not running")
            else:
                page.wait_for_selector('#chips [data-preset="lost-bonds"]', timeout=20000)
                page.click('#chips [data-preset="lost-bonds"]')
                page.wait_for_function("window.__stampede.eng().u?.run?.preset === 'lost-bonds' && !window.__stampede.eng().u.run.responses.length", timeout=120000)
                check("bond rumor has calibrated badge", page.inner_text('#runBadge') == 'Calibrated to a real 2023 bank run', page.inner_text('#runBadge'))
                page.click(f'#speed button[data-s="{args.speed}"]'); page.wait_for_timeout(500)
                st0 = get_json(base + "/api/ui/state") or {}
                if st0.get("paused"):
                    page.click("#play")
                t0 = time.time(); st = wait_finished("bond rumor", page)
                if check("beat 1: the bond rumor runs to the end", st is not None, f"{time.time() - t0:.0f} s" if st else f"not finished in {args.run_timeout} s"):
                    page.wait_for_selector("#closed.show", timeout=30000)
                    title = page.inner_text("#closedTitle"); text = page.inner_text("#closedText")
                    (out / ('bond_state.json' if 'closed' in title.lower() else 'pick_state.json')).write_text(json.dumps(st, indent=2))
                    check("finished run uses asked-to-withdraw words", 'asked to withdraw' in text, text)
                    check("beat 1 cash card names the shortfall", page.inner_text('#cashVal') == 'Out of cash' and 'short' in page.inner_text('#cashSub') and 'paid out' in page.inner_text('#cashSub'), page.inner_text('#cashVal') + ' | ' + page.inner_text('#cashSub'))
                    check("beat 1 cash card label", page.inner_text('#cashBar .lab span') == 'Cash and bonds on hand')
                    check("beat 1: the end card says the bank closed (the 2023 replay)", "closed" in title.lower(), f"{title} | {text}")
                    nxt = page.locator('#closed [data-goto="search"]')
                    check("beat 1: the end card has 'Next: the engine tries the plans'", nxt.is_visible() and "Next" in nxt.inner_text(), nxt.inner_text())
                    check("beat 1 headline still neutral after the run", head() == NEUTRAL, head()[:90])
                    page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "beat1_end.png"), full_page=True)
                    screen_words(page, "the bond rumor at the end")
                    nxt.click(); page.wait_for_timeout(1200)

            # beat 2: the wall
            if page.evaluate("document.querySelector('.tab.active').id") != "tab-search":
                page.click('.tabs [data-tab="search"]'); page.wait_for_timeout(1200)
            check("beat 2 is showing", page.evaluate("document.querySelector('.tab.active').id") == "tab-search")
            check("beat 2 starts with the neutral question", head() == NEUTRAL, head()[:110])
            check("beat 2 hides live bank controls", not page.locator('#play').is_visible() and not page.locator('#reset').is_visible() and not page.locator('#speed').is_visible())
            check("beat 2 waits for Play", page.evaluate('window.__stampede.wall().paused && window.__stampede.wall().t===0'))
            page.click('#wallSpeed [data-wall-speed="4"]'); page.click('#wallPause')
            page.wait_for_function('window.__stampede.wall().t > 300')
            check("beat 2 question stays during playback", head() == NEUTRAL)
            page.wait_for_function('window.__stampede.wall().phase === "final"', timeout=30000)
            check("beat 2 reveals the answer on completion", head().startswith(ANSWER), head()[:110])
            page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "beat2_end.png"))
            checks.append({"check": "beat 2 timeline labels don't overlap", "ok": None, "detail": "look at beat2_end.png (canvas text)"})
            print("look beat 2 timeline labels: see beat2_end.png", flush=True)

            page.click('#wallReplay'); page.wait_for_timeout(200)
            check("beat 2 Reset hides the answer", head() == NEUTRAL and page.evaluate('window.__stampede.wall().paused && window.__stampede.wall().t===0'))

            # beat 3: what helps
            page.click('#tab-search [data-goto="why"]'); page.wait_for_timeout(1500)
            check("beat 3 is showing", page.evaluate("document.querySelector('.tab.active').id") == "tab-why")
            check("beat 3 headline gives the answer", head().startswith(ANSWER), head()[:110])
            body = page.inner_text("#whyBody"); m = COUNT.search(body)
            check("beat 3 count matches the search file", bool(m) and int(m.group(1)) == total and (m.group(2) is None or int(m.group(2)) == wall_n),
                  m.group(0) if m else "no count line")
            page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "beat3.png"), full_page=True)
            for n in ("10", "25", "50"):
                page.click(f'#tab-why .runsPick button[data-runs="{n}"]'); page.wait_for_timeout(2000)
                m = COUNT.search(page.inner_text("#whyBody"))
                check(f"selector {n}: headline answers and the count shows", head().startswith(ANSWER) and bool(m),
                      {"headline": head()[:70], "count": m.group(0) if m else None})

            # keys: M beat 1, C recap, W beat 3, P beat 2
            for key, want, tab in (("m", NEUTRAL, "tab-crowd"), ("c", NEUTRAL, "tab-recap"), ("w", ANSWER, "tab-why"), ("p", NEUTRAL, "tab-search")):
                page.keyboard.press(key); page.wait_for_timeout(500)
                active = page.evaluate("document.querySelector('.tab.active').id")
                ok = active == tab and (head() == want if want == NEUTRAL else head().startswith(want))
                check(f"key {key.upper()} opens {tab} with the right headline", ok, {"tab": active, "headline": head()[:60]})

            # the engine's pick, from beat 1
            page.keyboard.press("w"); page.wait_for_timeout(800)
            chip = page.locator("#whyBody button.primary[data-plan]")
            if not engine:
                skip("the engine's pick runs live", "engine not running")
            elif not chip.is_visible():
                check("the live-plan button is on What helps", False, "not visible")
            else:
                label = chip.inner_text(); check("the live-plan button is on What helps", True, label)
                chip.click()
                page.wait_for_function("!window.__stampede.eng().loading && window.__stampede.eng().u?.run?.kind === 'scenario' && window.__stampede.eng().u.run.responses.length > 0", timeout=120000)
                verdict = page.inner_text("#verdict")
                check("live pick has what-if badge", page.inner_text('#runBadge') == 'What-if on the same calibrated model', page.inner_text('#runBadge'))
                check("the engine's pick starts", "Engine's pick" in verdict, verdict[:120])
                page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "pick_start.png"))
                page.wait_for_timeout(4000); screen_words(page, "the engine's pick")
                if args.skip_runs:
                    skip("the engine's pick keeps the bank open", "--skip-runs")
                else:
                    page.click(f'#speed button[data-s="{args.speed}"]'); page.wait_for_timeout(500)
                    st0 = get_json(base + "/api/ui/state") or {}
                    if st0.get("paused"):
                        page.click("#play")
                    t0 = time.time(); st = wait_finished("engine's pick", page)
                    if check("the engine's pick runs to the end", st is not None, f"{time.time() - t0:.0f} s" if st else f"not finished in {args.run_timeout} s"):
                        page.wait_for_selector("#closed.show", timeout=30000)
                        title = page.inner_text("#closedTitle"); text = page.inner_text("#closedText")
                        (out / 'pick_state.json').write_text(json.dumps(st, indent=2))
                        check("finished run uses asked-to-withdraw words", 'asked to withdraw' in text, text)
                        check("the engine's pick keeps the bank open", "made it" in title.lower(), f"{title} | {text}")
                        page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "pick_end.png"), full_page=True)
                        screen_words(page, "the engine's pick at the end")
            check("no script errors", not errors, errors[:5])
            check("every request the page made succeeded", not failed_loads, failed_loads[:8])
            check("the search was never started", not blocked, blocked)
      except Exception as exc:   # a step that raises still gets a report
          aborted = f"{type(exc).__name__}: {exc}"[:600]
          check("walkthrough finished without an exception", False, aborted)
          try:
              page.evaluate("window.scrollTo(0,0)"); page.screenshot(path=str(out / "abort.png"))
          except Exception:
              pass
      finally:
          try:
              reset(page)   # leave the server idle
          except Exception:
              pass
          browser.close()

    failed = [c for c in checks if c["ok"] is False]
    report = {"url": base, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "passed": sum(c["ok"] is True for c in checks),
              "failed": len(failed), "skipped_or_look": sum(c["ok"] is None for c in checks), "checks": checks, "errors": errors,
              "failed_loads": failed_loads, "blocked_search_requests": blocked, "aborted": aborted}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{report['passed']} passed, {report['failed']} failed, {report['skipped_or_look']} to look at or skipped; report in {out / 'report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
