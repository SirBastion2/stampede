"""Render the presenter view's three beats against saved results, with all writes blocked.

Checks data-derived wording, details access, absent-result states, and live-plan
request wiring (intercepted, never sent). Requires Playwright and Chromium.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://localhost:8000')
    ap.add_argument('--html', type=Path, default=Path('ui/live/stampede_ui.html'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'report.json').unlink(missing_ok=True)
    base = args.url.rstrip('/')
    cases, errors, blocked, planned = [], [], [], []
    held = []; hold_old_state = False; fake_plan_running = False; new_state = None
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width':1280,'height':900})
        page.on('pageerror', lambda e: errors.append(str(e)))

        def guard(route):
            nonlocal fake_plan_running
            r = route.request
            if r.method != 'GET':
                # Verify this button's real payload without starting the engine.
                if r.url in (base+'/api/control', base+'/api/preset'):
                    planned.append({'url':r.url, 'body':r.post_data_json})
                    if r.url == base+'/api/preset': fake_plan_running = True
                    return route.fulfill(json={'ok':True})
                blocked.append(r.url)
                return route.abort()
            if hold_old_state and '/api/ui/state?' in r.url and not held:
                held.append(route)
                return
            if fake_plan_running and '/api/ui/state?' in r.url:
                return route.fulfill(json=new_state)
            if r.url == base+'/ui':
                return route.fulfill(path=str(args.html.resolve()), content_type='text/html')
            route.continue_()

        page.route('**/*', guard)
        page.goto(base+'/ui')
        page.wait_for_function('window.__stampede.wall().data !== null')
        page.wait_for_timeout(300)
        assert 'Compare its responses' in page.locator('#headline').inner_text()
        assert not page.locator('#responseTools').get_attribute('open')
        assert not page.locator('#wallRun').is_visible()
        assert page.locator('#perf').inner_text().endswith('simulated customers')
        for theme in ['light', 'dark']:
            page.emulate_media(color_scheme=theme)
            for width in [1280, 1600]:
                page.set_viewport_size({'width':width, 'height':900})
                for n in [10,25,50]:
                    page.locator('.tabs [data-tab="why"]').click()
                    page.locator(f'#tab-why .runsPick button[data-runs="{n}"]').click()
                    page.wait_for_function('n => window.__stampede.wall().runs===n && window.__stampede.wall().data', arg=n)
                    page.wait_for_timeout(250)
                    d = page.evaluate('''() => {const d=window.__stampede.wall().data;const p=d.byPlan[d.pick];
                      return {n:p.open_by_seed.length, open:p.open_by_seed.filter(Boolean).length,
                        margin:(p.closest*100).toFixed(1), holds:d.sensitivity.top1_holds, of:d.sensitivity.of,
                        excluded:d.considered.unverified, pick:d.pick, steps:p.steps,
                        margin_millions:Math.round(p.closest*d.deposits_usd/1e6)};}''')
                    body = page.locator('#whyBody').inner_text()
                    assert f'Open on {d["open"]} of {d["n"]} bond-rumor days' in body
                    assert f'about ${d["margin_millions"]}M ({d["margin"]}% of deposits)' in body
                    assert f'{d["holds"]} of {d["of"]} tests' in body
                    assert 'plans need their own simulations' not in body
                    assert ('3:17 PM' if n==25 else '3:18 PM') in body and '3:57 PM' in body
                    assert 'arranging to borrow against them before the run' in body
                    assert len(body.split()) < 240
                    assert page.locator('#whyBody > details[open]').count()==0
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    assert not any(code in body for code in ['pledge_collateral','big_bank_deposits','cash-path','fire_median'])
                    page.screenshot(path=str(args.out/f'why_{theme}_{width}_{n}.png'), full_page=True)
                    cases.append({'theme':theme,'width':width,'days':n,'visible_words':len(body.split()),'metrics':d})
        # Detailed claims and withheld plans remain discoverable, including their provenance.
        for ident in ['whyAlternatives','whyEvidence','whyMore']:
            page.locator(f'#{ident} > summary').click()
        assert 'Early intervention means' in page.locator('#whyEvidence').inner_text()
        assert f'{d["excluded"]} plans need their own simulations' in page.locator('#whyEvidence').inner_text()
        assert page.locator('#whyMore table tbody tr').count()>3
        page.locator('#whyEvidence details > summary').click()
        assert page.locator('#whyEvidence details li').count()==111
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(args.out/'details.png'), full_page=True)
        # Hold an actual old idle-state response until the new plan's acknowledgement.
        old_state = page.request.get(base+'/api/ui/state').json()
        new_state = {**old_state, 'run':{**old_state['run'], 'preset':'lost-bonds', 'kind':'scenario'}}
        hold_old_state = True
        page.wait_for_timeout(400)
        assert len(held)==1
        page.locator('#whyBody .primary[data-plan]').click()
        page.wait_for_function('document.querySelector(".tab.active").id==="tab-crowd"')
        page.wait_for_function('!window.__stampede.eng().loading')
        held[0].fulfill(json=old_state)
        hold_old_state = False
        page.wait_for_timeout(300)
        assert len(planned)==2, planned
        assert planned[0]['body']=={'action':'reset'}
        assert planned[1]['body']['plan']==[{'name':name,'minute':when} for name,when in d['steps']]
        assert 'Bonds pledged in advance' in page.locator('#verdict').inner_text()
        # Missing winner must not make success claims or expose a long table by default.
        page.evaluate('window.__stampede.wall().data.pick=null')
        page.locator('.tabs [data-tab="why"]').click()
        page.locator('#whySort [data-sort="rank"]').click()
        assert 'No validated useful plan' in page.locator('#whyBody').inner_text()
        assert not page.locator('#whyBody > .wh-top').count()
        # The explicit sort action opens the full comparison; the fallback adds no new automatic disclosure.
        page.locator('.tabs [data-tab="crowd"]').click()
        page.locator('#responseTools > summary').click()
        assert page.locator('#responseTools [data-response]').count()==3
        page.locator('#limitsBtn').click()
        page.locator('#limits').wait_for(state='visible')
        assert 'Known gaps' in page.locator('#limits').inner_text()
        page.locator('#limits [data-close]').click()
        # Same warning input, with only the local run-end flag changed.
        warnings = page.evaluate('''() => {
          const api=window.__stampede,s=api.sim(),w={minutes:[10,20],days:[[true,.1,.2],[false,.1,.2]]};
          s.over=false;const ongoing=api.warning(w,15,.15);
          s.over=true;const finished=api.warning(null,15,.15);
          return {ongoing,finished};
        }''')
        assert 'Without help:' in warnings['ongoing'] and warnings['finished']==''
        assert not errors and not blocked, (errors, blocked)
        browser.close()
    report={'passed':len(cases)+3,'failed':0,'cases':cases,'page_errors':errors,
            'blocked_unexpected_writes':blocked,'intercepted_live_plan_requests':planned,
            'details_and_exclusions_accessible':True,'missing_winner_honest':True,'tool_cards_and_model_limits_accessible':True,'warning_disappears_after_run':True,'stale_state_does_not_erase_plan':True}
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print(f'{len(cases)+3} checks passed; no writes sent to the server')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
