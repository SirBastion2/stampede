"""Verify manual saved-replay controls, data fidelity and text/graphic separation.

Uses real saved data and blocks every non-GET request. Screenshots are written
for 10/25/50 days at each desktop width. Requires Playwright Chromium.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright

INSTRUMENT = r'''
window.replayChecks = {tiles:[], texts:[]};
const rawTile = drawTile;
drawTile = function(ctx,p,x,y,s,...rest) {
  const tile={plan:p.plan,x,y,s,texts:[],dots:[]};window.replayChecks.tiles.push(tile);window.currentReplayTile=tile;
  try {return rawTile(ctx,p,x,y,s,...rest);}finally{window.currentReplayTile=null;}
};
const proto=CanvasRenderingContext2D.prototype, rawText=proto.fillText,rawRect=proto.fillRect,rawClear=proto.clearRect,rawArc=proto.arc;
proto.clearRect=function(...args){if(this.canvas.id==='wallCv') window.replayChecks={tiles:[],texts:[]};return rawClear.apply(this,args);};
proto.fillText=function(text,x,y,...rest) {
 if(this.canvas.id==='wallCv' && this.globalAlpha>.9){
  const m=this.measureText(text), b={text:String(text),x:x-m.actualBoundingBoxLeft,y:y-m.actualBoundingBoxAscent,w:m.actualBoundingBoxLeft+m.actualBoundingBoxRight,h:m.actualBoundingBoxAscent+m.actualBoundingBoxDescent};
  window.replayChecks.texts.push(b);window.currentReplayTile?.texts.push(b);
 }
 return rawText.call(this,text,x,y,...rest);
};
proto.arc=function(x,y,r,...rest){if(this.canvas.id==='wallCv' && window.currentReplayTile && !window.currentReplayTile.circle)window.currentReplayTile.circle={x:x-r,y:y-r,w:r*2,h:r*2};return rawArc.call(this,x,y,r,...rest);};
proto.fillRect=function(x,y,w,h){if(this.canvas.id==='wallCv' && w<=3 && h<=3 && window.currentReplayTile)window.currentReplayTile.dots.push({x,y,w,h});return rawRect.call(this,x,y,w,h);};
window.replayInternals={seriesAt,groupCells,groupsTally,groupStandings,cache:miniVaultCache};
'''


def overlaps(a, b):
    return min(a['x']+a['w'], b['x']+b['w'])-max(a['x'],b['x']) > .5 and min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']) > .5


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',default='http://localhost:8000')
    ap.add_argument('--html',type=Path,default=Path('ui/live/stampede_ui.html'))
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'report.json').unlink(missing_ok=True)
    html=a.html.read_text().replace('/* test hook for headless checks */',INSTRUMENT+'\n/* test hook for headless checks */')
    blocked=[];errors=[];cases=[];base=a.url.rstrip('/');held=[];hold25=False
    with sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':1600,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        def guard(r):
            if r.request.method!='GET':blocked.append(r.request.url);return r.abort()
            if r.request.url==base+'/ui':return r.fulfill(body=html,content_type='text/html')
            if hold25 and r.request.url==base+'/api/search?runs=25':held.append(r);return
            r.continue_()
        page.route('**/*',guard);page.goto(base+'/ui')
        page.wait_for_function('window.__stampede.wall().data && !window.__stampede.wall().loading')
        page.locator('.tabs [data-tab="search"]').click()
        for width in [1600,1280,1024]:
            page.set_viewport_size({'width':width,'height':1000 if width==1600 else 900})
            for n in [10,25,50]:
                page.locator(f'#tab-search [data-runs="{n}"]').click()
                page.wait_for_function('n=>window.__stampede.wall().data && !window.__stampede.wall().loading && window.__stampede.wall().runs===n',arg=n)
                page.locator('#wallReplay').click()
                page.wait_for_timeout(300)
                start=page.evaluate('window.__stampede.wall().t')
                assert start==0 and page.locator('#wallPause').inner_text()=='Play'
                assert 'Compare its responses' in page.locator('#headline').inner_text()
                assert page.locator('#wallReplay').is_enabled()
                assert not page.locator('#play').is_visible() and not page.locator('#reset').is_visible()
                assert page.locator('#wallSpeed').is_visible()
                page.screenshot(path=str(a.out/f'{width}_{n}_ready.png'),full_page=True)
                # Speed alone does not start a replay. Pause freezes time and geometry.
                page.locator('#wallSpeed [data-wall-speed="2"]').click();page.wait_for_timeout(180)
                assert page.evaluate('window.__stampede.wall().t')==0
                page.locator('#wallPause').click();page.wait_for_timeout(180)
                assert page.evaluate('window.__stampede.wall().t')>0
                page.locator('#wallPause').click();page.wait_for_timeout(100)
                frozen=page.evaluate('JSON.stringify({t:window.__stampede.wall().t,tiles:[...window.__stampede.wall().gt.values()]})')
                page.wait_for_timeout(200)
                assert page.evaluate('JSON.stringify({t:window.__stampede.wall().t,tiles:[...window.__stampede.wall().gt.values()]})')==frozen
                page.locator('#wallPause').click();page.wait_for_timeout(100)
                page.locator('.tabs [data-tab="why"]').click();page.wait_for_timeout(100)
                stopped=page.evaluate('window.__stampede.wall().t')
                page.locator('.tabs [data-tab="search"]').click();page.wait_for_timeout(150)
                assert page.evaluate('window.__stampede.wall().paused && window.__stampede.wall().t')==stopped
                # State changes and later phases use the exact saved outcomes and series.
                for minute,phase in [(0,'play'),(240,'play'),(720,'play'),(1441,'final')]:
                    page.evaluate('''({minute,phase})=>{const w=window.__stampede.wall();w.paused=true;w.t=minute;w.phase=phase;w.dirty=true;w.gt?.clear();window.scrollTo(0,0);}''',{'minute':minute,'phase':phase})
                    page.wait_for_timeout(180)
                    headline=page.locator('#headline').inner_text()
                    assert headline.startswith('Best tested plan:') if phase=='final' else 'Compare its responses' in headline
                    measured=page.evaluate('''()=>({tiles:window.replayChecks.tiles,texts:window.replayChecks.texts,cache:window.replayInternals.cache.size,width:document.querySelector('#wallCv').clientWidth,height:document.querySelector('#wallCv').clientHeight,overflow:document.documentElement.scrollWidth>innerWidth})''')
                    defects=[]
                    for tile in measured['tiles']:
                        for text in tile['texts']:
                            if text['text']=='asked' or text['text'].endswith('%') or text['text'].startswith('#'):continue
                            if any(overlaps(text,dot) for dot in tile['dots']):defects.append({'plan':tile['plan'],'text':text['text']})
                            if text['x']<tile['x']-1 or text['x']+text['w']>tile['x']+tile['s']+1: defects.append({'text_outside_tile':text['text']})
                    for text in measured['texts']:
                        if text['x'] < -1 or text['y'] < -1 or text['x']+text['w'] > measured['width']+1 or text['y']+text['h'] > measured['height']+1: defects.append({'canvas_text_outside':text['text']})
                    if phase=='final':
                        circles=[tile['circle'] for tile in measured['tiles'] if 'circle' in tile]
                        assert len(circles)==4
                        assert max(c['y'] for c in circles)-min(c['y'] for c in circles)<.01
                        assert max(c['h'] for c in circles)-min(c['h'] for c in circles)<.01
                    assert not defects,(width,n,minute,defects)
                    assert not measured['overflow'] and measured['cache']<=256
                    page.screenshot(path=str(a.out/f'{width}_{n}_{minute}_{phase}.png'),full_page=True)
                    cases.append({'width':width,'days':n,'minute':minute,'phase':phase,'text_graphic_collisions':defects,'cache_entries':measured['cache']})
        # Compare compressed interpolation with the old 30->10->frame interpolation.
        fidelity=page.evaluate('''()=>{
          const d=window.__stampede.wall().data, api=window.replayInternals;let error=0,checked=0;
          const cells=api.groupCells(d);
          for(let g=0;g<d.groups.length;g+=7)for(let k=0;k<d.groupPlans.length;k+=3){
            const source=d.groups[g].cells[k].s, expanded=[];
            for(let m=10;m<=1440;m+=10){const i=Math.floor(m/30)-1,a=i<0?[0,1,0,0]:source[Math.min(i,source.length-1)],b=source[Math.min(i+1,source.length-1)],f=(m%30)/30;expanded.push([m,...a.map((v,j)=>v+(b[j]-v)*f)]);}
            for(const t of [0,1,9,10,29,30,31,347.6,387,719.5,1440]){
              const a=api.seriesAt(cells[g][k],t),b=api.seriesAt({mean_series:expanded},t);a.forEach((v,i)=>{error=Math.max(error,Math.abs(v-b[i]));checked++;});
            }
          }
          // Event-indexed sidebar counts equal the original filter rule, also after seeking backwards.
          for(const t of [0,180,240,540,1000,1441,200]){
            const actual=api.groupStandings(d,t);
            for(const row of actual){const want=d.groups.slice(0,50).filter(g=>{const c=g.cells[row.k];return c.k==='useful'||c.m==null||t<c.m;}).length;if(row.still!==want)throw Error('standing mismatch');}
          }
          return {checked,max_error:error};
        }''')
        assert fidelity['max_error']<1e-12
        # Keyboard controls remain local to the replay; every count completes naturally at 4x.
        page.locator('#wallReplay').click();page.keyboard.press('Space');page.wait_for_timeout(150)
        assert not page.evaluate('window.__stampede.wall().paused')
        page.keyboard.press('4');page.keyboard.press('2');page.keyboard.press('Space')
        assert page.evaluate('window.__stampede.wall().paused && window.__stampede.wall().speed===2')
        page.keyboard.press('r');page.keyboard.press('r')
        assert page.evaluate('window.__stampede.wall().paused && window.__stampede.wall().t===0')
        page.set_viewport_size({'width':1600,'height':1000})
        for n in [10,25,50]:
            page.locator(f'#tab-search [data-runs="{n}"]').click()
            page.wait_for_function('n=>window.__stampede.wall().data && !window.__stampede.wall().loading && window.__stampede.wall().runs===n',arg=n)
            page.locator('#wallReplay').click();page.locator('#wallSpeed [data-wall-speed="4"]').click();page.locator('#wallPause').click()
            page.wait_for_function('window.__stampede.wall().t>250',timeout=15000)
            page.screenshot(path=str(a.out/f'natural_{n}_playing.png'),full_page=True)
            assert 'Compare its responses' in page.locator('#headline').inner_text()
            page.wait_for_function('window.__stampede.wall().phase==="final" && window.__stampede.wall().paused',timeout=30000)
            assert page.evaluate('window.__stampede.wall().t')==1441
            assert page.locator('#headline').inner_text().startswith('Best tested plan:')
            page.screenshot(path=str(a.out/f'natural_{n}_complete.png'),full_page=True)
            cases.append({'kind':'natural_completion','days':n,'speed':4,'paused_at_end':True})
        # An older count response may arrive after the latest choice; the latest choice wins.
        page.reload();page.wait_for_function('window.__stampede.wall().data && !window.__stampede.wall().loading')
        page.locator('.tabs [data-tab="search"]').click()
        payload25=page.request.get(base+'/api/search?runs=25').json();hold25=True
        page.locator('#tab-search [data-runs="25"]').click();page.wait_for_timeout(150);assert len(held)==1
        page.locator('#tab-search [data-runs="50"]').click();page.wait_for_function('window.__stampede.wall().data && window.__stampede.wall().runs===50')
        held[0].fulfill(json=payload25);hold25=False;page.wait_for_timeout(250)
        assert page.evaluate('window.__stampede.wall().runs===50 && window.__stampede.wall().paused && window.__stampede.wall().t===0')
        assert not blocked and not errors,(blocked,errors)
        browser.close()
    report={'passed':len(cases)+4,'failed':0,'cases':cases,'interpolation':fidelity,'controls_verified_for_all_counts':True,'navigation_pauses':True,'late_count_response_ignored':True,'natural_completion_all_counts':True,'keyboard_controls_local':True,'blocked_writes':blocked,'page_errors':errors}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
