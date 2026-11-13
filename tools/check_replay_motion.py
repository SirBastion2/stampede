"""Check every painted replay frame for overlapping cards, with all writes blocked."""
import argparse,json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright

SHIM=r'''
window.motion={frame:0,checked:-1,enabled:false,boxes:[],frames:0,badFrames:0,examples:[],redCenters:0,maxRedCenterOffset:0};
const proto=CanvasRenderingContext2D.prototype, clear=proto.clearRect,image=proto.drawImage,round=proto.roundRect,arc=proto.arc,tile=drawTile,draw=drawWall;
proto.roundRect=function(x,y,w,h,...rest){if(!this.canvas.id&&!this.canvas._motionFrame)this.canvas._motionFrame={x:x+w/2,y:y+h/2,red:this.strokeStyle==='#ef4b55'};return round.call(this,x,y,w,h,...rest);};
proto.arc=function(x,y,r,...rest){const f=this.canvas._motionFrame;if(!this.canvas.id&&f?.red&&!f.checked){f.checked=true;window.motion.redCenters++;window.motion.maxRedCenterOffset=Math.max(window.motion.maxRedCenterOffset,Math.abs(x-f.x),Math.abs(y-f.y));}return arc.call(this,x,y,r,...rest);};
proto.clearRect=function(...args){if(this.canvas.id==='wallCv'){window.motion.frame++;window.motion.boxes=[];}return clear.apply(this,args);};
drawTile=function(ctx,p,x,y,s,alpha,...rest){const previous=window.inMotionTile;window.inMotionTile=ctx.canvas.id==='wallCv';
 try{if(window.inMotionTile&&p.single&&s>=MIN_SIZE&&alpha>.1)window.motion.boxes.push({x,y,s,g:p.plan.split('#').at(-1)});return tile(ctx,p,x,y,s,alpha,...rest);}finally{window.inMotionTile=previous;}};
proto.drawImage=function(img,x,y,w,h,...rest){if(this.canvas.id==='wallCv'&&!window.inMotionTile&&this.globalAlpha>.1&&w>=1)window.motion.boxes.push({x,y,s:w,g:window.motionGroup});return image.call(this,img,x,y,w,h,...rest);};
drawWall=function(...args){const result=draw(...args),m=window.motion;if(!m.enabled||m.checked===m.frame)return result;m.checked=m.frame;m.frames++;
 const groups=new Map();for(const b of m.boxes){if(!groups.has(b.g))groups.set(b.g,[]);groups.get(b.g).push(b);}
 let bad=false;for(const boxes of groups.values())for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
  const a=boxes[i],b=boxes[j],dx=Math.min(a.x+a.s,b.x+b.s)-Math.max(a.x,b.x),dy=Math.min(a.y+a.s,b.y+b.s)-Math.max(a.y,b.y);
  if(dx>1&&dy>1){bad=true;if(m.examples.length<6)m.examples.push({minute:wall.t,a,b,dx,dy});}
 }
 if(bad)m.badFrames++;return result;};
'''

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--html',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--speeds',default='1,2,4');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
source=a.html.read_text();shim=SHIM.replace('MIN_SIZE','6' if 'function miniCard(' in source else '14')
html=source.replace('G.forEach((g, i) => {','G.forEach((g, i) => { window.motionGroup=g.seed;').replace('/* test hook for headless checks */',shim+'\n/* test hook for headless checks */')
errors=[];blocked=[];cases=[]
with sync_playwright() as p:
 b=p.chromium.launch();page=b.new_page(viewport={'width':1600,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
 def guard(r):
  if r.request.method!='GET':blocked.append(r.request.url);return r.abort()
  if r.request.url=='http://localhost:8000/ui':return r.fulfill(body=html,content_type='text/html')
  r.continue_()
 page.route('**/*',guard)
 for n in [10,25,50]:
  for speed in map(int,a.speeds.split(',')):
   page.goto('http://localhost:8000/ui');page.wait_for_function('window.__stampede.wall().data && !window.__stampede.wall().loading');page.locator('.tabs [data-tab="search"]').click()
   page.locator(f'#tab-search [data-runs="{n}"]').click();page.wait_for_function('n=>window.__stampede.wall().data && !window.__stampede.wall().loading && window.__stampede.wall().runs===n',arg=n)
   page.evaluate('''speed=>{const w=window.__stampede.wall();w.t=110;w.phase='play';w.paused=false;w.speed=speed;w.last=0;window.motion.frames=0;window.motion.badFrames=0;window.motion.examples=[];window.motion.redCenters=0;window.motion.maxRedCenterOffset=0;window.motion.enabled=true;}''',speed)
   page.wait_for_timeout(12500/speed)
   result=page.evaluate('''()=>{window.motion.enabled=false;window.__stampede.wall().paused=true;return {frames:window.motion.frames,badFrames:window.motion.badFrames,examples:window.motion.examples,redCenters:window.motion.redCenters,maxRedCenterOffset:window.motion.maxRedCenterOffset};}''')
   cases.append({'days':n,'speed':speed,**result});print(n,speed,result['frames'],'frames;',result['badFrames'],'with overlap',flush=True)
   page.screenshot(path=str(a.out/f'{n}_{speed}_elimination.png'),full_page=True)
 b.close()
r={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':cases,'bad_frames':sum(c['badFrames'] for c in cases),'painted_frames_checked':sum(c['frames'] for c in cases),'max_red_center_offset':max(c['maxRedCenterOffset'] for c in cases),'red_centers_checked':sum(c['redCenters'] for c in cases),'page_errors':errors,'blocked_writes':blocked}
(a.out/'report.json').write_text(json.dumps(r,indent=2));raise SystemExit(bool(r['bad_frames'] or r['max_red_center_offset']>.01 or errors or blocked))
