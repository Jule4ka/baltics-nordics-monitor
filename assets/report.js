
const D = __DATA__;
const THEME_LABEL = {}; (D.themes||[]).forEach(t=>THEME_LABEL[t.id]=t.label);   // id -> name (for tooltips + multi-label tags)

// ---- act 1: Baltic defence-share choropleth ----
(function(){
  const M = D.map, sh = D.shares;
  const op = p => (0.15 + 0.85 * (p / 100));     // sequential: fill-opacity of accent on an absolute 0–100% scale
  const by = {}; sh.forEach(s => by[s.code] = s);
  const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const fmt = iso => { if(!iso) return '—'; const [y,m,d] = iso.split('-'); return `${+d} ${MON[+m-1]} ${y}`; };
  const svg = document.getElementById('cmap');
  svg.setAttribute('viewBox', M.viewBox);
  const homes = D.homes || {};
  let g = '';
  for (const code in M.paths){
    const s = by[code] || {country:code, pct:0, matched:0, total:0};
    const home = homes[code];
    const path = `<path class="geo" d="${M.paths[code]}" fill-opacity="${op(s.pct).toFixed(3)}">`
       + `<title>${s.country} — ${s.pct}% defence (${s.matched}/${s.total})`
       + `${home ? ' · click to open the source' : ''}</title></path>`;
    g += home
      ? `<a class="geo-link" href="${home}" target="_blank" rel="noopener">${path}</a>`
      : path;
  }
  for (const code in M.centroids){
    const [cx, cy] = M.centroids[code], s = by[code] || {pct:0};
    g += `<text class="geo-lab" x="${cx}" y="${cy-3}">${code}</text>`
       + `<text class="geo-sub" x="${cx}" y="${cy+15}">${s.pct}%</text>`;
  }
  svg.innerHTML = g;
  document.getElementById('clegend').innerHTML = sh.map(s =>
    `<div class="cl-row"><span class="cl-sw" style="opacity:${op(s.pct).toFixed(3)}"></span>`
    + `<span class="cl-txt"><span class="cl-name">${s.country}</span>`
    + `<span class="cl-dates">${fmt(s.d0)} – ${fmt(s.d1)}</span></span>`
    + `<span class="cl-num"><span class="cl-pct">${s.pct}%</span>`
    + `<span class="cl-frac">${s.matched}/${s.total}</span></span></div>`
  ).join('') + `<div class="scale mono"><span>0%</span><span class="scale-bar"></span><span>100%</span></div>`;
  const cov = D.coverage || {};
  const totAll = sh.reduce((a,s)=>a+s.total, 0);
  document.getElementById('cov').innerHTML =
    `<span class="star">✴</span> Articles collected <b>${fmt(cov.from)}</b> → <b>${fmt(cov.to)}</b>`
    + ` · <b>${D.points.length}</b> defence-themed of <b>${totAll}</b> total headlines`;
})();

// ---- act 1.5: temporal timeline — daily stacked bars by country + range brush ----
(function(){
  const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const fmt = iso => { if(!iso) return '—'; const [y,m,d] = iso.split('-'); return `${+d} ${MON[+m-1]}`; };
  const fmtY = iso => { const [y,m,d] = iso.split('-'); return `${+d} ${MON[+m-1]} ${y}`; };
  const addDays = (iso,n) => { const dt = new Date(iso+'T00:00:00Z'); dt.setUTCDate(dt.getUTCDate()+n); return dt.toISOString().slice(0,10); };

  // map source label -> country code via shares, then order EE, LV, LT
  const code = {}; (D.shares||[]).forEach(s => code[s.country] = s.code);
  const ORDER = ['EE','LV','LT'];
  const CLS = {EE:'ee', LV:'lv', LT:'lt'};
  const NAME = {EE:'Estonia · ERR', LV:'Latvia · LSM', LT:'Lithuania · LRT'};

  // build the full day span (including zero days) from coverage, falling back to points
  const ds = D.points.map(p=>p.d).filter(Boolean).sort();
  const cov = D.coverage || {};
  const from = cov.from || ds[0], to = cov.to || ds[ds.length-1];
  if(!from || !to) return;
  const days = []; for(let d=from; d<=to; d=addDays(d,1)){ days.push(d); if(days.length>1000) break; }
  const idx = {}; days.forEach((d,i)=>idx[d]=i);

  const counts = days.map(()=>({EE:0,LV:0,LT:0}));
  const arts = days.map(()=>[]);
  D.points.forEach(p=>{ const c = code[p.s]; const i = idx[p.d];
    if(c==null || i==null || !(c in counts[i])) return; counts[i][c]++; arts[i].push(p); });
  const maxTot = Math.max(1, ...counts.map(c=>c.EE+c.LV+c.LT));

  const hidden = new Set();               // toggled-off country codes
  let lo = 0, hi = days.length-1;         // selected day-index window
  const N = days.length;
  const loEl = document.getElementById('tlLo'), hiEl = document.getElementById('tlHi');
  loEl.max = hiEl.max = N-1; hiEl.value = N-1;

  function legend(){
    document.getElementById('tlLegend').innerHTML = ORDER.map(c=>{
      const tot = counts.reduce((a,d)=>a+d[c],0);
      return `<div class="leg${hidden.has(c)?' off':''}" data-c="${c}" tabindex="0">`
        + `<span class="sw tl-seg ${CLS[c]}" style="border-radius:2px"></span>${NAME[c]}`
        + `<span class="lc">${tot}</span></div>`;
    }).join('');
    document.querySelectorAll('#tlLegend .leg').forEach(el=>{
      const toggle=()=>{const c=el.dataset.c; hidden.has(c)?hidden.delete(c):hidden.add(c);
        el.classList.toggle('off'); chart(); listing();};
      el.onclick=toggle;
      el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
    });
  }

  function dayTotal(i){ return ORDER.reduce((a,c)=>a + (hidden.has(c)?0:counts[i][c]), 0); }

  function chart(){
    const el = document.getElementById('tlChart');
    el.innerHTML = days.map((d,i)=>{
      const tot = dayTotal(i);
      const segs = ORDER.filter(c=>!hidden.has(c)).map(c=>{
        const n = counts[i][c]; if(!n) return '';
        return `<div class="tl-seg ${CLS[c]}" style="height:${(n/tot*100).toFixed(2)}%"></div>`;
      }).join('');
      const parts = ORDER.map(c=>`${c} ${counts[i][c]}`).join(' · ');
      const out = (i<lo||i>hi) ? ' out' : '';
      const barH = (tot/maxTot*100).toFixed(2);
      return `<div class="tl-col${out}" data-i="${i}" title="${fmtY(d)} — ${tot} article${tot!==1?'s':''} (${parts})">`
        + `<div class="tl-bar" style="height:${barH}%">${segs}</div></div>`;
    }).join('');
    el.querySelectorAll('.tl-col').forEach(col=>{
      col.onclick=()=>{ const i=+col.dataset.i; lo=hi=i; loEl.value=hi==0?0:i; hiEl.value=i; sync(); };
    });
  }

  function slider(){
    const rng = document.getElementById('tlRange');
    const a = lo/(N-1||1)*100, b = hi/(N-1||1)*100;
    rng.style.left = a+'%'; rng.style.width = (b-a)+'%';
  }

  function listing(){
    let picked = [];
    for(let i=lo;i<=hi;i++) picked = picked.concat(arts[i].filter(p=>!hidden.has(code[p.s])));
    picked.sort((x,y)=> y.d.localeCompare(x.d) || x.h.localeCompare(y.h));
    const span = lo===hi ? fmtY(days[lo]) : `${fmt(days[lo])} – ${fmtY(days[hi])}`;
    document.getElementById('tlSel').innerHTML =
      `<b>${picked.length}</b> article${picked.length!==1?'s':''} · ${span}`;
    document.getElementById('tlArtsLabel').textContent =
      picked.length ? `read ${picked.length} article${picked.length!==1?'s':''} in view` : 'no articles in this range';
    document.getElementById('tlList').innerHTML = picked.map(p=>
      `<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}`
      + `<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
  }

  function sync(){ chart(); slider(); listing(); }

  loEl.oninput=()=>{ lo=Math.min(+loEl.value,+hiEl.value); hi=Math.max(+loEl.value,+hiEl.value); sync(); };
  hiEl.oninput=loEl.oninput;
  document.getElementById('tlReset').onclick=()=>{ lo=0; hi=N-1; loEl.value=0; hiEl.value=N-1;
    hidden.clear(); legend(); sync(); };

  const artsBox = document.getElementById('tlArts'), artsH = artsBox.querySelector('.tl-arts-h');
  const toggleArts=()=>{ const open=artsBox.classList.toggle('open'); artsH.setAttribute('aria-expanded',open); };
  artsH.onclick=toggleArts;
  artsH.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggleArts();}};

  // axis: first · middle · last
  document.getElementById('tlAxis').innerHTML =
    `<span>${fmtY(days[0])}</span><span class="tl-ax-mid">${fmt(days[Math.floor((N-1)/2)])}</span><span>${fmtY(days[N-1])}</span>`;

  legend(); sync();
})();

const cvs = document.getElementById('map'), ctx = cvs.getContext('2d');
const tip = document.getElementById('tip'), wrap = cvs.parentElement;
// start with every theme visible; click a legend entry to toggle it off/on
const hidden = new Set();
let pts = [], labelPos = [], hover = -1, anim = 1;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const isDark = () => { const t = document.documentElement.getAttribute('data-theme');
  return t === 'dark' || (t !== 'light' && matchMedia('(prefers-color-scheme:dark)').matches); };
let C = isDark() ? D.colorsDark : D.colors;      // theme-aware categorical palette

// header stats — the numeric three, then an "Updated" tile: the DATE in the same
// number font, plus an info icon that reveals the exact time (seconds + timezone)
// on hover or click.
const gDate = (D.generated||'—').split(' ')[0];    // "YYYY-MM-DD"
const gFull = D.generatedFull || D.generated || '';
const popLines = gFull.split(' · ').map(p=>`<span class="ip-l">${p}</span>`).join('');
const info = `<span class="stat-info"><button type="button" class="info-btn" aria-label="Show exact update time">`
  + `<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7"/><path d="M8 7v4"/><circle cx="8" cy="4.6" r=".9" fill="currentColor" stroke="none"/></svg>`
  + `</button><span class="info-pop mono" role="tooltip">${popLines}</span></span>`;
document.getElementById('stats').innerHTML = [
  [D.points.length,'Articles',''], [D.sources,'Sources',''],
  [D.themes.length,'Themes',''], [`${gDate}${info}`,'Updated','stat-meta']
].map(s=>`<div class="stat ${s[2]}"><div class="n">${s[0]}</div><div class="l">${s[1]}</div></div>`).join('');
// click the info icon to pin the tooltip open (hover works on its own; this is for touch)
document.querySelectorAll('#stats .info-btn').forEach(b=>{
  b.onclick=e=>{ e.stopPropagation(); b.closest('.stat-info').classList.toggle('open'); };
});
document.addEventListener('click',()=>document.querySelectorAll('#stats .stat-info.open')
  .forEach(el=>el.classList.remove('open')));

// legend — toggle themes on/off; rebuilt on theme change so swatches track colours
const leg = document.getElementById('legend');
function renderLegend(){
  leg.innerHTML = D.themes.map(t =>
    `<div class="leg${hidden.has(t.id)?' off':''}" data-c="${t.id}" tabindex="0">`
    + `<span class="sw" style="background:${C[t.id]}"></span>${t.label}`
    + `<span class="lc">${t.count}</span></div>`
  ).join('');
  leg.querySelectorAll('.leg').forEach(el=>{
    const toggle=()=>{const id=+el.dataset.c; hidden.has(id)?hidden.delete(id):hidden.add(id);
      el.classList.toggle('off'); draw();};
    el.onclick=toggle;
    el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
  });
}

// per-theme cards — the article list is hidden until you select (click) a theme
function renderCards(){
  document.getElementById('cards').innerHTML = D.themes.map(t=>{
    const arts = D.points.filter(p=>(p.cs||[p.c]).includes(t.id)).sort((a,b)=>b.d.localeCompare(a.d));
    const col = C[t.id];
    const li = arts.map(p=>{
      const also = (p.cs||[]).filter(id=>id!==t.id).map(id=>THEME_LABEL[id]).filter(Boolean);
      const tag = also.length ? ` · also ${also.join(', ')}` : '';
      return `<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}${tag}</span></a></li>`;
    }).join('');
    return `<div class="card" style="--dot:${col}"><h3 class="card-h">`
      + `<span class="sw" style="width:9px;height:9px;border-radius:50%;background:${col};display:inline-block"></span>`
      + `${t.label}<span class="lc">${t.count}</span><span class="caret">▶</span></h3>`
      + `<p class="card-hint">click to read ${t.count} article${t.count>1?'s':''}</p>`
      + `<ul class="card-arts">${li}</ul></div>`;
  }).join('');
  document.querySelectorAll('#cards .card').forEach(card=>{
    card.querySelector('.card-h').onclick = () => card.classList.toggle('open');
  });
}

// act 3 — keyword frequency bars; click a term to reveal its articles
function renderFreq(){
  const K = D.keywords || [];
  const maxn = Math.max(...K.map(k=>k.n), 1);
  document.getElementById('freq').innerHTML = K.map(k=>{
    const li = k.arts.map(i=>D.points[i]).sort((a,b)=>b.d.localeCompare(a.d))
      .map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
    return `<div class="freq-item"><div class="freq-row" tabindex="0" role="button" aria-expanded="false">`
      + `<span class="freq-term">${k.term}</span>`
      + `<div class="freq-track"><div class="freq-fill" style="width:${(k.n/maxn*100).toFixed(1)}%"></div></div>`
      + `<span class="freq-n mono">${k.n}</span></div>`
      + `<ul class="freq-arts">${li}</ul></div>`;
  }).join('');
  document.querySelectorAll('#freq .freq-row').forEach(row=>{
    const item = row.parentElement;
    const toggle=()=>{const open=item.classList.toggle('open'); row.setAttribute('aria-expanded', open);};
    row.onclick=toggle;
    row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
  });
}
// make an absolutely-positioned overlay draggable so it never permanently hides dots.
// a real drag (moved past a few px) suppresses the click that would otherwise toggle.
function makeDraggable(el){
  let sx, sy, ox, oy, pid, down = false, dragging = false, moved = false;
  el.style.cursor = 'grab'; el.title = 'drag to move';
  el.addEventListener('pointerdown', e => {
    down = true; dragging = false; moved = false; pid = e.pointerId;
    sx = e.clientX; sy = e.clientY;
    const r = el.getBoundingClientRect(), pr = el.offsetParent.getBoundingClientRect();
    ox = r.left - pr.left; oy = r.top - pr.top;
  });
  el.addEventListener('pointermove', e => {
    if(!down) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if(!dragging && Math.abs(dx) + Math.abs(dy) > 4){    // only NOW does it become a drag
      dragging = true; moved = true;
      el.style.right = 'auto'; el.style.left = ox + 'px'; el.style.top = oy + 'px';
      el.style.cursor = 'grabbing'; el.setPointerCapture(pid);
    }
    if(dragging){ el.style.left = (ox + dx) + 'px'; el.style.top = (oy + dy) + 'px'; }
  });
  const end = () => { down = false; if(dragging){ el.style.cursor = 'grab';
    try{ el.releasePointerCapture(pid); }catch(_){} } dragging = false; };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
  el.addEventListener('click', e => { if(moved){ e.stopPropagation(); e.preventDefault(); moved = false; } }, true);
}

// edits — ERR stories whose headline was revised after publishing (newest change first).
// Each item shows the version history: the first-seen headline, then each later wording,
// with the date it was first captured. Hidden entirely when nothing has been edited.
function renderEdits(){
  const E = D.edits || [];
  const panel = document.getElementById('editsPanel'), head = document.getElementById('editsH');
  const box = document.getElementById('edits');
  if(!box) return;
  if(!E.length){ if(panel) panel.style.display='none'; if(head) head.style.display='none'; return; }
  box.innerHTML = E.map(e=>{
    const vs = e.versions.map((v,i)=>{
      const last = i===e.versions.length-1;
      const mark = i ? '↳' : '•';
      return `<li class="ed-v ${last?'ed-cur':'ed-old'}"><span class="ed-mark">${mark}</span>`
        + `<a href="${v.u}" target="_blank" rel="noopener">${v.h}</a>`
        + `<span class="ed-when mono">${v.t}</span></li>`;   // 'DD Mon HH:MM CET/CEST'
    }).join('');
    return `<div class="ed-item"><div class="ed-meta mono"><span class="ed-src">${e.source}</span>`
      + `<span class="ed-n">${e.n} versions</span></div><ul class="ed-vers">${vs}</ul></div>`;
  }).join('');
}

renderLegend(); renderCards(); renderFreq(); renderEdits();
makeDraggable(leg);

// bounds
const xs=D.points.map(p=>p.x), ys=D.points.map(p=>p.y);
let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
const px=(maxX-minX)*0.08||1, py=(maxY-minY)*0.08||1;
minX-=px;maxX+=px;minY-=py;maxY+=py;

function layout(){
  const dpr=devicePixelRatio||1, w=wrap.clientWidth, h=cvs.clientHeight;
  cvs.width=w*dpr; cvs.height=h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const pad=34;
  pts=D.points.map(p=>({
    ...p,
    sx:pad+(p.x-minX)/(maxX-minX)*(w-2*pad),
    sy:pad+(maxY-p.y)/(maxY-minY)*(h-2*pad)
  }));
  labelPos=D.themes.map(t=>({
    id:t.id, label:t.label,
    sx:pad+(t.cx-minX)/(maxX-minX)*(w-2*pad),
    sy:pad+(maxY-t.cy)/(maxY-minY)*(h-2*pad)
  }));
}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function draw(){
  const w=wrap.clientWidth,h=cvs.clientHeight;
  ctx.clearRect(0,0,w,h);
  // faint grid
  ctx.strokeStyle=css('--grid');ctx.lineWidth=1;
  for(let i=1;i<6;i++){const gx=i/6*w;ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,h);ctx.stroke();
    const gy=i/6*h;ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke();}
  const ground=css('--panel');
  pts.forEach((p,i)=>{
    if(hidden.has(p.c))return;
    const r=Math.max(0,(i===hover?8:5.5)*anim);
    ctx.beginPath();ctx.arc(p.sx,p.sy,r,0,7);
    ctx.fillStyle=C[p.c];ctx.globalAlpha=i===hover?1:0.85;ctx.fill();
    ctx.globalAlpha=1;ctx.lineWidth=i===hover?2:1.2;ctx.strokeStyle=ground;ctx.stroke();
    if(i===hover){ctx.beginPath();ctx.arc(p.sx,p.sy,r+4,0,7);
      ctx.strokeStyle=C[p.c];ctx.globalAlpha=.4;ctx.lineWidth=1.5;ctx.stroke();ctx.globalAlpha=1;}
  });
  // theme identity now lives entirely in the legend — no labels drawn on the map
}
function nearest(mx,my){
  let best=-1,bd=169;
  pts.forEach((p,i)=>{if(hidden.has(p.c))return;
    const d=(p.sx-mx)**2+(p.sy-my)**2;if(d<bd){bd=d;best=i;}});
  return best;
}
cvs.addEventListener('mousemove',e=>{
  const rc=cvs.getBoundingClientRect(),mx=e.clientX-rc.left,my=e.clientY-rc.top;
  const h=nearest(mx,my);
  if(h!==hover){hover=h;draw();}
  if(h>=0){const p=pts[h];cvs.style.cursor='pointer';
    tip.innerHTML=`<b>${p.h}</b><span class="meta">${p.s} · ${p.d} · ${(p.cs||[p.c]).map(id=>THEME_LABEL[id]).filter(Boolean).join(', ')}</span>`;
    tip.style.opacity=1;
    let tx=mx+16,ty=my+16;
    if(tx+tip.offsetWidth>wrap.clientWidth)tx=mx-tip.offsetWidth-16;
    if(ty+tip.offsetHeight>cvs.clientHeight)ty=my-tip.offsetHeight-16;
    tip.style.left=tx+'px';tip.style.top=ty+'px';
  }else{cvs.style.cursor='default';tip.style.opacity=0;}
});
cvs.addEventListener('mouseleave',()=>{hover=-1;tip.style.opacity=0;draw();});
cvs.addEventListener('click',()=>{if(hover>=0)window.open(pts[hover].u,'_blank','noopener');});
new ResizeObserver(()=>{layout();draw();}).observe(wrap);
layout();
if(reduce){anim=1;draw();}
else{anim=0;const t0=performance.now();
  (function run(t){anim=Math.max(0,Math.min(1,(t-t0)/500));draw();if(anim<1)requestAnimationFrame(run);})(t0);}
// on theme toggle: swap palette, rebuild legend + cards swatches, redraw canvas
function onTheme(){ C = isDark() ? D.colorsDark : D.colors; renderLegend(); renderCards(); draw(); }
new MutationObserver(onTheme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',onTheme);

// ── collapsible sections ─────────────────────────────────────────────────────
// Each `.section-h` becomes a toggle; everything between it and the next
// `.section-h` is wrapped into a body that starts collapsed, so the page opens as
// a list of sections you expand on demand (numbers, timeline, themes, words, …).
// Canvases (the maps) redraw on expand via their ResizeObserver; we also fire a
// resize event as a belt-and-suspenders for anything measuring width.
(function(){
  const main = document.querySelector('main');
  if(!main) return;
  const heads = [...main.children].filter(el => el.classList.contains('section-h'));
  heads.forEach(h=>{
    const body = document.createElement('div');
    body.className = 'act-body';
    let el = h.nextElementSibling;
    while(el && !el.classList.contains('section-h')){
      const next = el.nextElementSibling;
      body.appendChild(el);
      el = next;
    }
    h.insertAdjacentElement('afterend', body);
    const caret = document.createElement('span');
    caret.className = 'act-caret'; caret.setAttribute('aria-hidden','true');
    caret.textContent = '▸';
    h.insertBefore(caret, h.firstChild);
    h.classList.add('act-h');
    h.setAttribute('role','button');
    h.setAttribute('tabindex','0');
    h.setAttribute('aria-expanded','false');
    const toggle = ()=>{
      const open = body.classList.toggle('open');
      h.classList.toggle('open', open);
      h.setAttribute('aria-expanded', open ? 'true' : 'false');
      if(open) window.dispatchEvent(new Event('resize'));   // nudge canvases to redraw
    };
    h.addEventListener('click', toggle);
    h.addEventListener('keydown', e=>{
      if(e.key==='Enter' || e.key===' '){ e.preventDefault(); toggle(); }
    });
  });
})();
