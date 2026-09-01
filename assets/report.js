
const D = __DATA__;

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

// act 4 — named entities (people / organizations / places). Each row carries a coverage-
// tone chip, a weekly-mentions sparkline and an EE/LV/LT country strip; click to expand the
// "seen alongside" co-occurrences + the articles. Hidden entirely when spaCy produced nothing.
function renderEntities(){
  const E = D.entities || {};
  const box = document.getElementById('entities'); if(!box) return;
  const groups = [['people','People'],['orgs','Organizations'],['places','Places']];
  const has = groups.some(([k])=>(E[k]||[]).length);
  if(!has){
    const panel=document.getElementById('entPanel'), head=document.getElementById('entH');
    if(panel) panel.style.display='none'; if(head) head.style.display='none'; return;
  }
  // tone -> red(escalatory)/blue(de-escalatory) mixed from neutral grey by magnitude
  const entTone = t => { const a=Math.min(1,Math.abs(t||0)), tgt=(t||0)>=0?[208,59,59]:[42,120,214],
    mix=x=>Math.round(150+(x-150)*a); return `rgb(${mix(tgt[0])},${mix(tgt[1])},${mix(tgt[2])})`; };
  const spark = s => { if(!s||s.length<2) return '<span class="ent-spark"></span>';
    const mx=Math.max(1,...s), W=60, H=16,
      pts=s.map((v,i)=>`${(i/(s.length-1)*W).toFixed(1)},${(H-1-(v/mx)*(H-2)).toFixed(1)}`).join(' ');
    return `<svg class="ent-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><polyline points="${pts}"/></svg>`; };
  const geo = cc => { if(!cc) return '<span class="ent-geo"></span>';
    const o=['EE','LV','LT'], tot=o.reduce((a,c)=>a+(cc[c]||0),0)||1;
    return `<span class="ent-geo" title="${o.map(c=>c+' '+(cc[c]||0)).join(' · ')}">`
      + o.map(c=>`<i class="eg ${c.toLowerCase()}" style="width:${((cc[c]||0)/tot*100).toFixed(0)}%"></i>`).join('')
      + `</span>`; };
  box.innerHTML = groups.map(([k,label])=>{
    const items = E[k]||[]; if(!items.length) return '';
    const rows = items.map(e=>{
      const li = e.arts.map(i=>D.points[i]).filter(Boolean).sort((a,b)=>b.d.localeCompare(a.d))
        .map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
      const withHtml = (e.with&&e.with.length)
        ? `<div class="ent-with"><span class="ew-lab">seen alongside</span>`
          + e.with.map(w=>`<span class="ew-chip">${w.name}<b>${w.n}</b></span>`).join('') + `</div>`
        : '';
      const tone = (typeof e.tone==='number')
        ? `<span class="ent-tone" style="background:${entTone(e.tone)}" title="mean coverage tone ${e.tone>0?'+':''}${e.tone}">${e.tone>0?'+':''}${e.tone.toFixed(1)}</span>`
        : '<span class="ent-tone blank"></span>';
      return `<div class="ent-item"><div class="ent-row" tabindex="0" role="button" aria-expanded="false">`
        + tone + `<span class="ent-name">${e.name}</span>` + spark(e.series) + geo(e.countries)
        + `<span class="ent-n mono">${e.n}</span></div>`
        + `<div class="ent-arts">${withHtml}<ul>${li}</ul></div></div>`;
    }).join('');
    return `<div class="ent-col"><h3 class="ent-h">${label}</h3>${rows}</div>`;
  }).join('');
  box.querySelectorAll('.ent-row').forEach(row=>{
    const item = row.parentElement;
    const toggle=()=>{const open=item.classList.toggle('open'); row.setAttribute('aria-expanded', open);};
    row.onclick=toggle;
    row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
  });
}

renderFreq(); renderEntities();

// ── collapsible sections ─────────────────────────────────────────────────────
// Each `.section-h` becomes a toggle; everything between it and the next
// `.section-h` is wrapped into a body that starts collapsed, so the page opens as
// a list of sections you expand on demand (numbers, timeline, words, …).
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

// ── act 6 — relationship network (force web + adjacency matrix) ───────────────
// Built from D.entities co-occurrences ("with"). No external libraries: the force
// layout, pan/zoom, drag and both canvases are hand-rolled so the page stays
// self-contained and offline. Nodes coloured by kind (people/orgs/places); the
// dossier reuses each entity's series, countries, tone and articles.
(function(){
  const E = D.entities || {}; const stage = document.getElementById('netStage');
  if(!stage) return;
  const groups=[['people','Person'],['orgs','Organization'],['places','Place']];
  const has = groups.some(([k])=>(E[k]||[]).length);
  const panel=document.getElementById('netPanel'), secH=document.getElementById('netSecH');
  if(!has){ if(panel) panel.style.display='none'; if(secH) secH.style.display='none'; return; }

  const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
  const GC = {people:'--accent',orgs:'--indigo',places:'--green'};
  const groupColor = g => css(GC[g]||'--muted');
  const groupLabel = g => (groups.find(x=>x[0]===g)||[,'—'])[1];
  const WEEKS = E.weeks || [];
  const MONS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const fmtWk = d => { if(!d) return ''; const [,m,dd]=d.split('-'); return `${+dd} ${MONS[+m-1]}`; };

  // build nodes + undirected links
  const nodes=[], byId=new Map();
  groups.forEach(([k])=>(E[k]||[]).forEach(e=>{
    const n={id:e.name,group:k,n:e.n,tone:e.tone,countries:e.countries||{},series:e.series||[],
      arts:e.arts||[],x:0,y:0,vx:0,vy:0,fixed:false,deg:0};
    nodes.push(n); byId.set(e.name,n);
  }));
  const pkey=(a,b)=> a<b?a+' '+b:b+' '+a;
  const seen=new Map();
  groups.forEach(([k])=>(E[k]||[]).forEach(e=>{
    (e.with||[]).forEach(w=>{ if(!byId.has(w.name)||w.name===e.name) return;
      const key=pkey(e.name,w.name); seen.set(key,Math.max(seen.get(key)||0,w.n)); });
  }));
  const links=[]; const pair=new Map();
  seen.forEach((v,key)=>{ const [a,b]=key.split(' '); const s=byId.get(a),t=byId.get(b);
    if(!s||!t) return; const l={s,t,value:v}; links.push(l); pair.set(key,v); s.deg++; t.deg++; });
  const pairVal=(a,b)=> a===b?0:(pair.get(pkey(a,b))||0);
  const valMax=Math.max(1,...links.map(l=>l.value));
  const nMax=Math.max(...nodes.map(n=>n.n));
  nodes.forEach(n=> n.r = 4 + Math.sqrt(n.n)*1.9);
  const wScale=v=>0.5+Math.sqrt(v)*0.4;

  // state
  const active={people:true,orgs:true,places:true};
  let threshold=4, hoverN=null, selN=null, view='web', inited=false;
  let W=0,H=0,DPR=Math.max(1,window.devicePixelRatio||1), tf={x:0,y:0,k:1};
  const nodeShown=n=>active[n.group];
  const linkShown=l=>active[l.s.group]&&active[l.t.group]&&l.value>=threshold;
  const neighbors=n=>{ const s=new Set([n.id]); links.forEach(l=>{ if((l.s===n||l.t===n)&&linkShown(l)){s.add(l.s.id);s.add(l.t.id);} }); return s; };

  // canvases
  const web=document.getElementById('netWeb'), wx=web.getContext('2d');
  const mx=document.getElementById('netMx'), mc=mx.getContext('2d');
  const tip=document.getElementById('netTip');

  // ── force layout (Fruchterman–Reingold-ish) ──
  let alpha=0, raf=0;
  function seed(){
    const R=Math.max(80,Math.min(W,H)/3);
    nodes.forEach((n,i)=>{ const a=i/nodes.length*6.2832, rr=R*(0.35+Math.random()*0.65);
      n.x=Math.cos(a)*rr; n.y=Math.sin(a)*rr; n.vx=n.vy=0; });
  }
  function tick(){
    const k=Math.sqrt((W*H)/(nodes.length+1))*0.42, k2=k*k, temp=alpha*k*0.9, GRAV=0.045;
    for(const n of nodes){ n._fx=0; n._fy=0; }
    for(let i=0;i<nodes.length;i++){ const a=nodes[i]; if(!nodeShown(a)) continue;
      for(let j=i+1;j<nodes.length;j++){ const b=nodes[j]; if(!nodeShown(b)) continue;
        let dx=a.x-b.x, dy=a.y-b.y, d=Math.hypot(dx,dy)||0.01;
        const rep=k2/d; const ux=dx/d, uy=dy/d;
        a._fx+=ux*rep; a._fy+=uy*rep; b._fx-=ux*rep; b._fy-=uy*rep;
        const md=a.r+b.r+6; if(d<md){ const push=(md-d)*0.5; a._fx+=ux*push; a._fy+=uy*push; b._fx-=ux*push; b._fy-=uy*push; }
      }
    }
    for(const l of links){ if(!linkShown(l)) continue; const a=l.s,b=l.t;
      let dx=a.x-b.x, dy=a.y-b.y, d=Math.hypot(dx,dy)||0.01, ux=dx/d, uy=dy/d;
      const att=d*d/k*(0.25+0.75*l.value/valMax)*0.006;
      a._fx-=ux*att; a._fy-=uy*att; b._fx+=ux*att; b._fy+=uy*att;
    }
    for(const n of nodes){ if(!nodeShown(n)) continue; n._fx-=n.x*GRAV; n._fy-=n.y*GRAV;
      if(n.fixed) continue;
      let d=Math.hypot(n._fx,n._fy)||0.01, lim=Math.min(d,temp);
      n.x+=n._fx/d*lim; n.y+=n._fy/d*lim;
    }
    alpha*=0.975;
  }
  function loop(){ raf=0; if(alpha>0.02){ tick(); drawWeb(); if(view==='web') raf=requestAnimationFrame(loop); } else drawWeb(); }
  function reheat(a){ alpha=Math.max(alpha,a||0.5); if(!raf&&view==='web') raf=requestAnimationFrame(loop); }

  // ── web render ──
  function drawWeb(){
    wx.setTransform(DPR,0,0,DPR,0,0); wx.clearRect(0,0,W,H);
    wx.save(); wx.translate(W/2+tf.x,H/2+tf.y); wx.scale(tf.k,tf.k);
    const focus=hoverN||selN, hi=focus?neighbors(focus):null;
    wx.lineCap='round';
    const edge=css('--gold');
    for(const l of links){ if(!linkShown(l)) continue;
      const on=hi?(hi.has(l.s.id)&&hi.has(l.t.id)&&(l.s===focus||l.t===focus)):false;
      if(hi&&!on) continue;
      wx.beginPath(); wx.moveTo(l.s.x,l.s.y); wx.lineTo(l.t.x,l.t.y);
      wx.lineWidth=wScale(l.value)/Math.sqrt(tf.k);
      wx.strokeStyle=edge; wx.globalAlpha=on?0.9:0.16; wx.stroke();
    }
    wx.globalAlpha=1;
    const panelC=css('--panel'), inkC=css('--ink');
    for(const n of nodes){ if(!nodeShown(n)) continue;
      const dim=hi?!hi.has(n.id):false, isF=n===focus;
      wx.globalAlpha=dim?0.16:1;
      wx.beginPath(); wx.arc(n.x,n.y,n.r,0,6.2832); wx.fillStyle=groupColor(n.group); wx.fill();
      wx.lineWidth=(isF?2.6:1.2)/tf.k; wx.strokeStyle=isF?css('--gold'):panelC; wx.stroke();
      const show=n.r>12||isF||(hi&&hi.has(n.id))||tf.k>1.35;
      if(show&&!dim){ const fs=Math.max(10,Math.min(15,8+n.r*0.34))/tf.k;
        wx.font=`${isF?700:600} ${fs}px "Iowan Old Style",Palatino,Georgia,serif`;
        wx.textAlign='center'; wx.textBaseline='top'; const lab=n.id.replace(/^the /,'');
        wx.lineWidth=3/tf.k; wx.strokeStyle=panelC; wx.globalAlpha=0.85; wx.strokeText(lab,n.x,n.y+n.r+2/tf.k);
        wx.fillStyle=inkC; wx.globalAlpha=1; wx.fillText(lab,n.x,n.y+n.r+2/tf.k);
      }
      wx.globalAlpha=1;
    }
    wx.restore();
  }
  function webNodeAt(px,py){ const x=(px-W/2-tf.x)/tf.k, y=(py-H/2-tf.y)/tf.k; let best=null,bd=1e9;
    for(const n of nodes){ if(!nodeShown(n)) continue; const dx=n.x-x,dy=n.y-y,d=dx*dx+dy*dy,rr=(n.r+4)*(n.r+4);
      if(d<rr&&d<bd){bd=d;best=n;} } return best; }

  // pan / zoom / drag
  let dragNode=null, panStart=null, downXY=null, moved=false;
  web.addEventListener('pointerdown',e=>{ if(e.button) return; const r=web.getBoundingClientRect();
    const px=e.clientX-r.left, py=e.clientY-r.top; downXY=[e.clientX,e.clientY]; moved=false;
    const n=webNodeAt(px,py);
    if(n){ dragNode=n; n.fixed=true; web.setPointerCapture(e.pointerId); web.classList.add('drag'); reheat(0.3); }
    else { panStart=[px,py,tf.x,tf.y]; web.setPointerCapture(e.pointerId); web.classList.add('drag'); }
  });
  web.addEventListener('pointermove',e=>{ const r=web.getBoundingClientRect();
    const px=e.clientX-r.left, py=e.clientY-r.top;
    if(dragNode){ moved=true; dragNode.x=(px-W/2-tf.x)/tf.k; dragNode.y=(py-H/2-tf.y)/tf.k; tip.style.opacity=0; reheat(0.25); return; }
    if(panStart){ moved=true; tf.x=panStart[2]+(px-panStart[0]); tf.y=panStart[3]+(py-panStart[1]); drawWeb(); return; }
    const n=webNodeAt(px,py); web.style.cursor=n?'pointer':'grab';
    if(n!==hoverN){ hoverN=n; if(alpha<=0.02) drawWeb(); }
    if(n){ tip.style.opacity=1; tip.style.left=Math.min(px+14,W-244)+'px'; tip.style.top=(py+14)+'px';
      tip.innerHTML=`<div class="tn">${n.id}</div><div class="tm">${groupLabel(n.group)} · ${n.n} mentions · ${n.deg} links</div>`;
    } else tip.style.opacity=0;
  });
  function endWeb(e){ if(dragNode){ dragNode.fixed=false; dragNode=null; } panStart=null; web.classList.remove('drag'); }
  web.addEventListener('pointerup',endWeb); web.addEventListener('pointercancel',endWeb);
  web.addEventListener('mouseleave',()=>{ hoverN=null; tip.style.opacity=0; if(alpha<=0.02) drawWeb(); });
  web.addEventListener('click',e=>{ if(downXY){ const dx=e.clientX-downXY[0],dy=e.clientY-downXY[1]; if(moved||dx*dx+dy*dy>25) return; }
    const r=web.getBoundingClientRect(); const n=webNodeAt(e.clientX-r.left,e.clientY-r.top);
    if(n) select(n); else closeDetail();
  });
  web.addEventListener('wheel',e=>{ e.preventDefault(); const r=web.getBoundingClientRect();
    const px=e.clientX-r.left, py=e.clientY-r.top; const f=Math.exp(-e.deltaY*0.0012);
    const nk=Math.max(0.3,Math.min(4,tf.k*f)); const wxs=(px-W/2-tf.x)/tf.k, wys=(py-H/2-tf.y)/tf.k;
    tf.x=px-W/2-wxs*nk; tf.y=py-H/2-wys*nk; tf.k=nk; drawWeb();
  },{passive:false});

  // ── matrix render ──
  const CELL=17, PAD=6, ROWLAB=158, COLLAB=150; let order=[], mHover=null;
  const hex2rgb=h=>{ h=h.trim(); if(h[0]==='#'){ if(h.length===4) h='#'+h[1]+h[1]+h[2]+h[2]+h[3]+h[3]; return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]; }
    const m=h.match(/\d+/g)||[150,150,150]; return [+m[0],+m[1],+m[2]]; };
  let _g=[189,131,34], _p=[249,242,226];
  function cellColor(v){ if(v<threshold) return null; const t=Math.sqrt(v/valMax), a=0.14+t*0.86;
    return `rgb(${Math.round(_p[0]+(_g[0]-_p[0])*a)},${Math.round(_p[1]+(_g[1]-_p[1])*a)},${Math.round(_p[2]+(_g[2]-_p[2])*a)})`; }
  function matrixOrder(){ const gr={people:0,orgs:1,places:2}; let ns=nodes.filter(nodeShown);
    const m=document.getElementById('netOrder').value;
    if(m==='group') ns.sort((a,b)=>gr[a.group]-gr[b.group]||b.n-a.n);
    else if(m==='mentions') ns.sort((a,b)=>b.n-a.n);
    else ns.sort((a,b)=>a.id.replace(/^the /,'').localeCompare(b.id.replace(/^the /,'')));
    order=ns; }
  function drawMatrix(){ matrixOrder(); _g=hex2rgb(css('--gold')); _p=hex2rgb(css('--panel'));
    const N=order.length, w=ROWLAB+N*CELL+PAD, h=COLLAB+N*CELL+PAD;
    mx.style.width=w+'px'; mx.style.height=h+'px'; mx.width=w*DPR; mx.height=h*DPR;
    mc.setTransform(DPR,0,0,DPR,0,0); mc.clearRect(0,0,w,h);
    const X0=ROWLAB,Y0=COLLAB, hr=mHover?mHover[0]:-1, hcc=mHover?mHover[1]:-1;
    if(mHover){ mc.fillStyle=css('--border2'); mc.globalAlpha=.5; mc.fillRect(0,Y0+hr*CELL,w,CELL); mc.fillRect(X0+hcc*CELL,0,CELL,h); mc.globalAlpha=1; }
    for(let i=0;i<N;i++) for(let j=0;j<N;j++){
      if(i===j){ mc.fillStyle=groupColor(order[i].group); mc.globalAlpha=.9; mc.fillRect(X0+j*CELL+3,Y0+i*CELL+3,CELL-6,CELL-6); mc.globalAlpha=1; continue; }
      const v=pairVal(order[i].id,order[j].id), col=cellColor(v); if(!col) continue;
      mc.fillStyle=col; mc.fillRect(X0+j*CELL+0.5,Y0+i*CELL+0.5,CELL-1,CELL-1);
    }
    if(document.getElementById('netOrder').value==='group'){ mc.strokeStyle=css('--border'); mc.lineWidth=1.3; mc.globalAlpha=.5;
      let prev=order[0]&&order[0].group;
      for(let i=1;i<N;i++){ if(order[i].group!==prev){ const p=Y0+i*CELL, q=X0+i*CELL;
        mc.beginPath();mc.moveTo(X0,p);mc.lineTo(X0+N*CELL,p);mc.stroke();
        mc.beginPath();mc.moveTo(q,Y0);mc.lineTo(q,Y0+N*CELL);mc.stroke(); prev=order[i].group; } } mc.globalAlpha=1; }
    mc.textBaseline='middle';
    for(let i=0;i<N;i++){ const nm=order[i].id.replace(/^the /,''), col=groupColor(order[i].group), emph=(i===hr||i===hcc);
      mc.font=`${emph?700:500} 11px ui-sans-serif,system-ui,sans-serif`;
      mc.textAlign='right'; mc.fillStyle=emph?css('--ink'):col; mc.fillText(nm,ROWLAB-8,Y0+i*CELL+CELL/2,ROWLAB-12);
      mc.save(); mc.translate(X0+i*CELL+CELL/2,COLLAB-8); mc.rotate(-Math.PI/2); mc.textAlign='left';
      mc.fillStyle=emph?css('--ink'):col; mc.fillText(nm,0,0,COLLAB-12); mc.restore();
    }
  }
  function mxCellAt(px,py){ const N=order.length, i=Math.floor((py-COLLAB)/CELL), j=Math.floor((px-ROWLAB)/CELL);
    if(i<0||j<0||i>=N||j>=N) return null; return [i,j]; }
  mx.addEventListener('mousemove',e=>{ const r=mx.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top;
    const c=mxCellAt(px,py); if((c&&(!mHover||c[0]!==mHover[0]||c[1]!==mHover[1]))||(!c!==!mHover)){ mHover=c; drawMatrix(); }
    const sr=stage.getBoundingClientRect();
    if(c){ const A=order[c[0]],B=order[c[1]]; tip.style.opacity=1;
      tip.style.left=Math.min(e.clientX-sr.left+14,stage.clientWidth-244)+'px'; tip.style.top=(e.clientY-sr.top+14)+'px';
      tip.innerHTML=c[0]===c[1]?`<div class="tn">${A.id}</div><div class="tm">${A.n} mentions</div>`
        :`<div class="tn">${A.id} × ${B.id}</div><div class="tm">${pairVal(A.id,B.id)} shared article${pairVal(A.id,B.id)===1?'':'s'}</div>`;
      mx.style.cursor='pointer';
    } else tip.style.opacity=0;
  });
  mx.addEventListener('mouseleave',()=>{ mHover=null; tip.style.opacity=0; drawMatrix(); });
  mx.addEventListener('click',e=>{ const r=mx.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top;
    const c=mxCellAt(px,py); if(c){ select(order[c[0]]); return; }
    if(px<ROWLAB&&py>COLLAB){ const i=Math.floor((py-COLLAB)/CELL); if(order[i]) select(order[i]); }
  });

  // ── dossier ──
  const detail=document.getElementById('netDetail');
  function select(n){ selN=n;
    document.getElementById('ndKind').textContent=groupLabel(n.group);
    document.getElementById('ndKind').style.color=groupColor(n.group);
    document.getElementById('ndName').textContent=n.id;
    document.getElementById('ndMent').textContent=n.n;
    const tp=document.getElementById('ndTone'); const t=n.tone||0, a=Math.min(1,Math.abs(t)*2.2), pos=t>=0;
    const tg=pos?[192,57,47]:[42,107,208], mix=x=>Math.round(155+(x-155)*a);
    tp.style.background=`rgb(${mix(tg[0])},${mix(tg[1])},${mix(tg[2])})`;
    tp.textContent=(t>0?'+':'')+t.toFixed(2);
    document.getElementById('ndLinks').textContent=n.deg;
    const S=n.series||[], smax=Math.max(1,...S);
    document.getElementById('ndTL').innerHTML=S.map((v,i)=>`<div class="bar" style="height:${(v/smax*100).toFixed(0)}%" title="week of ${WEEKS[i]||''}: ${v} article${v===1?'':'s'}"></div>`).join('');
    document.getElementById('ndTSpan').textContent=`peak ${smax}/wk`;
    document.getElementById('ndTL0').textContent=WEEKS.length?fmtWk(WEEKS[0]):'';
    document.getElementById('ndTL1').textContent=WEEKS.length?fmtWk(WEEKS[WEEKS.length-1]):'';
    const c=n.countries||{}, tot=(c.EE||0)+(c.LV||0)+(c.LT||0)||1;
    document.getElementById('ndGeo').innerHTML=
      `<i class="ee" style="width:${(c.EE||0)/tot*100}%" title="Estonia ${c.EE||0}"></i>`+
      `<i class="lv" style="width:${(c.LV||0)/tot*100}%" title="Latvia ${c.LV||0}"></i>`+
      `<i class="lt" style="width:${(c.LT||0)/tot*100}%" title="Lithuania ${c.LT||0}"></i>`;
    document.getElementById('ndGeoKeys').innerHTML=
      `<span><i class="ee"></i>EE<b>${c.EE||0}</b></span><span><i class="lv"></i>LV<b>${c.LV||0}</b></span><span><i class="lt"></i>LT<b>${c.LT||0}</b></span>`;
    const nb=[]; links.forEach(l=>{ if(l.s===n) nb.push([l.t,l.value]); else if(l.t===n) nb.push([l.s,l.value]); });
    nb.sort((a,b)=>b[1]-a[1]); const mxv=nb.length?nb[0][1]:1;
    document.getElementById('ndNeigh').innerHTML = nb.map(([m,v])=>
      `<div class="nrow" data-id="${m.id.replace(/"/g,'&quot;')}"><span class="ndot" style="background:${groupColor(m.group)}"></span>`+
      `<div style="flex:1;min-width:0"><div class="nm">${m.id}</div><div class="nbar" style="width:${Math.max(6,v/mxv*100)}%"></div></div>`+
      `<span class="nv">${v}</span></div>`).join('') || '<div style="color:var(--faint);font-size:12px">No links at this threshold.</div>';
    document.querySelectorAll('#ndNeigh .nrow').forEach(row=>{ row.onclick=()=>{ const m=byId.get(row.dataset.id); if(m){ select(m); focusOn(m); } }; });
    const arts=(n.arts||[]).map(i=>D.points[i]).filter(Boolean).sort((a,b)=>b.d.localeCompare(a.d));
    document.getElementById('ndArtN').textContent=`(${arts.length})`;
    document.getElementById('ndArts').innerHTML=arts.map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`).join('')||'<li><a style="color:var(--faint)">No articles.</a></li>';
    detail.classList.add('open');
    if(view==='web'){ if(alpha<=0.02) drawWeb(); } else drawMatrix();
  }
  function closeDetail(){ selN=null; detail.classList.remove('open'); if(view==='web'){ if(alpha<=0.02) drawWeb(); } else drawMatrix(); }
  function focusOn(n){ if(view!=='web') return; const k=Math.max(1,tf.k); tf.x=-n.x*k; tf.y=-n.y*k; tf.k=k; drawWeb(); }
  document.getElementById('ndClose').onclick=closeDetail;

  // ── controls ──
  function setView(v){ view=v; panel.setAttribute('data-view',v);
    document.getElementById('v-web').classList.toggle('on',v==='web');
    document.getElementById('v-matrix').classList.toggle('on',v==='matrix');
    document.getElementById('v-web').setAttribute('aria-selected',v==='web');
    document.getElementById('v-matrix').setAttribute('aria-selected',v==='matrix');
    tip.style.opacity=0;
    if(v==='web'){ reheat(0.15); } else drawMatrix();
  }
  document.getElementById('v-web').onclick=()=>setView('web');
  document.getElementById('v-matrix').onclick=()=>setView('matrix');
  document.querySelectorAll('.nleg').forEach(b=>{ b.onclick=()=>{ const g=b.dataset.g; active[g]=!active[g];
    b.classList.toggle('off',!active[g]); if(view==='web') reheat(0.4); else { mHover=null; drawMatrix(); } }; });
  const thr=document.getElementById('netThr'), thrv=document.getElementById('netThrV');
  thr.oninput=()=>{ threshold=+thr.value; thrv.textContent=thr.value;
    if(view==='web') reheat(0.35); else drawMatrix(); };
  document.getElementById('netOrder').onchange=()=>{ mHover=null; drawMatrix(); };
  document.getElementById('netReset').onclick=()=>{ tf={x:0,y:0,k:1}; closeDetail(); reheat(0.3); };

  // ── size + lazy init (section starts collapsed → wait for real size) ──
  function resize(){ const r=stage.getBoundingClientRect(); if(!r.width||!r.height) return false;
    W=r.width; H=r.height; DPR=Math.max(1,window.devicePixelRatio||1);
    web.width=W*DPR; web.height=H*DPR; return true; }
  const ro=new ResizeObserver(()=>{ if(resize()){ if(!inited){ inited=true; seed(); reheat(1.0); } else if(view==='web') drawWeb(); } });
  ro.observe(stage);
  window.addEventListener('resize',()=>{ if(resize()&&view==='web') drawWeb(); });
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>{ if(view==='web') drawWeb(); else drawMatrix(); });
})();
