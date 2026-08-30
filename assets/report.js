
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
