#!/usr/bin/env python
"""
Semantic-embeddings sections of the Baltics Monitor.

Semantic map built on sentence embeddings, shown next to (not instead of) the
keyword-bucket themes. It's spliced into the MAIN report after those sections.
Pass --no-embed to skip it (e.g. a fast local build with no fastembed/umap).

NOTE: there is deliberately NO unsupervised clustering here. Discovered clusters
on ~200 short docs were vague and drifted run-to-run, so the map is coloured by
the STABLE keyword-bucket theme instead. Embeddings are used for what they're
actually good at: proximity, isolation, and neighbour agreement.

What it does, for TWO text scopes side by side:
  * "content"  — headline + full article body
  * "headline" — headline only
For each scope:
  1. Embed every article into a semantic vector (meaning, not word overlap),
     cached by URL so each run only embeds NEW articles (incremental).
  2. UMAP -> 2D for the map (metric="cosine": compare directions, not distance,
     because in high dimensions Euclidean distance collapses). Points are
     coloured by their keyword-bucket theme.
  3. Cosine nearest-neighbours per article -> two lists:
     * anomalies   — the most ISOLATED stories (far from everything else)
     * mislabels   — stories whose semantic neighbours mostly carry a DIFFERENT
                     regex theme than the buckets assigned (a bucket-audit).

Model: configurable. Default is a runnable, CI-light BGE base via fastembed
(ONNX, no torch). BAAI/bge-large-en-v1.5 (the workshop's recommended, higher
MTEB score) is a one-flag swap: --embed-model BAAI/bge-large-en-v1.5 (installs
sentence-transformers/torch on first use if fastembed lacks the model).

Entry points:
  * render.render() calls build_report_fragment() to splice these sections in
  * or standalone:  venv/Scripts/python.exe embeddings_analysis.py
"""
import os
import re
import numpy as np

# The pipeline was split into focused modules; pull what we need from each.
import config
import assets
import pipeline

CACHE_DIR = os.path.join(config.OUT_DIR, "emb_cache")
# Runnable, CI-light default (fastembed/ONNX). bge-large is the higher-quality pick.
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
SCOPES = ("content", "headline")


# ── embedding (incremental, cached by URL) ──────────────────────────────────
def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _encoder(model):
    """Return (encode_fn, backend_name). Prefer fastembed (ONNX, light)."""
    try:
        from fastembed import TextEmbedding
        te = TextEmbedding(model_name=model)          # raises if model unsupported

        def enc(texts):
            return np.array(list(te.embed(list(texts))))
        return enc, "fastembed"
    except Exception as e:
        from sentence_transformers import SentenceTransformer
        print(f"[embed] fastembed unavailable for {model} ({e.__class__.__name__}); "
              f"using sentence-transformers")
        st = SentenceTransformer(model)

        def enc(texts):
            return np.asarray(st.encode(list(texts), show_progress_bar=False))
        return enc, "sentence-transformers"


def get_embeddings(urls, texts, scope, model):
    """Embed texts; cache vectors by URL so only NEW urls are re-embedded."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{scope}__{_slug(model)}.npz")
    cache = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        cache = {u: v for u, v in zip(z["urls"].tolist(), z["vecs"])}
    missing = [(u, t) for u, t in zip(urls, texts) if u not in cache]
    if missing:
        enc, backend = _encoder(model)
        vecs = _normalize(enc([t for _, t in missing]))
        for (u, _), v in zip(missing, vecs):
            cache[u] = v
        us = np.array(list(cache.keys()), dtype=object)
        vs = np.array([cache[u] for u in us], dtype=np.float32)
        np.savez(path, urls=us, vecs=vs)
        print(f"[embed:{scope}] +{len(missing)} new via {backend} ({model}); {len(cache)} cached")
    else:
        print(f"[embed:{scope}] all {len(urls)} already cached ({model})")
    return np.array([cache[u] for u in urls], dtype=np.float32)


# ── reduce + cluster ────────────────────────────────────────────────────────
def _umap(emb, n_components, min_dist):
    from umap import UMAP
    n = len(emb)
    n_neighbors = max(2, min(15, n - 1))
    nc = max(2, min(n_components, n - 2))
    return UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=nc,
                metric="cosine", random_state=42).fit_transform(emb)


def _neighbors(emb, k=5):
    """Cosine nearest neighbours (emb is unit-normalised, so dot == cosine)."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    idx = np.argsort(-sim, axis=1)[:, :k]
    iso = 1.0 - sim.max(axis=1)                        # isolation: far from everything
    return idx, iso


# ── build one scope's payload ───────────────────────────────────────────────
def build_view(df, scope, model, theme_names):
    urls = df["url"].tolist()
    if scope == "content":
        texts = (df["headline"].fillna("") + ". " + df["body"].fillna("")).tolist()
    else:
        texts = df["headline"].fillna("").tolist()

    emb = get_embeddings(urls, texts, scope, model)
    xy = _umap(emb, n_components=2, min_dist=0.1)      # 2-D map only; no clustering
    nn, iso = _neighbors(emb, 5)

    themes = df["theme"].tolist()                     # regex primary theme id per article
    pts = []
    for i, r in enumerate(df.itertuples()):
        neigh = [int(j) for j in nn[i]]
        # mislabel audit: do the semantic neighbours mostly carry a different regex theme?
        nb_themes = [themes[j] for j in neigh]
        same = nb_themes.count(themes[i])
        mis = bool(same <= 1 and len(nb_themes) >= 3)
        pts.append({
            "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
            "h": r.headline, "s": config.SRC_LABEL[r.source], "d": str(r.scrape_date), "u": r.url,
            "t": int(themes[i]), "nn": neigh, "iso": round(float(iso[i]), 3), "mis": mis,
        })
    return {
        "scope": scope, "model": model, "points": pts,
        "nMis": int(sum(p["mis"] for p in pts)),
        "themeNames": theme_names,
    }


# ── HTML / CSS / JS fragments (spliced into the main report) ─────────────────
EMB_CSS = """
/* embeddings prototype (opt-in, --embed) */
.emb-note{margin:6px 0 16px;padding:10px 14px;border:2px dashed var(--gold);border-radius:6px;
  background:color-mix(in srgb,var(--gold) 12%,transparent);font-size:13px;color:var(--muted);}
.emb-tabs{display:flex;gap:8px;margin:2px 0 10px;flex-wrap:wrap;}
.emb-tab{padding:7px 14px;border:2px solid var(--gold);border-radius:6px;cursor:pointer;
  background:var(--panel);color:var(--ink);font-size:13px;font-weight:600;}
.emb-tab.on{background:var(--gold);color:#2b140f;}
.emb-wrap{position:relative;}
.emb-canvas{display:block;width:100%;height:520px;touch-action:none;}
.emb-legend{position:absolute;top:12px;right:12px;max-width:250px;max-height:480px;overflow:auto;
  display:flex;flex-direction:column;gap:2px;background:color-mix(in srgb,var(--panel) 90%,transparent);
  backdrop-filter:blur(6px);border:1px solid var(--border);border-radius:10px;padding:8px;}
.emb-leg{display:flex;align-items:flex-start;gap:8px;padding:4px 6px;border-radius:6px;cursor:pointer;
  font-size:11.5px;color:var(--muted);user-select:none;}
.emb-leg:hover{background:var(--border2);}.emb-leg.off{opacity:.36;}
.emb-leg .sw{width:10px;height:10px;border-radius:50%;flex:0 0 auto;margin-top:2px;}
.emb-leg .lc{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--faint);}
.emb-tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;z-index:6;
  background:var(--ink);color:var(--bg);padding:8px 11px;border-radius:8px;max-width:300px;
  font-size:12.5px;line-height:1.4;box-shadow:0 6px 20px rgba(0,0,0,.3);}
.emb-tip b{display:block;margin-bottom:3px;}.emb-tip .m{opacity:.72;font-size:11px;}
.emb-anom{padding:14px 20px 18px;}
.emb-anom h4{margin:0 0 8px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);}
.emb-anom ul{list-style:none;margin:0 0 14px;padding:0;max-height:320px;overflow:auto;}
.emb-anom li{padding:7px 0;border-top:1px solid var(--border2);font-size:13px;}
.emb-anom li:first-child{border-top:0;}
.emb-anom a{color:var(--ink);text-decoration:none;display:block;}.emb-anom a:hover{color:var(--accent);}
.emb-anom .src{display:block;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);margin-top:2px;}
.emb-badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:3px;margin-right:6px;
  background:var(--accent);color:var(--bg);letter-spacing:.04em;}
"""

_EMB_SECTION = """
    <p class="section-h"><svg class="sec-sign" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="7" r="2.2"/><circle cx="14" cy="6" r="2.2"/><circle cx="10" cy="14" r="2.2"/><path d="M7.8 8.2 12 12M12.2 7.2 11 12"/></svg> The semantic map — near = similar meaning (embeddings &rarr; UMAP)</p>
    <section class="panel">
      <div class="panel-h">
        <h2>Semantic map &amp; anomalies</h2>
        <span class="hint mono">near = similar MEANING (not just shared words) · colour = keyword theme</span>
      </div>
      <div style="padding:14px 20px 0">
        <p class="emb-note">A second view of the same articles, laid out by <b>meaning</b> rather than word overlap, and coloured by the same keyword themes as the map above. Two scopes: <b>content</b> embeds headline + body, <b>headline</b> embeds titles only. Below the map: the most <i>isolated</i> stories, and a bucket audit where an article's nearest neighbours mostly carry a different theme.</p>
        <div class="emb-tabs" id="embTabs"></div>
      </div>
      <div class="emb-wrap">
        <canvas class="emb-canvas" id="embCanvas"></canvas>
        <div class="emb-legend" id="embLegend"></div>
        <div class="emb-tip" id="embTip"></div>
      </div>
      <div class="emb-anom">
        <h4 id="embAnomH">Anomalies — the most isolated stories</h4>
        <ul id="embAnom"></ul>
        <h4 id="embMisH">Possible mislabels — semantic neighbours disagree with the regex theme</h4>
        <ul id="embMis"></ul>
      </div>
    </section>
"""

_EMB_SCRIPT = r"""
(function(){
  const EV = __EMB_DATA__;                       // {content:{...}, headline:{...}, colors, colorsDark}
  const scopes = Object.keys(EV).filter(k=>EV[k] && EV[k].points);
  if(!scopes.length) return;
  const TN = (EV[scopes[0]].themeNames)||[];
  const ANOM_N = 20;                             // how many most-isolated stories to list
  // colour = keyword-bucket theme (same palette as the main map), light/dark aware.
  const isDark=()=>{const t=document.documentElement.getAttribute('data-theme');
    return t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme:dark)').matches);};
  let COL = isDark()?EV.colorsDark:EV.colors;
  let cur = scopes[0];
  const hidden = new Set();                       // hidden THEME ids
  const cvs=document.getElementById('embCanvas'), ctx=cvs.getContext('2d');
  const tip=document.getElementById('embTip'), wrap=cvs.parentElement;
  let P=[], hover=-1, minX,maxX,minY,maxY;

  document.getElementById('embTabs').innerHTML = scopes.map(s=>{
    const v=EV[s]; const lab = s==='content'?'Content (headline + body)':'Headline only';
    return `<div class="emb-tab${s===cur?' on':''}" data-s="${s}">${lab} · ${v.points.length} stories</div>`;
  }).join('');
  document.querySelectorAll('#embTabs .emb-tab').forEach(t=>{
    t.onclick=()=>{cur=t.dataset.s; hidden.clear();
      document.querySelectorAll('#embTabs .emb-tab').forEach(x=>x.classList.toggle('on',x.dataset.s===cur));
      load(); };
  });

  function load(){
    const v=EV[cur];
    const xs=v.points.map(p=>p.x), ys=v.points.map(p=>p.y);
    minX=Math.min(...xs);maxX=Math.max(...xs);minY=Math.min(...ys);maxY=Math.max(...ys);
    const px=(maxX-minX)*0.06||1, py=(maxY-minY)*0.06||1; minX-=px;maxX+=px;minY-=py;maxY+=py;
    // legend = the keyword themes actually present in this scope, by count
    const cnt={}; v.points.forEach(p=>{cnt[p.t]=(cnt[p.t]||0)+1;});
    const ids=Object.keys(cnt).map(Number).sort((a,b)=>cnt[b]-cnt[a]);
    document.getElementById('embLegend').innerHTML = ids.map(id=>
      `<div class="emb-leg${hidden.has(id)?' off':''}" data-c="${id}" tabindex="0">`
      +`<span class="sw" style="background:${COL[id]}"></span>`
      +`<span>${TN[id]||('#'+id)}</span><span class="lc">${cnt[id]}</span></div>`
    ).join('');
    document.querySelectorAll('#embLegend .emb-leg').forEach(el=>{
      el.onclick=()=>{const id=+el.dataset.c; hidden.has(id)?hidden.delete(id):hidden.add(id);
        el.classList.toggle('off'); draw();};
    });
    // anomalies (most isolated) + mislabels (neighbours disagree with the bucket)
    const li = arr => arr.map(p=>{
      const also = TN[p.t]?`<span class="emb-badge">${TN[p.t]}</span>`:'';
      return `<li><a href="${p.u}" target="_blank" rel="noopener">${also}${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`;
    }).join('');
    const anom=v.points.slice().sort((a,b)=>b.iso-a.iso).slice(0,ANOM_N);
    const miss=v.points.filter(p=>p.mis).sort((a,b)=>b.iso-a.iso);
    document.getElementById('embAnomH').textContent=`Anomalies — the ${anom.length} most isolated stories`;
    document.getElementById('embAnom').innerHTML = anom.length?li(anom):'<li>none</li>';
    document.getElementById('embMisH').textContent=`Possible mislabels — ${miss.length} where semantic neighbours disagree with the regex theme`;
    document.getElementById('embMis').innerHTML = miss.length?li(miss):'<li>none</li>';
    layout(); draw();
  }
  function layout(){
    const dpr=devicePixelRatio||1, w=wrap.clientWidth, h=cvs.clientHeight, pad=30;
    cvs.width=w*dpr; cvs.height=h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
    P=EV[cur].points.map(p=>({...p,
      sx:pad+(p.x-minX)/(maxX-minX)*(w-2*pad),
      sy:pad+(maxY-p.y)/(maxY-minY)*(h-2*pad)}));
  }
  function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
  function draw(){
    const w=wrap.clientWidth,h=cvs.clientHeight; ctx.clearRect(0,0,w,h);
    ctx.strokeStyle=css('--grid');ctx.lineWidth=1;
    for(let i=1;i<6;i++){const gx=i/6*w;ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,h);ctx.stroke();
      const gy=i/6*h;ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke();}
    const ground=css('--panel');
    P.forEach((p,i)=>{ if(hidden.has(p.t))return;
      const r=(i===hover?8:5.5);
      ctx.beginPath();ctx.arc(p.sx,p.sy,r,0,7);
      ctx.fillStyle=COL[p.t]||'#888';ctx.globalAlpha=i===hover?1:.85;ctx.fill();ctx.globalAlpha=1;
      ctx.lineWidth=i===hover?2:1.1;ctx.strokeStyle=ground;ctx.stroke();
    });
  }
  function nearest(mx,my){let b=-1,bd=169;P.forEach((p,i)=>{if(hidden.has(p.t))return;
    const d=(p.sx-mx)**2+(p.sy-my)**2;if(d<bd){bd=d;b=i;}});return b;}
  cvs.addEventListener('mousemove',e=>{const rc=cvs.getBoundingClientRect(),mx=e.clientX-rc.left,my=e.clientY-rc.top;
    const hh=nearest(mx,my); if(hh!==hover){hover=hh;draw();}
    if(hh>=0){const p=P[hh];cvs.style.cursor='pointer';
      tip.innerHTML=`<b>${p.h}</b><span class="m">${p.s} · ${p.d}<br>theme: ${TN[p.t]||'—'} · isolation ${p.iso}${p.mis?'<br>⚠ neighbours mostly a different theme':''}</span>`;
      tip.style.opacity=1; let tx=mx+16,ty=my+16;
      if(tx+tip.offsetWidth>wrap.clientWidth)tx=mx-tip.offsetWidth-16;
      if(ty+tip.offsetHeight>cvs.clientHeight)ty=my-tip.offsetHeight-16;
      tip.style.left=tx+'px';tip.style.top=ty+'px';
    }else{cvs.style.cursor='default';tip.style.opacity=0;}});
  cvs.addEventListener('mouseleave',()=>{hover=-1;tip.style.opacity=0;draw();});
  cvs.addEventListener('click',()=>{if(hover>=0)window.open(P[hover].u,'_blank','noopener');});
  new ResizeObserver(()=>{layout();draw();}).observe(wrap);
  // follow the page's light/dark, like the main map
  const onTheme=()=>{COL=isDark()?EV.colorsDark:EV.colors; load();};
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',onTheme);
  new MutationObserver(onTheme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  load();
})();
"""


def build_report_fragment(df, model=DEFAULT_MODEL, theme_names=None):
    """Return (css, html, script) to splice into the main report. df must carry
    'body' and 'theme' (from topic_analysis.analyse)."""
    import json
    if theme_names is None:
        theme_names = [t[0] for t in config.THEMES] + [config.THEME_OTHER]
    payload = {sc: build_view(df, sc, model, theme_names) for sc in SCOPES}
    # theme palette (same as the main map), so the embed map colours by keyword theme
    payload["colors"] = config.COLORS_LIGHT
    payload["colorsDark"] = config.COLORS_DARK
    data_js = json.dumps(payload, ensure_ascii=False)
    script = _EMB_SCRIPT.replace("__EMB_DATA__", data_js)
    return EMB_CSS, _EMB_SECTION, script


def _standalone(df, model):
    """Standalone self-contained page (for running this module directly)."""
    css, htmlfrag, script = build_report_fragment(df, model)
    inner = (f"<style>{assets.CSS}{css}</style>\n<div class=\"shell\"><main>{htmlfrag}</main></div>\n"
             f"<script>{script}</script>")
    page = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Baltic Monitor — embeddings prototype</title></head>"
            f"<body>{inner}</body></html>")
    os.makedirs(config.OUT_DIR, exist_ok=True)
    out = os.path.join(config.OUT_DIR, "embeddings-prototype.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[embed] wrote {out} ({len(page)//1024} KB)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Embeddings prototype (standalone)")
    ap.add_argument("--embed-model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    df_all = pipeline.load_data()
    df = pipeline.filter_themes(df_all)
    df["body"] = pipeline.fetch_all(df["url"].tolist())
    df, _themes, _kw = pipeline.analyse(df)                 # gives df['theme'] for the audit
    _standalone(df, args.embed_model)


if __name__ == "__main__":
    main()
