#!/usr/bin/env python
"""
Baltics/Nordics Monitor - thematic topic analysis.

Pipeline:
  1. Load the three *-always-updated.csv files (fixing lrt_lt mojibake).
  2. Filter headlines to defence/geopolitics themes
     (Ukraine, Russia, defence, China, war, drones, NATO, ...).
  3. Fetch the full article body for each matched article (robust, skips failures).
  4. TF-IDF vectorise (headline + body).
  5. Dimensionality reduction: TruncatedSVD (LSA) -> t-SNE 2D layout.
  6. Clustering: KMeans on the LSA space, k chosen by silhouette.
  7. Per-cluster keyword labels via mean TF-IDF weight.
  8. Render a self-contained, dependency-free interactive HTML report
     (Canvas topic map) + a data JSON + a clustered CSV.

Run:  venv/Scripts/python.exe topic_analysis.py
Output: nordics-monitor.html  (drop into jule4ka.github.io/content/nordics-monitor/)
"""
import sys, re, json, warnings, datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FILES = {
    "err_ee": "err_ee-always-updated.csv",
    "lrt_lt": "lrt_lt-always-updated.csv",
    "lsm_lv": "lsm_lv-always-updated.csv",
}
SRC_LABEL = {"err_ee": "ERR (Estonia)", "lrt_lt": "LRT (Lithuania)", "lsm_lv": "LSM (Latvia)"}

# Theme filter: defence / geopolitics / Russia-Ukraine-China / war / drones
KEYWORDS = (
    r"\b(ukrain|russ|kremlin|putin|moscow|belarus|defen[cs]e|militar|nato|war\b|"
    r"warfare|drone|china|chinese|beijing|weapon|missile|army|troop|soldier|"
    r"sanction|wagner|zelensky|invasion|espionage|spy|sabotag|cyber|airspace|"
    r"submarine|frigate|warship|warplane|hybrid|mobilis|mobiliz|conscript|"
    r"artillery|shelling|front[\s-]?line|refugee|border)\w*"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NordicsMonitor/1.0; research)"}

# Site boilerplate to drop from article bodies (newsletter / correction / social prompts).
BOILERPLATE = re.compile(
    r"(suggested correction|select text and press|send a suggested|to the editor|"
    r"follow err|follow lrt|follow lsm|sign up|newsletter|subscribe|cookie|"
    r"all rights reserved|advertisement|read more|related articles|share this|"
    r"download the|on facebook|on twitter|©)", re.I)

# Extra domain stopwords (site chrome + geography that would dominate every cluster).
EXTRA_STOP = {
    "text", "press", "select", "suggested", "correction", "editor", "send",
    "follow", "facebook", "twitter", "newsletter", "subscribe", "sign", "cookies",
    "err", "lrt", "lsm", "news", "said", "says", "according", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "year", "years",
    "wants", "make", "did",
}

# Categorical palette for the clusters (legible on both light and dark grounds).
CLUSTER_COLORS = ["#4f8cff", "#2bb8a3", "#e0a13a", "#a781f5",
                  "#f0688a", "#6fbf3f", "#ff7a45", "#00b8d9", "#c026d3"]


def fix_mojibake(s):
    """Repair double-encoded UTF-8 (lrt_lt export)."""
    if not isinstance(s, str):
        return ""
    if any(m in s for m in ["Ã", "â€", "Å", "Â", " Â"]):
        try:
            return s.encode("cp1252", "ignore").decode("utf-8", "ignore")
        except Exception:
            return s
    return s


def load_data():
    frames = []
    for src, path in FILES.items():
        d = pd.read_csv(path, encoding="utf-8")
        d["source"] = src
        d["headline"] = d["headline"].map(fix_mojibake).str.replace("�", " ", regex=False).str.strip()
        frames.append(d[["url", "headline", "scrape_date", "source"]])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["url", "headline"]).drop_duplicates("url").reset_index(drop=True)
    return df


def filter_themes(df):
    mask = df["headline"].str.contains(KEYWORDS, case=False, na=False, regex=True)
    out = df[mask].reset_index(drop=True)
    print(f"[filter] {mask.sum()} / {len(df)} articles match the defence/geopolitics themes")
    return out


def fetch_body(url):
    """Best-effort article-body extraction: join the meaningful <p> tags."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup
        ps = [p.get_text(" ", strip=True) for p in root.find_all("p")]
        ps = [p for p in ps if len(p) > 40 and not BOILERPLATE.search(p)]
        return " ".join(ps)[:6000]
    except Exception:
        return ""


def fetch_all(urls):
    print(f"[fetch] downloading {len(urls)} article bodies ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        bodies = list(ex.map(fetch_body, urls))
    ok = sum(1 for b in bodies if b)
    print(f"[fetch] got body text for {ok}/{len(urls)} articles")
    return bodies


def analyse(df):
    corpus = (df["headline"].fillna("") + ". " + df["body"].fillna("")).tolist()
    stop = list(TfidfVectorizer(stop_words="english").get_stop_words() | EXTRA_STOP)
    vec = TfidfVectorizer(
        stop_words=stop, ngram_range=(1, 2),
        min_df=2, max_df=0.6, max_features=5000, sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(corpus)
    terms = np.array(vec.get_feature_names_out())

    n = X.shape[0]
    n_comp = min(50, n - 1, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)
    print(f"[svd] reduced TF-IDF ({X.shape[1]} terms) -> {n_comp} LSA dims "
          f"({svd.explained_variance_ratio_.sum():.0%} variance)")

    # choose k by silhouette, but require each cluster to hold >=3 articles
    best_k, best_s, best_labels = 2, -1, None
    for k in range(4, min(7, n // 8) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lab = km.fit_predict(Z)
        counts = np.bincount(lab)
        if counts.min() < 3:
            continue
        try:
            s = silhouette_score(Z, lab)
        except Exception:
            s = -1
        if s > best_s:
            best_k, best_s, best_labels = k, s, lab
    if best_labels is None:  # fallback if the >=3 constraint was never satisfiable
        km = KMeans(n_clusters=5, n_init=10, random_state=42)
        best_labels = km.fit_predict(Z)
        best_k, best_s = 5, silhouette_score(Z, best_labels)
    print(f"[cluster] KMeans k={best_k} (silhouette={best_s:.3f})")
    df["cluster"] = best_labels

    # 2D layout via t-SNE on LSA space
    perp = max(5, min(30, n // 3))
    xy = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=42).fit_transform(Z)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]

    # cluster keyword labels via mean TF-IDF per cluster
    Xa = X.toarray()
    labels = {}
    for c in sorted(df["cluster"].unique()):
        idx = np.where(best_labels == c)[0]
        mean_w = Xa[idx].mean(axis=0)
        top = terms[mean_w.argsort()[::-1][:6]]
        labels[int(c)] = ", ".join(top)
        print(f"   cluster {c} ({len(idx)} articles): {labels[int(c)]}")
    return df, labels


def build_payload(df, labels):
    clusters = [{"id": int(c), "label": labels[c],
                 "count": int((df["cluster"] == c).sum())}
                for c in sorted(df["cluster"].unique())]
    points = [{
        "x": round(float(r.x), 3), "y": round(float(r.y), 3), "c": int(r.cluster),
        "h": r.headline, "s": SRC_LABEL[r.source], "d": str(r.scrape_date), "u": r.url,
    } for r in df.itertuples()]
    return {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sources": len(df["source"].unique()),
        "clusters": clusters, "points": points, "colors": CLUSTER_COLORS,
    }


CSS = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#141821; --muted:#5c6472; --faint:#8a92a3;
  --border:#e3e6ec; --border2:#eef0f4; --accent:#c47d2b; --grid:#eceef2;
  --shadow:0 1px 2px rgba(20,24,33,.04),0 8px 24px rgba(20,24,33,.05);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0b0e14; --panel:#12161f; --ink:#e7eaf0; --muted:#9aa3b2; --faint:#6b7382;
    --border:#212734; --border2:#1a1f2a; --accent:#e6a13c; --grid:#1a1f2a;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme=light]{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#141821; --muted:#5c6472; --faint:#8a92a3;
  --border:#e3e6ec; --border2:#eef0f4; --accent:#c47d2b; --grid:#eceef2;
  --shadow:0 1px 2px rgba(20,24,33,.04),0 8px 24px rgba(20,24,33,.05);
}
:root[data-theme=dark]{
  --bg:#0b0e14; --panel:#12161f; --ink:#e7eaf0; --muted:#9aa3b2; --faint:#6b7382;
  --border:#212734; --border2:#1a1f2a; --accent:#e6a13c; --grid:#1a1f2a;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased;}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Code","Cascadia Mono",Consolas,monospace;}
.shell{max-width:1180px;margin:0 auto;padding:0 22px;}
header{padding:44px 0 26px;border-bottom:1px solid var(--border);}
.eyebrow{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);
  font-weight:600;margin:0 0 12px;display:flex;align-items:center;gap:10px;}
.eyebrow::before{content:"";width:26px;height:2px;background:var(--accent);display:inline-block;}
h1{font-size:clamp(28px,4.2vw,44px);line-height:1.05;margin:0 0 12px;font-weight:760;
  letter-spacing:-.02em;text-wrap:balance;max-width:16ch;}
.lede{color:var(--muted);font-size:16px;max-width:64ch;margin:0;}
.stats{display:flex;flex-wrap:wrap;gap:0;margin-top:26px;
  border:1px solid var(--border);border-radius:12px;overflow:hidden;background:var(--panel);}
.stat{flex:1 1 140px;padding:14px 18px;border-right:1px solid var(--border2);}
.stat:last-child{border-right:0}
.stat .n{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.01em;}
.stat .l{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--faint);margin-top:3px;}
main{padding:30px 0 10px;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:16px;
  box-shadow:var(--shadow);overflow:hidden;}
.panel-h{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  padding:16px 20px;border-bottom:1px solid var(--border2);flex-wrap:wrap;}
.panel-h h2{margin:0;font-size:15px;font-weight:650;letter-spacing:-.01em;}
.panel-h .hint{font-size:12px;color:var(--faint);}
.mapwrap{position:relative;}
#map{display:block;width:100%;height:560px;touch-action:none;}
#legend{position:absolute;top:14px;right:14px;display:flex;flex-direction:column;gap:2px;
  background:color-mix(in srgb,var(--panel) 88%,transparent);backdrop-filter:blur(6px);
  border:1px solid var(--border);border-radius:10px;padding:8px;max-width:230px;}
.leg{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:6px;cursor:pointer;
  font-size:11.5px;color:var(--muted);user-select:none;transition:background .12s;}
.leg:hover{background:var(--border2);}
.leg.off{opacity:.38;}
.leg .sw{width:9px;height:9px;border-radius:50%;flex:0 0 auto;}
.leg .lc{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--faint);}
#tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;z-index:5;
  background:var(--ink);color:var(--bg);padding:8px 11px;border-radius:8px;max-width:280px;
  font-size:12.5px;line-height:1.4;box-shadow:0 6px 20px rgba(0,0,0,.28);}
#tip b{display:block;margin-bottom:3px;font-weight:600;}
#tip .meta{opacity:.72;font-size:11px;}
.section-h{margin:38px 0 4px;font-size:13px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--faint);font-weight:600;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;margin:16px 0 10px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px 18px;
  border-top:3px solid var(--dot);}
.card h3{margin:0 0 3px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
  display:flex;align-items:center;gap:8px;}
.card h3 .lc{margin-left:auto;font-variant-numeric:tabular-nums;}
.card .kw{font-size:14px;color:var(--ink);font-weight:600;margin:0 0 11px;line-height:1.35;}
.card ul{margin:0;padding:0;list-style:none;}
.card li{margin:0;padding:7px 0;border-top:1px solid var(--border2);font-size:13px;}
.card li:first-child{border-top:0;}
.card a{color:var(--ink);text-decoration:none;display:block;}
.card a:hover{color:var(--accent);}
.card .src{display:block;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--faint);margin-top:2px;}
footer{border-top:1px solid var(--border);margin-top:34px;padding:22px 0 46px;
  color:var(--faint);font-size:12.5px;line-height:1.7;}
a:focus-visible,.leg:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px;}
@media (max-width:560px){#map{height:440px;}#legend{position:static;max-width:none;margin:10px;}}
"""

BODY = """
<div class="shell">
  <header>
    <p class="eyebrow mono">Baltic Defence &amp; Geopolitics Monitor</p>
    <h1>The week in Russia, Ukraine &amp; regional security news</h1>
    <p class="lede">Every article the Baltic public broadcasters published on Ukraine, Russia,
      defence, China, war and drones — embedded, clustered by theme, and mapped. Hover a point
      for the headline; click to read the source.</p>
    <div class="stats mono" id="stats"></div>
  </header>
  <main>
    <section class="panel">
      <div class="panel-h">
        <h2>Topic map</h2>
        <span class="hint mono">TF-IDF · LSA · t-SNE · KMeans — proximity ≈ shared language</span>
      </div>
      <div class="mapwrap">
        <canvas id="map"></canvas>
        <div id="legend" class="mono"></div>
        <div id="tip"></div>
      </div>
    </section>
    <p class="section-h">Clusters</p>
    <section class="grid" id="cards"></section>
  </main>
  <footer class="mono">
    Sources — ERR (Estonia), LRT (Lithuania), LSM (Latvia), English editions.<br>
    Method — headlines filtered to defence/geopolitics themes, full article bodies fetched,
    TF-IDF vectorised, reduced with TruncatedSVD (LSA), laid out with t-SNE, clustered with KMeans.
    Cluster labels are each cluster's top mean-TF-IDF terms.
  </footer>
</div>
"""

SCRIPT = """
const D = __DATA__;
const C = D.colors, cvs = document.getElementById('map'), ctx = cvs.getContext('2d');
const tip = document.getElementById('tip'), wrap = cvs.parentElement;
const hidden = new Set();
let pts = [], hover = -1, anim = 1;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

// header stats
document.getElementById('stats').innerHTML = [
  [D.points.length,'Articles'], [D.sources,'Sources'],
  [D.clusters.length,'Clusters'], [D.generated,'Updated']
].map(s=>`<div class="stat"><div class="n">${s[0]}</div><div class="l">${s[1]}</div></div>`).join('');

// legend
const leg = document.getElementById('legend');
leg.innerHTML = D.clusters.map(c=>{
  const kw = c.label.split(', ').slice(0,3).join(' · ');
  return `<div class="leg" data-c="${c.id}" tabindex="0"><span class="sw" style="background:${C[c.id%C.length]}"></span>${kw}<span class="lc">${c.count}</span></div>`;
}).join('');
leg.querySelectorAll('.leg').forEach(el=>{
  const toggle=()=>{const id=+el.dataset.c; hidden.has(id)?hidden.delete(id):hidden.add(id);
    el.classList.toggle('off'); draw();};
  el.onclick=toggle;
  el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
});

// cluster cards
document.getElementById('cards').innerHTML = D.clusters.map(c=>{
  const arts = D.points.filter(p=>p.c===c.id);
  const col = C[c.id%C.length];
  const li = arts.map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
  const kw = c.label.split(', ').slice(0,4).join(', ');
  return `<div class="card" style="--dot:${col}"><h3><span class="sw" style="width:9px;height:9px;border-radius:50%;background:${col};display:inline-block"></span>Cluster ${c.id}<span class="lc">${c.count}</span></h3><p class="kw">${kw}</p><ul>${li}</ul></div>`;
}).join('');

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
    const r=(i===hover?8:5.5)*anim;
    ctx.beginPath();ctx.arc(p.sx,p.sy,r,0,7);
    ctx.fillStyle=C[p.c%C.length];ctx.globalAlpha=i===hover?1:0.9;ctx.fill();
    ctx.globalAlpha=1;ctx.lineWidth=i===hover?2:1.2;ctx.strokeStyle=ground;ctx.stroke();
    if(i===hover){ctx.beginPath();ctx.arc(p.sx,p.sy,r+4,0,7);
      ctx.strokeStyle=C[p.c%C.length];ctx.globalAlpha=.4;ctx.lineWidth=1.5;ctx.stroke();ctx.globalAlpha=1;}
  });
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
    tip.innerHTML=`<b>${p.h}</b><span class="meta">${p.s} · ${p.d}</span>`;
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
  (function run(t){anim=Math.min(1,(t-t0)/500);draw();if(anim<1)requestAnimationFrame(run);})(t0);}
// redraw on theme toggle
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
"""


def render(df, labels):
    payload = build_payload(df, labels)
    data_js = json.dumps(payload, ensure_ascii=False)
    script = SCRIPT.replace("__DATA__", data_js)
    inner = f"<style>{CSS}</style>\n{BODY}\n<script>{script}</script>"

    standalone = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Baltic Defence &amp; Geopolitics Monitor</title></head>"
        f"<body>{inner}</body></html>"
    )
    with open("nordics-monitor.html", "w", encoding="utf-8") as f:
        f.write(standalone)
    with open("nordics-monitor-artifact.html", "w", encoding="utf-8") as f:
        f.write(inner)  # content-only, for the Claude Artifact wrapper
    with open("nordics-monitor-data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[render] wrote nordics-monitor.html ({len(standalone)//1024} KB, self-contained)")


def main():
    df = load_data()
    df = filter_themes(df)
    df["body"] = fetch_all(df["url"].tolist())
    df, labels = analyse(df)
    render(df, labels)
    df[["url", "headline", "source", "scrape_date", "cluster"]].to_csv(
        "nordics-monitor-clustered.csv", index=False, encoding="utf-8")
    print("[done] nordics-monitor.html + artifact + data json + clustered csv")


if __name__ == "__main__":
    main()
