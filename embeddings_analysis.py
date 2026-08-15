#!/usr/bin/env python
"""
Semantic-embeddings sections of the Baltics Monitor.

Semantic map built on sentence embeddings, shown next to (not instead of) the
keyword-bucket themes. It's spliced into the MAIN report after those sections.
Pass --no-embed to skip it (e.g. a fast local build with no fastembed/umap).

This section is INDEPENDENT of the keyword-bucket themes in config.py. Topics
here are DISCOVERED from the embeddings themselves (HDBSCAN), not the regex
buckets, and named semantically from each cluster's own most distinctive terms
(class-based TF-IDF). Colour = discovered cluster, not keyword theme.

What it does, for TWO text scopes side by side:
  * "content"  — headline + full article body
  * "headline" — headline only
For each scope:
  1. Embed every article into a semantic vector (meaning, not word overlap),
     cached by URL so each run only embeds NEW articles (incremental).
  2. UMAP -> low-D for clustering + UMAP -> 2D for the map (metric="cosine":
     compare directions, not distance, because in high dimensions Euclidean
     distance collapses).
  3. HDBSCAN over the low-D embedding -> data-driven clusters (no preset count;
     genuinely off-topic stories fall out as "Unclustered" noise).
  4. c-TF-IDF per cluster -> the terms most distinctive to that cluster, used as
     its semantic label/keywords. Points are coloured by cluster.
  5. Cosine nearest-neighbours per article -> anomalies: the most ISOLATED
     stories (far from everything else).

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
# Higher-quality proximity: bge-large (still fastembed/ONNX, no torch). Its cache
# file is keyed by model name, so switching models never clashes with old vectors.
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
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


# ── escalation tone (embedding-anchored, no extra model) ─────────────────────
# Tone = how much a story leans toward ESCALATION vs DE-ESCALATION, measured as
# cosine to two small sets of anchor phrases. Reuses the article embeddings we
# already have; only these ~10 phrases are embedded fresh. Domain-appropriate:
# a defence feed is uniformly "negative", so generic sentiment is flat — what
# moves is the escalation/de-escalation axis.
ESCALATION_ANCHORS = [
    "military escalation and the threat of war",
    "russian attack, shelling, missile strikes and invasion",
    "troop mobilisation, conscription and armed conflict",
    "airspace violation, drone incursion and sabotage",
    "rising tension, provocation and hostile threats",
]
DEESCALATION_ANCHORS = [
    "ceasefire and peace negotiations",
    "diplomacy, dialogue and international cooperation",
    "de-escalation and easing of tensions",
    "peaceful resolution, stability and security guarantees",
    "agreement, partnership and lifting of sanctions",
]


def _tone(emb, model):
    """Per-article escalation tone, centred on this feed's baseline and scaled to
    ~[-1, 1] (positive = more escalatory than the feed norm). Returns display
    values; the median-centre makes it a RELATIVE read across topics/weeks."""
    enc, _ = _encoder(model)
    esc = _normalize(enc(ESCALATION_ANCHORS)).mean(axis=0)
    de = _normalize(enc(DEESCALATION_ANCHORS)).mean(axis=0)
    esc = esc / (np.linalg.norm(esc) or 1.0)
    de = de / (np.linalg.norm(de) or 1.0)
    raw = emb @ esc - emb @ de                          # emb rows are unit-normalised
    raw = raw - np.median(raw)                          # relative to feed baseline
    scale = np.percentile(np.abs(raw), 95) or 1.0
    return np.clip(raw / scale, -1.0, 1.0)


def _week_start(dstr):
    """ISO week (Monday) that a 'YYYY-MM-DD' scrape date falls in, as a date str."""
    from datetime import date, timedelta
    try:
        y, m, d = (int(x) for x in str(dstr)[:10].split("-"))
        dt = date(y, m, d)
        return (dt - timedelta(days=dt.weekday())).isoformat()
    except Exception:
        return None


def _heatmap(pts, clusters):
    """Aggregate point tone into a topic × week grid + an overall weekly line.

    Rows = real clusters (noise excluded). Cells carry mean tone + article count
    so thin cells stay honest. Weekly buckets: ~1 month of data can't support a
    daily cut without most cells being 1-2 stories."""
    weeks = sorted({w for p in pts if (w := _week_start(p["d"]))})
    widx = {w: i for i, w in enumerate(weeks)}
    real = [c for c in clusters if c["label"] != "Unclustered"]

    def agg(subset):
        cells = []
        for w in weeks:
            vals = [p["to"] for p in subset if _week_start(p["d"]) == w]
            if vals:
                cells.append({"w": widx[w], "m": round(sum(vals) / len(vals), 3),
                              "n": len(vals)})
        return cells

    heat = [{"c": c["id"], "cells": agg([p for p in pts if p["c"] == c["id"]])}
            for c in real]
    line = agg(pts)
    return {"weeks": weeks, "rows": [c["id"] for c in real], "heat": heat, "line": line}


def _cluster(emb):
    """HDBSCAN over a UMAP-reduced embedding -> integer label per row (-1 = noise).

    UMAP-then-HDBSCAN is the BERTopic recipe: cosine UMAP to a handful of dims
    (Euclidean is meaningful there) so density clustering has something to bite.
    min_cluster_size scales with n so the topic granularity tracks the corpus.
    """
    from sklearn.cluster import HDBSCAN
    n = len(emb)
    if n < 12:
        return np.zeros(n, dtype=int)                  # too few docs to cluster
    red = _umap(emb, n_components=min(5, n - 2), min_dist=0.0)
    mcs = max(5, n // 25)                              # ~4-10 topics on a ~150-250 doc corpus
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=1,
                     metric="euclidean").fit_predict(red)
    return labels


# ── semantic labelling (class-based TF-IDF, à la BERTopic) ───────────────────
_LABEL_STOP = set(config.EXTRA_STOP) | set(config.FREQ_STOP) | {
    "say", "says", "said", "told", "reports", "report", "reported", "week",
    "government", "minister", "ministry", "official", "officials", "plan", "plans",
}


def _dedupe_terms(terms, k):
    """Trim a ranked term list to k CONCISE, non-redundant keywords.

    Collapses morphological variants and overlapping n-grams by a 4-letter word
    stem: 'russia'/'russian' -> one term, 'sanctions' drops the later 'russia
    sanctions', 'air defence' drops a later 'air'. Highest-ranked term wins its
    stem; anything re-using an already-claimed stem is skipped.
    """
    chosen, seen = [], set()
    for t in terms:
        stems = {w[:4] for w in t.split()}
        if stems & seen:
            continue
        seen |= stems
        chosen.append(t)
        if len(chosen) >= k:
            break
    return chosen


def _ctfidf_labels(texts, labels, topn=3):
    """For each cluster, its most DISTINCTIVE terms (class-based TF-IDF), deduped.

    Concatenate every cluster's docs into one 'class document', count terms, then
    weight by how concentrated a term is in one class vs the whole corpus. Returns
    {cluster_id: [keyword, ...]} for real clusters only (noise -1 is skipped).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    ids = sorted(i for i in set(labels) if i != -1)
    if not ids:
        return {}
    docs = [" ".join(t for t, l in zip(texts, labels) if l == cid) for cid in ids]
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = list(ENGLISH_STOP_WORDS | _LABEL_STOP)
        cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2),
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=1)
        X = cv.fit_transform(docs).toarray().astype(float)      # (n_classes, n_terms)
    except ValueError:
        return {cid: [] for cid in ids}
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.clip(X.sum(axis=1, keepdims=True), 1, None)      # per-class term freq
    A = X.sum() / max(1, len(ids))                              # avg words per class
    idf = np.log(1.0 + A / np.clip(X.sum(axis=0), 1, None))     # rarer-across-classes -> bigger
    ctfidf = tf * idf
    out = {}
    for row, cid in enumerate(ids):
        ranked = [vocab[j] for j in np.argsort(-ctfidf[row]) if ctfidf[row, j] > 0][:12]
        out[cid] = _dedupe_terms(ranked, topn)                 # concise, non-redundant
    return out


def _palette(k):
    """k visually distinct (light, dark) hex colours via golden-angle hues.

    Golden-angle spacing maximises separation between consecutively-assigned
    hues. Identity is carried by the legend + tooltip label (colour alone can't
    stay colour-blind-safe past a few hues); colour just reinforces grouping.
    """
    def hsl(h, s, l):
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c),
                   (x, 0, c), (c, 0, x)][int(h // 60) % 6]
        return "#%02x%02x%02x" % (round((r + m) * 255), round((g + m) * 255), round((b + m) * 255))
    light, dark = [], []
    for i in range(max(1, k)):
        h = (i * 137.508) % 360
        light.append(hsl(h, 0.62, 0.46))
        dark.append(hsl(h, 0.68, 0.62))
    return light, dark


NOISE_LIGHT, NOISE_DARK = "#8f8677", "#847b6c"          # "Unclustered" grey


# ── build one scope's payload ───────────────────────────────────────────────
def build_view(df, scope, model):
    urls = df["url"].tolist()
    if scope == "content":
        texts = (df["headline"].fillna("") + ". " + df["body"].fillna("")).tolist()
    else:
        texts = df["headline"].fillna("").tolist()

    emb = get_embeddings(urls, texts, scope, model)
    xy = _umap(emb, n_components=2, min_dist=0.1)      # 2-D layout for the map
    _nn, iso = _neighbors(emb, 5)
    raw = _cluster(emb)                                # HDBSCAN labels (-1 = noise)
    keywords = _ctfidf_labels(texts, raw, topn=4)      # {raw_id: [term, ...]}
    tone = _tone(emb, model)                           # per-article escalation tone

    # Renumber clusters 0..K-1 by size (largest first) for stable colours; noise last.
    real = sorted((i for i in set(raw) if i != -1),
                  key=lambda c: -int((raw == c).sum()))
    remap = {c: i for i, c in enumerate(real)}
    K = len(real)
    light, dark = _palette(K)

    clusters = []
    for i, c in enumerate(real):
        kw = keywords.get(c, [])
        clusters.append({
            "id": i, "label": " · ".join(kw[:2]) or f"topic {i + 1}",
            "keywords": kw, "size": int((raw == c).sum()),
            "color": light[i], "colorDark": dark[i],
        })
    if (raw == -1).any():                              # noise bucket -> last id
        clusters.append({
            "id": K, "label": "Unclustered", "keywords": [],
            "size": int((raw == -1).sum()),
            "color": NOISE_LIGHT, "colorDark": NOISE_DARK,
        })

    pts = []
    for i, r in enumerate(df.itertuples()):
        cid = remap.get(int(raw[i]), K)                # noise -> K
        pts.append({
            "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
            "h": r.headline, "s": config.SRC_LABEL[r.source], "d": str(r.scrape_date),
            "u": r.url, "c": cid, "iso": round(float(iso[i]), 3),
            "to": round(float(tone[i]), 3),
        })
    return {"scope": scope, "model": model, "points": pts, "clusters": clusters,
            "tone": _heatmap(pts, clusters)}


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
.emb-topic-item{border-top:1px solid var(--border2);}
.emb-topic-item:first-child{border-top:0;}
.emb-topic{display:flex;align-items:baseline;gap:9px;padding:8px 0;cursor:pointer;}
.emb-topic:hover{color:var(--accent);}
.emb-topic .caret{flex:0 0 auto;font-size:10px;color:var(--faint);transition:transform .12s;
  transform-origin:center;position:relative;top:-1px;width:10px;}
.emb-topic.open .caret{transform:rotate(90deg);}
.emb-topic .dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto;position:relative;top:1px;}
.emb-topic .kw{font-size:13.5px;color:var(--ink);font-weight:600;}
.emb-topic:hover .kw{color:var(--accent);}
.emb-topic .kw .g{color:var(--muted);font-weight:400;}
.emb-topic .ct{margin-left:auto;font-size:11px;font-variant-numeric:tabular-nums;color:var(--faint);
  white-space:nowrap;}
.emb-arts{display:none;list-style:none;margin:0 0 8px;padding:0 0 4px 30px;}
.emb-arts.open{display:block;}
.emb-arts li{padding:6px 0;border-top:1px solid var(--border2);font-size:12.5px;}
.emb-arts a{color:var(--muted);text-decoration:none;display:block;}
.emb-arts a:hover{color:var(--accent);}
.emb-arts .src{display:block;font-size:10px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);margin-top:2px;}
/* escalation-tone panel */
.emb-tone{padding:6px 20px 18px;position:relative;}
.emb-tone h4{margin:0 0 4px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.tone-key{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;letter-spacing:.04em;
  text-transform:none;color:var(--faint);font-weight:400;}
.tone-key i{width:14px;height:10px;border-radius:2px;display:inline-block;}
.tone-key .tk-neg{background:#2a78d6;}.tone-key .tk-mid{background:#f0efec;border:1px solid var(--border);}
.tone-key .tk-pos{background:#d03b3b;}
:root[data-theme="dark"] .tone-key .tk-neg{background:#3987e5;}
:root[data-theme="dark"] .tone-key .tk-mid{background:#383835;}
.tone-sub{margin:0 0 12px;font-size:12px;color:var(--muted);max-width:70ch;line-height:1.5;}
.tone-line{display:block;width:100%;height:56px;margin:0 0 6px;overflow:visible;}
.tone-heat{display:grid;gap:2px;overflow-x:auto;font-size:11px;}
.tone-heat .th-row{display:grid;grid-template-columns:var(--labw,150px) 1fr;gap:6px;align-items:center;}
.tone-heat .th-lab{color:var(--ink);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  text-align:right;font-size:11.5px;}
.tone-heat .th-cells{display:grid;gap:2px;grid-auto-flow:column;grid-auto-columns:1fr;}
.tone-heat .th-cell{height:22px;border-radius:3px;background:var(--border2);cursor:default;
  display:flex;align-items:center;justify-content:center;font-size:9.5px;color:rgba(0,0,0,.5);}
.tone-heat .th-head .th-cell{background:none;color:var(--faint);font-size:10px;letter-spacing:.03em;height:16px;}
.tone-heat .th-head .th-lab{color:var(--faint);font-weight:400;}
.tone-heat .th-all{padding-bottom:4px;margin-bottom:2px;border-bottom:1px solid var(--border);}
.tone-heat .th-all .th-cell{height:26px;font-size:10.5px;font-weight:600;}
.tone-heat .th-cell[data-m]:hover{outline:2px solid var(--ink);outline-offset:-2px;}
"""

_EMB_SECTION = """
    <p class="section-h"><svg class="sec-sign" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="7" r="2.2"/><circle cx="14" cy="6" r="2.2"/><circle cx="10" cy="14" r="2.2"/><path d="M7.8 8.2 12 12M12.2 7.2 11 12"/></svg> Discovered topics — clusters found by meaning (embeddings &rarr; HDBSCAN)</p>
    <section class="panel">
      <div class="panel-h">
        <h2>Semantic topics &amp; anomalies</h2>
        <span class="hint mono">topics DISCOVERED from the text, independent of the keyword themes above</span>
      </div>
      <div style="padding:14px 20px 0">
        <p class="emb-note">Built from scratch from the article <b>embeddings</b> (BGE-large), <b>not</b> the keyword buckets above. HDBSCAN groups the stories into <b>data-driven clusters</b>; each cluster is named by its own most distinctive terms (class-based TF-IDF). Colour = discovered cluster. Two scopes: <b>content</b> embeds headline + body, <b>headline</b> embeds titles only. Genuinely off-topic stories fall out as <i>Unclustered</i>.</p>
        <div class="emb-tabs" id="embTabs"></div>
      </div>
      <div class="emb-wrap">
        <canvas class="emb-canvas" id="embCanvas"></canvas>
        <div class="emb-legend" id="embLegend"></div>
        <div class="emb-tip" id="embTip"></div>
      </div>
      <div class="emb-tone">
        <h4>Escalation tone — is coverage heating up? <span class="tone-key"><i class="tk-neg"></i>de-escalation<i class="tk-mid"></i>baseline<i class="tk-pos"></i>escalation</span></h4>
        <p class="tone-sub">Each story scored on an <b>escalation vs de-escalation</b> axis (cosine to anchor phrases in the same embedding space), <i>relative to this feed's own baseline</i>. Rows = discovered topics by week; the top <b>All coverage</b> row is the overall trend. Colour = mean tone; deeper red = more escalatory. Number = articles that week.</p>
        <div class="tone-heat" id="toneHeat"></div>
        <div class="emb-tip" id="toneTip"></div>
      </div>
      <div class="emb-anom">
        <h4 id="embTopH">Discovered topics</h4>
        <ul id="embTop"></ul>
        <h4 id="embAnomH">Anomalies — the most isolated stories</h4>
        <ul id="embAnom"></ul>
      </div>
    </section>
"""

_EMB_SCRIPT = r"""
(function(){
  const EV = __EMB_DATA__;                       // {content:{points,clusters}, headline:{...}}
  const scopes = Object.keys(EV).filter(k=>EV[k] && EV[k].points);
  if(!scopes.length) return;
  const ANOM_N = 20;                             // how many most-isolated stories to list
  const isDark=()=>{const t=document.documentElement.getAttribute('data-theme');
    return t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme:dark)').matches);};
  let cur = scopes[0];
  const hidden = new Set();                       // hidden CLUSTER ids
  let CL=[], COL=[], NAME=[];                     // per-scope clusters / colour map / label map
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

  function refreshColors(){
    const d=isDark();
    COL={}; NAME={};
    CL.forEach(c=>{COL[c.id]=d?c.colorDark:c.color; NAME[c.id]=c.label;});
  }
  // diverging blue↔gray↔red tone scale (t in [-1,1]); theme-aware midpoint
  function toneRGB(t){
    const d=isDark();
    const neg=d?[57,135,229]:[42,120,214], mid=d?[56,56,53]:[240,239,236], pos=[208,59,59];
    const k=Math.max(-1,Math.min(1,t)), a=Math.abs(k), e=k<0?neg:pos;
    return [Math.round(mid[0]+(e[0]-mid[0])*a),
            Math.round(mid[1]+(e[1]-mid[1])*a),
            Math.round(mid[2]+(e[2]-mid[2])*a)];
  }
  const toneColor = t => `rgb(${toneRGB(t).join(',')})`;
  // black or white text, whichever reads on the cell (perceived luminance)
  function toneInk(t){const [r,g,b]=toneRGB(t);
    return (0.299*r+0.587*g+0.114*b)>150?'rgba(0,0,0,.62)':'rgba(255,255,255,.82)';}
  const wk = s => new Date(s+'T00:00').toLocaleDateString('en',{month:'short',day:'numeric'});
  function renderTone(v){
    const T=v.tone, heatEl=document.getElementById('toneHeat');
    if(!T||!T.weeks||!T.weeks.length){heatEl.innerHTML='';return;}
    const W=T.weeks.length;
    const rowHtml=(id,label,color,cells,cls)=>{
      const byW={}; cells.forEach(c=>byW[c.w]=c);
      let out='';
      for(let i=0;i<W;i++){const c=byW[i];
        out += c
          ? `<div class="th-cell" style="background:${toneColor(c.m)};color:${toneInk(c.m)}" data-l="${label}" data-w="${i}" data-m="${c.m}" data-n="${c.n}">${c.n}</div>`
          : `<div class="th-cell" style="background:transparent"></div>`;}
      return `<div class="th-row ${cls||''}"><div class="th-lab" style="color:${color||'var(--faint)'}">${label}</div><div class="th-cells">${out}</div></div>`;
    };
    let html = `<div class="th-row th-head"><div class="th-lab">topic</div><div class="th-cells">`
      + T.weeks.map(w=>`<div class="th-cell">${wk(w)}</div>`).join('') + `</div></div>`;
    html += rowHtml(-1,'All coverage','var(--ink)',T.line,'th-all');
    T.heat.forEach(r=> html += rowHtml(r.c, NAME[r.c], COL[r.c], r.cells));
    heatEl.innerHTML = html;
    const tt=document.getElementById('toneTip');
    heatEl.querySelectorAll('.th-cell[data-m]').forEach(el=>{
      el.onmousemove=e=>{const wr=el.closest('.emb-tone'), rc=wr.getBoundingClientRect();
        const m=+el.dataset.m;
        tt.innerHTML=`<b>${el.dataset.l}</b><span class="m">${wk(T.weeks[+el.dataset.w])} · ${el.dataset.n} stories<br>mean tone ${m>0?'+':''}${m.toFixed(2)}</span>`;
        tt.style.opacity=1;
        let tx=e.clientX-rc.left+14, ty=e.clientY-rc.top+14;
        if(tx+180>wr.clientWidth)tx=e.clientX-rc.left-190; tt.style.left=tx+'px'; tt.style.top=ty+'px';};
      el.onmouseleave=()=>{tt.style.opacity=0;};
    });
  }
  function load(){
    const v=EV[cur]; CL=v.clusters||[]; refreshColors();
    const xs=v.points.map(p=>p.x), ys=v.points.map(p=>p.y);
    minX=Math.min(...xs);maxX=Math.max(...xs);minY=Math.min(...ys);maxY=Math.max(...ys);
    const px=(maxX-minX)*0.06||1, py=(maxY-minY)*0.06||1; minX-=px;maxX+=px;minY-=py;maxY+=py;
    // legend = discovered clusters, by size (already sorted server-side)
    document.getElementById('embLegend').innerHTML = CL.map(c=>
      `<div class="emb-leg${hidden.has(c.id)?' off':''}" data-c="${c.id}" tabindex="0">`
      +`<span class="sw" style="background:${COL[c.id]}"></span>`
      +`<span>${c.label}</span><span class="lc">${c.size}</span></div>`
    ).join('');
    document.querySelectorAll('#embLegend .emb-leg').forEach(el=>{
      el.onclick=()=>{const id=+el.dataset.c; hidden.has(id)?hidden.delete(id):hidden.add(id);
        el.classList.toggle('off'); draw();};
    });
    // discovered-topics list: click a topic to expand the articles inside it
    document.getElementById('embTopH').textContent=`Discovered topics — ${CL.filter(c=>c.label!=='Unclustered').length} clusters (click to list articles)`;
    const artsFor = id => v.points.filter(p=>p.c===id)
      .sort((a,b)=> (a.d<b.d?1:a.d>b.d?-1:0))          // most recent first
      .map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}`
        +`<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
    document.getElementById('embTop').innerHTML = CL.map(c=>{
      const kw = c.keywords&&c.keywords.length
        ? c.keywords.map((w,i)=>i?`<span class="g">, ${w}</span>`:w).join('')
        : `<span class="g">${c.label}</span>`;
      return `<li class="emb-topic-item">`
        +`<div class="emb-topic" data-c="${c.id}"><span class="caret">▸</span>`
        +`<span class="dot" style="background:${COL[c.id]}"></span>`
        +`<span class="kw">${kw}</span><span class="ct">${c.size} stories</span></div>`
        +`<ul class="emb-arts" id="arts-${c.id}">${artsFor(c.id)}</ul></li>`;
    }).join('');
    document.querySelectorAll('#embTop .emb-topic').forEach(el=>{
      el.onclick=()=>{const ul=document.getElementById('arts-'+el.dataset.c);
        const open=ul.classList.toggle('open'); el.classList.toggle('open',open);};
    });
    // anomalies (most isolated)
    const li = arr => arr.map(p=>{
      const nm=NAME[p.c]&&NAME[p.c]!=='Unclustered'?`<span class="emb-badge">${NAME[p.c]}</span>`:'';
      return `<li><a href="${p.u}" target="_blank" rel="noopener">${nm}${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`;
    }).join('');
    const anom=v.points.slice().sort((a,b)=>b.iso-a.iso).slice(0,ANOM_N);
    document.getElementById('embAnomH').textContent=`Anomalies — the ${anom.length} most isolated stories`;
    document.getElementById('embAnom').innerHTML = anom.length?li(anom):'<li>none</li>';
    renderTone(v);
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
    P.forEach((p,i)=>{ if(hidden.has(p.c))return;
      const r=(i===hover?8:5.5);
      ctx.beginPath();ctx.arc(p.sx,p.sy,r,0,7);
      ctx.fillStyle=COL[p.c]||'#888';ctx.globalAlpha=i===hover?1:.85;ctx.fill();ctx.globalAlpha=1;
      ctx.lineWidth=i===hover?2:1.1;ctx.strokeStyle=ground;ctx.stroke();
    });
  }
  function nearest(mx,my){let b=-1,bd=169;P.forEach((p,i)=>{if(hidden.has(p.c))return;
    const d=(p.sx-mx)**2+(p.sy-my)**2;if(d<bd){bd=d;b=i;}});return b;}
  cvs.addEventListener('mousemove',e=>{const rc=cvs.getBoundingClientRect(),mx=e.clientX-rc.left,my=e.clientY-rc.top;
    const hh=nearest(mx,my); if(hh!==hover){hover=hh;draw();}
    if(hh>=0){const p=P[hh];cvs.style.cursor='pointer';
      tip.innerHTML=`<b>${p.h}</b><span class="m">${p.s} · ${p.d}<br>topic: ${NAME[p.c]||'—'} · isolation ${p.iso}</span>`;
      tip.style.opacity=1; let tx=mx+16,ty=my+16;
      if(tx+tip.offsetWidth>wrap.clientWidth)tx=mx-tip.offsetWidth-16;
      if(ty+tip.offsetHeight>cvs.clientHeight)ty=my-tip.offsetHeight-16;
      tip.style.left=tx+'px';tip.style.top=ty+'px';
    }else{cvs.style.cursor='default';tip.style.opacity=0;}});
  cvs.addEventListener('mouseleave',()=>{hover=-1;tip.style.opacity=0;draw();});
  cvs.addEventListener('click',()=>{if(hover>=0)window.open(P[hover].u,'_blank','noopener');});
  new ResizeObserver(()=>{layout();draw();}).observe(wrap);
  // follow the page's light/dark, like the main map
  const onTheme=()=>{refreshColors(); load();};
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',onTheme);
  new MutationObserver(onTheme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  load();
})();
"""


def build_report_fragment(df, model=DEFAULT_MODEL, **_ignored):
    """Return (css, html, script) to splice into the main report. df must carry
    'body' (from pipeline.fetch_all). Topics are discovered here, independent of
    the keyword themes, so no theme names are needed. Extra kwargs are accepted
    and ignored for backward-compatible callers."""
    import json
    payload = {sc: build_view(df, sc, model) for sc in SCOPES}
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
    df = pipeline.filter_themes(df_all)                     # defence-filtered set
    df["body"] = pipeline.fetch_all(df["url"].tolist())     # topics discovered from embeddings; no analyse() needed
    _standalone(df, args.embed_model)


if __name__ == "__main__":
    main()
