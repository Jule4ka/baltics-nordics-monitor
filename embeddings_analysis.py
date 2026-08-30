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

What it does, for the "content" scope — headline + full article body:
  1. Embed every article into a semantic vector (meaning, not word overlap),
     cached by URL so each run only embeds NEW articles (incremental).
  2. UMAP -> low-D for clustering + UMAP -> 2D for the map (metric="cosine":
     compare directions, not distance, because in high dimensions Euclidean
     distance collapses).
  3. HDBSCAN over the low-D embedding -> data-driven clusters (no preset count;
     genuinely off-topic stories fall out as "Other" noise).
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
SCOPES = ("content",)


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
    real = [c for c in clusters if c["label"] != "Other"]

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
    # Defence/geopolitics feeds are topically narrow, so most stories pile into one
    # dense mass. HDBSCAN's default "eom" (excess-of-mass) selection prefers that big
    # parent cluster, and around min_cluster_size ~= n/30 it collapses the whole feed
    # into just 2 mega-blobs — a sharp cliff, not a gradual drift (eom gave 15 topics
    # at mcs=8 but 2 at mcs=9/10 on a 272-doc corpus). Tuning the divisor only moves
    # the cliff; as the corpus grows, n//30 eventually lands back on it and we get 2
    # topics "again". "leaf" selection instead keeps the leaf sub-clusters, decomposing
    # that mass into ~10-15 readable sub-topics and degrading smoothly with mcs.
    mcs = max(6, n // 30)                              # ~10-15 topics on a ~150-260 doc corpus
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="euclidean",
                     cluster_selection_method="leaf").fit_predict(red)
    return labels


# A noise point joins a cluster if its cosine to that centroid clears the cluster's
# admission bar: the NOISE_ADMIT_PCTL-th percentile of members' own cosine to the
# centroid. LOWER percentile = lower bar = MORE points absorbed out of noise; higher
# keeps more as Other. 10 ≈ "closer than the cluster's most peripheral tenth",
# which folds in the obvious border cases while leaving genuine outliers alone.
# Reassignment runs in the ORIGINAL embedding space (cosine), not the UMAP layout,
# à la BERTopic's reduce_outliers.
NOISE_ADMIT_PCTL = 10


def _reduce_noise(emb, labels):
    """Fold HDBSCAN's -1 noise into the nearest real cluster when it's a plausible
    member, leaving only genuinely isolated stories as 'Other'.

    HDBSCAN is deliberately conservative — a topically narrow feed leaves lots of
    low-density border points flagged noise even though they sit right beside a
    cluster. For each real cluster we take its centroid and an admission bar (the
    NOISE_ADMIT_PCTL-th percentile of members' cosine to that centroid); a noise
    point joins the best-matching cluster only if it clears that cluster's bar.
    """
    labels = np.asarray(labels).copy()
    real = sorted(c for c in set(labels.tolist()) if c != -1)
    noise = np.where(labels == -1)[0]
    if not real or not len(noise):
        return labels
    cents, bars = {}, {}
    for c in real:
        m = emb[labels == c]
        cen = m.mean(axis=0)
        cen = cen / (np.linalg.norm(cen) or 1.0)
        cents[c] = cen
        bars[c] = float(np.percentile(m @ cen, NOISE_ADMIT_PCTL))
    C = np.stack([cents[c] for c in real])             # (n_clusters, dim)
    sims = emb[noise] @ C.T                             # (n_noise, n_clusters), cosine
    best = sims.argmax(axis=1)
    admitted = 0
    for row, i in enumerate(noise):
        c = real[best[row]]
        if sims[row, best[row]] >= bars[c]:
            labels[i] = c
            admitted += 1
    if admitted:
        print(f"[cluster] reduced noise: {admitted}/{len(noise)} outliers reassigned, "
              f"{len(noise) - admitted} left Other")
    return labels


# ── semantic labelling (class-based TF-IDF, à la BERTopic) ───────────────────
_LABEL_STOP = set(config.EXTRA_STOP) | set(config.FREQ_STOP) | {
    "say", "says", "said", "told", "reports", "report", "reported", "week",
    "government", "minister", "ministry", "official", "officials", "plan", "plans",
    "photo",                                           # caption/credit boilerplate
}

# Feed-universal fillers: this is a DEFENCE feed, so "defence/military/war" sit in
# almost every cluster and carry no distinguishing signal AS A SOLO WORD — but they
# ARE wanted inside a phrase ("ukraine war", "air defence", "armed forces"). So we do
# NOT stopword them out of the vocabulary (that would block those phrases); instead a
# single filler word is heavily penalised in ranking, while phrases containing one are
# not. Affects LABELS only, never the embeddings/clustering.
_FILLER = {
    "defense", "defenses", "defence", "defences", "military", "militaries",
    "security", "forces", "force", "army", "armies", "war", "warfare",
}

# Labels must read as ENGLISH. The sources are English editions of Baltic
# broadcasters, but photo credits, institution names and bylines drag local-language
# tokens into the article bodies, and c-TF-IDF loves them precisely because they're
# rare-and-distinctive — so a cluster ends up labelled "border · valsts". There's no
# structural signal separating a wanted proper noun (nato, kyiv, iran) from an
# unwanted one (a surname, a Latvian institution acronym), so we curate the leaks.
# Extend this as new foreign tokens surface; it only affects LABELS, not clustering.
_NON_ENGLISH = {
    # Latvian / Estonian / Lithuanian common words + caption boilerplate
    "valsts", "prezidents", "prezidenta", "prezidentu", "prezidents", "kanceleja",
    "latvijas", "latvija", "austrumu", "robezsardze",
    # foreign institution acronyms (used verbatim in the English text)
    "csdd", "vsat", "ppa", "otp", "riigikogu",
    # surnames / local place names that RECUR across a cluster's stories (the docfreq
    # filter already drops one-off names; these appear often enough to need listing)
    "liubajevas", "kiviselg", "viimsalu", "luik", "karu", "edgars", "katelynas",
}
_LABEL_STOP |= _NON_ENGLISH


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
        cv = CountVectorizer(stop_words=stop, ngram_range=(1, 3),
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=1)
        X = cv.fit_transform(docs).toarray().astype(float)      # (n_classes, n_terms)
    except ValueError:
        return {cid: [] for cid in ids}
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.clip(X.sum(axis=1, keepdims=True), 1, None)      # per-class term freq
    A = X.sum() / max(1, len(ids))                              # avg words per class
    idf = np.log(1.0 + A / np.clip(X.sum(axis=0), 1, None))     # rarer-across-classes -> bigger
    ctfidf = tf * idf

    # In-cluster document frequency: how many of a cluster's OWN articles use each term.
    # A label should describe the whole cluster, so a term must recur across several of
    # its stories — not sit in just one. This is what keeps one-off surnames and stray
    # foreign tokens (a photo credit, a byline) out of labels WITHOUT a curated list:
    # they appear in a single article, so they never clear the bar. Real themes recur.
    present = (cv.transform(texts) > 0)                          # (n_articles, n_terms) binary
    members = {cid: [i for i, l in enumerate(labels) if l == cid] for cid in ids}

    # Prefer readable MULTI-WORD phrases over cryptic single tokens:
    #   • n-gram length boost   — a bigram outranks a unigram of equal c-TF-IDF, so
    #     "ukraine war" surfaces above "ukraine".
    #   • solo-filler penalty   — a lone feed-universal word (war/defence/…) is pushed
    #     down, but the SAME word inside a phrase is not (it never gets the penalty).
    #   • phrase-friendly recurrence — a unigram must recur in ~15% of the cluster, but
    #     a phrase only needs to appear in a few stories (phrases are naturally rarer),
    #     otherwise the strict docfreq bar filters every phrase out.
    lengths = np.array([t.count(" ") + 1 for t in vocab])
    phrase_boost = np.where(lengths >= 3, 2.4, np.where(lengths == 2, 2.0, 1.0))
    solo_filler = np.array([1.0 if (lengths[j] == 1 and vocab[j] in _FILLER) else 0.0
                            for j in range(len(vocab))])
    cand = {}
    for row, cid in enumerate(ids):
        idx = members[cid]
        dfreq = np.asarray(present[idx].sum(axis=0)).ravel() if idx else np.zeros(len(vocab))
        uni_min = max(2, int(round(0.15 * len(idx))))           # unigram: recurs across the cluster
        thresh = np.where(lengths >= 2, 2, uni_min)             # phrase: just needs to recur ≥2×
        score = ctfidf[row] * phrase_boost * np.where(solo_filler > 0, 0.12, 1.0)
        ranked = [vocab[j] for j in np.argsort(-score)
                  if ctfidf[row, j] > 0 and dfreq[j] >= thresh[j]][:25]
        cand[cid] = _dedupe_terms(ranked, 8)                    # collapse morphological overlaps

    # Label = the top few distinctive terms (phrases first). Cross-cluster distinctness
    # comes naturally now that each label carries 2+ terms; exact-duplicate labels are
    # broken apart downstream by extending with the next term.
    return {cid: cand[cid][:topn] for cid in ids}


def _keybert_labels(texts, labels, emb, model, topn=4):
    """Label clusters with a BLEND of KeyBERT + c-TF-IDF.

    Candidate n-grams are proposed by recurrence, then scored by
        cosine(phrase_embedding, cluster_centroid)   [semantic centrality, BGE]
      × c-TF-IDF(term, cluster)                       [distinctiveness]
    so the winner is a phrase that is BOTH on-topic AND specific to the cluster — cosine
    alone rewards generic central words ("use", "incident"); c-TF-IDF alone rewards
    distinctive-but-cryptic ones. Reuses the map's model + the article embeddings, and
    falls back to c-TF-IDF ordering if candidate embedding fails, so the build never breaks."""
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    ids = sorted(i for i in set(labels) if i != -1)
    if not ids:
        return {}
    stop = list(ENGLISH_STOP_WORDS | _LABEL_STOP)
    try:
        cv = CountVectorizer(stop_words=stop, ngram_range=(1, 3),
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=1).fit(texts)
    except ValueError:
        return {cid: [] for cid in ids}
    vocab = np.array(cv.get_feature_names_out())
    lengths = np.array([t.count(" ") + 1 for t in vocab])
    counts = cv.transform(texts)                                 # (n_articles, n_terms) counts
    present = (counts > 0)
    members = {cid: [i for i, l in enumerate(labels) if l == cid] for cid in ids}

    # c-TF-IDF distinctiveness per (cluster, term): term freq in the class × how rare the
    # term is ACROSS classes — this is what pushes shared words ("russia") and feed-universal
    # fillers down, so they stop dominating every label.
    X = np.vstack([np.asarray(counts[members[cid]].sum(axis=0)).ravel() if members[cid]
                   else np.zeros(len(vocab)) for cid in ids]).astype(float)   # (n_classes, n_terms)
    tf = X / np.clip(X.sum(axis=1, keepdims=True), 1, None)
    A = X.sum() / max(1, len(ids))
    idf = np.log(1.0 + A / np.clip(X.sum(axis=0), 1, None))
    ctfidf = tf * idf                                            # (n_classes, n_terms)
    phrase_boost = np.where(lengths >= 3, 2.4, np.where(lengths == 2, 2.0, 1.0))
    solo_filler = np.array([0.12 if (lengths[j] == 1 and vocab[j] in _FILLER) else 1.0
                            for j in range(len(vocab))])

    # Per-cluster candidate pool: terms that RECUR in the cluster, kept by a phrase-aware
    # pre-score (c-TF-IDF × phrase boost) so distinctive PHRASES survive into the pool
    # instead of being crowded out by more-frequent single words. Capped at 30 to bound
    # how many phrases we embed.
    row_of = {cid: r for r, cid in enumerate(ids)}
    cand_sets, all_cands = {}, set()
    for cid in ids:
        idx = members[cid]
        if not idx:
            cand_sets[cid] = []
            continue
        dfreq = np.asarray(present[idx].sum(axis=0)).ravel()
        uni_min = max(2, int(round(0.15 * len(idx))))
        thresh = np.where(lengths >= 2, 2, uni_min)
        prescore = ctfidf[row_of[cid]] * phrase_boost * solo_filler
        elig = [j for j in range(len(vocab))
                if dfreq[j] >= thresh[j] and ctfidf[row_of[cid], j] > 0]
        elig.sort(key=lambda j: -prescore[j])
        cand_sets[cid] = [vocab[j] for j in elig[:30]]
        all_cands.update(cand_sets[cid])
    if not all_cands:
        return {cid: [] for cid in ids}

    # Embed every candidate phrase ONCE with the map's model, unit-normalise.
    try:
        cand_list = sorted(all_cands)
        enc, _ = _encoder(model)
        cvecs = _normalize(enc(cand_list))
        cidx = {t: i for i, t in enumerate(cand_list)}
    except Exception as e:
        print(f"[label] KeyBERT embedding failed ({e.__class__.__name__}); using c-TF-IDF order")
        return _ctfidf_labels(texts, labels, topn=topn)

    jof = {t: j for j, t in enumerate(vocab)}
    out = {}
    for cid in ids:
        cs = cand_sets[cid]
        if not cs:
            out[cid] = []
            continue
        cen = emb[members[cid]].mean(axis=0)
        cen = cen / (np.linalg.norm(cen) or 1.0)
        row = row_of[cid]

        def blend(t):
            j = jof[t]
            cos = max(0.0, float(cvecs[cidx[t]] @ cen))         # semantic centrality (BGE)
            return cos * ctfidf[row, j] * phrase_boost[j] * solo_filler[j]

        ranked = sorted(cs, key=lambda t: -blend(t))
        out[cid] = _dedupe_terms(ranked, 8)                     # collapse morphological overlaps
    print(f"[label] KeyBERT×c-TF-IDF blended labels for {len(ids)} clusters via {model}")
    return {cid: out[cid][:topn] for cid in ids}


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


NOISE_LIGHT, NOISE_DARK = "#8f8677", "#847b6c"          # "Other" grey


# ── LLM topic labels (descriptive phrases; graceful fallback to keywords) ────
# The c-TF-IDF terms make short but cryptic labels. When an Anthropic key is
# available, upgrade them to a readable descriptive phrase per cluster — one API
# call per build, cached by cluster CONTENT so unchanged clusters cost nothing on
# repeat builds. If the SDK/key is missing or the call fails, we keep the keyword
# labels (mirrors how the page already degrades to TF-IDF when embeddings are off).
LABEL_MODEL = "claude-opus-5"


def _llm_call(samples):
    """samples: {cid: {"keywords": [...], "headlines": [...]}} -> {cid: phrase}.
    Returns {} (caller keeps keyword labels) if the SDK/key is missing or errors."""
    try:
        import anthropic
    except Exception as e:
        print(f"[label] anthropic SDK unavailable ({e.__class__.__name__}); keeping keyword labels")
        return {}
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("[label] no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN set; keeping keyword labels")
        return {}
    import json
    blocks = []
    for cid, s in sorted(samples.items()):
        heads = "\n".join(f"     - {h}" for h in s["headlines"])
        blocks.append(f"Cluster {cid}\n   keywords: {', '.join(s['keywords']) or '(none)'}\n"
                      f"   sample headlines:\n{heads}")
    prompt = (
        "You are labelling clusters of Baltic & Nordic defence / geopolitics news "
        "for a monitoring dashboard.\n\n"
        "For EACH cluster below, write a short, specific, descriptive topic label: a "
        "natural noun phrase of about 3 to 6 words that captures what the cluster is "
        "about. Not a single keyword, not a full sentence. Make every label clearly "
        "distinct from the others. No trailing punctuation, no surrounding quotes.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn ONLY a JSON object mapping each cluster id (as a string) to its "
          'label, e.g. {"0": "Russia\'s war on Ukraine", '
          '"1": "NATO troop presence in the Baltics"}.'
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=LABEL_MODEL, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()   # tolerate fences
        data = json.loads(text)
        labels = {}
        for cid in samples:
            v = data.get(str(cid), data.get(cid))
            if isinstance(v, str) and v.strip():
                labels[cid] = v.strip().strip('"')
        print(f"[label] wrote {len(labels)}/{len(samples)} topic labels via {LABEL_MODEL}")
        return labels
    except Exception as e:
        print(f"[label] labelling failed ({e.__class__.__name__}: {e}); keeping keyword labels")
        return {}


def _llm_labels(samples):
    """Content-cached wrapper around _llm_call: only NEW/changed clusters hit the API."""
    if not samples:
        return {}
    import json
    import hashlib
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "labels.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    def key(s):
        blob = json.dumps(s, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    out, todo = {}, {}
    for cid, s in samples.items():
        k = key(s)
        if k in cache:
            out[cid] = cache[k]
        else:
            todo[cid] = (k, s)
    if todo:
        fresh = _llm_call({cid: s for cid, (k, s) in todo.items()})
        for cid, (k, s) in todo.items():
            if cid in fresh:
                out[cid] = fresh[cid]
                cache[k] = fresh[cid]
        if fresh:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            except Exception:
                pass
    return out


# ── stable topic identities across builds (topic tracking) ──────────────────
OTHER_ID = -1                       # the noise / "Other" bucket's fixed id
TOPIC_MATCH_MIN = 0.85             # cosine floor to call a new cluster the SAME topic as a stored one


def _topic_colors(idx):
    """Stable (light, dark) hex for a topic by its PERMANENT id — golden-angle hue, so a
    topic keeps its colour across builds and distinct topics stay well separated."""
    def hsl(h, s, l):
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c),
                   (x, 0, c), (c, 0, x)][int(h // 60) % 6]
        return "#%02x%02x%02x" % (round((r + m) * 255), round((g + m) * 255), round((b + m) * 255))
    h = (idx * 137.508) % 360
    return hsl(h, 0.62, 0.46), hsl(h, 0.68, 0.62)


def _track_topics(clusters, centroids, scope, model, run_date):
    """Give discovered clusters STABLE identities across builds.

    A topic's centroid + name + colour is persisted to analysis/emb_cache. On each build a
    newly-found cluster is matched to the most-similar stored topic (cosine ≥ TOPIC_MATCH_MIN,
    one-to-one, best pairs first); a match inherits that topic's id, NAME and COLOUR so they
    stay constant day to day, while an unmatched cluster becomes a brand-new topic. Long-dead
    topics are retired. Returns {local_id: {id, label, color, colorDark}}."""
    import json
    path = os.path.join(CACHE_DIR, f"topics__{scope}__{_slug(model)}.json")
    store = {"next": 0, "topics": []}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = {"next": 0, "topics": []}
    stored = store.get("topics", [])
    svecs = [np.asarray(t["centroid"], dtype=np.float32) for t in stored]
    n_before = len(stored)                                 # topics that existed before this build

    # rank all (new cluster, stored topic) pairs by cosine, then match greedily one-to-one
    pairs = []
    for cl in clusters:
        v = centroids[cl["id"]]
        for si, sv in enumerate(svecs):
            denom = (float(np.linalg.norm(v)) * float(np.linalg.norm(sv))) or 1.0
            pairs.append((float(v @ sv) / denom, cl["id"], si))
    pairs.sort(reverse=True)
    match, taken = {}, set()
    for cos, li, si in pairs:
        if cos < TOPIC_MATCH_MIN:
            break
        if li in match or si in taken:
            continue
        match[li] = si
        taken.add(si)

    result = {}
    for cl in clusters:
        li, v = cl["id"], centroids[cl["id"]]
        vround = [round(float(x), 5) for x in v]
        if li in match:                                    # existing topic — keep id/name/colour
            t = stored[match[li]]
            t["centroid"] = vround                         # refresh to current membership
            t["last"], t["misses"] = run_date, 0
        else:                                              # brand-new topic
            tid = store["next"]; store["next"] = tid + 1
            col, cold = _topic_colors(tid)
            t = {"id": tid, "centroid": vround, "label": cl["label"],
                 "color": col, "colorDark": cold,
                 "first": run_date, "last": run_date, "misses": 0}
            stored.append(t)
        result[li] = {"id": t["id"], "label": t["label"],
                      "color": t["color"], "colorDark": t["colorDark"]}

    for si in range(n_before):                             # age only PRE-EXISTING topics not matched
        if si not in taken:
            stored[si]["misses"] = int(stored[si].get("misses", 0)) + 1
    store["topics"] = [t for t in stored if int(t.get("misses", 0)) <= 120]   # retire long-dead
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)
    except Exception:
        pass
    return result


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
    raw = _reduce_noise(emb, raw)                      # fold border points out of noise
    # Up to 4 candidate terms per cluster, ranked KeyBERT-style — each phrase embedded with
    # the map's BGE model and scored by cosine to the cluster centroid (semantic centrality,
    # not word frequency). The LABEL joins the top 2; the list is kept as `keywords` chips.
    keywords = _keybert_labels(texts, raw, emb, model, topn=4)   # {raw_id: [term, ...]}
    tone = _tone(emb, model)                           # per-article escalation tone

    # Local cluster numbering 0..K-1 by size (largest first). Colours + STABLE cross-build
    # identities are assigned by _track_topics below; the noise bucket is handled separately.
    real = sorted((i for i in set(raw) if i != -1),
                  key=lambda c: -int((raw == c).sum()))
    remap = {c: i for i, c in enumerate(real)}
    K = len(real)

    # Cross-cluster term frequency: how many real clusters list each term. A label's
    # SECOND word should be a term DISTINCTIVE to this cluster (share == 1), so two
    # neighbouring topics never read as mere reorderings of the same shared words
    # ("ukraine · russia" vs "russia · ukraine").
    from collections import Counter
    share = Counter(t for c in real for t in set(keywords.get(c, [])))

    def _stems(t):
        return {w[:4] for w in t.split()}

    # Offline label = an identity term + a term distinctive to THIS cluster, joined with
    # " · " (phrases preferred for both) — e.g. "russia · long range". Kept unique by
    # extending on an exact collision. UPGRADED to a single fluent phrase by the LLM when
    # a key is available.
    used_labels = set()
    def _phrase_label(kw, i):
        if not kw:
            return f"topic {i + 1}"
        # identity term: a phrase if one is among the top candidates, else the top term
        lead = next((t for t in kw[:2] if " " in t), kw[0])
        lstem = _stems(lead)
        # distinctive second: the highest-RANKED term (c-TF-IDF order, so representative
        # not one-off) that is UNIQUE to this cluster (share == 1) and stem-distinct from
        # the lead; fall back to the next distinct term of any kind.
        rest = [t for t in kw if t != lead and not (_stems(t) & lstem)]
        second = (next((t for t in rest if share[t] == 1), None)  # top-ranked unique term
                  or (rest[0] if rest else None))                 # any distinct term
        parts = [lead] + ([second] if second else [])
        lab = " · ".join(parts)
        extra = [t for t in kw if t not in parts]
        while lab in used_labels and extra:             # disambiguate exact duplicates
            lab += " · " + extra.pop(0)
        used_labels.add(lab)
        return lab

    heads = df["headline"].fillna("").tolist()
    samples, clusters, centroids = {}, [], {}
    for i, c in enumerate(real):
        kw = keywords.get(c, [])
        members = [j for j in range(len(raw)) if raw[j] == c]
        cen = emb[members].mean(axis=0) if members else emb.mean(axis=0)
        cen = cen / (np.linalg.norm(cen) or 1.0)
        centroids[i] = cen                             # unit centroid, for topic tracking
        if members:                                    # representative headlines for the LLM prompt
            order = sorted(members, key=lambda j: -float(emb[j] @ cen))
            samples[i] = {"keywords": kw, "headlines": [heads[j] for j in order[:6] if heads[j]]}
        clusters.append({"id": i, "label": _phrase_label(kw, i),
                         "keywords": kw, "size": int((raw == c).sum())})
    # Upgrade the keyword labels to LLM-written descriptive phrases when possible.
    phrases = _llm_labels(samples)
    for cl in clusters:
        if cl["id"] in phrases:
            cl["label"] = phrases[cl["id"]]

    # Stable identities across builds: a topic keeps its id, name and colour day to day.
    run_date = str(max((str(d) for d in df["scrape_date"]), default=""))[:10]
    try:
        track = _track_topics(clusters, centroids, scope, model, run_date)
    except Exception as e:
        print(f"[topics] tracking unavailable ({type(e).__name__}); per-build identities")
        light, dark = _palette(K)
        track = {cl["id"]: {"id": cl["id"], "label": cl["label"],
                            "color": light[cl["id"]], "colorDark": dark[cl["id"]]}
                 for cl in clusters}

    local_to_stable, final = {}, []
    for cl in clusters:
        t = track[cl["id"]]
        local_to_stable[cl["id"]] = t["id"]
        final.append({"id": t["id"], "label": t["label"], "keywords": cl["keywords"],
                      "size": cl["size"], "color": t["color"], "colorDark": t["colorDark"]})
    final.sort(key=lambda c: -c["size"])               # legend order by size
    clusters = final
    if (raw == -1).any():                              # noise bucket -> fixed "Other" id
        clusters.append({"id": OTHER_ID, "label": "Other", "keywords": [],
                         "size": int((raw == -1).sum()),
                         "color": NOISE_LIGHT, "colorDark": NOISE_DARK})

    pts = []
    for i, r in enumerate(df.itertuples()):
        loc = remap.get(int(raw[i]))                   # local cluster id, or None for noise
        cid = local_to_stable[loc] if loc is not None else OTHER_ID
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
/* topic-river streamgraph */
.emb-stream{padding:6px 20px 10px;position:relative;}
.emb-stream h4{margin:0 0 4px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);}
.stream-svg{display:block;width:100%;height:220px;overflow:visible;}
.stream-band{stroke:var(--panel);stroke-width:.6;transition:opacity .12s;cursor:pointer;}
.stream-band.dim{opacity:.22;}
.stream-topics{display:flex;flex-wrap:wrap;gap:5px;margin:0 0 9px;}
.stream-topics .stx{display:inline-flex;align-items:center;gap:5px;font-size:11px;padding:2px 9px;
  border:1px solid var(--border);border-radius:20px;cursor:pointer;color:var(--ink);
  user-select:none;background:var(--panel);}
.stream-topics .stx .sw{width:9px;height:9px;border-radius:50%;flex:0 0 auto;}
.stream-topics .stx.off{opacity:.4;}
.stream-topics .stx:hover{border-color:var(--accent);}
.stream-axis{display:flex;justify-content:space-between;font-size:10.5px;color:var(--faint);
  font-variant-numeric:tabular-nums;margin-top:2px;}
.stream-ctrl{display:flex;align-items:center;gap:10px;margin:2px 0 8px;flex-wrap:wrap;
  font-size:11.5px;color:var(--faint);}
.stream-ctrl .sc-lab{letter-spacing:.08em;text-transform:uppercase;}
.stream-ctrl .sc-range{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--muted);}
.seg{display:inline-flex;border:1.5px solid var(--gold);border-radius:6px;overflow:hidden;}
.seg button{background:var(--panel);color:var(--ink);border:0;padding:4px 13px;font-size:12px;
  cursor:pointer;font-weight:600;}
.seg button+button{border-left:1.5px solid var(--gold);}
.seg button.on{background:var(--gold);color:#2b140f;}
/* dual-thumb range: two overlaid inputs with a shared track */
.stream-slider{position:relative;height:20px;margin-top:5px;}
.stream-slider::before{content:"";position:absolute;left:0;right:0;top:8px;height:4px;
  background:var(--border2);border-radius:2px;}
.stream-slider input[type=range]{position:absolute;left:0;top:0;width:100%;height:20px;margin:0;
  -webkit-appearance:none;appearance:none;background:transparent;pointer-events:none;}
.stream-slider input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;pointer-events:auto;
  width:15px;height:15px;border-radius:50%;background:var(--accent);border:2px solid var(--panel);
  cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.4);}
.stream-slider input[type=range]::-moz-range-thumb{pointer-events:auto;width:15px;height:15px;
  border-radius:50%;background:var(--accent);border:2px solid var(--panel);cursor:pointer;}
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
.tone-sub{margin:0 0 12px;font-size:12px;color:var(--muted);line-height:1.5;}
.tone-line{display:block;width:100%;height:56px;margin:0 0 6px;overflow:visible;}
.tone-heat{display:grid;gap:2px;overflow-x:auto;font-size:11px;}
.tone-heat .th-row{display:grid;grid-template-columns:var(--labw,210px) 1fr;gap:6px;align-items:center;}
.tone-heat .th-lab{color:var(--ink);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  text-align:right;font-size:11.5px;}
.tone-heat .th-cells{display:grid;gap:2px;grid-auto-flow:column;grid-auto-columns:1fr;}
.tone-heat .th-cell{height:22px;border-radius:3px;background:var(--border2);cursor:default;
  display:flex;align-items:center;justify-content:center;font-size:9.5px;color:rgba(0,0,0,.5);}
.tone-heat .th-head .th-cell{background:none;color:var(--faint);font-size:10px;letter-spacing:.03em;height:16px;}
.tone-heat .th-head .th-lab{color:var(--faint);font-weight:400;}
.tone-heat .th-all{padding-bottom:4px;margin-bottom:2px;border-bottom:1px solid var(--border);}
.tone-heat .th-all .th-cell{height:26px;font-size:10.5px;font-weight:600;}
.tone-heat .th-cell.th-hit{cursor:pointer;}
.tone-heat .th-cell[data-m]:hover{outline:2px solid var(--ink);outline-offset:-2px;}
.tone-heat .th-cell.sel{outline:2.5px solid var(--ink);outline-offset:-2px;box-shadow:0 0 0 2px var(--panel);}
/* click-to-list articles for a tone cell */
.tone-arts{margin-top:12px;}
.tone-arts:empty{display:none;}
.tone-arts .ta-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted);
  padding:8px 12px;background:color-mix(in srgb,var(--gold) 10%,var(--panel));border:1.5px solid var(--border);
  border-radius:6px 6px 0 0;font-variant-numeric:tabular-nums;}
.tone-arts .ta-head b{color:var(--ink);}
.tone-arts .ta-x{margin-left:auto;background:none;border:0;color:var(--faint);cursor:pointer;font-size:14px;
  line-height:1;padding:2px 4px;}
.tone-arts .ta-x:hover{color:var(--accent);}
.tone-arts .ta-list{list-style:none;margin:0;padding:2px 12px 6px;border:1.5px solid var(--border);border-top:0;
  border-radius:0 0 6px 6px;}
.tone-arts .ta-list li{padding:7px 0;border-top:1px solid var(--border2);font-size:12.5px;}
.tone-arts .ta-list li:first-child{border-top:0;}
.tone-arts .ta-list a{color:var(--muted);text-decoration:none;display:flex;align-items:baseline;gap:8px;}
.tone-arts .ta-list a:hover{color:var(--accent);}
.tone-arts .ta-tone{flex:none;font-size:10px;font-weight:700;font-variant-numeric:tabular-nums;
  padding:1px 6px;border-radius:3px;min-width:34px;text-align:center;}
.tone-arts .ta-list .src{display:block;font-size:10px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);margin-top:2px;}
"""

_EMB_SECTION = """
    <p class="section-h"><svg class="sec-sign" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="7" r="2.2"/><circle cx="14" cy="6" r="2.2"/><circle cx="10" cy="14" r="2.2"/><path d="M7.8 8.2 12 12M12.2 7.2 11 12"/></svg> Discovered topics — clusters found by meaning</p>
    <section class="panel">
      <div class="panel-h">
        <h2>Semantic topics &amp; anomalies</h2>
        <span class="hint mono">topics discovered from the text</span>
      </div>
      <div style="padding:14px 20px 0">
        <p class="emb-note">Built from scratch from the article <b>embeddings</b> (BGE), <b>not</b> the keyword buckets. HDBSCAN groups the stories into <b>data-driven clusters</b>, each named by the key phrase whose meaning sits closest to the cluster (embedding similarity × distinctiveness). Topics are <b>tracked across days</b>, so a topic keeps the same name and colour build to build. Embeds each story's <b>headline + body</b>; off-topic stories fall out as <i>Other</i>.</p>
        <div class="emb-tabs" id="embTabs"></div>
      </div>
      <div class="emb-wrap">
        <canvas class="emb-canvas" id="embCanvas"></canvas>
        <div class="emb-legend" id="embLegend"></div>
        <div class="emb-tip" id="embTip"></div>
      </div>
      <div class="emb-stream">
        <h4>Topic river — what's rising, fading, appearing</h4>
        <p class="tone-sub">Each band is a discovered topic; thickness = articles that period. <b>Select topics</b> with the chips, pick the <b>grain</b>, drag the <b>range handles</b> to zoom, and <b>click a band</b> to read its articles. Bands <b>swell, shrink, appear and vanish</b>; hover one for its count.</p>
        <div class="stream-ctrl">
          <span class="sc-lab">grain</span>
          <div class="seg" id="streamGrain">
            <button type="button" data-g="day">Day</button>
            <button type="button" data-g="week" class="on">Week</button>
            <button type="button" data-g="month">Month</button>
          </div>
          <span class="sc-range" id="streamSpan"></span>
        </div>
        <div class="stream-topics" id="streamTopics"></div>
        <svg class="stream-svg" id="topicStream" preserveAspectRatio="none"></svg>
        <div class="stream-slider" id="streamSlider">
          <input type="range" id="streamLo" min="0" max="1" value="0" aria-label="range start">
          <input type="range" id="streamHi" min="0" max="1" value="1" aria-label="range end">
        </div>
        <div class="stream-axis" id="streamAxis"></div>
        <div class="emb-tip" id="streamTip"></div>
        <div class="tone-arts" id="streamArts"></div>
      </div>
      <div class="emb-tone">
        <h4>Escalation tone — is coverage heating up? <span class="tone-key"><i class="tk-neg"></i>de-escalation<i class="tk-mid"></i>baseline<i class="tk-pos"></i>escalation</span></h4>
        <p class="tone-sub">Each story scored on an <b>escalation vs de-escalation</b> axis (cosine to anchor phrases in the same embedding space), <i>relative to this feed's own baseline</i>. Rows = discovered topics by week; the top <b>All coverage</b> row is the overall trend. Colour = mean tone; deeper red = more escalatory. Number = articles that week — <b>click any number to list those articles</b>.</p>
        <div class="tone-heat" id="toneHeat"></div>
        <div class="emb-tip" id="toneTip"></div>
        <div class="tone-arts" id="toneArts"></div>
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
  const EV = __EMB_DATA__;                       // {content:{points,clusters,...}}
  const scopes = Object.keys(EV).filter(k=>EV[k] && EV[k].points);
  if(!scopes.length) return;
  const ANOM_N = 20;                             // how many most-isolated stories to list
  const isDark=()=>{const t=document.documentElement.getAttribute('data-theme');
    return t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme:dark)').matches);};
  let cur = scopes[0];
  // start with every cluster visible; click a legend entry to toggle it off/on
  const hidden = new Set();                         // hidden CLUSTER ids
  let CL=[], COL=[], NAME=[];                     // per-scope clusters / colour map / label map
  const cvs=document.getElementById('embCanvas'), ctx=cvs.getContext('2d');
  const tip=document.getElementById('embTip'), wrap=cvs.parentElement;
  let P=[], hover=-1, minX,maxX,minY,maxY;

  // make the legend draggable so it never permanently hides dots; a real drag
  // (moved past a few px) suppresses the click that would otherwise toggle a cluster.
  function makeDraggable(el){
    let sx,sy,ox,oy,pid,down=false,dragging=false,moved=false;
    el.style.cursor='grab'; el.title='drag to move';
    el.addEventListener('pointerdown',e=>{
      down=true; dragging=false; moved=false; pid=e.pointerId;
      sx=e.clientX; sy=e.clientY;
      const r=el.getBoundingClientRect(), pr=el.offsetParent.getBoundingClientRect();
      ox=r.left-pr.left; oy=r.top-pr.top;
    });
    el.addEventListener('pointermove',e=>{
      if(!down)return;
      const dx=e.clientX-sx, dy=e.clientY-sy;
      if(!dragging && Math.abs(dx)+Math.abs(dy)>4){        // only NOW does it become a drag
        dragging=true; moved=true;
        el.style.right='auto'; el.style.left=ox+'px'; el.style.top=oy+'px';
        el.style.cursor='grabbing'; el.setPointerCapture(pid);
      }
      if(dragging){ el.style.left=(ox+dx)+'px'; el.style.top=(oy+dy)+'px'; }
    });
    const end=()=>{down=false; if(dragging){el.style.cursor='grab';
      try{el.releasePointerCapture(pid);}catch(_){}} dragging=false;};
    el.addEventListener('pointerup',end);
    el.addEventListener('pointercancel',end);
    el.addEventListener('click',e=>{if(moved){e.stopPropagation();e.preventDefault();moved=false;}},true);
  }
  makeDraggable(document.getElementById('embLegend'));

  // only show the scope switcher when there's more than one scope
  if(scopes.length>1){
    document.getElementById('embTabs').innerHTML = scopes.map(s=>{
      const v=EV[s]; const lab = s==='content'?'Content (headline + body)':'Headline only';
      return `<div class="emb-tab${s===cur?' on':''}" data-s="${s}">${lab} · ${v.points.length} stories</div>`;
    }).join('');
    document.querySelectorAll('#embTabs .emb-tab').forEach(t=>{
      t.onclick=()=>{cur=t.dataset.s; hidden.clear();
        document.querySelectorAll('#embTabs .emb-tab').forEach(x=>x.classList.toggle('on',x.dataset.s===cur));
        load(); };
    });
  }

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
  // topic river: interactive centred streamgraph, bucketed CLIENT-SIDE from the points so
  // grain (day/week/month), date range and topic filter are all live. Thickness = articles
  // that period; bands swell, shrink, appear and vanish. Topic filter reuses the map legend.
  let streamGrain='week', streamSel=null;    // streamSel: {g, B, lo, hi} selected bucket window
  const streamHidden=new Set();              // topic ids deselected in the river (its own filter)
  const dkey=(d,g)=>{ d=d.slice(0,10);
    if(g==='day') return d;
    if(g==='month') return d.slice(0,7);
    const [y,m,da]=d.split('-').map(Number); const dt=new Date(Date.UTC(y,m-1,da));
    dt.setUTCDate(dt.getUTCDate()-((dt.getUTCDay()+6)%7)); return dt.toISOString().slice(0,10); };
  const dlabel=(k,g)=>{ if(g==='month'){const [y,mo]=k.split('-');
      return new Date(+y,+mo-1,1).toLocaleDateString('en',{month:'short',year:'2-digit'});}
    return new Date(k+'T00:00').toLocaleDateString('en',{month:'short',day:'numeric'}); };
  function spanKeys(minD,maxD,g){                        // contiguous bucket keys min..max (gaps filled)
    const keys=[];
    if(g==='month'){let [y,m]=minD.slice(0,7).split('-').map(Number);
      const [ey,em]=maxD.slice(0,7).split('-').map(Number);
      let guard=0; while((y<ey||(y===ey&&m<=em))&&guard++<600){keys.push(`${y}-${String(m).padStart(2,'0')}`);
        if(++m>12){m=1;y++;}} return keys;}
    const step=g==='day'?1:7; const add=(k,n)=>{const dt=new Date(k+'T00:00:00Z');
      dt.setUTCDate(dt.getUTCDate()+n); return dt.toISOString().slice(0,10);};
    let cur=dkey(minD,g); const end=dkey(maxD,g); let guard=0;
    while(guard++<3000){keys.push(cur); if(cur>=end)break; cur=add(cur,step);} return keys;
  }
  function renderStream(v){
    const svg=document.getElementById('topicStream'), axis=document.getElementById('streamAxis');
    const tip=document.getElementById('streamTip'), spanEl=document.getElementById('streamSpan');
    const loEl=document.getElementById('streamLo'), hiEl=document.getElementById('streamHi');
    if(!svg) return;
    const pts=(v.points||[]).filter(p=>p.c!==-1 && p.d);
    if(pts.length<2){svg.innerHTML='';if(axis)axis.innerHTML='';return;}
    const g=streamGrain;
    const dates=pts.map(p=>p.d.slice(0,10)).sort();
    const keys=spanKeys(dates[0],dates[dates.length-1],g), B=keys.length;
    if(!streamSel||streamSel.g!==g||streamSel.B!==B) streamSel={g,B,lo:0,hi:B-1};
    if(loEl&&hiEl){loEl.max=hiEl.max=B-1; loEl.value=streamSel.lo; hiEl.value=streamSel.hi;}
    const lo=streamSel.lo, hi=streamSel.hi, selKeys=keys.slice(lo,hi+1), W=selKeys.length;
    const kof={}; selKeys.forEach((k,i)=>kof[k]=i);
    if(spanEl) spanEl.textContent=`${dlabel(keys[lo],g)} – ${dlabel(keys[hi],g)} · ${W} ${g}${W>1?'s':''}`;
    const topics=CL.filter(c=>c.id!==-1 && !streamHidden.has(c.id));
    const cnt={}; topics.forEach(t=>cnt[t.id]=new Array(W).fill(0));
    pts.forEach(p=>{const i=kof[dkey(p.d,g)]; if(i!=null&&cnt[p.c]) cnt[p.c][i]++;});
    let bands=topics.map(t=>({id:t.id,name:NAME[t.id],color:COL[t.id],cnt:cnt[t.id],
        total:cnt[t.id].reduce((a,b)=>a+b,0)})).filter(b=>b.total>0);
    if(!bands.length||W<2){svg.innerHTML='<text x="10" y="26" fill="'+css('--faint')+'" font-size="13">Not enough data in this range — widen it or change the grain.</text>';if(axis)axis.innerHTML='';return;}
    bands.sort((a,b)=>b.total-a.total);
    const ordered=[]; bands.forEach((t,i)=> i%2?ordered.push(t):ordered.unshift(t));  // big topics centred
    const Wpx=1000, Hpx=220, pad=8; svg.setAttribute('viewBox',`0 0 ${Wpx} ${Hpx}`);
    const maxTot=Math.max(1,...selKeys.map((_,w)=>ordered.reduce((a,t)=>a+t.cnt[w],0)));
    const ys=(Hpx-2*pad)/maxTot, xat=w=> W>1?(w/(W-1))*Wpx:Wpx/2;
    const top=ordered.map(()=>[]), bot=ordered.map(()=>[]);
    for(let w=0;w<W;w++){const tot=ordered.reduce((a,t)=>a+t.cnt[w],0); let cur=(Hpx-tot*ys)/2;
      ordered.forEach((t,k)=>{top[k][w]=cur; cur+=t.cnt[w]*ys; bot[k][w]=cur;});}
    let html='';
    ordered.forEach((t,k)=>{let d='';
      for(let w=0;w<W;w++) d+=(w?'L':'M')+xat(w).toFixed(1)+' '+top[k][w].toFixed(1);
      for(let w=W-1;w>=0;w--) d+='L'+xat(w).toFixed(1)+' '+bot[k][w].toFixed(1);
      html+=`<path class="stream-band" d="${d}Z" fill="${t.color}" data-k="${k}"></path>`;});
    svg.innerHTML=html;
    if(axis) axis.innerHTML=`<span>${dlabel(selKeys[0],g)}</span><span>${dlabel(selKeys[(W-1)>>1],g)}</span><span>${dlabel(selKeys[W-1],g)}</span>`;
    const paths=[...svg.querySelectorAll('.stream-band')];
    paths.forEach(el=>{const t=ordered[+el.dataset.k];
      el.onmousemove=e=>{const wr=el.closest('.emb-stream'), rc=svg.getBoundingClientRect();
        const frac=Math.max(0,Math.min(1,(e.clientX-rc.left)/rc.width)), w=Math.round(frac*(W-1));
        paths.forEach(p=>p.classList.toggle('dim',p!==el));
        tip.innerHTML=`<b>${t.name}</b><span class="m">${dlabel(selKeys[w],g)} · ${t.cnt[w]} ${t.cnt[w]===1?'story':'stories'}<br>${t.total} in view</span>`;
        tip.style.opacity=1;
        const pr=wr.getBoundingClientRect(); let tx=e.clientX-pr.left+14, ty=e.clientY-pr.top+14;
        if(tx+180>wr.clientWidth)tx=e.clientX-pr.left-190; tip.style.left=tx+'px'; tip.style.top=ty+'px';};
      el.onmouseleave=()=>{paths.forEach(p=>p.classList.remove('dim')); tip.style.opacity=0;};
      el.onclick=()=>{                                   // click a band -> list its articles in view
        const arts=document.getElementById('streamArts');
        if(arts.dataset.open===String(t.id)){arts.innerHTML='';arts.dataset.open='';return;}
        arts.dataset.open=String(t.id);
        const list=v.points.filter(p=>p.c===t.id && kof[dkey(p.d,g)]!=null)
          .sort((a,b)=>(a.d<b.d?1:a.d>b.d?-1:0));
        const items=list.map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}`
          +`<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
        arts.innerHTML=`<div class="ta-head"><b>${t.name}</b> · ${dlabel(selKeys[0],g)}–${dlabel(selKeys[W-1],g)}`
          +` · ${list.length} ${list.length===1?'story':'stories'}`
          +`<button class="ta-x" aria-label="close">✕</button></div><ul class="ta-list">${items}</ul>`;
        arts.querySelector('.ta-x').onclick=()=>{arts.innerHTML='';arts.dataset.open='';};
        arts.scrollIntoView({behavior:'smooth',block:'nearest'});
      };
    });
  }
  // dedicated topic selector for the river (chips); toggling re-renders the stream
  function renderChips(v){
    const box=document.getElementById('streamTopics'); if(!box) return;
    const topics=CL.filter(c=>c.id!==-1);
    box.innerHTML=topics.map(c=>`<span class="stx${streamHidden.has(c.id)?' off':''}" data-c="${c.id}" tabindex="0">`
      +`<span class="sw" style="background:${COL[c.id]}"></span>${NAME[c.id]}</span>`).join('');
    box.querySelectorAll('.stx').forEach(el=>{
      const toggle=()=>{const id=+el.dataset.c; streamHidden.has(id)?streamHidden.delete(id):streamHidden.add(id);
        el.classList.toggle('off'); renderStream(v);};
      el.onclick=toggle;
      el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}};
    });
  }
  // wire the grain buttons + range handles once; re-render on change
  function setupStream(){
    document.querySelectorAll('#streamGrain button').forEach(b=>{
      b.classList.toggle('on', b.dataset.g===streamGrain);
      b.onclick=()=>{streamGrain=b.dataset.g; streamSel=null;
        document.querySelectorAll('#streamGrain button').forEach(x=>x.classList.toggle('on',x.dataset.g===streamGrain));
        renderStream(EV[cur]);};
    });
    const loEl=document.getElementById('streamLo'), hiEl=document.getElementById('streamHi');
    if(loEl&&hiEl){
      const onR=()=>{ if(!streamSel) return; let a=+loEl.value, b=+hiEl.value;
        if(a>b){const t=a;a=b;b=t;} if(b-a<1){ if(b<streamSel.B-1)b=a+1; else a=b-1; }
        streamSel.lo=a; streamSel.hi=b; renderStream(EV[cur]); };
      loEl.oninput=onR; hiEl.oninput=onR;
    }
  }
  function renderTone(v){
    const T=v.tone, heatEl=document.getElementById('toneHeat');
    if(!T||!T.weeks||!T.weeks.length){heatEl.innerHTML='';return;}
    const W=T.weeks.length;
    const rowHtml=(id,label,color,cells,cls)=>{
      const byW={}; cells.forEach(c=>byW[c.w]=c);
      let out='';
      for(let i=0;i<W;i++){const c=byW[i];
        out += c
          ? `<div class="th-cell th-hit" style="background:${toneColor(c.m)};color:${toneInk(c.m)}" data-c="${id}" data-l="${label}" data-w="${i}" data-m="${c.m}" data-n="${c.n}">${c.n}</div>`
          : `<div class="th-cell" style="background:transparent"></div>`;}
      return `<div class="th-row ${cls||''}"><div class="th-lab" title="${label}" style="color:${color||'var(--faint)'}">${label}</div><div class="th-cells">${out}</div></div>`;
    };
    let html = `<div class="th-row th-head"><div class="th-lab">topic</div><div class="th-cells">`
      + T.weeks.map(w=>`<div class="th-cell">${wk(w)}</div>`).join('') + `</div></div>`;
    html += rowHtml(-1,'All coverage','var(--ink)',T.line,'th-all');
    T.heat.forEach(r=> html += rowHtml(r.c, NAME[r.c], COL[r.c], r.cells));
    heatEl.innerHTML = html;
    const tt=document.getElementById('toneTip');
    const artsEl=document.getElementById('toneArts'); artsEl.innerHTML=''; let openKey=null;
    // Monday-start ISO week for a 'YYYY-MM-DD' date — matches _week_start() server-side
    const weekStart=d=>{const [y,m,da]=d.slice(0,10).split('-').map(Number);
      const dt=new Date(Date.UTC(y,m-1,da)); dt.setUTCDate(dt.getUTCDate()-((dt.getUTCDay()+6)%7));
      return dt.toISOString().slice(0,10);};
    heatEl.querySelectorAll('.th-cell[data-m]').forEach(el=>{
      el.onmousemove=e=>{const wr=el.closest('.emb-tone'), rc=wr.getBoundingClientRect();
        const m=+el.dataset.m;
        tt.innerHTML=`<b>${el.dataset.l}</b><span class="m">${wk(T.weeks[+el.dataset.w])} · ${el.dataset.n} stories<br>mean tone ${m>0?'+':''}${m.toFixed(2)} · click to list</span>`;
        tt.style.opacity=1;
        let tx=e.clientX-rc.left+14, ty=e.clientY-rc.top+14;
        if(tx+180>wr.clientWidth)tx=e.clientX-rc.left-190; tt.style.left=tx+'px'; tt.style.top=ty+'px';};
      el.onmouseleave=()=>{tt.style.opacity=0;};
      el.onclick=()=>{
        const cid=+el.dataset.c, wi=+el.dataset.w, key=cid+':'+wi, wkStr=T.weeks[wi];
        heatEl.querySelectorAll('.th-cell.sel').forEach(x=>x.classList.remove('sel'));
        if(openKey===key){openKey=null; artsEl.innerHTML=''; return;}   // toggle closed
        openKey=key; el.classList.add('sel');
        const m=+el.dataset.m;
        const arts=v.points.filter(p=>(cid===-1||p.c===cid)&&weekStart(p.d)===wkStr)
          .sort((a,b)=>(a.d<b.d?1:a.d>b.d?-1:0));
        const items=arts.map(p=>{
          const t=p.to, sign=t>0?'+':'', lab=t>=.15?'escalatory':t<=-.15?'de-escalation':'neutral';
          return `<li><a href="${p.u}" target="_blank" rel="noopener">`
            +`<span class="ta-tone" style="background:${toneColor(t)};color:${toneInk(t)}" title="tone ${sign}${t.toFixed(2)} · ${lab}">${sign}${t.toFixed(2)}</span>`
            +`${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`;}).join('');
        artsEl.innerHTML=`<div class="ta-head"><b>${el.dataset.l}</b> · week of ${wk(wkStr)}`
          +` · ${arts.length} ${arts.length===1?'story':'stories'} · mean tone ${m>0?'+':''}${m.toFixed(2)}`
          +`<button class="ta-x" aria-label="close">✕</button></div><ul class="ta-list">${items}</ul>`;
        artsEl.querySelector('.ta-x').onclick=()=>{openKey=null; artsEl.innerHTML='';
          heatEl.querySelectorAll('.th-cell.sel').forEach(x=>x.classList.remove('sel'));};
        artsEl.scrollIntoView({behavior:'smooth',block:'nearest'});
      };
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
        el.classList.toggle('off'); draw();};              // map legend filters the map only
    });
    // discovered-topics list: click a topic to expand the articles inside it
    document.getElementById('embTopH').textContent=`Discovered topics — ${CL.filter(c=>c.label!=='Other').length} clusters (click to list articles)`;
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
      const nm=NAME[p.c]&&NAME[p.c]!=='Other'?`<span class="emb-badge">${NAME[p.c]}</span>`:'';
      return `<li><a href="${p.u}" target="_blank" rel="noopener">${nm}${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`;
    }).join('');
    const anom=v.points.slice().sort((a,b)=>b.iso-a.iso).slice(0,ANOM_N);
    document.getElementById('embAnomH').textContent=`Anomalies — the ${anom.length} most isolated stories`;
    document.getElementById('embAnom').innerHTML = anom.length?li(anom):'<li>none</li>';
    renderChips(v); renderStream(v); setupStream();
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
