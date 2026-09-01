#!/usr/bin/env python
"""
Data pipeline for the Baltics Monitor: load -> filter -> fetch -> analyse.

This is the number-crunching half (no HTML here — that's render.py):

  load_data()      read + dedupe the three source CSVs (repairs lrt_lt mojibake)
  filter_themes()  keep only headlines that hit the defence KEYWORDS
  compute_shares() per-country "how much of the news is defence?" for act 1
  fetch_all()      download each matched article's body (8 in parallel)
  analyse()        keyword-bucket theme tagging + act-3 headline word frequencies
  build_payload()  pack everything the front-end needs into one JSON-able dict

No ML here: themes are the hand-authored keyword buckets in config.THEMES (regex,
multi-label), not clustering. Semantic topic DISCOVERY (embeddings -> UMAP ->
HDBSCAN) lives in embeddings_analysis.py.
"""
import re
import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    FILES, SRC_CODE, SRC_LABEL, SRC_HOME, KEYWORDS, HEADERS, BOILERPLATE,
    EXTRA_STOP, FREQ_STOP, canon, THEMES, THEME_OTHER, THEME_PATS, THEME_RANK,
    COLORS_LIGHT, COLORS_DARK, BALTIC_MAP,
)


try:
    import ftfy as _ftfy
except Exception:                                     # optional; falls back to the heuristic below
    _ftfy = None


def fix_mojibake(s):
    """Repair mis-decoded / double-encoded text in scraped headlines.

    Prefers ftfy (fixes the full range — 'NausÄda' -> 'Nausėda', 'chiefâ€™s' -> 'chief's');
    falls back to a cp1252->utf-8 heuristic when ftfy isn't installed."""
    if not isinstance(s, str):
        return ""
    if _ftfy is not None:
        try:
            return _ftfy.fix_text(s)
        except Exception:
            pass
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
    # Dedupe on URL only: this drops exact repeats but KEEPS re-slugged copies of a
    # story whose headline ERR edited (same numeric id, new slug) — those edits are
    # editorially interesting and we want to see them as distinct rows.
    df = df.dropna(subset=["url", "headline"]).drop_duplicates("url").reset_index(drop=True)
    return df


def _now_ams():
    """(short, full) Amsterdam-time stamps for the SAME instant (DST-aware).
      short: 'YYYY-MM-DD HH:MM Amsterdam'  (for the header tile)
      full : 'YYYY-MM-DD HH:MM:SS CEST (UTC+02:00) · Europe/Amsterdam'  (for the tooltip)
    Falls back to UTC if zoneinfo is unavailable."""
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        a = now.astimezone(ZoneInfo("Europe/Amsterdam"))
        off = a.strftime("%z")                                # +0200
        off = off[:3] + ":" + off[3:] if len(off) == 5 else off
        short = a.strftime("%Y-%m-%d %H:%M Amsterdam")
        full = f"{a.strftime('%Y-%m-%d %H:%M:%S')} {a.strftime('%Z')} (UTC{off}) · Europe/Amsterdam"
        return short, full
    except Exception:
        s = now.strftime("%Y-%m-%d %H:%M UTC")
        return s, now.strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_ams(dt_str):
    """A UTC 'YYYY-MM-DD_HH.MM.SS' scrape stamp -> 'DD Mon HH:MM' in Amsterdam time
    (DST-aware). Scrapes run in CI on a UTC clock, so the raw stamp is UTC; the
    'Amsterdam time' label lives in the section caption. Falls back to the date."""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.datetime.strptime(str(dt_str), "%Y-%m-%d_%H.%M.%S")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(ZoneInfo("Europe/Amsterdam")).strftime("%d %b %H:%M")
    except Exception:
        return str(dt_str)[:10]


def find_edits():
    """Same story, changed headline. ERR re-slugs a URL when it edits a headline
    while keeping the numeric article id (news.err.ee/<id>/<slug>); load_data
    dedupes by URL, so BOTH versions survive as distinct rows. Group by that id,
    keep articles seen under >1 distinct headline, and return them newest-change-
    first, each with its ordered headline versions and when each was first captured.
    Only DEFENCE/geopolitics stories are reported (an id counts if ANY of its headline
    versions hits KEYWORDS) — headline edits on unrelated news aren't interesting here.
    (Only ERR exposes a stable per-article id, so only ERR edits are detected.)"""
    frames = []
    for src, path in FILES.items():
        try:
            d = pd.read_csv(path, encoding="utf-8")
        except Exception:
            continue
        d["source"] = src
        d["headline"] = d["headline"].map(fix_mojibake).str.replace("�", " ", regex=False).str.strip()
        d["aid"] = d["url"].str.extract(r"news\.err\.ee/(\d+)")[0] if src == "err_ee" else None
        frames.append(d)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True).dropna(subset=["aid", "url", "headline"])
    when = "scrape_datetime" if "scrape_datetime" in df.columns else "scrape_date"
    df = df.sort_values(when).drop_duplicates("url")          # first-seen row per URL
    # keep only ids that are defence-related in at least one of their headline versions
    matched = df["headline"].str.contains(KEYWORDS, case=False, na=False, regex=True)
    defence_ids = set(df.loc[matched, "aid"])
    edits = []
    for aid, g in df.groupby("aid"):
        if aid not in defence_ids:                            # skip non-defence stories
            continue
        g = g.sort_values(when)
        versions, seen, last_raw = [], set(), ""
        for r in g.itertuples():
            if r.headline in seen:
                continue
            seen.add(r.headline)
            last_raw = str(getattr(r, when))                  # raw UTC stamp, sortable
            versions.append({"h": r.headline, "u": r.url, "d": str(r.scrape_date),
                             "t": _fmt_ams(getattr(r, when))})   # 'DD Mon HH:MM' Amsterdam
        if len(versions) < 2:                                 # unchanged -> not an edit
            continue
        edits.append({"id": str(aid), "source": SRC_LABEL.get(g.iloc[0]["source"], ""),
                      "n": len(versions), "versions": versions, "last": last_raw})
    edits.sort(key=lambda e: e["last"], reverse=True)         # newest change first (raw UTC)
    print(f"[edits] {len(edits)} article(s) had their headline revised after publishing")
    return edits


def filter_themes(df):
    mask = df["headline"].str.contains(KEYWORDS, case=False, na=False, regex=True)
    out = df[mask].reset_index(drop=True)
    print(f"[filter] {mask.sum()} / {len(df)} articles match the defence/geopolitics themes")
    return out


def compute_shares(df_all, df_matched):
    """Per-country share of headlines that touch defence/geopolitics."""
    tot = df_all.groupby("source").size()
    mat = df_matched.groupby("source").size()
    rows = []
    for src in FILES:
        t = int(tot.get(src, 0))
        m = int(mat.get(src, 0))
        dates = df_all.loc[df_all["source"] == src, "scrape_date"].dropna().astype(str)
        rows.append({
            "code": SRC_CODE[src], "country": SRC_LABEL[src], "matched": m, "total": t,
            "pct": round(m / t * 100, 1) if t else 0.0,
            "d0": dates.min() if len(dates) else "",   # earliest scrape date seen
            "d1": dates.max() if len(dates) else "",   # latest scrape date seen
        })
    rows.sort(key=lambda r: r["pct"], reverse=True)
    for r in rows:
        print(f"[share] {r['country']}: {r['matched']}/{r['total']} = {r['pct']}% "
              f"defence  ({r['d0']} → {r['d1']})")
    return rows


def fetch_body(url):
    """Best-effort article-body extraction: join the meaningful <p> tags."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.content, "html.parser")   # bytes -> BS detects charset (avoids mojibake)
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
    # Article -> theme is pure keyword-bucket matching (config.THEMES regex); no TF-IDF,
    # SVD or t-SNE. Semantic topic DISCOVERY lives in embeddings_analysis.py. We keep
    # sklearn's English stopword set only for the act-3 headline word-frequency tally.
    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)

    # Multi-label: tag each article with EVERY theme its keywords hit (headline x3),
    # pick the strongest as the colour/primary. No headline hit anywhere -> keep only
    # the single best body theme; nothing at all -> "Other defence".
    OTHER_ID = len(THEMES)

    def assign(h, b):
        hits = [(len(p.findall(h)), len(p.findall(b))) for p in THEME_PATS]
        members = [i for i, (hh, _) in enumerate(hits) if hh > 0]
        if not members:
            body = [i for i, (_, bb) in enumerate(hits) if bb > 0]
            if body:
                members = [max(body, key=lambda i: (hits[i][1], THEME_RANK[i], -i))]
            else:
                return OTHER_ID, [OTHER_ID]
        # primary: most headline hits, then body hits, then rank (NATO loses), then order
        primary = max(members, key=lambda i: (hits[i][0], hits[i][1], THEME_RANK[i], -i))
        return primary, sorted(members)

    assigned = [assign(h, b) for h, b in
                zip(df["headline"].fillna(""), df["body"].fillna(""))]
    df["theme"] = [p for p, _ in assigned]          # primary theme (colour)
    df["themes"] = [m for _, m in assigned]          # all themes it belongs to

    # Per-theme metadata: MEMBERSHIP count (an article counts under each of its themes).
    names = [t[0] for t in THEMES] + [THEME_OTHER]
    members_col = df["themes"].tolist()
    themes = []
    for tid in range(len(THEMES) + 1):
        idx = [i for i, ms in enumerate(members_col) if tid in ms]
        if not idx:
            continue
        themes.append({"id": tid, "label": names[tid], "count": len(idx)})
        print(f"[theme] {names[tid]} ({len(idx)})")
    themes.sort(key=lambda t: t["count"], reverse=True)

    # Global keyword frequency (act 3): in how many article HEADLINES each merged term
    # appears (titles only, not body), and which articles — so each term lists its stories.
    tok = re.compile(r"[a-zA-Z][a-zA-Z]+")
    stopset = set(stop) | FREQ_STOP
    docs_by_term = {}
    for i, text in enumerate(df["headline"].fillna("").astype(str).tolist()):
        seen = set()
        for w in tok.findall(text.lower()):
            if w in stopset or len(w) < 3:
                continue
            c = canon(w)
            if c in stopset or len(c) < 3:
                continue
            seen.add(c)
        for c in seen:
            docs_by_term.setdefault(c, []).append(i)
    ranked = sorted(docs_by_term.items(), key=lambda kv: len(kv[1]), reverse=True)[:16]
    keywords = [{"term": term, "n": len(idxs), "arts": idxs} for term, idxs in ranked]
    print("[keywords] " + ", ".join(f"{k['term']}({k['n']})" for k in keywords[:10]))
    return df, themes, keywords


def build_payload(df, themes, keywords, shares):
    points = [{
        "h": r.headline, "s": SRC_LABEL[r.source], "d": str(r.scrape_date), "u": r.url,
    } for r in df.itertuples()]
    d0 = [s["d0"] for s in shares if s["d0"]]
    d1 = [s["d1"] for s in shares if s["d1"]]
    coverage = {"from": min(d0) if d0 else "", "to": max(d1) if d1 else ""}
    gen_short, gen_full = _now_ams()
    # named entities (people/orgs/places) via spaCy — degrades to {} if spaCy is absent.
    # Enriched with per-entity weekly trend, coverage tone, country split + co-occurrence.
    import ner
    body = df["body"].fillna("") if "body" in df.columns else ""
    ent_texts = (df["headline"].fillna("") + ". " + body).tolist()
    ent_dates = [str(x)[:10] for x in df["scrape_date"]]
    ent_countries = [SRC_CODE.get(s, "") for s in df["source"]]
    ent_tones = df["tone"].tolist() if "tone" in df.columns else None
    entities = ner.extract_entities(ent_texts, dates=ent_dates, sources=ent_countries,
                                    tones=ent_tones)
    return {
        "generated": gen_short,
        "generatedFull": gen_full,
        "sources": len(df["source"].unique()),
        "themes": themes, "points": points, "keywords": keywords,
        "colors": COLORS_LIGHT, "colorsDark": COLORS_DARK,
        "shares": shares, "map": BALTIC_MAP, "coverage": coverage,
        "homes": {SRC_CODE[s]: SRC_HOME[s] for s in SRC_HOME},
        "edits": find_edits(), "entities": entities,
    }
