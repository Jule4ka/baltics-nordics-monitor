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
Output: analysis/nordics-monitor.html  (drop into jule4ka.github.io/content/nordics-monitor/)
"""
import sys, os, re, json, warnings, datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = "data"        # the always-updated source CSVs live here
OUT_DIR = "analysis"     # generated report + data artifacts land here
FILES = {
    "err_ee": f"{DATA_DIR}/err_ee-always-updated.csv",
    "lrt_lt": f"{DATA_DIR}/lrt_lt-always-updated.csv",
    "lsm_lv": f"{DATA_DIR}/lsm_lv-always-updated.csv",
}
SRC_LABEL = {"err_ee": "ERR (Estonia)", "lrt_lt": "LRT (Lithuania)", "lsm_lv": "LSM (Latvia)"}
SRC_CODE = {"err_ee": "EE", "lrt_lt": "LT", "lsm_lv": "LV"}

# Real EE/LV/LT borders, projected to a shared SVG viewbox (see scratchpad/build_baltic_svg.py).
# Baked in so the pipeline stays offline; geometry is static, only the shading changes per run.
BALTIC_MAP = {
    "viewBox": "0 0 440 625",
    "paths": {
        "EE": "M382.6,231.6 L381.2,231.9 L373.1,230.6 L364.1,226.3 L360.2,223.1 L356.4,223.1 L351.7,225.2 L335.1,231.3 L331.0,229.9 L321.4,223.9 L316.6,217.4 L305.9,204.4 L305.0,201.3 L303.6,198.8 L292.2,195.6 L287.9,190.8 L284.4,190.1 L279.2,187.7 L265.8,177.5 L262.5,176.5 L261.7,178.2 L261.9,180.7 L261.1,182.1 L259.4,182.0 L256.2,178.2 L252.5,174.9 L240.9,181.2 L236.7,182.8 L233.1,183.2 L214.7,191.4 L209.1,195.8 L206.7,195.4 L207.3,191.2 L214.9,170.4 L216.3,153.9 L219.1,151.7 L219.9,149.4 L218.7,144.1 L210.8,140.7 L207.6,141.2 L204.7,146.9 L201.7,151.0 L194.7,153.5 L188.6,149.2 L174.5,143.4 L170.9,135.8 L170.1,128.1 L162.6,120.6 L159.5,111.9 L160.8,105.8 L167.5,101.7 L169.5,98.3 L160.9,98.8 L159.2,98.0 L158.8,94.8 L155.0,84.1 L158.4,79.9 L159.9,75.8 L157.1,72.3 L157.8,68.3 L160.0,64.3 L158.7,55.0 L167.1,50.0 L175.4,46.6 L192.9,44.8 L191.1,36.3 L198.2,35.9 L210.1,25.6 L221.9,27.4 L239.0,20.4 L271.8,20.5 L276.3,16.4 L275.5,12.3 L275.6,8.0 L281.8,9.2 L292.2,8.5 L330.9,17.0 L340.4,17.0 L353.6,25.7 L360.7,28.0 L381.7,28.0 L414.0,31.8 L420.4,25.9 L421.0,24.4 L424.1,27.7 L428.0,33.0 L429.0,36.0 L427.7,37.8 L423.8,39.3 L422.9,41.0 L421.2,43.7 L416.7,44.2 L414.3,46.3 L411.5,55.3 L406.2,70.2 L398.3,81.5 L392.0,87.7 L389.2,92.5 L387.4,98.2 L387.0,104.0 L393.1,135.5 L393.0,141.2 L391.6,147.1 L390.5,153.0 L391.4,158.2 L395.4,167.0 L399.6,180.2 L401.3,188.6 L404.1,191.7 L406.8,193.9 L407.4,195.3 L407.3,196.8 L405.9,198.5 L393.7,202.9 L392.1,206.6 L390.8,210.8 L385.4,216.9 L383.8,222.7 L382.8,229.3 L382.6,231.6ZM107.7,115.8 L111.9,118.4 L115.7,117.6 L119.5,115.8 L127.9,117.5 L147.0,130.4 L148.7,133.9 L137.3,135.5 L134.7,139.4 L132.0,142.2 L128.8,143.1 L123.3,148.7 L115.8,154.0 L114.3,157.2 L100.8,156.6 L93.5,158.6 L87.5,164.6 L85.1,176.2 L80.7,185.2 L76.3,188.4 L71.7,188.9 L70.6,185.5 L71.0,182.2 L80.8,169.4 L82.8,165.3 L77.9,163.5 L73.9,159.0 L65.0,153.9 L63.4,149.7 L65.6,149.4 L67.5,148.2 L69.9,144.7 L71.0,140.7 L63.9,129.0 L67.5,127.2 L72.0,127.6 L76.6,131.0 L81.7,127.0 L83.8,126.4 L87.4,127.8 L90.9,120.1 L99.4,117.6 L103.6,115.2 L107.7,115.8ZM125.5,94.0 L120.8,99.3 L117.9,97.2 L116.4,94.7 L110.3,106.5 L103.4,108.5 L99.3,106.2 L99.7,101.8 L95.7,90.2 L89.7,86.8 L81.3,86.5 L75.1,81.7 L98.7,78.4 L101.2,72.9 L106.0,67.1 L109.6,66.5 L112.7,67.8 L113.2,72.3 L114.0,74.1 L124.7,76.6 L128.9,84.2 L130.5,93.3 L125.5,94.0ZM149.9,123.3 L145.1,124.4 L133.6,116.9 L136.3,111.8 L139.6,109.8 L149.3,112.9 L150.6,120.7 L149.9,123.3Z",
        "LV": "M338.6,428.8 L335.7,428.2 L327.4,424.9 L320.5,420.0 L316.3,413.4 L309.1,404.5 L304.4,399.9 L297.0,394.1 L284.6,382.4 L280.1,379.7 L258.1,374.6 L250.1,372.3 L242.8,359.0 L240.4,351.3 L236.8,350.0 L232.9,351.6 L228.6,353.1 L218.7,362.1 L215.5,363.4 L209.4,363.6 L195.0,365.5 L188.5,362.3 L177.1,358.7 L171.0,358.1 L165.5,358.2 L141.3,354.6 L136.9,358.5 L132.4,359.2 L128.1,353.2 L122.7,351.5 L116.8,353.6 L106.0,353.8 L93.1,351.9 L76.8,350.4 L74.4,351.1 L56.2,359.0 L51.8,360.2 L32.1,373.6 L16.5,386.1 L14.7,366.1 L15.6,326.1 L18.0,306.3 L28.8,294.7 L34.2,285.7 L37.3,273.7 L38.3,262.6 L40.5,253.4 L56.1,227.1 L68.5,224.3 L85.3,216.9 L104.1,210.9 L107.7,218.6 L109.5,224.5 L132.1,246.1 L137.9,253.3 L146.6,278.1 L167.6,290.7 L184.0,286.7 L191.2,280.6 L204.3,269.3 L210.2,261.1 L211.4,253.2 L209.1,219.2 L205.5,204.5 L206.7,195.4 L209.1,195.8 L214.7,191.4 L233.1,183.2 L236.7,182.8 L240.9,181.2 L252.5,174.9 L256.2,178.2 L259.4,182.0 L261.1,182.1 L261.9,180.7 L261.7,178.2 L262.5,176.5 L265.8,177.5 L279.2,187.7 L284.4,190.1 L287.9,190.8 L292.2,195.6 L303.6,198.8 L305.0,201.3 L305.9,204.4 L316.6,217.4 L321.4,223.9 L331.0,229.9 L335.1,231.3 L351.7,225.2 L356.4,223.1 L360.2,223.1 L364.1,226.3 L373.1,230.6 L381.2,231.9 L382.6,231.6 L389.5,232.1 L391.9,233.8 L393.5,242.1 L401.3,248.6 L408.5,254.0 L410.3,256.5 L410.9,261.3 L410.4,267.0 L409.5,269.9 L406.5,273.3 L403.9,281.8 L403.5,289.9 L399.3,303.9 L400.3,304.2 L409.0,301.7 L411.5,303.1 L413.4,306.2 L414.0,315.0 L416.9,319.0 L419.8,325.2 L420.7,330.0 L426.3,335.7 L426.7,339.4 L430.1,352.5 L431.4,360.1 L432.0,365.9 L430.3,373.4 L428.9,378.4 L427.1,378.1 L422.1,379.4 L414.2,385.5 L402.5,399.7 L399.5,402.9 L396.4,413.8 L395.7,414.9 L388.9,414.4 L387.0,414.1 L380.2,414.3 L365.3,411.5 L359.5,413.4 L351.9,424.3 L349.0,426.0 L340.2,427.5 L338.6,428.8Z",
        "LT": "M11.4,469.9 L8.0,469.1 L14.6,456.9 L17.2,449.0 L18.9,437.7 L20.5,434.2 L20.5,439.3 L19.9,447.8 L15.7,462.3 L11.4,469.9ZM116.4,567.6 L113.9,562.5 L111.4,553.2 L111.6,545.8 L113.1,538.4 L120.1,516.6 L119.8,513.1 L114.6,507.0 L108.3,502.6 L104.8,493.2 L92.0,492.7 L79.9,493.2 L76.1,492.7 L64.6,488.8 L53.5,482.5 L46.0,478.8 L39.8,474.6 L36.4,470.3 L31.1,471.5 L27.5,471.5 L27.5,470.8 L25.5,463.1 L27.6,451.3 L23.8,434.0 L17.4,413.3 L16.9,391.1 L16.5,386.1 L32.1,373.6 L51.8,360.2 L56.2,359.0 L74.4,351.1 L76.8,350.4 L93.1,351.9 L106.0,353.8 L116.8,353.6 L122.7,351.5 L128.1,353.2 L132.4,359.2 L136.9,358.5 L141.3,354.6 L165.5,358.2 L171.0,358.1 L177.1,358.7 L188.5,362.3 L195.0,365.5 L209.4,363.6 L215.5,363.4 L218.7,362.1 L228.6,353.1 L232.9,351.6 L236.8,350.0 L240.4,351.3 L242.8,359.0 L250.1,372.3 L258.1,374.6 L280.1,379.7 L284.6,382.4 L297.0,394.1 L304.4,399.9 L309.1,404.5 L316.3,413.4 L320.5,420.0 L327.4,424.9 L335.7,428.2 L338.6,428.8 L338.4,433.5 L337.0,441.6 L334.3,452.0 L331.4,460.1 L330.7,463.2 L332.9,465.8 L343.7,467.0 L348.3,468.4 L349.2,470.5 L346.8,473.3 L343.3,475.7 L341.8,477.8 L339.0,485.7 L321.1,484.7 L318.7,486.3 L317.6,489.9 L316.7,494.1 L314.3,499.1 L309.5,503.5 L302.1,505.1 L296.0,508.0 L291.4,517.1 L288.0,529.4 L288.1,538.0 L288.6,542.9 L288.2,545.6 L285.9,548.7 L282.1,556.6 L279.0,565.5 L277.9,570.3 L278.4,572.5 L281.9,572.6 L286.9,574.4 L289.5,577.9 L290.5,582.0 L290.5,586.4 L289.6,588.8 L285.6,590.5 L279.3,590.6 L275.7,588.5 L275.0,586.9 L276.7,582.6 L275.4,577.4 L272.9,574.4 L267.6,578.8 L262.5,578.8 L256.5,582.7 L252.5,589.0 L248.8,591.3 L238.5,590.0 L236.0,592.8 L233.8,605.6 L232.6,608.1 L224.1,607.6 L215.8,612.6 L206.5,616.8 L201.7,613.9 L199.1,610.7 L194.0,611.3 L188.5,612.7 L184.8,611.9 L180.6,612.2 L172.5,614.7 L162.4,613.9 L158.1,611.8 L157.7,609.8 L158.0,604.8 L157.9,597.1 L156.3,590.2 L151.4,584.2 L146.3,580.0 L139.8,575.6 L135.0,573.7 L132.4,573.2 L131.8,570.7 L130.8,568.5 L128.6,566.6 L123.8,564.1 L119.7,563.5 L116.4,567.6Z",
    },
    "centroids": {"EE": [296.2, 127.2], "LV": [267.2, 303.0], "LT": [203.3, 494.2]},
}

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

# Extra stopwords for the act-3 frequency chart only: generic filler + the countries'
# own names/cities (self-references that would otherwise dominate and tell us nothing).
FREQ_STOP = {
    "time", "state", "states", "way", "week", "day", "days", "people", "month",
    "today", "year", "years", "country", "countries", "new", "also", "first", "two",
    "lithuania", "latvia", "estonia", "lithuanian", "latvian", "estonian",
    "baltic", "baltics", "vilnius", "tallinn", "riga",
    "added", "situation", "possible", "according", "public",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "does", "including", "used"
}

# Merge overlapping word forms for the frequency chart so variants count as one term:
# adjective/demonym forms and plurals collapse to a single canonical word
# (russia/russian -> russia, drones -> drone, ministers/ministries -> minister/ministry).
CANON = {
    "russian": "russia", "russians": "russia",
    "ukrainian": "ukraine", "ukrainians": "ukraine",
    "belarus": "belarus", "belarusian": "belarus", "belarusians": "belarus",
    "chinese": "china", "estonian": "estonia", "latvian": "latvia",
    "lithuanian": "lithuania", "defense": "defence", "militarily": "military",
    "ministries": "ministry", "ministers": "minister",
}

def canon(w):
    """Collapse a token to its canonical form (variant map first, then simple plural)."""
    if w in CANON:
        return CANON[w]
    if w.endswith("ies") and len(w) > 5:                 # embassies -> embassy
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        return w[:-1]                                    # drones -> drone, troops -> troop
    return w

# Categorical palette for the clusters (legible on both light and dark grounds).
# Thematic keyword buckets for the topic map (act 2). Each matched article is
# assigned to the bucket its keywords hit hardest (headline weighted x3); the
# fallback is "Other defence". Colour + an on-map label carry the theme, while
# proximity (t-SNE) carries semantic similarity — no opaque clustering.
THEMES = [
    ("Ukraine",            r"ukrain|zelensk|kyiv|kharkiv|donbas|donetsk|kherson|zaporizh|odesa"),
    ("Russia & Kremlin",   r"russ|kremlin|putin|moscow|wagner|lavrov|medvedev"),
    ("NATO & defence",     r"nato|defen[cs]e|militar|troop|soldier|brigade|deploy|exercise|weapon|"
                           r"missile|artillery|himars|atacms|warship|frigate|submarine|\barmy\b|"
                           r"rearm|conscript|mobilis|mobiliz"),
    ("Drones & airspace",  r"drone|airspace|uav|incursion|airport|warplane|aircraft|\bjet\b"),
    ("Border & migration", r"border|migrant|refugee|belarus|frontier|fence|smuggl|crossing|barbed"),
    ("Hybrid & sabotage",  r"sabotag|cyber|espionage|\bspy|hybrid|disinfo|jamming|\bgps\b|"
                           r"undersea|cable|provocation|shadow fleet"),
    ("China",              r"china|chinese|beijing"),
]
THEME_OTHER = "Other defence"
THEME_PATS = [re.compile(p, re.I) for _, p in THEMES]

# Categorical palette (validated reference hues), light + dark step per theme id.
# Identity is carried mainly by the on-map labels; colour reinforces. Last = grey "Other".
THEME_COLORS = [
    ("#2a78d6", "#3987e5"),  # 0 Ukraine            blue
    ("#eb6834", "#d95926"),  # 1 Russia & Kremlin   orange
    ("#1baf7a", "#199e70"),  # 2 NATO & defence     aqua
    ("#eda100", "#c98500"),  # 3 Drones & airspace  yellow
    ("#e87ba4", "#d55181"),  # 4 Border & migration magenta
    ("#1f9e46", "#37b25c"),  # 5 Hybrid & sabotage  green
    ("#7a68c4", "#9085e9"),  # 6 China              violet
    ("#9a8f7c", "#8f836d"),  # 7 Other defence      warm grey
]
COLORS_LIGHT = [c[0] for c in THEME_COLORS]
COLORS_DARK = [c[1] for c in THEME_COLORS]


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

    # 2D layout via t-SNE on LSA space (proximity ≈ shared language) — no clustering
    perp = max(5, min(30, n // 3))
    xy = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=42).fit_transform(Z)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]

    # Assign each article to the theme its keywords hit hardest (headline weighted x3).
    def theme_of(h, b):
        best_i, best_s = len(THEMES), 0            # default index = "Other defence"
        for i, pat in enumerate(THEME_PATS):
            s = 3 * len(pat.findall(h)) + len(pat.findall(b))
            if s > best_s:
                best_s, best_i = s, i
        return best_i
    df["theme"] = [theme_of(h, b) for h, b in
                   zip(df["headline"].fillna(""), df["body"].fillna(""))]

    # Per-theme metadata: count, top mean-TF-IDF terms, and the on-map label centroid.
    Xa = X.toarray()
    names = [t[0] for t in THEMES] + [THEME_OTHER]
    themes = []
    for tid in range(len(THEMES) + 1):
        idx = np.where(df["theme"].values == tid)[0]
        if len(idx) == 0:
            continue
        mean_w = Xa[idx].mean(axis=0)
        kw = ", ".join(terms[mean_w.argsort()[::-1][:5]])
        themes.append({
            "id": tid, "label": names[tid], "kw": kw, "count": int(len(idx)),
            "cx": round(float(df["x"].values[idx].mean()), 3),
            "cy": round(float(df["y"].values[idx].mean()), 3),
        })
        print(f"[theme] {names[tid]} ({len(idx)}): {kw}")
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
        "x": round(float(r.x), 3), "y": round(float(r.y), 3), "c": int(r.theme),
        "h": r.headline, "s": SRC_LABEL[r.source], "d": str(r.scrape_date), "u": r.url,
    } for r in df.itertuples()]
    d0 = [s["d0"] for s in shares if s["d0"]]
    d1 = [s["d1"] for s in shares if s["d1"]]
    coverage = {"from": min(d0) if d0 else "", "to": max(d1) if d1 else ""}
    return {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sources": len(df["source"].unique()),
        "themes": themes, "points": points, "keywords": keywords,
        "colors": COLORS_LIGHT, "colorsDark": COLORS_DARK,
        "shares": shares, "map": BALTIC_MAP, "coverage": coverage,
    }


CSS = """
:root{
  --bg:#e7d5b3; --panel:#f6ecd2; --ink:#2b140f; --muted:#6d4634; --faint:#9c7a5a;
  --border:#c1972f; --border2:#e2cd9c; --accent:#a52f1d; --gold:#bd8322; --green:#3c7a4e;
  --grid:#d7bf8f; --shadow:0 2px 4px rgba(70,35,15,.12),0 16px 36px rgba(70,35,15,.18);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#180c0a; --panel:#26130d; --ink:#f4e7cd; --muted:#c6a986; --faint:#8c7455;
    --border:#84591f; --border2:#3a2315; --accent:#e8614a; --gold:#e6ad4c; --green:#5fb17b;
    --grid:#35221a; --shadow:0 2px 6px rgba(0,0,0,.5),0 18px 44px rgba(0,0,0,.62);
  }
}
:root[data-theme=light]{
  --bg:#e7d5b3; --panel:#f6ecd2; --ink:#2b140f; --muted:#6d4634; --faint:#9c7a5a;
  --border:#c1972f; --border2:#e2cd9c; --accent:#a52f1d; --gold:#bd8322; --green:#3c7a4e;
  --grid:#d7bf8f; --shadow:0 2px 4px rgba(70,35,15,.12),0 16px 36px rgba(70,35,15,.18);
}
:root[data-theme=dark]{
  --bg:#180c0a; --panel:#26130d; --ink:#f4e7cd; --muted:#c6a986; --faint:#8c7455;
  --border:#84591f; --border2:#3a2315; --accent:#e8614a; --gold:#e6ad4c; --green:#5fb17b;
  --grid:#35221a; --shadow:0 2px 6px rgba(0,0,0,.5),0 18px 44px rgba(0,0,0,.62);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased;}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Code","Cascadia Mono",Consolas,monospace;}
/* faint woven-textile watermark (Baltic diamond net) in the paper margins, behind all content */
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.09;
  background:var(--gold);
  -webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='30'%20height='30'%3E%3Cpath%20d='M15%200L30%2015L15%2030L0%2015Z'%20fill='none'%20stroke='%23000'%20stroke-width='1.1'/%3E%3Ccircle%20cx='15'%20cy='15'%20r='1.2'%20fill='%23000'/%3E%3C/svg%3E") repeat;
          mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='30'%20height='30'%3E%3Cpath%20d='M15%200L30%2015L15%2030L0%2015Z'%20fill='none'%20stroke='%23000'%20stroke-width='1.1'/%3E%3Ccircle%20cx='15'%20cy='15'%20r='1.2'%20fill='%23000'/%3E%3C/svg%3E") repeat;}
.shell{max-width:1180px;margin:0 auto;padding:0 22px;position:relative;z-index:1;}
header{padding:46px 0 30px;border-bottom:3px double var(--gold);}
.eyebrow{display:inline-flex;align-items:center;gap:9px;font-size:11px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--panel);background:var(--accent);font-weight:700;
  padding:7px 14px;border-radius:3px;margin:0 0 20px;box-shadow:0 3px 0 var(--gold);}
.eyebrow::before{content:"\\2734";font-size:13px;line-height:1;color:var(--gold);display:inline-block;}
h1{font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  font-size:clamp(35px,5.4vw,60px);line-height:1.01;margin:0 0 15px;font-weight:700;
  letter-spacing:-.02em;text-wrap:balance;max-width:15ch;}
.lede{color:var(--muted);font-size:16px;max-width:64ch;margin:0;}
.stats{display:flex;flex-wrap:wrap;gap:0;margin-top:28px;
  border:2px solid var(--gold);border-radius:6px;overflow:hidden;background:var(--panel);}
.stat{flex:1 1 140px;padding:16px 18px;border-right:1px solid var(--border);}
.stat:last-child{border-right:0}
.stat .n{font-size:28px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-.01em;color:var(--accent);}
.stat .l{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--faint);margin-top:3px;}
main{padding:30px 0 10px;}
.panel{background:var(--panel);border:2px solid var(--gold);border-radius:7px;
  box-shadow:var(--shadow);overflow:hidden;}
.panel-h{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  padding:15px 20px;border-bottom:2px solid var(--gold);flex-wrap:wrap;
  background:color-mix(in srgb,var(--gold) 15%,var(--panel));}
.panel-h h2{margin:0;font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  font-size:18px;font-weight:700;letter-spacing:-.005em;color:var(--ink);}
.panel-h .hint{font-size:12px;color:var(--muted);}
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
.section-h{margin:44px 0 6px;font-size:14px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);font-weight:700;display:flex;align-items:center;gap:13px;}
.sec-sign{width:30px;height:30px;flex:0 0 auto;color:var(--accent);padding:4px;
  background:color-mix(in srgb,var(--gold) 22%,var(--panel));
  border:1.5px solid var(--gold);border-radius:50%;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;margin:16px 0 10px;}
.card{background:var(--panel);border:1.5px solid var(--border);border-radius:7px;padding:15px 18px;
  border-top:5px solid var(--dot);box-shadow:var(--shadow);}
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
/* quantitative: Baltic defence-share choropleth */
.country-wrap{display:flex;gap:28px;padding:20px 22px 24px;flex-wrap:wrap;align-items:center;}
.country-map{flex:0 0 auto;width:min(300px,72vw);}
.country-map svg{width:100%;height:auto;display:block;overflow:visible;}
.geo{stroke:var(--panel);stroke-width:1.4;fill:var(--accent);transition:fill-opacity .25s,filter .12s;}
.geo:hover{filter:brightness(1.07);}
.geo-lab{fill:var(--ink);paint-order:stroke;stroke:var(--panel);stroke-width:3.6px;stroke-linejoin:round;
  font-weight:800;font-size:16px;text-anchor:middle;pointer-events:none;}
.geo-sub{fill:var(--ink);paint-order:stroke;stroke:var(--panel);stroke-width:3px;stroke-linejoin:round;
  font-weight:700;font-size:13px;text-anchor:middle;pointer-events:none;font-variant-numeric:tabular-nums;}
.country-legend{flex:1 1 210px;min-width:200px;}
.cl-row{display:flex;align-items:center;gap:11px;padding:10px 0;border-top:1px solid var(--border2);}
.cl-row:first-child{border-top:0;}
.cl-sw{width:13px;height:13px;border-radius:3px;background:var(--accent);flex:0 0 auto;}
.cl-name{font-size:13.5px;font-weight:600;color:var(--ink);}
.cl-pct{margin-left:auto;font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--ink);}
.cl-frac{font-size:11.5px;color:var(--faint);margin-left:9px;font-variant-numeric:tabular-nums;}
.scale{display:flex;align-items:center;gap:9px;margin-top:15px;padding-top:13px;
  border-top:1px solid var(--border2);font-size:11px;color:var(--faint);}
.scale-bar{height:8px;flex:1;border-radius:4px;
  background:linear-gradient(90deg,color-mix(in srgb,var(--accent) 20%,var(--panel)),var(--accent));}
.section-num{color:var(--accent);font-weight:700;margin-right:2px;}
/* Baltic woven ornament band (diamond + cross "josta" motif), tinted via CSS mask */
.weave{height:16px;background:var(--accent);opacity:.9;
  -webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='32'%20height='16'%3E%3Cpath%20d='M16%201L32%208L16%2015L0%208Z'%20fill='none'%20stroke='%23000'%20stroke-width='1.6'/%3E%3Cpath%20d='M16%205V11M13%208H19'%20stroke='%23000'%20stroke-width='1.5'/%3E%3C/svg%3E") repeat-x center/auto 16px;
          mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='32'%20height='16'%3E%3Cpath%20d='M16%201L32%208L16%2015L0%208Z'%20fill='none'%20stroke='%23000'%20stroke-width='1.6'/%3E%3Cpath%20d='M16%205V11M13%208H19'%20stroke='%23000'%20stroke-width='1.5'/%3E%3C/svg%3E") repeat-x center/auto 16px;}
.header-weave{margin:22px 0 4px;padding:7px 0;border-top:2px solid var(--gold);border-bottom:2px solid var(--gold);}
.footer-weave{margin:10px 0 24px;padding:7px 0;border-top:2px solid var(--gold);border-bottom:2px solid var(--gold);}
/* legend: stacked name + date span, right-aligned numbers */
.cl-txt{display:flex;flex-direction:column;gap:1px;min-width:0;}
.cl-dates{font-size:11px;color:var(--faint);font-variant-numeric:tabular-nums;letter-spacing:.01em;}
.cl-num{margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:0;}
.cl-num .cl-pct{margin-left:0;}
.cl-num .cl-frac{margin-left:0;}
/* coverage caption under the country panel */
.cov{margin:0 22px;padding:13px 0 18px;border-top:1px dashed var(--border);
  font-size:12.5px;color:var(--muted);display:flex;align-items:center;gap:9px;flex-wrap:wrap;}
.cov b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;}
.cov .star{color:var(--accent);}
/* small honesty caveat note */
.note{max-width:64ch;margin:2px 0 0;font-size:12.5px;color:var(--faint);line-height:1.5;}
.note b{color:var(--muted);font-weight:600;}
/* collapsible theme cards — article list revealed on select */
.card-h{cursor:pointer;user-select:none;}
.card .caret{margin-left:10px;color:var(--faint);transition:transform .15s;font-size:11px;}
.card.open .caret{transform:rotate(90deg);}
.card-arts{display:none;}
.card.open .card-arts{display:block;}
.card-hint{font-size:11px;color:var(--faint);margin:2px 0 0;}
.card.open .card-hint{display:none;}
/* act 3 — keyword frequency bars (click a term to list its articles) */
.freq{padding:16px 22px 20px;display:flex;flex-direction:column;gap:4px;}
.freq-row{display:grid;grid-template-columns:120px 1fr 42px;align-items:center;gap:12px;
  padding:5px 0;cursor:pointer;border-radius:6px;}
.freq-row:hover .freq-term{color:var(--accent);}
.freq-row:hover .freq-fill{filter:brightness(1.06);}
.freq-row:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.freq-term{font-size:13px;color:var(--ink);text-align:right;font-weight:500;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.freq-track{height:14px;background:var(--border2);border-radius:4px;overflow:hidden;}
.freq-fill{height:100%;background:var(--accent);border-radius:4px;min-width:3px;
  transition:width .5s cubic-bezier(.2,.7,.2,1);}
.freq-n{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums;text-align:left;}
.freq-arts{display:none;list-style:none;margin:2px 0 8px;padding:2px 12px 2px 132px;
  max-height:360px;overflow-y:auto;}
.freq-item.open .freq-arts{display:block;}
.freq-arts li{padding:7px 0;border-top:1px solid var(--border2);font-size:13px;}
.freq-arts li:first-child{border-top:0;}
.freq-arts a{color:var(--ink);text-decoration:none;display:block;}
.freq-arts a:hover{color:var(--accent);}
.freq-arts .src{display:block;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);margin-top:2px;}
@media (max-width:560px){.freq-arts{padding-left:0;}}
@media (max-width:560px){#map{height:440px;}#legend{position:static;max-width:none;margin:10px;}
  .country-wrap{gap:16px;}.country-map{width:min(240px,66vw);}}
"""

BODY = """
<div class="shell">
  <header>
    <p class="eyebrow mono">Baltic Defence &amp; Geopolitics Monitor</p>
    <h1>The week in Russia, Ukraine &amp; regional security news</h1>
    <p class="lede">How much of the Baltic public broadcasters' news is about defence and security —
      and what is it actually about? First the numbers by country, then the themes: every matching
      article embedded, coloured by topic and mapped by shared language.</p>
    <div class="stats mono" id="stats"></div>
  </header>
  <div class="weave header-weave" aria-hidden="true"></div>
  <main>
    <p class="section-h"><svg class="sec-sign" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" aria-hidden="true"><path d="M10 1.5 L18.5 10 L10 18.5 L1.5 10 Z"/><rect x="4" y="4" width="12" height="12"/><circle cx="10" cy="10" r="1.4" fill="currentColor" stroke="none"/></svg> The numbers — how much is defence?</p>
    <section class="panel">
      <div class="panel-h">
        <h2>Defence &amp; security share by country</h2>
        <span class="hint mono">matching headlines ÷ all headlines</span>
      </div>
      <div class="country-wrap">
        <div class="country-map"><svg id="cmap" role="img"
          aria-label="Map of Estonia, Latvia and Lithuania shaded by share of defence news"></svg></div>
        <div class="country-legend" id="clegend"></div>
      </div>
      <div class="cov" id="cov"></div>
    </section>
    <p class="note"><b>Read with care:</b> each broadcaster's English page carries a different mix
      (LSM's homepage includes weather, culture &amp; local news; LRT's is a curated
      <em>news-in-english</em> feed) and was scraped a different number of times. These shares
      reflect what landed on each page over the period — not the newsroom's overall editorial
      priorities, and they are not strictly comparable between countries.</p>

    <p class="section-h"><svg class="sec-sign" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true"><circle cx="10" cy="10" r="3.8"/><path d="M10 1.5V4M10 16V18.5M1.5 10H4M16 10H18.5M4 4l1.9 1.9M14.1 14.1L16 16M16 4l-1.9 1.9M5.9 14.1L4 16"/></svg> The themes — what is it about?</p>
    <section class="panel">
      <div class="panel-h">
        <h2>Topic map</h2>
        <span class="hint mono">near = similar wording · colour &amp; label = theme</span>
      </div>
      <div class="mapwrap">
        <canvas id="map"></canvas>
        <div id="legend" class="mono"></div>
        <div id="tip"></div>
      </div>
    </section>
    <p class="section-h">By theme — select one to read its articles</p>
    <section class="grid" id="cards"></section>

    <p class="section-h"><svg class="sec-sign" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 8L10 3L17 8M3 12.5L10 7.5L17 12.5M3 17L10 12L17 17"/></svg> The words — what comes up most</p>
    <section class="panel">
      <div class="panel-h">
        <h2>Most frequent terms</h2>
        <span class="hint mono">articles mentioning each word · click a word to list them</span>
      </div>
      <div class="freq" id="freq"></div>
    </section>
  </main>
  <div class="weave footer-weave" aria-hidden="true"></div>
  <footer class="mono">
    Sources — ERR (Estonia), LRT (Lithuania), LSM (Latvia), English editions.<br>
    Method — headlines filtered to defence/geopolitics themes, full article bodies fetched,
    TF-IDF vectorised, reduced with TruncatedSVD (LSA) and laid out with t-SNE so nearby dots
    share wording. Each article is coloured by the theme its keywords match most; theme labels
    sit on the map. Keyword lists are each theme's top mean-TF-IDF terms.
  </footer>
</div>
"""

SCRIPT = """
const D = __DATA__;

// ---- act 1: Baltic defence-share choropleth ----
(function(){
  const M = D.map, sh = D.shares;
  const maxp = Math.max(...sh.map(s=>s.pct), 1);
  const op = p => (0.2 + 0.8 * (p / maxp));      // sequential: fill-opacity of accent by share
  const by = {}; sh.forEach(s => by[s.code] = s);
  const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const fmt = iso => { if(!iso) return '—'; const [y,m,d] = iso.split('-'); return `${+d} ${MON[+m-1]} ${y}`; };
  const svg = document.getElementById('cmap');
  svg.setAttribute('viewBox', M.viewBox);
  let g = '';
  for (const code in M.paths){
    const s = by[code] || {country:code, pct:0, matched:0, total:0};
    g += `<path class="geo" d="${M.paths[code]}" fill-opacity="${op(s.pct).toFixed(3)}">`
       + `<title>${s.country} — ${s.pct}% defence (${s.matched}/${s.total})</title></path>`;
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
  ).join('') + `<div class="scale mono"><span>0%</span><span class="scale-bar"></span><span>${maxp}%</span></div>`;
  const cov = D.coverage || {};
  const totAll = sh.reduce((a,s)=>a+s.total, 0);
  document.getElementById('cov').innerHTML =
    `<span class="star">✴</span> Articles collected <b>${fmt(cov.from)}</b> → <b>${fmt(cov.to)}</b>`
    + ` · <b>${D.points.length}</b> defence-themed of <b>${totAll}</b> total headlines`;
})();

const cvs = document.getElementById('map'), ctx = cvs.getContext('2d');
const tip = document.getElementById('tip'), wrap = cvs.parentElement;
const hidden = new Set();
let pts = [], labelPos = [], hover = -1, anim = 1;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const isDark = () => { const t = document.documentElement.getAttribute('data-theme');
  return t === 'dark' || (t !== 'light' && matchMedia('(prefers-color-scheme:dark)').matches); };
let C = isDark() ? D.colorsDark : D.colors;      // theme-aware categorical palette

// header stats
document.getElementById('stats').innerHTML = [
  [D.points.length,'Articles'], [D.sources,'Sources'],
  [D.themes.length,'Themes'], [D.generated,'Updated']
].map(s=>`<div class="stat"><div class="n">${s[0]}</div><div class="l">${s[1]}</div></div>`).join('');

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
    const arts = D.points.filter(p=>p.c===t.id);
    const col = C[t.id];
    const li = arts.map(p=>`<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`).join('');
    return `<div class="card" style="--dot:${col}"><h3 class="card-h">`
      + `<span class="sw" style="width:9px;height:9px;border-radius:50%;background:${col};display:inline-block"></span>`
      + `${t.label}<span class="lc">${t.count}</span><span class="caret">▶</span></h3>`
      + `<p class="kw">${t.kw}</p>`
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
    const li = k.arts.map(i=>{ const p = D.points[i];
      return `<li><a href="${p.u}" target="_blank" rel="noopener">${p.h}<span class="src">${p.s} · ${p.d}</span></a></li>`;
    }).join('');
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
renderLegend(); renderCards(); renderFreq();

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
  // theme labels on the map — the primary identity channel; colour only reinforces
  ctx.textAlign='center';ctx.textBaseline='middle';ctx.lineJoin='round';
  ctx.font='600 12.5px ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif';
  const ink=css('--ink');
  labelPos.forEach(L=>{
    if(hidden.has(L.id))return;
    ctx.lineWidth=4;ctx.strokeStyle=ground;ctx.globalAlpha=.9;ctx.strokeText(L.label,L.sx,L.sy);
    ctx.globalAlpha=1;ctx.fillStyle=ink;ctx.fillText(L.label,L.sx,L.sy);
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
  (function run(t){anim=Math.max(0,Math.min(1,(t-t0)/500));draw();if(anim<1)requestAnimationFrame(run);})(t0);}
// on theme toggle: swap palette, rebuild legend + cards swatches, redraw canvas
function onTheme(){ C = isDark() ? D.colorsDark : D.colors; renderLegend(); renderCards(); draw(); }
new MutationObserver(onTheme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',onTheme);
"""


def render(df, themes, keywords, shares):
    payload = build_payload(df, themes, keywords, shares)
    data_js = json.dumps(payload, ensure_ascii=False)
    script = SCRIPT.replace("__DATA__", data_js)
    inner = f"<style>{CSS}</style>\n{BODY}\n<script>{script}</script>"

    standalone = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Baltic Defence &amp; Geopolitics Monitor</title></head>"
        f"<body>{inner}</body></html>"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/nordics-monitor.html", "w", encoding="utf-8") as f:
        f.write(standalone)
    with open(f"{OUT_DIR}/nordics-monitor-artifact.html", "w", encoding="utf-8") as f:
        f.write(inner)  # content-only, for the Claude Artifact wrapper
    with open(f"{OUT_DIR}/nordics-monitor-data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[render] wrote {OUT_DIR}/nordics-monitor.html ({len(standalone)//1024} KB, self-contained)")


def main():
    df_all = load_data()
    df = filter_themes(df_all)
    shares = compute_shares(df_all, df)
    df["body"] = fetch_all(df["url"].tolist())
    df, themes, keywords = analyse(df)
    render(df, themes, keywords, shares)
    df[["url", "headline", "source", "scrape_date", "theme"]].to_csv(
        f"{OUT_DIR}/nordics-monitor-clustered.csv", index=False, encoding="utf-8")
    print(f"[done] {OUT_DIR}/ nordics-monitor.html + artifact + data json + clustered csv")


if __name__ == "__main__":
    main()
