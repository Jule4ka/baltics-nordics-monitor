# Baltics Monitor

An automated **defence & geopolitics news monitor** for the Baltic states.

Once a day it scrapes the English editions of the three Baltic public broadcasters,
keeps only the security-related stories (Russia, Ukraine, NATO, China, drones, war,
sabotage…), analyses them, and builds **one self-contained interactive web page**.
It then publishes that page to a personal website. There's no server — everything
runs on **GitHub Actions**.

**Live page:** <https://jule4ka.github.io/content/baltics_monitor/>

> ℹ️ **Scope:** today this covers the **Baltics** (hence `baltics-monitor.*`).
> The repo is named `baltics-nordics-monitor` because the plan is to grow it —
> Nordics, Central Europe, eventually the whole of cozy little Europe.

---

## Table of contents

1. [The two repos (important)](#the-two-repos-important)
2. [Sources](#sources)
3. [Data flow at a glance](#data-flow-at-a-glance)
4. [Project structure](#project-structure)
5. [How it works, step by step](#how-it-works-step-by-step)
6. [The report — what's on the page](#the-report--whats-on-the-page)
7. [Running it locally](#running-it-locally)
8. [The daily automation (GitHub Actions)](#the-daily-automation-github-actions)
9. [Publishing explained (the complicated part)](#publishing-explained-the-complicated-part)
10. [One-time setup: the token](#one-time-setup-the-token)
11. [Caveats & gotchas](#caveats--gotchas)
12. [Troubleshooting](#troubleshooting)

---

## The two repos (important)

This project spans **two separate GitHub repositories**. Keeping them straight is the
key to understanding everything else:

| Repo | Role |
|---|---|
| **`Jule4ka/baltics-nordics-monitor`** (this one) | **The engine.** Scrapers, the analysis script, and the daily GitHub Action that produces the page. |
| **`Jule4ka/Jule4ka.github.io`** | **The website.** A [Nikola](https://getnikola.com/) static site (Julia's personal page). The finished monitor is published *into* this repo as one of its pages. |

Every day, the engine repo builds the page and **pushes it into the website repo**,
which then serves it. See [Publishing explained](#publishing-explained-the-complicated-part).

## Sources

| Broadcaster | Country | Scraper notebook | Master data file |
|---|---|---|---|
| ERR | 🇪🇪 Estonia | `scrapers/err_ee.ipynb` | `data/err_ee-always-updated.csv` |
| LRT | 🇱🇹 Lithuania | `scrapers/lrt_lt.ipynb` | `data/lrt_lt-always-updated.csv` |
| LSM | 🇱🇻 Latvia | `scrapers/lsm_lv.ipynb` | `data/lsm_lv-always-updated.csv` |

Only the **English editions** are scraped.

## Data flow at a glance

```
  scrapers/
  ┌─ err_ee.ipynb ─┐
  ├─ lrt_lt.ipynb ─┤   scrape homepages      ┌───────────────────┐
  └─ lsm_lv.ipynb ─┘  ───────────────────▶   │  data/*.csv        │  (archive of headlines)
        (LSM also uses scrapers/scrape.py +  └─────────┬─────────┘
         Playwright to beat Cloudflare)                │
                                                        ▼
                                            ┌───────────────────────┐
                                            │  topic_analysis.py     │  filter → fetch bodies →
                                            │                        │  keyword themes + word freq →
                                            │                        │  spaCy NER → embeddings
                                            │                        │  (BERTopic/HDBSCAN) + search →
                                            │                        │  build
                                            └───────────┬───────────┘
                                                        ▼
                                            ┌───────────────────────┐
                                            │ analysis/              │
                                            │  baltics-monitor.html  │  ← the finished report
                                            └───────────┬───────────┘
                                                        ▼
                        copy into jul_site/files/  +  nikola build + github_deploy
                                                        ▼
                                   https://jule4ka.github.io/content/baltics_monitor/
```

## Project structure

```
baltics-nordics-monitor/
├── data/                              # INPUTS (scraped headlines)
│   ├── <src>-always-updated.csv       #   master archive per source (deduplicated by URL)
│   └── err_ee/  lrt_lt/  lsm_lv/       #   a timestamped snapshot of every scrape run
│
├── analysis/                          # OUTPUTS (regenerated every run)
│   ├── baltics-monitor.html           #   the report — full self-contained page (THIS is published)
│   ├── baltics-monitor-artifact.html  #   content-only variant (no <html>/<head>; for embedding)
│   ├── baltics-monitor-data.json      #   the computed data behind the page
│   ├── baltics-monitor-clustered.csv  #   every matched article tagged with its theme(s)
│   ├── body_cache.json                #   COMMITTED — scraped article bodies keyed by URL (download once)
│   └── emb_cache/                     #   COMMITTED — cached embeddings, keyed by URL (incremental)
│
│   # ── the pipeline, split into focused modules (edit whichever part you mean) ──
├── topic_analysis.py                  # thin CLI entrypoint — just wires the steps together
├── config.py                          # sources, the KEYWORDS filter, THEME buckets, palette
├── pipeline.py                        # load → filter → fetch → analyse (keyword buckets + word freq)
├── ner.py                             # spaCy named-entity extraction (people / orgs / places)
├── render.py                          # assembles the self-contained page (embeddings folded in)
├── embeddings_analysis.py             # semantic layer: fastembed → UMAP → BERTopic/HDBSCAN,
│                                      #   + the in-browser transformers.js search index
├── assets.py                          # loads the three front-end files below
├── assets/                            # THE FRONT-END — edit these directly
│   ├── report.css                     #   styles
│   ├── report.html                    #   page body markup
│   └── report.js                      #   the interactive Canvas map + charts
│
├── scrapers/                          # THE SCRAPERS — the three notebooks + LSM helper
│   ├── err_ee.ipynb  lrt_lt.ipynb  lsm_lv.ipynb   # one notebook per broadcaster
│   ├── scrape.py                      #   Playwright helper: fetches the LSM homepage past Cloudflare
│   └── lsm_page.html                  #   scrape.py's rendered LSM homepage (intermediate)
│                                      #   (each notebook chdir's to the repo root, so data/ paths are unchanged)
└── .github/workflows/                 # the once-a-day automation
```

The embeddings + search sections are now part of the main page (after the keyword-bucket,
timeline and entity sections), not a separate file. Build a fast, embeddings-free page
locally (keyword-bucket / timeline / entity sections only) with
`python topic_analysis.py --no-embed`.

## How it works, step by step

### 1. Scrape (the three notebooks)

Each notebook fetches its broadcaster's homepage, extracts each **headline + URL**, and
**appends to two files**:
- a timestamped snapshot in `data/<src>/` (a permanent record of that run), and
- the master `data/<src>-always-updated.csv` — deduplicated by URL, so only genuinely
  **new** articles are added over time.

Two of the three are easy; one is not:
- **ERR** and **LRT** are fetched with a plain `requests.get()`.
- **LSM** sits behind **Cloudflare**, which blocks plain requests. So `lsm_lv.ipynb`
  first runs **`scrapers/scrape.py`**, which drives a headless **Playwright + stealth** browser
  to clear the Cloudflare challenge and save the rendered HTML to `scrapers/lsm_page.html` for
  the notebook to parse. (This is why LSM needs a real browser — see *Caveats*.)

### 2. Analyse & build the report (`topic_analysis.py`)

1. **Load** the three master CSVs, dedupe by URL. A load-time `ftfy` pass repairs any
   mojibake that slipped into older rows (see the note below on `lrt_lt`).
2. **Filter** headlines to the defence/geopolitics keyword set (Ukraine, Russia, NATO,
   drones, border, brigade, ammunition, smuggling, China, sabotage, …) — a single
   case-insensitive regex (`config.KEYWORDS`) matched against the **headline only**.
3. **Per-country share** — how many of each source's headlines matched (`matched ÷ total`),
   plus each source's date range (the country map + the "defence share" numbers).
4. **Fetch the full article body** for each matched article (8 in parallel; failures
   skipped). Bodies are **cached by URL** in `analysis/body_cache.json`, so the first run
   downloads everything and later runs fetch only the *newly matched* articles — the slow
   download happens once. Bodies feed the theme tagging, the entity text and the embeddings.
5. **Assign keyword-bucket themes** — hand-authored regex buckets, **not** clustering.
   Each article is tagged with **every** bucket its words hit (headline weighted ×3 over
   body) and coloured by the strongest, so it can appear under more than one theme
   (*multi-label*). Buckets: Ukraine, Russia & Kremlin, NATO & defence, Drones & airspace,
   Border & migration, Hybrid warfare, China & Indo-Pacific, or "Other defence". "NATO &
   defence" is the broad catch-all, so it loses scoring ties to a more specific theme.
   *(There is no TF-IDF / SVD / t-SNE / KMeans here any more — the old vector layout and
   clustering were removed; the semantic map now lives entirely in the embeddings step.)*
6. **Keyword frequency** — count, across the **headlines**, in how many articles each word
   appears (document frequency, not total hits), after dropping stopwords + the Baltic
   countries' own names/capitals and merging variants (russia/russian → russia, drones →
   drone, …). Top 16 terms, each keeping the list of articles that use it.
7. **Named entities** (`ner.py`) — spaCy (`en_core_web_sm`) pulls people / organizations /
   places from headline + body for the entity network (degrades to none if spaCy is absent).
8. **Embeddings layer** (`embeddings_analysis.py`, below) — discovered topics, escalation
   tone, and the in-browser search index.
9. **Render** (`render.py`) the report to `analysis/baltics-monitor.html` (+ the JSON, the
   tagged CSV, and the content-only artifact variant). The embeddings + search sections are
   spliced in here, after the keyword-bucket / timeline / entity sections.

> **Why the `lrt_lt` CSV used to look mangled.** The LRT scraper fetched its homepage with
> `requests` and parsed `response.text`; when the server omits a charset, `requests` guesses
> latin-1, which turns UTF-8 (`'`, `"`, Lithuanian `ė/š/ų`) into mojibake in the saved CSV.
> The scraper now parses `response.content` (bytes), so BeautifulSoup sniffs the real
> encoding — fresh rows are clean at the source. The existing rows were repaired in a
> one-time pass, and `pipeline.load_data()` still runs `ftfy` on load as a safety net for any
> legacy mojibake.

The page is **fully self-contained** — inline CSS/JS, the Baltic map baked in as SVG,
no external requests — so it works as a plain file, in an email, or embedded anywhere.

### The embeddings sections (`embeddings_analysis.py`)

`embeddings_analysis.py` runs the modern "semantic" stack from the
[DataHarvest 2026 workshop](https://github.com/resolveworks/dataharvest2026)
**alongside** the curated themes — it does **not** replace them. Its sections are now
**part of the main page** (spliced in after the keyword-bucket sections). It runs by
default; pass `--no-embed` for a fast, embeddings-free local build:

```bash
venv/Scripts/python.exe topic_analysis.py             # full page (with embeddings + search)
venv/Scripts/python.exe topic_analysis.py --no-embed  # fast: keyword-bucket / timeline / entity sections only
# lighter embedding model (bge-base instead of the default bge-large):
venv/Scripts/python.exe topic_analysis.py --embed-model BAAI/bge-base-en-v1.5
# or run the module on its own -> analysis/embeddings-prototype.html
venv/Scripts/python.exe embeddings_analysis.py
```

What it does (scope = `content`, i.e. headline + body):

1. **Embed** each article into a semantic vector — meaning, not word overlap. Vectors are
   **cached by URL** (`analysis/emb_cache/`), so each run only embeds *new* articles.
   *(Caveat: the cache is keyed by URL only, so if an article's text changes but its URL
   doesn't, the old vector is reused until that URL next appears new.)*
2. **UMAP → 2-D** map with `metric="cosine"` (compare *directions*, not distance — in high
   dimensions Euclidean distance collapses; embedding meaning lives in the angle).
3. **Cluster** with **BERTopic** when installed, else a hand-rolled **UMAP → 5-D → HDBSCAN**
   (identical recipe). `cluster == -1` are **outliers**; borderline noise is then folded into
   the nearest cluster, à la BERTopic's `reduce_outliers`. Set `BALTICS_NO_BERTOPIC=1` to
   force the built-in path.
4. **Label** each cluster with a KeyBERT-style, class-based-TF-IDF phrase (the term whose
   *meaning* sits closest to the cluster centroid), and **track topics across days** so a
   topic keeps its name + colour build to build.
5. **Escalation-tone** scoring per article (cosine to anchor phrases), a **topic-river**
   streamgraph, and an **anomalies** list of the most isolated stories.

**In-browser semantic search (transformers.js).** The corpus is also embedded with a small
model (`BAAI/bge-small-en-v1.5`) whose weights are int8-quantised and inlined (~0.5 KB/article).
The search box embeds *your query* client-side with the transformers.js twin
(`Xenova/bge-small-en-v1.5`, loaded on first use from a CDN) and ranks articles by cosine
similarity. This is the one feature that touches the network — and only when you search; the
model is cached after the first (~30 MB) download and the rest of the page needs no network.

**Model & install.** Default is `BAAI/bge-large-en-v1.5` via **fastembed** (ONNX, no torch —
top of the English [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard));
`--embed-model BAAI/bge-base-en-v1.5` is a lighter one-flag swap. HDBSCAN ships in
scikit-learn ≥1.9; the extra deps are `umap-learn`, `fastembed` and (optional) `bertopic`
(plus `sentence-transformers` only if you pick a model fastembed lacks). Text is the
broadcasters' **English** editions, so an English model fits.

> **Read it as an aid, not the primary labels:** cluster names are unstable run-to-run and a
> many-cluster scatter can't be made colour-blind-safe by colour alone. It's a
> **navigation / anomaly-finding** aid — the curated keyword buckets remain the primary,
> stable themes. Every heavy dep (`fastembed`, `umap-learn`, `bertopic`) is imported lazily
> and installed in a **non-fatal** CI step: if any fails, the page falls back cleanly
> (BERTopic → built-in HDBSCAN; whole embedding stack missing → keyword-bucket page with no
> embeddings/search).

## The report — what's on the page

Top to bottom:

1. **The numbers — how much is defence?**
   A map of Estonia/Latvia/Lithuania, each country shaded by its defence-news share,
   with per-country date ranges and a coverage caption. (Read the on-page caveat: the
   shares are **not** strictly comparable between countries — see *Caveats*.)
2. **The timeline — when, and from where?**
   Defence headlines per day, stacked by country; drag the range handles to zoom and
   click a span to list its articles. A legend toggles each country on/off.
3. **The words — what comes up most.**
   A bar chart of the most frequent headline terms; click a word to list the articles
   whose headline contains it (newest first).
4. **The players — who is named beside whom.**
   The spaCy entity network (people / orgs / places; an edge = a shared article), with a
   web view, a co-occurrence matrix view, and a per-entity dossier (mentions, tone, weekly
   trend, country split, neighbours).
5. **Search — find stories by meaning.**
   A semantic search box (transformers.js, in-browser) that ranks articles by meaning
   rather than keywords.
6. **Discovered topics — clusters found by meaning.**
   One panel (*"Topic map & anomalies"*): the embeddings map (BERTopic/HDBSCAN topics,
   coloured and labelled), a topic-river streamgraph, an escalation-tone heatmap, the
   clickable topic list, and an anomalies list. Built from the text, *independent* of the
   keyword buckets.

The design is a deliberate **Baltic woven-textile** look — a madder-red + gold + cream
palette (deep-wine in dark mode), a patterned masthead, gold-framed panels, folk-sign
section markers (Auseklis / Saule / Egle), and woven ornament bands. It adapts to light
and dark automatically.

## Running it locally

```bash
# one-time setup
python -m venv venv
venv/Scripts/python -m pip install jupyter lxml pandas requests beautifulsoup4 html5lib scikit-learn playwright playwright-stealth
venv/Scripts/python -m playwright install chromium
# analysis extras (headline mojibake repair + spaCy entities)
venv/Scripts/python -m pip install ftfy spacy && venv/Scripts/python -m spacy download en_core_web_sm
# semantic layer (embeddings + search + BERTopic clustering); all optional — the build
# degrades gracefully if any are missing. transformers.js needs NO install (CDN, in-browser).
venv/Scripts/python -m pip install fastembed umap-learn bertopic

# scrape (run each notebook), then build the report
jupyter nbconvert --to notebook --execute scrapers/err_ee.ipynb --stdout
jupyter nbconvert --to notebook --execute scrapers/lrt_lt.ipynb --stdout
jupyter nbconvert --to notebook --execute scrapers/lsm_lv.ipynb --stdout
venv/Scripts/python topic_analysis.py        # writes analysis/baltics-monitor.html
```

**Run everything from the repo root** — all paths (`data/`, `analysis/`) are relative to
it, and that's where CI runs too. Open `analysis/baltics-monitor.html` in a browser to
preview.

> On Windows, if `scikit-learn` fails to import with an "Application Control policy"
> error, that's **Smart App Control** blocking its unsigned binary — it has to be turned
> off (permanent) for the local run to work, or use the containerised `.devcontainer/`.

## The daily automation (GitHub Actions)

Automation is split across **two** workflows (both also runnable on demand via the
**Run workflow** button), scheduled on opposite hours so a scrape and a build never
collide:

- **`.github/workflows/scrape.yml`** (`scrape`) — **every 2 hours, on even hours**
  (`0 */2 * * *`). Installs Playwright + Chromium, executes the three scraper notebooks,
  and commits the refreshed `data/` back to **this** repo. It does **not** build the page.
- **`.github/workflows/dashboard.yml`** (`dashboard`) — **twice a day, on odd hours**
  (01:00 and 13:00 UTC). Installs the analysis deps (core must-succeed; embeddings, NER,
  BERTopic and the LLM-labeller each in their own optional `continue-on-error` step), runs
  `topic_analysis.py`, commits the refreshed `analysis/` back to this repo, and
  **publishes** the report to the website repo (next section).

To trigger by hand: **Actions → "scrape"** or **"dashboard" → Run workflow**.

## Publishing explained (the complicated part)

Here it is in full — including *why* it's shaped this way.

**The goal:** get the finished report onto the Nikola site so that clicking
**"Baltic monitor"** in the site's `Content` menu opens the report as its own
**full-screen page** at `https://jule4ka.github.io/content/baltics_monitor/`.

**The Nikola facts that matter:**
- The site's **source** lives on the **`src`** branch, under `jul_site/`.
- Nikola turns two kinds of source into a site:
  - **`jul_site/files/…`** — copied to the output **verbatim** (no processing at all).
  - **`jul_site/pages/…`** — run through Nikola's **HTML compiler + blog theme**.
- Our report is a **whole HTML document** (its own `<html>`, a huge `<style>`, scripts).
  A `pages/` page would be run through the compiler, which **strips the `<style>`** and
  wraps the content in the blog theme — so it renders **unstyled/broken**. (We hit this.)

**So the report goes under `files/`, not `pages/`.** Because `files/` is copied
verbatim, the whole document is served **exactly as generated** — full-screen, with its
own design and no site chrome around it. This is the simplest, most robust approach, and
it's why there is **no `make_site_page.py`** anymore (an earlier attempt wrapped the
report inside a `pages/` page via an iframe; we went back to the simpler full-page
version).

**What the workflow's publish step actually does:**
1. Clones the website repo's **`src`** branch (using the token — see next section).
2. Copies the report to **`jul_site/files/content/baltics_monitor/index.html`**. Because
   `files/` maps to the site root, this serves at **`/content/baltics_monitor/`** — the
   URL the `Content` menu link in `conf.py` points to.
3. Removes any older placements from earlier experiments so nothing collides.
4. Runs **`nikola build`** then **`nikola github_deploy`**, which builds the site and
   publishes it (commits source to `src`, pushes the built output to the deploy branch).

Result: **"Baltic monitor"** in the menu opens the full-screen report at
**`/content/baltics_monitor/`**.

## One-time setup: the token

The publish step needs to push into the **other** repo, so it uses a secret:

- **Secret name:** `PAGES_PUSH_TOKEN` (in **this** repo → Settings → Secrets and
  variables → Actions).
- **Value:** a GitHub **Personal Access Token** with **write access to
  `Jule4ka/Jule4ka.github.io`**:
  - *Fine-grained PAT* → that repo only, **Contents: Read and write**, or
  - *Classic PAT* → `repo` scope.
- Fine-grained tokens **expire**; when one does, the publish step starts failing and you
  regenerate it and **update** the secret. (A classic token with no expiry is
  set-and-forget but broader.)

## Caveats & gotchas

- **The per-country % shares are not directly comparable.** Each broadcaster's English
  page carries a different mix (LSM's homepage includes weather/culture/local news;
  LRT's is a curated *news-in-english* feed) and was scraped a different number of times.
  The numbers reflect *what landed on each page*, not newsroom priorities. (There's an
  on-page note saying this.)
- **Dates are scrape dates, not publish dates** — "collected between X and Y".
- **LSM depends on a real browser.** If Playwright/Chromium isn't installed or Cloudflare
  blocks the runner, `scrapers/scrape.py` fails — and `lsm_lv.ipynb` now runs it with
  `subprocess(check=True)` so it **fails loudly** instead of silently re-parsing a stale
  `scrapers/lsm_page.html`. Cloudflare is harsher on datacenter IPs, so CI may occasionally be
  blocked; the most reliable LSM refresh is a local run.
- **The Nikola build runs in CI.** If a future `Nikola[extras]` install/build breaks on
  the runner's Python, pin `python-version` in the workflow to whatever the site builds
  on locally.

## Troubleshooting

- **The published page looks unstyled / broken.** The report ended up under `pages/`
  (which compiles + themes it) instead of `files/`. It must be copied **verbatim** to
  `jul_site/files/content/baltics_monitor/index.html`.
- **Publish step fails with a 403/auth error.** `PAGES_PUSH_TOKEN` is missing, expired,
  or lacks write access to `Jule4ka/Jule4ka.github.io`. Recreate it (above).
- **LSM data stopped updating but ERR/LRT are fine.** Cloudflare blocked the CI scrape;
  check the `lsm_lv` step logs, or refresh LSM from a local run.
- **A scheduled run went red.** Open the failing step in the **Actions** tab — it's
  almost always either the LSM/Cloudflare scrape or the Nikola build.
```
