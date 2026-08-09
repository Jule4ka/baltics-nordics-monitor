# Baltics Monitor

An automated **defence & geopolitics news monitor** for the Baltic states.

Once a day it scrapes the English editions of the three Baltic public broadcasters,
keeps only the security-related stories (Russia, Ukraine, NATO, China, drones, war,
sabotage…), analyses them, and builds **one self-contained interactive web page**.
It then publishes that page to a personal website. There's no server — everything
runs on **GitHub Actions**.

**Live page:** <https://jule4ka.github.io/content/baltics_monitor/>

> ⚠️ **Naming quirk:** the generated files are historically called
> `nordics-monitor.*`, but this is the **Baltics** monitor. The names are just legacy.

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
| ERR | 🇪🇪 Estonia | `err_ee.ipynb` | `data/err_ee-always-updated.csv` |
| LRT | 🇱🇹 Lithuania | `lrt_lt.ipynb` | `data/lrt_lt-always-updated.csv` |
| LSM | 🇱🇻 Latvia | `lsm_lv.ipynb` | `data/lsm_lv-always-updated.csv` |

Only the **English editions** are scraped.

## Data flow at a glance

```
  ┌─ err_ee.ipynb ─┐
  ├─ lrt_lt.ipynb ─┤   scrape homepages      ┌───────────────────┐
  └─ lsm_lv.ipynb ─┘  ───────────────────▶   │  data/*.csv        │  (archive of headlines)
        (LSM also uses scrape.py +           └─────────┬─────────┘
         Playwright to beat Cloudflare)                │
                                                        ▼
                                            ┌───────────────────────┐
                                            │  topic_analysis.py     │  filter → fetch bodies →
                                            │                        │  TF-IDF → themes → build
                                            └───────────┬───────────┘
                                                        ▼
                                            ┌───────────────────────┐
                                            │ analysis/              │
                                            │  nordics-monitor.html  │  ← the finished report
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
│   ├── nordics-monitor.html           #   the report — full self-contained page (THIS is published)
│   ├── nordics-monitor-artifact.html  #   content-only variant (no <html>/<head>; for embedding)
│   ├── nordics-monitor-data.json      #   the computed data behind the page
│   └── nordics-monitor-clustered.csv  #   every matched article tagged with its theme
│
├── err_ee.ipynb  lrt_lt.ipynb  lsm_lv.ipynb   # the three scrapers
├── scrape.py                          # Playwright helper: fetches the LSM homepage past Cloudflare
├── topic_analysis.py                  # the analysis + report builder (the heart of the project)
├── lsm_page.html                      # scrape.py's rendered LSM homepage (intermediate file)
└── .github/workflows/                 # the once-a-day automation
```

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
  first runs **`scrape.py`**, which drives a headless **Playwright + stealth** browser
  to clear the Cloudflare challenge and save the rendered HTML to `lsm_page.html` for
  the notebook to parse. (This is why LSM needs a real browser — see *Caveats*.)

### 2. Analyse & build the report (`topic_analysis.py`)

1. **Load** the three master CSVs (repairing `lrt_lt`'s text encoding), dedupe by URL.
2. **Filter** headlines to the defence/geopolitics keyword set (Ukraine, Russia, NATO,
   drones, border, China, sabotage, …).
3. **Per-country share** — for Act 1, how many of each source's headlines matched
   (`matched ÷ total`), plus each source's date range.
4. **Fetch the full article body** for every matched article (8 in parallel; failures
   skipped).
5. **Vectorise & lay out** — TF-IDF → TruncatedSVD (LSA) → **t-SNE** into a 2-D map,
   so articles with similar wording sit near each other. *(This is the "proximity"; there
   is no more KMeans clustering — that was removed because the clusters overlapped and
   meant nothing.)*
6. **Assign a theme** to each article by the keyword bucket its words hit hardest
   (Ukraine, Russia & Kremlin, NATO & defence, Drones & airspace, Border & migration,
   Hybrid & sabotage, China, or "Other").
7. **Keyword frequency** — count, across the **headlines**, how many articles mention
   each word (merging variants: russia/russian → russia, drones → drone, …).
8. **Render** the report to `analysis/nordics-monitor.html` (+ the JSON, the tagged CSV,
   and the content-only artifact variant).

The page is **fully self-contained** — inline CSS/JS, the Baltic map baked in as SVG,
no external requests — so it works as a plain file, in an email, or embedded anywhere.

## The report — what's on the page

Three "acts":

1. **The numbers — how much is defence?**
   A map of Estonia/Latvia/Lithuania, each country shaded by its defence-news share,
   with per-country date ranges and a coverage caption. (Read the on-page caveat: the
   shares are **not** strictly comparable between countries — see *Caveats*.)
2. **The themes — what is it about?**
   The proximity map: each article is a dot, coloured and labelled by its theme; nearby
   dots share wording. A legend toggles themes; below it, one card per theme lists its
   articles (newest first) when you click it.
3. **The words — what comes up most.**
   A bar chart of the most frequent headline terms; click a word to list the articles
   whose headline contains it (newest first).

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

# scrape (run each notebook), then build the report
jupyter nbconvert --to notebook --execute err_ee.ipynb --stdout
jupyter nbconvert --to notebook --execute lrt_lt.ipynb --stdout
jupyter nbconvert --to notebook --execute lsm_lv.ipynb --stdout
venv/Scripts/python topic_analysis.py        # writes analysis/nordics-monitor.html
```

**Run everything from the repo root** — all paths (`data/`, `analysis/`) are relative to
it, and that's where CI runs too. Open `analysis/nordics-monitor.html` in a browser to
preview.

> On Windows, if `scikit-learn` fails to import with an "Application Control policy"
> error, that's **Smart App Control** blocking its unsigned binary — it has to be turned
> off (permanent) for the local run to work, or use the containerised `.devcontainer/`.

## The daily automation (GitHub Actions)

`.github/workflows/dklsdjknsdkjnsd.yml` runs **once a day at 05:00 UTC** (and on demand
via the **Run workflow** button). Each run:

1. installs dependencies (incl. Playwright + Chromium),
2. executes the three scraper notebooks,
3. runs `topic_analysis.py`,
4. commits the refreshed `data/` + `analysis/` back to **this** repo,
5. **publishes** the report to the website repo (next section).

To trigger it by hand: **Actions → "daily" → Run workflow**.

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
  blocks the runner, `scrape.py` fails — and `lsm_lv.ipynb` now runs it with
  `subprocess(check=True)` so it **fails loudly** instead of silently re-parsing a stale
  `lsm_page.html`. Cloudflare is harsher on datacenter IPs, so CI may occasionally be
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
