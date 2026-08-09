# Baltics / Nordics Monitor

An automated **defence & geopolitics news monitor** for the Baltic states. Every
two hours it scrapes the English editions of the three Baltic public
broadcasters, keeps only the security-related stories (Russia, Ukraine, NATO,
China, drones, war, sabotage…), and renders a single self-contained interactive
web page — a **country map of how much of each broadcaster's news is defence**,
followed by a **theme map that clusters the actual articles**. It then publishes
that page to a GitHub Pages site. No server; it runs entirely on GitHub Actions.

---

## Sources

| Broadcaster | Country | Notebook | Master CSV |
|---|---|---|---|
| ERR | 🇪🇪 Estonia | `err_ee.ipynb` | `data/err_ee-always-updated.csv` |
| LRT | 🇱🇹 Lithuania | `lrt_lt.ipynb` | `data/lrt_lt-always-updated.csv` |
| LSM | 🇱🇻 Latvia | `lsm_lv.ipynb` | `data/lsm_lv-always-updated.csv` |

## Project structure

```
baltics_nordics_monitor/
├── data/                              # inputs
│   ├── <src>-always-updated.csv       #   master, deduplicated-by-URL archive per source
│   └── err_ee/  lrt_lt/  lsm_lv/       #   timestamped snapshot of every scrape run
├── analysis/                          # generated outputs (regenerated every run)
│   ├── nordics-monitor.html           #   the report — self-contained, this is what gets published
│   ├── nordics-monitor-artifact.html  #   content-only variant (for a Claude Artifact wrapper)
│   ├── nordics-monitor-data.json      #   the computed data behind the page
│   └── nordics-monitor-clustered.csv  #   every matched article tagged with its cluster
├── err_ee.ipynb  lrt_lt.ipynb  lsm_lv.ipynb   # the three scrapers
├── scrape.py                          # Playwright helper that fetches LSM past Cloudflare
├── topic_analysis.py                  # the analysis + report builder
├── lsm_page.html                      # scrape.py's rendered LSM homepage (intermediate)
└── .github/workflows/                 # the every-2-hours automation
```

## How it works

### 1. Scrape (the notebooks)
Each notebook fetches its broadcaster's homepage, extracts headline + URL, and
**appends to two places**: a timestamped snapshot in `data/<src>/`, and the
master `data/<src>-always-updated.csv` (deduplicated by URL, so only genuinely
new articles are added).

- **ERR** and **LRT** fetch with a plain `requests.get()`.
- **LSM** sits behind **Cloudflare**, which blocks plain requests. So
  `lsm_lv.ipynb` first runs **`scrape.py`**, which drives a headless
  **Playwright + stealth** browser to clear the Cloudflare challenge and save the
  rendered page to `lsm_page.html` for the notebook to parse. (This is why LSM
  needs a real browser installed, not just `requests` — see *Caveats*.)

### 2. Analyse & render (`topic_analysis.py`)
1. Load the three master CSVs (repairing `lrt_lt` text encoding), dedupe by URL.
2. **Filter** headlines to the defence/geopolitics keyword set.
3. Fetch the **full article body** for each match (8 in parallel, failures skipped).
4. **TF-IDF** vectorise → **TruncatedSVD (LSA)** → **t-SNE** 2-D layout.
5. **KMeans** clustering (k chosen by silhouette), labelled by top per-cluster terms.
6. Render the report:
   - **Act 1 — the numbers:** a Baltic **choropleth**, each country shaded by its
     defence share (matched ÷ total headlines), with per-country date ranges.
   - **Act 2 — the themes:** the interactive **topic map** of clustered articles.

The page is **fully self-contained** (inline CSS/JS, borders baked in as SVG) —
no external requests — so it works as a static file and in a Claude Artifact.

## Running it locally

```bash
# one-time
python -m venv venv
venv/Scripts/python -m pip install jupyter lxml pandas requests beautifulsoup4 html5lib scikit-learn playwright playwright-stealth
venv/Scripts/python -m playwright install chromium

# scrape (run each notebook), then build the report
jupyter nbconvert --to notebook --execute err_ee.ipynb --stdout
jupyter nbconvert --to notebook --execute lrt_lt.ipynb --stdout
jupyter nbconvert --to notebook --execute lsm_lv.ipynb --stdout
venv/Scripts/python topic_analysis.py        # writes analysis/nordics-monitor.html
```

Notebooks use paths relative to the repo root — run them from the repo root
(that's also where CI runs them).

## Automation & publishing

`.github/workflows/` runs **once a day** (05:00 UTC) and on demand:
1. installs deps (incl. Playwright + Chromium),
2. executes the three scraper notebooks,
3. runs `topic_analysis.py`,
4. commits the refreshed `data/` + `analysis/` back to this repo,
5. **publishes** to the personal **Nikola** site (`Jule4ka/Jule4ka.github.io`),
   mirroring the existing `nordics-monitor`. The report is a full self-contained HTML
   document, so it goes under **`files/`** (copied to the site root verbatim — no
   compiler, no theme) at `jul_site/files/content/baltics_monitor/index.html`. A page
   under `pages/` would instead be run through Nikola's HTML compiler, which strips the
   top-level `<style>` block and wraps it in the blog theme → unstyled. Then
   `nikola build` + `nikola github_deploy` publishes it, live at
   **`jule4ka.github.io/content/baltics_monitor/`**.

Publishing needs a repo secret **`PAGES_PUSH_TOKEN`** — a Personal Access Token
with **write** access to `Jule4ka/Jule4ka.github.io`. Set it in this repo →
Settings → Secrets and variables → Actions.

## Caveats

- **LSM depends on a real browser.** If Playwright/Chromium isn't installed, or
  Cloudflare blocks the runner, `scrape.py` fails. `lsm_lv.ipynb` now runs it via
  `subprocess(..., check=True)`, so a failure **hard-fails the notebook loudly**
  instead of silently re-parsing a stale `lsm_page.html`. Cloudflare is stricter
  on datacenter IPs, so CI runs may still be blocked intermittently — the most
  reliable LSM refresh is a local run.
- **Dates are scrape dates, not publish dates** — the coverage range shows when
  articles were collected.
```
