# scrape.py — run with:  python scrapers/scrape.py
# Fetches the LSM homepage (gets past Cloudflare via stealth) and saves the
# rendered HTML to scrapers/lsm_page.html for the notebook to parse.
import os
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright_stealth import Stealth

URL = "https://eng.lsm.lv/"
# Write next to THIS file (scrapers/), not the caller's CWD, so the outputs land
# in the scrapers folder whether launched from the repo root, CI, or scrapers/.
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "lsm_page.html")
DEBUG_HTML = os.path.join(HERE, "lsm_debug.html")
DEBUG_PNG = os.path.join(HERE, "lsm_debug.png")

# headless=True is REQUIRED here: this runs unattended on a server/CI with no
# display, so a headed window would crash the browser before navigation.
with Stealth().use_sync(sync_playwright()) as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    try:
        # "commit" returns as soon as the response arrives, so a Cloudflare
        # interstitial can't stall navigation past the timeout; then we wait
        # for the real content selector to appear after the challenge clears.
        page.goto(URL, wait_until="commit", timeout=40000)
        page.wait_for_selector(".list-article", timeout=40000)   # waits out the challenge
    except PWTimeout:
        # Most likely a Cloudflare challenge we didn't clear. Dump what we got
        # so the block page is inspectable instead of failing blind.
        page.screenshot(path=DEBUG_PNG, full_page=True)
        with open(DEBUG_HTML, "w", encoding="utf-8") as f:
            f.write(page.content())
        print(f"FAILED: '.list-article' never appeared. Page title: {page.title()!r}",
              file=sys.stderr)
        print("Wrote lsm_debug.html + lsm_debug.png for inspection.", file=sys.stderr)
        browser.close()
        sys.exit(1)
    html = page.content()
    browser.close()

with open(PAGE, "w", encoding="utf-8") as f:
    f.write(html)

print("saved scrapers/lsm_page.html")
