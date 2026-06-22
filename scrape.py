# scrape.py — run with:  python scrape.py
# Fetches the LSM homepage (gets past Cloudflare via stealth) and
# saves the rendered HTML to lsm_page.html for the notebook to parse.
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

URL = "https://eng.lsm.lv/"

with Stealth().use_sync(sync_playwright()) as p:
    browser = p.chromium.launch(headless=True)   # visible window clears CF best
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_selector(".list-article", timeout=40000)   # waits out the challenge
    html = page.content()
    browser.close()

with open("lsm_page.html", "w", encoding="utf-8") as f:
    f.write(html)

print("saved lsm_page.html")
