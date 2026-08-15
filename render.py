#!/usr/bin/env python
"""
Report renderer: turn the analysed DataFrame into the self-contained HTML page.

Assembly is deliberately simple — glue the front-end assets together and splice
in the data:

    <style> report.css [+ embed css] </style>
    report.html  (with the embed sections injected before </main>)
    <script> report.js  (with __DATA__ replaced by the payload JSON) </script>
    <script> embed js </script>

The embeddings sections (semantic map with HDBSCAN-discovered topics + anomalies)
are now part of the MAIN page, sitting AFTER the original keyword-bucket sections —
not a separate file. If the embedding stack isn't installed (fastembed / umap),
we log a note and fall back to the TF-IDF-only page so a build never hard-fails.
"""
import os
import json

from config import OUT_DIR, THEMES, THEME_OTHER
from assets import CSS, BODY, SCRIPT
from pipeline import build_payload


def _wrap(inner, title="Baltic Defence &amp; Geopolitics Monitor"):
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title}</title></head><body>{inner}</body></html>"
    )


def _embed_fragment(df, embed_model):
    """Build the embeddings sections, or (None, None, None) if the stack is missing.

    Returns (extra_css, html_sections, extra_script). The heavy deps (fastembed,
    umap-learn) are imported lazily inside embeddings_analysis, so a missing one
    surfaces here as an exception we swallow — the page is still built without it.
    """
    try:
        import embeddings_analysis as emb
        # topics are discovered from the embeddings here, independent of the keyword themes
        return emb.build_report_fragment(df, model=embed_model or emb.DEFAULT_MODEL)
    except Exception as e:
        print(f"[embed] skipped — building TF-IDF-only page ({type(e).__name__}: {e})")
        return None, None, None


def render(df, themes, keywords, shares, embed=True, embed_model=None):
    payload = build_payload(df, themes, keywords, shares)
    data_js = json.dumps(payload, ensure_ascii=False)
    script = SCRIPT.replace("__DATA__", data_js)

    css, body, extra_script = CSS, BODY, ""
    if embed:
        ecss, ehtml, escript = _embed_fragment(df, embed_model)
        if ehtml is not None:
            css = CSS + ecss
            body = BODY.replace("</main>", ehtml + "\n  </main>")  # after the original sections
            extra_script = f"\n<script>{escript}</script>"

    inner = f"<style>{css}</style>\n{body}\n<script>{script}</script>{extra_script}"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/baltics-monitor.html", "w", encoding="utf-8") as f:
        f.write(_wrap(inner))
    with open(f"{OUT_DIR}/baltics-monitor-artifact.html", "w", encoding="utf-8") as f:
        f.write(inner)  # content-only, for the Claude Artifact wrapper
    with open(f"{OUT_DIR}/baltics-monitor-data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    kb = len(_wrap(inner)) // 1024
    tag = "with embeddings" if extra_script else "TF-IDF only"
    print(f"[render] wrote {OUT_DIR}/baltics-monitor.html ({kb} KB, self-contained, {tag})")
