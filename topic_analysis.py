#!/usr/bin/env python
"""
Baltics Monitor — command-line entrypoint.

This file is deliberately thin: it just wires the pieces together. The actual
work lives in focused modules so each part is easy to find and edit:

  config.py      sources, the defence KEYWORDS filter, THEME buckets, palette
  pipeline.py    load -> filter -> fetch -> analyse (regex keyword-bucket themes)
  assets/        report.css / report.html / report.js  (the whole front-end)
  assets.py      loads those three files
  render.py      assembles the self-contained page (embeddings folded into it)
  embeddings_analysis.py   the semantic-embeddings sections (fastembed -> UMAP -> HDBSCAN)

Run:
  venv/Scripts/python.exe topic_analysis.py                 # full page (with embeddings)
  venv/Scripts/python.exe topic_analysis.py --no-embed      # fast: TF-IDF sections only
  venv/Scripts/python.exe topic_analysis.py --embed-model BAAI/bge-large-en-v1.5

Output (analysis/): baltics-monitor.html  (published to
jule4ka.github.io/content/baltics_monitor/) + -artifact.html + -data.json + -clustered.csv
"""
import sys
import warnings
import argparse

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import OUT_DIR, THEMES, THEME_OTHER
from pipeline import load_data, filter_themes, compute_shares, fetch_all, analyse
from render import render


def main():
    ap = argparse.ArgumentParser(description="Baltics Monitor report generator")
    ap.add_argument("--no-embed", dest="embed", action="store_false",
                    help="skip the semantic-embeddings sections (faster; needs no "
                         "fastembed/umap). Default: embeddings ARE built into the page.")
    ap.add_argument("--embed-model", default=None,
                    help="embedding model id (default: bge-large via fastembed; "
                         "pass e.g. BAAI/bge-base-en-v1.5 for a lighter run)")
    args = ap.parse_args()

    df_all = load_data()
    df = filter_themes(df_all)
    shares = compute_shares(df_all, df)
    df["body"] = fetch_all(df["url"].tolist())
    df, themes, keywords = analyse(df)
    # per-article escalation tone (for entity coverage-tone); reuses cached embeddings
    if args.embed:
        try:
            import embeddings_analysis as emb
            df["tone"] = emb.article_tone(df, args.embed_model or emb.DEFAULT_MODEL)
        except Exception as e:
            print(f"[tone] skipped ({type(e).__name__})")
    render(df, themes, keywords, shares, embed=args.embed, embed_model=args.embed_model)

    names = [t[0] for t in THEMES] + [THEME_OTHER]
    out = df[["url", "headline", "source", "scrape_date", "theme"]].copy()
    out["theme_name"] = df["theme"].map(lambda i: names[i])
    out["all_themes"] = df["themes"].map(lambda ms: "; ".join(names[i] for i in ms))
    out.to_csv(f"{OUT_DIR}/baltics-monitor-clustered.csv", index=False, encoding="utf-8")
    print(f"[done] {OUT_DIR}/ baltics-monitor.html + artifact + data json + clustered csv")


if __name__ == "__main__":
    main()
