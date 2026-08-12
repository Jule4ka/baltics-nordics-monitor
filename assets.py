#!/usr/bin/env python
"""
Front-end asset loader.

The report's CSS / HTML / JS live as plain files under assets/ so they can be
edited with proper syntax highlighting (they used to be big triple-quoted string
constants inside topic_analysis.py). This module just reads them once at import
and exposes them as CSS / BODY / SCRIPT, exactly the names the renderer expects.

Edit the look & feel here:
  assets/report.css   — styles
  assets/report.html  — page body markup (the <div class="shell"><main>…)
  assets/report.js     — the interactive Canvas map + charts (uses __DATA__ token)
"""
import os

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load(name):
    with open(os.path.join(_DIR, name), encoding="utf-8") as f:
        return f.read()


CSS = load("report.css")     # was topic_analysis.CSS
BODY = load("report.html")   # was topic_analysis.BODY
SCRIPT = load("report.js")   # was topic_analysis.SCRIPT  (contains the __DATA__ token)
