#!/usr/bin/env python
"""Wrap the standalone monitor report in a Nikola PAGE for the jul_site (Nikola) site.

A page under Nikola's `pages/` is run through the HTML compiler + blog theme, which
strips a full document's top-level <style> and lets its body/* rules leak into the
theme. To keep the page under pages/ (served at /content/baltics_monitor/) while
rendering the design intact, we embed the whole standalone report inside an
`<iframe srcdoc="...">` — an isolated browsing context, so nothing is stripped or
leaks. A tiny script auto-sizes the frame to its content height.

Usage:
    python make_site_page.py analysis/nordics-monitor.html <out.html>
"""
import io
import sys


def build(report_html: str) -> str:
    # srcdoc is a double-quoted attribute: escape & first, then ".
    srcdoc = report_html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        "<!--\n"
        ".. title: Baltic Defence & Geopolitics Monitor\n"
        ".. slug: baltics_monitor\n"          # folder pages/content/ -> /content/baltics_monitor/
        ".. hidetitle: true\n"
        "-->\n"
        '<iframe id="baltics-monitor" title="Baltic Defence &amp; Geopolitics Monitor" '
        'srcdoc="' + srcdoc + '" '
        'style="width:100%;border:0;display:block;min-height:100vh" scrolling="no"></iframe>\n'
        "<script>(function(){var f=document.getElementById('baltics-monitor');"
        "function r(){try{f.style.height="
        "f.contentWindow.document.documentElement.scrollHeight+'px';}catch(e){}}"
        "f.addEventListener('load',r);setInterval(r,1200);})();</script>\n"
    )


def main():
    src, dst = sys.argv[1], sys.argv[2]
    report = io.open(src, encoding="utf-8").read()
    page = build(report)
    io.open(dst, "w", encoding="utf-8").write(page)
    print(f"wrote {dst} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
