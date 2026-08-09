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
import json
import sys


def build(report_html: str) -> str:
    # Embed the report as a JS string and build the <iframe srcdoc> at runtime.
    # Nikola's HTML page compiler strips <style> and mangles escaped attributes, but it
    # leaves <script> content untouched — so we carry the whole document inside a script.
    # json.dumps makes a valid JS string; escape "</" so a nested </script> can't close us.
    js = json.dumps(report_html).replace("</", "<\\/")
    return (
        "<!--\n"
        ".. title: Baltic Defence & Geopolitics Monitor\n"
        ".. slug: baltics_monitor\n"          # folder pages/content/ -> /content/baltics_monitor/
        ".. hidetitle: true\n"
        "-->\n"
        '<div id="baltics-monitor-mount"></div>\n'
        "<script>\n"
        "var BALTICS_MONITOR_HTML = " + js + ";\n"
        "(function(){\n"
        "  var f = document.createElement('iframe');\n"
        "  f.title = 'Baltic Defence \\u0026 Geopolitics Monitor';\n"
        "  f.setAttribute('style','width:100%;border:0;min-height:100vh;display:block');\n"
        "  f.srcdoc = BALTICS_MONITOR_HTML;\n"
        "  function resize(){try{f.style.height="
        "f.contentWindow.document.documentElement.scrollHeight+'px';}catch(e){}}\n"
        "  f.addEventListener('load', function(){resize(); setInterval(resize, 1200);});\n"
        "  document.getElementById('baltics-monitor-mount').appendChild(f);\n"
        "})();\n"
        "</script>\n"
    )


def main():
    src, dst = sys.argv[1], sys.argv[2]
    report = io.open(src, encoding="utf-8").read()
    page = build(report)
    io.open(dst, "w", encoding="utf-8").write(page)
    print(f"wrote {dst} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
