#!/usr/bin/env python
"""
Configuration & domain vocabulary for the Baltics Monitor.

Everything here is data you might want to tune by hand: which sources to read,
the defence/geopolitics keyword filter, the theme buckets, and the colour
palette. No logic runs at import beyond building a couple of compiled regexes.

  • FILES / SRC_*          — the three source CSVs and how each is labelled
  • BALTIC_MAP             — baked-in country borders for the SVG map
  • KEYWORDS               — the headline filter (an article must hit this to count)
  • THEMES / THEME_*       — hand-authored keyword buckets (NOT ML clusters)
  • THEME_COLORS           — the categorical palette (light + dark)
"""
import re

DATA_DIR = "data"        # the always-updated source CSVs live here
OUT_DIR = "analysis"     # generated report + data artifacts land here
FILES = {
    "err_ee": f"{DATA_DIR}/err_ee-always-updated.csv",
    "lrt_lt": f"{DATA_DIR}/lrt_lt-always-updated.csv",
    "lsm_lv": f"{DATA_DIR}/lsm_lv-always-updated.csv",
}
SRC_LABEL = {"err_ee": "ERR (Estonia)", "lrt_lt": "LRT (Lithuania)", "lsm_lv": "LSM (Latvia)"}
SRC_CODE = {"err_ee": "EE", "lrt_lt": "LT", "lsm_lv": "LV"}
# Landing page each source is scraped from — the country shapes link back to these.
SRC_HOME = {"err_ee": "https://news.err.ee/",
            "lrt_lt": "https://www.lrt.lt/en/news-in-english",
            "lsm_lv": "https://eng.lsm.lv"}

# Real EE/LV/LT borders, projected to a shared SVG viewbox (see scratchpad/build_baltic_svg.py).
# Baked in so the pipeline stays offline; geometry is static, only the shading changes per run.
BALTIC_MAP = {
    "viewBox": "0 0 440 625",
    "paths": {
        "EE": "M382.6,231.6 L381.2,231.9 L373.1,230.6 L364.1,226.3 L360.2,223.1 L356.4,223.1 L351.7,225.2 L335.1,231.3 L331.0,229.9 L321.4,223.9 L316.6,217.4 L305.9,204.4 L305.0,201.3 L303.6,198.8 L292.2,195.6 L287.9,190.8 L284.4,190.1 L279.2,187.7 L265.8,177.5 L262.5,176.5 L261.7,178.2 L261.9,180.7 L261.1,182.1 L259.4,182.0 L256.2,178.2 L252.5,174.9 L240.9,181.2 L236.7,182.8 L233.1,183.2 L214.7,191.4 L209.1,195.8 L206.7,195.4 L207.3,191.2 L214.9,170.4 L216.3,153.9 L219.1,151.7 L219.9,149.4 L218.7,144.1 L210.8,140.7 L207.6,141.2 L204.7,146.9 L201.7,151.0 L194.7,153.5 L188.6,149.2 L174.5,143.4 L170.9,135.8 L170.1,128.1 L162.6,120.6 L159.5,111.9 L160.8,105.8 L167.5,101.7 L169.5,98.3 L160.9,98.8 L159.2,98.0 L158.8,94.8 L155.0,84.1 L158.4,79.9 L159.9,75.8 L157.1,72.3 L157.8,68.3 L160.0,64.3 L158.7,55.0 L167.1,50.0 L175.4,46.6 L192.9,44.8 L191.1,36.3 L198.2,35.9 L210.1,25.6 L221.9,27.4 L239.0,20.4 L271.8,20.5 L276.3,16.4 L275.5,12.3 L275.6,8.0 L281.8,9.2 L292.2,8.5 L330.9,17.0 L340.4,17.0 L353.6,25.7 L360.7,28.0 L381.7,28.0 L414.0,31.8 L420.4,25.9 L421.0,24.4 L424.1,27.7 L428.0,33.0 L429.0,36.0 L427.7,37.8 L423.8,39.3 L422.9,41.0 L421.2,43.7 L416.7,44.2 L414.3,46.3 L411.5,55.3 L406.2,70.2 L398.3,81.5 L392.0,87.7 L389.2,92.5 L387.4,98.2 L387.0,104.0 L393.1,135.5 L393.0,141.2 L391.6,147.1 L390.5,153.0 L391.4,158.2 L395.4,167.0 L399.6,180.2 L401.3,188.6 L404.1,191.7 L406.8,193.9 L407.4,195.3 L407.3,196.8 L405.9,198.5 L393.7,202.9 L392.1,206.6 L390.8,210.8 L385.4,216.9 L383.8,222.7 L382.8,229.3 L382.6,231.6ZM107.7,115.8 L111.9,118.4 L115.7,117.6 L119.5,115.8 L127.9,117.5 L147.0,130.4 L148.7,133.9 L137.3,135.5 L134.7,139.4 L132.0,142.2 L128.8,143.1 L123.3,148.7 L115.8,154.0 L114.3,157.2 L100.8,156.6 L93.5,158.6 L87.5,164.6 L85.1,176.2 L80.7,185.2 L76.3,188.4 L71.7,188.9 L70.6,185.5 L71.0,182.2 L80.8,169.4 L82.8,165.3 L77.9,163.5 L73.9,159.0 L65.0,153.9 L63.4,149.7 L65.6,149.4 L67.5,148.2 L69.9,144.7 L71.0,140.7 L63.9,129.0 L67.5,127.2 L72.0,127.6 L76.6,131.0 L81.7,127.0 L83.8,126.4 L87.4,127.8 L90.9,120.1 L99.4,117.6 L103.6,115.2 L107.7,115.8ZM125.5,94.0 L120.8,99.3 L117.9,97.2 L116.4,94.7 L110.3,106.5 L103.4,108.5 L99.3,106.2 L99.7,101.8 L95.7,90.2 L89.7,86.8 L81.3,86.5 L75.1,81.7 L98.7,78.4 L101.2,72.9 L106.0,67.1 L109.6,66.5 L112.7,67.8 L113.2,72.3 L114.0,74.1 L124.7,76.6 L128.9,84.2 L130.5,93.3 L125.5,94.0ZM149.9,123.3 L145.1,124.4 L133.6,116.9 L136.3,111.8 L139.6,109.8 L149.3,112.9 L150.6,120.7 L149.9,123.3Z",
        "LV": "M338.6,428.8 L335.7,428.2 L327.4,424.9 L320.5,420.0 L316.3,413.4 L309.1,404.5 L304.4,399.9 L297.0,394.1 L284.6,382.4 L280.1,379.7 L258.1,374.6 L250.1,372.3 L242.8,359.0 L240.4,351.3 L236.8,350.0 L232.9,351.6 L228.6,353.1 L218.7,362.1 L215.5,363.4 L209.4,363.6 L195.0,365.5 L188.5,362.3 L177.1,358.7 L171.0,358.1 L165.5,358.2 L141.3,354.6 L136.9,358.5 L132.4,359.2 L128.1,353.2 L122.7,351.5 L116.8,353.6 L106.0,353.8 L93.1,351.9 L76.8,350.4 L74.4,351.1 L56.2,359.0 L51.8,360.2 L32.1,373.6 L16.5,386.1 L14.7,366.1 L15.6,326.1 L18.0,306.3 L28.8,294.7 L34.2,285.7 L37.3,273.7 L38.3,262.6 L40.5,253.4 L56.1,227.1 L68.5,224.3 L85.3,216.9 L104.1,210.9 L107.7,218.6 L109.5,224.5 L132.1,246.1 L137.9,253.3 L146.6,278.1 L167.6,290.7 L184.0,286.7 L191.2,280.6 L204.3,269.3 L210.2,261.1 L211.4,253.2 L209.1,219.2 L205.5,204.5 L206.7,195.4 L209.1,195.8 L214.7,191.4 L233.1,183.2 L236.7,182.8 L240.9,181.2 L252.5,174.9 L256.2,178.2 L259.4,182.0 L261.1,182.1 L261.9,180.7 L261.7,178.2 L262.5,176.5 L265.8,177.5 L279.2,187.7 L284.4,190.1 L287.9,190.8 L292.2,195.6 L303.6,198.8 L305.0,201.3 L305.9,204.4 L316.6,217.4 L321.4,223.9 L331.0,229.9 L335.1,231.3 L351.7,225.2 L356.4,223.1 L360.2,223.1 L364.1,226.3 L373.1,230.6 L381.2,231.9 L382.6,231.6 L389.5,232.1 L391.9,233.8 L393.5,242.1 L401.3,248.6 L408.5,254.0 L410.3,256.5 L410.9,261.3 L410.4,267.0 L409.5,269.9 L406.5,273.3 L403.9,281.8 L403.5,289.9 L399.3,303.9 L400.3,304.2 L409.0,301.7 L411.5,303.1 L413.4,306.2 L414.0,315.0 L416.9,319.0 L419.8,325.2 L420.7,330.0 L426.3,335.7 L426.7,339.4 L430.1,352.5 L431.4,360.1 L432.0,365.9 L430.3,373.4 L428.9,378.4 L427.1,378.1 L422.1,379.4 L414.2,385.5 L402.5,399.7 L399.5,402.9 L396.4,413.8 L395.7,414.9 L388.9,414.4 L387.0,414.1 L380.2,414.3 L365.3,411.5 L359.5,413.4 L351.9,424.3 L349.0,426.0 L340.2,427.5 L338.6,428.8Z",
        "LT": "M11.4,469.9 L8.0,469.1 L14.6,456.9 L17.2,449.0 L18.9,437.7 L20.5,434.2 L20.5,439.3 L19.9,447.8 L15.7,462.3 L11.4,469.9ZM116.4,567.6 L113.9,562.5 L111.4,553.2 L111.6,545.8 L113.1,538.4 L120.1,516.6 L119.8,513.1 L114.6,507.0 L108.3,502.6 L104.8,493.2 L92.0,492.7 L79.9,493.2 L76.1,492.7 L64.6,488.8 L53.5,482.5 L46.0,478.8 L39.8,474.6 L36.4,470.3 L31.1,471.5 L27.5,471.5 L27.5,470.8 L25.5,463.1 L27.6,451.3 L23.8,434.0 L17.4,413.3 L16.9,391.1 L16.5,386.1 L32.1,373.6 L51.8,360.2 L56.2,359.0 L74.4,351.1 L76.8,350.4 L93.1,351.9 L106.0,353.8 L116.8,353.6 L122.7,351.5 L128.1,353.2 L132.4,359.2 L136.9,358.5 L141.3,354.6 L165.5,358.2 L171.0,358.1 L177.1,358.7 L188.5,362.3 L195.0,365.5 L209.4,363.6 L215.5,363.4 L218.7,362.1 L228.6,353.1 L232.9,351.6 L236.8,350.0 L240.4,351.3 L242.8,359.0 L250.1,372.3 L258.1,374.6 L280.1,379.7 L284.6,382.4 L297.0,394.1 L304.4,399.9 L309.1,404.5 L316.3,413.4 L320.5,420.0 L327.4,424.9 L335.7,428.2 L338.6,428.8 L338.4,433.5 L337.0,441.6 L334.3,452.0 L331.4,460.1 L330.7,463.2 L332.9,465.8 L343.7,467.0 L348.3,468.4 L349.2,470.5 L346.8,473.3 L343.3,475.7 L341.8,477.8 L339.0,485.7 L321.1,484.7 L318.7,486.3 L317.6,489.9 L316.7,494.1 L314.3,499.1 L309.5,503.5 L302.1,505.1 L296.0,508.0 L291.4,517.1 L288.0,529.4 L288.1,538.0 L288.6,542.9 L288.2,545.6 L285.9,548.7 L282.1,556.6 L279.0,565.5 L277.9,570.3 L278.4,572.5 L281.9,572.6 L286.9,574.4 L289.5,577.9 L290.5,582.0 L290.5,586.4 L289.6,588.8 L285.6,590.5 L279.3,590.6 L275.7,588.5 L275.0,586.9 L276.7,582.6 L275.4,577.4 L272.9,574.4 L267.6,578.8 L262.5,578.8 L256.5,582.7 L252.5,589.0 L248.8,591.3 L238.5,590.0 L236.0,592.8 L233.8,605.6 L232.6,608.1 L224.1,607.6 L215.8,612.6 L206.5,616.8 L201.7,613.9 L199.1,610.7 L194.0,611.3 L188.5,612.7 L184.8,611.9 L180.6,612.2 L172.5,614.7 L162.4,613.9 L158.1,611.8 L157.7,609.8 L158.0,604.8 L157.9,597.1 L156.3,590.2 L151.4,584.2 L146.3,580.0 L139.8,575.6 L135.0,573.7 L132.4,573.2 L131.8,570.7 L130.8,568.5 L128.6,566.6 L123.8,564.1 L119.7,563.5 L116.4,567.6Z",
    },
    "centroids": {"EE": [296.2, 127.2], "LV": [267.2, 303.0], "LT": [203.3, 494.2]},
}

# Theme filter: defence / geopolitics / Russia-Ukraine-China / war / drones.
# An article's HEADLINE must hit this to enter the report at all (see pipeline.filter_themes).
# 'deepfake|disinfo|misinform' were added so info-warfare / malign-influence stories
# (e.g. EU rules on AI deepfakes) qualify as hybrid-warfare candidates.
KEYWORDS = (
    r"\b(ukrain|russ|kremlin|putin|moscow|belarus|defen[cs]e|militar|nato|war\b|"
    r"warfare|drone|china|chinese|beijing|taiwan|weapon|missile|army|troop|soldier|"
    r"sanction|wagner|zelensky|invasion|espionage|spy|sabotag|cyber|airspace|"
    r"submarine|frigate|warship|warplane|hybrid|mobilis|mobiliz|conscript|"
    r"deepfake|disinfo|misinform|terror|radicali[sz]|extremis|"
    r"readiness|preparedness|resilience|civil\s+defen[cs]e|shelter|evacuat|"
    r"naval|navy|tank|migrant|threat|special\s+forces|false.?flag|\bhack|"
    r"intelligence\s+service|counter.?intelligence|\bcia\b|\bfsb\b|\bgru\b|\bkgb\b|"
    r"artillery|shelling|front[\s-]?line|refugee|border)\w*"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NordicsMonitor/1.0; research)"}

# Site boilerplate to drop from article bodies (newsletter / correction / social prompts).
BOILERPLATE = re.compile(
    r"(suggested correction|select text and press|send a suggested|to the editor|"
    r"follow err|follow lrt|follow lsm|sign up|newsletter|subscribe|cookie|"
    r"all rights reserved|advertisement|read more|related articles|share this|"
    r"download the|on facebook|on twitter|©)", re.I)

# Extra domain stopwords (site chrome + geography that would dominate every cluster).
EXTRA_STOP = {
    "text", "press", "select", "suggested", "correction", "editor", "send",
    "follow", "facebook", "twitter", "newsletter", "subscribe", "sign", "cookies",
    "err", "lrt", "lsm", "news", "said", "says", "according", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "year", "years",
    "wants", "make", "did",
}

# Extra stopwords for the act-3 frequency chart only: generic filler + the countries'
# own names/cities (self-references that would otherwise dominate and tell us nothing).
FREQ_STOP = {
    "time", "state", "states", "way", "week", "day", "days", "people", "month",
    "today", "year", "years", "country", "countries", "new", "also", "first", "two",
    "lithuania", "latvia", "estonia", "lithuanian", "latvian", "estonian",
    "baltic", "baltics", "vilnius", "tallinn", "riga",
    "added", "situation", "possible", "according", "public",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "does", "including", "used"
}

# Merge overlapping word forms for the frequency chart so variants count as one term:
# adjective/demonym forms and plurals collapse to a single canonical word
# (russia/russian -> russia, drones -> drone, ministers/ministries -> minister/ministry).
CANON = {
    "russian": "russia", "russians": "russia",
    "ukrainian": "ukraine", "ukrainians": "ukraine",
    "belarus": "belarus", "belarusian": "belarus", "belarusians": "belarus",
    "chinese": "china", "estonian": "estonia", "latvian": "latvia",
    "lithuanian": "lithuania", "defense": "defence", "militarily": "military",
    "ministries": "ministry", "ministers": "minister",
}


def canon(w):
    """Collapse a token to its canonical form (variant map first, then simple plural)."""
    if w in CANON:
        return CANON[w]
    if w.endswith("ies") and len(w) > 5:                 # embassies -> embassy
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        return w[:-1]                                    # drones -> drone, troops -> troop
    return w


# Thematic keyword buckets for the topic map (act 2). NOT clusters: each matched
# article is tagged with EVERY bucket its keywords hit (headline weighted x3);
# the strongest match becomes its colour/primary theme, and it also appears under
# any other bucket it hits (multi-label). Articles matching none fall back to
# "Other defence". "NATO & defence" is the broad catch-all, so on a scoring tie it
# yields to a more specific theme (see THEME_RANK / pipeline.analyse()).
THEMES = [
    ("Ukraine",            r"ukrain|zelensk|kyiv|kharkiv|donbas|donetsk|kherson|zaporizh|odesa"),
    ("Russia & Kremlin",   r"russ|kremlin|putin|moscow|wagner|lavrov|medvedev"),
    ("NATO & defence",     r"nato|defen[cs]e|militar|troop|soldier|brigade|deploy|exercise|weapon|"
                           r"missile|artillery|himars|atacms|warship|frigate|submarine|\barmy\b|"
                           r"rearm|conscript|mobilis|mobiliz|readiness|preparedness|resilience|"
                           r"civil\s+defen[cs]e|shelter|evacuat|naval|navy|\btank|special\s+forces"),
    ("Drones & airspace",  r"drone|airspace|uav|incursion|airport|warplane|aircraft|\bjet\b"),
    ("Border & migration", r"border|migrant|refugee|belarus|frontier|\bfence|smuggl|crossing|barbed"),
    ("Hybrid warfare",
                           r"sabotag|cyber|espionage|\bspy|hybrid|disinfo|deepfake|misinform|"
                           r"propaganda|interfer|meddl|cognitive|influence\s+oper|jamming|\bgps\b|"
                           r"undersea|\bcable|shadow fleet|provocation|coercion|"
                           r"terror|radicali[sz]|extremis|false.?flag|\bhack|counter.?intelligence|"
                           r"intelligence\s+service|\bcia\b|\bfsb\b|\bgru\b|\bkgb\b"),
    ("China & Indo-Pacific",
                           r"\bchina\b|chinese|beijing|taiwan|xi jinping|indo.?pacific|"
                           r"south china"),
]
THEME_OTHER = "Other defence"
THEME_PATS = [re.compile(p, re.I) for _, p in THEMES]

# Primary-theme tie-break, index-aligned to THEMES (higher wins on a tie).
# "NATO & defence" (id 2) is the broad sink, so it loses ties to any specific theme.
THEME_RANK = [2, 2, 1, 2, 2, 2, 2]

# Categorical palette (validated against dataviz/validate_palette.js), light + dark
# step per theme id. Identity is carried by the legend + hover tooltip; colour
# reinforces. A 7-hue scatter cannot clear the all-pairs colour-blind floor by
# colour alone (it never can past 3 hues), so theme names ride in the tooltip as
# the secondary channel. Last id = grey "Other". Passes every ADJACENT gate in
# both modes; worst adjacent CVD ΔE 9.1 light / 8.4 dark.
THEME_COLORS = [
    ("#2a78d6", "#3987e5"),  # 0 Ukraine                    blue
    ("#eb6834", "#d95926"),  # 1 Russia & Kremlin           orange
    ("#1baf7a", "#199e70"),  # 2 NATO & defence             aqua-green
    ("#eda100", "#c98500"),  # 3 Drones & airspace          yellow
    ("#e87ba4", "#d55181"),  # 4 Border & migration         magenta
    ("#4a3aa7", "#9085e9"),  # 5 Hybrid warfare             violet
    ("#b02a37", "#d0574f"),  # 6 China & Indo-Pacific       crimson
    ("#9a8f7c", "#8f836d"),  # 7 Other defence              warm grey
]
COLORS_LIGHT = [c[0] for c in THEME_COLORS]
COLORS_DARK = [c[1] for c in THEME_COLORS]
