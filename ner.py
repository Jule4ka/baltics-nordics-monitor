#!/usr/bin/env python
"""Named-entity recognition over the defence corpus (spaCy).

Pulls the people (PERSON), organizations (ORG) and places (GPE/LOC) out of each article's
headline+body, aggregates them across the corpus, and returns the top entities per group
with the article indices that mention them (positional, matching build_payload's points
order — so the front-end can list an entity's articles).

Graceful: if spaCy or its model isn't installed, returns {} and the page simply omits the
section — the same degrade-don't-crash pattern the embeddings extras use.
"""
import re

# spaCy entity label -> display group
_GROUP = {"PERSON": "people", "ORG": "orgs", "GPE": "places", "LOC": "places"}

# entity surfaces that are noise as "key players" (feed-generic or ambiguous)
_ENT_STOP = {
    "reuters", "ap", "afp", "bns", "err", "lrt", "lsm", "delfi",
    "the baltic times", "baltic news service",
}

# adjectival demonyms spaCy mis-tags as PERSON/ORG/GPE ("Lithuanian said…", "Ukrainian forces")
_DEMONYMS = {
    "lithuanian", "lithuanians", "latvian", "latvians", "estonian", "estonians",
    "russian", "russians", "ukrainian", "ukrainians", "belarusian", "belarusians",
    "polish", "european", "europeans", "american", "americans", "german", "germans",
    "french", "british", "chinese", "finnish", "swedish", "norwegian", "nordic",
    "western", "eastern", "soviet",
}

# cities/regions spaCy often mislabels as PERSON/ORG — force them into places
_PLACES_FORCE = {
    "vilnius", "kaunas", "riga", "tallinn", "kyiv", "kiev", "moscow", "minsk",
    "warsaw", "brussels", "kaliningrad", "narva", "klaipeda", "st petersburg",
}

_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_MOJIBAKE = re.compile(r"[âÂÃÄÅ€œ]")               # leftover bad-encoding markers


def _clean(s):
    """Normalise an entity surface: collapse spaces, drop possessive + edge punctuation."""
    s = " ".join(s.split())
    s = re.sub(r"[’']s$", "", s)                  # trailing 's / ’s
    s = _EDGE.sub("", s)
    if len(s) < 2 or s.isdigit():
        return ""
    return s


def extract_entities(texts, top=15, min_articles=2, body_chars=2000):
    """texts: iterable of article strings (headline + body). Returns
    {"people": [...], "orgs": [...], "places": [...]}, each a list of
    {"name", "n" (articles mentioning it), "arts" (article indices)}, or {} if spaCy is
    unavailable."""
    texts = list(texts)
    try:
        import spacy
    except Exception as e:
        print(f"[ner] spaCy not installed ({type(e).__name__}); skipping entities")
        return {}
    try:                                               # parser/lemmatizer not needed for NER -> faster
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler"])
    except Exception as e:
        print(f"[ner] model en_core_web_sm unavailable ({type(e).__name__}); skipping entities")
        return {}

    groups = {"people": {}, "orgs": {}, "places": {}}  # group -> canon -> {forms:{surf:ct}, arts:set}
    for i, doc in enumerate(nlp.pipe((t[:body_chars] for t in texts), batch_size=64)):
        seen = set()                                   # count each entity once per article
        for ent in doc.ents:
            g = _GROUP.get(ent.label_)
            if not g:
                continue
            surf = _clean(ent.text)
            canon = surf.lower()
            if not surf or canon in _ENT_STOP or canon in _DEMONYMS or _MOJIBAKE.search(surf):
                continue
            if canon in _PLACES_FORCE:                 # correct city mislabels -> places
                g = "places"
            if (g, canon) in seen:
                continue
            seen.add((g, canon))
            rec = groups[g].setdefault(canon, {"forms": {}, "arts": set()})
            rec["forms"][surf] = rec["forms"].get(surf, 0) + 1
            rec["arts"].add(i)

    # drop one-offs
    for g in groups:
        groups[g] = {c: r for c, r in groups[g].items() if len(r["arts"]) >= min_articles}

    # cross-group dedup: keep a name only in the group where it's most mentioned, so a city
    # mis-tagged as a person/org ("Vilnius", "Kaunas") collapses into places.
    best = {}
    for g, d in groups.items():
        for c, r in d.items():
            if c not in best or len(r["arts"]) > best[c][1]:
                best[c] = (g, len(r["arts"]))
    for g, d in groups.items():
        for c in list(d):
            if best[c][0] != g:
                del d[c]

    out = {}
    for g, d in groups.items():
        items = [{"canon": c, "name": max(r["forms"], key=r["forms"].get),
                  "arts": set(r["arts"])} for c, r in d.items()]
        # merge a lone surname into a full name ending with it ("Putin" -> "Vladimir Putin")
        fulls = [it for it in items if " " in it["canon"]]
        for it in list(items):
            if " " in it["canon"]:
                continue
            host = next((f for f in fulls if f["canon"].split()[-1] == it["canon"]), None)
            if host:
                host["arts"] |= it["arts"]
                items.remove(it)
        for it in items:
            it["n"] = len(it["arts"])
        items.sort(key=lambda x: -x["n"])
        out[g] = [{"name": it["name"], "n": it["n"], "arts": sorted(it["arts"])}
                  for it in items[:top]]
    total = sum(len(v) for v in out.values())
    print(f"[ner] {total} entities across {len(texts)} articles (people/orgs/places)")
    return out
