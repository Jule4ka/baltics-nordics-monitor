- use a map somehow?
- do I need 2 ways to extract topics?
- use like Bumba (intelligence tools)
- think which essay I can write on these stories (maybe somethign from anomalies??)

## Entity feature ideas (brainstorm)
done:
- [x] 1. entity mention trends over time (per-entity weekly sparkline)
- [x] 2. entity coverage tone (mean escalation-tone chip)
- [x] 3. co-occurrence — "seen alongside" (entities sharing articles)
- [x] 5. entity x country split (EE/LV/LT mini-bar)
todo / maybe:
- [ ] 4. click an entity to cross-filter the map + other panels
- [ ] 6. plot the GPE places on the Baltic map (intensity)

## Parked ideas
- LLM-written topic labels + entity alias canonicalisation (needs ANTHROPIC_API_KEY)
- tune topic-tracking match threshold if topics drift too much over time
- trend/label options tried: c-TF-IDF vs KeyBERT vs blend (blend won); representative-headline labels (not used)

Full descriptions of the entity ideas (for reference):
- 1 trends: river/sparkline per entity — when did Putin/NATO/CIA surge?
- 2 tone: avg escalation-tone of an entity's articles — who's in the hottest coverage (red↔blue chip)
- 3 co-occurrence: for a clicked entity, other entities sharing its articles (Putin↔Zelensky↔NATO); could grow into a network graph
- 4 cross-filter: click an entity to highlight its articles on the map + filter other panels
- 5 country: which players dominate EE vs LV vs LT coverage
- 6 places→map: plot GPE places as intensity on the Baltic map

## Analysis & intelligence roadmap
done:
- [x] entity co-occurrence NETWORK (force-directed web + adjacency matrix) — grew out of co-occurrence idea 3; merged the entity list + network into one "The players" section, per-entity dossier (mentions, tone, weekly timeline, EE/LV/LT split, seen-alongside, articles)
- [x] cross-country framing — "same story, three newsrooms": match ONE event across EE/LV/LT by embedding similarity (cosine ≥ 0.85), show the three framings side by side with each country's escalation tone + the tone spread
- [x] embedding topic granularity retuned to ~10 well-separated topics (HDBSCAN min_cluster_size n//26, fewest "Other")
- [x] removed vestigial TF-IDF / SVD / t-SNE from pipeline.analyse — topics now come ONLY from embeddings (answers "do I need 2 ways to extract topics?" — no)
- [x] dedicated EE/LV/LT country palette (teal / plum / rose), distinct from the entity-kind red/blue/green

todo / maybe:
- [ ] trending topics — momentum score per discovered topic (recent weeks vs baseline), ▲/▼ badge + sparkline, a "Trending now" strip; cross-build trend via the persisted topic store
- [ ] bigger / different embedding model — try BAAI/bge-m3 (multilingual, stronger; useful if we ever feed original ET/LV/LT text), intfloat/e5-large-v2, Alibaba gte-large-en-v1.5, or an API model (Voyage / OpenAI); compare topic quality. `--embed-model` already supports the swap (cache is keyed by model)
- [ ] LLM analysis (Claude, needs ANTHROPIC_API_KEY) — the "Bumba / intelligence tools" idea:
    - fluent topic names + per-topic 2-sentence summaries (hook already exists: _llm_labels)
    - event / timeline extraction across the corpus
    - stance / framing classification per country (quantify who spins what, beyond the tone axis)
- [ ] relation extraction between named entities (who-did-what-to-whom) to LABEL the network edges, not just "shared an article"
- [ ] essay material — mine anomalies + the biggest cross-country tone gaps for a story angle