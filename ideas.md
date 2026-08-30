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