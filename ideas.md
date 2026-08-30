- use a map somehow?
- do I need 2 ways to extract topics?
- use like Bumba (intelligence tools)
- think which essay I can write on these stories (maybe somethign from anomalies??)


Ideas for the entities — here's what I'd do, roughly best-first:

1. Entity mention trends over time — the same river/sparkline treatment for people & orgs: when did "Putin", "NATO", "CIA" surge? Reuses the streamgraph infra you just got. High synergy.
2. Entity tone — average escalation-tone of the articles each entity appears in → "who shows up in the hottest coverage" (a red↔blue chip next to each name). Reuses the sentiment score already computed.
3. Co-occurrence / "seen alongside" — for a clicked entity, show which other entities share its articles (Putin ↔ Zelensky ↔ NATO). Reveals the relationship web; could be a mini network graph.
4. Entity → cross-filter — click an entity to highlight its articles on the semantic map and filter the other panels. Turns entities into a lens over everything.
5. Entity × country — which players dominate Estonian vs Latvian vs Lithuanian coverage (small heatmap).
6. Places → the map — plot the GPE places as intensity on the existing Baltic map.

My pick: #1 (entity trends) for immediate payoff, or #2 (entity tone) as a cheap, striking addition.

Which appeals? And separately — the interactive trends upgrade is uncommitted; want me to commit it now (and still pending: push everything, and the headline mojibake fix)?