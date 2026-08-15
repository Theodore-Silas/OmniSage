# SearchAgent Wiki Schema

## Purpose
This Wiki is the persistent knowledge layer for SearchAgent. Every search result is compiled into structured, interlinked Markdown pages that grow richer with each query.

## Directory Structure
- concepts/    — Cross-source concept summaries (e.g., "RAG Methods Comparison")
- entities/    — Entity pages (tools like "Milvus", papers like "RAPTOR 2024", people, orgs)
- queries/     — Archived search results (one page per unique query)
- synthesis/   — Cross-entity comparison tables, timelines, survey pages
- sources/     — Snapshots of original search results (immutable, one per source URL)
- index.md     — Global catalog of all Wiki pages
- log.md       — Append-only operation log
- error_book.yaml — Persistent error tracking

## Page Template
Every Wiki page MUST follow this structure:

```markdown
---
type: concept | entity | query | synthesis | source
created: ISO8601
updated: ISO8601
tags: [tag1, tag2]
status: draft | reviewed | established
confidence: 0.0 - 1.0
---

# Title

> One-sentence summary in blockquote

## Key Facts
- Fact 1 (cited as [1])
- Fact 2 (cited as [2])

## Analysis
Detailed analysis content.

## Related Pages
- [[path/to/related-page]] — relationship description

## Sources
- [[sources/timestamp-slug]] — source description

## Contradictions
(Explicitly mark conflicting information if any)
```

## Workflow Rules

### Search
1. Search Wiki first (check index.md + full-text)
2. If found: answer from Wiki with citation to source pages
3. If not found: external multi-source search → compile result as Wiki page → answer

### Persist (after every search)
1. Save fetched content as sources/YYYY-MM-DD-query-slug.md
2. Save full answer as queries/YYYY-MM-DD-query-slug.md
3. Extract entities → update or create entities/*.md pages
4. If cross-source synthesis exists → update or create concepts/*.md
5. Update index.md with new/updated pages
6. Append to log.md

### Quality Standards
1. Every claim must be traceable to a specific source page
2. Contradictions MUST be explicitly marked, never hidden
3. Pages with confidence < 0.5 must be marked status: draft
4. Use [[wikilink]] for all cross-page references
