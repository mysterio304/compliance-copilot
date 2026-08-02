# Compliance Copilot

Multilingual agentic RAG system that maps Chinese supplier specifications to
Eurasian Economic Union (EAEU / ЕАЭС) certification requirements and returns
clause-level citations.

An importer receiving a Chinese supplier's spec sheet has to work out which
technical regulations (технические регламенты) apply, which clauses trigger,
and what conformity assessment route is required — currently slow manual work
done by paid consultants. This system takes a supplier spec as input and
returns the applicable regulations, the specific clauses, the assessment
route, and a citation for every claim.

v1 scope covers electronics: ТР ТС 004/2011, ТР ТС 020/2011, ТР ЕАЭС 037/2016,
ТР ТС 010/2011, ТР ТС 008/2011. The regulatory corpus is Russian; supplier
specs and queries may be Chinese, English, or Russian — retrieval is
cross-lingual.

Full architecture, metrics, and limitations are documented here once the
retrieval and agent layers exist (Phase 3 onward). Corpus provenance —
including a known deviation from the original structural assumption — is in
[`data/sources.md`](data/sources.md).

## Running locally

Requires `uv`, Docker, and Docker Compose.

```bash
cp .env.example .env   # fill in DEEPSEEK_API_KEY / LANGFUSE_* when Phase 5 needs them
uv sync
uv run pytest
uv run ruff check .
uv run mypy --strict src

docker compose up -d
curl http://localhost:8000/health
docker compose down

# Corpus ingestion
uv run python scripts/ingest_corpus.py   # data/raw_pdfs/*.pdf -> data/processed/*.json
uv run python scripts/corpus_stats.py    # measured document/clause/chunk counts
```

## Corpus (measured, Phase 2)

| Regulation | Clauses | Chunks |
|---|---:|---:|
| ТР ТС 004/2011 | 87 | 87 |
| ТР ТС 020/2011 | 101 | 102 |
| ТР ЕАЭС 037/2016 | 100 | 100 |
| ТР ТС 010/2011 | 268 | 274 |
| ТР ТС 008/2011 | 118 | 122 |
| **Total** | **674** | **685** |

All 5 documents, all Russian. Re-run `scripts/corpus_stats.py` after any
ingestion change — this table should match its output, not be hand-edited.

## Status

- [x] Phase 1 — Foundation (schemas, config, containerized API skeleton)
- [x] Phase 2 — Ingestion (Docling parsing, clause extraction, chunking)
- [ ] Phase 3 — Hybrid retrieval
- [ ] Phase 4 — Evaluation harness
- [ ] Phase 5 — Agent loop
- [ ] Phase 6 — API and CI
