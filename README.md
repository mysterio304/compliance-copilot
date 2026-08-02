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

# Corpus ingestion and embedding
uv run python scripts/ingest_corpus.py   # data/raw_pdfs/*.pdf -> data/processed/*.json
uv run python scripts/corpus_stats.py    # measured document/clause/chunk counts
uv run python scripts/load_corpus.py     # embeds with BGE-M3, writes to Postgres
```

Note: the Postgres container publishes on host port **5433**, not 5432 — this
avoids colliding with a locally-installed Postgres, if you have one. Only the
host-side port differs; nothing inside docker-compose's network is affected.

The API's Docker image currently includes the full ingestion/retrieval stack
(Docling, BGE-M3/FlagEmbedding, PyTorch) even though no endpoint uses them yet —
that's Phase 6 work. On Linux/ARM64 (e.g. Colima on Apple Silicon), PyTorch's
wheel pulls in NVIDIA CUDA packages unconditionally even though everything here
runs on CPU; there's no CPU-only wheel for that platform to redirect to, so the
build downloads several GB it doesn't use. Worth revisiting — either at Phase 6
when the API actually needs these deps, or by splitting them into an optional
dependency group sooner if the image weight becomes a problem before then.

## Retrieval (measured, Phase 3)

Hybrid dense+sparse search over BGE-M3 embeddings, both stored natively in
pgvector (`vector(1024)` for dense, `sparsevec(250002)` for sparse lexical
weights — 250002 is the actual BGE-M3/XLM-RoBERTa tokenizer vocab size),
fused with reciprocal rank fusion.

Cross-lingual retrieval — the hard requirement — was verified against the real
corpus, not synthetic data: a Chinese query for cadmium (镉) or hexavalent
chromium (六价铬), and an English query for "low voltage equipment", each
correctly retrieve the matching Russian clause at or near top-1. A fourth pair
(lead: 铅/lead → Свинец) is measurably weaker — real cross-lingual similarity
(cosine ~0.38–0.43) that doesn't yet outrank corpus noise for this short,
common word. See `tests/test_retrieval.py` for the specifics; this is a
tracked limitation, not a hidden one.

The relevance floor (checked against dense cosine similarity) is what makes
"insufficient evidence" possible: a nonsense query returns no hits rather than
a weak match.

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
- [x] Phase 3 — Hybrid retrieval (BGE-M3 dense+sparse, RRF fusion, relevance floor)
- [ ] Phase 4 — Evaluation harness
- [ ] Phase 5 — Agent loop
- [ ] Phase 6 — API and CI
