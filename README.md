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
uv run python eval/run_eval.py           # score against the committed baseline
uv run python scripts/tune_retrieval.py  # sweep fusion_weight/top_k/relevance_floor
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

## Evaluation harness (measured, Phase 4 — pre-agent baseline)

10 hand-labelled gold cases in `eval/cases/`, scored against 5 metrics in
`eval/metrics.py`. Built and run *before* the agent exists (Phase 5) — every
case here queries `hybrid_search` directly with the whole `input_spec` text,
with no query planning or applicability reasoning in front of it yet. This is
deliberately the weakest the system will ever measure; it exists so Phase 5's
agent has something concrete to improve against, not to look good on its own.

Retrieval-only baseline (`top_k=10`, before Phase 5 tuning) vs. after
`scripts/tune_retrieval.py` (`top_k=5`, now the default — see Retrieval
Tuning below):

| Metric | Before (top_k=10) | After (top_k=5) |
|---|---:|---:|
| retrieval_recall_at_k | 0.125 | 0.125 |
| citation_precision | 0.560 | 0.580 |
| regulation_set_f1 | 0.726 | 0.757 |
| false_positive_rate | 0.524 | 0.190 |
| abstention_accuracy | 0.000 | 0.000 |

Two results were investigated rather than taken at face value:

- **abstention_accuracy = 0.0** (both configs — top_k doesn't touch this) —
  the relevance floor alone can't tell "vague query" from "topically similar
  to something." A spec that just says "no technical details provided" still
  scores 0.56–0.59 cosine similarity against clauses that are *about*
  documentation requirements — related vocabulary, wrong reason. This is
  exactly why Phase 5's agent has an explicit `extract_product_attributes`
  step: `ProductAttributes.is_empty` catches this deterministically instead of
  relying on the retrieval threshold. Verified live: the agent correctly
  abstains (empty findings, confidence 0.0, explicit unresolved_questions) on
  the same vague-spec case this metric is measuring.
- **false_positive_rate 0.524 → 0.190** — driven by boilerplate preamble
  clauses ("Настоящий технический регламент... разработан в соответствии с
  Соглашением...") nearly identical across all five regulations, weakly
  matching almost any query. `top_k=5` (found by sweeping, not guessing —
  below) leaves less room for that noise to make the merged result set at
  all. The agent's per-regulation-scoped retrieval (`search_regulations(...,
  regulation_code=...)`) attacks the same problem from a different angle,
  scoping evidence to one document instead of the whole corpus.

`eval/run_eval.py` writes `eval/results/latest.json` and compares it against
the committed `eval/results/baseline.json` (now the tuned config's numbers),
exiting non-zero on a regression beyond tolerance (default 5 points).
`--update-baseline` overwrites the baseline deliberately; nothing else should.

## Retrieval tuning (measured, Phase 5)

`scripts/tune_retrieval.py` sweeps `fusion_weight`, `top_k`, and
`relevance_floor` against the same 10 gold cases and scoring as
`eval/run_eval.py`. Full sweep table and reasoning in that script's commit
message. Two findings worth knowing before touching these values:

- **`top_k=5` beats `top_k=10`** on every metric except recall (unchanged) —
  now the default.
- **`fusion_weight` is currently inert** at the shipped `relevance_floor=0.35`:
  every value from 0.0 to 1.0 produces byte-identical metrics. The floor is
  checked against `dense_score` specifically (`retrieval/hybrid.py`), so a
  chunk sparse search favors but dense search doesn't independently rank has
  `dense_score=0` and gets floor-filtered regardless of fusion weight.
  Confirmed by re-sweeping at `floor=0.0`, where fusion_weight changed every
  metric. Left at the neutral 0.5 default — no measured downside at the
  shipped floor, and if the floor's dense-only anchoring is ever revisited,
  `1.0` (dense-heavy) looked best on precision/F1/FPR in that sweep, at some
  cost to recall vs. `0.5`.

## Agent (Phase 5)

LangGraph pipeline in `agent/`: extract product attributes → plan which
regulation families to investigate → retrieve per family → assess
applicability → assemble report → validate. Every LLM call
(`agent/llm.py::call_structured`) is Pydantic-validated against DeepSeek's
JSON mode, retried once with the validation error fed back on failure, and
raises rather than returning anything unvalidated on a second failure. The
validate node checks every citation's `clause_id` actually exists in the
database; on failure it routes back to reassess once, then finalizes a
partial report with the dropped findings stated explicitly.

Verified live end to end (not mocked) in `tests/test_agent.py` and
`tests/test_agent_tools.py`: a real spec produces findings with valid
citations; a vague spec abstains; a night-light spec with child-adjacent
marketing correctly excludes toy safety at the planning stage rather than
pattern-matching on "children." One real reasoning gap found and left
documented rather than silently patched: the agent reasoned about a lamp's
5V DC *output* rather than its 100-240V AC *adapter* (the actual low-voltage
scope) on one live run — a prompt-quality issue, not a code bug, and out of
scope for this phase's ask (retrieval tuning, not prompt engineering).

`@observe(name="compliance_agent_run")` roots a Langfuse trace at
`run_agent`; every nested `call_structured` generation attaches to it
automatically. Not yet verified against a real Langfuse project — no
`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` configured in this environment; the
`@observe()` decorator degrades gracefully (logs, doesn't raise) when unset,
confirmed directly.

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
- [x] Phase 4 — Evaluation harness (10 gold cases, 5 metrics, pre-agent baseline)
- [x] Phase 5 — Agent loop (LangGraph pipeline, structured outputs, validation retry, tuned retrieval)
- [ ] Phase 6 — API and CI
