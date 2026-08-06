from dataclasses import dataclass

from pgvector import SparseVector
from sqlalchemy import Row, select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.db.models import BGE_M3_VOCAB_SIZE, ChunkRow, ClauseRow, DocumentRow
from compliance_copilot.retrieval.embedder import embed_query

# Standard constant from the original RRF paper (Cormack et al.) — dampens the
# influence of rank 1 vs rank 2 so fusion isn't dominated by whichever list
# happens to rank one chunk first. Not exposed as a tunable; fusion_weight and
# relevance_floor are the two knobs Phase 5 tunes against the eval harness.
_RRF_K = 60

_RankedHit = tuple[str, str, str, str, float]  # chunk_id, clause_id, document_id, text, score


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    clause_id: str
    document_id: str
    text: str
    dense_score: float
    sparse_score: float
    fused_score: float


async def _search_dense(
    session: AsyncSession, query_dense: list[float], top_k: int, regulation_code: str | None
) -> list[_RankedHit]:
    distance = ChunkRow.dense_embedding.cosine_distance(query_dense)
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.clause_id,
            ClauseRow.document_id,
            ChunkRow.text,
            distance.label("distance"),
        )
        .join(ClauseRow, ChunkRow.clause_id == ClauseRow.id)
        .where(ChunkRow.dense_embedding.is_not(None))
    )
    if regulation_code is not None:
        stmt = stmt.join(DocumentRow, ClauseRow.document_id == DocumentRow.id).where(
            DocumentRow.regulation_code == regulation_code
        )
    stmt = stmt.order_by(distance).limit(top_k)
    result = await session.execute(stmt)
    return [(row.id, row.clause_id, row.document_id, row.text, 1 - row.distance) for row in result]


async def _dense_scores_for_ids(
    session: AsyncSession, query_dense: list[float], chunk_ids: list[str]
) -> dict[str, float]:
    """Real cosine similarity for specific chunks, regardless of whether they
    placed in dense search's own top_k. Needed because a chunk sparse search
    surfaces can have a real, floor-clearing dense similarity without ranking
    highly enough in dense's independent top_k to appear in dense_ranked —
    defaulting such a chunk's dense_score to 0.0 (as if it had none) silently
    discards that real value and floor-filters it incorrectly."""
    if not chunk_ids:
        return {}
    distance = ChunkRow.dense_embedding.cosine_distance(query_dense)
    stmt = select(ChunkRow.id, distance.label("distance")).where(
        ChunkRow.id.in_(chunk_ids), ChunkRow.dense_embedding.is_not(None)
    )
    result = await session.execute(stmt)
    return {row.id: 1 - row.distance for row in result}


# Every EAEU technical regulation defines its scope in two halves: what it
# covers ("Настоящий технический регламент ... распространяется на ...", or
# "Действие настоящего технического регламента распространяется на ..." in the
# ТР ЕАЭС 037/2016 phrasing) and what it carves out ("... не распространяется
# на ...", typically one clause enumerating categories а)–к). Matched with \s+
# rather than literal spaces because the corpus text comes out of PDF extraction
# with runs of whitespace inside sentences.
#
# The two are near-identical lexically — the exclusion half differs by one "не"
# in a clause that can run past 1600 characters — so embedding similarity cannot
# tell them apart, and ranking surfaces one about as often as the other. That is
# precisely why both have to be supplied structurally rather than hoped for.
_SCOPE_INCLUSION_SIGNATURE = r"распространя\w*\s+на"
_SCOPE_EXCLUSION_SIGNATURE = r"не\s+распространя\w*\s+на"


async def _earliest_matching_chunk(
    session: AsyncSession, regulation_code: str, signature: str, *, excluding: str | None = None
) -> Row[tuple[str, str, str, str]] | None:
    stmt = (
        select(ChunkRow.id, ChunkRow.clause_id, ClauseRow.document_id, ChunkRow.text)
        .join(ClauseRow, ChunkRow.clause_id == ClauseRow.id)
        .join(DocumentRow, ClauseRow.document_id == DocumentRow.id)
        .where(
            DocumentRow.regulation_code == regulation_code,
            ClauseRow.text.op("~")(signature),
        )
    )
    if excluding is not None:
        stmt = stmt.where(~ClauseRow.text.op("~")(excluding))
    stmt = stmt.order_by(ClauseRow.char_start, ChunkRow.id).limit(1)
    return (await session.execute(stmt)).first()


async def _scope_clause_hits(
    session: AsyncSession,
    query_dense: list[float],
    regulation_code: str,
    exclude_chunk_ids: set[str],
) -> list[RetrievalHit]:
    """Both halves of a regulation's scope statement, scored against the query
    like any other hit.

    Applicability is *defined* by these two clauses, so an assessor asked "does
    this regulation apply" needs both in evidence every time. Supplying only the
    positive half is not enough and was itself a defect: a product can sit
    squarely inside the stated scope and still be carved out by name, which is
    what happens with electric toys, used goods, medical devices and measuring
    instruments under ТР ЕАЭС 037/2016. Leaving either to chance in the top_k
    produces confidently wrong verdicts in both directions — a missed inclusion
    reads as a false negative, a missed exclusion as a false positive.

    Earliest-in-document-order resolves the regulations whose later articles
    reuse the same phrasing.
    """
    inclusion = await _earliest_matching_chunk(
        session,
        regulation_code,
        _SCOPE_INCLUSION_SIGNATURE,
        # A clause carrying the negated phrasing is the exclusion half, even
        # though it also matches the inclusion pattern as a substring.
        excluding=_SCOPE_EXCLUSION_SIGNATURE,
    )
    exclusion = await _earliest_matching_chunk(session, regulation_code, _SCOPE_EXCLUSION_SIGNATURE)

    rows = [row for row in (inclusion, exclusion) if row is not None]
    wanted = [row for row in rows if row.id not in exclude_chunk_ids]
    if not wanted:
        return []

    scores = await _dense_scores_for_ids(session, query_dense, [row.id for row in wanted])
    return [
        RetrievalHit(
            chunk_id=row.id,
            clause_id=row.clause_id,
            document_id=row.document_id,
            text=row.text,
            dense_score=scores.get(row.id, 0.0),
            sparse_score=0.0,
            # Ranked last deliberately: included for coverage, not because it
            # out-competed anything on relevance.
            fused_score=0.0,
        )
        for row in wanted
    ]


async def _search_sparse(
    session: AsyncSession, query_sparse: SparseVector, top_k: int, regulation_code: str | None
) -> list[_RankedHit]:
    neg_inner_product = ChunkRow.sparse_embedding.max_inner_product(query_sparse)
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.clause_id,
            ClauseRow.document_id,
            ChunkRow.text,
            neg_inner_product.label("neg_ip"),
        )
        .join(ClauseRow, ChunkRow.clause_id == ClauseRow.id)
        .where(ChunkRow.sparse_embedding.is_not(None))
    )
    if regulation_code is not None:
        stmt = stmt.join(DocumentRow, ClauseRow.document_id == DocumentRow.id).where(
            DocumentRow.regulation_code == regulation_code
        )
    stmt = stmt.order_by(neg_inner_product).limit(top_k)
    result = await session.execute(stmt)
    return [(row.id, row.clause_id, row.document_id, row.text, -row.neg_ip) for row in result]


async def hybrid_search(
    session: AsyncSession,
    query: str,
    embedding_model: str,
    top_k: int,
    fusion_weight: float,
    relevance_floor: float,
    regulation_code: str | None = None,
    ensure_scope_clause: bool = False,
) -> list[RetrievalHit]:
    """Hybrid dense+sparse retrieval over BGE-M3 embeddings, fused with reciprocal
    rank fusion (fusion_weight favors dense at 1.0, sparse at 0.0).

    The relevance floor is checked against each hit's real dense cosine
    similarity — the one bounded, well-understood signal here (sparse inner
    products and RRF scores don't have a fixed scale to threshold against).
    A chunk sparse search surfaces but dense's own top_k doesn't include gets
    a supplemental targeted lookup for its actual dense score rather than
    being assumed to have none — otherwise the floor would filter it out
    based on a fabricated 0.0 regardless of its real similarity, which also
    made fusion_weight practically inert (a sparse-favored chunk could never
    survive the floor to be re-ranked by fusion in the first place). If
    nothing clears the floor, this returns an empty list — the caller reports
    insufficient evidence rather than surfacing a weak match.

    regulation_code optionally scopes the search to one document (e.g. "ТР ТС
    004/2011") — used by the agent's per-regulation-family retrieval step so a
    family's evidence isn't diluted by generic boilerplate from other documents.

    ensure_scope_clause appends both halves of that regulation's scope statement
    — what it covers and what it excludes — if the ranked results missed them,
    past both the relevance floor and the top_k cut. It is opt-in rather than
    default because it answers an agent-level question ("what defines whether
    this regulation applies?") rather than a retrieval one, and the retrieval
    baseline is measured without it.
    """
    query_embedding = embed_query(query, embedding_model)
    query_sparse = SparseVector(query_embedding.sparse, BGE_M3_VOCAB_SIZE)

    dense_ranked = await _search_dense(session, query_embedding.dense, top_k, regulation_code)
    sparse_ranked = await _search_sparse(session, query_sparse, top_k, regulation_code)

    fused_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    meta: dict[str, tuple[str, str, str]] = {}  # chunk_id -> (clause_id, document_id, text)

    for rank, (chunk_id, clause_id, document_id, text, score) in enumerate(dense_ranked, start=1):
        rrf_term = fusion_weight * (1 / (_RRF_K + rank))
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_term
        dense_scores[chunk_id] = score
        meta[chunk_id] = (clause_id, document_id, text)

    for rank, (chunk_id, clause_id, document_id, text, score) in enumerate(sparse_ranked, start=1):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1 - fusion_weight) * (
            1 / (_RRF_K + rank)
        )
        sparse_scores[chunk_id] = score
        meta[chunk_id] = (clause_id, document_id, text)

    sparse_only_ids = [chunk_id for chunk_id in fused_scores if chunk_id not in dense_scores]
    dense_scores.update(
        await _dense_scores_for_ids(session, query_embedding.dense, sparse_only_ids)
    )

    hits = [
        RetrievalHit(
            chunk_id=chunk_id,
            clause_id=meta[chunk_id][0],
            document_id=meta[chunk_id][1],
            text=meta[chunk_id][2],
            dense_score=dense_scores.get(chunk_id, 0.0),
            sparse_score=sparse_scores.get(chunk_id, 0.0),
            fused_score=fused_score,
        )
        for chunk_id, fused_score in fused_scores.items()
        if dense_scores.get(chunk_id, 0.0) >= relevance_floor
    ]
    hits.sort(key=lambda hit: hit.fused_score, reverse=True)
    top_hits = hits[:top_k]

    if ensure_scope_clause and regulation_code is not None:
        top_hits.extend(
            await _scope_clause_hits(
                session,
                query_embedding.dense,
                regulation_code,
                {hit.chunk_id for hit in top_hits},
            )
        )

    return top_hits
