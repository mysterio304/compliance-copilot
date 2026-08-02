from dataclasses import dataclass

from pgvector import SparseVector
from sqlalchemy import select
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
) -> list[RetrievalHit]:
    """Hybrid dense+sparse retrieval over BGE-M3 embeddings, fused with reciprocal
    rank fusion (fusion_weight favors dense at 1.0, sparse at 0.0).

    The relevance floor is checked against each hit's dense cosine similarity —
    the one bounded, well-understood signal here (sparse inner products and RRF
    scores don't have a fixed scale to threshold against). A hit that never
    appeared in the dense top_k has no dense score and fails the floor by
    construction. If nothing clears the floor, this returns an empty list — the
    caller reports insufficient evidence rather than surfacing a weak match.

    regulation_code optionally scopes the search to one document (e.g. "ТР ТС
    004/2011") — used by the agent's per-regulation-family retrieval step so a
    family's evidence isn't diluted by generic boilerplate from other documents.
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
    return hits[:top_k]
