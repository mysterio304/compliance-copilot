from dataclasses import dataclass
from typing import Literal

from pgvector import SparseVector
from sqlalchemy import Row, Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.db.models import BGE_M3_VOCAB_SIZE, ChunkRow, ClauseRow, DocumentRow
from compliance_copilot.retrieval.embedder import embed_query

# Standard constant from the original RRF paper (Cormack et al.) — dampens the
# influence of rank 1 vs rank 2 so fusion isn't dominated by whichever list
# happens to rank one chunk first. Not exposed as a tunable; fusion_weight and
# relevance_floor are the two knobs Phase 5 tunes against the eval harness.
_RRF_K = 60

# chunk_id, clause_id, document_id, clause_path, text, score
_RankedHit = tuple[str, str, str, str, str, float]


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    clause_id: str
    document_id: str
    clause_path: str
    text: str
    dense_score: float
    sparse_score: float
    fused_score: float
    # Set only on structurally-injected scope clauses (ensure_scope_clause):
    # which half of the scope statement this is. Downstream prompts render it
    # as an explicit polarity marker, because the clause text alone cannot
    # carry it — an item from an exclusions annex ("Провода, шнуры, кабели…")
    # and an item from the covered-products annex ("оборудование световое…")
    # are both bare equipment lists, and a live trace caught the assessor
    # reading a covered-products item as an exclusion and carving a desk lamp
    # out of EMC with it.
    scope_half: Literal["inclusion", "exclusion"] | None = None


async def _search_dense(
    session: AsyncSession, query_dense: list[float], top_k: int, regulation_code: str | None
) -> list[_RankedHit]:
    distance = ChunkRow.dense_embedding.cosine_distance(query_dense)
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.clause_id,
            ClauseRow.document_id,
            ClauseRow.clause_path,
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
    return [
        (row.id, row.clause_id, row.document_id, row.clause_path, row.text, 1 - row.distance)
        for row in result
    ]


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
# covers ("Настоящий технический регламент ... распространяется на ...") and
# what it carves out. Matched with \s+ rather than literal spaces because the
# corpus text comes out of PDF extraction with runs of whitespace inside
# sentences.
#
# The exclusion signature is deliberately looser than the inclusion one,
# because the carve-outs live in two grammatical shapes: article bodies say
# "не распространяется на ...", while annex headers say "..., на которые не
# распространяется настоящий технический регламент". Both matter — ТР ТС
# 020/2011 keeps its EMC-passive equipment list (cables, switches without
# active circuits, high-voltage gear without electronics) in Приложение 1, and
# ТР ТС 008/2011 keeps its not-actually-toys list there too. An annex header
# match alone is useless to an assessor, though: the header names the list and
# the items are their own clauses, so headers get expanded to their children.
#
# Inclusion and exclusion are near-identical lexically — one "не" apart — so
# embedding similarity cannot tell them apart, and ranking surfaces one about
# as often as the other. That is precisely why both are supplied structurally
# rather than hoped for.
_SCOPE_INCLUSION_SIGNATURE = r"распространя\w*\s+на"
_SCOPE_EXCLUSION_SIGNATURE = r"не\s+распространя"

_ScopeRow = Row[tuple[str, str, str, str, str]]


def _scope_row_select() -> Select[tuple[str, str, str, str, str]]:
    return (
        select(
            ChunkRow.id,
            ChunkRow.clause_id,
            ClauseRow.document_id,
            ChunkRow.text,
            ClauseRow.clause_path,
        )
        .join(ClauseRow, ChunkRow.clause_id == ClauseRow.id)
        .join(DocumentRow, ClauseRow.document_id == DocumentRow.id)
        .order_by(ClauseRow.char_start, ChunkRow.id)
    )


async def _matching_chunks(
    session: AsyncSession, regulation_code: str, signature: str, *, excluding: str | None = None
) -> list[_ScopeRow]:
    stmt = _scope_row_select().where(
        DocumentRow.regulation_code == regulation_code,
        ClauseRow.text.op("~")(signature),
    )
    if excluding is not None:
        stmt = stmt.where(~ClauseRow.text.op("~")(excluding))
    return list((await session.execute(stmt)).all())


async def _annex_child_chunks(
    session: AsyncSession, regulation_code: str, annex_path: str
) -> list[_ScopeRow]:
    stmt = _scope_row_select().where(
        DocumentRow.regulation_code == regulation_code,
        ClauseRow.clause_path.like(f"{annex_path},%"),
    )
    return list((await session.execute(stmt)).all())


def _is_annex_header(clause_path: str) -> bool:
    # Annex roots render as bare "Приложение N"; their items carry a comma
    # ("Приложение 1, пункт 3"). An exclusion match on a bare annex path is a
    # list header whose actual content lives in the children.
    return clause_path.startswith("Приложение") and "," not in clause_path


async def _scope_clause_hits(
    session: AsyncSession,
    query_dense: list[float],
    regulation_code: str,
    exclude_chunk_ids: set[str],
) -> list[RetrievalHit]:
    """Both halves of a regulation's scope statement, scored against the query
    like any other hit.

    Applicability is *defined* by these clauses, so an assessor asked "does
    this regulation apply" needs them in evidence every time. Supplying only the
    positive half was itself a defect: a product can sit squarely inside the
    stated scope and still be carved out by name. Supplying only the *earliest*
    exclusion clause was the same defect one level down — three audited false
    positives (an extension cord, a solar garden light, a 10 kV switchgear
    panel) were all excluded by ТР ТС 020/2011's Приложение 1, which the
    Статья 1 injection never covered. Exclusions are taken wherever the
    signature matches, with annex headers expanded to their item clauses.

    A missed inclusion reads as a false negative, a missed exclusion as a false
    positive — both directions of confidently wrong.
    """
    inclusions = await _matching_chunks(
        session,
        regulation_code,
        _SCOPE_INCLUSION_SIGNATURE,
        # A clause carrying the negated phrasing is the exclusion half, even
        # though it also matches the inclusion pattern as a substring.
        excluding=_SCOPE_EXCLUSION_SIGNATURE,
    )
    exclusions = await _matching_chunks(session, regulation_code, _SCOPE_EXCLUSION_SIGNATURE)

    halves: list[tuple[_ScopeRow, Literal["inclusion", "exclusion"]]] = [
        (row, "inclusion")
        for row in inclusions[:1]  # earliest scope statement
    ]
    for row in exclusions:
        halves.append((row, "exclusion"))
        if _is_annex_header(row.clause_path):
            children = await _annex_child_chunks(session, regulation_code, row.clause_path)
            halves.extend((child, "exclusion") for child in children)

    seen: set[str] = set(exclude_chunk_ids)
    wanted: list[tuple[_ScopeRow, Literal["inclusion", "exclusion"]]] = []
    for row, half in halves:
        if row.id not in seen:
            seen.add(row.id)
            wanted.append((row, half))
    if not wanted:
        return []

    scores = await _dense_scores_for_ids(session, query_dense, [row.id for row, _ in wanted])
    return [
        RetrievalHit(
            chunk_id=row.id,
            clause_id=row.clause_id,
            document_id=row.document_id,
            clause_path=row.clause_path,
            text=row.text,
            dense_score=scores.get(row.id, 0.0),
            sparse_score=0.0,
            # Ranked last deliberately: included for coverage, not because it
            # out-competed anything on relevance.
            fused_score=0.0,
            scope_half=half,
        )
        for row, half in wanted
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
            ClauseRow.clause_path,
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
    return [
        (row.id, row.clause_id, row.document_id, row.clause_path, row.text, -row.neg_ip)
        for row in result
    ]


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
    # chunk_id -> (clause_id, document_id, clause_path, text)
    meta: dict[str, tuple[str, str, str, str]] = {}

    for rank, (chunk_id, clause_id, document_id, clause_path, text, score) in enumerate(
        dense_ranked, start=1
    ):
        rrf_term = fusion_weight * (1 / (_RRF_K + rank))
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_term
        dense_scores[chunk_id] = score
        meta[chunk_id] = (clause_id, document_id, clause_path, text)

    for rank, (chunk_id, clause_id, document_id, clause_path, text, score) in enumerate(
        sparse_ranked, start=1
    ):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1 - fusion_weight) * (
            1 / (_RRF_K + rank)
        )
        sparse_scores[chunk_id] = score
        meta[chunk_id] = (clause_id, document_id, clause_path, text)

    sparse_only_ids = [chunk_id for chunk_id in fused_scores if chunk_id not in dense_scores]
    dense_scores.update(
        await _dense_scores_for_ids(session, query_embedding.dense, sparse_only_ids)
    )

    hits = [
        RetrievalHit(
            chunk_id=chunk_id,
            clause_id=meta[chunk_id][0],
            document_id=meta[chunk_id][1],
            clause_path=meta[chunk_id][2],
            text=meta[chunk_id][3],
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
