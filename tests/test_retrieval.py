"""Integration tests against a real, running Postgres+pgvector instance with the
real corpus loaded — no mocks. Requires, before running:

    docker compose up -d db
    uv run python scripts/ingest_corpus.py
    uv run python scripts/load_corpus.py

Cross-lingual retrieval is the hard requirement for this phase: a query in one
language must retrieve the correct clause in another. The three pairs below
(cadmium, hexavalent chromium, low-voltage equipment) were verified interactively
to rank the correct clause at or near top-1. A fourth pair (lead/铅 -> Свинец) is
included too, but as a softer cosine-similarity check, not a top-k assertion —
measured against the real corpus, "lead" is a short, common word that several
unrelated chunks also score respectably against, so the correct clause places
outside the top few results even though the encoder does capture real cross-
lingual relatedness (cosine ~0.38-0.43 vs. ~0.15-0.25 for unrelated pairs). That
gap is real and worth tracking as a retrieval-quality limitation, not something
to paper over by only testing the cases that already work.
"""

from collections.abc import AsyncIterator

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_copilot.config import get_settings
from compliance_copilot.db.models import ChunkRow
from compliance_copilot.db.session import make_engine, make_session_factory
from compliance_copilot.retrieval.embedder import embed_query
from compliance_copilot.retrieval.hybrid import hybrid_search

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def embedding_model() -> str:
    return get_settings().embedding_model_name


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    # Function-scoped: pytest-asyncio gives each test its own event loop, and an
    # asyncpg connection pool is bound to the loop it was created in — a
    # module-scoped engine here fails with "attached to a different loop" on the
    # second test.
    eng = make_engine(get_settings().database_url)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return make_session_factory(engine)


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as s:
        count = (await s.execute(select(ChunkRow.id).limit(1))).first()
        if count is None:
            pytest.skip(
                "No embedded chunks in the database — run "
                "scripts/ingest_corpus.py and scripts/load_corpus.py first."
            )
        yield s


@pytest.mark.parametrize(
    ("query", "expected_substring"),
    [
        ("镉", "Кадмий"),  # Chinese "cadmium" -> Russian clause
        ("六价铬", "Шестивалентный хром"),  # Chinese "hexavalent chromium"
        ("low voltage equipment", "Низковольтное оборудование"),  # English
    ],
)
async def test_cross_lingual_retrieval_finds_correct_clause(
    session: AsyncSession, embedding_model: str, query: str, expected_substring: str
) -> None:
    hits = await hybrid_search(
        session, query, embedding_model, top_k=5, fusion_weight=0.5, relevance_floor=0.0
    )
    assert any(expected_substring in hit.text for hit in hits), (
        f"query {query!r} did not retrieve a clause containing {expected_substring!r} "
        f"in the top 5: {[hit.text[:40] for hit in hits]}"
    )


@pytest.mark.parametrize("query", ["铅", "lead"])
async def test_cross_lingual_lead_has_real_but_weaker_similarity(
    session: AsyncSession, embedding_model: str, query: str
) -> None:
    stmt = select(ChunkRow).where(ChunkRow.clause_id == "tr_eaes_037_2016__54")  # "Свинец"
    chunk = (await session.execute(stmt)).scalar_one()
    assert chunk.dense_embedding is not None

    query_embedding = embed_query(query, embedding_model)
    dense = np.array(chunk.dense_embedding)
    query_dense = np.array(query_embedding.dense)
    norm_product = np.linalg.norm(dense) * np.linalg.norm(query_dense)
    cosine = float(np.dot(dense, query_dense) / norm_product)

    assert cosine > 0.3, (
        f"expected cross-lingual similarity for {query!r} -> 'Свинец' to be real "
        f"(>0.3) even if not top-ranked; got {cosine:.3f}"
    )


async def test_relevance_floor_returns_empty_for_nonsense_query(
    session: AsyncSession, embedding_model: str
) -> None:
    hits = await hybrid_search(
        session,
        "quantum banana spaceship recipe",
        embedding_model,
        top_k=5,
        fusion_weight=0.5,
        relevance_floor=0.5,
    )
    assert hits == []


async def test_relevant_query_clears_the_floor(session: AsyncSession, embedding_model: str) -> None:
    hits = await hybrid_search(
        session,
        "входное напряжение оборудования",
        embedding_model,
        top_k=5,
        fusion_weight=0.5,
        relevance_floor=0.3,
    )
    assert len(hits) > 0
    assert all(hit.dense_score >= 0.3 for hit in hits)


async def test_hybrid_search_orders_hits_by_descending_fused_score(
    session: AsyncSession, embedding_model: str
) -> None:
    hits = await hybrid_search(
        session,
        "требования безопасности низковольтного оборудования",
        embedding_model,
        top_k=10,
        fusion_weight=0.5,
        relevance_floor=0.0,
    )
    scores = [hit.fused_score for hit in hits]
    assert scores == sorted(scores, reverse=True)
