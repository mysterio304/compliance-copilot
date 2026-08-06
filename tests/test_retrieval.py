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

import json
from collections.abc import AsyncIterator

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_copilot.config import get_settings
from compliance_copilot.db.models import ChunkRow, ClauseRow
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


async def test_compact_query_beats_diluted_prose_for_a_specific_substance_mention(
    session: AsyncSession, embedding_model: str
) -> None:
    """A specific fact (lead content) gets diluted into irrelevance when embedded
    as one query alongside a long, mostly-unrelated prose spec — found live: a
    realistic full spec mentioning lead content retrieved zero lead-related
    clauses in the top 10, worse than even the bare word "lead" alone. This is
    why agent/nodes.py's retrieve_node queries with the compact extracted
    ProductAttributes JSON instead of raw spec_text — verifying that choice
    here, not just asserting it works from the node itself."""
    long_diluted_spec = (
        "Product: Wired Earbuds\n"
        "Materials: ABS housing, copper wire, silicone tips\n"
        "Hazardous substances declaration: contains lead (Pb) content below 0.1% "
        "by weight, no cadmium, no mercury present"
    )
    compact_query = '{"materials": ["ABS housing", "copper wire", "lead (Pb) below 0.1%"]}'

    diluted_hits = await hybrid_search(
        session,
        long_diluted_spec,
        embedding_model,
        top_k=10,
        fusion_weight=0.5,
        relevance_floor=0.35,
        regulation_code="ТР ЕАЭС 037/2016",
    )
    compact_hits = await hybrid_search(
        session,
        compact_query,
        embedding_model,
        top_k=10,
        fusion_weight=0.5,
        relevance_floor=0.35,
        regulation_code="ТР ЕАЭС 037/2016",
    )

    assert not any("свинец" in hit.text.lower() for hit in diluted_hits)
    assert any("свинец" in hit.text.lower() for hit in compact_hits)


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


async def test_fusion_weight_changes_results_at_shipped_relevance_floor(
    session: AsyncSession, embedding_model: str
) -> None:
    """Regression test for a real bug: dense_score used to default to 0.0 for
    any chunk found only via sparse search, so the relevance floor (checked
    against dense_score) silently discarded a sparse-favored chunk's real
    dense similarity and floor-filtered it regardless of fusion_weight —
    making fusion_weight practically inert whenever the floor was active.
    Fixed by looking up the real dense score for sparse-only discoveries
    instead of assuming they have none."""
    query = (
        "Product: LED desk lamp, 5V DC input via USB-C adapter\n"
        "100-240V AC 50/60Hz (adapter)\nMaterials: ABS housing, aluminium heat sink"
    )
    result_sets = {}
    for fusion_weight in (0.0, 0.5, 1.0):
        hits = await hybrid_search(
            session,
            query,
            embedding_model,
            top_k=10,
            fusion_weight=fusion_weight,
            relevance_floor=0.35,
        )
        result_sets[fusion_weight] = tuple(hit.clause_id for hit in hits)

    assert len({result_sets[0.0], result_sets[0.5], result_sets[1.0]}) > 1, (
        "fusion_weight produced identical results across 0.0/0.5/1.0 at the "
        "shipped relevance_floor — the inertness bug may have regressed"
    )


async def test_ensure_scope_clause_supplies_the_clause_ranking_alone_misses(
    session: AsyncSession, embedding_model: str
) -> None:
    """Regression test for a real false negative: asked whether toy safety
    applied to a plush teddy bear, the agent answered "no" — correctly, given
    its evidence. Ranked retrieval had handed it Приложение 1 and the
    electrical-properties clauses but not Статья 1 пункт 1, the clause that
    states which products the regulation covers, and the agent won't assert
    applicability it has no citation for. The scope clause now comes along
    regardless of where it ranks."""
    teddy_bear_attributes = json.dumps(
        {
            "product_name": "Plush Teddy Bear",
            "voltage": None,
            "frequency": None,
            "materials": ["polyester textile fabric", "polyester fiber stuffing"],
            "wireless_bands": [],
            "has_batteries": False,
            "intended_user": "children",
            "notes": None,
        }
    )

    async def clause_id_for(path: str) -> str:
        return (
            await session.execute(
                select(ClauseRow.id).where(
                    ClauseRow.document_id == "tr_ts_008_2011",
                    ClauseRow.clause_path == path,
                )
            )
        ).scalar_one()

    # Both halves of the scope statement: what the regulation covers, and the
    # clause that carves categories back out of it.
    scope_clause_id = await clause_id_for("Статья 1, пункт 1")
    exclusion_clause_id = await clause_id_for("Статья 1, пункт 2")

    # Shipped parameters, not test-friendly ones: the point is that the miss
    # happens in the configuration the system actually runs.
    settings = get_settings()

    ranked_only = await hybrid_search(
        session,
        teddy_bear_attributes,
        embedding_model,
        top_k=settings.retrieval_top_k,
        fusion_weight=settings.retrieval_fusion_weight,
        relevance_floor=settings.retrieval_relevance_floor,
        regulation_code="ТР ТС 008/2011",
    )
    assert scope_clause_id not in {hit.clause_id for hit in ranked_only}, (
        "ranking alone now surfaces the scope clause for this query — if that is "
        "a genuine retrieval improvement, this test no longer guards anything"
    )

    with_scope = await hybrid_search(
        session,
        teddy_bear_attributes,
        embedding_model,
        top_k=settings.retrieval_top_k,
        fusion_weight=settings.retrieval_fusion_weight,
        relevance_floor=settings.retrieval_relevance_floor,
        regulation_code="ТР ТС 008/2011",
        ensure_scope_clause=True,
    )
    with_scope_ids = {hit.clause_id for hit in with_scope}
    assert scope_clause_id in with_scope_ids
    # The exclusion half matters as much: supplying only the positive scope let
    # the assessor affirm applicability for categories the regulation carves out
    # by name, since nothing in its evidence said they were carved out.
    assert exclusion_clause_id in with_scope_ids

    for clause_id in (scope_clause_id, exclusion_clause_id):
        hit = next(hit for hit in with_scope if hit.clause_id == clause_id)
        assert hit.dense_score > 0.0, f"{clause_id} should carry its real similarity, not a stub"
