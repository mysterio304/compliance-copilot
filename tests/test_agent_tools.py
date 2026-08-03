"""Integration tests against the real DB and, for extraction, the real DeepSeek
API — no mocks. Requires docker compose up -d db, the corpus ingested and
loaded, and DEEPSEEK_API_KEY set for the extraction tests."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_copilot.agent.tools import (
    extract_product_attributes,
    fetch_clause,
    list_regulation_structure,
    search_regulations,
)
from compliance_copilot.config import Settings, get_settings
from compliance_copilot.db.models import ChunkRow
from compliance_copilot.db.session import make_engine, make_session_factory

# asyncio_mode = "auto" in pyproject.toml already treats async defs as asyncio
# tests; no pytestmark needed (this file mixes async and sync tests).


@pytest.fixture(scope="module")
def settings() -> Settings:
    return get_settings()


@pytest.fixture
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    eng = make_engine(settings.database_url)
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
        has_data = (await s.execute(select(ChunkRow.id).limit(1))).first()
        if has_data is None:
            pytest.skip("No embedded chunks — run ingest/load scripts first.")
        yield s


async def test_search_regulations_returns_hits(session: AsyncSession, settings: Settings) -> None:
    hits = await search_regulations(session, settings, "входное напряжение", "ru", top_k=5)
    assert len(hits) > 0


async def test_search_regulations_respects_regulation_filter(
    session: AsyncSession, settings: Settings
) -> None:
    hits = await search_regulations(
        session,
        settings,
        "требования безопасности",
        "ru",
        top_k=10,
        regulation_code="ТР ТС 004/2011",
    )
    assert len(hits) > 0
    assert all(hit.document_id == "tr_ts_004_2011" for hit in hits)


async def test_fetch_clause_returns_context(session: AsyncSession) -> None:
    ctx = await fetch_clause(session, "tr_ts_004_2011__3")
    assert ctx is not None
    assert ctx.clause.id == "tr_ts_004_2011__3"
    assert ctx.preceding is not None or ctx.following is not None


async def test_fetch_clause_returns_none_for_missing_id(session: AsyncSession) -> None:
    ctx = await fetch_clause(session, "does_not_exist__0")
    assert ctx is None


async def test_list_regulation_structure_returns_ordered_paths(session: AsyncSession) -> None:
    toc = await list_regulation_structure(session, "ТР ТС 004/2011")
    assert len(toc) > 0
    assert toc[0].startswith("Предисловие") or toc[0].startswith("Статья")


async def test_list_regulation_structure_empty_for_unknown_regulation(
    session: AsyncSession,
) -> None:
    toc = await list_regulation_structure(session, "ТР ТС 999/9999")
    assert toc == []


@pytest.mark.llm
def test_extract_product_attributes_finds_real_values(settings: Settings) -> None:
    attrs = extract_product_attributes(
        settings,
        "Product: LED desk lamp, 5V DC input via USB-C adapter\n"
        "100-240V AC 50/60Hz (adapter)\nMaterials: ABS housing, aluminium heat sink",
    )
    assert attrs.voltage is not None
    assert not attrs.is_empty


@pytest.mark.llm
def test_extract_product_attributes_flags_vague_spec_as_empty(settings: Settings) -> None:
    attrs = extract_product_attributes(
        settings,
        "Product: Widget X200. No further technical details, materials, voltage, "
        "or intended use information has been provided by the supplier.",
    )
    assert attrs.is_empty
