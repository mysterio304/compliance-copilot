"""API endpoint tests against the real DB (and, for /analyse, the real
DeepSeek API — marked @pytest.mark.llm) — no mocks."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from compliance_copilot.api.main import create_app, lifespan
from compliance_copilot.config import get_settings
from compliance_copilot.db.models import ChunkRow
from compliance_copilot.db.session import make_engine, make_session_factory


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return Path("data/raw_pdfs/tr_ts_004_2011.pdf").read_bytes()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with lifespan(app):
        # skip gracefully if the corpus isn't loaded, matching the other
        # integration tests' precondition check
        settings = get_settings()
        engine = make_engine(settings.database_url)
        session_factory = make_session_factory(engine)
        async with session_factory() as session:
            has_data = (await session.execute(select(ChunkRow.id).limit(1))).first()
        await engine.dispose()
        if has_data is None:
            pytest.skip("No embedded chunks — run ingest/load scripts first.")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_list_documents_returns_all_five_regulations(client: AsyncClient) -> None:
    response = await client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    codes = {doc["regulation_code"] for doc in body}
    assert "ТР ТС 004/2011" in codes
    assert all(doc["clause_count"] > 0 for doc in body)


async def test_resolve_clause_returns_context(client: AsyncClient) -> None:
    response = await client.get("/clauses/tr_ts_004_2011__3")
    assert response.status_code == 200
    body = response.json()
    assert body["clause"]["id"] == "tr_ts_004_2011__3"


async def test_resolve_clause_404_for_missing_id(client: AsyncClient) -> None:
    response = await client.get("/clauses/does_not_exist__0")
    assert response.status_code == 404


async def test_analyse_requires_spec_text_or_file(client: AsyncClient) -> None:
    response = await client.post("/analyse")
    assert response.status_code == 400


@pytest.mark.llm
async def test_analyse_with_spec_text_returns_report(client: AsyncClient) -> None:
    response = await client.post(
        "/analyse",
        data={
            "spec_text": (
                "Product: Plush Teddy Bear\n"
                "Materials: polyester textile fabric, polyester fiber stuffing\n"
                "Power source: none — no electronic parts, no batteries\n"
                "Intended use: children ages 0+, soft toy for play"
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert any(f["regulation_code"] == "ТР ТС 008/2011" for f in body["findings"])
    for finding in body["findings"]:
        assert len(finding["citations"]) >= 1


@pytest.mark.llm
async def test_analyse_with_pdf_upload_returns_report(
    client: AsyncClient, sample_pdf_bytes: bytes
) -> None:
    response = await client.post(
        "/analyse", files={"file": ("spec.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    body = response.json()
    assert "findings" in body


async def test_analyse_rejects_non_pdf_upload(client: AsyncClient) -> None:
    response = await client.post(
        "/analyse", files={"file": ("spec.txt", b"not a pdf", "text/plain")}
    )
    assert response.status_code == 400
