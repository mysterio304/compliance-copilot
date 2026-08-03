"""Agent graph tests. The end-to-end tests hit the real DB and the real
DeepSeek API — no mocks. The validation/retry tests use a synthetic state
with a fabricated bad clause_id, since a real LLM is unlikely to organically
produce an invalid citation (assess_applicability's extra_validate already
rejects citations outside the retrieved set before they reach the report) —
this is the one path in the graph deliberately exercised without a live call."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from compliance_copilot.agent.graph import run_agent
from compliance_copilot.agent.nodes import finalize_partial_report_node, make_validate_node
from compliance_copilot.config import Settings, get_settings
from compliance_copilot.db.models import ChunkRow
from compliance_copilot.db.session import make_engine, make_session_factory
from compliance_copilot.schemas.citations import Citation
from compliance_copilot.schemas.reports import ComplianceFinding, ComplianceReport


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


async def test_agent_produces_findings_with_valid_citations_for_real_spec(
    session: AsyncSession, settings: Settings
) -> None:
    result = await run_agent(
        settings,
        session,
        "Product: LED desk lamp, 5V DC input via USB-C adapter\n"
        "100-240V AC 50/60Hz (adapter)\nMaterials: ABS housing, aluminium heat sink\n"
        "No wireless functionality",
    )
    report = result["report"]
    assert isinstance(report, ComplianceReport)
    assert len(report.findings) > 0
    for finding in report.findings:
        assert len(finding.citations) >= 1
    # every finding survived validation (no errors on the terminal state)
    assert result.get("validation_errors") in (None, [])


async def test_agent_abstains_on_vague_spec(session: AsyncSession, settings: Settings) -> None:
    result = await run_agent(
        settings,
        session,
        "Product: Widget X200. No further technical details, materials, voltage, "
        "or intended use information has been provided by the supplier.",
    )
    report = result["report"]
    assert report.findings == []
    assert report.overall_confidence == 0.0
    assert len(report.unresolved_questions) > 0
    assert result.get("planned_regulations") == []


async def test_agent_excludes_toy_regulation_for_non_toy_despite_child_marketing(
    session: AsyncSession, settings: Settings
) -> None:
    result = await run_agent(
        settings,
        session,
        "Product: Cartoon Animal Night Light\n"
        "Description: decorative bedroom lighting fixture shaped like a cartoon "
        "fox, soft LED glow, wall-pluggable\nInput: 220V AC 50Hz\n"
        "Materials: PC plastic shade, ABS base\n"
        "Marketing: sold in the children's bedroom decor section, but explicitly "
        "not intended for play — a lighting fixture, not a toy",
    )
    report = result["report"]
    predicted = {f.regulation_code for f in report.findings}
    assert "ТР ТС 008/2011" not in predicted


async def test_validate_node_flags_nonexistent_clause_id(session: AsyncSession) -> None:
    validate_node = make_validate_node(session)
    fake_finding = ComplianceFinding(
        regulation_code="ТР ТС 004/2011",
        applies=True,
        reasoning="synthetic test finding",
        citations=[
            Citation(
                clause_id="does_not_exist__0",
                document_id="tr_ts_004_2011",
                quoted_span="synthetic",
                relevance_score=0.9,
            )
        ],
    )
    report = ComplianceReport(
        product_summary="synthetic",
        findings=[fake_finding],
        overall_confidence=0.9,
    )
    result = await validate_node({"report": report})
    assert len(result["validation_errors"]) == 1
    assert "does_not_exist__0" in result["validation_errors"][0]
    assert result["validation_attempts"] == 1


async def test_finalize_partial_report_drops_invalid_findings_and_explains_why() -> None:
    good_finding = ComplianceFinding(
        regulation_code="ТР ТС 020/2011",
        applies=True,
        reasoning="valid finding",
        citations=[
            Citation(
                clause_id="tr_ts_020_2011__1",
                document_id="tr_ts_020_2011",
                quoted_span="real",
                relevance_score=0.8,
            )
        ],
    )
    bad_finding = ComplianceFinding(
        regulation_code="ТР ТС 004/2011",
        applies=True,
        reasoning="invalid finding",
        citations=[
            Citation(
                clause_id="does_not_exist__0",
                document_id="tr_ts_004_2011",
                quoted_span="fake",
                relevance_score=0.9,
            )
        ],
    )
    report = ComplianceReport(
        product_summary="synthetic",
        findings=[good_finding, bad_finding],
        overall_confidence=0.85,
    )
    state = {
        "report": report,
        "validation_errors": [
            "ТР ТС 004/2011: cited clause_id 'does_not_exist__0' does not exist in the database"
        ],
    }
    result = await finalize_partial_report_node(state)
    partial_report = result["report"]
    assert len(partial_report.findings) == 1
    assert partial_report.findings[0].regulation_code == "ТР ТС 020/2011"
    assert any("dropped" in q for q in partial_report.unresolved_questions)
