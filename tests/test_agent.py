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
from compliance_copilot.agent.llm import StructuredOutputError
from compliance_copilot.agent.models import ScopeAssessment
from compliance_copilot.agent.nodes import (
    _majority_verdict,
    finalize_partial_report_node,
    make_validate_node,
)
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


@pytest.mark.llm
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

    # Regression check for a real, found-live reasoning bug: the agent used to
    # conclude LVD doesn't apply by reasoning about the lamp's own 5V DC
    # output rather than its 100-240V AC adapter — the actual equipment in
    # scope. Verified fixed 3/3 on live runs before this assertion was added.
    lvd_finding = next((f for f in report.findings if f.regulation_code == "ТР ТС 004/2011"), None)
    assert lvd_finding is not None
    assert lvd_finding.applies is True


@pytest.mark.llm
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


@pytest.mark.llm
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


class _CountingCaller:
    """Returns the queued verdicts in order, counting how many calls were made
    — the call count is the point of these tests as much as the verdict is."""

    def __init__(self, verdicts: list[bool | type[Exception]]) -> None:
        self.verdicts = verdicts
        self.calls = 0

    async def __call__(self) -> ScopeAssessment:
        verdict = self.verdicts[self.calls]
        self.calls += 1
        if isinstance(verdict, type):
            raise verdict("synthetic failure")
        return ScopeAssessment(
            in_scope=verdict, reasoning=f"call {self.calls}", cited_clause_ids=["c__0"]
        )


async def test_majority_verdict_stops_at_two_when_they_agree() -> None:
    # Two agreeing calls are already the majority of three, so the third would
    # cost a request without being able to change the outcome.
    caller = _CountingCaller([True, True, False])
    result = await _majority_verdict(caller, lambda a: a.in_scope)
    assert result.in_scope is True
    assert caller.calls == 2


async def test_majority_verdict_breaks_a_tie_with_a_third_call() -> None:
    caller = _CountingCaller([True, False, False])
    result = await _majority_verdict(caller, lambda a: a.in_scope)
    assert result.in_scope is False
    assert caller.calls == 3


async def test_majority_verdict_returns_the_winning_response_whole() -> None:
    """The reasoning and citations reported must be the ones belonging to the
    verdict reported — a report that cites the losing side's evidence would be
    worse than either answer on its own."""
    caller = _CountingCaller([False, True, True])
    result = await _majority_verdict(caller, lambda a: a.in_scope)
    assert result.in_scope is True
    assert result.reasoning == "call 2"


async def test_majority_verdict_survives_one_failed_call() -> None:
    # Previously a single StructuredOutputError sank the whole assessment;
    # with two samples in flight, one surviving response is enough.
    caller = _CountingCaller([StructuredOutputError, True])
    result = await _majority_verdict(caller, lambda a: a.in_scope)
    assert result.in_scope is True
    assert caller.calls == 2


async def test_majority_verdict_raises_when_every_call_fails() -> None:
    caller = _CountingCaller([StructuredOutputError, StructuredOutputError])
    with pytest.raises(StructuredOutputError):
        await _majority_verdict(caller, lambda a: a.in_scope)
