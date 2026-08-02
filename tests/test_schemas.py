import pytest
from pydantic import ValidationError

from compliance_copilot.schemas import (
    Chunk,
    Citation,
    Clause,
    ComplianceFinding,
    ComplianceReport,
    SourceDocument,
)


@pytest.fixture
def citation() -> Citation:
    return Citation(
        clause_id="tr_ts_004_2011__st4_p2",
        document_id="tr_ts_004_2011",
        quoted_span="устройство должно соответствовать требованиям",
        relevance_score=0.91,
    )


def test_source_document_round_trips_expected_fields() -> None:
    doc = SourceDocument(
        id="tr_ts_004_2011",
        title="О безопасности низковольтного оборудования",
        regulation_code="ТР ТС 004/2011",
        language="ru",
        publication_date="2011-08-16",
        source_url="https://docs.cntd.ru/document/902320567",
    )
    assert doc.regulation_code == "ТР ТС 004/2011"
    assert doc.language == "ru"


def test_clause_rejects_inverted_span() -> None:
    with pytest.raises(ValidationError, match="char_end"):
        Clause(
            id="c1",
            document_id="tr_ts_004_2011",
            clause_path="Статья 4, пункт 2",
            text="...",
            char_start=100,
            char_end=50,
            language="ru",
        )


def test_clause_accepts_valid_span() -> None:
    clause = Clause(
        id="c1",
        document_id="tr_ts_004_2011",
        clause_path="Статья 4, пункт 2",
        text="...",
        char_start=50,
        char_end=100,
        language="ru",
    )
    assert clause.char_end > clause.char_start


def test_chunk_allows_missing_embedding_until_phase_3() -> None:
    chunk = Chunk(
        id="chunk1",
        clause_id="c1",
        text="...",
        embedding_model="BAAI/bge-m3",
    )
    assert chunk.dense_vector is None
    assert chunk.sparse_weights is None


def test_compliance_finding_requires_at_least_one_citation(citation: Citation) -> None:
    finding = ComplianceFinding(
        regulation_code="ТР ТС 004/2011",
        applies=True,
        reasoning="Adapter output is within LVD voltage scope.",
        conformity_assessment_route="declaration_of_conformity",
        citations=[citation],
    )
    assert len(finding.citations) == 1


def test_compliance_finding_rejects_empty_citations() -> None:
    with pytest.raises(ValidationError, match="citations"):
        ComplianceFinding(
            regulation_code="ТР ТС 004/2011",
            applies=True,
            reasoning="An uncited claim must be unrepresentable.",
            conformity_assessment_route=None,
            citations=[],
        )


def test_compliance_report_assembles_findings(citation: Citation) -> None:
    finding = ComplianceFinding(
        regulation_code="ТР ТС 004/2011",
        applies=True,
        reasoning="Adapter output is within LVD voltage scope.",
        conformity_assessment_route="declaration_of_conformity",
        citations=[citation],
    )
    report = ComplianceReport(
        product_summary="LED desk lamp, 5V DC input via USB-C adapter",
        findings=[finding],
        unresolved_questions=[],
        overall_confidence=0.82,
    )
    assert report.findings[0].applies is True
    assert report.overall_confidence == pytest.approx(0.82)


def test_compliance_report_defaults_unresolved_questions_to_empty_list(
    citation: Citation,
) -> None:
    finding = ComplianceFinding(
        regulation_code="ТР ТС 004/2011",
        applies=True,
        reasoning="...",
        citations=[citation],
    )
    report = ComplianceReport(
        product_summary="...",
        findings=[finding],
        overall_confidence=0.5,
    )
    assert report.unresolved_questions == []
