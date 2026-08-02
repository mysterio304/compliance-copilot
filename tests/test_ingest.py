"""Ingestion tests over a real fixture PDF (ТР ТС 004/2011 — the smallest, cleanest
regulation in the corpus): clause hierarchy is preserved and character spans map
back to the extracted source text exactly. No synthetic/mocked PDF content — this
runs the real Docling parser against a real, authentic-text PDF."""

from pathlib import Path

import pytest

from compliance_copilot.ingest.chunker import MAX_CHUNK_CHARS, chunk_clause
from compliance_copilot.ingest.clause_extractor import extract_clauses, heading_to_clause_path
from compliance_copilot.ingest.pdf_parser import ParsedBlock, parse_pdf
from compliance_copilot.schemas.documents import Clause

FIXTURE_PDF = Path(__file__).parent.parent / "data" / "raw_pdfs" / "tr_ts_004_2011.pdf"


@pytest.fixture(scope="module")
def parsed_blocks() -> list[ParsedBlock]:
    return parse_pdf(FIXTURE_PDF)


@pytest.fixture(scope="module")
def extracted() -> tuple[list[Clause], str]:
    blocks = parse_pdf(FIXTURE_PDF)
    return extract_clauses(blocks, document_id="tr_ts_004_2011", language="ru")


def test_parse_pdf_recovers_headings(parsed_blocks: list[ParsedBlock]) -> None:
    heading_texts = [b.text for b in parsed_blocks if b.kind == "heading"]
    assert any("Статья 1" in t for t in heading_texts)
    assert any("Статья 9" in t for t in heading_texts)
    assert any(t.startswith("Приложение") for t in heading_texts)


def test_parse_pdf_drops_front_matter(parsed_blocks: list[ParsedBlock]) -> None:
    # "Содержание" (table of contents label) and the title block precede the first
    # heading ("Предисловие") and must not appear as blocks.
    assert not any(b.text == "Содержание" for b in parsed_blocks)
    assert parsed_blocks[0].kind == "heading"


def test_clause_hierarchy_covers_every_article(extracted: tuple[list[Clause], str]) -> None:
    clauses, _full_text = extracted
    article_numbers = {
        clause.clause_path.split(",")[0]
        for clause in clauses
        if clause.clause_path.startswith("Статья")
    }
    expected = {f"Статья {n}" for n in range(1, 10)}
    assert expected <= article_numbers


def test_clause_hierarchy_includes_appendix(extracted: tuple[list[Clause], str]) -> None:
    clauses, _full_text = extracted
    assert any(c.clause_path.startswith("Приложение") for c in clauses)


def test_numbered_points_produce_punkt_paths(extracted: tuple[list[Clause], str]) -> None:
    clauses, _full_text = extracted
    assert any(c.clause_path == "Статья 1, пункт 1" for c in clauses)
    assert any(c.clause_path == "Статья 1, пункт 2" for c in clauses)


def test_every_clause_span_maps_back_to_source_text(extracted: tuple[list[Clause], str]) -> None:
    clauses, full_text = extracted
    assert len(clauses) > 0
    for clause in clauses:
        assert full_text[clause.char_start : clause.char_end] == clause.text
        assert clause.char_end - clause.char_start == len(clause.text)


def test_clause_document_id_is_consistent(extracted: tuple[list[Clause], str]) -> None:
    clauses, _full_text = extracted
    assert all(c.document_id == "tr_ts_004_2011" for c in clauses)


@pytest.mark.parametrize(
    ("heading_text", "expected_path"),
    [
        ("Статья 4. Требования безопасности", "Статья 4"),
        ("Статья 12. Что-то", "Статья 12"),
        ("Приложение", "Приложение"),
        ("Приложение № 2 Перечень чего-то", "Приложение 2"),
        ("Приложение 3. Заголовок", "Приложение 3"),
        ("III. Правила обращения на рынке", "Раздел III"),
        ("Предисловие", "Предисловие"),
    ],
)
def test_heading_to_clause_path_patterns(heading_text: str, expected_path: str) -> None:
    assert heading_to_clause_path(heading_text) == expected_path


def test_chunk_clause_keeps_short_clause_as_single_chunk() -> None:
    clause = Clause(
        id="c1",
        document_id="doc1",
        clause_path="Статья 1, пункт 1",
        text="Короткий пункт.",
        char_start=0,
        char_end=15,
        language="ru",
    )
    chunks = chunk_clause(clause, embedding_model="BAAI/bge-m3")
    assert len(chunks) == 1
    assert chunks[0].clause_id == "c1"
    assert chunks[0].text == clause.text


def test_chunk_clause_splits_oversized_clause_and_preserves_clause_id() -> None:
    sentence = "Это предложение о требованиях безопасности низковольтного оборудования. "
    long_text = sentence * (MAX_CHUNK_CHARS // len(sentence) + 5)
    clause = Clause(
        id="c2",
        document_id="doc1",
        clause_path="Статья 4, пункт 1",
        text=long_text.strip(),
        char_start=0,
        char_end=len(long_text.strip()),
        language="ru",
    )
    chunks = chunk_clause(clause, embedding_model="BAAI/bge-m3")
    assert len(chunks) > 1
    assert all(c.clause_id == "c2" for c in chunks)
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
