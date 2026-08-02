import re
from dataclasses import dataclass, field

from compliance_copilot.ingest.pdf_parser import ParsedBlock
from compliance_copilot.schemas.documents import Clause, Language

_ARTICLE_RE = re.compile(r"^Статья\s+(\d+)\.?\s*(.*)$")
_APPENDIX_RE = re.compile(r"^Приложение\s*(?:№\s*)?(\d+)?\.?\s*(.*)$")
_ROMAN_RE = re.compile(r"^([IVXLCDM]+)\.\s+(.*)$")
_NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s*")

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _is_valid_roman(candidate: str) -> bool:
    """Guards against false positives: real roman numerals only, in a sane range."""
    if not candidate or not all(c in _ROMAN_VALUES for c in candidate):
        return False
    total = 0
    prev = 0
    for char in reversed(candidate):
        value = _ROMAN_VALUES[char]
        total += value if value >= prev else -value
        prev = value
    return 0 < total <= 50


def heading_to_clause_path(heading_text: str) -> str:
    """Maps a recognized heading to its citation path. Handles the three structures
    observed across the corpus: "Статья N" (ТР ТС 004/010/008/020), roman-numeral
    "Раздел" sections (ТР ЕАЭС 037/2016), and "Приложение" (numbered or bare)."""
    if match := _ARTICLE_RE.match(heading_text):
        return f"Статья {match.group(1)}"
    if match := _APPENDIX_RE.match(heading_text):
        number = match.group(1)
        return f"Приложение {number}" if number else "Приложение"
    if (match := _ROMAN_RE.match(heading_text)) and _is_valid_roman(match.group(1)):
        return f"Раздел {match.group(1)}"
    return heading_text


@dataclass
class _OpenClause:
    clause_path: str
    start: int
    end: int
    texts: list[str] = field(default_factory=list)


def extract_clauses(
    blocks: list[ParsedBlock], document_id: str, language: Language
) -> tuple[list[Clause], str]:
    """Segments parsed blocks into citable Clause objects, keyed by статья/раздел/
    приложение plus, where present, numbered пункт. Char spans are computed against
    a single reconstructed full_text so clause.text == full_text[start:end] always
    holds — that identity is what later tests check.

    Returns (clauses, full_text).
    """
    full_text = ""
    clauses: list[Clause] = []
    heading_path: str | None = None
    open_clause: _OpenClause | None = None

    def add_to_full_text(text: str) -> tuple[int, int]:
        nonlocal full_text
        if full_text:
            full_text += "\n\n"
        start = len(full_text)
        full_text += text
        return start, len(full_text)

    def flush() -> None:
        nonlocal open_clause
        if open_clause is not None and open_clause.texts:
            clauses.append(
                Clause(
                    id=f"{document_id}__{len(clauses)}",
                    document_id=document_id,
                    clause_path=open_clause.clause_path,
                    text=full_text[open_clause.start : open_clause.end],
                    char_start=open_clause.start,
                    char_end=open_clause.end,
                    language=language,
                )
            )
        open_clause = None

    for block in blocks:
        if block.kind == "heading":
            flush()
            heading_path = heading_to_clause_path(block.text)
            add_to_full_text(block.text)
            continue

        assert heading_path is not None, "body block encountered before any heading"

        match = _NUMBERED_ITEM_RE.match(block.text)
        if match:
            flush()
            block_start, block_end = add_to_full_text(block.text)
            clause_text_start = block_start + match.end()
            open_clause = _OpenClause(
                clause_path=f"{heading_path}, пункт {match.group(1)}",
                start=clause_text_start,
                end=block_end,
                texts=[block.text[match.end() :]],
            )
        else:
            block_start, block_end = add_to_full_text(block.text)
            if open_clause is None:
                open_clause = _OpenClause(
                    clause_path=heading_path, start=block_start, end=block_end, texts=[block.text]
                )
            else:
                open_clause.end = block_end
                open_clause.texts.append(block.text)

    flush()
    return clauses, full_text
