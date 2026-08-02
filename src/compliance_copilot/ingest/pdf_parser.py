from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docling.document_converter import DocumentConverter

BlockKind = Literal["heading", "body"]

_HEADING_LABELS = frozenset({"section_header", "title"})


@dataclass(frozen=True)
class ParsedBlock:
    """One structural unit recovered by Docling, in reading order."""

    kind: BlockKind
    text: str


def parse_pdf(pdf_path: Path) -> list[ParsedBlock]:
    """Parse a regulation PDF with Docling, preserving heading hierarchy.

    Front matter (title block, "Содержание"/table of contents) that precedes the
    first heading is dropped — it is never a citable clause. Everything from the
    first heading onward is kept in reading order.
    """
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    document = result.document

    blocks: list[ParsedBlock] = []
    seen_heading = False
    for item, _level in document.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        label = getattr(item, "label", None)
        is_heading = label in _HEADING_LABELS
        if is_heading:
            seen_heading = True
        if not seen_heading:
            continue

        text = text.strip()
        # Docling strips the enumeration marker ("1.", "2.") out of list_item text
        # into a separate attribute; reattach it so downstream пункт-numbering
        # detection sees the same inline "N. ..." shape regardless of how Docling
        # classified the paragraph (list_item vs. plain text).
        if getattr(item, "enumerated", False) and (marker := getattr(item, "marker", None)):
            text = f"{marker} {text}"

        blocks.append(ParsedBlock(kind="heading" if is_heading else "body", text=text))
    return blocks
