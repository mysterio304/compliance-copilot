from pathlib import Path

from docling.document_converter import DocumentConverter


def extract_plain_text(pdf_path: Path) -> str:
    """Plain-text extraction for arbitrary PDFs (supplier spec sheets) — distinct
    from pdf_parser.parse_pdf, which assumes EAEU regulation structure (drops
    front matter before the first heading, splits on статья/пункт boundaries).
    A supplier spec has no such structure to preserve; we just need its text."""
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()
