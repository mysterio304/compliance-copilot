from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from compliance_copilot.ingest.chunker import chunk_clause
from compliance_copilot.ingest.clause_extractor import extract_clauses
from compliance_copilot.ingest.pdf_parser import parse_pdf
from compliance_copilot.schemas.documents import Chunk, Clause, Language, SourceDocument


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    title: str
    regulation_code: str
    language: Language
    publication_date: date
    source_url: str
    pdf_path: Path


class IngestedDocument(BaseModel):
    document: SourceDocument
    clauses: list[Clause]
    chunks: list[Chunk]


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    with manifest_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base_dir = manifest_path.parent.parent  # manifest lives at data/, paths are repo-relative
    return [
        ManifestEntry(
            doc_id=entry["doc_id"],
            title=entry["title"],
            regulation_code=entry["regulation_code"],
            language=entry["language"],
            publication_date=date.fromisoformat(entry["publication_date"]),
            source_url=entry["source_url"],
            pdf_path=base_dir / entry["pdf_path"],
        )
        for entry in raw["documents"]
    ]


def ingest_document(entry: ManifestEntry, embedding_model: str) -> IngestedDocument:
    blocks = parse_pdf(entry.pdf_path)
    clauses, _full_text = extract_clauses(
        blocks, document_id=entry.doc_id, language=entry.language
    )
    chunks = [chunk for clause in clauses for chunk in chunk_clause(clause, embedding_model)]
    document = SourceDocument(
        id=entry.doc_id,
        title=entry.title,
        regulation_code=entry.regulation_code,
        language=entry.language,
        publication_date=entry.publication_date,
        source_url=entry.source_url,
    )
    return IngestedDocument(document=document, clauses=clauses, chunks=chunks)


def ingest_corpus(manifest_path: Path, embedding_model: str) -> list[IngestedDocument]:
    return [ingest_document(entry, embedding_model) for entry in load_manifest(manifest_path)]
