import json
from pathlib import Path

from pgvector import SparseVector
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.db.models import BGE_M3_VOCAB_SIZE, ChunkRow, ClauseRow, DocumentRow
from compliance_copilot.ingest.pipeline import IngestedDocument
from compliance_copilot.retrieval.embedder import embed_texts


def load_ingested_document(path: Path) -> IngestedDocument:
    return IngestedDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


async def embed_and_store_document(
    session: AsyncSession, ingested: IngestedDocument, embedding_model: str
) -> int:
    """Embeds every chunk of one document with BGE-M3 and (re)writes it to Postgres.
    Idempotent: deletes any existing rows for this document_id first, so re-running
    the loader after a re-ingestion doesn't leave stale chunks behind.
    """
    document_id = ingested.document.id
    await session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))

    session.add(
        DocumentRow(
            id=document_id,
            title=ingested.document.title,
            regulation_code=ingested.document.regulation_code,
            language=ingested.document.language,
            publication_date=ingested.document.publication_date,
            source_url=ingested.document.source_url,
        )
    )
    for clause in ingested.clauses:
        session.add(
            ClauseRow(
                id=clause.id,
                document_id=clause.document_id,
                clause_path=clause.clause_path,
                text=clause.text,
                char_start=clause.char_start,
                char_end=clause.char_end,
                language=clause.language,
            )
        )

    chunk_texts = [chunk.text for chunk in ingested.chunks]
    embeddings = embed_texts(chunk_texts, embedding_model) if chunk_texts else []
    for chunk, embedding in zip(ingested.chunks, embeddings, strict=True):
        session.add(
            ChunkRow(
                id=chunk.id,
                clause_id=chunk.clause_id,
                text=chunk.text,
                embedding_model=chunk.embedding_model,
                dense_embedding=embedding.dense,
                sparse_embedding=SparseVector(embedding.sparse, BGE_M3_VOCAB_SIZE),
            )
        )

    await session.commit()
    return len(ingested.chunks)
