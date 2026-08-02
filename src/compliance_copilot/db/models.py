from datetime import date as date_

from pgvector.sqlalchemy import SPARSEVEC, Vector
from sqlalchemy import Date, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# BGE-M3 dense output dimension and its XLM-RoBERTa tokenizer vocab size, confirmed
# against the actual model (not assumed) — see AutoTokenizer.from_pretrained("BAAI/bge-m3").
BGE_M3_DENSE_DIM = 1024
BGE_M3_VOCAB_SIZE = 250_002


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    regulation_code: Mapped[str] = mapped_column(String)
    language: Mapped[str] = mapped_column(String(2))
    publication_date: Mapped[date_] = mapped_column(Date)
    source_url: Mapped[str] = mapped_column(Text)

    clauses: Mapped[list["ClauseRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ClauseRow(Base):
    __tablename__ = "clauses"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    clause_path: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int]
    char_end: Mapped[int]
    language: Mapped[str] = mapped_column(String(2))

    document: Mapped["DocumentRow"] = relationship(back_populates="clauses")
    chunks: Mapped[list["ChunkRow"]] = relationship(
        back_populates="clause", cascade="all, delete-orphan"
    )


class ChunkRow(Base):
    """Dense/sparse embeddings are nullable: a chunk exists once ingested (Phase 2)
    and gains embeddings once embedded (Phase 3) — the two steps are separable.

    No ANN index (HNSW/ivfflat) yet: at ~700 rows a brute-force scan is exact and
    fast enough. Add one (with a deliberately chosen operator class and training
    data) once corpus size actually makes it necessary — not before.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    clause_id: Mapped[str] = mapped_column(ForeignKey("clauses.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String)
    dense_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(BGE_M3_DENSE_DIM), nullable=True
    )
    sparse_embedding: Mapped[object | None] = mapped_column(
        SPARSEVEC(BGE_M3_VOCAB_SIZE), nullable=True
    )

    clause: Mapped["ClauseRow"] = relationship(back_populates="chunks")
