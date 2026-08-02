import re

from compliance_copilot.schemas.documents import Chunk, Clause

MAX_CHUNK_CHARS = 2000

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-Z\d])")


def _split_into_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s]


def chunk_clause(clause: Clause, embedding_model: str) -> list[Chunk]:
    """Chunks on clause boundaries, not fixed token windows: a clause that fits
    within MAX_CHUNK_CHARS is one chunk. Only clauses that exceed it get split,
    by sentence, so every fragment still carries its parent clause_id (and, via
    that id, the clause_path used for citation)."""
    if len(clause.text) <= MAX_CHUNK_CHARS:
        return [
            Chunk(
                id=f"{clause.id}__chunk0",
                clause_id=clause.id,
                text=clause.text,
                embedding_model=embedding_model,
            )
        ]

    chunks: list[Chunk] = []
    current = ""
    for sentence in _split_into_sentences(clause.text):
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    id=f"{clause.id}__chunk{len(chunks)}",
                    clause_id=clause.id,
                    text=current,
                    embedding_model=embedding_model,
                )
            )
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(
            Chunk(
                id=f"{clause.id}__chunk{len(chunks)}",
                clause_id=clause.id,
                text=current,
                embedding_model=embedding_model,
            )
        )
    return chunks
