from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.agent.llm import call_structured
from compliance_copilot.config import Settings
from compliance_copilot.db.models import ClauseRow, DocumentRow
from compliance_copilot.retrieval.hybrid import RetrievalHit, hybrid_search
from compliance_copilot.schemas.documents import Clause
from compliance_copilot.schemas.product import ProductAttributes


def _row_to_clause(row: ClauseRow) -> Clause:
    return Clause(
        id=row.id,
        document_id=row.document_id,
        clause_path=row.clause_path,
        text=row.text,
        char_start=row.char_start,
        char_end=row.char_end,
        language=row.language,
    )


async def search_regulations(
    session: AsyncSession,
    settings: Settings,
    query: str,
    lang: str,
    top_k: int,
    regulation_code: str | None = None,
    ensure_scope_clause: bool = False,
) -> list[RetrievalHit]:
    """Hybrid retrieval tool. `lang` is accepted for interface parity with the
    brief's tool signature and cross-lingual intent, but isn't otherwise used —
    BGE-M3 retrieval is cross-lingual by construction, not by a language filter."""
    del lang
    return await hybrid_search(
        session,
        query,
        settings.embedding_model_name,
        top_k,
        settings.retrieval_fusion_weight,
        settings.retrieval_relevance_floor,
        regulation_code=regulation_code,
        ensure_scope_clause=ensure_scope_clause,
    )


class ClauseContext(BaseModel):
    clause: Clause
    preceding: Clause | None = None
    following: Clause | None = None


async def fetch_clause(session: AsyncSession, clause_id: str) -> ClauseContext | None:
    """Full clause text plus its immediate neighbors in the source document, so
    the agent can read a citation in context rather than as an isolated snippet."""
    stmt = select(ClauseRow).where(ClauseRow.id == clause_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None

    preceding_row = (
        await session.execute(
            select(ClauseRow)
            .where(ClauseRow.document_id == row.document_id, ClauseRow.char_start < row.char_start)
            .order_by(ClauseRow.char_start.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    following_row = (
        await session.execute(
            select(ClauseRow)
            .where(ClauseRow.document_id == row.document_id, ClauseRow.char_start > row.char_start)
            .order_by(ClauseRow.char_start.asc())
            .limit(1)
        )
    ).scalar_one_or_none()

    return ClauseContext(
        clause=_row_to_clause(row),
        preceding=_row_to_clause(preceding_row) if preceding_row else None,
        following=_row_to_clause(following_row) if following_row else None,
    )


async def list_regulation_structure(session: AsyncSession, regulation_code: str) -> list[str]:
    """Table of contents for a regulation: its clause_paths in document order,
    for navigating to related clauses beyond what retrieval surfaced."""
    stmt = (
        select(ClauseRow.clause_path)
        .join(DocumentRow, ClauseRow.document_id == DocumentRow.id)
        .where(DocumentRow.regulation_code == regulation_code)
        .order_by(ClauseRow.char_start)
    )
    result = await session.execute(stmt)
    return [row[0] for row in result]


_EXTRACTION_SYSTEM_PROMPT = """You extract structured technical attributes from a \
supplier product specification. The spec may be in Chinese, English, Russian, or a \
mix. Only extract what is explicitly stated or directly and unambiguously implied — \
never guess or infer a plausible-sounding value for a field that isn't actually \
present in the text. Leave a field null (or an empty list) if it isn't present. \
An empty/near-empty result is the correct output for a spec that doesn't contain \
enough information — do not fabricate values to avoid returning an empty result."""


def extract_product_attributes(settings: Settings, spec_text: str) -> ProductAttributes:
    """LLM-based structured extraction from raw supplier spec text."""
    return call_structured(
        settings,
        _EXTRACTION_SYSTEM_PROMPT,
        spec_text,
        ProductAttributes,
    )


__all__ = [
    "ClauseContext",
    "extract_product_attributes",
    "fetch_clause",
    "list_regulation_structure",
    "search_regulations",
]
