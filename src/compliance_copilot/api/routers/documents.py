from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.api.dependencies import get_session
from compliance_copilot.api.schemas import DocumentInventoryItem
from compliance_copilot.db.models import ClauseRow, DocumentRow

router = APIRouter()


@router.get("/documents", response_model=list[DocumentInventoryItem])
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> list[DocumentInventoryItem]:
    """Corpus inventory: every ingested regulation and how many clauses it has."""
    stmt = (
        select(DocumentRow, func.count(ClauseRow.id).label("clause_count"))
        .outerjoin(ClauseRow, ClauseRow.document_id == DocumentRow.id)
        .group_by(DocumentRow.id)
        .order_by(DocumentRow.regulation_code)
    )
    result = await session.execute(stmt)
    return [
        DocumentInventoryItem(
            id=document.id,
            title=document.title,
            regulation_code=document.regulation_code,
            language=document.language,
            publication_date=document.publication_date,
            source_url=document.source_url,
            clause_count=clause_count,
        )
        for document, clause_count in result
    ]
