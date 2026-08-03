from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.agent.tools import ClauseContext, fetch_clause
from compliance_copilot.api.dependencies import get_session

router = APIRouter()


@router.get("/clauses/{clause_id}", response_model=ClauseContext)
async def resolve_clause(
    clause_id: str, session: AsyncSession = Depends(get_session)
) -> ClauseContext:
    """Resolves a citation's clause_id to its full text and document-order
    neighbors, so a reader can verify any claim in a ComplianceReport."""
    context = await fetch_clause(session, clause_id)
    if context is None:
        raise HTTPException(status_code=404, detail=f"Clause '{clause_id}' not found")
    return context
