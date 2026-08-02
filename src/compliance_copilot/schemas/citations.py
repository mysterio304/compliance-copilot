from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Points a claim back to the exact clause and quoted span that supports it."""

    model_config = ConfigDict(frozen=True)

    clause_id: str
    document_id: str
    quoted_span: str
    relevance_score: float = Field(ge=0.0, le=1.0)
