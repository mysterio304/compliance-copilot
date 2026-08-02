from pydantic import BaseModel, ConfigDict, Field

from compliance_copilot.schemas.citations import Citation


class ComplianceFinding(BaseModel):
    """A single regulation's applicability verdict. Uncited claims are unrepresentable:
    citations is non-empty by construction, not by convention."""

    model_config = ConfigDict(frozen=True)

    regulation_code: str
    applies: bool
    reasoning: str
    conformity_assessment_route: str | None = None
    citations: list[Citation] = Field(min_length=1)


class ComplianceReport(BaseModel):
    """The full answer returned to the importer. unresolved_questions carries anything
    retrieval couldn't resolve — the agent's "I don't know", made structured instead of prose."""

    model_config = ConfigDict(frozen=True)

    product_summary: str
    findings: list[ComplianceFinding]
    unresolved_questions: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
