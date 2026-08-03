from typing import TypedDict

from compliance_copilot.retrieval.hybrid import RetrievalHit
from compliance_copilot.schemas.product import ProductAttributes
from compliance_copilot.schemas.reports import ComplianceFinding, ComplianceReport


class AgentState(TypedDict, total=False):
    spec_text: str
    attributes: ProductAttributes
    planned_regulations: list[str]
    plan_reasoning: str
    retrieved_by_regulation: dict[str, list[RetrievalHit]]
    findings: list[ComplianceFinding]
    skipped_regulations: list[str]
    report: ComplianceReport
    validation_errors: list[str]
    validation_attempts: int
