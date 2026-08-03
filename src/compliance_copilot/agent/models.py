"""LLM input/output models internal to the agent graph — distinct from
schemas/, which holds the domain models shared across layers (API, DB,
eval). These only ever exist transiently inside a graph run."""

from pydantic import BaseModel, Field

KNOWN_REGULATIONS: dict[str, str] = {
    "ТР ТС 004/2011": (
        "safety of low-voltage equipment — electrical devices operating at "
        "50-1000V AC or 75-1500V DC"
    ),
    "ТР ТС 020/2011": (
        "electromagnetic compatibility — technical means that create or are "
        "susceptible to electromagnetic interference"
    ),
    "ТР ЕАЭС 037/2016": (
        "restriction of hazardous substances (RoHS) in electrical and "
        "electronic products"
    ),
    "ТР ТС 010/2011": "safety of machines and/or equipment",
    "ТР ТС 008/2011": (
        "safety of toys — applies to ALL products intended for children's play, "
        "regardless of whether they have any electrical, electronic, or mechanical "
        "component. Covers physical, mechanical, chemical, flammability, and "
        "choking hazards as much as electrical ones — a plush toy or wooden block "
        "with zero electronics is squarely in scope, not exempt"
    ),
}


class RegulationPlan(BaseModel):
    candidate_regulations: list[str] = Field(
        description="Subset of the known regulation codes plausibly relevant to this product"
    )
    reasoning: str


class ApplicabilityAssessment(BaseModel):
    applies: bool
    reasoning: str
    conformity_assessment_route: str | None = Field(
        default=None, description='e.g. "declaration_of_conformity", "certification"'
    )
    cited_clause_ids: list[str] = Field(
        min_length=1,
        description="clause_id values from the candidate clauses that support this conclusion",
    )
