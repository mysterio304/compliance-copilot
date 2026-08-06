"""LLM input/output models internal to the agent graph — distinct from
schemas/, which holds the domain models shared across layers (API, DB,
eval). These only ever exist transiently inside a graph run."""

from pydantic import BaseModel, Field

KNOWN_REGULATIONS: dict[str, str] = {
    "ТР ТС 004/2011": (
        "safety of low-voltage equipment — electrical devices operating at "
        "50-1000V AC or 75-1500V DC. IMPORTANT: assess the voltage of every "
        "electrical component the product ships with, not just the end device's "
        "own operating voltage — a bundled power adapter/charger/power supply "
        "that steps mains AC (e.g. 100-240V AC) down to the device's low output "
        "voltage (e.g. 5V DC) is itself equipment operating in the regulated "
        "range and is in scope, even when the end device's own output voltage "
        "is well below it. Do not conclude 'not applicable' by looking only at "
        "the device's output voltage when a mains-connected adapter is present."
    ),
    "ТР ТС 020/2011": (
        "electromagnetic compatibility — technical means that create or are "
        "susceptible to electromagnetic interference"
    ),
    "ТР ЕАЭС 037/2016": (
        "restriction of hazardous substances (RoHS) in electrical and electronic products"
    ),
    "ТР ТС 010/2011": (
        "safety of machines and/or equipment — applies regardless of power "
        "source. A machine does NOT need a motor, engine, or any electrical "
        "drive to be in scope: hand-operated and purely mechanical equipment "
        "with moving parts (cranks, gears, burrs, blades, presses) is squarely "
        "covered. Do not conclude 'not applicable' merely because the product "
        "is manual/unpowered"
    ),
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


class ScopeAssessment(BaseModel):
    """First half of the applicability question: does the regulation's stated
    scope reach this product at all?"""

    in_scope: bool
    reasoning: str
    conformity_assessment_route: str | None = Field(
        default=None, description='e.g. "declaration_of_conformity", "certification"'
    )
    cited_clause_ids: list[str] = Field(
        min_length=1,
        description="clause_id values from the candidate clauses that support this conclusion",
    )


class ExclusionAssessment(BaseModel):
    """Second half: is a product the positive scope reaches nonetheless carved
    back out by one of the regulation's named exclusions?

    Split from the scope question rather than folded into it because a single
    prompt asking both at once cannot be balanced. Phrasing it to look for
    reasons a regulation applies made the agent miss exclusions entirely
    (electric toys, used goods, medical devices were all affirmed under
    ТР ЕАЭС 037/2016, which excludes each by name); rewriting the same prompt to
    look for exclusions swung it the other way, dropping the regulation from
    products it genuinely covers. Two one-directional questions combined in code
    (`applies = in_scope and not excluded`) have no such balance to lose.
    """

    excluded: bool
    reasoning: str
    cited_clause_ids: list[str] = Field(
        min_length=1,
        description="clause_id values of the exclusion clause(s) this conclusion rests on",
    )
