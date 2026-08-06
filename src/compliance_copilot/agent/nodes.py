import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.agent.llm import call_structured
from compliance_copilot.agent.models import (
    KNOWN_REGULATIONS,
    ExclusionAssessment,
    RegulationPlan,
    ScopeAssessment,
)
from compliance_copilot.agent.state import AgentState
from compliance_copilot.agent.tools import extract_product_attributes, search_regulations
from compliance_copilot.config import Settings
from compliance_copilot.db.models import ClauseRow
from compliance_copilot.retrieval.hybrid import RetrievalHit
from compliance_copilot.schemas.citations import Citation
from compliance_copilot.schemas.product import ProductAttributes
from compliance_copilot.schemas.reports import ComplianceFinding, ComplianceReport

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]

# How much of a candidate clause the applicability assessor gets to read. This
# was 500, which silently truncated 21% of the corpus (145 of 685 chunks) and
# produced wrong verdicts rather than degraded ones: ТР ЕАЭС 037/2016's
# exclusion list is one 1290-character clause enumerating а)–к), so at 500 the
# model saw the two exclusions that fit and none of the eight that didn't. It
# then affirmed applicability for electric toys, used goods, medical devices,
# measuring instruments, photovoltaics and vehicle equipment — every one of
# which that clause excludes by name, in text it was never shown. Sized against
# the measured distribution instead of a round number: p99 clause length is
# 2255 characters, so truncation is now the rare exception it was assumed to be.
MAX_CLAUSE_CHARS_FOR_ASSESSMENT = 2500


def _summarize_attributes(attrs: ProductAttributes) -> str:
    if attrs.is_empty:
        base = attrs.product_name or "Unnamed product"
        return f"{base} — insufficient technical detail provided by the supplier."
    parts = [attrs.product_name or "Product"]
    if attrs.voltage:
        parts.append(f"voltage: {attrs.voltage}")
    if attrs.frequency:
        parts.append(f"frequency: {attrs.frequency}")
    if attrs.materials:
        parts.append(f"materials: {', '.join(attrs.materials)}")
    if attrs.wireless_bands:
        parts.append(f"wireless: {', '.join(attrs.wireless_bands)}")
    if attrs.intended_user:
        parts.append(f"intended user: {attrs.intended_user}")
    return "; ".join(parts)


def _overall_confidence(findings: list[ComplianceFinding]) -> float:
    if not findings:
        return 0.0
    scores = [citation.relevance_score for finding in findings for citation in finding.citations]
    return sum(scores) / len(scores) if scores else 0.0


def make_extract_attributes_node(settings: Settings) -> NodeFn:
    async def extract_attributes_node(state: AgentState) -> dict[str, Any]:
        attrs = await asyncio.to_thread(extract_product_attributes, settings, state["spec_text"])
        return {"attributes": attrs}

    return extract_attributes_node


def make_plan_node(settings: Settings) -> NodeFn:
    async def plan_node(state: AgentState) -> dict[str, Any]:
        attrs = state["attributes"]
        if attrs.is_empty:
            return {
                "planned_regulations": [],
                "plan_reasoning": "Product attributes could not be extracted from the spec.",
            }

        known_list = "\n".join(f"- {code}: {desc}" for code, desc in KNOWN_REGULATIONS.items())
        known_codes = set(KNOWN_REGULATIONS)

        def validate_plan(plan: RegulationPlan) -> None:
            unknown = set(plan.candidate_regulations) - known_codes
            if unknown:
                raise ValueError(f"unknown regulation codes: {unknown}. Use only listed codes.")

        plan = await asyncio.to_thread(
            call_structured,
            settings,
            (
                "Given the product attributes below, decide which of these EAEU "
                f"technical regulations are plausibly relevant to investigate:\n{known_list}\n\n"
                "Include a regulation only if there is a real reason to check it — do "
                "not include one just because the product is a manufactured item. "
                "Return an empty list if none plausibly apply."
            ),
            attrs.model_dump_json(),
            RegulationPlan,
            extra_validate=validate_plan,
        )
        return {
            "planned_regulations": plan.candidate_regulations,
            "plan_reasoning": plan.reasoning,
        }

    return plan_node


def make_retrieve_node(settings: Settings, session: AsyncSession) -> NodeFn:
    async def retrieve_node(state: AgentState) -> dict[str, Any]:
        # Query with the extracted attributes, not the raw spec_text. Found via
        # live testing, not assumed: embedding a long prose spec as one query
        # dilutes a specific fact (e.g. "contains lead") across everything else
        # in the text, sometimes losing it entirely from the top-k where the
        # bare fact alone would have surfaced it. The compact structured
        # attributes carry the same substance with far less to dilute it —
        # verified to match or beat spec_text retrieval on every case checked,
        # including ones that already worked.
        query = state["attributes"].model_dump_json()
        retrieved: dict[str, list[RetrievalHit]] = {}
        for regulation_code in state["planned_regulations"]:
            hits = await search_regulations(
                session,
                settings,
                query,
                "auto",
                settings.retrieval_top_k,
                regulation_code=regulation_code,
                # The next node is asked whether this regulation applies, and the
                # clause defining its scope is what settles that — so it must be
                # in evidence rather than depending on it placing in the top_k.
                ensure_scope_clause=True,
            )
            retrieved[regulation_code] = hits
        return {"retrieved_by_regulation": retrieved}

    return retrieve_node


def _clause_listing(hits: list[RetrievalHit]) -> str:
    return "\n".join(
        f"- clause_id={hit.clause_id} | {hit.text[:MAX_CLAUSE_CHARS_FOR_ASSESSMENT]}"
        for hit in hits
    )


def _citations_for(hits: list[RetrievalHit], clause_ids: list[str]) -> list[Citation]:
    hits_by_id = {hit.clause_id: hit for hit in hits}
    return [
        Citation(
            clause_id=clause_id,
            document_id=hits_by_id[clause_id].document_id,
            quoted_span=hits_by_id[clause_id].text,
            relevance_score=hits_by_id[clause_id].dense_score,
        )
        for clause_id in clause_ids
    ]


def _check_cited_ids(cited: list[str], allowed: set[str]) -> None:
    unknown = set(cited) - allowed
    if unknown:
        raise ValueError(f"cited_clause_ids contains ids not in the provided candidates: {unknown}")


async def _majority_verdict[T](
    make_call: Callable[[], Awaitable[T]], verdict_of: Callable[[T], bool]
) -> T:
    """Best of three on the boolean each assessment turns on, returning the
    winning response whole so its reasoning and citations stay consistent with
    the verdict reported.

    `temperature=0` does not make DeepSeek deterministic — it is MoE, and
    provider-side routing varies with batching. Measured over three identical
    runs of the 22 gold cases, 2 of them changed answer, both on borderline
    verdicts. Identical input reaching different conclusions is a defect in a
    compliance system regardless of which answer is right, so the assessor's
    two questions are decided by vote rather than by a single sample.

    Only two calls are made up front: if they agree they are the majority
    already, whatever a third would have said. The third is a tiebreak, so the
    usual cost is 2 calls, not 3. Both run concurrently, keeping latency at
    roughly one call.

    One call failing validation no longer sinks the assessment either — the
    surviving response is used, unvoted. That is strictly better than the
    single-call version, which had nothing to fall back on.
    """
    outcomes = await asyncio.gather(make_call(), make_call(), return_exceptions=True)
    succeeded = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    if not succeeded:
        raise next(outcome for outcome in outcomes if isinstance(outcome, BaseException))
    if len(succeeded) == 1 or verdict_of(succeeded[0]) == verdict_of(succeeded[1]):
        return succeeded[0]

    tiebreak = await make_call()
    return succeeded[0] if verdict_of(tiebreak) == verdict_of(succeeded[0]) else succeeded[1]


def _scope_prompt(regulation_code: str, clause_listing: str, correction: str) -> str:
    return (
        f"Question 1 of 2 about {regulation_code} ({KNOWN_REGULATIONS[regulation_code]}).\n\n"
        "Does this regulation's stated scope reach this product at all? Consider only what the "
        "regulation says it COVERS — its 'распространяется на' clause and any product list "
        "(приложение / перечень) among the clauses below. Ignore exclusions completely: a "
        "separate question handles those, so do not answer in_scope=false because something "
        "looks carved out.\n\n"
        "A product may bundle several physically distinct components (the device itself, a "
        "separate mains adapter/charger/power supply, a removable battery). If the regulation "
        "reaches any one of them, it reaches the product — so a low-voltage device is in scope "
        "through its mains adapter when the attributes say one is included. Judge only the "
        "components the attributes actually state: do not assume an unstated mains adapter for "
        "something described as battery-powered, PoE-powered, or supplied without one.\n\n"
        "Cite the clause(s) your answer rests on. Keep reasoning to 2-3 sentences."
        f"{correction}\n\nCandidate clauses:\n{clause_listing}"
    )


def _exclusion_prompt(regulation_code: str, clause_listing: str, correction: str) -> str:
    return (
        f"Question 2 of 2 about {regulation_code} ({KNOWN_REGULATIONS[regulation_code]}). That "
        "the regulation's positive scope reaches this product is already settled — do not "
        "revisit it.\n\n"
        "Is the product carved back out by one of this regulation's named exclusions? At least "
        "one clause below says what the regulation does NOT cover ('не распространяется на'), "
        "usually as a single long clause listing unrelated categories а), б), в), … Read that "
        "list to its end: the category that matches, if any, can be anywhere in it.\n\n"
        "Exclusions are narrow. Answer excluded=true only when the product clearly falls into a "
        "listed category, matching on substance rather than wording — the spec and the clause "
        "describe the same thing differently and in different languages ('refurbished' is "
        "'бывшие в употреблении'; a part 'for installation in vehicles' is equipment 'для "
        "использования на … транспорте'). If nothing clearly matches, answer excluded=false: a "
        "partial or arguable resemblance is not an exclusion.\n\n"
        "Cite the exclusion clause you checked against. Keep reasoning to 2-3 sentences."
        f"{correction}\n\nCandidate clauses:\n{clause_listing}"
    )


async def _assess_one_regulation(
    settings: Settings,
    attrs: ProductAttributes,
    regulation_code: str,
    hits: list[RetrievalHit],
    correction: str,
) -> ComplianceFinding:
    """Applicability as two one-directional questions combined in code, rather
    than one prompt asked to weigh inclusion and exclusion against each other —
    see ExclusionAssessment for why that balance could not be struck in a single
    prompt. The exclusion question is skipped when the positive scope doesn't
    reach the product: the verdict is already settled and the call would be
    asking whether something outside the scope is also carved out of it."""
    allowed_ids = {hit.clause_id for hit in hits}
    listing = _clause_listing(hits)
    attributes_json = attrs.model_dump_json()

    def check_scope(assessment: ScopeAssessment) -> None:
        _check_cited_ids(assessment.cited_clause_ids, allowed_ids)

    def check_exclusion(assessment: ExclusionAssessment) -> None:
        _check_cited_ids(assessment.cited_clause_ids, allowed_ids)

    async def ask_scope() -> ScopeAssessment:
        return await asyncio.to_thread(
            call_structured,
            settings,
            _scope_prompt(regulation_code, listing, correction),
            attributes_json,
            ScopeAssessment,
            extra_validate=check_scope,
        )

    async def ask_exclusion() -> ExclusionAssessment:
        return await asyncio.to_thread(
            call_structured,
            settings,
            _exclusion_prompt(regulation_code, listing, correction),
            attributes_json,
            ExclusionAssessment,
            extra_validate=check_exclusion,
        )

    scope = await _majority_verdict(ask_scope, lambda assessment: assessment.in_scope)
    if not scope.in_scope:
        return ComplianceFinding(
            regulation_code=regulation_code,
            applies=False,
            reasoning=scope.reasoning,
            citations=_citations_for(hits, scope.cited_clause_ids),
        )

    exclusion = await _majority_verdict(ask_exclusion, lambda assessment: assessment.excluded)
    if exclusion.excluded:
        return ComplianceFinding(
            regulation_code=regulation_code,
            applies=False,
            # Cite the exclusion, not the scope clause: the exclusion is what
            # decided this verdict, and it's what a reader needs to check.
            reasoning=f"Within the regulation's stated scope, but excluded: {exclusion.reasoning}",
            citations=_citations_for(hits, exclusion.cited_clause_ids),
        )

    return ComplianceFinding(
        regulation_code=regulation_code,
        applies=True,
        reasoning=f"{scope.reasoning} No exclusion applies: {exclusion.reasoning}",
        conformity_assessment_route=scope.conformity_assessment_route,
        citations=_citations_for(hits, scope.cited_clause_ids),
    )


def make_assess_applicability_node(settings: Settings) -> NodeFn:
    async def assess_applicability_node(state: AgentState) -> dict[str, Any]:
        attrs = state["attributes"]
        validation_errors = state.get("validation_errors") or []
        correction = (
            f"\n\nNote: a previous attempt failed validation: {'; '.join(validation_errors)}. "
            "Only cite clause_id values from the candidate list."
            if validation_errors
            else ""
        )

        findings: list[ComplianceFinding] = []
        skipped: list[str] = []

        for regulation_code, hits in state["retrieved_by_regulation"].items():
            if not hits:
                skipped.append(f"{regulation_code}: no evidence above the relevance floor.")
                continue
            findings.append(
                await _assess_one_regulation(settings, attrs, regulation_code, hits, correction)
            )

        return {"findings": findings, "skipped_regulations": skipped}

    return assess_applicability_node


async def assemble_report_node(state: AgentState) -> dict[str, Any]:
    attrs = state["attributes"]
    findings = state["findings"]
    unresolved = list(state.get("skipped_regulations") or [])
    if attrs.is_empty:
        unresolved.insert(
            0,
            "Supplier specification did not contain enough technical detail to "
            "determine applicable regulations.",
        )

    report = ComplianceReport(
        product_summary=_summarize_attributes(attrs),
        findings=findings,
        unresolved_questions=unresolved,
        overall_confidence=_overall_confidence(findings),
    )
    return {"report": report}


def make_validate_node(session: AsyncSession) -> NodeFn:
    async def validate_node(state: AgentState) -> dict[str, Any]:
        report = state["report"]
        errors: list[str] = []
        for finding in report.findings:
            for citation in finding.citations:
                stmt = select(ClauseRow.id).where(ClauseRow.id == citation.clause_id)
                exists = (await session.execute(stmt)).first()
                if exists is None:
                    errors.append(
                        f"{finding.regulation_code}: cited clause_id "
                        f"'{citation.clause_id}' does not exist in the database"
                    )
        attempts = state.get("validation_attempts", 0) + 1
        return {"validation_errors": errors, "validation_attempts": attempts}

    return validate_node


async def finalize_partial_report_node(state: AgentState) -> dict[str, Any]:
    """Second validation failure: drop findings with invalid citations and say
    so explicitly, rather than silently returning a report that looks complete."""
    report = state["report"]
    errors = state["validation_errors"]
    bad_regulation_codes = {error.split(":", 1)[0] for error in errors}

    kept_findings = [f for f in report.findings if f.regulation_code not in bad_regulation_codes]
    unresolved = [
        *report.unresolved_questions,
        "Validation failed twice; the following findings were dropped rather than "
        f"reported with unverifiable citations: {'; '.join(errors)}",
    ]
    partial_report = ComplianceReport(
        product_summary=report.product_summary,
        findings=kept_findings,
        unresolved_questions=unresolved,
        overall_confidence=_overall_confidence(kept_findings),
    )
    return {"report": partial_report}
