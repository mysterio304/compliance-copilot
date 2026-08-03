import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.agent.llm import call_structured
from compliance_copilot.agent.models import (
    KNOWN_REGULATIONS,
    ApplicabilityAssessment,
    RegulationPlan,
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
            )
            retrieved[regulation_code] = hits
        return {"retrieved_by_regulation": retrieved}

    return retrieve_node


def make_assess_applicability_node(settings: Settings) -> NodeFn:
    async def assess_applicability_node(state: AgentState) -> dict[str, Any]:
        attrs = state["attributes"]
        retrieved_by_regulation = state["retrieved_by_regulation"]
        validation_errors = state.get("validation_errors") or []

        findings: list[ComplianceFinding] = []
        skipped: list[str] = []

        for regulation_code, hits in retrieved_by_regulation.items():
            if not hits:
                skipped.append(f"{regulation_code}: no evidence above the relevance floor.")
                continue

            allowed_ids = {hit.clause_id for hit in hits}
            clause_listing = "\n".join(
                f"- clause_id={hit.clause_id} | {hit.text[:500]}" for hit in hits
            )
            correction = (
                f"\n\nNote: a previous attempt failed validation: {'; '.join(validation_errors)}. "
                "Only cite clause_id values from the list below."
                if validation_errors
                else ""
            )

            def validate_citations(
                assessment: ApplicabilityAssessment, allowed: set[str] = allowed_ids
            ) -> None:
                unknown = set(assessment.cited_clause_ids) - allowed
                if unknown:
                    raise ValueError(
                        f"cited_clause_ids contains ids not in the provided candidates: {unknown}"
                    )

            assessment = await asyncio.to_thread(
                call_structured,
                settings,
                (
                    f"Decide whether {regulation_code} ({KNOWN_REGULATIONS[regulation_code]}) "
                    "applies to this product, given the product attributes and the candidate "
                    "clauses retrieved from that regulation below. A product described in the "
                    "attributes may bundle multiple physically distinct electrical components "
                    "(e.g. the device itself, a separate power adapter/charger/power supply, a "
                    "removable battery) — assess each component that is plausibly in scope, not "
                    "only the primary device, since a regulation can genuinely apply to one "
                    f"component even when it doesn't apply to another.{correction}\n\n"
                    f"Candidate clauses:\n{clause_listing}"
                ),
                attrs.model_dump_json(),
                ApplicabilityAssessment,
                extra_validate=validate_citations,
            )

            hits_by_id = {hit.clause_id: hit for hit in hits}
            citations = [
                Citation(
                    clause_id=clause_id,
                    document_id=hits_by_id[clause_id].document_id,
                    quoted_span=hits_by_id[clause_id].text,
                    relevance_score=hits_by_id[clause_id].dense_score,
                )
                for clause_id in assessment.cited_clause_ids
            ]
            findings.append(
                ComplianceFinding(
                    regulation_code=regulation_code,
                    applies=assessment.applies,
                    reasoning=assessment.reasoning,
                    conformity_assessment_route=assessment.conformity_assessment_route,
                    citations=citations,
                )
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
