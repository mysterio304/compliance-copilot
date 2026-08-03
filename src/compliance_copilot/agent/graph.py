from typing import Any

from langfuse import observe
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.agent.nodes import (
    assemble_report_node,
    finalize_partial_report_node,
    make_assess_applicability_node,
    make_extract_attributes_node,
    make_plan_node,
    make_retrieve_node,
    make_validate_node,
)
from compliance_copilot.agent.state import AgentState
from compliance_copilot.config import Settings

# "Route back once with the error; on second failure, return a partial report" —
# one retry means validate_node's second pass (attempt count reaches 2) is the
# one that finalizes instead of retrying again.
MAX_VALIDATION_ATTEMPTS = 2


def _route_after_validate(state: AgentState) -> str:
    if not state.get("validation_errors"):
        return "done"
    if state.get("validation_attempts", 0) >= MAX_VALIDATION_ATTEMPTS:
        return "finalize_partial"
    return "retry"


def build_graph(settings: Settings, session: AsyncSession) -> Any:
    """extract product attributes -> plan regulation families -> retrieve per
    family -> assess applicability -> assemble report -> validate, with
    validate routing back to assess_applicability once on failure and
    finalizing a partial (explicitly flagged) report on a second failure."""
    # LangGraph's add_node overloads don't resolve cleanly under mypy --strict for
    # a `TypedDict, total=False` state with plain-dict partial-update returns —
    # a known friction point, not a real type-safety gap here (every node's
    # return shape is asserted by its NodeFn signature in nodes.py already).
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("extract_attributes", make_extract_attributes_node(settings))  # type: ignore[call-overload]
    graph.add_node("plan", make_plan_node(settings))  # type: ignore[call-overload]
    graph.add_node("retrieve", make_retrieve_node(settings, session))  # type: ignore[call-overload]
    graph.add_node("assess_applicability", make_assess_applicability_node(settings))  # type: ignore[call-overload]
    graph.add_node("assemble_report", assemble_report_node)
    graph.add_node("validate", make_validate_node(session))  # type: ignore[arg-type]
    graph.add_node("finalize_partial", finalize_partial_report_node)

    graph.set_entry_point("extract_attributes")
    graph.add_edge("extract_attributes", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "assess_applicability")
    graph.add_edge("assess_applicability", "assemble_report")
    graph.add_edge("assemble_report", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"done": END, "retry": "assess_applicability", "finalize_partial": "finalize_partial"},
    )
    graph.add_edge("finalize_partial", END)

    return graph.compile()


@observe(name="compliance_agent_run")
async def run_agent(settings: Settings, session: AsyncSession, spec_text: str) -> AgentState:
    """Root of the trace tree: every call_structured generation made by any node
    during this run nests under this span automatically via Langfuse's context
    propagation, so a full run — extraction, planning, every per-regulation
    applicability call — is inspectable as one trace, not scattered spans."""
    compiled = build_graph(settings, session)
    result: AgentState = await compiled.ainvoke({"spec_text": spec_text})
    return result
