"""Agent-level prediction shape for the five metrics in eval/metrics.py.

The retrieval harness (eval/run_eval.py) scores what hybrid_search surfaces.
That is the right thing to measure for recall and for retrieval tuning, but it
is the wrong layer for abstention_accuracy and false_positive_rate: retrieval
never decides that a regulation applies, it only returns the top-k chunks
nearest the query, so a clause from a must_not_return regulation appearing in
that list is not the system asserting anything. Measured against the agent —
which does decide — the same two cases that score as false positives at the
retrieval layer come back clean.

Hence this module: same five metric definitions, fed by what the agent actually
concluded. Two mappings matter here and are deliberate:

- predicted_regulations comes from findings with applies=True, not from the
  regulations the citations belong to. A finding that says a regulation does
  *not* apply still carries citations (the schema requires them — that's the
  point of the citation constraint), and counting those as predictions would
  score a correct negative verdict as a false positive.
- abstained means the agent asserted no applicable regulation. That is the
  structural form the brief's "the system says I don't know" takes here: empty
  findings plus a populated unresolved_questions, which is exactly what
  ProductAttributes.is_empty produces upstream.
"""

from dataclasses import dataclass

from compliance_copilot.schemas.reports import ComplianceReport
from eval.metrics import RetrievedClause

# clause_id -> (document_id, regulation_code, clause_path)
ClauseIndex = dict[str, tuple[str, str, str]]


@dataclass(frozen=True)
class AgentPrediction:
    case_id: str
    retrieved: list[RetrievedClause]
    abstained: bool
    asserted_regulations: frozenset[str]

    @property
    def predicted_regulations(self) -> set[str]:
        return set(self.asserted_regulations)


def prediction_from_report(
    case_id: str, report: ComplianceReport, clause_index: ClauseIndex
) -> AgentPrediction:
    """Project a ComplianceReport onto the shape the metrics consume.

    Only citations attached to applies=True findings become `retrieved` — the
    metrics read that list as "the clauses the system stands behind," and
    citations supporting a negative verdict are not that.
    """
    applied = [finding for finding in report.findings if finding.applies]
    cited_clause_ids = [citation.clause_id for finding in applied for citation in finding.citations]
    # Order-preserving dedupe: one clause cited by two findings shouldn't count twice.
    seen: set[str] = set()
    retrieved = []
    for clause_id in cited_clause_ids:
        if clause_id in seen or clause_id not in clause_index:
            continue
        seen.add(clause_id)
        document_id, regulation_code, clause_path = clause_index[clause_id]
        retrieved.append(
            RetrievedClause(
                document_id=document_id,
                regulation_code=regulation_code,
                clause_path=clause_path,
            )
        )
    return AgentPrediction(
        case_id=case_id,
        retrieved=retrieved,
        abstained=not applied,
        asserted_regulations=frozenset(finding.regulation_code for finding in applied),
    )
