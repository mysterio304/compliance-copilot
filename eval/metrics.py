"""The five metric definitions, shared by both evaluation harnesses.

Each metric is expressed against the `Prediction` protocol below rather than a
concrete type, because two different layers get scored with them:

- eval/run_eval.py (Phase 4, retrieval-only): predictions are the top-k
  hybrid_search hits for the whole input_spec, resolved to
  (regulation_code, clause_path) via the clauses table. Fast and free — no LLM
  in the loop — which is what makes it usable as the CI gate.
- eval/run_eval_agent.py (Phase 5+): predictions are what the agent actually
  concluded, with real Citation objects behind them.

Which layer you score changes what a number means, and two of these metrics are
only meaningful at the agent layer. Retrieval makes no applicability decision —
it returns the k nearest chunks and nothing else — so at that layer
`false_positive_rate` counts "a clause from a forbidden regulation was among the
top-k" (not an assertion that it applies), and `abstention_accuracy` can only
ask whether anything cleared the relevance floor, which measures topical
similarity rather than whether the spec carried enough information to decide.
Measured on the corpus, the two insufficient-information cases score *higher*
than a genuine product spec does, so no floor value separates them; the agent's
deterministic ProductAttributes.is_empty check is what actually implements
abstention. Read the retrieval-layer values for those two metrics as
diagnostics of the retrieval distribution, not as system behavior.

`citation_precision` is likewise a proxy at the retrieval layer — a retrieved
clause counts as "supporting the claim" if it comes from a regulation in the
case's expected_regulations list — and a direct measurement at the agent layer,
where real citations exist to score.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from eval.gold_case import GoldCase


@dataclass(frozen=True)
class RetrievedClause:
    document_id: str
    regulation_code: str
    clause_path: str


class Prediction(Protocol):
    """What the five metrics below need from a prediction, regardless of which
    layer produced it. The retrieval harness supplies CasePrediction; the agent
    harness supplies AgentPrediction (eval/agent_metrics.py), whose
    predicted_regulations comes from findings the agent actually asserted rather
    than from the regulations its citations happen to belong to. Keeping the
    metric definitions parameterized over this protocol is what lets both
    harnesses be scored by the same code instead of two copies that drift."""

    @property
    def retrieved(self) -> list[RetrievedClause]: ...

    @property
    def abstained(self) -> bool: ...

    @property
    def predicted_regulations(self) -> set[str]: ...


@dataclass(frozen=True)
class CasePrediction:
    case_id: str
    retrieved: list[RetrievedClause]
    abstained: bool

    @property
    def predicted_regulations(self) -> set[str]:
        return {clause.regulation_code for clause in self.retrieved}


CaseAndPrediction = tuple[GoldCase, Prediction]


def retrieval_recall_at_k(pairs: Sequence[CaseAndPrediction]) -> float | None:
    """Of the cases with at least one gold expected_clause, what fraction of
    those clauses (by document_id + clause_path) appear among the retrieved
    hits? Macro-averaged over cases; cases with no expected_clauses (e.g.
    abstention cases) don't participate — there's nothing to recall."""
    scores = []
    for case, prediction in pairs:
        if not case.expected_clauses:
            continue
        retrieved_keys = {(c.document_id, c.clause_path) for c in prediction.retrieved}
        found = sum(
            1
            for expected in case.expected_clauses
            if (expected.doc, expected.clause) in retrieved_keys
        )
        scores.append(found / len(case.expected_clauses))
    return sum(scores) / len(scores) if scores else None


def citation_precision(pairs: Sequence[CaseAndPrediction]) -> float | None:
    """Of the clauses retrieval actually returned, what fraction come from a
    regulation that genuinely applies (per expected_regulations)? See module
    docstring — this is a retrieval-only proxy for the citation-precision
    metric described in the brief, not a measurement of agent-generated
    citations (there is no agent yet)."""
    scores = []
    for case, prediction in pairs:
        if not prediction.retrieved:
            continue
        supported = sum(
            1 for c in prediction.retrieved if c.regulation_code in case.expected_regulations
        )
        scores.append(supported / len(prediction.retrieved))
    return sum(scores) / len(scores) if scores else None


def regulation_set_f1(pairs: Sequence[CaseAndPrediction]) -> float | None:
    """Standard set F1 between predicted and expected regulation codes,
    macro-averaged. Abstention cases (empty expected_regulations by design)
    are excluded — that outcome is measured by abstention_accuracy instead."""
    scores = []
    for case, prediction in pairs:
        if case.expect_abstain:
            continue
        expected = set(case.expected_regulations)
        predicted = prediction.predicted_regulations
        if not expected and not predicted:
            scores.append(1.0)
            continue
        if not predicted or not expected:
            scores.append(0.0)
            continue
        true_positives = len(expected & predicted)
        precision = true_positives / len(predicted)
        recall = true_positives / len(expected)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        scores.append(f1)
    return sum(scores) / len(scores) if scores else None


def false_positive_rate(pairs: Sequence[CaseAndPrediction]) -> float | None:
    """Across every (case, forbidden regulation) pair drawn from must_not_return,
    how often does the forbidden regulation show up in what was predicted?"""
    total = 0
    triggered = 0
    for case, prediction in pairs:
        for forbidden in case.must_not_return:
            total += 1
            if forbidden in prediction.predicted_regulations:
                triggered += 1
    return triggered / total if total else None


def abstention_accuracy(pairs: Sequence[CaseAndPrediction]) -> float | None:
    """Of the cases gold-labeled as insufficient-information, how many did
    retrieval actually decline (nothing cleared the relevance floor)?"""
    abstain_cases = [(case, prediction) for case, prediction in pairs if case.expect_abstain]
    if not abstain_cases:
        return None
    correct = sum(1 for _case, prediction in abstain_cases if prediction.abstained)
    return correct / len(abstain_cases)


def compute_all_metrics(pairs: Sequence[CaseAndPrediction]) -> dict[str, float | None]:
    return {
        "retrieval_recall_at_k": retrieval_recall_at_k(pairs),
        "citation_precision": citation_precision(pairs),
        "regulation_set_f1": regulation_set_f1(pairs),
        "false_positive_rate": false_positive_rate(pairs),
        "abstention_accuracy": abstention_accuracy(pairs),
    }
