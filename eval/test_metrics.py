"""Unit tests for the metric math itself — pure functions, no DB, no retrieval.
Integration-level correctness (does the metric reflect real retrieval quality)
is exercised by scripts/run_eval.py against the live corpus instead."""

from eval.gold_case import ExpectedClause, GoldCase
from eval.metrics import (
    CasePrediction,
    RetrievedClause,
    abstention_accuracy,
    citation_precision,
    false_positive_rate,
    regulation_set_f1,
    retrieval_recall_at_k,
)


def _case(**overrides: object) -> GoldCase:
    defaults: dict[str, object] = {
        "id": "case_x",
        "input_spec": "...",
        "language": "en",
    }
    defaults.update(overrides)
    return GoldCase.model_validate(defaults)


def _hit(regulation_code: str, clause_path: str = "Статья 1, пункт 1") -> RetrievedClause:
    return RetrievedClause(
        document_id=regulation_code, regulation_code=regulation_code, clause_path=clause_path
    )


def test_recall_at_k_full_when_all_expected_clauses_retrieved() -> None:
    case = _case(expected_clauses=[ExpectedClause(doc="d1", clause="c1")])
    prediction = CasePrediction(
        case_id="case_x",
        retrieved=[RetrievedClause(document_id="d1", regulation_code="R1", clause_path="c1")],
        abstained=False,
    )
    assert retrieval_recall_at_k([(case, prediction)]) == 1.0


def test_recall_at_k_zero_when_expected_clause_missing() -> None:
    case = _case(expected_clauses=[ExpectedClause(doc="d1", clause="c1")])
    prediction = CasePrediction(case_id="case_x", retrieved=[], abstained=True)
    assert retrieval_recall_at_k([(case, prediction)]) == 0.0


def test_recall_at_k_ignores_cases_with_no_expected_clauses() -> None:
    case = _case(expect_abstain=True)
    prediction = CasePrediction(case_id="case_x", retrieved=[], abstained=True)
    assert retrieval_recall_at_k([(case, prediction)]) is None


def test_citation_precision_penalizes_irrelevant_hits() -> None:
    case = _case(expected_regulations=["R1"])
    prediction = CasePrediction(
        case_id="case_x", retrieved=[_hit("R1"), _hit("R2")], abstained=False
    )
    assert citation_precision([(case, prediction)]) == 0.5


def test_regulation_set_f1_perfect_match() -> None:
    case = _case(expected_regulations=["R1", "R2"])
    prediction = CasePrediction(
        case_id="case_x", retrieved=[_hit("R1"), _hit("R2")], abstained=False
    )
    assert regulation_set_f1([(case, prediction)]) == 1.0


def test_regulation_set_f1_partial_match() -> None:
    case = _case(expected_regulations=["R1", "R2"])
    prediction = CasePrediction(
        case_id="case_x", retrieved=[_hit("R1"), _hit("R3")], abstained=False
    )
    # precision = 1/2, recall = 1/2 -> f1 = 0.5
    assert regulation_set_f1([(case, prediction)]) == 0.5


def test_regulation_set_f1_excludes_abstain_cases() -> None:
    case = _case(expect_abstain=True)
    prediction = CasePrediction(case_id="case_x", retrieved=[], abstained=True)
    assert regulation_set_f1([(case, prediction)]) is None


def test_false_positive_rate_counts_forbidden_regulations() -> None:
    case = _case(must_not_return=["R1", "R2"])
    prediction = CasePrediction(case_id="case_x", retrieved=[_hit("R1")], abstained=False)
    assert false_positive_rate([(case, prediction)]) == 0.5


def test_false_positive_rate_none_when_no_forbidden_regulations_specified() -> None:
    case = _case()
    prediction = CasePrediction(case_id="case_x", retrieved=[], abstained=True)
    assert false_positive_rate([(case, prediction)]) is None


def test_abstention_accuracy_rewards_correct_decline() -> None:
    case = _case(expect_abstain=True)
    prediction = CasePrediction(case_id="case_x", retrieved=[], abstained=True)
    assert abstention_accuracy([(case, prediction)]) == 1.0


def test_abstention_accuracy_penalizes_guessing_when_it_should_decline() -> None:
    case = _case(expect_abstain=True)
    prediction = CasePrediction(case_id="case_x", retrieved=[_hit("R1")], abstained=False)
    assert abstention_accuracy([(case, prediction)]) == 0.0


def test_abstention_accuracy_none_when_no_abstain_cases() -> None:
    case = _case(expected_regulations=["R1"])
    prediction = CasePrediction(case_id="case_x", retrieved=[_hit("R1")], abstained=False)
    assert abstention_accuracy([(case, prediction)]) is None
