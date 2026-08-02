"""Unit tests for the regression-detection logic in run_eval.py — pure function,
no DB needed."""

from eval.run_eval import check_regressions


def test_no_regression_when_metrics_match_baseline() -> None:
    metrics = {"retrieval_recall_at_k": 0.5, "false_positive_rate": 0.2}
    baseline = {"retrieval_recall_at_k": 0.5, "false_positive_rate": 0.2}
    assert check_regressions(metrics, baseline, tolerance=0.05) == []


def test_regression_flagged_when_higher_is_better_metric_drops() -> None:
    metrics = {"retrieval_recall_at_k": 0.3}
    baseline = {"retrieval_recall_at_k": 0.5}
    regressions = check_regressions(metrics, baseline, tolerance=0.05)
    assert len(regressions) == 1
    assert "retrieval_recall_at_k" in regressions[0]
    assert "dropped" in regressions[0]


def test_no_regression_when_drop_is_within_tolerance() -> None:
    metrics = {"retrieval_recall_at_k": 0.48}
    baseline = {"retrieval_recall_at_k": 0.5}
    assert check_regressions(metrics, baseline, tolerance=0.05) == []


def test_regression_flagged_when_false_positive_rate_rises() -> None:
    # false_positive_rate is "lower is better" — a rise is the regression direction.
    metrics = {"false_positive_rate": 0.4}
    baseline = {"false_positive_rate": 0.1}
    regressions = check_regressions(metrics, baseline, tolerance=0.05)
    assert len(regressions) == 1
    assert "rose" in regressions[0]


def test_no_regression_when_metric_improves() -> None:
    metrics = {"retrieval_recall_at_k": 0.9, "false_positive_rate": 0.0}
    baseline = {"retrieval_recall_at_k": 0.5, "false_positive_rate": 0.3}
    assert check_regressions(metrics, baseline, tolerance=0.05) == []


def test_none_values_are_skipped() -> None:
    metrics = {"abstention_accuracy": None}
    baseline = {"abstention_accuracy": 0.8}
    assert check_regressions(metrics, baseline, tolerance=0.05) == []
