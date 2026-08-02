#!/usr/bin/env python3
"""Retrieval-only evaluation harness (Phase 4 — built before the agent exists).

For each gold case, runs hybrid_search directly against the whole input_spec
text (no LLM query planning yet — that's Phase 5), resolves hits to
(regulation_code, clause_path), scores against the 5 metrics, prints a table,
writes eval/results/latest.json, and compares against eval/results/baseline.json
with a configurable tolerance. Exits non-zero if any metric regresses beyond
that tolerance.

Usage: uv run python eval/run_eval.py [--cases-dir DIR] [--tolerance 0.05] [--update-baseline]
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from compliance_copilot.config import Settings, get_settings  # noqa: E402
from compliance_copilot.db.models import ChunkRow, ClauseRow, DocumentRow  # noqa: E402
from compliance_copilot.db.session import make_engine, make_session_factory  # noqa: E402
from compliance_copilot.retrieval.hybrid import hybrid_search  # noqa: E402
from eval.gold_case import GoldCase, load_cases  # noqa: E402
from eval.metrics import CasePrediction, RetrievedClause, compute_all_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# False-positive rate is the only "lower is better" metric here.
HIGHER_IS_BETTER = {
    "retrieval_recall_at_k": True,
    "citation_precision": True,
    "regulation_set_f1": True,
    "false_positive_rate": False,
    "abstention_accuracy": True,
}


async def _resolve_clauses(
    session: AsyncSession, clause_ids: list[str]
) -> dict[str, tuple[str, str, str]]:
    """clause_id -> (document_id, regulation_code, clause_path)."""
    if not clause_ids:
        return {}
    stmt = (
        select(
            ClauseRow.id,
            ClauseRow.document_id,
            ClauseRow.clause_path,
            DocumentRow.regulation_code,
        )
        .join(DocumentRow, ClauseRow.document_id == DocumentRow.id)
        .where(ClauseRow.id.in_(clause_ids))
    )
    result = await session.execute(stmt)
    return {row.id: (row.document_id, row.regulation_code, row.clause_path) for row in result}


async def run_case(session: AsyncSession, case: GoldCase, settings: Settings) -> CasePrediction:
    hits = await hybrid_search(
        session,
        case.input_spec,
        settings.embedding_model_name,
        settings.retrieval_top_k,
        settings.retrieval_fusion_weight,
        settings.retrieval_relevance_floor,
    )
    if not hits:
        return CasePrediction(case_id=case.id, retrieved=[], abstained=True)

    resolved = await _resolve_clauses(session, [hit.clause_id for hit in hits])
    retrieved = [
        RetrievedClause(
            document_id=resolved[hit.clause_id][0],
            regulation_code=resolved[hit.clause_id][1],
            clause_path=resolved[hit.clause_id][2],
        )
        for hit in hits
        if hit.clause_id in resolved
    ]
    return CasePrediction(case_id=case.id, retrieved=retrieved, abstained=False)


def print_metrics_table(
    metrics: Mapping[str, float | None], baseline: Mapping[str, float] | None
) -> None:
    print(f"{'Metric':<28}{'Value':>10}{'Baseline':>12}{'Δ':>10}")
    for name, value in metrics.items():
        value_str = f"{value:.3f}" if value is not None else "N/A"
        if baseline and name in baseline and value is not None:
            base_value = baseline[name]
            delta = value - base_value
            print(f"{name:<28}{value_str:>10}{base_value:>12.3f}{delta:>+10.3f}")
        else:
            print(f"{name:<28}{value_str:>10}{'-':>12}{'-':>10}")


def check_regressions(
    metrics: Mapping[str, float | None], baseline: Mapping[str, float], tolerance: float
) -> list[str]:
    regressions = []
    for name, value in metrics.items():
        if value is None or name not in baseline:
            continue
        base_value = baseline[name]
        higher_is_better = HIGHER_IS_BETTER[name]
        delta = value - base_value if higher_is_better else base_value - value
        if delta < -tolerance:
            direction = "dropped" if higher_is_better else "rose"
            regressions.append(
                f"{name} {direction} from {base_value:.3f} to {value:.3f} "
                f"(tolerance {tolerance:.3f})"
            )
    return regressions


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    cases = load_cases(args.cases_dir)

    async with session_factory() as session:
        count = (await session.execute(select(ChunkRow.id).limit(1))).first()
        if count is None:
            print(
                "No embedded chunks in the database — run scripts/ingest_corpus.py "
                "and scripts/load_corpus.py first.",
                file=sys.stderr,
            )
            return 1

        predictions = [await run_case(session, case, settings) for case in cases]

    await engine.dispose()

    pairs = list(zip(cases, predictions, strict=True))
    metrics = compute_all_metrics(pairs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = RESULTS_DIR / "baseline.json"
    latest_path = RESULTS_DIR / "latest.json"

    baseline: dict[str, float] | None = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print_metrics_table(metrics, baseline)
    latest_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if baseline is None:
        baseline_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\nNo prior baseline — this run is now eval/results/baseline.json.")
        return 0

    if args.update_baseline:
        baseline_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\neval/results/baseline.json updated to this run.")
        return 0

    regressions = check_regressions(metrics, baseline, args.tolerance)
    if regressions:
        print("\nRegressions beyond tolerance:")
        for regression in regressions:
            print(f"  - {regression}")
        return 1

    print("\nNo regressions beyond tolerance.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=Path(__file__).resolve().parent / "cases")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite baseline.json with this run's metrics instead of comparing.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
