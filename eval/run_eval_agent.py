#!/usr/bin/env python3
"""Agent-level evaluation harness — the same five metrics as eval/run_eval.py,
scored against what the LangGraph agent actually concludes rather than what
retrieval surfaces.

Why both exist: run_eval.py is the fast, free CI gate for retrieval quality
(no LLM calls, seconds per run). It cannot measure abstention or false
positives, because retrieval makes no applicability decision — see
eval/agent_metrics.py for the full argument. This harness does measure them,
at the cost of one full agent run per case (real DeepSeek calls, minutes per
run), so it is a deliberate, separately-baselined step rather than something
CI runs on every push.

Runs sequentially on purpose: DeepSeek is pinned to temperature=0 for
determinism, and firing ten concurrent multi-node agent runs at it trades that
determinism plus rate-limit headroom for a few minutes of wall clock.

Usage: uv run python eval/run_eval_agent.py [--cases-dir DIR] [--tolerance 0.05]
                                            [--update-baseline] [--only CASE_ID]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from compliance_copilot.agent.graph import run_agent  # noqa: E402
from compliance_copilot.agent.llm import StructuredOutputError  # noqa: E402
from compliance_copilot.config import Settings, get_settings  # noqa: E402
from compliance_copilot.db.models import ChunkRow, ClauseRow, DocumentRow  # noqa: E402
from compliance_copilot.db.session import make_engine, make_session_factory  # noqa: E402
from eval.agent_metrics import AgentPrediction, ClauseIndex, prediction_from_report  # noqa: E402
from eval.gold_case import GoldCase, load_cases  # noqa: E402
from eval.metrics import CaseAndPrediction, compute_all_metrics  # noqa: E402
from eval.run_eval import (  # noqa: E402
    check_regressions,
    format_metrics_markdown,
    print_metrics_table,
    write_github_step_summary,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_PATH = RESULTS_DIR / "agent_baseline.json"
LATEST_PATH = RESULTS_DIR / "agent_latest.json"


async def _clause_index(session: AsyncSession, clause_ids: list[str]) -> ClauseIndex:
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


async def run_case(session: AsyncSession, case: GoldCase, settings: Settings) -> AgentPrediction:
    state = await run_agent(settings, session, case.input_spec)
    report = state["report"]
    cited = [citation.clause_id for finding in report.findings for citation in finding.citations]
    index = await _clause_index(session, cited)
    return prediction_from_report(case.id, report, index)


def print_case_line(case: GoldCase, prediction: AgentPrediction) -> None:
    verdict = (
        "abstained" if prediction.abstained else ", ".join(sorted(prediction.predicted_regulations))
    )
    forbidden = sorted(set(case.must_not_return) & prediction.predicted_regulations)
    flag = "  <- FORBIDDEN" if forbidden else ""
    if case.expect_abstain and not prediction.abstained:
        flag = "  <- SHOULD HAVE ABSTAINED"
    print(f"  {case.id}: {verdict}{flag}")


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    cases = load_cases(args.cases_dir)
    if args.only:
        cases = [case for case in cases if case.id in args.only]
        if not cases:
            print(f"No cases matched --only {args.only}", file=sys.stderr)
            return 1

    predictions: list[AgentPrediction] = []
    failures: list[str] = []

    async with session_factory() as session:
        if (await session.execute(select(ChunkRow.id).limit(1))).first() is None:
            print(
                "No embedded chunks in the database — run scripts/ingest_corpus.py "
                "and scripts/load_corpus.py first.",
                file=sys.stderr,
            )
            return 1

        print(f"Running {len(cases)} cases through the agent (real LLM calls)...")
        for case in cases:
            try:
                prediction = await run_case(session, case, settings)
            except StructuredOutputError as exc:
                # Don't let one bad LLM response silently shrink the denominator —
                # record it, keep going, and fail the run at the end.
                failures.append(f"{case.id}: {exc}")
                print(f"  {case.id}: ERROR — {exc}")
                continue
            predictions.append(prediction)
            print_case_line(case, prediction)

    await engine.dispose()

    scored = [case for case in cases if case.id in {p.case_id for p in predictions}]
    pairs: list[CaseAndPrediction] = list(zip(scored, predictions, strict=True))
    metrics = compute_all_metrics(pairs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline: dict[str, float] | None = None
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    print()
    print_metrics_table(metrics, baseline)
    write_github_step_summary(format_metrics_markdown(metrics, baseline))
    LATEST_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if failures:
        print(f"\n{len(failures)} case(s) failed to produce a report:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if args.only:
        print("\nPartial run (--only): baseline not written or compared.")
        return 0

    if baseline is None:
        BASELINE_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nNo prior baseline — this run is now {BASELINE_PATH.name}.")
        return 0

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n{BASELINE_PATH.name} updated to this run.")
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
        help=f"Overwrite {BASELINE_PATH.name} with this run's metrics instead of comparing.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="CASE_ID",
        help="Run only these case ids (skips baseline write/compare).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
