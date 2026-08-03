#!/usr/bin/env python3
"""Sweeps retrieval parameters (fusion_weight, top_k, relevance_floor) against
the eval harness and records every configuration's metrics — the point being
that the final choice is picked by measurement, not intuition.

Usage: uv run python scripts/tune_retrieval.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.gold_case import GoldCase, load_cases  # noqa: E402
from eval.metrics import compute_all_metrics  # noqa: E402
from eval.run_eval import run_case  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from compliance_copilot.config import Settings, get_settings  # noqa: E402
from compliance_copilot.db.models import ChunkRow  # noqa: E402
from compliance_copilot.db.session import make_engine, make_session_factory  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent.parent / "eval" / "cases"

# (fusion_weight, top_k, relevance_floor). Baseline first, then one axis varied
# at a time against it, then a couple of combined candidates informed by what
# the single-axis sweeps show.
GRID: list[tuple[float, int, float]] = [
    (0.5, 10, 0.35),  # baseline (current .env defaults)
    (0.0, 10, 0.35),  # sparse only
    (0.3, 10, 0.35),
    (0.7, 10, 0.35),
    (1.0, 10, 0.35),  # dense only
    (0.5, 5, 0.35),
    (0.5, 15, 0.35),
    (0.5, 20, 0.35),
    (0.5, 10, 0.25),
    (0.5, 10, 0.45),
    (0.5, 10, 0.55),
]


async def run_config(
    base_settings: Settings,
    session: AsyncSession,
    cases: list[GoldCase],
    fusion_weight: float,
    top_k: int,
    relevance_floor: float,
) -> dict[str, float | None]:
    settings = base_settings.model_copy(
        update={
            "retrieval_fusion_weight": fusion_weight,
            "retrieval_top_k": top_k,
            "retrieval_relevance_floor": relevance_floor,
        }
    )
    predictions = [await run_case(session, case, settings) for case in cases]
    pairs = list(zip(cases, predictions, strict=True))
    return compute_all_metrics(pairs)


def fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


async def main() -> None:
    base_settings = get_settings()
    engine = make_engine(base_settings.database_url)
    session_factory = make_session_factory(engine)
    cases = load_cases(CASES_DIR)

    results: list[tuple[float, int, float, dict[str, float | None]]] = []
    async with session_factory() as session:
        has_data = (await session.execute(select(ChunkRow.id).limit(1))).first()
        if has_data is None:
            print(
                "No embedded chunks — run scripts/ingest_corpus.py and "
                "scripts/load_corpus.py first.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        for fusion_weight, top_k, relevance_floor in GRID:
            metrics = await run_config(
                base_settings, session, cases, fusion_weight, top_k, relevance_floor
            )
            results.append((fusion_weight, top_k, relevance_floor, metrics))
            print(f"fusion={fusion_weight} top_k={top_k} floor={relevance_floor} done")

    await engine.dispose()

    print()
    print(
        f"{'fusion':>7}{'top_k':>7}{'floor':>7}{'recall@k':>10}"
        f"{'precision':>11}{'reg_f1':>8}{'fpr':>8}{'abstain':>9}"
    )
    for fusion_weight, top_k, relevance_floor, metrics in results:
        print(
            f"{fusion_weight:>7}{top_k:>7}{relevance_floor:>7}"
            f"{fmt(metrics['retrieval_recall_at_k']):>10}"
            f"{fmt(metrics['citation_precision']):>11}"
            f"{fmt(metrics['regulation_set_f1']):>8}"
            f"{fmt(metrics['false_positive_rate']):>8}"
            f"{fmt(metrics['abstention_accuracy']):>9}"
        )


if __name__ == "__main__":
    asyncio.run(main())
