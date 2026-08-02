#!/usr/bin/env python3
"""Embeds every chunk from data/processed/*.json with BGE-M3 (dense + sparse) and
writes documents/clauses/chunks into Postgres.

Usage: uv run python scripts/load_corpus.py [--dir data/processed]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compliance_copilot.config import get_settings  # noqa: E402
from compliance_copilot.db.session import (  # noqa: E402
    init_models,
    make_engine,
    make_session_factory,
)
from compliance_copilot.retrieval.loader import (  # noqa: E402
    embed_and_store_document,
    load_ingested_document,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    settings = get_settings()
    engine = make_engine(settings.database_url)
    await init_models(engine)
    session_factory = make_session_factory(engine)

    json_files = sorted(args.dir.glob("*.json"))
    if not json_files:
        print(f"No ingested documents found in {args.dir}. Run scripts/ingest_corpus.py first.")
        raise SystemExit(1)

    async with session_factory() as session:
        for path in json_files:
            ingested = load_ingested_document(path)
            chunk_count = await embed_and_store_document(
                session, ingested, settings.embedding_model_name
            )
            print(f"{ingested.document.regulation_code}: {chunk_count} chunks embedded and stored")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
