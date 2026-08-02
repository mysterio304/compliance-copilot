#!/usr/bin/env python3
"""Runs the ingestion pipeline over the regulation corpus manifest, writing one
JSON file per document (SourceDocument + its Clauses + its Chunks) to data/processed/.

Usage: uv run python scripts/ingest_corpus.py [--manifest PATH] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compliance_copilot.config import get_settings  # noqa: E402
from compliance_copilot.ingest.pipeline import ingest_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/corpus_manifest.yaml"), help="Manifest path"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed"), help="Output directory for JSON"
    )
    args = parser.parse_args()

    embedding_model = get_settings().embedding_model_name
    args.out.mkdir(parents=True, exist_ok=True)

    ingested = ingest_corpus(args.manifest, embedding_model)
    for doc in ingested:
        out_path = args.out / f"{doc.document.id}.json"
        out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"{doc.document.regulation_code}: {len(doc.clauses)} clauses, "
            f"{len(doc.chunks)} chunks -> {out_path}"
        )


if __name__ == "__main__":
    main()
