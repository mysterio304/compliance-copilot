#!/usr/bin/env python3
"""Prints corpus size as a measured number: document count, clause count, chunk
count, and a language breakdown, read from the ingested JSON in data/processed/.

Usage: uv run python scripts/corpus_stats.py [--dir data/processed]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compliance_copilot.ingest.pipeline import IngestedDocument  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    json_files = sorted(args.dir.glob("*.json"))
    if not json_files:
        print(f"No ingested documents found in {args.dir}. Run scripts/ingest_corpus.py first.")
        raise SystemExit(1)

    documents = [
        IngestedDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in json_files
    ]

    total_clauses = sum(len(doc.clauses) for doc in documents)
    total_chunks = sum(len(doc.chunks) for doc in documents)
    language_counts: Counter[str] = Counter(doc.document.language for doc in documents)

    print(f"Documents: {len(documents)}")
    print(f"Clauses:   {total_clauses}")
    print(f"Chunks:    {total_chunks}")
    print("Language breakdown (by document):")
    for language, count in sorted(language_counts.items()):
        print(f"  {language}: {count}")
    print()
    print(f"{'Regulation':<22}{'Title':<55}{'Clauses':>8}{'Chunks':>8}")
    for doc in documents:
        print(
            f"{doc.document.regulation_code:<22}{doc.document.title[:53]:<55}"
            f"{len(doc.clauses):>8}{len(doc.chunks):>8}"
        )


if __name__ == "__main__":
    main()
