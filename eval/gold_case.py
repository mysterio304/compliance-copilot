from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ExpectedClause(BaseModel):
    doc: str
    clause: str


class GoldCase(BaseModel):
    id: str
    input_spec: str
    language: str
    expected_regulations: list[str] = Field(default_factory=list)
    expected_clauses: list[ExpectedClause] = Field(default_factory=list)
    must_not_return: list[str] = Field(default_factory=list)
    # Not in the brief's example schema, but a required signal: without it there is
    # no way to distinguish "correctly found nothing applies" from "should have
    # abstained." Explicit here rather than inferred from an empty
    # expected_regulations list.
    expect_abstain: bool = False
    notes: str = ""


def load_cases(cases_dir: Path) -> list[GoldCase]:
    cases = [
        GoldCase.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(cases_dir.glob("*.yaml"))
    ]
    if not cases:
        raise ValueError(f"No gold cases found in {cases_dir}")
    return cases
