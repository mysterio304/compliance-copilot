from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["ru", "zh", "en"]


class SourceDocument(BaseModel):
    """A single regulation PDF, ingested once and cited many times."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    regulation_code: str = Field(description='e.g. "ТР ТС 004/2011"')
    language: Language
    publication_date: date
    source_url: str


class Clause(BaseModel):
    """The citable unit: a статья/пункт/приложение, with its span in the source text."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    clause_path: str = Field(description='e.g. "Статья 4, пункт 2"')
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    language: Language

    @model_validator(mode="after")
    def check_span_order(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than char_start ({self.char_start})"
            )
        return self


class Chunk(BaseModel):
    """An embeddable fragment of a clause. 1:1 with a clause unless the clause exceeds
    embedding context, in which case multiple chunks share the same clause_id and clause_path."""

    model_config = ConfigDict(frozen=True)

    id: str
    clause_id: str
    text: str
    embedding_model: str
    dense_vector: list[float] | None = None
    sparse_weights: dict[str, float] | None = None
