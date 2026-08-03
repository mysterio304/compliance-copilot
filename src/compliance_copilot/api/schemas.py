"""API-specific response models — thin wrappers around schemas/ for shapes the
domain models don't already cover, rather than domain models themselves."""

from datetime import date

from pydantic import BaseModel


class DocumentInventoryItem(BaseModel):
    id: str
    title: str
    regulation_code: str
    language: str
    publication_date: date
    source_url: str
    clause_count: int
