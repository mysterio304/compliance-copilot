import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from compliance_copilot.agent.graph import run_agent
from compliance_copilot.agent.llm import StructuredOutputError
from compliance_copilot.api.dependencies import get_session, get_settings_dependency
from compliance_copilot.config import Settings
from compliance_copilot.ingest.plain_text import extract_plain_text
from compliance_copilot.schemas.reports import ComplianceReport

router = APIRouter()


async def _text_from_upload(file: UploadFile) -> str:
    filename = (file.filename or "").lower()
    if file.content_type != "application/pdf" and not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp.flush()
        return await asyncio.to_thread(extract_plain_text, Path(tmp.name))


@router.post("/analyse", response_model=ComplianceReport)
async def analyse(
    spec_text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings_dependency),
) -> ComplianceReport:
    """Accepts a supplier spec as raw text or an uploaded PDF, runs the agent,
    and returns the ComplianceReport. Exactly one of spec_text/file is required."""
    if file is None and not (spec_text and spec_text.strip()):
        raise HTTPException(status_code=400, detail="Provide spec_text or upload a PDF file")

    text = await _text_from_upload(file) if file is not None else spec_text
    if text is None or not text.strip():
        raise HTTPException(status_code=400, detail="No usable text extracted from input")

    try:
        result = await run_agent(settings, session, text)
    except StructuredOutputError as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM output failed validation: {exc}"
        ) from exc
    return result["report"]
