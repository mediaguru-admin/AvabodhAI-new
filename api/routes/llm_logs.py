"""
api/routes/llm_logs.py
----------------------
Read API for the LLM execution log (pipeline/llm_log.py) — one row per
chat-completion / vision call made on the caller's behalf. Scoped to the
caller's tenant + org unit, like every other route (and RLS backs it up).

Prompt/response text is omitted from the list unless include_text=true;
GET /llm-logs/{id} always returns it.
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_tenant_id, get_org_unit_id
from db.database import get_db_session_fastapi
from db.models import LLMExecutionLog

router = APIRouter()

_TEXT_FIELDS = ("system_prompt", "user_prompt", "response_text")


def _to_dict(row: LLMExecutionLog, include_text: bool) -> dict:
    data = {c.name: getattr(row, c.name) for c in LLMExecutionLog.__table__.columns}
    data["id"] = str(row.id)
    if not include_text:
        for field in _TEXT_FIELDS:
            data.pop(field)
    return data


@router.get("/", summary="List LLM execution logs")
async def list_llm_logs(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    purpose: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None, description="success | failed | timeout | rate_limited"),
    client_id: Optional[str] = Query(default=None, description="Caller's X-Client-ID"),
    model_name: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(default=None, description="document | chat_thread"),
    entity_id: Optional[str] = Query(default=None),
    created_from: Optional[datetime] = Query(default=None),
    created_to: Optional[datetime] = Query(default=None),
    include_text: bool = Query(default=False, description="Include prompt/response text"),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    query = db.query(LLMExecutionLog).filter(
        LLMExecutionLog.tenant_id == tenant_id, LLMExecutionLog.org_unit_id == org_unit_id,
    )
    for column, value in (
        (LLMExecutionLog.purpose, purpose),
        (LLMExecutionLog.status, status),
        (LLMExecutionLog.client_id, client_id),
        (LLMExecutionLog.model_name, model_name),
        (LLMExecutionLog.entity_type, entity_type),
        (LLMExecutionLog.entity_id, entity_id),
    ):
        if value:
            query = query.filter(column == value)
    if created_from:
        query = query.filter(LLMExecutionLog.created_at >= created_from)
    if created_to:
        query = query.filter(LLMExecutionLog.created_at <= created_to)

    total = query.count()
    rows = (
        query.order_by(LLMExecutionLog.created_at.desc())
        .offset((page - 1) * per_page).limit(per_page).all()
    )
    return {
        "items": [_to_dict(r, include_text) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/{log_id}", summary="Get one LLM execution log, including prompt/response text")
async def get_llm_log(
    log_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    row = db.query(LLMExecutionLog).filter(
        LLMExecutionLog.id == log_id,
        LLMExecutionLog.tenant_id == tenant_id,
        LLMExecutionLog.org_unit_id == org_unit_id,
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"LLM log '{log_id}' not found")
    return _to_dict(row, include_text=True)
