import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.kb_schemas import (
    KBUploadResponse,
    KBStatusResponse,
    KBRetrieveRequest,
    KBChatRequest,
    KBChatResponse
)
from db.database import get_async_db_session_fastapi, AsyncSessionFactory
from db.models import KBDocument
from pipeline.kb_pipeline import ingest_document, retrieve_relevant_context, chat_with_kb
from pipeline.attribute_store import get_document_attributes, upsert_definition
import json

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "./kb_uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def bg_ingestion_wrapper(file_path: str, owner_id: str, document_id: str):
    """
    Wrapper for running the ingestion pipeline in the background.
    Creates a dedicated session to prevent request session closure issues.
    """
    async with AsyncSessionFactory() as session:
        try:
            await ingest_document(file_path, owner_id, document_id, session)
        except Exception:
            logger.exception("Background ingestion failed for document %s", document_id)
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    logger.warning("Could not delete temp file: %s", file_path)


@router.post(
    "/documents",
    response_model=KBUploadResponse,
    status_code=202,
    summary="Upload document to Knowledge Base"
)
@router.post(
    "/upload",
    response_model=KBUploadResponse,
    status_code=202,
    include_in_schema=False,
)
async def upload_kb_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    owner_id: str = Form(...),
    category: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    active_attributes: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db_session_fastapi)
):
    # 1. Create a unique document ID
    doc_id = uuid.uuid4()

    # 2. Save file temporarily
    # Never trust a client-provided filename as a path component.
    safe_filename = Path(file.filename or "upload").name
    temp_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{safe_filename}")
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logger.exception("Failed to write temp file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 3. Create document record in database
    kb_doc = KBDocument(
        id=doc_id,
        owner_id=owner_id,
        file_name=safe_filename,
        status="processing",
        created_at=datetime.now(timezone.utc)
    )
    db.add(kb_doc)
    await db.commit()

    # NestJS sends active extraction definitions as upload metadata.
    try:
        definitions = json.loads(active_attributes or "[]")
        for definition in definitions if isinstance(definitions, list) else []:
            if definition.get("key") and definition.get("description"):
                upsert_definition(owner_id, None, definition)
    except Exception:
        logger.exception("Could not save active attribute definitions")

    # 4. Dispatch background task
    background_tasks.add_task(bg_ingestion_wrapper, temp_path, owner_id, str(doc_id))

    return KBUploadResponse(kbDocumentId=doc_id, status="processing")


@router.get(
    "/status/{kbDocumentId}",
    response_model=KBStatusResponse,
    summary="Get status of KB document"
)
async def get_kb_document_status(
    kbDocumentId: uuid.UUID,
    db: AsyncSession = Depends(get_async_db_session_fastapi)
):
    stmt = select(KBDocument).filter_by(id=kbDocumentId)
    result = await db.execute(stmt)
    kb_doc = result.scalar_one_or_none()

    if not kb_doc:
        raise HTTPException(status_code=404, detail="KB Document not found")

    return KBStatusResponse(
        kbDocumentId=kb_doc.id,
        status=kb_doc.status,
        error_message=kb_doc.error_message,
        attributes=get_document_attributes(str(kb_doc.owner_id), None, str(kb_doc.id)) if kb_doc.status == "ready" else {},
    )


@router.post(
    "/retrieve",
    response_model=List[str],
    summary="Retrieve compressed document context"
)
async def retrieve_context(
    request: KBRetrieveRequest
):
    try:
        chunks = await retrieve_relevant_context(request.owner_id, request.query)
        return chunks
    except Exception as e:
        logger.exception("Retrieve context failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat",
    response_model=KBChatResponse,
    response_model_exclude_defaults=True,
    summary="Chat with Knowledge Base"
)
async def chat_kb(
    request: KBChatRequest,
    db: AsyncSession = Depends(get_async_db_session_fastapi)
):
    try:
        response = await chat_with_kb(
            owner_id=request.owner_id,
            message=request.message,
            conversation_id=request.conversation_id,
            db=db
        )
        if isinstance(response, str):
            return KBChatResponse(response=response)
        return KBChatResponse(
            response=str(response.get("response", "")),
            conversation_id=response.get("conversation_id"),
            sources=response.get("sources", []),
        )
    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(e))
