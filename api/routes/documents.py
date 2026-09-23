"""
api/routes/documents.py
-----------------------
Rewritten 2026-08-21 (Qdrant migration). Upload is now ASYNC: the request
registers the document (Postgres, status=UPLOADED) and schedules
pipeline/ingest.py::process_document() as a FastAPI BackgroundTasks job,
returning immediately — the caller polls GET /documents/{id} (or
GET /documents/{id}?...) for status. This matches IMPLEMENTATION_PLAN
(2).md's decision: in-process BackgroundTasks, no Celery/Redis for v1.

Endpoint paths/methods are unchanged (requirement: existing consumers —
clariona-core's knowledge-base integration, the misinformation-detection
engine — must not need to change how they call in). Response shapes are
additive only: new `status`/`status_detail`/`metadata` fields, existing
fields unchanged.

Also fixes Phase H #1 (path traversal — the on-disk filename is now
server-generated, never the client-supplied `file.filename`) and #2
(uploaded files are now tenant/org-scoped on disk, not one flat directory).
"""

import json
import mimetypes
import os
import uuid
from urllib.parse import quote
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException, Depends, Query, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_tenant_id, get_org_unit_id
from pipeline import object_store
from api.schemas.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDetailResponse,
    DocumentUpdateRequest,
    DocumentUpdateResponse,
    ReprocessResponse,
    DeleteResponse,
)
from db.database import get_db_session_fastapi
from db.models import Document
from pipeline.extractor import SUPPORTED_EXTENSIONS
from pipeline.storage import register_document, set_stored_path
from pipeline import ingest, vector_store
from utils.signed_link import generate_file_token, verify_file_token
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS


def _validate_file_extension(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"File type '{ext}' not supported. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    return ext


def _to_utc_datetime(d: Optional[date]) -> Optional[datetime]:
    if d is None:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """datetime -> ISO 8601 string for Qdrant payload fields (see pipeline/ingest.py's effective_from/effective_to docstring)."""
    return dt.isoformat() if dt else None


def _parse_metadata_form(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("metadata must be a JSON object")
        return parsed
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {e}")


def _tenant_upload_dir(tenant_id: str, org_unit_id: str) -> Path:
    """
    Phase H #2 — tenant/org-scoped on disk, not one flat directory. Path
    components come from already-validated header values (api.dependencies
    rejects empty/malformed tenant/org headers before this is reached), not
    raw unsanitized user input.
    """
    root = Path(settings.UPLOAD_DIR).resolve()
    d = (root / tenant_id / org_unit_id).resolve()

    # Defence in depth. api.dependencies rejects ids containing path
    # separators or "..", so this should be unreachable - but this is the
    # line that actually writes user bytes to a location derived from a
    # header, so it verifies the result rather than trusting the check
    # upstream to still be there after some future refactor.
    if not d.is_relative_to(root):
        logger.error("Refusing upload path outside UPLOAD_DIR: %s", d)
        raise HTTPException(status_code=400, detail="Invalid tenant or org unit identifier")

    d.mkdir(parents=True, exist_ok=True)
    return d


def _compute_bytes_hash(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _resolve_local_source(stored_path: str) -> tuple[str, bool]:
    """
    Turn a documents.stored_path into a path the ingestion job can actually
    open, plus whether that path is a throwaway WE created.

    stored_path has two forms now - a local path, or an 's3://' URI - and
    three separate call sites feed it to pipeline/ingest.py. Resolving it in
    one place is what stops them drifting: before this, the duplicate-retry
    path handed an 's3://' URI straight to the file parser (which cannot
    open it), and the reprocess path handed over a real local original that
    then got deleted as if it were scratch.

    The returned bool is passed through as delete_local_when_done, so the
    rule is uniform everywhere: delete only what this function downloaded.
    """
    if object_store.is_s3_uri(stored_path):
        return str(object_store.download_to_temp(stored_path)), True
    return stored_path, False


def _stored_file_exists(record: Document) -> bool:
    """
    2026-08-23 — every existence check on stored_path has to go through
    here now that the path has two forms. An s3:// URI is NOT a local path,
    so os.path.exists() on it is always False; checking it directly made
    every S3-backed preview link 404. We trust the URI rather than issuing
    a HEAD per request: this runs on the list endpoint too (once per
    document per page), and a round trip each would be a real cost to
    catch a case that only happens if someone deletes from the bucket
    behind the app's back.
    """
    if not record.stored_path:
        return False
    if object_store.is_s3_uri(record.stored_path):
        return True
    return os.path.exists(record.stored_path)


def _build_preview_link(record: Document, tenant_id: str, org_unit_id: str, request: Request) -> Optional[str]:
    """Web-scraped: source_path IS the URL. Uploaded: signed link to GET /documents/{id}/file (see utils/signed_link.py)."""
    if not record.source_path:
        return None
    if record.source_path.startswith("http://") or record.source_path.startswith("https://"):
        return record.source_path
    if not _stored_file_exists(record):
        return None
    token = generate_file_token(str(record.id), tenant_id, org_unit_id)
    base = settings.PUBLIC_BASE_URL.rstrip("/") if settings.PUBLIC_BASE_URL else str(request.base_url).rstrip("/")
    return f"{base}/documents/{record.id}/file?token={token}"


def _to_response(record: Document, tenant_id: str, org_unit_id: str, request: Request, elapsed_sec: Optional[float] = None) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        id=record.id, doc_name=record.doc_name, summary_text=record.summary_text or "",
        key_topics=record.key_topics, page_count=record.page_count or 0,
        chunk_count=record.chunk_count or 0, source_path=record.source_path,
        language=record.language or "English", model_used=record.model_used, doc_hash=record.doc_hash,
        tenant_id=record.tenant_id, org_unit_id=record.org_unit_id,
        category=record.category, effective_from=record.effective_from, effective_to=record.effective_to,
        is_ground_truth=record.is_ground_truth,
        title=record.title, author=record.author, document_type=record.document_type, domain=record.domain,
        key_entities=record.key_entities, mentioned_dates=record.mentioned_dates,
        target_audience=record.target_audience, sentiment=record.sentiment,
        confidentiality_level=record.confidentiality_level, metadata_status=record.metadata_status,
        image_count=record.image_count or 0, preview_link=_build_preview_link(record, tenant_id, org_unit_id, request),
        status=record.status, status_detail=record.status_detail, metadata=record.metadata_json,
        created_at=record.created_at, updated_at=record.updated_at, elapsed_sec=elapsed_sec,
    )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201, summary="Upload a document (async ingestion)")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF, DOCX, PPTX, XLSX, TXT, CSV, MD, HTML, or image file"),
    category: Optional[str] = Form(default=None),
    effective_from: Optional[date] = Form(default=None),
    effective_to: Optional[date] = Form(default=None),
    is_ground_truth: bool = Form(default=False, description="Optional filter, no longer a hard retrieval gate"),
    metadata: Optional[str] = Form(default=None, description="JSON object — unknown keys stored, allowlisted keys become filterable in search"),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
):
    ext = _validate_file_extension(file.filename)
    metadata_dict = _parse_metadata_form(metadata)

    raw_bytes = await file.read()
    file_hash = _compute_bytes_hash(raw_bytes)

    # Dedup — return the cached record, re-schedule ingestion only if the
    # existing one previously failed.
    from pipeline.storage import check_duplicate
    existing = check_duplicate(file_hash, tenant_id, org_unit_id)
    if existing:
        logger.info("Duplicate upload for tenant=%s org_unit=%s hash=%s (status=%s)", tenant_id, org_unit_id, file_hash[:12], existing.status)
        if existing.status == "FAILED":
            # stored_path may be an s3:// URI, which the extractor cannot
            # open - resolve it to a real local file first.
            try:
                retry_path, retry_is_temp = _resolve_local_source(existing.stored_path)
            except Exception as e:
                logger.exception("Could not fetch %s for duplicate retry: %s", existing.stored_path, e)
                raise HTTPException(status_code=502, detail="Could not retrieve the stored file. Contact support with the X-Request-ID response header if this persists.")
            background_tasks.add_task(
                ingest.process_document, str(existing.id), tenant_id, org_unit_id,
                retry_path, existing.doc_name, existing.doc_hash,
                existing.is_ground_truth, metadata_dict,
                _iso(existing.effective_from), _iso(existing.effective_to),
                retry_is_temp,
            )
            from pipeline.storage import set_status
            set_status(existing.id, tenant_id, org_unit_id, "PROCESSING")
        return _to_response(existing, tenant_id, org_unit_id, request, elapsed_sec=0.0)

    # Phase H #1 fix — server-generated on-disk filename, never
    # file.filename (client-controlled, was previously used unsanitized —
    # path traversal). Phase H #2 — tenant/org-scoped directory.
    dest_dir = _tenant_upload_dir(tenant_id, org_unit_id)
    dest_path = dest_dir / f"{uuid.uuid4()}{ext}"
    dest_path.write_bytes(raw_bytes)
    logger.info("File saved: %s (tenant=%s org_unit=%s)", dest_path, tenant_id, org_unit_id)

    effective_from_dt = _to_utc_datetime(effective_from)
    effective_to_dt = _to_utc_datetime(effective_to)

    record, _created = register_document(
        doc_name=file.filename, file_hash=file_hash, tenant_id=tenant_id, org_unit_id=org_unit_id,
        file_type=ext.lstrip("."), file_size=len(raw_bytes), source="upload",
        source_path=str(dest_path), stored_path=str(dest_path),
        category=category, effective_from=effective_from_dt, effective_to=effective_to_dt,
        is_ground_truth=is_ground_truth, metadata=metadata_dict,
    )

    # 2026-08-23 — with a bucket configured, S3 becomes the durable copy and
    # the local file is demoted to scratch space that ingestion reads once
    # and then deletes (pipeline/ingest.py). Uploaded AFTER register_document
    # so the object key can carry the real document id.
    #
    # A failure here fails the whole request on purpose. Silently keeping the
    # local copy instead would leave the caller believing their file is
    # durably stored when it is only on a container's disk — the exact
    # failure mode this change exists to remove. The Postgres row is marked
    # FAILED so it doesn't sit in PROCESSING with no job coming.
    if object_store.is_enabled():
        try:
            key = object_store.build_key(tenant_id, org_unit_id, str(record.id), ext)
            uri = object_store.upload_file(str(dest_path), key)
            set_stored_path(record.id, tenant_id, org_unit_id, uri)
            record.stored_path = uri
        except Exception as e:
            logger.exception("S3 upload failed for document %s: %s", record.id, e)
            from pipeline.storage import set_status
            set_status(record.id, tenant_id, org_unit_id, "FAILED", "S3 upload failed")
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=502,
                detail="Could not store the uploaded file. Contact support with the X-Request-ID response header if this persists.",
            )

    # Ingestion always reads the LOCAL path — it was just written here, so
    # there is no reason to round-trip it back out of S3.
    # delete_local_when_done is keyed on whether S3 actually took the file
    # (stored_path became an s3:// URI above), NOT on whether S3 is merely
    # configured. With no bucket, this scratch file IS the only copy.
    background_tasks.add_task(
        ingest.process_document, str(record.id), tenant_id, org_unit_id,
        str(dest_path), file.filename, file_hash, is_ground_truth, metadata_dict,
        _iso(effective_from_dt), _iso(effective_to_dt),
        object_store.is_s3_uri(record.stored_path),
    )

    return _to_response(record, tenant_id, org_unit_id, request)


@router.get("/", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=10, ge=1, le=100),
    document_type: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None, description="Filter by ingestion status: UPLOADED | PROCESSING | READY | FAILED"),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    offset = (page - 1) * per_page
    query = db.query(Document).filter(Document.tenant_id == tenant_id, Document.org_unit_id == org_unit_id)
    if document_type:
        query = query.filter(Document.document_type == document_type.lower())
    if domain:
        query = query.filter(Document.domain == domain.lower())
    if category:
        query = query.filter(Document.category == category)
    if status:
        query = query.filter(Document.status == status.upper())

    total = query.count()
    records = query.order_by(Document.created_at.desc()).offset(offset).limit(per_page).all()
    items = [
        DocumentListItem(
            id=rec.id, doc_name=rec.doc_name, page_count=rec.page_count or 0, chunk_count=rec.chunk_count or 0,
            model_used=rec.model_used, document_type=rec.document_type, domain=rec.domain, category=rec.category,
            is_ground_truth=rec.is_ground_truth, image_count=rec.image_count or 0, status=rec.status,
            created_at=rec.created_at, summary_preview=(rec.summary_text[:200] if rec.summary_text else None),
        )
        for rec in records
    ]
    return DocumentListResponse(total=total, page=page, per_page=per_page, documents=items)


def _get_owned_document(doc_id: uuid.UUID, tenant_id: str, org_unit_id: str, db: Session) -> Document:
    record = db.query(Document).filter(
        Document.id == doc_id, Document.tenant_id == tenant_id, Document.org_unit_id == org_unit_id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return record


@router.get("/{doc_id}", response_model=DocumentDetailResponse, summary="Get document details (includes preview data + ingestion status)")
async def get_document(
    doc_id: uuid.UUID, request: Request,
    tenant_id: str = Depends(get_tenant_id), org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = _get_owned_document(doc_id, tenant_id, org_unit_id, db)
    resp = _to_response(record, tenant_id, org_unit_id, request)
    return DocumentDetailResponse(**resp.model_dump(exclude={"elapsed_sec"}))


@router.get("/{doc_id}/preview-url", summary="Get a fresh preview link for a document")
async def get_document_preview_url(
    doc_id: uuid.UUID, request: Request,
    tenant_id: str = Depends(get_tenant_id), org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = _get_owned_document(doc_id, tenant_id, org_unit_id, db)
    preview_link = _build_preview_link(record, tenant_id, org_unit_id, request)
    if not preview_link:
        raise HTTPException(status_code=404, detail="No file available for this document")
    return {"doc_id": str(doc_id), "preview_link": preview_link}


@router.get("/{doc_id}/file", summary="Stream the original uploaded file (signed link only)")
async def get_document_file(
    doc_id: uuid.UUID,
    token: str = Query(..., description="Signed token from a document's preview_link"),
    db: Session = Depends(get_db_session_fastapi),
):
    payload = verify_file_token(token)
    if not payload or payload.get("doc_id") != str(doc_id):
        raise HTTPException(status_code=403, detail="Invalid or expired link")

    # No headers on this route (plain link/<iframe>) — set RLS context
    # explicitly from the token's own verified payload, same pattern as
    # db.database._apply_rls_context().
    db.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": payload["tenant_id"]})
    db.execute(text("SELECT set_config('app.org_unit_id', :v, true)"), {"v": payload["org_unit_id"]})

    record = db.query(Document).filter(
        Document.id == doc_id, Document.tenant_id == payload["tenant_id"], Document.org_unit_id == payload["org_unit_id"],
    ).first()
    if not record or not record.stored_path:
        raise HTTPException(status_code=404, detail="File not found")

    # 2026-08-23 — S3-backed documents stream through this endpoint rather
    # than redirecting to a presigned URL. A 307 to amazonaws.com would
    # change this endpoint's contract for live callers (clariona-core
    # consumes this API), and would add a second credential with its own
    # expiry alongside the signed token that already authorises this
    # request. Revisit only if preview bandwidth becomes the bottleneck.
    if object_store.is_s3_uri(record.stored_path):
        try:
            body = object_store.open_stream(record.stored_path)
        except Exception as e:
            logger.warning("S3 fetch failed for document %s (%s): %s", doc_id, record.stored_path, e)
            raise HTTPException(status_code=404, detail="File no longer exists in storage")
        # Content-Disposition is built with RFC 5987/6266 percent-encoding,
        # not an f-string. doc_name is the uploaded filename and only its
        # EXTENSION is validated (_validate_file_extension), so a name
        # containing a double quote would otherwise close the quoted string
        # early and let the uploader inject extra header parameters - e.g. a
        # filename* that overrides the name the victim's browser saves under.
        # FileResponse on the local branch below does this encoding itself;
        # this branch has to do it explicitly.
        safe_name = quote(record.doc_name or "document", safe="")

        # media_type inferred, not hardcoded: 'inline' plus
        # application/octet-stream makes every browser DOWNLOAD instead of
        # render, which silently broke the iframe preview this endpoint
        # exists for - and only for S3-backed documents, so it looked
        # intermittent. FileResponse infers this from the filename already.
        media_type, _ = mimetypes.guess_type(record.doc_name or "")

        return StreamingResponse(
            body,
            media_type=media_type or "application/octet-stream",
            headers={"Content-Disposition": f"inline; filename*=UTF-8''{safe_name}"},
        )

    if not os.path.exists(record.stored_path):
        raise HTTPException(status_code=404, detail="File no longer exists on disk")
    return FileResponse(record.stored_path, filename=record.doc_name)


@router.patch("/{doc_id}", response_model=DocumentUpdateResponse, summary="Update document name")
async def update_document(
    doc_id: uuid.UUID, payload: DocumentUpdateRequest,
    tenant_id: str = Depends(get_tenant_id), org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = _get_owned_document(doc_id, tenant_id, org_unit_id, db)
    record.doc_name = payload.doc_name
    db.commit()
    db.refresh(record)
    return DocumentUpdateResponse(id=record.id, doc_name=record.doc_name, updated_at=record.updated_at)


@router.post("/{doc_id}/reprocess", response_model=ReprocessResponse, status_code=202, summary="Retry ingestion for a FAILED or READY document")
async def reprocess_document(
    doc_id: uuid.UUID, background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id), org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = _get_owned_document(doc_id, tenant_id, org_unit_id, db)
    if not record.stored_path:
        raise HTTPException(status_code=422, detail="This document has no local file to reprocess (web-scraped documents are not reprocessable this way)")

    # 2026-08-23 — an S3-backed document normally has NO local copy by now:
    # ingestion deletes it as soon as it finishes. Pull it back down first.
    # This is the case that simply could not work before S3 — once a
    # container was replaced, a local-only file was gone and the document
    # was permanently unreprocessable.
    #
    # Downloaded synchronously, before the background task is scheduled, so
    # a fetch failure surfaces as a real HTTP error to the caller instead of
    # failing invisibly inside a job nobody is watching.
    try:
        local_path, is_temp = _resolve_local_source(record.stored_path)
    except Exception as e:
        logger.exception("Could not fetch %s from S3 for reprocess: %s", record.stored_path, e)
        raise HTTPException(status_code=502, detail="Could not retrieve the stored file. Contact support with the X-Request-ID response header if this persists.")

    from pipeline.storage import set_status
    set_status(doc_id, tenant_id, org_unit_id, "PROCESSING")
    background_tasks.add_task(
        ingest.process_document, str(doc_id), tenant_id, org_unit_id,
        local_path, record.doc_name, record.doc_hash,
        record.is_ground_truth, record.metadata_json or {},
        _iso(record.effective_from), _iso(record.effective_to),
        is_temp,      # only the temp WE downloaded is disposable
    )
    return ReprocessResponse(id=doc_id, status="PROCESSING")


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(
    doc_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id), org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = _get_owned_document(doc_id, tenant_id, org_unit_id, db)
    # Qdrant has no cascade — delete its points explicitly, before the
    # Postgres row (deterministic order: if Qdrant deletion fails, the
    # Postgres row still exists and the delete can be retried).
    vector_store.delete_document_points(tenant_id=tenant_id, document_id=str(doc_id))
    vector_store.delete_document_summary(tenant_id=tenant_id, document_id=str(doc_id))
    if object_store.is_s3_uri(record.stored_path):
        object_store.delete(record.stored_path)          # non-fatal by design
    elif record.stored_path and os.path.exists(record.stored_path):
        try:
            os.remove(record.stored_path)
        except OSError as e:
            logger.warning("Failed to remove file %s: %s", record.stored_path, e)
    db.delete(record)
    db.commit()
    return DeleteResponse(id=doc_id)
