"""
api/routes/internal.py
-----------------------
Server-to-server endpoints for other internal services calling this API
directly — as opposed to end-user/browser traffic (documents/chat/search)
or the separate /kb subsystem. Currently just one route: MDS
(misinformation-detection) evidence retrieval.

MDS used to query document_chunks/document_summaries directly via raw
SQL. That schema no longer exists — chunk storage moved to Qdrant (see
pipeline/vector_store.py, the 2026-08-21 migration); document_summaries
was renamed to documents. This endpoint replaces that direct-Postgres
access with a proper API call, same pattern Core already uses for /chat.

Auth: same trust model as every other route in this repo — X-Tenant-ID/
X-Org-Unit-ID headers, required by TenantGuardMiddleware like every
non-exempt path, resolved via api.dependencies.get_tenant_id/
get_org_unit_id same as every other route. query_embedding is the one
truly mandatory body field alongside those two headers.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.dependencies import get_tenant_id, get_org_unit_id
from pipeline.retriever import build_filter
from pipeline import vector_store
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_QUERY_EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small — matches vector_store's collection config


class EvidenceRetrievalRequest(BaseModel):
    query_embedding: list[float] = Field(
        ..., min_length=_QUERY_EMBEDDING_DIM, max_length=_QUERY_EMBEDDING_DIM,
        description="Pre-computed query embedding, 1536-dim (OpenAI text-embedding-3-small)",
    )
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(
        default=0.60, ge=-1.0, le=1.0,
        description="Minimum cosine similarity a chunk must clear to be returned",
    )
    org_ids: Optional[list[str]] = Field(
        default=None,
        description="Widen retrieval to multiple org units within your tenant. "
                     "Defaults to the caller's own org unit (X-Org-Unit-ID header) only.",
    )
    metadata: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Allowlisted document metadata filters, e.g. {\"country\": [\"India\"]}. "
                     "Unknown keys are rejected with a 422 — see pipeline.retriever.FILTERABLE_METADATA_KEYS "
                     "for the current allowlist.",
    )


class EvidenceChunk(BaseModel):
    chunk_id: str
    chunk_text: str
    similarity_score: float


class EvidenceRetrievalResponse(BaseModel):
    results: list[EvidenceChunk]


@router.post(
    "/retrieve-evidence",
    response_model=EvidenceRetrievalResponse,
    summary="Vector-only ground-truth evidence retrieval (MDS)",
)
async def retrieve_evidence(
    payload: EvidenceRetrievalRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
):
    """
    Replaces MDS's old direct SQL join against document_chunks/
    document_summaries (is_ground_truth = TRUE, effective_from/
    effective_to window, tenant/org scoping via Postgres RLS) — that
    schema no longer exists. Same filtering semantics here via
    build_filter()'s is_ground_truth + as_of params: is_ground_truth is
    denormalized onto every chunk's own Qdrant payload (same as it used
    to be denormalized onto every document_chunks row), and as_of=today
    reproduces "effective_from is null or already passed" /
    "effective_to is null or not yet expired" exactly.

    Vector-only (dense/cosine) for v1, per MDS's own spec — their
    fts_query hybrid path is not implemented here; add a sparse leg
    (vector_store.search(mode="hybrid", ...)) later if they need it.

    similarity_score is Qdrant's raw cosine similarity for this
    dense-only query — no metric conversion needed, the collection is
    already configured for cosine distance (standard for OpenAI
    embeddings), matching MDS's requested "similarity_metric": "cosine".
    """
    try:
        query_filter = build_filter(
            tenant_id=tenant_id, org_unit_id=org_unit_id, org_ids=payload.org_ids,
            is_ground_truth=True, as_of=date.today(), metadata=payload.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # run_in_threadpool: this repo runs uvicorn --workers 1 — a
        # blocking Qdrant call here would stall the whole API for every
        # other in-flight request, same reasoning as search.py/chat.py.
        results = await run_in_threadpool(
            vector_store.search,
            query_filter=query_filter, dense_vector=payload.query_embedding,
            mode="semantic", limit=payload.top_k, score_threshold=payload.similarity_threshold,
        )
    except Exception as e:
        logger.exception("MDS evidence retrieval failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Evidence retrieval failed. Contact support with the X-Request-ID response header if this persists.",
        )

    return EvidenceRetrievalResponse(
        results=[
            EvidenceChunk(
                chunk_id=r["id"],
                # Image-role chunks carry their content in image_caption,
                # not chunk_text (which is empty/absent for them) — a match
                # on an image chunk is real evidence, not something to drop,
                # so fall back to its caption instead of returning blank text.
                chunk_text=r.get("chunk_text") or r.get("image_caption") or "",
                similarity_score=r["score"],
            )
            for r in results
        ]
    )
