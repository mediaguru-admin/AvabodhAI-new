"""
api/routes/health.py
--------------------
Health check endpoints for monitoring.
"""

from fastapi import APIRouter
from sqlalchemy import text

from db.database import engine
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get("/", summary="Basic health check")
async def health():
    return {"status": "ok", "service": "Avabodh API"}


@router.get("/db", summary="Database health check")
async def health_db():
    """Check if PostgreSQL is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return {"status": "error", "database": str(e)}


@router.get("/models", summary="Local ML model health check")
async def health_models():
    """
    Actually TRIES to load the two local ONNX models that retrieval quality
    depends on, rather than reporting on config.

    2026-09-23 — added after both were found failing silently in
    production-like conditions. fastembed loads them lazily, and
    pipeline/retriever.py catches each failure, logs a warning, and
    continues with degraded results: a SPLADE failure turns hybrid search
    into dense-only (no keyword matching at all), and a reranker failure
    means results are never re-sorted by relevance. The chat answer still
    comes back and still looks completely normal, so the only way anyone
    noticed was a wrong figure in an answer days later.

    Both models are memory-hungry and fail with ONNXRuntime "bad
    allocation" when the box is short on RAM, which makes this
    INTERMITTENT — the same deployment can be healthy and degraded hours
    apart with no code change. That is exactly why this probes live
    instead of trusting a startup check.

    Returns 200 either way — "degraded" is a real, serviceable state, not
    an outage. Poll this; alert on status != "ok".
    """
    from pipeline import embedder

    results: dict[str, dict] = {}

    try:
        embedder.rerank("healthcheck", ["healthcheck document"])
        results["reranker"] = {"status": "ok"}
    except Exception as e:
        results["reranker"] = {"status": "unavailable", "error": str(e)[:200],
                               "impact": "results are not re-sorted by relevance"}

    try:
        indices, _ = embedder.embed_sparse_query("healthcheck")
        results["sparse_splade"] = {"status": "ok", "terms": len(indices)}
    except Exception as e:
        results["sparse_splade"] = {"status": "unavailable", "error": str(e)[:200],
                                    "impact": "hybrid search degrades to dense-only, no keyword matching"}

    degraded = [name for name, r in results.items() if r["status"] != "ok"]
    if degraded:
        logger.error("Model health check: DEGRADED — unavailable: %s", ", ".join(degraded))
    return {"status": "degraded" if degraded else "ok", "degraded": degraded, "models": results}


@router.get("/qdrant", summary="Qdrant health check")
async def health_qdrant():
    """Check if Qdrant (the chunk/vector store) is reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        return {"status": "error", "qdrant": str(e)}