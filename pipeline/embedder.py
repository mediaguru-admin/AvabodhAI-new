"""
pipeline/embedder.py
--------------------
Embedding + reranking singletons only — NO database writes here any more
(that moved to pipeline/vector_store.py::upsert_chunks, called from
pipeline/ingest.py). This module used to also build DocumentChunk ORM rows
and bulk_save_objects() them into Postgres; that responsibility is gone
now that chunk storage lives in Qdrant.

Three models, all lazily constructed once (module-level singletons —
fastembed's SPLADE/reranker models are backed by ONNX sessions that are
expensive to (re)load, and are safe to reuse across requests):

  dense  — OpenAI (settings.EMBEDDING_MODEL, e.g. text-embedding-3-small)
  sparse — fastembed SPLADE (settings.SPARSE_MODEL) — learned term
           weighting + expansion, chosen 2026-08-21 over BM25 (subsumes
           BM25's exact-match behavior, still lightweight/CPU-only via
           onnxruntime, no torch needed for this leg)
  rerank — fastembed cross-encoder (settings.RERANK_MODEL,
           BAAI/bge-reranker-v2-m3) — CPU-capable, no GPU/torch, same
           fastembed/ONNX dependency family as the sparse model
"""

import time
from functools import lru_cache
from typing import Optional

from fastembed import SparseTextEmbedding
# TextCrossEncoder lives in its own submodule, not re-exported at the
# fastembed top level (confirmed against the installed fastembed==0.8.0 —
# `from fastembed import TextCrossEncoder` raises ImportError).
#
# 2026-08-21 (v3, reverted back from FlagEmbedding): tried loading the
# originally-decided "BAAI/bge-reranker-v2-m3" via BAAI's own FlagEmbedding
# library (real torch/transformers weights, not fastembed's ONNX
# conversion) — that model isn't in fastembed's supported list at all, so
# switching runtimes was the only way to use it under that exact name. In
# practice it was too heavy for this dev box: torch+transformers pulled in
# ~1.1GB of weights, hit a Windows page-file allocation failure on first
# load, needed a transformers<5 pin to dodge a separate FlagEmbedding/
# transformers-5.x incompatibility, and even once working, the lazy
# first-load cost alone stalled a real request for 392s (confirmed live in
# avabodh.log) because nothing warmed it up ahead of the first user
# request. Reverted to fastembed's TextCrossEncoder — ONNX runtime,
# already a dependency for SPLADE, no torch — using a model fastembed
# actually supports (see RERANK_MODEL in config/settings.py).
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache
def _dense_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL)


@lru_cache
def _sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.SPARSE_MODEL)


@lru_cache
def _rerank_model() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=settings.RERANK_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Dense (OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

def embed_dense_query(text: str) -> list[float]:
    return _dense_client().embed_query(text)


def embed_dense_batch(texts: list[str], batch_size: Optional[int] = None) -> list[list[float]]:
    """
    Batched dense embedding — mirrors the old store_chunk_embeddings() batching behavior.

    Each batch gets settings.EMBEDDING_MAX_RETRIES retries (most failures
    here are transient: rate limits, network blips) before giving up.
    Exhausting retries on any one batch stops the whole document rather
    than skipping that batch and moving to the next — a skipped batch
    means its chunks are silently absent from the vector store with no
    indication anything is missing, which is worse than an explicit,
    retryable failure. A correlated failure (rate limit, bad key) would
    likely fail the next batch too anyway, so there's little to gain from
    pressing on.
    """
    if not texts:
        return []
    batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
    max_retries = max(0, settings.EMBEDDING_MAX_RETRIES)
    client = _dense_client()
    out: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch = texts[i : i + batch_size]
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):  # +1 for the initial try, +1 for range's exclusivity
            try:
                out.extend(client.embed_documents(batch))
                logger.info("Dense-embedded batch %d/%d (attempt %d)", batch_num, total_batches, attempt)
                last_error = None
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Dense embedding batch %d/%d failed (attempt %d/%d): %s",
                    batch_num, total_batches, attempt, max_retries + 1, e,
                )
                if attempt <= max_retries:
                    time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s, ...
        if last_error is not None:
            raise RuntimeError(
                f"Dense embedding failed at batch {batch_num} after {max_retries + 1} attempt(s): {last_error}"
            ) from last_error
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sparse (fastembed SPLADE)
# ─────────────────────────────────────────────────────────────────────────────

def embed_sparse_query(text: str) -> tuple[list[int], list[float]]:
    """One sparse embedding, for a query at search time."""
    result = next(_sparse_model().embed([text]))
    return list(result.indices.tolist()), list(result.values.tolist())


def embed_sparse_batch(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """
    Batched sparse embedding, for chunk text at ingest time.

    Chunked into settings.SPARSE_BATCH_SIZE-sized calls rather than one
    single_model().embed(texts) call — SPLADE's forward pass allocates a
    (batch, seq_len, vocab) float32 array, so passing all of a large
    document's chunks (e.g. 50) at once demands one huge allocation
    (~745MiB observed) instead of several small, bounded ones.
    """
    if not texts:
        return []
    batch_size = settings.SPARSE_BATCH_SIZE
    model = _sparse_model()
    out: list[tuple[list[int], list[float]]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out.extend(
            (list(r.indices.tolist()), list(r.values.tolist()))
            for r in model.embed(batch)
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reranker (fastembed cross-encoder)
# ─────────────────────────────────────────────────────────────────────────────

def rerank(query: str, documents: list[str]) -> list[float]:
    """
    Returns one relevance score per document, same order as the input
    list — caller sorts/zips these back against their own candidate
    objects (see pipeline/retriever.py::search()).
    """
    if not documents:
        return []
    scores = _rerank_model().rerank(query, documents)
    return list(scores)
