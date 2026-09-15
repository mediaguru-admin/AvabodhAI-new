"""
pipeline/vector_store.py
-------------------------
Single point of Qdrant access — nothing else in this codebase imports
qdrant_client directly. Same discipline pipeline/retriever.py's old
_build_where() enforced for SQL: one chokepoint, so isolation logic can't
drift between call sites.

Chunk text, dense vector (OpenAI), sparse vector (fastembed SPLADE), and
all filterable metadata (tenant_id, org_unit_id, document_id, effective
dates, allowlisted client metadata) live in Qdrant now — see
AvabodhAIDocumentRetrievalArchitecture.md and IMPLEMENTATION_PLAN (2).md.
Postgres (db/models.py::Document) holds only document-level registry state.

Qdrant has NO Row-Level-Security equivalent. Every read and every delete
below takes tenant_id (and callers are expected to also filter org_unit_id
via pipeline/retriever.py::build_filter()) — there is no "unscoped" query
path in this module. Tenant isolation for chunk data is enforced ENTIRELY
here; there is no second layer the way Postgres has RLS as a backstop.

Two collections:
  avabodh_chunks         — document + image chunks (role='text'|'image')
  avabodh_chat_messages  — chat message vectors (replaces the old
                            ChatMessage.embedding pgvector column)

Point IDs are deterministic (uuid5 of stable inputs), NOT random — this is
what makes upsert_chunks() idempotent: re-running ingestion for the same
document (a retry, a reprocess) overwrites the same points instead of
creating duplicates. See IMPLEMENTATION_PLAN (2).md §2.2 / architecture
doc §26.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, Union

from qdrant_client import QdrantClient, models

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Namespace for deterministic chunk point IDs — fixed, arbitrary UUID,
# never change this or every existing point ID changes on next deploy.
_POINT_NS = uuid.UUID("6f6b6e77-3a1a-4e6e-9c7a-5d6a2f9b1c3e")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "splade"


def _dense_vector_size() -> int:
    return settings.OLLAMA_EMBEDDING_DIMENSIONS if settings.use_ollama else settings.EMBEDDING_DIMENSIONS


def _client() -> QdrantClient:
    """
    Module-singleton-ish client. Not cached in a global on purpose — the
    qdrant-client library itself pools connections internally, and a
    plain function call here keeps this trivially mockable/patchable in
    tests (tests point QDRANT_URL at QdrantClient(":memory:") instead).
    """
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def chunk_point_id(document_id: str, role: str, chunk_index: int) -> str:
    """Deterministic point ID — same (document_id, role, chunk_index) always maps to the same point."""
    return str(uuid.uuid5(_POINT_NS, f"{document_id}:{role}:{chunk_index}"))


# ─────────────────────────────────────────────────────────────────────────────
# Collection provisioning — called by scripts/init_qdrant.py, and once at
# app startup (main.py lifespan), idempotently.
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collections() -> None:
    """
    Create avabodh_chunks / avabodh_chat_messages (+ payload indexes) if
    they don't already exist. Safe to call on every startup — Qdrant's
    create_collection is not idempotent by itself (raises if it already
    exists), so existence is checked first.
    """
    client = _client()
    _ensure_chunks_collection(client)
    _ensure_chat_collection(client)


def _ensure_chunks_collection(client: QdrantClient) -> None:
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=_dense_vector_size(),
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        logger.info("Created Qdrant collection '%s'.", settings.QDRANT_COLLECTION)

    # Only fields actually used in filters get an index — per architecture
    # doc §27 ("payload indexes should only be created for fields used in
    # filtering"). is_tenant=True on tenant_id is Qdrant's own recommended
    # multi-tenancy hint (routes tenant-scoped queries more efficiently).
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "tenant_id", models.KeywordIndexParams(type="keyword", is_tenant=True))
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "org_unit_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "document_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "doc_name", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "role", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "is_ground_truth", models.PayloadSchemaType.BOOL)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "effective_from", models.PayloadSchemaType.DATETIME)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "effective_to", models.PayloadSchemaType.DATETIME)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "chunk_index", models.PayloadSchemaType.INTEGER)
    _ensure_payload_index(client, settings.QDRANT_COLLECTION, "image_bytes_hash", models.PayloadSchemaType.KEYWORD)
    for key in FILTERABLE_METADATA_KEYS:
        _ensure_payload_index(client, settings.QDRANT_COLLECTION, f"meta_{key}", models.PayloadSchemaType.KEYWORD)


def _ensure_chat_collection(client: QdrantClient) -> None:
    if not client.collection_exists(settings.QDRANT_CHAT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_CHAT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=_dense_vector_size(),
                    distance=models.Distance.COSINE,
                ),
            },
        )
        logger.info("Created Qdrant collection '%s'.", settings.QDRANT_CHAT_COLLECTION)

    _ensure_payload_index(client, settings.QDRANT_CHAT_COLLECTION, "tenant_id", models.KeywordIndexParams(type="keyword", is_tenant=True))
    _ensure_payload_index(client, settings.QDRANT_CHAT_COLLECTION, "org_unit_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.QDRANT_CHAT_COLLECTION, "thread_id", models.PayloadSchemaType.KEYWORD)


def _ensure_payload_index(client: QdrantClient, collection: str, field_name: str, schema) -> None:
    try:
        client.create_payload_index(collection_name=collection, field_name=field_name, field_schema=schema)
    except Exception as e:
        # Already exists (or a transient issue) — idempotent by design,
        # don't fail startup over a duplicate-index error.
        logger.debug("Payload index '%s' on '%s' not (re)created: %s", field_name, collection, e)


# Metadata keys clients are allowed to filter search by. Kept controlled
# (architecture doc §10/§28 Rule 6) rather than indexing arbitrary
# client-supplied keys — an unbounded set of payload indexes doesn't scale
# and lets a caller shape the schema by accident.
FILTERABLE_METADATA_KEYS = {
    "category", "country", "state", "department",
    "classification", "language", "source", "document_type",
}


# ─────────────────────────────────────────────────────────────────────────────
# Chunk points — write path
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkPoint:
    """One chunk (text or image) ready to upsert into avabodh_chunks."""
    document_id: str
    tenant_id: str
    org_unit_id: str
    role: str                      # "text" | "image"
    chunk_index: int
    chunk_text: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    doc_hash: str = ""
    doc_name: str = ""
    total_chunks: int = 1
    chunk_size: int = 0
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_type: Optional[str] = None
    topic: Optional[str] = None
    table_html: Optional[str] = None   # original <table> markup, when this chunk contains a table (see pipeline/chunker.py)
    is_ground_truth: bool = False
    effective_from: Optional[str] = None   # ISO 8601 or None
    effective_to: Optional[str] = None
    embedding_model: str = ""
    source_path: Optional[str] = None
    language: str = "English"
    created_at: Optional[str] = None
    metadata: dict = field(default_factory=dict)   # allowlisted keys only — see FILTERABLE_METADATA_KEYS
    image_fields: dict = field(default_factory=dict)  # image_url, image_caption, image_type, etc. (role='image' only)

    def to_point(self) -> models.PointStruct:
        payload = {
            "tenant_id": self.tenant_id,
            "org_unit_id": self.org_unit_id,
            "document_id": self.document_id,
            "doc_hash": self.doc_hash,
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_text": self.chunk_text,
            "chunk_size": self.chunk_size,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "chunk_type": self.chunk_type,
            "topic": self.topic,
            "table_html": self.table_html,
            "role": self.role,
            "is_ground_truth": self.is_ground_truth,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "embedding_model": self.embedding_model,
            "source_path": self.source_path,
            "language": self.language,
            "created_at": self.created_at,
            **{f"meta_{k}": v for k, v in self.metadata.items() if k in FILTERABLE_METADATA_KEYS},
            **self.image_fields,
        }
        return models.PointStruct(
            id=chunk_point_id(self.document_id, self.role, self.chunk_index),
            vector={
                DENSE_VECTOR_NAME: self.dense_vector,
                SPARSE_VECTOR_NAME: models.SparseVector(indices=self.sparse_indices, values=self.sparse_values),
            },
            payload=payload,
        )


def upsert_chunks(points: list[ChunkPoint], batch_size: int = 100) -> int:
    """Batched upsert — idempotent (deterministic point IDs, see chunk_point_id())."""
    if not points:
        return 0
    client = _client()
    total = 0
    for i in range(0, len(points), batch_size):
        batch = [p.to_point() for p in points[i : i + batch_size]]
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=batch)
        total += len(batch)
    logger.info("Upserted %d chunk point(s) into '%s'.", total, settings.QDRANT_COLLECTION)
    return total


def image_hash_exists(tenant_id: str, org_unit_id: str, image_bytes_hash: str) -> bool:
    """
    Dedup check — True if an image with this exact byte hash is already
    indexed for this tenant+org_unit. Replaces the old Postgres-based
    check_image_hash_exists() (db/models.py::DocumentChunk.image_bytes_hash
    no longer exists — chunk storage moved to Qdrant).
    """
    client = _client()
    count = client.count(
        collection_name=settings.QDRANT_COLLECTION,
        count_filter=models.Filter(must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(key="org_unit_id", match=models.MatchValue(value=org_unit_id)),
            models.FieldCondition(key="role", match=models.MatchValue(value="image")),
            models.FieldCondition(key="image_bytes_hash", match=models.MatchValue(value=image_bytes_hash)),
        ]),
        exact=False,
    )
    return count.count > 0


def count_chunks(query_filter: models.Filter) -> int:
    """
    Cheap filtered count (no vector search) — used by pipeline/retriever.py
    to size SEARCH_CANDIDATES dynamically per query: a 4-page document's
    entire chunk set fits well under the old fixed candidate count, but a
    60-page document's true best match can rank outside a fixed top-20
    window and never reach reranking at all. exact=False (approximate
    count) is intentional — this only needs to be right enough to decide
    "roughly how many chunks does this document have," not exact to the
    point of adding real query latency.
    """
    client = _client()
    result = client.count(
        collection_name=settings.QDRANT_COLLECTION,
        count_filter=query_filter,
        exact=False,
    )
    return result.count


def delete_document_points(tenant_id: str, document_id: str) -> None:
    """
    Delete every chunk (text + image) belonging to one document, in one
    filter-based call — Qdrant supports delete-by-filter natively (unlike
    Pinecone serverless, which the original plan assumed), so this needs
    no prior enumeration of point IDs from Postgres.
    """
    client = _client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ])
        ),
    )
    logger.info("Deleted Qdrant points for document_id=%s (tenant=%s).", document_id, tenant_id)


def scroll_document_chunks(
    tenant_id: str,
    org_unit_id: str,
    document_id: str,
    role: Optional[str] = None,
    limit: int = 10_000,
) -> list[dict]:
    """
    All chunks for one document, ordered by chunk_index — backs
    GET /search/chunks/{doc_id}. Qdrant's scroll() doesn't sort server-side,
    so this fetches (bounded by `limit`) and sorts client-side; `limit`
    is generous but not unbounded, since a single document's chunk count
    is expected to stay well under it in practice.
    """
    client = _client()
    must = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
        models.FieldCondition(key="org_unit_id", match=models.MatchValue(value=org_unit_id)),
        models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
    ]
    if role:
        must.append(models.FieldCondition(key="role", match=models.MatchValue(value=role)))

    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=models.Filter(must=must),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    rows = [{"id": str(p.id), **p.payload} for p in points]
    rows.sort(key=lambda r: r.get("chunk_index", 0))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Search — hybrid (dense + sparse, Qdrant-native RRF fusion)
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query_filter: models.Filter,
    dense_vector: Optional[list[float]] = None,
    sparse_indices: Optional[list[int]] = None,
    sparse_values: Optional[list[float]] = None,
    mode: str = "hybrid",
    limit: int = 15,
    score_threshold: Optional[float] = None,
    query_point_id: Optional[Union[str, int]] = None,
    lookup_from_collection: Optional[str] = None,
) -> list[dict]:
    """
    mode: "hybrid" (dense + sparse prefetch, fused server-side via RRF),
    "semantic" (dense only), "keyword" (sparse only).

    score_threshold: minimum score a hit must clear to be returned (for
    hybrid, this applies to the fused RRF score, not either leg's raw
    score — the two legs aren't on comparable scales, so per-leg
    thresholding isn't offered here). None (default) falls back to
    settings.SEARCH_SCORE_THRESHOLD; pass an explicit value to override
    that per-call. Both being None/unset means no floor, matching
    Qdrant's own default behavior.

    query_point_id + lookup_from_collection: query using an EXISTING
    point's own stored vector instead of a freshly-embedded query — e.g.
    "find chunks similar to this other chunk" (query_point_id from
    avabodh_chunks itself) or "find chunks similar to this past chat
    message" (query_point_id from avabodh_chat_messages,
    lookup_from_collection=settings.QDRANT_CHAT_COLLECTION). When
    lookup_from_collection is set, Qdrant looks the point up in that
    DIFFERENT collection (its `lookup_from`) rather than the one being
    searched; when unset, the point is looked up in the same collection
    being queried. Overrides dense_vector/sparse_indices/sparse_values —
    only one query source (a fresh vector, or an existing point) applies
    per call.

    Returns dicts with `id`, `score`, and the point's payload merged in —
    same shape regardless of mode, so callers don't need to branch.
    """
    client = _client()
    threshold = score_threshold if score_threshold is not None else settings.SEARCH_SCORE_THRESHOLD
    lookup_from = models.LookupLocation(collection=lookup_from_collection) if lookup_from_collection else None

    if mode == "semantic":
        query_value = query_point_id if query_point_id is not None else dense_vector
        if query_value is None:
            raise ValueError("semantic search requires dense_vector or query_point_id")
        result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_value,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            score_threshold=threshold,
            lookup_from=lookup_from,
            with_payload=True,
        )
    elif mode == "keyword":
        query_value = query_point_id
        if query_value is None:
            if sparse_indices is None or sparse_values is None:
                raise ValueError("keyword search requires sparse_indices/sparse_values or query_point_id")
            query_value = models.SparseVector(indices=sparse_indices, values=sparse_values)
        result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_value,
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            score_threshold=threshold,
            lookup_from=lookup_from,
            with_payload=True,
        )
    else:  # hybrid
        dense_query = query_point_id if query_point_id is not None else dense_vector
        sparse_query = query_point_id
        if sparse_query is None:
            if sparse_indices is None or sparse_values is None:
                raise ValueError("hybrid search requires both dense and sparse vectors, or query_point_id")
            sparse_query = models.SparseVector(indices=sparse_indices, values=sparse_values)
        if dense_query is None:
            raise ValueError("hybrid search requires both dense and sparse vectors, or query_point_id")
        result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                # 2026-08-22: prefetch limit was hardcoded to the static
                # settings.SEARCH_CANDIDATES — meaning even when the
                # caller (pipeline/retriever.py) requests a larger dynamic
                # `limit` for a big document, RRF fusion still only had a
                # small fixed pool per leg to fuse from, silently
                # capping recall regardless of what the caller asked for.
                # Using `limit` itself here keeps the prefetch pool at
                # least as large as what fusion is actually being asked
                # to return.
                models.Prefetch(
                    query=dense_query, using=DENSE_VECTOR_NAME,
                    filter=query_filter, limit=limit,
                    lookup_from=lookup_from,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter, limit=limit,
                    lookup_from=lookup_from,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )

    return [{"id": str(pt.id), "score": pt.score, **(pt.payload or {})} for pt in result.points]


def batch_search(
    query_filter: models.Filter,
    dense_vectors: list[list[float]],
    sparse_vectors: list[tuple[list[int], list[float]]],
    limit: int = 10,
    score_threshold: Optional[float] = None,
) -> list[list[dict]]:
    """
    Hybrid search for N queries in ONE network round trip via Qdrant's
    query_batch_points(), instead of N sequential query_points() calls —
    the fix for multi-query retrieval (pipeline/chat.py's condensed
    primary_query + alternate_queries) previously doing one full
    embed+search round trip per variant, serially. Server-side, Qdrant
    still evaluates each request independently (this is request batching
    for transport, not a single fused multi-vector query) — the RRF fusion
    below is still per-query; merging ACROSS queries is the caller's job
    (pipeline/retriever.py::multi_query_search()), same as before.

    dense_vectors/sparse_vectors must be the same length and same order as
    the queries they came from — pair with pipeline/embedder.py's
    embed_dense_batch()/embed_sparse_batch(), which already batch multiple
    texts into one OpenAI/SPLADE call each, so an N-query multi-query
    search now costs 1 dense-embed call + 1 sparse-embed call + 1 Qdrant
    call total, not N of each.
    """
    if len(dense_vectors) != len(sparse_vectors):
        raise ValueError("dense_vectors and sparse_vectors must be the same length")
    if not dense_vectors:
        return []

    client = _client()
    threshold = score_threshold if score_threshold is not None else settings.SEARCH_SCORE_THRESHOLD
    requests = [
        models.QueryRequest(
            prefetch=[
                # 2026-08-22: use `limit` (the caller's dynamic candidate
                # count, see pipeline/retriever.py::_dynamic_search_candidates())
                # instead of the static settings.SEARCH_CANDIDATES — same
                # fix as vector_store.py::search()'s hybrid branch, same
                # reasoning: a fixed small prefetch pool per leg caps
                # recall regardless of what limit the caller actually asked for.
                models.Prefetch(
                    query=dense_vec, using=DENSE_VECTOR_NAME,
                    filter=query_filter, limit=limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_idx, values=sparse_val),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter, limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            filter=query_filter,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )
        for dense_vec, (sparse_idx, sparse_val) in zip(dense_vectors, sparse_vectors)
    ]

    responses = client.query_batch_points(collection_name=settings.QDRANT_COLLECTION, requests=requests)
    return [
        [{"id": str(pt.id), "score": pt.score, **(pt.payload or {})} for pt in resp.points]
        for resp in responses
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Chat message vectors (avabodh_chat_messages) — backs GET /chat/search
# ─────────────────────────────────────────────────────────────────────────────

def upsert_chat_message(
    message_id: str,
    tenant_id: str,
    org_unit_id: str,
    thread_id: str,
    role: str,
    content: str,
    dense_vector: list[float],
    created_at: Optional[str] = None,
) -> None:
    client = _client()
    client.upsert(
        collection_name=settings.QDRANT_CHAT_COLLECTION,
        points=[models.PointStruct(
            id=message_id,
            vector={DENSE_VECTOR_NAME: dense_vector},
            payload={
                "tenant_id": tenant_id,
                "org_unit_id": org_unit_id,
                "thread_id": thread_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "created_at": created_at,
            },
        )],
    )


def delete_thread_messages(tenant_id: str, thread_id: str) -> None:
    client = _client()
    client.delete(
        collection_name=settings.QDRANT_CHAT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="thread_id", match=models.MatchValue(value=thread_id)),
            ])
        ),
    )


def search_chat(
    tenant_id: str,
    org_unit_id: str,
    dense_vector: list[float],
    top_k: int = 10,
) -> list[dict]:
    client = _client()
    result = client.query_points(
        collection_name=settings.QDRANT_CHAT_COLLECTION,
        query=dense_vector,
        using=DENSE_VECTOR_NAME,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(key="org_unit_id", match=models.MatchValue(value=org_unit_id)),
        ]),
        limit=top_k,
        with_payload=True,
    )
    return [{"id": str(pt.id), "score": pt.score, **(pt.payload or {})} for pt in result.points]
