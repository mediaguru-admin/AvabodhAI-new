"""
pipeline/retriever.py
---------------------
Rewritten 2026-08-21 (Qdrant migration). Retrieval is now:

1. build_filter() — the SINGLE isolation chokepoint. Qdrant has no RLS
   equivalent, so this function is the only thing standing between one
   tenant's search and another's data. EVERY read in this module goes
   through it. tenant_id is mandatory; org_unit_id defaults to [header
   org] if not explicitly widened by the caller.
2. search() — dense (OpenAI) + sparse (fastembed SPLADE) via
   pipeline/vector_store.py, fused server-side by Qdrant (RRF), or either
   leg alone for "semantic"/"keyword" mode.
3. rerank via fastembed cross-encoder (pipeline/embedder.py::rerank) over
   the fused candidates.
4. retrieve() — used by chat: same search() + rerank, plus the
   image-in-chat merge (unchanged behavior from before this rewrite).

is_ground_truth is no longer a hard gate (previously unconditionally
enforced) — it's now an OPTIONAL filter, per Vijay's decision
(IMPLEMENTATION_PLAN (2).md decisions table). Pass is_ground_truth=True
to restrict to ground-truth-only documents; omit it to search everything.
"""

import time
from datetime import date
from typing import Optional

import numpy as np
from qdrant_client import models

from pipeline import embedder, vector_store
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# 2026-09-23: shortlist_documents() below was measured spending 2.35s on
# count_document_summaries() ALONE, on a local Qdrant instance with a few
# hundred points — for a call whose only purpose is "decide whether to
# bother narrowing," made fresh on every single chat query. The document
# count changes only on upload/delete, so a short TTL cache turns this
# into a non-issue: {(tenant_id, org_unit_id): (count, fetched_at)}.
# Staleness cost is self-limiting and cheap either way it's wrong — a
# tenant that JUST crossed DOC_SHORTLIST_LIMIT might narrow one query
# "too early" for up to the TTL, and shortlist_documents()'s own
# docstring already treats narrowing-when-not-yet-warranted as a lossy-
# but-tolerable extra hop, not a correctness bug.
_DOC_COUNT_CACHE: dict[tuple[str, str], tuple[int, float]] = {}
_DOC_COUNT_CACHE_TTL_SECONDS = 60

# Metadata keys clients are allowed to filter by — mirrors
# vector_store.FILTERABLE_METADATA_KEYS (the set that actually gets
# indexed); kept here too so build_filter() can validate/422 without
# importing vector_store's index-creation concerns into this module.
FILTERABLE_METADATA_KEYS = vector_store.FILTERABLE_METADATA_KEYS


def build_filter(
    tenant_id: str,
    org_unit_id: str,
    org_ids: Optional[list[str]] = None,
    document_ids: Optional[list[str]] = None,
    doc_name: Optional[str] = None,
    role: Optional[str] = None,
    is_ground_truth: Optional[bool] = None,
    metadata: Optional[dict[str, list[str]]] = None,
    as_of: Optional[date] = None,
) -> models.Filter:
    """
    The single isolation + filtering chokepoint for every Qdrant read.
    tenant_id is always required and always applied. org_unit_id (from
    the X-Org-Unit-ID header) is the default scope; pass org_ids to widen
    to multiple departments within the SAME tenant — never across tenants.

    metadata keys not in FILTERABLE_METADATA_KEYS raise ValueError — the
    caller (api/routes/search.py) turns that into a 422, per architecture
    doc §10/§28 Rule 6 ("keep filterable metadata controlled").
    """
    must: list[models.Condition] = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
    ]

    scoped_orgs = org_ids if org_ids else [org_unit_id]
    must.append(models.FieldCondition(key="org_unit_id", match=models.MatchAny(any=scoped_orgs)))

    if document_ids:
        must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=[str(d) for d in document_ids])))
    if doc_name:
        must.append(models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)))
    if role:
        must.append(models.FieldCondition(key="role", match=models.MatchValue(value=role)))
    if is_ground_truth is not None:
        must.append(models.FieldCondition(key="is_ground_truth", match=models.MatchValue(value=is_ground_truth)))

    if metadata:
        for key, values in metadata.items():
            if key not in FILTERABLE_METADATA_KEYS:
                raise ValueError(f"'{key}' is not a filterable metadata key. Allowed: {sorted(FILTERABLE_METADATA_KEYS)}")
            if values:
                must.append(models.FieldCondition(key=f"meta_{key}", match=models.MatchAny(any=values)))

    if as_of is not None:
        as_of_str = as_of.isoformat()
        # (effective_from IS NULL OR effective_from <= as_of)
        must.append(models.Filter(should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key="effective_from")),
            models.FieldCondition(key="effective_from", range=models.DatetimeRange(lte=as_of_str)),
        ]))
        # (effective_to IS NULL OR effective_to >= as_of)
        must.append(models.Filter(should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key="effective_to")),
            models.FieldCondition(key="effective_to", range=models.DatetimeRange(gte=as_of_str)),
        ]))

    return models.Filter(must=must)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _mmr_select(candidates: list[dict], top_k: int, lambda_mult: float) -> list[dict]:
    """
    Maximal Marginal Relevance: greedily picks top_k candidates balancing
    relevance (rerank/fusion score) against diversity (how different a
    candidate is from what's already been picked) — instead of naive
    top-K-by-score, which can end up entirely clustered around one or two
    chunks/documents that happen to score highest, crowding out other
    genuinely relevant content once shortlist_documents() has already
    narrowed the search to several relevant documents.

    lambda_mult: 1.0 = pure relevance (identical to plain top-K), 0.0 =
    pure diversity. Falls back to plain top-K if any candidate is missing
    a dense vector (with_vectors=False upstream, or an older code path) —
    MMR needs real embeddings to measure diversity, this must never be a
    hard requirement for search to work at all.
    """
    if not candidates:
        return []
    if not all(c.get("_dense_vector") for c in candidates):
        return candidates[:top_k]

    scores = [c.get("rerank_score", c.get("score", 0.0)) for c in candidates]
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    normalized = [((s - lo) / spread) if spread > 0 else 1.0 for s in scores]

    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < top_k:
        if not selected:
            best_idx = max(remaining, key=lambda i: normalized[i])
        else:
            def mmr_score(i: int) -> float:
                relevance = normalized[i]
                diversity = max(
                    _cosine_similarity(candidates[i]["_dense_vector"], candidates[j]["_dense_vector"])
                    for j in selected
                )
                return lambda_mult * relevance - (1 - lambda_mult) * diversity
            best_idx = max(remaining, key=mmr_score)
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def _dynamic_search_candidates(query_filter: models.Filter) -> int:
    """
    How many pre-rerank candidates to fetch for THIS query's scope —
    min(actual chunk count in scope, SEARCH_CANDIDATES_MAX), so a small
    document doesn't waste time fetching more candidates than it has, and
    a large document isn't capped at a fixed count that only covered a
    fraction of it. Falls back to the static settings.SEARCH_CANDIDATES on
    any count failure (Qdrant hiccup, etc.) — this must never be a hard
    blocker for search working at all.
    """
    try:
        chunk_count = vector_store.count_chunks(query_filter)
        if chunk_count > 0:
            return min(chunk_count, settings.SEARCH_CANDIDATES_MAX)
    except Exception as e:
        logger.warning("Dynamic SEARCH_CANDIDATES sizing failed, falling back to static value: %s", e)
    return settings.SEARCH_CANDIDATES


def search(
    query: str,
    query_filter: models.Filter,
    mode: str = "hybrid",
    top_k: int = 10,
    do_rerank: bool = True,
    score_threshold: Optional[float] = None,
    query_point_id: Optional[str] = None,
    lookup_from_collection: Optional[str] = None,
    degraded: Optional[list[str]] = None,
) -> list[dict]:
    """
    mode: "hybrid" (dense+sparse, Qdrant-native RRF fusion — default),
    "semantic" (dense only), "keyword" (sparse only).

    When do_rerank, fetches settings.SEARCH_CANDIDATES candidates first
    (recall-oriented), then reranks with the fastembed cross-encoder and
    truncates to top_k (precision-oriented) — architecture doc §18.

    score_threshold: per-call override of settings.SEARCH_SCORE_THRESHOLD
    (.env-tunable) — None (default) just uses whatever's configured there.

    query_point_id + lookup_from_collection: bypass embedding `query`
    entirely and search using an EXISTING point's own stored vector
    instead — e.g. "find chunks similar to this chunk" (query_point_id
    from this same collection) or "find chunks similar to this past chat
    message" (query_point_id from avabodh_chat_messages,
    lookup_from_collection=settings.QDRANT_CHAT_COLLECTION). See
    pipeline/vector_store.py::search() for the full mechanics. `query`
    (the text arg) still needs to be passed but is ignored for embedding
    when query_point_id is set — only used for the rerank pass below.

    degraded: 2026-09-23 — optional out-param, mutated in place (never
    reassigned) with a short code per component that silently failed and
    fell back for THIS call: "sparse_embedding", "dense_embedding",
    "reranking". None (the default) means "caller doesn't care" — every
    failure path below still degrades and returns results exactly as
    before; this only adds a way to observe that it happened. Internal
    only: api/routes/chat.py persists the collected list onto
    ChatMessage.degraded (db/models.py) for later debugging, and it is
    deliberately NEVER put on ChatMessageResponse or any other API
    surface — an end user has no useful action to take on "the keyword
    search leg failed," this is for explaining a bad answer after the
    fact, not for display.

    Each leg is independently fault-tolerant, same property the old
    pgvector-era hybrid_search() had (its own docstring: "if either
    backend has a bad day, hybrid_search() degrades to whichever side is
    still working"): if mode="hybrid" and ONE of dense/sparse embedding
    fails, this degrades to the other leg alone rather than failing the
    whole search. Only fails outright if the mode being asked for has no
    surviving leg (e.g. mode="semantic" and dense embedding itself fails).
    """
    dense_vector = sparse_indices = sparse_values = None

    if query_point_id is None:
        if mode in ("hybrid", "semantic"):
            try:
                dense_vector = embedder.embed_dense_query(query)
            except Exception as e:
                logger.warning("Dense embedding failed (mode=%s): %s", mode, e)
                if degraded is not None:
                    degraded.append("dense_embedding")
                if mode == "semantic":
                    return []
                mode = "keyword"   # degrade hybrid -> keyword-only
        if mode in ("hybrid", "keyword"):
            try:
                sparse_indices, sparse_values = embedder.embed_sparse_query(query)
            except Exception as e:
                logger.error(
                    "DEGRADED: sparse embedding failed (mode=%s) — hybrid search is running "
                    "dense-only for this query, with no keyword matching: %s", mode, e,
                )
                if degraded is not None:
                    degraded.append("sparse_embedding")
                if mode == "keyword":
                    return []
                mode = "semantic"   # degrade hybrid -> semantic-only

    fetch_limit = _dynamic_search_candidates(query_filter) if do_rerank else top_k

    try:
        hits = vector_store.search(
            query_filter=query_filter,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            mode=mode,
            limit=fetch_limit,
            score_threshold=score_threshold,
            query_point_id=query_point_id,
            lookup_from_collection=lookup_from_collection,
            with_vectors=do_rerank,   # only needed for the MMR step below
        )
    except Exception as e:
        logger.error("Qdrant search failed (mode=%s): %s", mode, e)
        return []

    if not hits:
        return []

    if do_rerank and len(hits) > 1:
        try:
            scores = embedder.rerank(query, [h.get("chunk_text", "") for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
        except Exception as e:
            logger.error(
                "DEGRADED: reranking failed — results are NOT re-sorted by relevance for this "
                "query, falling back to raw fusion order: %s", e,
            )
            if degraded is not None:
                degraded.append("reranking")
    hits.sort(key=lambda h: h.get("rerank_score", h.get("score", 0.0)), reverse=True)

    if do_rerank:
        selected = _mmr_select(hits, top_k, settings.MMR_LAMBDA)
    else:
        selected = hits[:top_k]

    out = []
    for h in selected:
        h = dict(h)
        h.pop("_dense_vector", None)   # internal-only, MMR's input — never leaks past this function
        h["chunk_id"] = h["id"]
        h["similarity"] = h.get("rerank_score", h.get("score", 0.0))
        h["search_type"] = mode
        out.append(h)
    return out


def _cached_document_count(tenant_id: str, org_unit_id: str, summary_filter: models.Filter) -> int:
    """
    vector_store.count_document_summaries(), cached — see the module-level
    _DOC_COUNT_CACHE comment above for why. Raises through on a real
    failure exactly like an uncached call would; shortlist_documents()'s
    except block is what turns that into "search the whole corpus."
    """
    key = (tenant_id, org_unit_id)
    cached = _DOC_COUNT_CACHE.get(key)
    now = time.monotonic()
    if cached is not None and (now - cached[1]) < _DOC_COUNT_CACHE_TTL_SECONDS:
        return cached[0]

    count = vector_store.count_document_summaries(summary_filter)
    _DOC_COUNT_CACHE[key] = (count, now)
    return count


def shortlist_documents(
    query: str, tenant_id: str, org_unit_id: str, limit: Optional[int] = None,
) -> list[str]:
    """
    First-stage retrieval: narrows to the most relevant document_ids
    (via their embedded summaries, see pipeline/ingest.py's post-
    summarization upsert) BEFORE any chunk-level search runs, so
    chunk-level search's candidate pool (SEARCH_CANDIDATES_MAX) applies
    within a handful of relevant documents instead of the whole corpus.
    Without this, a fixed candidate pool doesn't scale — retrieval quality
    degrades as the number of documents grows, regardless of how large
    that pool is, since it's shared across every document in scope.

    Called only when the caller hasn't already scoped the query to a
    specific document (retrieve()/retrieve_multi() below) — a query that's
    already scoped has nothing to shortlist.

    Returns [] on any failure (no summary points yet — an older document,
    or Qdrant down — a tenant with zero documents, or a genuinely empty
    result), AND when the tenant/org has DOC_SHORTLIST_LIMIT documents or
    fewer — narrowing to "the top 5" out of a knowledge base that only
    HAS 5 (or fewer) documents doesn't shortlist anything, it just adds a
    lossy extra hop for no benefit, and at anything close to that count it
    risks silently excluding real, relevant documents whose summary
    happened to embed slightly less well than the ones that made the cut
    (see count_document_summaries() below — same dynamic-limit pattern as
    _dynamic_search_candidates() uses for chunks, for the same reason: a
    fixed cap must never bite at a scale where it wasn't needed yet).
    Callers MUST treat [] as "search the whole corpus," never as a hard
    failure — this is a precision optimization, not a correctness
    requirement.
    """
    limit = limit or settings.DOC_SHORTLIST_LIMIT

    summary_filter = models.Filter(must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
        models.FieldCondition(key="org_unit_id", match=models.MatchValue(value=org_unit_id)),
    ])

    try:
        doc_count = _cached_document_count(tenant_id, org_unit_id, summary_filter)
        if doc_count <= limit:
            return []
    except Exception as e:
        logger.warning("Document count check failed, falling back to full-corpus search: %s", e)
        return []

    try:
        dense_vector = embedder.embed_dense_query(query)
        sparse_indices, sparse_values = embedder.embed_sparse_query(query)
    except Exception as e:
        logger.warning("Document shortlist embedding failed, falling back to full-corpus search: %s", e)
        return []

    try:
        hits = vector_store.search_document_summaries(
            query_filter=summary_filter,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            limit=limit,
        )
    except Exception as e:
        logger.warning("Document shortlist search failed, falling back to full-corpus search: %s", e)
        return []

    return [h["document_id"] for h in hits if h.get("document_id")]


def retrieve(
    query: str,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    is_ground_truth: Optional[bool] = None,
    degraded: Optional[list[str]] = None,
) -> list[dict]:
    """
    Chat's retrieval entry point — hybrid search + rerank, scoped to one
    tenant/org. Kept as a thin wrapper (search() already does the real
    work) so chat.py's call site doesn't need to build a Filter itself.

    When the caller hasn't already scoped to one document (doc_filter),
    this shortlists the most relevant documents first (shortlist_documents())
    before searching their chunks — see that function's docstring.

    degraded: see search()'s docstring — passed straight through.
    """
    document_ids = None
    if not doc_filter:
        document_ids = shortlist_documents(query=query, tenant_id=tenant_id, org_unit_id=org_unit_id) or None

    query_filter = build_filter(
        tenant_id=tenant_id, org_unit_id=org_unit_id, document_ids=document_ids,
        doc_name=doc_filter, role=role_filter, is_ground_truth=is_ground_truth,
    )
    results = search(query=query, query_filter=query_filter, mode="hybrid", top_k=top_k, degraded=degraded)
    if not results:
        logger.info("No chunks found for query: %s", query[:50])
    return results


def multi_query_search(
    queries: list[str],
    query_filter: models.Filter,
    mode: str = "hybrid",
    top_k: int = 10,
    degraded: Optional[list[str]] = None,
) -> list[dict]:
    """
    Retrieves candidates for EACH query variant independently (no
    per-variant rerank — reranking happens once, on the merged union,
    below), merges by chunk id (keeping the best fusion score seen across
    variants), then reranks the union against queries[0] (the primary/
    condensed query — the clearest single statement of intent) before
    truncating to top_k.

    queries[0] is expected to be a standalone, context-complete query
    (see pipeline/chat.py::generate_search_queries) — the alternates
    exist purely to broaden recall, not as a replacement for a good
    primary query.

    2026-08-21: previously issued one full embed+search round trip PER
    query variant, serially (N dense-embed calls, N sparse-embed calls, N
    Qdrant calls for N queries). Both embedder.embed_dense_batch() and
    embed_sparse_batch() already batch multiple texts into one call each
    (built for chunk ingestion, reused here as-is), and
    vector_store.batch_search() sends all N hybrid queries to Qdrant in
    ONE query_batch_points() round trip — so an N-query multi-query search
    now costs exactly 1 dense-embed call + 1 sparse-embed call + 1 Qdrant
    call, regardless of N, instead of 3N. mode is currently always hybrid
    here in practice (the only caller passes "hybrid"); non-hybrid modes
    fall back to the old per-query search() loop since batch_search() only
    implements the hybrid path.
    """
    if len(queries) == 1:
        return search(query=queries[0], query_filter=query_filter, mode=mode, top_k=top_k, do_rerank=True,
                      degraded=degraded)

    candidates_limit = _dynamic_search_candidates(query_filter)

    by_id: dict[str, dict] = {}
    batched_ok = False
    if mode == "hybrid":
        try:
            dense_vectors = embedder.embed_dense_batch(queries)
            sparse_vectors = embedder.embed_sparse_batch(queries)
            batches = vector_store.batch_search(
                query_filter=query_filter,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                limit=candidates_limit,
                with_vectors=True,   # needed for the MMR step below
            )
            for hits in batches:
                for h in hits:
                    existing = by_id.get(h["id"])
                    if existing is None or h.get("score", 0.0) > existing.get("score", 0.0):
                        by_id[h["id"]] = h
            batched_ok = True
        except Exception as e:
            logger.warning("Batched multi-query search failed — falling back to per-query search: %s", e)

    if not batched_ok:
        for q in queries:
            hits = search(query=q, query_filter=query_filter, mode=mode, top_k=candidates_limit, do_rerank=False,
                          degraded=degraded)
            for h in hits:
                existing = by_id.get(h["id"])
                if existing is None or h.get("score", 0.0) > existing.get("score", 0.0):
                    by_id[h["id"]] = h

    candidates = list(by_id.values())
    if not candidates:
        return []

    try:
        scores = embedder.rerank(queries[0], [c.get("chunk_text", "") for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
    except Exception as e:
        logger.error(
            "DEGRADED: multi-query reranking failed — results are NOT re-sorted by relevance for "
            "this query, falling back to per-variant fusion order: %s", e,
        )
        if degraded is not None:
            degraded.append("reranking")
    candidates.sort(key=lambda c: c.get("rerank_score", c.get("score", 0.0)), reverse=True)

    selected = _mmr_select(candidates, top_k, settings.MMR_LAMBDA)

    out = []
    for h in selected:
        h = dict(h)
        h.pop("_dense_vector", None)
        h["chunk_id"] = h["id"]
        h["similarity"] = h.get("rerank_score", h.get("score", 0.0))
        h["search_type"] = mode
        out.append(h)
    return out


def retrieve_multi(
    queries: list[str],
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    is_ground_truth: Optional[bool] = None,
    degraded: Optional[list[str]] = None,
) -> list[dict]:
    """
    Multi-query variant of retrieve() — see multi_query_search()'s docstring
    for the merge/rerank strategy, and retrieve()'s docstring for the
    document-shortlist stage this applies the same way (shortlisted once,
    against queries[0] — the condensed primary query).

    degraded: see search()'s docstring — passed straight through.
    """
    document_ids = None
    if not doc_filter and queries:
        document_ids = shortlist_documents(query=queries[0], tenant_id=tenant_id, org_unit_id=org_unit_id) or None

    query_filter = build_filter(
        tenant_id=tenant_id, org_unit_id=org_unit_id, document_ids=document_ids,
        doc_name=doc_filter, role=role_filter, is_ground_truth=is_ground_truth,
    )
    results = multi_query_search(queries=queries, query_filter=query_filter, mode="hybrid", top_k=top_k,
                                 degraded=degraded)
    if not results:
        logger.info("No chunks found for multi-query: %s", queries[0][:50] if queries else "")
    return results
