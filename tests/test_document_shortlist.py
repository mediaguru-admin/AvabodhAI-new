"""
tests/test_document_shortlist.py
-----------------------------------
The scale-ready retrieval fix: pipeline/retriever.py::shortlist_documents()
(first-stage, document-level retrieval) and _mmr_select() (diversity
re-ranking of the final chunk selection) — see the design discussion this
implements: a fixed chunk-level candidate pool doesn't scale with the
number of documents in a knowledge base, so queries narrow to a handful of
relevant DOCUMENTS first, before chunk-level search runs inside just those.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline import retriever


@pytest.fixture(autouse=True)
def _clear_doc_count_cache():
    """
    retriever._DOC_COUNT_CACHE (2026-09-23) is module-level and persists
    across tests in the same process — without this, whichever test runs
    first would poison every later test's document-count result.
    """
    retriever._DOC_COUNT_CACHE.clear()
    yield
    retriever._DOC_COUNT_CACHE.clear()


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert retriever._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_does_not_crash(self):
        assert retriever._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# _mmr_select
# ---------------------------------------------------------------------------

class TestMmrSelect:
    def test_empty_candidates_returns_empty(self):
        assert retriever._mmr_select([], top_k=5, lambda_mult=0.5) == []

    def test_falls_back_to_plain_top_k_when_vectors_missing(self):
        """MMR needs real embeddings for diversity — must never be a hard
        requirement for search to work at all."""
        candidates = [
            {"id": "a", "rerank_score": 0.9},
            {"id": "b", "rerank_score": 0.5},
            {"id": "c", "rerank_score": 0.1},
        ]
        result = retriever._mmr_select(candidates, top_k=2, lambda_mult=0.5)
        assert [c["id"] for c in result] == ["a", "b"]

    def test_respects_top_k(self):
        candidates = [
            {"id": str(i), "rerank_score": 1.0 - i * 0.1, "_dense_vector": [1.0, 0.0]}
            for i in range(5)
        ]
        result = retriever._mmr_select(candidates, top_k=3, lambda_mult=0.5)
        assert len(result) == 3

    def test_pure_diversity_avoids_near_duplicate_of_top_pick(self):
        """lambda=0 (pure diversity): after picking the top-relevance
        candidate, the next pick must be the genuinely different one, not
        its near-duplicate, even though the near-duplicate has the second
        -highest relevance score."""
        candidates = [
            {"id": "top", "rerank_score": 1.0, "_dense_vector": [1.0, 0.0]},
            {"id": "near_duplicate_of_top", "rerank_score": 0.95, "_dense_vector": [0.99, 0.01]},
            {"id": "genuinely_different", "rerank_score": 0.5, "_dense_vector": [0.0, 1.0]},
        ]
        result = retriever._mmr_select(candidates, top_k=2, lambda_mult=0.0)
        ids = [c["id"] for c in result]
        assert ids[0] == "top"
        assert ids[1] == "genuinely_different"

    def test_pure_relevance_matches_plain_top_k(self):
        """lambda=1.0 should behave identically to plain top-K by score,
        even with vectors present — diversity term should have zero weight."""
        candidates = [
            {"id": "a", "rerank_score": 0.9, "_dense_vector": [1.0, 0.0]},
            {"id": "b", "rerank_score": 0.8, "_dense_vector": [1.0, 0.0]},  # identical vector to "a"
            {"id": "c", "rerank_score": 0.1, "_dense_vector": [0.0, 1.0]},
        ]
        result = retriever._mmr_select(candidates, top_k=2, lambda_mult=1.0)
        assert [c["id"] for c in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# shortlist_documents
# ---------------------------------------------------------------------------

class TestShortlistDocuments:
    """
    2026-09-23: shortlist_documents() used a flat DOC_SHORTLIST_LIMIT with
    no regard for how many documents actually exist — a knowledge base
    with exactly (or fewer than) DOC_SHORTLIST_LIMIT documents got
    silently narrowed anyway, meaning real, relevant documents were
    excluded from every unscoped query for no reason at all (confirmed
    live: a 10-document tenant with DOC_SHORTLIST_LIMIT=5 had HALF its
    knowledge base excluded from every broad question, producing
    confidently wrong answers with no error surfaced). Every test below
    that exercises the "shortlisting actually happens" path now explicitly
    mocks count_document_summaries() to return more than the limit — the
    dedicated TestDynamicLimit class below covers the fix itself.
    """

    def test_embedding_failure_returns_empty_never_raises(self):
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=50), \
             patch.object(retriever.embedder, "embed_dense_query", side_effect=ConnectionError("down")):
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1")
        assert result == []

    def test_search_failure_returns_empty_never_raises(self):
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=50), \
             patch.object(retriever.embedder, "embed_dense_query", return_value=[0.1, 0.2]), \
             patch.object(retriever.embedder, "embed_sparse_query", return_value=([1], [0.5])), \
             patch.object(retriever.vector_store, "search_document_summaries", side_effect=RuntimeError("qdrant down")):
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1")
        assert result == []

    def test_success_returns_document_ids_in_order(self):
        hits = [
            {"id": "p1", "document_id": "doc-a", "doc_name": "A", "score": 0.9},
            {"id": "p2", "document_id": "doc-b", "doc_name": "B", "score": 0.7},
        ]
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=50), \
             patch.object(retriever.embedder, "embed_dense_query", return_value=[0.1, 0.2]), \
             patch.object(retriever.embedder, "embed_sparse_query", return_value=([1], [0.5])), \
             patch.object(retriever.vector_store, "search_document_summaries", return_value=hits) as search_mock:
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1")

        assert result == ["doc-a", "doc-b"]
        assert search_mock.call_args.kwargs["limit"] == retriever.settings.DOC_SHORTLIST_LIMIT

    def test_custom_limit_is_honored(self):
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=50), \
             patch.object(retriever.embedder, "embed_dense_query", return_value=[0.1]), \
             patch.object(retriever.embedder, "embed_sparse_query", return_value=([1], [0.5])), \
             patch.object(retriever.vector_store, "search_document_summaries", return_value=[]) as search_mock:
            retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=3)

        assert search_mock.call_args.kwargs["limit"] == 3


class TestDynamicLimit:
    """The actual bug fix: never narrow below the true document count."""

    def test_corpus_at_exactly_the_limit_is_not_narrowed(self):
        """The exact regression: 5 documents, DOC_SHORTLIST_LIMIT=5 —
        must not shortlist (search_document_summaries must never even be
        called), since narrowing "5 out of 5" excludes nothing usefully
        but still risks dropping one on a close embedding-score call."""
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=5), \
             patch.object(retriever.embedder, "embed_dense_query") as embed_mock, \
             patch.object(retriever.vector_store, "search_document_summaries") as search_mock:
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=5)

        assert result == []
        embed_mock.assert_not_called()
        search_mock.assert_not_called()

    def test_corpus_under_the_limit_is_not_narrowed(self):
        """The user's exact reported scenario: 10 documents were getting
        narrowed to 5 even though 10 <= a reasonable limit should mean no
        narrowing at all in the intended design — this specifically
        locks in "fewer documents than the limit -> never narrow"."""
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=3), \
             patch.object(retriever.vector_store, "search_document_summaries") as search_mock:
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=5)

        assert result == []
        search_mock.assert_not_called()

    def test_corpus_over_the_limit_is_narrowed(self):
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=11), \
             patch.object(retriever.embedder, "embed_dense_query", return_value=[0.1]), \
             patch.object(retriever.embedder, "embed_sparse_query", return_value=([1], [0.5])), \
             patch.object(retriever.vector_store, "search_document_summaries",
                           return_value=[{"id": "p1", "document_id": "doc-a"}]) as search_mock:
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=5)

        assert result == ["doc-a"]
        search_mock.assert_called_once()

    def test_count_failure_falls_back_to_full_corpus_search(self):
        with patch.object(retriever.vector_store, "count_document_summaries", side_effect=RuntimeError("down")), \
             patch.object(retriever.vector_store, "search_document_summaries") as search_mock:
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1")

        assert result == []
        search_mock.assert_not_called()


class TestDocumentCountCache:
    """
    2026-09-23: count_document_summaries() measured at 2.35s per call on a
    local Qdrant instance — pure per-call overhead for a value that only
    changes on document upload/delete, being re-fetched on every single
    chat query. _cached_document_count() wraps it with a short TTL cache.
    """

    def test_second_call_within_ttl_uses_cache(self):
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=3) as count_mock:
            retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=5)
            retriever.shortlist_documents(query="q2", tenant_id="t1", org_unit_id="o1", limit=5)

        count_mock.assert_called_once()

    def test_expired_ttl_refetches(self):
        # _cached_document_count() calls time.monotonic() exactly once per
        # shortlist_documents() invocation (to get `now`) — one value per call.
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=3) as count_mock, \
             patch.object(retriever.time, "monotonic", side_effect=[0.0, 999.0]):
            retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=5)
            retriever.shortlist_documents(query="q2", tenant_id="t1", org_unit_id="o1", limit=5)

        assert count_mock.call_count == 2

    def test_different_tenants_are_cached_independently(self):
        with patch.object(retriever.vector_store, "count_document_summaries", return_value=3) as count_mock:
            retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1", limit=5)
            retriever.shortlist_documents(query="q", tenant_id="t2", org_unit_id="o1", limit=5)

        assert count_mock.call_count == 2

    def test_failure_is_not_cached_and_propagates_to_caller(self):
        """A transient Qdrant failure must not get "cached" as a fake
        result, and shortlist_documents()'s own except block (not this
        cache) is what turns it into a safe fallback."""
        with patch.object(retriever.vector_store, "count_document_summaries",
                          side_effect=RuntimeError("down")):
            result = retriever.shortlist_documents(query="q", tenant_id="t1", org_unit_id="o1")
        assert result == []
        assert retriever._DOC_COUNT_CACHE == {}


# ---------------------------------------------------------------------------
# retrieve() / retrieve_multi() — shortlist wiring
# ---------------------------------------------------------------------------

class TestRetrieveShortlistWiring:
    def test_scoped_query_skips_shortlisting(self):
        """doc_filter already scopes to one document — nothing to shortlist."""
        with patch.object(retriever, "shortlist_documents") as shortlist_mock, \
             patch.object(retriever, "search", return_value=[]) as search_mock:
            retriever.retrieve(query="q", tenant_id="t1", org_unit_id="o1", doc_filter="report.pdf")

        shortlist_mock.assert_not_called()
        search_mock.assert_called_once()

    def test_unscoped_query_shortlists_first(self):
        with patch.object(retriever, "shortlist_documents", return_value=["doc-a", "doc-b"]) as shortlist_mock, \
             patch.object(retriever, "build_filter", return_value="FILTER") as build_filter_mock, \
             patch.object(retriever, "search", return_value=[]):
            retriever.retrieve(query="q", tenant_id="t1", org_unit_id="o1")

        shortlist_mock.assert_called_once()
        assert build_filter_mock.call_args.kwargs["document_ids"] == ["doc-a", "doc-b"]

    def test_empty_shortlist_falls_back_to_unscoped_search(self):
        """shortlist_documents() returning [] (no summary points yet, or a
        genuine Qdrant hiccup) must fall back to searching everything, not
        become a hard filter that matches nothing."""
        with patch.object(retriever, "shortlist_documents", return_value=[]), \
             patch.object(retriever, "build_filter", return_value="FILTER") as build_filter_mock, \
             patch.object(retriever, "search", return_value=[]):
            retriever.retrieve(query="q", tenant_id="t1", org_unit_id="o1")

        assert build_filter_mock.call_args.kwargs["document_ids"] is None

    def test_multi_shortlists_against_primary_query(self):
        with patch.object(retriever, "shortlist_documents", return_value=["doc-a"]) as shortlist_mock, \
             patch.object(retriever, "build_filter", return_value="FILTER"), \
             patch.object(retriever, "multi_query_search", return_value=[]):
            retriever.retrieve_multi(queries=["condensed query", "variant"], tenant_id="t1", org_unit_id="o1")

        assert shortlist_mock.call_args.kwargs["query"] == "condensed query"
