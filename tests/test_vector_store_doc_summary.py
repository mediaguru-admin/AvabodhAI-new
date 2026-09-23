"""
tests/test_vector_store_doc_summary.py
-----------------------------------------
pipeline/vector_store.py's document-summary collection (backs
pipeline/retriever.py::shortlist_documents()) and the with_vectors=True
support added to search()/batch_search() for MMR diversity re-ranking.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from pipeline import vector_store

_FILTER = models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value="t1"))])


def _patch_client(fake_client):
    return patch.object(vector_store, "_client", return_value=fake_client)


class TestDocumentSummaryPointId:
    def test_deterministic_same_input_same_id(self):
        a = vector_store.document_summary_point_id("doc-1")
        b = vector_store.document_summary_point_id("doc-1")
        assert a == b

    def test_different_documents_different_ids(self):
        a = vector_store.document_summary_point_id("doc-1")
        b = vector_store.document_summary_point_id("doc-2")
        assert a != b

    def test_distinct_from_chunk_point_id_namespace(self):
        """Must not collide with a chunk point's own ID for the same
        document_id — they live in different collections, but sharing the
        exact same UUID would still be a latent footgun."""
        summary_id = vector_store.document_summary_point_id("doc-1")
        chunk_id = vector_store.chunk_point_id("doc-1", "text", 0)
        assert summary_id != chunk_id


class TestUpsertDocumentSummary:
    def test_blank_summary_is_a_no_op(self):
        fake_client = MagicMock()
        with _patch_client(fake_client):
            vector_store.upsert_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="   ", dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5],
            )
        fake_client.upsert.assert_not_called()

    def test_success_upserts_once(self):
        fake_client = MagicMock()
        with _patch_client(fake_client):
            vector_store.upsert_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="a real summary", dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5],
            )
        fake_client.upsert.assert_called_once()
        _, kwargs = fake_client.upsert.call_args
        assert kwargs["collection_name"] == vector_store.settings.QDRANT_DOC_SUMMARY_COLLECTION

    def test_retries_then_succeeds(self):
        fake_client = MagicMock()
        fake_client.upsert.side_effect = [ConnectionError("down"), None]
        with _patch_client(fake_client), \
             patch.object(vector_store.settings, "QDRANT_UPSERT_MAX_RETRIES", 2), \
             patch.object(vector_store.time, "sleep", lambda _s: None):
            vector_store.upsert_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="a real summary", dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5],
            )
        assert fake_client.upsert.call_count == 2

    def test_exhausted_retries_logs_and_does_not_raise(self):
        """Non-fatal by design — chunk indexing already succeeded by the
        time this runs; losing the shortlist entry for one document must
        never undo that."""
        fake_client = MagicMock()
        fake_client.upsert.side_effect = ConnectionError("down")
        with _patch_client(fake_client), \
             patch.object(vector_store.settings, "QDRANT_UPSERT_MAX_RETRIES", 1), \
             patch.object(vector_store.time, "sleep", lambda _s: None):
            vector_store.upsert_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="a real summary", dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5],
            )  # must not raise
        assert fake_client.upsert.call_count == 2  # 1 initial + 1 retry


class TestDeleteDocumentSummary:
    def test_calls_delete_with_correct_filter(self):
        fake_client = MagicMock()
        with _patch_client(fake_client):
            vector_store.delete_document_summary(tenant_id="t1", document_id="doc-1")
        fake_client.delete.assert_called_once()
        _, kwargs = fake_client.delete.call_args
        assert kwargs["collection_name"] == vector_store.settings.QDRANT_DOC_SUMMARY_COLLECTION


class TestSearchDocumentSummaries:
    def test_returns_mapped_dicts(self):
        fake_point = MagicMock(id="p1", score=0.8, payload={"document_id": "doc-1", "doc_name": "a.pdf"})
        fake_result = MagicMock(points=[fake_point])
        fake_client = MagicMock()
        fake_client.query_points.return_value = fake_result
        with _patch_client(fake_client):
            hits = vector_store.search_document_summaries(
                query_filter=_FILTER, dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5], limit=5,
            )
        assert hits == [{"id": "p1", "score": 0.8, "document_id": "doc-1", "doc_name": "a.pdf"}]
        assert fake_client.query_points.call_args.kwargs["collection_name"] == vector_store.settings.QDRANT_DOC_SUMMARY_COLLECTION


class TestSearchWithVectors:
    def test_default_does_not_return_dense_vector(self):
        fake_point = MagicMock(id="p1", score=0.8, payload={"chunk_text": "hi"}, vector=None)
        fake_result = MagicMock(points=[fake_point])
        fake_client = MagicMock()
        fake_client.query_points.return_value = fake_result
        with _patch_client(fake_client):
            hits = vector_store.search(
                query_filter=_FILTER, dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5],
            )
        assert "_dense_vector" not in hits[0]

    def test_with_vectors_true_returns_dense_vector(self):
        fake_point = MagicMock(
            id="p1", score=0.8, payload={"chunk_text": "hi"},
            vector={"dense": [0.1, 0.2], "splade": MagicMock()},
        )
        fake_result = MagicMock(points=[fake_point])
        fake_client = MagicMock()
        fake_client.query_points.return_value = fake_result
        with _patch_client(fake_client):
            hits = vector_store.search(
                query_filter=_FILTER, dense_vector=[0.1], sparse_indices=[1], sparse_values=[0.5],
                with_vectors=True,
            )
        assert hits[0]["_dense_vector"] == [0.1, 0.2]
        assert fake_client.query_points.call_args.kwargs["with_vectors"] is True
