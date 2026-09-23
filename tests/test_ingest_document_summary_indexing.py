"""
tests/test_ingest_document_summary_indexing.py
--------------------------------------------------
pipeline/ingest.py::_index_document_summary() (embeds + upserts a
document's summary for the shortlist stage) and _clear_document_index()
(clears both chunks AND the summary point before re-ingesting, so a
reprocess that fails early can't leave a stale summary vector pointing at
a document whose chunks were just wiped).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pipeline import ingest


class TestIndexDocumentSummary:
    def test_blank_summary_is_a_no_op(self):
        with patch.object(ingest.embedder, "embed_dense_query") as embed_mock, \
             patch.object(ingest.vector_store, "upsert_document_summary") as upsert_mock:
            ingest._index_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf", summary_text="   ",
            )
        embed_mock.assert_not_called()
        upsert_mock.assert_not_called()

    def test_success_embeds_and_upserts(self):
        with patch.object(ingest.embedder, "embed_dense_query", return_value=[0.1, 0.2]), \
             patch.object(ingest.embedder, "embed_sparse_query", return_value=([1], [0.5])), \
             patch.object(ingest.vector_store, "upsert_document_summary") as upsert_mock:
            ingest._index_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="a real summary",
            )
        upsert_mock.assert_called_once_with(
            document_id="doc-1", tenant_id="t1", org_unit_id="o1",
            doc_name="a.pdf", summary_text="a real summary",
            dense_vector=[0.1, 0.2], sparse_indices=[1], sparse_values=[0.5],
        )

    def test_embedding_failure_is_logged_and_does_not_raise(self):
        """Non-fatal: a document whose summary indexing fails just won't
        be shortlisted first (shortlist_documents() falls back to
        full-corpus search) — must never fail an otherwise-successful
        ingestion over this."""
        with patch.object(ingest.embedder, "embed_dense_query", side_effect=ConnectionError("down")), \
             patch.object(ingest.vector_store, "upsert_document_summary") as upsert_mock:
            ingest._index_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="a real summary",
            )  # must not raise
        upsert_mock.assert_not_called()

    def test_upsert_failure_is_logged_and_does_not_raise(self):
        with patch.object(ingest.embedder, "embed_dense_query", return_value=[0.1]), \
             patch.object(ingest.embedder, "embed_sparse_query", return_value=([1], [0.5])), \
             patch.object(ingest.vector_store, "upsert_document_summary", side_effect=RuntimeError("qdrant down")):
            ingest._index_document_summary(
                document_id="doc-1", tenant_id="t1", org_unit_id="o1", doc_name="a.pdf",
                summary_text="a real summary",
            )  # must not raise


class TestClearDocumentIndex:
    def test_clears_both_chunks_and_summary(self):
        with patch.object(ingest.vector_store, "delete_document_points") as delete_chunks_mock, \
             patch.object(ingest.vector_store, "delete_document_summary") as delete_summary_mock:
            ingest._clear_document_index(document_id="doc-1", tenant_id="t1")

        delete_chunks_mock.assert_called_once_with(tenant_id="t1", document_id="doc-1")
        delete_summary_mock.assert_called_once_with(tenant_id="t1", document_id="doc-1")
