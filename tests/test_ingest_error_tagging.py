"""
tests/test_ingest_error_tagging.py
-----------------------------------
pipeline/ingest.py's _run_stage helper, and that process_document /
process_web_document actually use it on every fatal stage.

Before this, a FAILED document's status_detail was just str(e) for
whatever the underlying library raised — for a raw socket-level timeout
that's the bare string "timed out", with no indication of which stage was
running or what kind of error it even was. _run_stage tags every fatal
stage with its name and the real exception type, so status_detail (what
the webhook/check-status path pushes to the UI) is actually diagnosable
without cross-referencing live server logs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline import ingest


class TestRunStage:
    def test_passes_through_return_value_on_success(self):
        result = ingest._run_stage("Some stage", lambda x: x * 2, 21)
        assert result == 42

    def test_wraps_exception_with_stage_name_and_type(self):
        def _boom():
            raise TimeoutError("timed out")

        with pytest.raises(RuntimeError) as exc_info:
            ingest._run_stage("Text extraction", _boom)

        assert str(exc_info.value) == "Text extraction failed: TimeoutError: timed out"

    def test_preserves_original_exception_as_cause(self):
        original = ValueError("bad input")

        def _boom():
            raise original

        with pytest.raises(RuntimeError) as exc_info:
            ingest._run_stage("Chunking", _boom)

        assert exc_info.value.__cause__ is original

    def test_forwards_args_and_kwargs(self):
        calls = []

        def _record(a, b, keyword=None):
            calls.append((a, b, keyword))
            return "ok"

        result = ingest._run_stage("Stage", _record, 1, 2, keyword="three")
        assert result == "ok"
        assert calls == [(1, 2, "three")]


class TestProcessDocumentFailureReporting:
    def test_extraction_timeout_produces_stage_tagged_status_detail(self):
        """The exact bug report this fixes: a bare socket timeout during
        extraction used to surface to the UI as just 'timed out' — now it
        must say which stage failed and the real exception type."""
        with patch.object(ingest.storage, "set_status", MagicMock()) as set_status_mock, \
             patch.object(ingest.webhook, "notify_status_change", MagicMock()) as webhook_mock, \
             patch.object(ingest.vector_store, "delete_document_points", MagicMock()), \
             patch.object(ingest.vector_store, "delete_document_summary", MagicMock()), \
             patch.object(ingest.extractor, "extract_file", MagicMock(side_effect=TimeoutError("timed out"))):
            ingest.process_document(
                document_id="doc-1", tenant_id="tenant-1", org_unit_id="org-1",
                file_path="report.pdf", doc_name="report.pdf", doc_hash="hash123",
            )

        failed_calls = [c for c in set_status_mock.call_args_list if c.args[3] == "FAILED"]
        assert len(failed_calls) == 1
        status_detail = failed_calls[0].kwargs["status_detail"]
        assert status_detail == "Text extraction failed: TimeoutError: timed out"

        failed_webhook_calls = [c for c in webhook_mock.call_args_list if c.args[3] == "FAILED"]
        assert len(failed_webhook_calls) == 1
        assert failed_webhook_calls[0].kwargs["status_detail"] == status_detail

    def test_chunking_failure_is_tagged_separately_from_extraction(self):
        with patch.object(ingest.storage, "set_status", MagicMock()) as set_status_mock, \
             patch.object(ingest.webhook, "notify_status_change", MagicMock()), \
             patch.object(ingest.vector_store, "delete_document_points", MagicMock()), \
             patch.object(ingest.vector_store, "delete_document_summary", MagicMock()), \
             patch.object(ingest.extractor, "extract_file", MagicMock(return_value=MagicMock())), \
             patch.object(ingest.chunker, "chunk_document", MagicMock(side_effect=ValueError("bad structure"))):
            ingest.process_document(
                document_id="doc-2", tenant_id="tenant-1", org_unit_id="org-1",
                file_path="report.pdf", doc_name="report.pdf", doc_hash="hash456",
            )

        failed_calls = [c for c in set_status_mock.call_args_list if c.args[3] == "FAILED"]
        assert len(failed_calls) == 1
        assert failed_calls[0].kwargs["status_detail"] == "Chunking failed: ValueError: bad structure"

    def test_indexing_failure_after_successful_extraction_and_chunking(self):
        fake_chunk = MagicMock()
        fake_extracted = MagicMock(image_elements=[], table_elements=[])

        with patch.object(ingest.storage, "set_status", MagicMock()) as set_status_mock, \
             patch.object(ingest.webhook, "notify_status_change", MagicMock()), \
             patch.object(ingest.vector_store, "delete_document_points", MagicMock()), \
             patch.object(ingest.vector_store, "delete_document_summary", MagicMock()), \
             patch.object(ingest.extractor, "extract_file", MagicMock(return_value=fake_extracted)), \
             patch.object(ingest.chunker, "chunk_document", MagicMock(return_value=[fake_chunk])), \
             patch.object(ingest, "_index_text_chunks", MagicMock(side_effect=ConnectionError("refused"))):
            ingest.process_document(
                document_id="doc-3", tenant_id="tenant-1", org_unit_id="org-1",
                file_path="report.pdf", doc_name="report.pdf", doc_hash="hash789",
            )

        failed_calls = [c for c in set_status_mock.call_args_list if c.args[3] == "FAILED"]
        assert len(failed_calls) == 1
        assert failed_calls[0].kwargs["status_detail"] == "Chunk embedding/indexing failed: ConnectionError: refused"
