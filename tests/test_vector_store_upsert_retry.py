"""
tests/test_vector_store_upsert_retry.py
------------------------------------------
pipeline/vector_store.py::upsert_chunks retry behavior. A transient Qdrant
blip on one batch must not immediately fail a document whose chunking and
embedding (real, paid-for OpenAI calls) already succeeded — retry up to
settings.QDRANT_UPSERT_MAX_RETRIES times before giving up. Retries are safe
here specifically because point IDs are deterministic (chunk_point_id()),
so a retried upsert overwrites the same points rather than duplicating.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline import vector_store


def _make_point(index: int) -> vector_store.ChunkPoint:
    return vector_store.ChunkPoint(
        document_id="doc-1",
        tenant_id="tenant-1",
        org_unit_id="org-1",
        role="text",
        chunk_index=index,
        chunk_text=f"chunk {index}",
        dense_vector=[0.1, 0.2],
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.5],
    )


def _patch_client(fake_client):
    return patch.object(vector_store, "_client", return_value=fake_client)


class TestSucceedsWithoutRetry:
    def test_single_batch_no_retries_needed(self):
        fake_client = MagicMock()
        with _patch_client(fake_client):
            total = vector_store.upsert_chunks([_make_point(0)], batch_size=10)
        assert total == 1
        assert fake_client.upsert.call_count == 1


class TestRetriesThenSucceeds:
    def test_transient_failure_recovers_within_retry_budget(self):
        fake_client = MagicMock()
        fake_client.upsert.side_effect = [ConnectionError("down"), None]
        with _patch_client(fake_client), \
             patch.object(vector_store.settings, "QDRANT_UPSERT_MAX_RETRIES", 3), \
             patch.object(vector_store.time, "sleep", lambda _s: None):
            total = vector_store.upsert_chunks([_make_point(0)], batch_size=10)

        assert total == 1
        assert fake_client.upsert.call_count == 2

    def test_default_retry_budget_is_five(self):
        """The actual number you asked for: at least 5 retries before
        declaring the upload failed."""
        fake_client = MagicMock()
        fake_client.upsert.side_effect = ConnectionError("down")
        with _patch_client(fake_client), \
             patch.object(vector_store.time, "sleep", lambda _s: None):
            with pytest.raises(RuntimeError):
                vector_store.upsert_chunks([_make_point(0)], batch_size=10)

        # 1 initial attempt + 5 retries (the real, unpatched default) = 6.
        assert vector_store.settings.QDRANT_UPSERT_MAX_RETRIES == 5
        assert fake_client.upsert.call_count == 6


class TestExhaustedRetriesFailsTheWholeUpload:
    def test_second_batch_failing_stops_before_a_third_batch(self):
        fake_client = MagicMock()
        fake_client.upsert.side_effect = [
            None,                  # batch 1: succeeds
            ConnectionError("x"),  # batch 2: fails every attempt
            ConnectionError("x"),
        ]
        with _patch_client(fake_client), \
             patch.object(vector_store.settings, "QDRANT_UPSERT_MAX_RETRIES", 1), \
             patch.object(vector_store.time, "sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="batch 2"):
                vector_store.upsert_chunks(
                    [_make_point(0), _make_point(1), _make_point(2)], batch_size=1,
                )

        # batch 1 (1 call) + batch 2 (1 initial + 1 retry = 2 calls) = 3.
        # A 3rd batch must never be attempted.
        assert fake_client.upsert.call_count == 3

    def test_empty_points_list_is_a_no_op(self):
        fake_client = MagicMock()
        with _patch_client(fake_client):
            total = vector_store.upsert_chunks([])
        assert total == 0
        fake_client.upsert.assert_not_called()
