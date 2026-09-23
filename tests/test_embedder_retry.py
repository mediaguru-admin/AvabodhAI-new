"""
tests/test_embedder_retry.py
------------------------------
pipeline/embedder.py::embed_dense_batch retry/stop behavior.

Design decision this covers: on a batch failure, retry up to
settings.EMBEDDING_MAX_RETRIES times, then STOP the whole document rather
than skipping the failed batch and continuing to the next one. Skipping
would leave that batch's chunks silently absent from the vector store —
worse than an explicit, retryable failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline import embedder


@pytest.fixture(autouse=True)
def _clear_dense_client_cache():
    embedder._dense_client.cache_clear()
    yield
    embedder._dense_client.cache_clear()


def _patch_client(fake_client):
    return patch.object(embedder, "_dense_client", return_value=fake_client)


class TestSucceedsWithoutRetry:
    def test_single_batch_no_retries_needed(self):
        fake_client = MagicMock()
        fake_client.embed_documents.return_value = [[0.1, 0.2]]
        with _patch_client(fake_client):
            result = embedder.embed_dense_batch(["chunk one"], batch_size=10)
        assert result == [[0.1, 0.2]]
        assert fake_client.embed_documents.call_count == 1


class TestRetriesThenSucceeds:
    def test_transient_failure_recovers_within_retry_budget(self):
        fake_client = MagicMock()
        fake_client.embed_documents.side_effect = [
            TimeoutError("rate limited"),
            [[0.1, 0.2]],
        ]
        with _patch_client(fake_client), \
             patch.object(embedder.settings, "EMBEDDING_MAX_RETRIES", 2), \
             patch.object(embedder.time, "sleep", lambda _s: None):
            result = embedder.embed_dense_batch(["chunk one"], batch_size=10)

        assert result == [[0.1, 0.2]]
        assert fake_client.embed_documents.call_count == 2

    def test_retry_count_is_env_controlled(self):
        """Confirms EMBEDDING_MAX_RETRIES actually bounds the attempt count."""
        fake_client = MagicMock()
        fake_client.embed_documents.side_effect = ConnectionError("down")
        with _patch_client(fake_client), \
             patch.object(embedder.settings, "EMBEDDING_MAX_RETRIES", 4), \
             patch.object(embedder.time, "sleep", lambda _s: None):
            with pytest.raises(RuntimeError):
                embedder.embed_dense_batch(["chunk one"], batch_size=10)

        # 1 initial attempt + 4 retries = 5 total calls
        assert fake_client.embed_documents.call_count == 5


class TestStopsRatherThanSkippingOnExhaustedRetries:
    def test_second_batch_exhausting_retries_stops_before_a_third_batch(self):
        """The actual design decision: a batch that fails after retries
        must stop the whole document, not be skipped in favor of moving on
        to the next batch — a skipped batch is silent data loss."""
        fake_client = MagicMock()
        fake_client.embed_documents.side_effect = [
            [[0.1]],              # batch 1: succeeds
            ConnectionError("x"), # batch 2: fails every attempt
            ConnectionError("x"),
            ConnectionError("x"),
        ]
        with _patch_client(fake_client), \
             patch.object(embedder.settings, "EMBEDDING_MAX_RETRIES", 2), \
             patch.object(embedder.time, "sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="batch 2"):
                embedder.embed_dense_batch(["a", "b", "c"], batch_size=1)

        # batch 1 (1 call) + batch 2 (1 initial + 2 retries = 3 calls) = 4.
        # A 4th chunk ("c", batch 3) must never be attempted.
        assert fake_client.embed_documents.call_count == 4

    def test_raises_not_returns_partial_results_on_exhausted_retries(self):
        fake_client = MagicMock()
        fake_client.embed_documents.side_effect = RuntimeError("persistent failure")
        with _patch_client(fake_client), \
             patch.object(embedder.settings, "EMBEDDING_MAX_RETRIES", 0), \
             patch.object(embedder.time, "sleep", lambda _s: None):
            with pytest.raises(RuntimeError):
                embedder.embed_dense_batch(["chunk"], batch_size=10)
