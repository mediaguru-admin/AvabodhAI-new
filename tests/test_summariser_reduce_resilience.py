"""
tests/test_summariser_reduce_resilience.py
--------------------------------------------
The bug this covers: the Map step (pipeline/summariser.py::summarise_document)
makes one billed LLM call per chunk and produces real data (chunk_summaries).
Previously, if the single Reduce call that combines them failed for any
reason, the whole function raised, the caller (pipeline/ingest.py::
_summarise_and_apply) caught it and stored summary_text="" — discarding
every chunk-level call that had already succeeded and been paid for.

Fix: retry Reduce a few times (most failures are transient), and if it's
still failing, fall back to the joined chunk summaries instead of losing
the Map results. These tests bypass real LangChain/OpenAI wiring — only
the prompt objects (MAP_PROMPT/CHUNK_METADATA_PROMPT/COMBINE_PROMPT) are
faked, so `_build_llm`'s real (but never-invoked) ChatOpenAI construction
is left alone.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from db.models import ChunkMetadataOutput
from pipeline import summariser


class _FakeChain:
    """Stand-in for a fully-built LangChain runnable chain — ignores
    whatever it's further piped into (the llm, the output parser) and
    just calls the test-provided function on .invoke()."""

    def __init__(self, invoke_fn):
        self._invoke_fn = invoke_fn

    def __or__(self, _other):
        return self

    def invoke(self, inputs):
        return self._invoke_fn(inputs)


class _FakePrompt:
    """Stand-in for MAP_PROMPT/CHUNK_METADATA_PROMPT/COMBINE_PROMPT —
    `prompt | llm` normally builds a real chain; here it just captures the
    test's invoke function and ignores the llm entirely, since these tests
    are about summarise_document's retry/fallback control flow, not real
    prompt/LLM/parser behavior."""

    def __init__(self, invoke_fn):
        self._invoke_fn = invoke_fn

    def __or__(self, _llm):
        return _FakeChain(self._invoke_fn)


def _chunk_metadata_prompt(summary_prefix: str = "chunk summary"):
    counter = {"n": 0}

    def _invoke(_inputs):
        counter["n"] += 1
        return ChunkMetadataOutput(summary=f"{summary_prefix} {counter['n']}")

    return _FakePrompt(_invoke)


def _combine_prompt(reduce_side_effects: list):
    """reduce_side_effects: list of either an Exception instance (that
    attempt raises) or a string (that attempt succeeds with this text)."""
    calls = {"n": 0}

    def _invoke(_inputs):
        index = calls["n"]
        calls["n"] += 1
        effect = reduce_side_effects[index]
        if isinstance(effect, Exception):
            raise effect
        return effect

    return _FakePrompt(_invoke), calls


def _make_chunks(n: int) -> list[Document]:
    return [Document(page_content=f"chunk {i} content") for i in range(n)]


class TestReduceSucceedsNormally:
    def test_no_retry_needed_reduce_fallback_not_used(self):
        with patch.object(summariser, "CHUNK_METADATA_PROMPT", _chunk_metadata_prompt()), \
             patch.object(summariser, "COMBINE_PROMPT", _combine_prompt(["final combined summary"])[0]), \
             patch.object(summariser, "DOCUMENT_METADATA_PROMPT", _FakePrompt(lambda _i: (_ for _ in ()).throw(RuntimeError("no doc metadata in this test")))):
            result = summariser.summarise_document(_make_chunks(3), doc_name="doc.pdf")

        assert result["reduce_fallback_used"] is False
        assert result["summary_text"] == "final combined summary"
        assert result["chunk_count"] == 3
        assert len(result["chunk_metadata"]) == 3


class TestReduceRetriesThenSucceeds:
    def test_transient_failure_recovers_without_losing_map_results(self):
        combine_prompt, calls = _combine_prompt([
            TimeoutError("rate limited"),
            "final summary after one retry",
        ])
        with patch.object(summariser, "CHUNK_METADATA_PROMPT", _chunk_metadata_prompt()), \
             patch.object(summariser, "COMBINE_PROMPT", combine_prompt), \
             patch.object(summariser, "DOCUMENT_METADATA_PROMPT", _FakePrompt(lambda _i: (_ for _ in ()).throw(RuntimeError("skip")))), \
             patch.object(summariser.time, "sleep", lambda _s: None):
            result = summariser.summarise_document(_make_chunks(2), doc_name="doc.pdf")

        assert calls["n"] == 2
        assert result["reduce_fallback_used"] is False
        assert result["summary_text"] == "final summary after one retry"


class TestReduceFailsAfterAllRetries:
    def test_falls_back_to_joined_chunk_summaries_never_raises(self):
        """The exact bug this fixes: Reduce failing completely must not
        discard the already-succeeded, already-paid-for Map results."""
        combine_prompt, calls = _combine_prompt([
            ConnectionError("down"),
            ConnectionError("still down"),
            ConnectionError("still down"),
        ])
        with patch.object(summariser, "CHUNK_METADATA_PROMPT", _chunk_metadata_prompt(summary_prefix="real chunk data")), \
             patch.object(summariser, "COMBINE_PROMPT", combine_prompt), \
             patch.object(summariser, "DOCUMENT_METADATA_PROMPT", _FakePrompt(lambda _i: (_ for _ in ()).throw(RuntimeError("skip")))), \
             patch.object(summariser.time, "sleep", lambda _s: None):
            result = summariser.summarise_document(_make_chunks(3), doc_name="doc.pdf")

        assert calls["n"] == 3  # exhausted all retry attempts
        assert result["reduce_fallback_used"] is True
        # The Map step's real output (paid for, already computed) must
        # survive in the fallback summary, not be discarded.
        assert "real chunk data" in result["summary_text"]
        assert result["chunk_metadata"][0].summary.startswith("real chunk data")

    def test_document_metadata_extraction_unaffected_by_reduce_fallback(self):
        """Confirms the pre-existing, correct non-fatal behavior for the
        document-metadata step is untouched by this fix."""
        combine_prompt, _ = _combine_prompt([
            ConnectionError("down"), ConnectionError("down"), ConnectionError("down"),
        ])
        with patch.object(summariser, "CHUNK_METADATA_PROMPT", _chunk_metadata_prompt()), \
             patch.object(summariser, "COMBINE_PROMPT", combine_prompt), \
             patch.object(summariser, "DOCUMENT_METADATA_PROMPT", _FakePrompt(lambda _i: (_ for _ in ()).throw(RuntimeError("doc metadata down too")))), \
             patch.object(summariser.time, "sleep", lambda _s: None):
            result = summariser.summarise_document(_make_chunks(2), doc_name="doc.pdf")

        assert result["reduce_fallback_used"] is True
        assert result["document_metadata"] is None  # non-fatal, as before
