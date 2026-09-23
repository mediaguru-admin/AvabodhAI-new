"""
tests/test_memory_window.py
----------------------------
pipeline/memory.py::load_memory_from_db()'s window mechanism, rewritten
2026-09-23: the verbatim window is no longer a fixed "last N turns" — it
is "every message after ChatThread.rolling_summary_through", so it always
tiles exactly with the summary (which covers "up to
rolling_summary_through") with no gap, regardless of how far behind
pipeline/chat_storage.py::update_rolling_summary() has fallen.

These tests mock the SQLAlchemy Session directly (db.query(...) returns
different mocks per model) since load_memory_from_db() takes a raw
Session, not an ORM layer already covered by fixtures elsewhere.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from db.models import ChatThread, ChatMessage
from pipeline import memory as memory_mod


def _fake_message(role: str, content: str, has_image_caption: str = None):
    return SimpleNamespace(
        id=uuid.uuid4(), role=role, content=content,
        image_caption=has_image_caption,
    )


def _mock_db(thread_rolling_summary_through: int, thread_rolling_summary, all_messages: list):
    """
    Builds a fake Session whose .query(ChatThread)... returns a thread
    stub, and whose .query(ChatMessage)... returns whatever slice of
    all_messages the real OFFSET would have returned — i.e. the mock
    itself performs the slicing a real Postgres OFFSET would do, so the
    test asserts on the OFFSET value passed in, not on hand-sliced data.
    """
    thread_query = MagicMock()
    thread_query.filter.return_value.first.return_value = SimpleNamespace(
        rolling_summary_through=thread_rolling_summary_through,
        rolling_summary=thread_rolling_summary,
    )

    message_query = MagicMock()
    captured_offset = {}

    def _offset(n):
        captured_offset["value"] = n
        result = MagicMock()
        result.all.return_value = all_messages[n:]
        return result

    message_query.filter.return_value.order_by.return_value.offset.side_effect = _offset

    db = MagicMock()
    db.query.side_effect = lambda model: thread_query if model is ChatThread else message_query
    return db, captured_offset


class TestSteadyStateFallsBackToConfiguredWindow:
    """When the summarizer is keeping pace, rolling_summary_through sits
    at total - MEMORY_WINDOW_SIZE*2, so the dynamic window collapses to
    exactly the same size as the old fixed-N behavior."""

    def test_offset_equals_rolling_summary_through(self):
        messages = [_fake_message("human" if i % 2 == 0 else "ai", f"msg {i}") for i in range(20)]
        db, captured = _mock_db(thread_rolling_summary_through=10, thread_rolling_summary="old stuff", all_messages=messages)

        memory, rolling_summary, window_ids = memory_mod.load_memory_from_db(
            thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1", db=db,
        )

        assert captured["value"] == 10
        assert rolling_summary == "old stuff"
        # 20 - 10 = 10 messages -> 5 turns injected
        assert len(memory.chat_memory.messages) == 10
        assert len(window_ids) == 10

    def test_no_thread_row_defaults_offset_to_zero(self):
        """A missing/undeletable thread row must not crash — offset 0
        (whole history) is the safe fallback, same as rolling_summary=None."""
        thread_query = MagicMock()
        thread_query.filter.return_value.first.return_value = None
        message_query = MagicMock()
        message_query.filter.return_value.order_by.return_value.offset.return_value.all.return_value = []
        db = MagicMock()
        db.query.side_effect = lambda model: thread_query if model is ChatThread else message_query

        memory, rolling_summary, window_ids = memory_mod.load_memory_from_db(
            thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1", db=db,
        )
        assert rolling_summary is None
        message_query.filter.return_value.order_by.return_value.offset.assert_called_once_with(0)


class TestLaggingSummarizerWindowGrows:
    """The actual fix: when rolling_summary_through hasn't advanced (the
    background job is slow/hasn't run), the window must include EVERY
    message since that point, not just the configured N — nothing may
    fall into the gap between the two."""

    def test_window_exceeds_configured_size_when_summary_is_stale(self):
        # 30 messages, but rolling_summary_through stuck at 4 (way behind
        # where a keeping-pace summarizer would have it for 30 messages).
        messages = [_fake_message("human" if i % 2 == 0 else "ai", f"msg {i}") for i in range(30)]
        db, captured = _mock_db(thread_rolling_summary_through=4, thread_rolling_summary="stale", all_messages=messages)

        memory, rolling_summary, window_ids = memory_mod.load_memory_from_db(
            thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1", db=db,
        )

        assert captured["value"] == 4
        # 30 - 4 = 26 messages injected verbatim -- far more than
        # MEMORY_WINDOW_SIZE*2 (10) would have allowed under the old fixed window.
        assert len(memory.chat_memory.messages) == 26
        assert len(window_ids) == 26

    def test_langchain_k_trimming_does_not_silently_truncate_the_grown_window(self):
        """Regression guard for the exact bug this would reintroduce:
        ConversationBufferWindowMemory's own k must be raised alongside
        the query, or LangChain would quietly re-trim the deliberately
        oversized window back down to MEMORY_WINDOW_SIZE turns, recreating
        the gap this whole mechanism exists to close."""
        messages = [_fake_message("human" if i % 2 == 0 else "ai", f"msg {i}") for i in range(40)]
        db, _ = _mock_db(thread_rolling_summary_through=0, thread_rolling_summary=None, all_messages=messages)

        memory, _, _ = memory_mod.load_memory_from_db(
            thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1", db=db,
        )
        assert len(memory.chat_memory.messages) == 40


class TestMessageFetchFailureDegradesSafely:
    def test_query_exception_returns_empty_memory_not_a_crash(self):
        thread_query = MagicMock()
        thread_query.filter.return_value.first.return_value = SimpleNamespace(
            rolling_summary_through=5, rolling_summary="s",
        )
        message_query = MagicMock()
        message_query.filter.return_value.order_by.return_value.offset.side_effect = RuntimeError("db down")
        db = MagicMock()
        db.query.side_effect = lambda model: thread_query if model is ChatThread else message_query

        memory, rolling_summary, window_ids = memory_mod.load_memory_from_db(
            thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1", db=db,
        )
        assert memory.chat_memory.messages == []
        assert window_ids == set()
        # rolling_summary itself still loaded fine -- independent try block
        assert rolling_summary == "s"
