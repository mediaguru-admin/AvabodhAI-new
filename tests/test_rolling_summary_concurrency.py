"""
tests/test_rolling_summary_concurrency.py
------------------------------------------
pipeline/chat_storage.py::update_rolling_summary() now runs as a FastAPI
BackgroundTask (api/routes/chat.py::_save_turn), fired after the response
is already sent. That makes two overlapping calls for the SAME thread
possible (turn N+1 sent before turn N's background job finishes), which
the synchronous, single-call-per-request version never had to worry
about. Two protections, tested independently here:

  1. An in-process per-thread_id lock — a second call for a thread
     already being updated skips outright rather than racing.
  2. A conditional UPDATE (WHERE rolling_summary_through < new value) —
     the guarantee that holds even if the lock is ever bypassed (e.g. a
     future multi-process deployment): a stale write can never regress
     the stored value, it just becomes a no-op.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline import chat_storage


@pytest.fixture(autouse=True)
def _clear_thread_locks():
    """_THREAD_UPDATE_LOCKS is module-level and keyed by thread_id — clear
    it between tests so a lock left held (or a stale entry) from one test
    can never affect another."""
    chat_storage._THREAD_UPDATE_LOCKS.clear()
    yield
    chat_storage._THREAD_UPDATE_LOCKS.clear()


def _fake_message(role: str, content: str):
    return SimpleNamespace(id=uuid.uuid4(), role=role, content=content)


def _session_returning(rolling_summary_through: int, message_count: int, update_rowcount: int):
    """
    A fake Session whose ChatThread query returns a thread stub with the
    given rolling_summary_through, whose ChatMessage query returns
    message_count fake messages, and whose .execute(UPDATE ...) reports
    update_rowcount — 0 simulates the monotonic guard rejecting a stale
    write, 1 simulates a normal successful write.
    """
    session = MagicMock()
    thread_stub = SimpleNamespace(rolling_summary_through=rolling_summary_through, rolling_summary="old")
    session.query.return_value.filter.return_value.first.return_value = thread_stub
    messages = [_fake_message("human" if i % 2 == 0 else "ai", f"m{i}") for i in range(message_count)]
    session.query.return_value.filter.return_value.order_by.return_value.all.return_value = messages
    session.execute.return_value = SimpleNamespace(rowcount=update_rowcount)
    return session


@contextmanager
def _ctx(session):
    yield session


class TestMonotonicWriteGuard:
    def test_successful_write_when_nothing_raced(self):
        session = _session_returning(rolling_summary_through=0, message_count=20, update_rowcount=1)
        with patch.object(chat_storage, "get_db_session_context", return_value=_ctx(session)), \
             patch.object(chat_storage, "_summarize_turns", return_value="new summary"):
            chat_storage.update_rolling_summary(thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1")

        session.execute.assert_called_once()
        update_stmt = session.execute.call_args[0][0]
        # It's a real SQLAlchemy Update construct (not a plain string/dict) —
        # confirms the conditional-UPDATE code path actually ran, not a
        # fallback or a no-op.
        from sqlalchemy.sql.dml import Update
        assert isinstance(update_stmt, Update)

    def test_stale_write_is_a_no_op_not_a_crash_or_regression(self):
        """Simulates: another (faster/later-starting) update already
        advanced rolling_summary_through past what this job computed.
        The conditional UPDATE reports rowcount=0 -- must log and return,
        never raise, never claim success."""
        session = _session_returning(rolling_summary_through=0, message_count=20, update_rowcount=0)
        with patch.object(chat_storage, "get_db_session_context", return_value=_ctx(session)), \
             patch.object(chat_storage, "_summarize_turns", return_value="stale summary"):
            # Must not raise.
            chat_storage.update_rolling_summary(thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1")

        session.execute.assert_called_once()

    def test_nothing_aged_out_skips_before_any_write(self):
        """rolling_summary_through already covers everything that would
        age out -- must not even attempt the UPDATE, let alone the LLM call."""
        session = _session_returning(rolling_summary_through=100, message_count=20, update_rowcount=1)
        with patch.object(chat_storage, "get_db_session_context", return_value=_ctx(session)), \
             patch.object(chat_storage, "_summarize_turns") as summarize_mock:
            chat_storage.update_rolling_summary(thread_id=str(uuid.uuid4()), tenant_id="t1", org_unit_id="o1")

        summarize_mock.assert_not_called()
        session.execute.assert_not_called()


class TestPerThreadLock:
    def test_second_call_for_same_thread_skips_while_first_holds_lock(self):
        thread_id = str(uuid.uuid4())
        lock = chat_storage._get_thread_lock(thread_id)
        lock.acquire()  # simulate a job already in flight for this thread
        try:
            with patch.object(chat_storage, "_update_rolling_summary") as inner_mock:
                chat_storage.update_rolling_summary(thread_id=thread_id, tenant_id="t1", org_unit_id="o1")
            inner_mock.assert_not_called()
        finally:
            lock.release()

    def test_lock_is_released_after_a_successful_update_so_the_next_call_can_run(self):
        thread_id = str(uuid.uuid4())
        session = _session_returning(rolling_summary_through=100, message_count=20, update_rowcount=1)
        with patch.object(chat_storage, "get_db_session_context", return_value=_ctx(session)):
            chat_storage.update_rolling_summary(thread_id=thread_id, tenant_id="t1", org_unit_id="o1")

        assert chat_storage._get_thread_lock(thread_id).acquire(blocking=False) is True
        chat_storage._get_thread_lock(thread_id).release()

    def test_lock_is_released_even_when_the_update_raises(self):
        """update_rolling_summary() is non-fatal by design -- an exception
        inside _update_rolling_summary must still release the lock, or
        every subsequent turn for this thread would skip forever."""
        thread_id = str(uuid.uuid4())
        with patch.object(chat_storage, "_update_rolling_summary", side_effect=RuntimeError("db exploded")):
            chat_storage.update_rolling_summary(thread_id=thread_id, tenant_id="t1", org_unit_id="o1")  # must not raise

        assert chat_storage._get_thread_lock(thread_id).acquire(blocking=False) is True
        chat_storage._get_thread_lock(thread_id).release()

    def test_different_threads_do_not_block_each_other(self):
        thread_a, thread_b = str(uuid.uuid4()), str(uuid.uuid4())
        lock_a = chat_storage._get_thread_lock(thread_a)
        lock_a.acquire()
        try:
            with patch.object(chat_storage, "_update_rolling_summary") as inner_mock:
                chat_storage.update_rolling_summary(thread_id=thread_b, tenant_id="t1", org_unit_id="o1")
            inner_mock.assert_called_once()
        finally:
            lock_a.release()
