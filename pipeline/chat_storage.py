"""
pipeline/chat_storage.py
------------------------
Saves chat messages and threads to PostgreSQL.
Both human and AI messages saved as separate rows.
Message embeddings generated and stored for semantic search.

Multi-tenancy + department isolation: every function here takes BOTH
tenant_id and org_unit_id, and either stamps them onto a new row or
filters an existing lookup by both together. Thread ownership is
validated once (create_thread / get_thread), and downstream functions
that operate purely on thread_id (save_human_message, save_ai_message,
update_thread_title, increment_message_count) trust that the caller
already confirmed the thread belongs to this tenant_id+org_unit_id — but
they still stamp both onto every row they write, so a bad thread_id can
never result in a cross-tenant or cross-department row being created
even if the upstream check were ever skipped.
"""

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import update as sa_update
from sqlalchemy.orm import Session
from sqlalchemy.orm import make_transient

from db.models import ChatThread, ChatMessage
from db.database import get_db_session_context
from pipeline import embedder, vector_store
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_ROLLING_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["existing_summary", "new_turns"],
    template="""You maintain a running summary of an ongoing conversation between a
user and a document-assistant chatbot. Update the summary below to also
cover the NEW turns, without losing any fact, decision, name, number, or
open question already captured in the existing summary. Be concise but
specific — this summary is the only record of these turns once they
scroll out of the chatbot's raw context.

EXISTING SUMMARY:
{existing_summary}

NEW TURNS TO FOLD IN:
{new_turns}

UPDATED SUMMARY:""",
)


def _summarize_turns(existing_summary: Optional[str], new_turns_text: str, thread_id: str) -> str:
    """
    Fold new_turns_text into existing_summary via one LLM call. Retries a
    couple of times (transient failures — rate limits, timeouts), then
    falls back to appending the raw turn text rather than losing it —
    same "never discard already-real data" rule as
    pipeline/summariser.py's Reduce-step retry+fallback.
    """
    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.REDUCE_MODEL,
        temperature=0,
        max_tokens=settings.MEMORY_SUMMARY_MAX_TOKENS,
        timeout=settings.CHAT_REQUEST_TIMEOUT,
        max_retries=settings.CHAT_MAX_RETRIES,
    )
    chain = _ROLLING_SUMMARY_PROMPT | llm | StrOutputParser()

    ATTEMPTS = 3
    last_error: Optional[Exception] = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return chain.invoke({
                "existing_summary": existing_summary or "(none yet — this is the first summarised slice)",
                "new_turns": new_turns_text,
            }).strip()
        except Exception as e:
            last_error = e
            logger.warning(
                "Rolling summary attempt %d/%d failed for thread %s: %s",
                attempt, ATTEMPTS, thread_id[:8], e,
            )
            if attempt < ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    logger.error(
        "Rolling summary LLM failed after %d attempts for thread %s (%s) — falling back to "
        "raw turn text instead of losing these turns from memory entirely",
        ATTEMPTS, thread_id[:8], last_error,
    )
    fallback = (existing_summary + "\n\n" if existing_summary else "") + new_turns_text
    return fallback[:4000]


# 2026-09-23: update_rolling_summary() now runs as a FastAPI BackgroundTask
# (api/routes/chat.py::_save_turn), fired after the response is already
# sent. That makes two overlapping calls for the SAME thread possible —
# e.g. the user sends turn N+1 before turn N's background job has
# finished. Without protection, two such jobs would each read the
# CURRENT rolling_summary_through independently, compute independently,
# and whichever one COMMITS LAST would win — even if it started from
# staler data and covers FEWER messages than the one it overwrites,
# silently regressing rolling_summary_through and losing already-folded
# turns. Two layers guard against that:
#
#   1. _THREAD_UPDATE_LOCKS (below) — an in-process lock per thread_id.
#      A second call arriving while one is already running for that
#      thread is skipped outright, not queued: it would just recompute a
#      near-identical range, and the next real turn re-triggers this
#      anyway. Sufficient on its own for THIS deployment specifically
#      (Dockerfile runs uvicorn --workers 1, a single process — no
#      cross-process race is even possible), but it stops being
#      sufficient the moment that ever changes.
#   2. The conditional UPDATE in _update_rolling_summary() below — the
#      guarantee that actually holds regardless of process topology. The
#      write only applies if rolling_summary_through in the ROW is still
#      lower than the value being written, checked atomically at write
#      time by the database itself, not assumed from when the job
#      started. A stale/slow job's write becomes a silent no-op instead
#      of a regression.
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_UPDATE_LOCKS: dict[str, threading.Lock] = {}


def _get_thread_lock(thread_id: str) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_UPDATE_LOCKS.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _THREAD_UPDATE_LOCKS[thread_id] = lock
        return lock


def update_rolling_summary(thread_id: str, tenant_id: str, org_unit_id: str) -> None:
    """
    Runs as a background task after a turn's response is already sent
    (api/routes/chat.py::_save_turn) — the user is never waiting on this.
    Folds any messages that have now aged OUT of the recency window
    (MEMORY_WINDOW_SIZE turns, pipeline/memory.py) into
    ChatThread.rolling_summary, so a long thread's older turns are
    compressed forward instead of vanishing once load_memory_from_db()'s
    window no longer reaches them.

    rolling_summary_through tracks how many messages (chronological) are
    already folded in, so a thread with no new aged-out messages this turn
    (still within its first MEMORY_WINDOW_SIZE turns) is a cheap no-op —
    checked before any LLM call. See the module-level comment above this
    function for the concurrency guarantee (never backdated) that running
    as a background task now requires.

    NON-FATAL, like _index_message_in_qdrant() above: by the time this runs
    the answer has already been generated (and billed) and both messages
    are already committed. A failure here must never turn a successful
    turn into a 500 — the caller is past the point where it could do
    anything useful with the error (this doesn't even run on the request
    path anymore), and the only consequence of skipping an update is that
    the same aged-out messages get folded in on the next turn instead.
    _summarize_turns() already handles LLM failure on its own; this
    guards the DB work around it (connection drop, session error, a
    malformed thread_id).
    """
    lock = _get_thread_lock(thread_id)
    if not lock.acquire(blocking=False):
        logger.info(
            "Rolling summary update for thread %s already in progress — skipping "
            "(the next turn will pick up anything this one doesn't cover)",
            thread_id[:8],
        )
        return
    try:
        _update_rolling_summary(thread_id, tenant_id, org_unit_id)
    except Exception as e:
        logger.warning(
            "Rolling summary update failed for thread %s (non-fatal, will retry next turn): %s",
            thread_id[:8], e,
        )
    finally:
        lock.release()


def _update_rolling_summary(thread_id: str, tenant_id: str, org_unit_id: str) -> None:
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id),
            ChatThread.tenant_id == tenant_id,
            ChatThread.org_unit_id == org_unit_id,
        ).first()
        if not thread:
            return

        messages = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.thread_id == uuid.UUID(thread_id),
                ChatMessage.tenant_id == tenant_id,
                ChatMessage.org_unit_id == org_unit_id,
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

        aged_out_end = len(messages) - (settings.MEMORY_WINDOW_SIZE * 2)
        already_summarized = thread.rolling_summary_through or 0
        if aged_out_end <= already_summarized:
            return  # nothing has aged out of the window since the last update

        new_slice = messages[already_summarized:aged_out_end]
        if not new_slice:
            return

        turn_text = "\n".join(
            f"{'Human' if m.role == 'human' else 'Assistant'}: {m.content}"
            for m in new_slice
        )

        new_summary = _summarize_turns(thread.rolling_summary, turn_text, thread_id)

        # Monotonic write — see the module-level comment above
        # update_rolling_summary() for why. This is a conditional UPDATE,
        # not the ORM attribute-set-then-commit pattern this replaced:
        # that pattern always overwrites whatever's in the row NOW,
        # racing on nothing but which job's commit lands last. The WHERE
        # clause below makes the write a no-op — not a regression — if
        # another update already advanced rolling_summary_through past
        # this job's value by the time it actually writes.
        result = session.execute(
            sa_update(ChatThread)
            .where(
                ChatThread.id == uuid.UUID(thread_id),
                ChatThread.rolling_summary_through < aged_out_end,
            )
            .values(rolling_summary=new_summary, rolling_summary_through=aged_out_end)
        )
        if result.rowcount == 0:
            logger.info(
                "Rolling summary update for thread %s skipped — another update already "
                "advanced rolling_summary_through past %d in the meantime",
                thread_id[:8], aged_out_end,
            )
            return

        logger.info(
            "Rolling summary updated for thread %s — folded in %d message(s), now covers first %d",
            thread_id[:8], len(new_slice), aged_out_end,
        )


def _index_message_in_qdrant(message_id, tenant_id: str, org_unit_id: str, thread_id: str, role: str, content: str) -> None:
    """
    2026-08-21: chat-message vectors moved to Qdrant's avabodh_chat_messages
    collection (ChatMessage.embedding/embedding_model columns removed —
    see db/models.py). Non-fatal — GET /chat/search just won't find this
    message if indexing fails, the message itself is still saved in Postgres.
    """
    try:
        dense_vector = embedder.embed_dense_query(content)
        vector_store.upsert_chat_message(
            message_id=str(message_id), tenant_id=tenant_id, org_unit_id=org_unit_id,
            thread_id=str(thread_id), role=role, content=content,
            dense_vector=dense_vector,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.warning("Chat message Qdrant indexing failed (non-fatal): %s", e)


def create_thread(
    tenant_id: str,
    org_unit_id: str,
    title: Optional[str] = None,
    user_id: Optional[str] = None,
    doc_filter: Optional[str] = None,
) -> ChatThread:
    """Create a new chat thread, stamped with tenant_id + org_unit_id, and return it."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = ChatThread(
            tenant_id = tenant_id,
            org_unit_id = org_unit_id,
            title     = title,
            user_id   = user_id,
            doc_filter= doc_filter,
        )
        session.add(thread)
        session.flush()
        session.expunge(thread)
        make_transient(thread)
        logger.info("Created thread: %s (tenant=%s org_unit=%s)", thread.id, tenant_id, org_unit_id)
        return thread


def update_thread_title(thread_id: str, tenant_id: str, org_unit_id: str, title: str) -> None:
    """Update thread title — called after first message. Scoped to tenant+org_unit."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id),
            ChatThread.tenant_id == tenant_id,
            ChatThread.org_unit_id == org_unit_id,
        ).first()
        if thread:
            thread.title = title
            thread.updated_at = datetime.now(timezone.utc)
            session.add(thread)
            logger.info("Updated thread title: %s -> %s", thread_id[:8], title)


def increment_message_count(thread_id: str, tenant_id: str, org_unit_id: str) -> None:
    """Increment message counter on thread. Scoped to tenant+org_unit."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id),
            ChatThread.tenant_id == tenant_id,
            ChatThread.org_unit_id == org_unit_id,
        ).first()
        if thread:
            thread.message_count = (thread.message_count or 0) + 1
            thread.updated_at = datetime.now(timezone.utc)
            session.add(thread)


def save_human_message(
    thread_id: str,
    tenant_id: str,
    org_unit_id: str,
    content: str,
) -> ChatMessage:
    """
    Save human message as its own row. Indexed into Qdrant for
    GET /chat/search (see _index_message_in_qdrant).
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        msg = ChatMessage(
            tenant_id       = tenant_id,
            org_unit_id     = org_unit_id,
            thread_id       = uuid.UUID(thread_id),
            role            = "human",
            content         = content,
        )
        session.add(msg)
        session.flush()
        message_id = msg.id
        session.expunge(msg)
        make_transient(msg)

    _index_message_in_qdrant(message_id, tenant_id, org_unit_id, thread_id, "human", content)
    logger.info("Saved human message to thread %s", thread_id[:8])
    return msg


def save_ai_message(
    thread_id: str,
    tenant_id: str,
    org_unit_id: str,
    content: str,
    sources: Optional[list] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    has_image: bool = False,
    image_caption: Optional[str] = None,
    degraded: Optional[list] = None,
) -> ChatMessage:
    """
    Save AI message as its own row — separate from human message.
    Stores sources (which chunks were used) and token usage.
    Generates and stores embedding.

    has_image / image_caption: set when this turn was answered using
    a user-attached image (multimodal chat). image_caption is the GPT-4o
    Vision caption of that image, kept for thread history / display.

    degraded: which retrieval components silently fell back while
    answering this turn (pipeline/retriever.py::search()'s degraded
    out-param, e.g. ["sparse_embedding"] or ["reranking"]) — internal
    debugging data only (db/models.py::ChatMessage.degraded), never
    exposed on the API response.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        msg = ChatMessage(
            tenant_id         = tenant_id,
            org_unit_id       = org_unit_id,
            thread_id         = uuid.UUID(thread_id),
            role              = "ai",
            content           = content,
            sources           = sources or [],
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            degraded          = degraded or None,
            has_image         = has_image,
            image_caption     = image_caption,
        )
        session.add(msg)
        session.flush()
        message_id = msg.id
        session.expunge(msg)
        make_transient(msg)

    _index_message_in_qdrant(message_id, tenant_id, org_unit_id, thread_id, "ai", content)
    logger.info("Saved AI message to thread %s | sources=%d | has_image=%s",
               thread_id[:8], len(sources or []), has_image)
    return msg


def get_thread(thread_id: str, tenant_id: str, org_unit_id: str, db: Session) -> Optional[ChatThread]:
    """
    Get thread by ID, scoped to tenant_id + org_unit_id. Returns None if
    the thread doesn't exist OR belongs to a different tenant/department —
    callers should treat all cases identically (404), never distinguish
    them in the response, to avoid leaking whether a given thread_id
    exists at all.
    """
    return db.query(ChatThread).filter(
        ChatThread.id == uuid.UUID(thread_id),
        ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()


def get_thread_messages(thread_id: str, tenant_id: str, org_unit_id: str, db: Session) -> list[ChatMessage]:
    """Get all messages for a thread ordered by time, scoped to tenant_id + org_unit_id."""
    return (
        db.query(ChatMessage)
        .filter(
            ChatMessage.thread_id == uuid.UUID(thread_id),
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.org_unit_id == org_unit_id,
        )
        .order_by(ChatMessage.created_at.asc())
        .all()
    )