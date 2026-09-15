"""
api/routes/chat.py
------------------
All chat endpoints.

Multimodal chat:
  POST /chat/message accepts an optional image (base64) alongside the query.

  When image is attached — TWO parallel paths:
    Path 1: GPT-4o Vision captions the image → caption embedded →
            pgvector similarity search finds related document chunks
    Path 2: Final LLM call uses GPT-4o with BOTH the image bytes AND
            the retrieved context — GPT-4o literally sees the pixels
            and answers using document knowledge simultaneously

Multi-tenancy + department isolation: every endpoint takes tenant_id AND
org_unit_id from the X-Tenant-ID / X-Org-Unit-ID headers. Thread
creation/lookup, message storage, and document retrieval are all scoped
by BOTH together — a chat can never surface another tenant's, or another
department's, documents or thread history.
"""

import base64
import uuid
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from openai import OpenAI

from api.dependencies import get_tenant_id, get_org_unit_id
from api.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    SourceReference,
    ThreadCreateRequest,
    ThreadUpdateRequest,
    ThreadResponse,
    ThreadListResponse,
    MessageListResponse,
    DeleteResponse,
)
from db.database import get_db_session_fastapi
from db.models import ChatThread, ChatMessage
from pipeline.retriever import retrieve, retrieve_multi, search, build_filter
from pipeline import embedder, vector_store, storage
from pipeline.memory import load_memory_from_db, build_prompt_with_history, select_attachable_crops
from pipeline.chat import chat_complete, chat_stream, generate_thread_title, generate_search_queries
from pipeline.chat_storage import (
    create_thread, update_thread_title, increment_message_count,
    save_human_message, save_ai_message,
    get_thread, get_thread_messages,
)
from pipeline.image_processor import (
    caption_image_with_vision,
    build_image_embedding_text,
    should_skip_image,
)
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

# 2026-08-23: the model used for BOTH legs of a user-attached chat image —
# captioning it for retrieval, and answering with it. Deliberately
# settings.MAP_MODEL, not settings.VISION_MODEL: VISION_MODEL (gpt-4o) is
# the ingestion-time reader for document charts and tables, where a
# one-time cost buys a caption reused forever. Live chat traffic is
# per-request and stays on the cheaper model.
#
# Note this path is NOT gated by settings.VISION_ENABLED — that switch
# governs ingestion spend only. A user attaching an image to their
# question always gets it read (see force=True below).
CHAT_IMAGE_MODEL = settings.MAP_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Image-aware retrieval helper
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve_with_image(
    query:         str,
    image_base64:  str,
    image_media_type: str,
    tenant_id:     str,
    org_unit_id:   str,
    top_k:         int = 5,
    doc_filter:    Optional[str] = None,
) -> tuple[list[dict], Optional[str]]:
    """
    When user attaches an image:
    1. Caption it with GPT-4o Vision to understand what it shows
    2. Build composite embedding text from caption
    3. Use that text to search Qdrant (finds both text AND image chunks) —
       scoped to tenant_id + org_unit_id like every other retrieval path
    4. Merge with a plain text-query search, rerank the UNION against the
       original query once (fixes the old version's separate top_k slices
       being sorted on two different similarity scales before merging)
    Returns (chunks, image_caption_text)
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        # force=True — settings.VISION_ENABLED is an INGESTION cost switch
        # (see config/settings.py). The image here was attached by the user
        # to this question, and captioning it is what makes their own image
        # searchable against the knowledge base; turning that off would
        # break the feature rather than save ingestion spend, so this path
        # deliberately ignores the toggle. The answering call below
        # (_chat_complete_with_image) sends the same image straight to
        # gpt-4o and was never routed through the toggle at all.
        caption = caption_image_with_vision(
            image_bytes=image_bytes, surrounding_text=query,
            force=True, model=settings.MAP_MODEL,
        )

        if caption is None:
            logger.warning("Vision captioning of user image failed — falling back to text search")
            return retrieve(query=query, tenant_id=tenant_id, org_unit_id=org_unit_id,
                           top_k=top_k, doc_filter=doc_filter), None

        caption_search_text = build_image_embedding_text(
            caption=caption, doc_name="user_query", surrounding_text=query,
        )

        query_filter = build_filter(tenant_id=tenant_id, org_unit_id=org_unit_id, doc_name=doc_filter)
        image_chunks = search(query=caption_search_text, query_filter=query_filter,
                               mode="hybrid", top_k=top_k * 2, do_rerank=False)
        text_chunks = search(query=query, query_filter=query_filter,
                              mode="hybrid", top_k=top_k * 2, do_rerank=False)

        # Merge by chunk id, then rerank the UNION against the original
        # query once — a single consistent ranking, not two separately
        # top_k'd lists stitched together.
        by_id = {c["id"]: c for c in image_chunks + text_chunks}
        merged = list(by_id.values())
        if merged:
            scores = embedder.rerank(query, [c.get("chunk_text", "") for c in merged])
            for c, s in zip(merged, scores):
                c["similarity"] = float(s)
            merged.sort(key=lambda c: c["similarity"], reverse=True)

        logger.info(
            "Image-aware retrieval: caption type=%s, found %d merged chunks",
            caption.image_type, len(merged[:top_k]),
        )
        return merged[:top_k], caption.caption

    except Exception as e:
        logger.warning("Image-aware retrieval failed — falling back to text: %s", e)
        return retrieve(query=query, tenant_id=tenant_id, org_unit_id=org_unit_id,
                       top_k=top_k, doc_filter=doc_filter), None


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal LLM call — GPT-4o sees image + retrieved context
# ─────────────────────────────────────────────────────────────────────────────

def _chat_complete_with_image(
    prompt:           str,
    image_base64:     str,
    image_media_type: str,
    crop_paths:       Optional[list[str]] = None,
) -> dict:
    """
    Call GPT-4o with both the image bytes AND the text prompt.
    GPT-4o literally sees the image pixels alongside the document context.
    Returns {"content": str, "prompt_tokens": int, "completion_tokens": int}

    crop_paths: 2026-08-23 — the retrieved documents' OWN visual crops
    (pipeline/memory.py::select_attachable_crops()), attached here too, so
    this path gets the same "read the chart yourself" treatment the
    text-only path does. Ordering is load-bearing: text prompt, then the
    document crops IN SELECTION ORDER, then the user's uploaded image
    LAST. The prompt refers to them as "ATTACHED VISUAL k of N" (memory.py
    writes those pointers off the same helper), so a document visual must
    never be pushed out of position by the user's own attachment — hence
    the user's image goes at the end, not the front where it used to sit.
    """
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        content: list[dict] = [{"type": "text", "text": prompt}]
        for path in crop_paths or []:
            try:
                with open(path, "rb") as f:
                    crop_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else "jpeg"
                mime = "jpeg" if ext == "jpg" else ext
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{crop_b64}", "detail": "high"},
                })
            except Exception as e:
                logger.warning("Failed to load document crop '%s' for multimodal call (skipping it): %s", path, e)
        content.append({
            "type": "image_url",
            "image_url": {
                "url":    f"data:{image_media_type};base64,{image_base64}",
                "detail": "high",
            },
        })

        response = client.chat.completions.create(
            model=CHAT_IMAGE_MODEL,
            max_tokens=1500,
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )

        return {
            "content":           response.choices[0].message.content,
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        logger.error("Multimodal GPT-4o call failed: %s", e)
        raise RuntimeError(f"Multimodal LLM call failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared chat preparation
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve_chunks_sync(request: ChatMessageRequest, tenant_id: str, org_unit_id: str, memory) -> tuple:
    """
    All actual retrieval work — dense/sparse query embedding, the Qdrant
    call(s), and cross-encoder reranking (pipeline/embedder.py::rerank(),
    ~10-15s of pure CPU-bound torch inference on a GPU-less box) — is
    synchronous. This MUST be called via run_in_threadpool() from the
    route handler, never awaited/invoked directly on the event loop: this
    repo runs uvicorn with --workers 1 (single process, single event
    loop — Dockerfile/CLAUDE.md), so a blocking call made directly there
    freezes the ENTIRE API, not just the request that triggered it — every
    other in-flight request (even an unrelated GET /health/) stalls until
    it returns. Confirmed live: a chat request appeared to hang
    indefinitely and only "resolved" when the dev server was stopped —
    that was the event loop being fully blocked, not a real deadlock; the
    call was still running to completion underneath the frozen loop.

    Returns (chunks, image_caption) — same shape _prepare_chat needs.
    """
    if request.image_base64:
        return _retrieve_with_image(
            query            = request.query,
            image_base64     = request.image_base64,
            image_media_type = request.image_media_type or "image/jpeg",
            tenant_id        = tenant_id,
            org_unit_id      = org_unit_id,
            top_k            = request.top_k,
            doc_filter       = request.doc_filter,
        )

    history_messages = memory.chat_memory.messages
    if history_messages:
        document_summary = None
        if request.doc_filter:
            document_summary = storage.get_summary_by_name(request.doc_filter, tenant_id, org_unit_id)
        search_queries = generate_search_queries(request.query, history_messages, document_summary=document_summary)
        chunks = retrieve_multi(
            queries=search_queries, tenant_id=tenant_id, org_unit_id=org_unit_id,
            top_k=request.top_k, doc_filter=request.doc_filter,
        )
    else:
        chunks = retrieve(
            query=request.query, tenant_id=tenant_id, org_unit_id=org_unit_id,
            top_k=request.top_k, doc_filter=request.doc_filter,
        )
    return chunks, None


async def _prepare_chat(request: ChatMessageRequest, tenant_id: str, org_unit_id: str, db: Session) -> tuple:
    is_new_thread = False
    if request.thread_id is None:
        # A thread created implicitly by the global chat endpoint is a global
        # conversation.  The document filter belongs to this request only;
        # persisting it on the thread makes the UI treat the first document as
        # the scope for every later message.  Explicit document-chat threads
        # are still created through POST /chat/threads and retain doc_filter.
        thread = create_thread(tenant_id=tenant_id, org_unit_id=org_unit_id, doc_filter=None)
        thread_id = str(thread.id)
        is_new_thread = True
    else:
        thread_id = str(request.thread_id)
        # Scoped by tenant_id + org_unit_id — a thread_id belonging to
        # another tenant OR another department returns None here exactly
        # like a nonexistent one.
        thread = get_thread(thread_id, tenant_id, org_unit_id, db)
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")

    memory = load_memory_from_db(thread_id, tenant_id, org_unit_id, db)

    # ── Retrieval — image-aware if image attached, always scoped ───────────
    # 2026-08-21 — conversational retrieval fix (query condensation, see
    # generate_search_queries) lives inside _retrieve_chunks_sync now. The
    # run_in_threadpool() wrapper below is the actual bug fix from this
    # pass: this whole call chain (embedding, Qdrant, reranking) is
    # blocking CPU/IO work — see _retrieve_chunks_sync's docstring for why
    # calling it directly here would freeze the entire API, not just this
    # request.
    chunks, image_caption = await run_in_threadpool(_retrieve_chunks_sync, request, tenant_id, org_unit_id, memory)

    prompt = build_prompt_with_history(
        query          = request.query,
        memory         = memory,
        context_chunks = chunks,
        doc_filter     = request.doc_filter,
    )

    return thread_id, is_new_thread, memory, chunks, prompt, image_caption


def _build_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_name":    c["doc_name"],
            "chunk_index": c["chunk_index"],
            "chunk_text":  c["chunk_text"][:200],
            "similarity":  c.get("similarity"),
            "role":        c.get("role", "text"),
            "image_type":  c.get("image_type"),
            "image_url":   c.get("image_url"),
            "page_number":     c.get("page_number"),
            "section_heading": c.get("section_heading"),
            "table_html":      c.get("table_html"),
        }
        for c in chunks
    ]


async def _save_turn(
    thread_id: str, tenant_id: str, org_unit_id: str, is_new_thread: bool,
    query: str, answer: str, sources: list,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    image_caption: Optional[str] = None,
) -> Optional[str]:
    # run_in_threadpool: save_*_message()/increment_message_count() are
    # synchronous SQLAlchemy calls, and generate_thread_title() below makes
    # a blocking OpenAI call — same event-loop-freezing concern as the
    # retrieval/chat-completion calls in send_message() above.
    await run_in_threadpool(
        save_human_message, thread_id=thread_id, tenant_id=tenant_id, org_unit_id=org_unit_id, content=query,
    )
    await run_in_threadpool(
        save_ai_message,
        thread_id=thread_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
        content=answer, sources=sources,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        has_image=image_caption is not None, image_caption=image_caption,
    )
    await run_in_threadpool(increment_message_count, thread_id, tenant_id, org_unit_id)

    thread_title = None
    if is_new_thread:
        thread_title = await run_in_threadpool(generate_thread_title, query)
        await run_in_threadpool(update_thread_title, thread_id, tenant_id, org_unit_id, thread_title)
    return thread_title


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/message — full JSON response (multimodal)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=200,
    summary="Send a message and get full JSON response",
    description=(
        "Send a text query, optionally with an image (base64). "
        "When image is provided, GPT-4o Vision sees the actual image pixels "
        "alongside retrieved document context — dual-path: "
        "similarity search finds related knowledge, GPT-4o answers about the image."
    ),
)
async def send_message(
    request: ChatMessageRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread_id, is_new_thread, memory, chunks, prompt, image_caption = await _prepare_chat(
        request, tenant_id, org_unit_id, db
    )

    # Fallback when no chunks found
    if not chunks:
        fallback_msg = (
            "I can see the image you've shared, but I don't have relevant information "
            "in the knowledge base to answer your question about it."
            if request.image_base64
            else "I don't have relevant information in the documents to answer your question."
        )
        await _save_turn(
            thread_id, tenant_id, org_unit_id, is_new_thread, request.query, fallback_msg, [],
            image_caption=image_caption,
        )
        return ChatMessageResponse(
            message_id   = uuid.uuid4(),
            thread_id    = uuid.UUID(thread_id),
            role         = "ai",
            content      = fallback_msg,
            sources      = [],
            thread_title = (await run_in_threadpool(generate_thread_title, request.query)) if is_new_thread else None,
            created_at   = __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            image_understood = bool(request.image_base64),
            image_caption = image_caption,
        )

    # ── LLM call — multimodal if image attached, text-only otherwise ──────
    # run_in_threadpool: both branches make a blocking OpenAI HTTP call
    # (langchain_openai's ChatOpenAI.invoke() / the raw openai client's
    # chat.completions.create() are both synchronous) — same event-loop-
    # freezing risk as the retrieval call above, just smaller (seconds,
    # not ~10-15s), still enough to stall concurrent requests under
    # --workers 1.
    # 2026-08-23: the retrieved documents' OWN visual crops — charts,
    # diagrams and table regions alike (was tables-only). These are the
    # ORIGINAL pixels, re-attached so the model reads the values itself
    # with the question in hand, instead of answering from the prose
    # caption GPT-4o Vision wrote once at ingestion, before the question
    # existed. select_attachable_crops() is the SAME helper
    # pipeline/memory.py::build_prompt_with_history() used to number the
    # "ATTACHED VISUAL k of N" pointers in the prompt — one shared source
    # so the prompt's numbering and what's actually attached here can't
    # drift apart. Computed before the branch so BOTH answering paths
    # attach the identical set.
    crop_paths = [c["crop_image_path"] for c in select_attachable_crops(chunks)]

    if request.image_base64:
        logger.info("Multimodal chat: image + text query, thread=%s", thread_id[:8])
        result = await run_in_threadpool(
            _chat_complete_with_image,
            prompt, request.image_base64, request.image_media_type or "image/jpeg",
            crop_paths or None,
        )
        image_understood = True
    else:
        result = await run_in_threadpool(chat_complete, prompt, crop_paths or None)
        image_understood = False

    answer  = result["content"]
    sources = _build_sources(chunks)

    thread_title = await _save_turn(
        thread_id=thread_id, tenant_id=tenant_id, org_unit_id=org_unit_id, is_new_thread=is_new_thread,
        query=request.query, answer=answer, sources=sources,
        prompt_tokens=result.get("prompt_tokens"),
        completion_tokens=result.get("completion_tokens"),
        image_caption=image_caption,
    )

    return ChatMessageResponse(
        message_id       = uuid.uuid4(),
        thread_id        = uuid.UUID(thread_id),
        role             = "ai",
        content          = answer,
        sources          = [SourceReference(**s) for s in sources],
        thread_title     = thread_title,
        created_at       = __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        image_understood = image_understood,
        image_caption    = image_caption,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thread CRUD — all tenant + org-unit scoped
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/threads", response_model=ThreadResponse, status_code=201, summary="Create a new thread")
async def create_thread_endpoint(
    request: ThreadCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = create_thread(
        tenant_id=tenant_id, org_unit_id=org_unit_id, title=request.title,
        user_id=request.user_id, doc_filter=request.doc_filter,
    )
    return ThreadResponse(
        id=thread.id, title=thread.title, user_id=thread.user_id,
        doc_filter=thread.doc_filter, message_count=thread.message_count or 0,
        created_at=thread.created_at, updated_at=thread.updated_at,
        tenant_id=thread.tenant_id, org_unit_id=thread.org_unit_id,
    )


@router.get("/threads", response_model=ThreadListResponse, summary="List all threads")
async def list_threads(
    page: int = Query(default=1, ge=1), per_page: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    offset = (page - 1) * per_page
    base_query = db.query(ChatThread).filter(
        ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    )
    total  = base_query.count()
    threads = base_query.order_by(ChatThread.updated_at.desc()).offset(offset).limit(per_page).all()
    return ThreadListResponse(total=total, threads=[
        ThreadResponse(id=t.id, title=t.title, user_id=t.user_id, doc_filter=t.doc_filter,
                       message_count=t.message_count or 0, created_at=t.created_at,
                       updated_at=t.updated_at, tenant_id=t.tenant_id, org_unit_id=t.org_unit_id)
        for t in threads
    ])


@router.get("/threads/{thread_id}", response_model=ThreadResponse, summary="Get thread by ID")
async def get_thread_endpoint(
    thread_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    return ThreadResponse(id=thread.id, title=thread.title, user_id=thread.user_id,
                          doc_filter=thread.doc_filter, message_count=thread.message_count or 0,
                          created_at=thread.created_at, updated_at=thread.updated_at,
                          tenant_id=thread.tenant_id, org_unit_id=thread.org_unit_id)


@router.patch("/threads/{thread_id}", response_model=ThreadResponse, summary="Update thread title")
async def update_thread_endpoint(
    thread_id: uuid.UUID, request: ThreadUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    thread.title = request.title
    db.commit()
    db.refresh(thread)
    return ThreadResponse(id=thread.id, title=thread.title, user_id=thread.user_id,
                          doc_filter=thread.doc_filter, message_count=thread.message_count or 0,
                          created_at=thread.created_at, updated_at=thread.updated_at,
                          tenant_id=thread.tenant_id, org_unit_id=thread.org_unit_id)


@router.delete("/threads/{thread_id}", response_model=DeleteResponse, summary="Delete thread and all its messages")
async def delete_thread_endpoint(
    thread_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    # Postgres chat_messages cascade-deletes automatically via ON DELETE
    # CASCADE; Qdrant has no cascade, so its points are deleted explicitly.
    vector_store.delete_thread_messages(tenant_id=tenant_id, thread_id=str(thread_id))
    db.delete(thread)
    db.commit()
    return DeleteResponse(id=thread_id)


@router.get("/threads/{thread_id}/messages", summary="Get all messages in a thread")
async def get_messages_endpoint(
    thread_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.thread_id == thread_id,
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.org_unit_id == org_unit_id,
        )
        .order_by(ChatMessage.created_at.asc()).all()
    )
    return {
        "thread_id": str(thread_id), "thread_title": thread.title, "total": len(messages),
        "messages": [
            {"id": str(m.id), "role": m.role, "content": m.content,
             "sources": m.sources or [], "created_at": str(m.created_at)}
            for m in messages
        ]
    }


@router.get("/search", summary="Semantic search across chat history")
async def search_chat_history(
    query: str = Query(..., min_length=1), top_k: int = Query(default=5, ge=1, le=20),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    """
    2026-08-21: now backed by Qdrant's avabodh_chat_messages collection
    (pipeline/vector_store.py::search_chat) instead of chat_messages.embedding
    (that pgvector column is gone). thread_title is looked up from Postgres
    per hit — chat message vectors don't carry it in their payload.
    """
    query_vector = embedder.embed_dense_query(query)
    try:
        hits = vector_store.search_chat(
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            dense_vector=query_vector, top_k=top_k,
        )
        results = []
        for h in hits:
            thread = db.query(ChatThread).filter(ChatThread.id == uuid.UUID(h["thread_id"])).first()
            results.append({
                "id": h["message_id"], "role": h["role"], "content": h["content"][:300],
                "thread_id": h["thread_id"], "thread_title": thread.title if thread else None,
                "similarity": round(float(h["score"]), 4), "created_at": h.get("created_at"),
            })
        return {"query": query, "total": len(results), "results": results}
    except Exception as e:
        logger.exception("Chat history search failed (query=%r): %s", query, e)
        raise HTTPException(status_code=500, detail="Search failed. Contact support with the X-Request-ID response header if this persists.")
