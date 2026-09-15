"""
pipeline/chat.py
----------------
Core chat pipeline:
1. Build prompt from memory + retrieved chunks
2. LLM call with streaming support
3. Collect full response after stream
4. Generate thread title (first message only)

Two modes:
- stream=True  → yields tokens one by one via SSE
- stream=False → returns complete response at once
"""

import base64
import os
import time
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _build_llm(streaming: bool = False, for_answer: bool = False) -> ChatOpenAI:
    """
    for_answer: True for the call that actually answers a user's question
    (chat_complete / chat_stream), False for incidental helper calls like
    thread-title generation.

    2026-08-23 — why the split. Both used to be settings.MAP_MODEL, the
    model picked for cheap per-chunk ingestion summaries. That meant a
    chart crop re-attached to an answer was read by gpt-4o-mini, while a
    user attaching their OWN image went down api/routes/chat.py's
    multimodal path and got gpt-4o. Same pixels, weaker reader, purely
    because of which door the image came through. Answers now use
    settings.CHAT_MODEL (gpt-4o); helper calls stay on MAP_MODEL, so a
    six-word thread title doesn't cost gpt-4o rates. Ingestion is
    untouched — MAP_MODEL still drives summarisation exactly as before.

    Temperature comes from settings.CHAT_TEMPERATURE (0.1) rather than the
    hardcoded 0.3 it was: this call does exact-figure extraction against a
    fixed response schema, and both hold together better as sampling
    tightens.
    """
    temperature = settings.CHAT_TEMPERATURE
    if settings.use_ollama:
        if ChatOllama is not None:
            return ChatOllama(
                model=settings.OLLAMA_CHAT_MODEL,
                base_url=settings.ollama_url,
                temperature=temperature,
                max_tokens=1024,
                streaming=streaming,
            )
        return ChatOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_url.rstrip('/')}/v1",
            model=settings.OLLAMA_CHAT_MODEL,
            temperature=temperature,
            max_tokens=1024,
            streaming=streaming,
        )
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.CHAT_MODEL if for_answer else settings.MAP_MODEL,
        temperature=temperature,
        max_tokens=1024,
        streaming=streaming,
    )


def generate_thread_title(first_query: str) -> str:
    """
    Auto-generate a short thread title from the first message.
    Called only once per thread — on first message.
    """
    try:
        llm = _build_llm(streaming=False)
        messages = [
            SystemMessage(content="Generate a short 4-6 word title for this conversation. Return only the title, nothing else."),
            HumanMessage(content=first_query),
        ]
        response = llm.invoke(messages)
        title = response.content.strip().strip('"').strip("'")
        return title[:100]
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return first_query[:50]


class SearchQueries(BaseModel):
    """
    2026-08-21 — fixes a real conversational-retrieval bug found in
    testing: a follow-up like "which was the least?" was embedded and
    searched on its OWN, with no idea what "least" refers to (the topic
    lived in the PREVIOUS turn, not this one) — retrieval found nothing
    relevant, and the LLM correctly (if confusingly) said "not enough
    information" about context it never actually received.

    2026-08-21 (v2) — a second real bug found in testing: the LLM would
    silently substitute a synonym for a specific document term while
    condensing ("Afrobeats streams" -> "revenue generated from Afrobeats
    streams"), which sent retrieval after the wrong metric entirely. Field
    descriptions below now explicitly forbid that.
    """
    primary_query: str = Field(
        description="A standalone, fully self-contained rewrite of the user's latest message — "
                    "resolve any pronoun or implicit reference (\"it\", \"least\", \"that one\") "
                    "using the conversation history, so this reads as a complete question with no "
                    "missing context. Preserve every specific noun/term from the conversation "
                    "EXACTLY as written (e.g. \"streams\", \"revenue\", \"booking value\" are "
                    "different things — never substitute one for another, even if they seem related)."
    )
    alternate_queries: list[str] = Field(
        default_factory=list, max_length=2,
        description="Up to 2 additional phrasings or related-angle queries covering the SAME "
                    "information need as primary_query, worded differently to broaden retrieval "
                    "coverage (different sentence structure, a narrower or broader framing) — but "
                    "still preserving the same exact specific terms/nouns as primary_query, not "
                    "synonyms for them."
    )


def generate_search_queries(query: str, history_messages: list, document_summary: Optional[str] = None) -> list[str]:
    """
    Query condensation + multi-query expansion, combined into one LLM
    call. Only meaningful when there's actual history to condense against
    — the caller (api/routes/chat.py) skips this entirely on a thread's
    first message, where the raw query already IS standalone.

    document_summary: the target document's stored summary_text (Postgres
    Document.summary_text, looked up by the caller via
    pipeline/storage.py::get_summary_by_name() when request.doc_filter
    scopes this chat to one document) — grounds condensation in what the
    document is actually about, not just the last few chat turns. None
    when no single document is in scope (doc_filter unset) — omitted from
    the prompt entirely rather than guessed at.

    Returns up to 3 queries: [primary_query, *alternate_queries]. Falls
    back to [query] (today's exact single-query behavior) on any failure
    — this must never be a hard blocker for chat working at all.
    """
    if not history_messages:
        return [query]

    try:
        history_str = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history_messages
        )
        summary_block = (
            f"DOCUMENT SUMMARY (what the in-scope document is about — use this to disambiguate "
            f"terms, not to answer the question):\n{document_summary}\n\n"
            if document_summary else ""
        )
        # Use the same provider fallback as the rest of the chat pipeline.
        # Calling ChatOpenAI directly here fails on Ollama-only deployments
        # as soon as a thread has history (the first message skips this path).
        llm = _build_llm(streaming=False, for_answer=False)
        structured_llm = llm.with_structured_output(SearchQueries)
        result: SearchQueries = structured_llm.invoke(
            f"{summary_block}"
            f"CONVERSATION SO FAR:\n{history_str}\n\n"
            f"ORIGINAL PROMPT (the user's latest message, verbatim): {query}\n\n"
            "Using the conversation (and document summary, if given) above to resolve pronouns "
            "and implicit references in the ORIGINAL PROMPT, respond with search queries in "
            "exactly this shape:\n"
            "  primary_query   — a standalone, self-contained rewrite of the ORIGINAL PROMPT\n"
            "  alternate_1     — a different phrasing/angle of the same information need\n"
            "  alternate_2     — another different phrasing/angle of the same information need\n"
            "Do not substitute synonyms for specific terms from the conversation — preserve them "
            "exactly."
        )
        queries = [result.primary_query] + list(result.alternate_queries)[:2]
        queries = [q.strip() for q in queries if q and q.strip()]
        final = queries[:3] if queries else [query]
        logger.info("Condensed query %r (with %d history msgs) -> %r", query, len(history_messages), final)
        return final
    except Exception as e:
        logger.warning("Search query generation failed — falling back to raw query %r: %s", query, e)
        return [query]


def chat_complete(prompt: str, image_paths: Optional[list[str]] = None) -> dict:
    """
    Non-streaming LLM call.
    Returns complete response dict with content and token usage.

    image_paths: 2026-08-22 — actual table-crop images (from
    api/routes/chat.py, collected off retrieved chunks whose
    table_html extraction was unreliable — see
    image_processor.py::table_html_is_reliable() and
    pipeline/memory.py's "(Visual — see attached image)" sublabel) to
    attach directly to this call, so the model reads the table's real
    pixels itself instead of trusting only the one-time Vision caption
    made at ingestion. detail="high" — a table's small text/numbers need
    the multi-tile resolution; "low" downscales to a single 512x512 tile
    and would likely blur exactly the values this exists to recover.
    None/empty behaves exactly as before (plain text-only call).
    """
    llm = _build_llm(streaming=False, for_answer=True)
    start = time.time()

    try:
        if image_paths:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for path in image_paths:
                try:
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpeg"
                    mime = "jpeg" if ext == "jpg" else ext
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "high"},
                    })
                except Exception as e:
                    logger.warning("Failed to load table crop image '%s' for chat_complete (skipping it): %s", path, e)
            messages = [HumanMessage(content=content)]
        else:
            messages = [HumanMessage(content=prompt)]

        response = llm.invoke(messages)
        elapsed = round(time.time() - start, 2)

        return {
            "content":           response.content.strip(),
            "prompt_tokens":     getattr(response.usage_metadata, "input_tokens", None),
            "completion_tokens": getattr(response.usage_metadata, "output_tokens", None),
            "elapsed_sec":       elapsed,
        }
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise RuntimeError(f"LLM call failed: {e}")


async def chat_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Streaming LLM call.
    Yields tokens one by one as SSE events.
    Also collects full response for storage after stream ends.

    Usage:
        async for token in chat_stream(prompt):
            yield f"data: {token}\n\n"
    """
    llm = _build_llm(streaming=True, for_answer=True)
    full_response = []

    try:
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                full_response.append(token)
                yield token

        # Signal stream end
        yield "[DONE]"

    except Exception as e:
        logger.error("Streaming LLM call failed: %s", e)
        yield f"[ERROR]: {e}"

    # Store full response on the generator object for saving to DB later
    chat_stream.last_response = "".join(full_response)
