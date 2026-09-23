"""
pipeline/memory.py
------------------
Chat memory management using ConversationBufferWindowMemory.

Loads past N turns from chat_messages table and injects them into prompt.
This is how the chatbot remembers conversation history.

Window = last N turns only (not entire history) to avoid token overflow.
"""
import io
import os
import uuid
from typing import Optional
import pandas as pd
from sqlalchemy.orm import Session
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from db.models import ChatMessage, ChatThread
from pipeline import embedder, vector_store
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _flatten_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    pandas.read_html() resolves a multi-row header (rowspan/colspan) into
    a MultiIndex, e.g. columns [('Rank','Rank'), ('Demographics','Area')]
    — collapse each into one readable label ("Rank", "Demographics /
    Area") for the markdown header row. Drops pandas' own "Unnamed: N"
    placeholder for cells that were genuinely blank in the source table,
    and de-duplicates a level repeated against itself (single-row headers
    read_html still wraps in a 1-level MultiIndex-shaped tuple sometimes).
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for col in df.columns:
        parts = [str(c) for c in col if str(c) and not str(c).startswith("Unnamed")]
        deduped = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]
        new_cols.append(" / ".join(deduped) if deduped else "")
    df.columns = new_cols
    return df


def _html_table_to_markdown(html: str) -> str:
    """
    Convert <table> markup (pipeline/chunker.py's table_html, sourced from
    unstructured's el.metadata.text_as_html) into GitHub-flavored markdown
    pipe tables for the POSSIBLY HELPFUL TABLES prompt section.

    Content-unaltered, format-only change — every cell value is carried
    over exactly as extracted, nothing is summarized or reworded (unlike
    the GPT-4o Vision table-crop caption path, which necessarily
    compresses a table into prose). Markdown tables are also more
    token-efficient than raw HTML (no tag overhead) and are what LLMs are
    most commonly trained/evaluated on for tabular reasoning.

    2026-08-22: uses pandas.read_html() rather than a hand-rolled
    BeautifulSoup row/cell walk or markdownify (already a project
    dependency via pipeline/scraper.py) — tested both against a table
    with a rowspan/colspan header (a real pattern in our tables) and both
    produced a MISALIGNED markdown table: the second header row gets
    treated as an ordinary data row, so column counts stop matching
    row-to-row. pandas.read_html() correctly resolves that into a proper
    MultiIndex instead (see _flatten_table_columns above) — this is the
    actual gap that made pandas worth the extra dependency (tabulate,
    for DataFrame.to_markdown()) over what was already in the stack.

    A chunk can carry MULTIPLE concatenated <table> blocks (chunker.py
    doesn't split them) — read_html() already returns one DataFrame per
    <table> found; each is converted and joined with a blank line.

    Falls back to returning the original HTML unchanged if pandas can't
    parse it (malformed markup, or no <table> found at all) — safer than
    silently dropping the table's data from the prompt.
    """
    try:
        dfs = pd.read_html(io.StringIO(html))
    except Exception as e:
        logger.debug("Table HTML->markdown conversion failed, sending raw HTML instead: %s", e)
        return html
    if not dfs:
        return html

    md_blocks = []
    for df in dfs:
        try:
            df = _flatten_table_columns(df).fillna("")   # empty <td></td> cells -> "", not the literal string "nan"
            md_blocks.append(df.to_markdown(index=False))
        except Exception as e:
            logger.debug("Markdown rendering failed for one table block, skipping it: %s", e)
            continue

    return "\n\n".join(md_blocks) if md_blocks else html


def select_attachable_crops(context_chunks: list[dict], limit: Optional[int] = None) -> list[dict]:
    """
    The retrieved chunks whose ORIGINAL crop image will be attached to the
    answering LLM call, in the order they'll be attached.

    2026-08-23 — this is deliberately a SHARED helper: both
    build_prompt_with_history() (which writes "ATTACHED VISUAL k of N"
    pointers into the prompt) and api/routes/chat.py (which actually
    encodes and attaches the files) must agree on exactly which chunks are
    attached and in what order, or the prompt's numbering points at the
    wrong image. One function, called by both, is the only way that can't
    silently drift.

    os.path.exists() is checked, not assumed: crop_image_path is written
    into the Qdrant payload at ingestion, and points ingested BEFORE crops
    were persisted for charts/diagrams (or whose crop file was since
    cleaned up) carry a path that no longer resolves. Those fall back to
    their caption text rather than crashing the call — see
    build_prompt_with_history().
    """
    if limit is None:
        limit = getattr(settings, "MAX_ATTACHED_CROPS", 6) or 6
    selected = []
    for c in context_chunks or []:
        if c.get("role") != "image":
            continue
        path = c.get("crop_image_path")
        if not path:
            continue
        try:
            if not os.path.exists(path):
                logger.debug("Crop file missing on disk, falling back to caption: %s", path)
                continue
        except Exception:
            continue
        selected.append(c)
        if len(selected) >= limit:
            break
    return selected


def load_memory_from_db(
    thread_id: str, tenant_id: str, org_unit_id: str, db: Session,
) -> tuple[ConversationBufferWindowMemory, Optional[str], set[str]]:
    """
    Returns (memory, rolling_summary, window_message_ids):
      - memory: ConversationBufferWindowMemory holding every message since
        rolling_summary_through, injected verbatim.
      - rolling_summary: ChatThread.rolling_summary — everything UP TO
        rolling_summary_through, compressed (see pipeline/chat_storage.py::
        update_rolling_summary()). None for a thread with no summary yet.
      - window_message_ids: the ChatMessage.id values already included
        verbatim in `memory` — passed to select_relevant_history() below
        so semantic recall never re-surfaces a message the prompt is
        already sending in full.

    2026-09-23 — the window is no longer a fixed "last N turns". It is
    defined as "every message after rolling_summary_through", whatever
    that count happens to be. Summary and window are two halves of one
    partition of the thread's full history — they meet exactly, by
    construction, instead of being computed independently and only
    USUALLY lining up.

    Why that matters: pipeline/chat_storage.py::update_rolling_summary()
    can lag behind — a slow LLM call, a transient failure it's still
    retrying, or (if it's ever moved off the request path, e.g. to a
    background task) simply not having run yet for this turn. With the
    OLD fixed-N window, a lagging summarizer meant messages that had aged
    out of the window but weren't in the summary yet were in NEITHER —
    invisible to the model. With the window derived from
    rolling_summary_through instead, a lagging summarizer just means
    the window is bigger for this one prompt; nothing is ever dropped.

    settings.MEMORY_WINDOW_SIZE still does exactly what it did before in
    the normal case: update_rolling_summary() only advances
    rolling_summary_through once a thread has more than
    MEMORY_WINDOW_SIZE turns past it (see that function), so in steady
    state (summarizer keeping pace) rolling_summary_through sits at
    total_messages - MEMORY_WINDOW_SIZE*2 and this query naturally returns
    the same "last N turns" as before. MEMORY_WINDOW_SIZE is the
    configured fallback/steady-state size, not a hard cap — it only stops
    being the literal window size when the background job is behind,
    which is exactly the case where showing MORE history is correct.

    tenant_id + org_unit_id: filtered here even though thread_id alone is
    already effectively unique to one tenant+org_unit (every
    ChatMessage.thread_id points at exactly one ChatThread, which belongs
    to exactly one tenant and department, and the caller already
    validated thread ownership via get_thread() before reaching this
    call). Filtering by both anyway means this query stays correct even
    if that upstream check is ever refactored away.
    """
    window_message_ids: set[str] = set()
    rolling_summary: Optional[str] = None
    rolling_summary_through = 0

    try:
        thread = db.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id),
            ChatThread.tenant_id == tenant_id,
            ChatThread.org_unit_id == org_unit_id,
        ).first()
        if thread:
            rolling_summary = thread.rolling_summary
            rolling_summary_through = thread.rolling_summary_through or 0
    except Exception as e:
        logger.warning("Rolling summary load failed for thread %s: %s", thread_id, e)

    # 2026-08-23 — this used to be ORDER BY created_at ASC LIMIT 10 (no
    # offset), which took the OLDEST ten messages, not the newest — the
    # window froze at a thread's first five turns and never advanced.
    # Confirmed against live data (a 24-message thread whose memory still
    # held only its opening turns, where "can you repeat the last
    # response?" got an answer from twenty messages earlier). Not just a
    # display problem — pipeline/chat.py::generate_search_queries()
    # condenses follow-ups against these same messages, so the stale
    # window sent retrieval after the wrong topic too.
    #
    # ASC + OFFSET(rolling_summary_through), no LIMIT: returns exactly
    # "everything not yet folded into the summary" — see this function's
    # docstring for why that replaced a fixed LIMIT. Postgres can seek to
    # the offset via the thread_id/created_at index rather than pulling
    # the whole thread back to slice it in Python, same performance shape
    # as the old DESC+LIMIT+reverse approach in the steady-state case.
    try:
        messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.thread_id == thread_id,
                ChatMessage.tenant_id == tenant_id,
                ChatMessage.org_unit_id == org_unit_id,
            )
            .order_by(ChatMessage.created_at.asc())
            .offset(rolling_summary_through)
            .all()
        )
    except Exception as e:
        logger.warning("Memory load failed for thread %s: %s", thread_id, e)
        messages = []

    # k must be at least as large as what we're about to inject —
    # ConversationBufferWindowMemory does its OWN trimming to the last k
    # exchanges, which would otherwise silently re-truncate this back down
    # to settings.MEMORY_WINDOW_SIZE turns even when we deliberately
    # included more to cover a lagging summarizer. len(messages) (a
    # message count) is always >= the true turn count, so it's a safe
    # upper bound without needing to pre-count pairs.
    memory = ConversationBufferWindowMemory(
        k=max(settings.MEMORY_WINDOW_SIZE, len(messages)),
        return_messages=True,
        memory_key="chat_history",
        input_key="query",
        output_key="answer",
    )

    try:
        # Inject messages into memory in pairs (human, ai)
        i = 0
        while i < len(messages) - 1:
            human_msg = messages[i]
            ai_msg    = messages[i + 1]

            if human_msg.role == "human" and ai_msg.role == "ai":
                # 2026-08-23 — carry a past turn's ATTACHED IMAGE into the
                # history. The image a user attaches to a question is
                # captioned, used once, and discarded (it is never stored
                # on disk or in Qdrant); only ChatMessage.image_caption
                # survives, and nothing read it back. So a perfectly
                # reasonable follow-up — "what about that chart I sent?" —
                # hit a model with no idea an image had ever existed, one
                # turn after answering about it correctly. Folding the
                # caption into that turn's user text is the cheap fix: no
                # schema change, no extra call, and it only affects turns
                # that actually had an image.
                #
                # Truncated at 400 chars: a Vision caption can run to ~800
                # tokens, and up to five of these can sit in one window —
                # untrimmed they would crowd out the conversation they are
                # meant to support.
                query_text = human_msg.content
                caption = (getattr(ai_msg, "image_caption", None) or "").strip()
                if caption:
                    if len(caption) > 400:
                        caption = caption[:400].rstrip() + "..."
                    query_text = (
                        query_text
                        + chr(10)
                        + "[The user attached an image with this question. It showed: "
                        + caption + "]"
                    )

                memory.save_context(
                    {"query": query_text},
                    {"answer": ai_msg.content},
                )
                window_message_ids.add(str(human_msg.id))
                window_message_ids.add(str(ai_msg.id))
                i += 2
            else:
                i += 1

        logger.info(
            "Loaded %d messages from thread %s into memory",
            len(messages), str(thread_id)[:8],
        )

    except Exception as e:
        logger.warning("Memory load failed for thread %s: %s", thread_id, e)

    return memory, rolling_summary, window_message_ids


def select_relevant_history(
    query: str, thread_id: str, tenant_id: str, org_unit_id: str,
    exclude_ids: set[str],
) -> list[dict]:
    """
    Semantic-recall layer: embeds the CURRENT query and matches it against
    every past message in this thread (already indexed in Qdrant's
    avabodh_chat_messages collection at save time — see
    pipeline/chat_storage.py::_index_message_in_qdrant), scoped to this
    thread_id only via vector_store.search_chat()'s thread_id filter.

    exclude_ids: the message ids already sent verbatim in the recency
    window (pipeline/memory.py::load_memory_from_db's window_message_ids)
    — filtered out here so a message never appears twice in the prompt.

    Non-fatal by design, same as shortlist_documents()/_mmr_select(): a
    failure here should degrade to "no extra recall," never break the
    chat turn — the recency window and rolling summary still carry the
    conversation on their own.
    """
    limit = settings.MEMORY_SEMANTIC_RECALL_LIMIT
    if not query or not query.strip() or limit <= 0:
        return []

    try:
        dense_vector = embedder.embed_dense_query(query)
        hits = vector_store.search_chat(
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            dense_vector=dense_vector,
            top_k=limit + len(exclude_ids),
            thread_id=thread_id,
        )
    except Exception as e:
        logger.warning("Semantic recall over chat history failed (non-fatal): %s", e)
        return []

    results = [h for h in hits if h.get("message_id") not in exclude_ids]
    return results[:limit]


def build_prompt_with_history(
    query: str,
    memory: ConversationBufferWindowMemory,
    context_chunks: list[dict],
    doc_filter: Optional[str] = None,
    attach_crops: bool = True,
    rolling_summary: Optional[str] = None,
    relevant_history: Optional[list[dict]] = None,
) -> str:
    """
    Build the full prompt string:
    - System instructions
    - Retrieved document context
    - Earlier conversation summary (turns older than the recency window)
    - Relevant earlier messages (semantic recall — specific older facts)
    - Conversation history (recency window, verbatim)
    - Current query

    attach_crops: whether the caller is actually going to attach the
    selected crop images to the LLM call. True (the default, and what
    api/routes/chat.py does on both its answering paths) makes every
    visual chunk with a usable crop render as an "ATTACHED VISUAL k of N"
    pointer with NO prose description — the model is expected to read the
    image. Set False only if the call genuinely can't carry images, in
    which case every visual falls back to its ingestion-time Vision
    caption, i.e. exactly the pre-2026-08-23 behavior.

    rolling_summary / relevant_history: pipeline/chat_storage.py's
    ChatThread.rolling_summary and pipeline/memory.py::
    select_relevant_history()'s output — both optional so a thread still
    inside its first recency window (nothing has aged out, no recall
    needed yet) renders exactly the pre-2026-09-23 prompt shape.
    """
    from typing import Optional

    # System message — tightened 2026-08-21: the previous version produced
    # rambling, self-contradicting answers (listing several partial facts,
    # THEN adding a wishy-washy "I don't have enough information" at the
    # end even when it had just demonstrated it did). This version forces
    # a direct, decisive answer instead.

    # 2026-08-23 — condensed from ten overlapping rules to seven, purely a
    # rewrite for length: every instruction the old version carried is still
    # here, just stated once instead of across two or three rules. The merges
    # were old 1+7 (source + consider everything) -> 1; old 2+2a+8 (brevity,
    # response shape, don't narrate) -> 2; old 2b+6+9 (calculate, tables are
    # calculable, match the query's intent) -> 4. Nothing was dropped and
    # nothing new was added — if behaviour changes, this rewrite is the cause
    # and it should be reverted, not patched.
    system = f"""You are Avabodh, a document assistant. Answer directly and precisely.

RULES:
1. SOURCE: Answer only from the data below — never outside knowledge, though related reasoning over that data is appreciated. Take ALL of it into account before answering, and apply semantic sense to complex queries.
2. RESPONSE SCHEMA — mandatory, no exceptions: the answer stated concisely in your own words but within one sentence, then the [Source: ...] tag at the end, nothing else.
3. PARTIAL DATA: State exactly what IS known, then stop — do not follow it with "I don't have enough information," that's a contradiction, and never open with what the documents do NOT say. Lead with the fact you have. Partial facts are still an answer.
4. CALCULATION: If the documents don't state something directly, find data to calculate it from — including POSSIBLY HELPFUL TABLES, which you may extract from, compute on, and reason over. Exact figures to 2 decimal places, never rounded off or approximate. If calculation isn't possible either, reason it out.
5. CITATION: Cite only the [Source: doc_name, page N] or [Source: doc_name, Section: heading] tag exactly as it appears in the context below — never invent a citation format.
6. IMAGE CAPTIONS: Some context is images (charts, tables, diagrams) captioned by GPT-4o Vision. Treat it with the same confidence as document text and cite it the same way.
7. ATTACHED VISUALS: Items listed in DOCUMENT CONTEXT as "ATTACHED VISUAL k of N" are actual images attached to this message, deliberately carrying no written description — the image itself is the data. Read the pixels: transcribe the exact axis values, labels, legend entries and row/column figures, and compute from them when the question has a relation to them. A visual with no description is NOT missing information; "the description doesn't say" is never a valid reason to refuse when the visual is attached.
8. DIVERSITY: The answer to the query/question can be across multiple chunks/data points, images or tables. You haev to handle them efficiecntly going through the entire content before replying. You might also need to perfrom cross-data calculations, make sure you handle them efficiently.
9. CONVERSATION MEMORY: EARLIER CONVERSATION SUMMARY, RELEVANT EARLIER MESSAGES, and CONVERSATION HISTORY are conversation context, not document sources — use them to understand what was already discussed, resolve follow-ups ("it", "that project", "the one you mentioned"), and avoid repeating yourself, but never cite them with a [Source: ...] tag; only DOCUMENT CONTEXT and POSSIBLY HELPFUL TABLES are citable sources.
"""
    # Document context from retrieved chunks — text and image chunks are
    # formatted differently so the LLM knows which parts came from GPT-4o
    # Vision's understanding of an image versus the document's actual text.
    # table_html (original <table> markup, see pipeline/chunker.py) is
    # pulled OUT into its own POSSIBLY HELPFUL TABLES section below, one
    # per chunk that has one, rather than inlined into DOCUMENT CONTEXT —
    # keeps prose and structured table data visually/semantically separate
    # for the LLM, each still tagged with the same [Source: ...] header so
    # a table can be traced back to the passage it came from.
    # 2026-08-23: which visual chunks get their ORIGINAL crop attached to
    # the LLM call — computed here so the "ATTACHED VISUAL k of N"
    # numbering written below matches, position for position, what
    # api/routes/chat.py attaches (it calls this same helper on the same
    # chunk list). id()-keyed because these are the identical dict objects
    # in both places, and chunk payloads have no guaranteed stable key
    # that survives every retrieval path.
    attached = select_attachable_crops(context_chunks) if attach_crops else []
    attached_pos = {id(c): i + 1 for i, c in enumerate(attached)}
    n_attached = len(attached)

    if context_chunks:
        context_parts = []
        table_parts = []
        for chunk in context_chunks:
            # page_number/section_heading — whichever is available — form
            # the citation tag the system prompt instructs the LLM to
            # reuse verbatim inline, so an answer can point back to a
            # specific page/section instead of just a doc name.
            location_bits = []
            if chunk.get("page_number") is not None:
                location_bits.append(f"page {chunk['page_number']}")
            if chunk.get("section_heading"):
                location_bits.append(f"Section: {chunk['section_heading']}")
            location = ", ".join(location_bits)
            tag = f"[Source: {chunk['doc_name']}{', ' + location if location else ''}]"

            if chunk.get("role") == "image":
                image_type = chunk.get("image_type") or "image"
                caption = chunk.get("image_caption") or chunk["chunk_text"]

                if id(chunk) in attached_pos:
                    pos = attached_pos[id(chunk)]
                    context_parts.append(
                        f"""{tag} ATTACHED VISUAL {pos} of {n_attached} — {image_type}

The image itself is attached to this message as visual {pos} of {n_attached} (document visuals come first, in this order, before any image the user uploaded). No written description is provided for it, on purpose — READ THE IMAGE.

- Read every value off the pixels yourself: axis values and units, bar/point/segment values, legend entries, row and column headers, every cell figure, and any text printed inside the image, and represent it in a way that makes sense.
- Where the question needs a total, subtotal, difference, share or ratio, CALCULATE it from the values you read. Exact figures to 2 decimal places — never "approximately", never a rounded-off ballpark.
- Prefer these values over any prose elsewhere in this prompt if the two disagree: the image is the primary source, the surrounding text is not.
- If one value is genuinely illegible, name which one and give a single-line best guess, clearly marked as a guess — do not let it silently drop out of a calculation.
- Cite anything taken from it as {tag}."""
                    )
                    continue
                if image_type == "table":
                    context_parts.append(
                        f"""{tag} Your motive is to enchance the quality of the data, USE THE FOLLOWING TO ANSWER THE QUESTION — it contains:

                        1. All the Key facts, numbers, and data points from text and tables (axis values, labeled figures, transcribed text) exactly how its represented.
                        2. Main topics and concepts discussed
                        3. Questions this content could answer
                        4. Visual content analysis (charts, diagrams, patterns in images)
                        5. Alternative search terms users might use
                        6. You need to provide a llm understandable structure of the following data keeping in mind it can also perfrom mathametical operations.


                        Extract the exact values it reports — do not just note that this data exists.

                        DATA: {caption}"""
                    )
                elif image_type in ("chart", "diagram"):
                    # Structured visuals: data lives in axes/values (chart)
                    # or in components/flow (diagram) — gridded/bounded,
                    # same shape of task as reading a table, just visual.
                    context_parts.append(
                        f"""{tag} Your motive is to enchance the quality of the data, USE THE FOLLOWING TO ANSWER THE QUESTION — it contains:

                        1. All the Key facts, numbers, and data points from this image (axis values, labeled figures, transcribed text) exactly how its represented.
                        2. Main topics and concepts this image depicts
                        3. Questions this image's content could answer
                        4. Visual content analysis (chart type, diagram structure, patterns shown)
                        5. Alternative search terms users might use
                        6. You need to provide a llm understandable structure of the following data keeping in mind it can also perfrom mathametical operations.

                        Extract the exact values it reports — do not just note that this data exists.

                        DATA: {caption}"""
                    )
                else:
                    # 2026-08-22: photo/screenshot/infographic/other —
                    # NOT gridded or bounded like a chart/table. An object
                    # photo can carry markings, stamps, engravings, printed
                    # or handwritten statements, codes, serial numbers —
                    # scattered across the image, not laid out in rows/
                    # columns. The "searchable description" checklist above
                    # doesn't fit this: the goal here isn't summarizing that
                    # text exists, it's decrypting exactly what it says, so
                    # nothing exact (a code, a number, a name) gets lost to
                    # paraphrasing before it ever reaches the answering LLM.
                    context_parts.append(
                        f"""{tag} Your motive is to enchance the quality of the data, USE THE FOLLOWING TO ANSWER THE QUESTION — it decrypts everything visible in this image:

                        1. What the object/subject physically is
                        2. VERBATIM transcription of every visible text, marking, stamp, engraving, label, code, or number — exact characters, not paraphrased or summarized, since an exact value may be the actual answer needed
                        3. Where each marking sits on the object, when position could matter (e.g. "stamped on the underside", "engraved along the left edge")
                        4. Notable condition, features, or anomalies visible
                        5. Any legible data — measurements, dates, serial numbers, quantities

                        Do not vaguely summarize ("text is visible on the object") — transcribe exactly what is legible. If something is illegible or partially obscured, say so explicitly, and add a single-line best-guess of what that part could plausibly have been (context, shape, partial characters) — one line maximum, clearly marked as a guess, not stated as fact.

                        DECRYPTED CONTENT: {caption}"""
                    )
                # 2026-08-22: a table whose table_html came out unreliable
                # (image_processor.py::table_html_is_reliable()) gets
                # captioned via Vision AND has its crop attached directly
                # to the final LLM call (api/routes/chat.py collects
                # crop_image_path from these same chunks) — sublabel it
                # "(Visual)" here so the LLM knows this entry is backed by
                # an actual attached image, not just the caption text.
                # Reached only when this visual's crop was NOT attached
                # (attach_crops=False, or the crop file is missing on
                # disk — a document ingested before crops were persisted).
                # The "see attached image" sublabel would be a lie in that
                # case, so it's dropped: this is caption-only fallback.
                if image_type == "table":
                    table_parts.append(f"{tag} (Vision reading of a table image — table_html extraction was unreliable; treat exact values here as approximate)\n{caption}")
            else:
                context_parts.append(f"{tag}\n{chunk['chunk_text']}")
                if chunk.get("table_html"):
                    table_parts.append(f"{tag} (Structured)\n{_html_table_to_markdown(chunk['table_html'])}")
        context_str = "\n\n".join(context_parts)
        tables_str = "\n\n".join(table_parts) if table_parts else "No tables retrieved for this query."
    else:
        context_str = "No relevant document context found."
        tables_str = "No tables retrieved for this query."

    # Conversation history from memory
    history_messages = memory.chat_memory.messages
    history_str = ""
    if history_messages:
        history_parts = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                history_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        history_str = "\n".join(history_parts)

    # Earlier conversation summary — everything older than the recency
    # window, compressed by pipeline/chat_storage.py::update_rolling_summary().
    summary_str = rolling_summary.strip() if rolling_summary and rolling_summary.strip() else "No earlier conversation to summarise yet."

    # Relevant earlier messages — semantic recall (select_relevant_history()
    # above), specific facts from anywhere in the thread that the recency
    # window and the (necessarily lossy) summary might not carry.
    recall_str = "None retrieved for this question."
    if relevant_history:
        recall_parts = []
        for h in relevant_history:
            role_label = "Human" if h.get("role") == "human" else "Assistant"
            recall_parts.append(f"{role_label} (earlier in this conversation): {h.get('content', '')}")
        recall_str = "\n".join(recall_parts)

    # Build full prompt
    prompt = f"""{system}

CURRENT QUESTION:
{query}

DOCUMENT CONTEXT:
{context_str}

POSSIBLY HELPFUL TABLES:
{tables_str}

EARLIER CONVERSATION SUMMARY (older turns, compressed):
{summary_str}

RELEVANT EARLIER MESSAGES (semantically matched to the current question, may be from earlier than the summary above):
{recall_str}

CONVERSATION HISTORY (most recent turns, verbatim):
{history_str if history_str else "No previous conversation."}


ANSWER:"""

    return prompt
