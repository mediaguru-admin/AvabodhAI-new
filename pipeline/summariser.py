"""
pipeline/summariser.py
----------------------
Steps 3 + 4 - Map + Reduce summarisation using OpenAI.

NEW: Metadata extraction added at both levels —
- Chunk-level metadata extracted in the SAME Map call (no extra cost)
- Document-level metadata extracted in ONE extra call after Reduce

Map step:    ThreadPoolExecutor — all chunks processed simultaneously,
             each call now returns summary + structured chunk metadata
Reduce step: Single LLM call combines all chunk summaries into one,
             followed by one structured-output call for document metadata
"""

import contextvars
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from db.models import ChunkMetadataOutput, DocumentMetadataOutput
from config.settings import get_settings
from pipeline.llm_log import llm_log_kwargs
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"]   = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = settings.LANGCHAIN_PROJECT

# ── Existing plain-text summary prompt (kept for reduce input) ────────────────
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a professional document analyst.
Read the following section and write a concise summary.
Focus on: key facts, important decisions, main arguments, critical data.
Keep it under 150 words. Be specific.

SECTION:
{text}

CONCISE SUMMARY:""",
)

# ── NEW — Structured prompt for chunk-level metadata + summary together ───────
CHUNK_METADATA_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional document analyst extracting structured metadata.
Read the following section and return:
- A concise summary (under 150 words, specific, factual)
- The section/heading this content likely belongs to (e.g. "Introduction", "Methodology"), or null if unclear
- The chunk type: paragraph, table, list, heading, or code
- A 2-3 word topic label for this specific section
- Named entities mentioned (people, organizations, locations) — empty list if none
- Whether this section contains numerical data, statistics, or tabular data
- Your confidence (0.0-1.0) in this extraction

SECTION:
{text}"""
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior document analyst writing an executive summary for a
document management system. The system that displays this summary does
NOT render markdown or preserve line breaks reliably — assume your
output may be shown as one continuous block of text with no formatting
applied at all. It must still read as clear, well-punctuated prose in
that worst case, not a broken list.

Write exactly four short paragraphs, in this order, as plain sentences.
Never use a dash, bullet, asterisk, or hash symbol anywhere — every
point must be a complete grammatical sentence, connected with ordinary
words like "First," "Second," "Additionally," and "Finally," never with
a line break or a symbol standing in for punctuation. Each paragraph
label below must be followed immediately by its content on the same
line, never left standing alone.

Overview: 3-4 sentences covering the big picture of the entire document.

Key Findings: 5-7 of the most important findings, each as one complete
sentence, joined naturally in a flowing paragraph (for example: "First,
... Second, ... Additionally, ... Finally, ...").

Main Topics: one sentence naming the 4-6 major topics discussed,
separated by commas, written as a normal sentence (for example: "This
document covers X, Y, and Z.").

Conclusion: one sentence capturing the overall takeaway.

Rules:
- Separate the four paragraphs with a single blank line, nothing else.
- Never put "Overview", "Key Findings", "Main Topics", or "Conclusion"
  on their own line — always followed directly by a colon and the
  sentence content on that same line.
- Be specific, not vague.
- Maximum 400 words total.
SECTION SUMMARIES:
{text}

EXECUTIVE SUMMARY:""",
)

# ── NEW — Document-level metadata extraction prompt ────────────────────────────
DOCUMENT_METADATA_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior document analyst extracting document-level metadata.
Based on the document summary and section summaries below, extract:
- title: the real document title (not a filename)
- author: if mentioned anywhere, else null
- document_type: one of resume, research_paper, contract, report, invoice, manual, article, other
- domain: one of legal, technical, financial, academic, medical, general
- detected_language: the language of the document
- key_entities: organizations, people, locations mentioned (max 10)
- mentioned_dates: any dates referenced (max 10)
- target_audience: one of technical, general, executive
- sentiment: one of positive, negative, neutral, critical
- confidentiality_level: one of public, internal, confidential — infer from content tone

DOCUMENT SUMMARY:
{summary}

SECTION SUMMARIES:
{sections}"""
)


def _build_llm(max_tokens: int, purpose: str) -> ChatOpenAI:
    if settings.use_ollama:
        if ChatOllama is not None:
            return ChatOllama(
                model=settings.OLLAMA_CHAT_MODEL,
                base_url=settings.ollama_url,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=max_tokens,
                **llm_log_kwargs(purpose),
            )
        return ChatOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_url.rstrip('/')}/v1",
            model=settings.OLLAMA_CHAT_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=max_tokens,
            **llm_log_kwargs(purpose),
        )
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MAP_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=max_tokens,
        # See pipeline/chat.py::_build_llm — these settings existed but were
        # wired to nothing, so every OpenAI call was unbounded. Matters most
        # here: the Map step fans out one call per chunk in a thread pool, so
        # one hung call held a worker for as long as the network let it.
        timeout=settings.LLM_REQUEST_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
        **llm_log_kwargs(purpose),
    )


def _summarise_chunk(args: tuple) -> tuple:
    """
    Summarise a single chunk — designed to run in a thread.
    Returns (index, summary_text) tuple.
    Kept separate from metadata extraction — used only as a fallback
    if structured extraction fails for a chunk.
    """
    index, text, llm = args
    try:
        chain = MAP_PROMPT | llm | StrOutputParser()
        summary = chain.invoke({"text": text[:3000]})
        logger.info("Chunk %d summarised", index + 1)
        return index, summary.strip()
    except Exception as e:
        logger.warning("Chunk %d failed: %s — using truncated text", index + 1, e)
        return index, text[:200]


def _extract_chunk_metadata(args: tuple) -> tuple:
    """
    NEW — Extract summary + structured metadata in ONE LLM call per chunk.
    Uses with_structured_output so Pydantic validates the shape directly.
    Runs in a thread alongside other chunks — same parallelism as before.

    Returns (index, ChunkMetadataOutput | None)
    """
    index, text, structured_llm = args
    try:
        chain = CHUNK_METADATA_PROMPT | structured_llm
        result: ChunkMetadataOutput = chain.invoke({"text": text[:3000]})
        logger.info("Chunk %d metadata extracted (topic=%s, type=%s)",
                   index + 1, result.topic, result.chunk_type)
        return index, result
    except Exception as e:
        logger.warning("Chunk %d metadata extraction failed: %s", index + 1, e)
        return index, None


@traceable(run_type="chain", name="MapReduce Summarisation - OpenAI Parallel")
def summarise_document(chunks: list[Document], doc_name: str = "document") -> dict:
    """
    Parallel Map-Reduce summarisation + metadata extraction.

    Map step:    All chunks sent to OpenAI simultaneously.
                 Each call now returns BOTH summary text AND structured
                 chunk metadata (section_heading, chunk_type, topic, entities)
                 in the SAME call — no extra API cost vs the old version.

    Reduce step: Single LLM call combines all chunk summaries into the
                 final structured summary, followed by ONE additional
                 structured-output call that extracts document-level
                 metadata (title, author, document_type, domain, etc).

    Returns dict with summary_text + document metadata + per-chunk metadata
    (chunk_metadata list, same order as input chunks, None for failed ones).
    """
    if not chunks:
        raise ValueError(f"No chunks provided for '{doc_name}'")

    logger.info(
        "Starting PARALLEL summarisation + metadata extraction for '%s' - %d chunks | model=%s",
        doc_name, len(chunks), settings.MAP_MODEL,
    )

    start = time.time()
    # One model instance per purpose so each call is tagged correctly in
    # the LLM execution log (pipeline/llm_log.py) — construction is cheap.
    fallback_llm = _build_llm(settings.MAP_MAX_TOKENS, "doc_chunk_summary")
    # Structured output LLM — forces Pydantic-validated ChunkMetadataOutput shape
    structured_llm = _build_llm(settings.MAP_MAX_TOKENS, "doc_chunk_metadata").with_structured_output(ChunkMetadataOutput)

    # ── Step 3: MAP — parallel processing with metadata ────────────────────
    chunk_args = [
        (i, chunk.page_content.strip(), structured_llm)
        for i, chunk in enumerate(chunks)
        if chunk.page_content.strip()
    ]

    chunk_summaries = [""] * len(chunk_args)
    chunk_metadata: list = [None] * len(chunk_args)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            # copy_context: worker threads don't inherit contextvars, and the
            # LLM log needs this document's tenant/entity context.
            executor.submit(contextvars.copy_context().run, _extract_chunk_metadata, args): args[0]
            for args in chunk_args
        }
        completed = 0
        for future in as_completed(futures):
            index, metadata_result = future.result()
            if metadata_result is not None:
                chunk_summaries[index] = metadata_result.summary
                chunk_metadata[index] = metadata_result
            else:
                # Fallback to plain summarisation if structured extraction failed
                _, text, _ = chunk_args[index]
                fallback_idx, fallback_summary = _summarise_chunk(
                    (index, text, fallback_llm)
                )
                chunk_summaries[fallback_idx] = fallback_summary
            completed += 1
            logger.info("Map progress: %d/%d chunks done", completed, len(chunk_args))

    # Remove empty summaries (keep metadata aligned by filtering both together)
    paired = [(s, m) for s, m in zip(chunk_summaries, chunk_metadata) if s]
    chunk_summaries = [s for s, _ in paired]
    chunk_metadata  = [m for _, m in paired]

    if not chunk_summaries:
        raise RuntimeError(f"All chunks failed for '{doc_name}'")

    map_time = round(time.time() - start, 2)
    logger.info(
        "Map step complete in %.2fs — %d chunk summaries + metadata ready",
        map_time, len(chunk_summaries),
    )

    # ── Step 4: REDUCE — single call ─────────────────────────────────────
    reduce_llm = _build_llm(settings.REDUCE_MAX_TOKENS, "doc_summary")
    combined = "\n\n".join(chunk_summaries)

    if len(combined) > 12000:
        logger.info("Combined text too long (%d chars) - truncating", len(combined))
        combined = combined[:12000]

    reduce_chain = COMBINE_PROMPT | reduce_llm | StrOutputParser()

    # The Map step above already paid for and produced real data (one
    # billed LLM call per chunk) — a single flaky Reduce call must never
    # throw that away. Retry a couple of times first (most Reduce
    # failures are transient: rate limits, timeouts), and if it's still
    # failing, fall back to the chunk summaries directly rather than
    # raising and losing the Map results entirely (the old behavior:
    # raise here -> caller catches it -> summary_text stored as "",
    # silently discarding every chunk-level call that already succeeded).
    REDUCE_MAX_ATTEMPTS = 3
    final_summary = None
    reduce_fallback_used = False
    last_reduce_error: Exception | None = None

    for attempt in range(1, REDUCE_MAX_ATTEMPTS + 1):
        try:
            final_summary = reduce_chain.invoke({"text": combined}).strip()

            # Deterministic safety net, not just a prompt instruction — collapse
            # ANY run of whitespace (newlines, double spaces, tabs) into exactly
            # one regular space. This is what actually guarantees a visible gap
            # between "...State House." and "Key Findings:" — a plain space
            # character survives essentially any frontend's text handling,
            # where a raw newline is what was silently vanishing before (see
            # the "KEY FINDINGS-President Tinubu" collision with zero space,
            # confirmed in testing against Clariona's actual UI). Relying on
            # the LLM to reliably include spacing on every single generation
            # isn't a guarantee; this line is.
            final_summary = re.sub(r"\s+", " ", final_summary).strip()
            break
        except Exception as e:
            last_reduce_error = e
            logger.warning(
                "Reduce step attempt %d/%d failed for '%s': %s",
                attempt, REDUCE_MAX_ATTEMPTS, doc_name, e,
            )
            if attempt < REDUCE_MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))  # 1s, then 2s

    if final_summary is None:
        logger.error(
            "Reduce step failed after %d attempts for '%s' (%s) — falling back to "
            "joined chunk summaries instead of discarding the Map results",
            REDUCE_MAX_ATTEMPTS, doc_name, last_reduce_error,
        )
        reduce_fallback_used = True
        final_summary = re.sub(r"\s+", " ", " ".join(chunk_summaries)).strip()[:4000]

    # ── NEW — Step 5: Document-level metadata extraction (ONE extra call) ──
    document_metadata = None
    try:
        metadata_llm = _build_llm(settings.MAP_MAX_TOKENS, "doc_metadata").with_structured_output(DocumentMetadataOutput)
        metadata_chain = DOCUMENT_METADATA_PROMPT | metadata_llm
        document_metadata = metadata_chain.invoke({
            "summary":  final_summary,
            "sections": combined[:8000],   # cap to control token cost
        })
        logger.info(
            "Document metadata extracted: type=%s domain=%s title=%s",
            document_metadata.document_type, document_metadata.domain,
            document_metadata.title[:50] if document_metadata.title else "(none)",
        )
    except Exception as e:
        logger.warning("Document metadata extraction failed (non-fatal): %s", e)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "Summarisation + metadata complete for '%s' | chunks=%d | map=%.2fs | total=%.2fs",
        doc_name, len(chunks), map_time, elapsed,
    )

    return {
        "summary_text":         final_summary,
        "map_model":            settings.MAP_MODEL,
        "reduce_model":         settings.REDUCE_MODEL,
        "chunk_count":          len(chunks),
        "elapsed_sec":          elapsed,
        "chunk_metadata":       chunk_metadata,     # list aligned with final chunk order
        "document_metadata":    document_metadata,  # DocumentMetadataOutput | None
        "reduce_fallback_used": reduce_fallback_used,  # True if Reduce failed after retries and summary_text is the raw joined chunk summaries, not a real Reduce pass
    }
