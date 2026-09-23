"""
pipeline/chunker.py
--------------------
Replaces pipeline/splitter.py's role in the main pipeline (splitter.py's
SemanticChunker required OpenAI embedding calls just to find chunk
boundaries; unstructured's chunk_by_title groups elements by document
structure — headings, list items, page breaks — instead, no extra
embedding cost).

clean_extracted_text() and is_meaningful_chunk() are reused verbatim from
splitter.py (unchanged logic, still correct for post-chunking noise
filtering regardless of which chunker produced the raw text).
"""

import re
from dataclasses import dataclass
from typing import Optional

from unstructured.chunking.title import chunk_by_title

from config.settings import get_settings
from pipeline.extractor import ExtractedDocument
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    text: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    table_html: Optional[str] = None   # original <table> markup, when this chunk contains a Table element


def clean_extracted_text(text: str) -> str:
    """Fixes hyphenation, removes noise, normalises whitespace. Unchanged from pipeline/splitter.py."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\W_]{1,5}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'([a-z]{2,})\n([a-z]{2,})', r'\1 \2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    return text.strip()


def is_meaningful_chunk(text: str) -> bool:
    """Returns False for chunks that are noise — too short, only numbers, or not enough real words. Unchanged from pipeline/splitter.py."""
    text = text.strip()
    if len(text) < 200:
        return False
    if re.match(r'^[\d\s\.\,\:\;\-\_\(\)]+$', text):
        return False
    words = [w for w in text.split() if len(w) > 2]
    if len(words) < 5:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False
    return True


def chunk_document(extracted: ExtractedDocument) -> list[Chunk]:
    """
    unstructured.chunking.title.chunk_by_title groups elements under their
    nearest heading, splitting further only when a group exceeds
    max_characters — this is what keeps chunks from cutting mid-sentence
    under normal conditions and what keeps a section's content together
    with its heading. multipage_sections=True (the library default, made
    explicit here) keeps a section together even when it spans a page
    break, rather than force-splitting at every page boundary.
    combine_text_under_n_chars merges runs of small elements (list items,
    short paragraphs) instead of emitting a chunk per element.

    Table handling (verified empirically against real documents, not just
    the docs — behavior differs by case): a Table element's
    metadata.text_as_html survives chunking whether the table ends up
    merged into a larger chunk with surrounding text, or isolated on its
    own — either way it's carried into Chunk.table_html below, and such a
    chunk is exempted from is_meaningful_chunk()'s prose-oriented noise
    filter (a short, number-heavy table is legitimate content, not noise).
    """
    try:
        title_chunks = chunk_by_title(
            extracted.elements,
            max_characters=settings.MAX_CHUNK_SIZE,
            new_after_n_chars=int(settings.MAX_CHUNK_SIZE * 0.9),
            combine_text_under_n_chars=settings.MIN_CHUNK_SIZE,
            multipage_sections=True,
            # overlap only applies when text-splitting an oversized
            # section (i.e. a genuine hard split) — normal chunks are
            # untouched since overlap_all defaults False. Small on
            # purpose: just enough to avoid losing a sentence's context
            # right at a forced split boundary, not meaningful duplication.
            overlap=settings.CHUNK_OVERLAP,
            # 2026-08-22 (requires unstructured>=0.23.0 — confirmed absent
            # in 0.18.32, present from 0.23.0 through the installed
            # 0.27.1): isolate_table=True means a Table element can only
            # ever START its own pre-chunk and is NEVER merged with
            # adjacent non-table elements. Direct fix for a real,
            # confirmed-live bug: a multi-column PDF layout caused a
            # table's chunk to get contaminated with unrelated prose from
            # a neighboring section (the "Critical Transport &
            # Infrastructure Corridors" table coming out full of Afrobeats/
            # Nollywood text). With isolation on, that contamination is
            # structurally impossible — a table chunk can only ever
            # contain that one table.
            isolate_table=True,
            # 2026-09-23: flipped to True — the large-table-got-fragmented
            # failure this was deliberately left disabled for has now been
            # confirmed live: a 17-row residential-projects table (Godrej
            # Properties Bengaluru area annexure) exceeded new_after_n_chars
            # (2700 = 0.9 * MAX_CHUNK_SIZE) and was hard-split mid-row,
            # silently dropping the last 5 rows from every chunk that
            # existed. Any "sum this table" question answered from whatever
            # subset of fragments got retrieved was then wrong by
            # construction, independent of retrieval/reranking correctness.
            # skip_table_chunking=True keeps a table as a single chunk
            # regardless of size — trades a larger chunk (and more prompt
            # tokens when it's retrieved) for the table never losing rows.
            # Table chunks are already exempt from is_meaningful_chunk()'s
            # size-oriented filter, so a bigger table chunk won't be
            # discarded either.
            skip_table_chunking=True,
            repeat_table_headers=True
        )
    except Exception as e:
        logger.error("chunk_by_title failed for '%s': %s", extracted.doc_name, e)
        raise

    cleaned: list[Chunk] = []
    discarded = 0

    for el in title_chunks:
        text = clean_extracted_text(str(el))
        table_html = getattr(el.metadata, "text_as_html", None)

        # A table chunk is legitimate content even if short/number-heavy —
        # only apply the prose noise filter to non-table chunks.
        if not table_html and not is_meaningful_chunk(text):
            discarded += 1
            continue
        if not text:
            discarded += 1
            continue

        page_number = getattr(el.metadata, "page_number", None)
        # Chunks spanning multiple original elements carry orig_elements —
        # use the first element's heading-ish text as a best-effort section
        # label; None is fine, this is a nice-to-have, not load-bearing.
        section_heading = None
        orig_elements = getattr(el.metadata, "orig_elements", None)
        if orig_elements:
            first_type = type(orig_elements[0]).__name__
            if first_type == "Title":
                section_heading = str(orig_elements[0])[:512]

        cleaned.append(Chunk(text=text, chunk_index=-1, total_chunks=-1,
                              page_number=page_number, section_heading=section_heading,
                              table_html=table_html))

    for i, c in enumerate(cleaned):
        c.chunk_index = i
        c.total_chunks = len(cleaned)

    logger.info(
        "Chunking complete for '%s' — %d chunks produced, %d discarded (too small)",
        extracted.doc_name, len(cleaned), discarded,
    )
    return cleaned
