"""
pipeline/extractor.py
----------------------
Content extraction for the main ingestion pipeline — replaces
pipeline/loader.py's role there (loader.py itself stays, but is now
/kb-only, see its module docstring).

Rewritten 2026-08-21 (v2): routes each file to unstructured's DEDICATED
partition_* function for its type, not the generic auto-detecting
partition(). Per unstructured's own docs
(docs.unstructured.io/open-source/core-functionality/partitioning):
dedicated functions exist specifically so type-specific parameters (table-
structure inference, OCR strategy, etc.) can be tuned per format — auto-
detection is framed there as the "unknown file type" fallback, not the
default path. Every dedicated call below was verified against the actual
installed unstructured==0.18.32 signatures (introspected directly, not
just doc text) before being wired in.

extract_file() dispatches by extension to a dedicated partitioner; if the
extension has no dedicated handler here, it falls back to the generic
partition() — exactly unstructured's own recommended fallback case.
extract_web_markdown() handles scraped pages (pipeline/scraper.py already
converts HTML -> markdown) the same way an uploaded .md file would be
partitioned, so both sources converge on one code path before chunking.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.csv import partition_csv
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.partition.md import partition_md
from unstructured.partition.auto import partition as partition_auto

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".docx", ".csv",
    ".pptx", ".html", ".htm", ".md", ".xlsx",
    ".png", ".jpg", ".jpeg", ".webp",
}

# Standalone image uploads route through pipeline/ingest.py's dedicated
# GPT-4o Vision captioning branch (IMPLEMENTATION_PLAN (2).md §4), NOT
# through this module at all — see pipeline/ingest.py::_process_standalone_image().
STANDALONE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-type dedicated partitioners. Each wraps a specific unstructured
# partition_* function with parameters chosen for this app's needs:
#   - infer_table_structure=True wherever it isn't already the type's
#     default, so Table elements carry metadata.text_as_html (original
#     table structure, not just flattened cell text) all the way through
#     chunking (verified empirically: text_as_html survives chunk_by_title
#     whether the table ends up isolated or merged with nearby text).
#   - PDF image regions ARE extracted here (extract_image_block_types,
#     extract_image_block_to_payload — see _partition_pdf below) as of
#     2026-08-21, replacing the earlier PyMuPDF-based extraction in
#     pipeline/image_processor.py. extract_file() below splits Image-
#     category elements out of the main element list into
#     ExtractedDocument.image_elements — pipeline/ingest.py feeds those
#     to image_processor.py::extract_images_from_elements() for GPT-4o
#     Vision captioning, same as before, just a different image source.
# ─────────────────────────────────────────────────────────────────────────────

def _partition_pdf(filepath: str) -> list:
    # infer_table_structure defaults to False for PDF specifically
    # (confirmed via signature introspection) — everything else here
    # already defaults True for their respective types, but PDF needs it
    # explicit or tables come through as flattened text only.
    #
    # extract_image_block_types=["Image"] + extract_image_block_to_payload=True
    # (2026-08-21): hi_res layout detection finds image REGIONS on the
    # RENDERED page — this also catches vector-drawn charts/figures that
    # are just shapes+text on the page, not an embedded raster object.
    # The previous approach (PyMuPDF's page.get_images(), extracting only
    # embedded XObject images) silently missed those entirely — confirmed
    # a real gap, not a hypothetical one, against an infographic-heavy
    # real document. extract_image_block_to_payload=True returns base64
    # bytes directly in element metadata, exactly what
    # pipeline/image_processor.py::caption_image_with_vision() already
    # needs — no disk I/O, no temp directory to manage/clean up.
    strategy = settings.UNSTRUCTURED_STRATEGY
    kwargs = dict(
        filename=filepath,
        strategy=strategy,
        infer_table_structure=True,
        languages=["eng"],
        extract_image_block_types=["Image","Table"],
        extract_image_block_to_payload=True,
    )
    try:
        return partition_pdf(**kwargs)
    except Exception as e:
        # 2026-08-21: hi_res's layout/table/OCR models are memory-hungry
        # per-page — confirmed against a real large/complex PDF, an
        # onnxruntime allocation failure (~745MB for one buffer) that
        # took the whole ingestion down. Graceful degradation beats a
        # hard failure: retry once with "fast" (pdfminer-based text
        # extraction, no layout model, no table/image detection) so the
        # document still ingests with usable text instead of nothing at
        # all. Only applies when hi_res was actually what was requested —
        # if "fast" was already the configured strategy, there's nothing
        # lighter to fall back to, so the original error still propagates.
        if strategy == "hi_res":
            logger.warning(
                "hi_res PDF partitioning failed for '%s' (%s) — retrying with strategy='fast' "
                "(loses table-structure/image-region extraction for this document, but the "
                "text still ingests instead of the whole upload failing).",
                filepath, e,
            )
            kwargs["strategy"] = "fast"
            kwargs["infer_table_structure"] = False   # table HTML needs hi_res's layout model — not available under "fast"
            kwargs.pop("extract_image_block_types", None)
            kwargs.pop("extract_image_block_to_payload", None)
            return partition_pdf(**kwargs)
        raise


def _partition_docx(filepath: str) -> list:
    return partition_docx(filename=filepath, infer_table_structure=True)


def _partition_pptx(filepath: str) -> list:
    return partition_pptx(filename=filepath, infer_table_structure=True, include_slide_notes=True)


def _partition_xlsx(filepath: str) -> list:
    # include_header=True keeps column names in the table's text/HTML —
    # without it, a sheet's numbers lose their column labels entirely.
    return partition_xlsx(filename=filepath, infer_table_structure=True, include_header=True)


def _partition_csv(filepath: str) -> list:
    return partition_csv(filename=filepath, infer_table_structure=True, include_header=True)


def _partition_html(filepath: str) -> list:
    # skip_headers_and_footers=True — for an UPLOADED .html/.htm file
    # (nav/footer chrome saved along with the page). Scraped web pages
    # take a different path (extract_web_markdown, below) — scraper.py
    # already strips nav/footer noise before this ever runs.
    return partition_html(filename=filepath, skip_headers_and_footers=True)


def _partition_text(filepath: str) -> list:
    return partition_text(filename=filepath)


def _partition_md(filepath: str) -> list:
    return partition_md(filename=filepath)


_DEDICATED_PARTITIONERS: dict[str, Callable[[str], list]] = {
    ".pdf": _partition_pdf,
    ".docx": _partition_docx,
    ".pptx": _partition_pptx,
    ".xlsx": _partition_xlsx,
    ".csv": _partition_csv,
    ".html": _partition_html,
    ".htm": _partition_html,
    ".txt": _partition_text,
    ".md": _partition_md,
}


@dataclass
class ExtractedDocument:
    elements: list[Any]          # non-image elements, fed to pipeline/chunker.py
    doc_name: str
    file_hash: str
    file_size: int
    source_path: str
    page_count: int = 0
    image_elements: list[Any] = field(default_factory=list)   # "Image"-category elements (PDF only) — fed to image_processor.py::extract_images_from_elements()
    table_elements: list[Any] = field(default_factory=list)   # "Table"-category elements (PDF only) — fed to image_processor.py::extract_images_from_elements()
    # PDF only: {1-indexed page_number: extractable text-layer character
    # count}. A page with a real text layer yields hundreds-to-thousands of
    # chars; a page whose content is drawn as vector paths or is a scanned
    # raster yields ~0, meaning ANYTHING unstructured extracted from it came
    # from OCR. Consumed by pipeline/ingest.py to decide whether a Table
    # element's text_as_html can be trusted — see _pdf_page_text_chars().
    page_text_chars: dict[int, int] = field(default_factory=dict)


def _pdf_page_text_chars(filepath: str) -> dict[int, int]:
    """
    Per-page extractable-text character count, 1-indexed to match
    unstructured's metadata.page_number.

    2026-09-23 — added after a confirmed live failure: the Godrej Q2 FY26
    annexure pages render their tables as VECTOR DRAWINGS with the cell
    text converted to outlines, so the pages carry ~77 text characters
    each (just the slide header/footer) against 300-475 drawing ops. No
    text layer means unstructured's hi_res path can only OCR them, and OCR
    of a dense 23-row numeric table silently dropped most rows while still
    emitting a structurally clean 12-column <table> — so the result LOOKED
    well-formed and passed every structural sanity check, but was missing
    the data. Chat then answered "21.79 million sq ft" for a figure whose
    real value is 34.01, with no error logged anywhere.

    The garbling itself turned out to be undetectable downstream (those
    tables score uniform=1.0 on cell-count consistency and have a LOWER
    junk-character ratio than several perfectly good tables in the same
    document — measured, not assumed). The text layer is the honest
    upstream signal instead: it says whether OCR was even involved.

    Returns {} on any failure — callers treat an absent entry as "unknown,
    assume fine", so this can never make extraction fail.
    """
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - pymupdf is a declared dependency
        logger.warning("pymupdf unavailable — skipping text-layer measurement for '%s'", filepath)
        return {}

    try:
        with pymupdf.open(filepath) as doc:
            return {i + 1: len(page.get_text().strip()) for i, page in enumerate(doc)}
    except Exception as e:
        logger.warning("Text-layer measurement failed for '%s' (non-fatal): %s", filepath, e)
        return {}


def _compute_file_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def extract_file(filepath: str) -> ExtractedDocument:
    """
    Extract structured elements from an uploaded file — routes to a
    dedicated partition_* function by extension; falls back to the
    generic auto-detecting partition() only when this module has no
    dedicated handler for the extension (matches unstructured's own
    documented recommendation for that function's use case).
    """
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    partitioner = _DEDICATED_PARTITIONERS.get(ext)
    try:
        if partitioner is not None:
            elements = partitioner(filepath)
        else:
            logger.info("No dedicated partitioner for '%s' — using generic auto-detect partition().", ext)
            elements = partition_auto(filename=filepath, strategy=settings.UNSTRUCTURED_STRATEGY)
    except Exception as e:
        logger.error("unstructured partition failed for '%s' (ext=%s): %s", filepath, ext, e)
        raise

    doc_name = os.path.basename(filepath)
    file_hash = _compute_file_hash(filepath)
    page_numbers = {
        el.metadata.page_number
        for el in elements
        if getattr(el.metadata, "page_number", None) is not None
    }

    # Split "Image"-category elements out — chunk_by_title shouldn't see
    # them mixed into text sections (an Image element's own text is empty;
    # its content is the base64 payload, not something chunk_by_title
    # should be title-grouping alongside prose). Only PDF partitioning
    # currently requests these (extract_image_block_types in _partition_pdf)
    # so this is a no-op for every other file type.
    text_elements = [el for el in elements if type(el).__name__ != "Image"]
    table_elements = [el for el in elements if type(el).__name__ == "Table"]
    image_elements = [el for el in elements if type(el).__name__ == "Image"]

    logger.info(
        "Extracted '%s' -> %d text element(s) + %d image element(s) + %d table element(s) across %d page(s) | ext=%s | hash=%s",
        doc_name, len(text_elements), len(image_elements), len(table_elements), len(page_numbers) or 1, ext, file_hash[:12],
    )

    return ExtractedDocument(
        elements=text_elements,
        doc_name=doc_name,
        file_hash=file_hash,
        file_size=os.path.getsize(filepath),
        source_path=os.path.abspath(filepath),
        page_count=len(page_numbers) or 1,
        image_elements=image_elements,
        table_elements=table_elements,
        page_text_chars=_pdf_page_text_chars(filepath) if ext == ".pdf" else {},
    )


def extract_web_markdown(markdown_text: str, doc_name: str, source_url: str) -> ExtractedDocument:
    """
    Same extraction path for scraped web pages — pipeline/scraper.py
    already converts HTML to markdown; this routes that markdown through
    partition_md, the SAME dedicated function an uploaded .md file uses,
    so chunking downstream doesn't need to know which source produced it.
    """
    try:
        elements = partition_md(text=markdown_text)
    except Exception as e:
        logger.error("unstructured partition_md failed for web doc '%s': %s", doc_name, e)
        raise

    file_hash = hashlib.sha256(markdown_text.encode("utf-8")).hexdigest()
    logger.info("Extracted web doc '%s' -> %d element(s) | hash=%s", doc_name, len(elements), file_hash[:12])

    return ExtractedDocument(
        elements=elements,
        doc_name=doc_name,
        file_hash=file_hash,
        file_size=len(markdown_text.encode("utf-8")),
        source_path=source_url,
        page_count=1,
    )
