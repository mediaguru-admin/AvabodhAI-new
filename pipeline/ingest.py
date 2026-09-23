"""
pipeline/ingest.py
-------------------
Background ingestion job — replaces pipeline/orchestrator.py. Runs via
FastAPI BackgroundTasks (in-process, no Celery/Redis — Vijay's docs'
choice, see the plan doc), by cli.py, and on reprocess.

Sequence: extract (unstructured) -> chunk (chunk_by_title) -> embed
(dense + sparse) -> index into Qdrant -> summarise (non-fatal) -> images
(non-fatal) -> mark READY. Any exception before chunk-indexing completes
marks the document FAILED with the error in status_detail — chunk
indexing is the retrieval-critical step; summary/image failures don't
fail the whole job (architecture doc: "PostgreSQL manages documents").

Idempotent: delete_document_points() runs first, so re-running this for
the same document_id (a retry, POST /documents/{id}/reprocess) overwrites
cleanly rather than duplicating or leaving stale chunks behind.
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document as LCDocument

from pipeline import extractor, chunker, embedder, vector_store, storage, summariser, object_store, webhook
from pipeline.image_processor import (
    extract_images_from_elements, extract_images_from_soup,
    caption_image_with_vision, build_image_embedding_text, compute_image_hash,
    table_html_is_reliable,
)
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _run_stage(stage_name: str, fn, *args, **kwargs):
    """
    Runs one fatal pipeline stage, and if it raises, re-raises tagged with
    which stage produced it and the real exception's type — not just its
    bare message. Without this, status_detail on a FAILED document ends up
    as whatever str(e) happens to be for the underlying library, which for
    something like a raw socket timeout is just "timed out" with zero
    indication of what was even being attempted or what kind of error it
    was (network? library bug? bad input?) — useless for diagnosing a
    specific failure from the UI alone, without cross-referencing live
    server logs at the exact timestamp.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"{stage_name} failed: {type(e).__name__}: {e}") from e


def _index_text_chunks(
    chunks: list[chunker.Chunk],
    document_id: str,
    tenant_id: str,
    org_unit_id: str,
    doc_hash: str,
    doc_name: str,
    is_ground_truth: bool,
    source_path: Optional[str],
    metadata: dict,
    effective_from: Optional[str] = None,
    effective_to: Optional[str] = None,
) -> int:
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    dense_vectors = embedder.embed_dense_batch(texts)
    sparse_vectors = embedder.embed_sparse_batch(texts)

    points = []
    for c, dense, (sp_idx, sp_val) in zip(chunks, dense_vectors, sparse_vectors):
        points.append(vector_store.ChunkPoint(
            document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
            role="text", chunk_index=c.chunk_index, chunk_text=c.text,
            dense_vector=dense, sparse_indices=sp_idx, sparse_values=sp_val,
            doc_hash=doc_hash, doc_name=doc_name, total_chunks=c.total_chunks,
            chunk_size=len(c.text), page_number=c.page_number,
            section_heading=c.section_heading, is_ground_truth=is_ground_truth,
            embedding_model=settings.EMBEDDING_MODEL, source_path=source_path,
            metadata=metadata, effective_from=effective_from, effective_to=effective_to,
            table_html=c.table_html,
        ))
    return vector_store.upsert_chunks(points)


def _index_image_chunks(
    images: list[dict],
    document_id: str,
    tenant_id: str,
    org_unit_id: str,
    doc_hash: str,
    doc_name: str,
    is_ground_truth: bool,
    metadata: dict,
    effective_from: Optional[str] = None,
    effective_to: Optional[str] = None,
) -> int:
    if not images:
        return 0
    texts = [img["embedding_text"] for img in images]
    dense_vectors = embedder.embed_dense_batch(texts)
    sparse_vectors = embedder.embed_sparse_batch(texts)

    points = []
    for img, dense, (sp_idx, sp_val) in zip(images, dense_vectors, sparse_vectors):
        points.append(vector_store.ChunkPoint(
            document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
            role="image", chunk_index=img["chunk_index"], chunk_text=img["embedding_text"],
            dense_vector=dense, sparse_indices=sp_idx, sparse_values=sp_val,
            doc_hash=doc_hash, doc_name=doc_name, total_chunks=img["total_chunks"],
            chunk_size=len(img["embedding_text"]), page_number=img.get("page_number"),
            is_ground_truth=is_ground_truth, embedding_model=settings.EMBEDDING_MODEL,
            source_path=img.get("source_path"), metadata=metadata,
            effective_from=effective_from, effective_to=effective_to,
            image_fields={
                "image_url": img.get("image_url"),
                "image_bytes_hash": img.get("image_bytes_hash"),
                "image_format": img.get("image_format"),
                "image_width": img.get("image_width"),
                "image_height": img.get("image_height"),
                "image_caption": img.get("image_caption"),
                "image_type": img.get("image_type"),
                "image_alt_text": img.get("image_alt_text"),
                "contains_chart": img.get("contains_chart"),
                "contains_table": img.get("contains_table"),
                "vision_confidence": img.get("vision_confidence"),
                "crop_image_path": img.get("crop_image_path"),
            },
        ))
    return vector_store.upsert_chunks(points)


def _clear_document_index(document_id: str, tenant_id: str) -> None:
    """
    Clears both this document's chunks AND its document-summary point
    (see _index_document_summary()) before re-ingesting. Without clearing
    the summary point too, a reprocess that fails before reaching
    summarization would leave a stale summary vector pointing at a
    document whose chunks were just wiped — shortlist_documents() could
    then shortlist a document with nothing actually retrievable inside it.
    A fresh summary point gets written again at the end of a successful
    run either way (deterministic ID, so it just overwrites).
    """
    vector_store.delete_document_points(tenant_id=tenant_id, document_id=str(document_id))
    vector_store.delete_document_summary(tenant_id=tenant_id, document_id=str(document_id))


def _index_document_summary(document_id: str, tenant_id: str, org_unit_id: str, doc_name: str, summary_text: str) -> None:
    """
    Embeds this document's summary and upserts it into
    vector_store.QDRANT_DOC_SUMMARY_COLLECTION — backs
    pipeline/retriever.py::shortlist_documents(), the first stage of
    retrieval that narrows to relevant documents before chunk-level
    search runs inside them. Best-effort: a failure here just means this
    document won't be shortlisted first (shortlist_documents() falls back
    to full-corpus search when nothing's shortlisted), never something
    that should fail an otherwise-successful ingestion.
    """
    if not summary_text.strip():
        return
    try:
        dense_vector = embedder.embed_dense_query(summary_text)
        sparse_indices, sparse_values = embedder.embed_sparse_query(summary_text)
        vector_store.upsert_document_summary(
            document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
            doc_name=doc_name, summary_text=summary_text,
            dense_vector=dense_vector, sparse_indices=sparse_indices, sparse_values=sparse_values,
        )
    except Exception as e:
        logger.warning(
            "Document-summary indexing failed for document %s (non-fatal, won't be shortlisted first): %s",
            document_id, e,
        )


def _summarise_and_apply(chunks: list[chunker.Chunk], document_id: str, tenant_id: str, org_unit_id: str, doc_name: str, page_count: int) -> None:
    """Non-fatal — chunk indexing (already done by the time this runs) is what matters for retrieval."""
    try:
        lc_docs = [LCDocument(page_content=c.text, metadata={"page": c.page_number}) for c in chunks]
        result = summariser.summarise_document(lc_docs, doc_name=doc_name)
        if result.get("reduce_fallback_used"):
            logger.warning(
                "Document %s: Reduce step failed after retries, summary is the "
                "raw joined chunk summaries, not a real Reduce pass",
                document_id,
            )
        storage.apply_summary(
            document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
            summary_text=result["summary_text"],
            key_topics=[],  # chunk-level topics aren't rolled up to a doc-level list here
            chunk_count=len(chunks), page_count=page_count,
            model_used=result["reduce_model"],
            document_metadata=result.get("document_metadata"),
        )
        _index_document_summary(
            document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
            doc_name=doc_name, summary_text=result["summary_text"],
        )
    except Exception as e:
        logger.warning("Summarisation failed for document %s (non-fatal): %s", document_id, e)
        storage.apply_summary(
            document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
            summary_text="", key_topics=[], chunk_count=len(chunks), page_count=page_count,
            model_used="", document_metadata=None,
        )


def _detect_image_format(image_bytes: bytes) -> str:
    if image_bytes[:4] == b"\x89PNG":
        return "png"
    if image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    if image_bytes[:4] == b"RIFF" and len(image_bytes) > 12 and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "jpeg"


def _process_standalone_image(
    document_id: str, tenant_id: str, org_unit_id: str, file_path: str, doc_name: str, doc_hash: str,
    is_ground_truth: bool, metadata: dict, effective_from: Optional[str], effective_to: Optional[str],
) -> None:
    """
    Standalone image upload (.png/.jpg/.jpeg/.webp) — IMPLEMENTATION_PLAN
    (2).md §4: GPT-4o Vision caption -> caption becomes the document
    summary -> document_type='image' -> ONE role='image' Qdrant chunk.
    Deliberately does NOT go through extractor.py/chunker.py (unstructured
    OCR on a bare image is the wrong tool here — Vision captioning is
    what the plan calls for, same as PDF/web images already get).
    """
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    caption = caption_image_with_vision(image_bytes=image_bytes)
    if caption is None:
        # A bare image file has no text of any kind — the Vision caption IS
        # the whole document. So unlike a PDF (which still ingests its text
        # and tables with Vision off), there is nothing to fall back to
        # here and the upload genuinely cannot proceed. Name the toggle
        # explicitly when that's the cause, so this doesn't read as an
        # OpenAI outage.
        if not settings.VISION_ENABLED:
            raise RuntimeError(
                "Cannot ingest a standalone image while VISION_ENABLED=false — "
                "the Vision caption is the only content a bare image file has. "
                "Set VISION_ENABLED=true to upload images as documents."
            )
        raise RuntimeError("Vision captioning failed for this image (and no alt text was available as a fallback)")

    embedding_text = build_image_embedding_text(caption=caption, doc_name=doc_name)

    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
    except Exception:
        width, height = 0, 0

    image_dict = {
        "doc_name": doc_name, "source_path": file_path, "page_number": None,
        "image_bytes_hash": compute_image_hash(image_bytes),
        "image_url": None, "image_format": _detect_image_format(image_bytes),
        "image_width": width, "image_height": height, "image_size_bytes": len(image_bytes),
        "image_caption": caption.caption, "image_type": caption.image_type,
        "image_alt_text": caption.suggested_alt_text,
        "contains_chart": caption.contains_chart, "contains_table": caption.contains_table,
        "vision_confidence": caption.confidence,
        "embedding_text": embedding_text, "chunk_index": 0, "total_chunks": 1,
    }
    indexed = _index_image_chunks(
        [image_dict], str(document_id), tenant_id, org_unit_id, doc_hash, doc_name, is_ground_truth, metadata,
        effective_from=effective_from, effective_to=effective_to,
    )
    if indexed:
        storage.set_image_count(document_id, tenant_id, org_unit_id, indexed)

    storage.apply_summary(
        document_id=document_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
        summary_text=caption.caption, key_topics=caption.key_elements,
        chunk_count=1, page_count=1, model_used=settings.VISION_MODEL,
        document_metadata=None, document_type_override="image",
    )


def process_document(
    document_id: str,
    tenant_id: str,
    org_unit_id: str,
    file_path: str,
    doc_name: str,
    doc_hash: str,
    is_ground_truth: bool = False,
    metadata: Optional[dict] = None,
    effective_from: Optional[str] = None,
    effective_to: Optional[str] = None,
    delete_local_when_done: bool = False,
) -> None:
    """
    Uploaded-file ingestion (PDF/DOCX/PPTX/XLSX/TXT/CSV/MD/images).

    delete_local_when_done: remove file_path once this job ends. Defaults to
    FALSE so the destructive behaviour is always something a caller opted
    into for a file it knows is disposable - never something this function
    infers. It previously inferred it from object_store.is_enabled(), i.e.
    "is S3 configured", which is not the same question as "does S3 have
    THIS file": reprocessing a document ingested before S3 existed passed
    its real local original here and the finally block deleted it. Only the
    caller knows whether the path it handed over is a throwaway copy.
    effective_from/effective_to: ISO 8601 strings (or None) — the SAME
    values already stored on the Document row in Postgres, duplicated
    onto every Qdrant chunk payload so `as_of` filtering (build_filter())
    works directly against Qdrant without a Postgres lookup, per
    architecture doc §11/§12/§27 Rule 7 ("effective dates belong in
    Qdrant"). Caller (api/routes/documents.py) is responsible for the
    datetime -> ISO string conversion.
    """
    metadata = metadata or {}
    storage.set_status(document_id, tenant_id, org_unit_id, "PROCESSING")
    webhook.notify_status_change(document_id, tenant_id, org_unit_id, "PROCESSING")

    try:
        _run_stage(
            "Clearing previous index", _clear_document_index,
            tenant_id=tenant_id, document_id=str(document_id),
        )

        if Path(file_path).suffix.lower() in extractor.STANDALONE_IMAGE_EXTENSIONS:
            _run_stage(
                "Standalone image processing", _process_standalone_image,
                document_id, tenant_id, org_unit_id, file_path, doc_name, doc_hash,
                is_ground_truth, metadata, effective_from, effective_to,
            )
            storage.set_status(document_id, tenant_id, org_unit_id, "READY")
            webhook.notify_status_change(document_id, tenant_id, org_unit_id, "READY")
            logger.info("Ingestion complete for document %s ('%s') — standalone image", document_id, doc_name)
            return

        extracted = _run_stage("Text extraction", extractor.extract_file, file_path)
        chunks = _run_stage("Chunking", chunker.chunk_document, extracted)
        # 2026-08-22: this used to raise as soon as `chunks` (TEXT chunks
        # only) came back empty — BEFORE the image/table-processing block
        # further down ever got a chance to run. An all-image PDF (scanned
        # brochure, infographic deck — a real, valid document type) has
        # zero text elements by design, so this failed every single time
        # even though the image pipeline would have found real content.
        # Confirmed live: "No meaningful content extracted" on a document
        # that was "mostly images" per the user's own upload. Only a
        # genuine empty/unparseable document (no text AND no
        # images/tables at all) should actually fail here.
        if not chunks and not extracted.image_elements and not extracted.table_elements:
            raise RuntimeError("No meaningful content extracted from this document")

        if chunks:
            _run_stage(
                "Chunk embedding/indexing", _index_text_chunks,
                chunks, str(document_id), tenant_id, org_unit_id, doc_hash, doc_name,
                is_ground_truth, extracted.source_path, metadata,
                effective_from=effective_from, effective_to=effective_to,
            )
            _summarise_and_apply(chunks, document_id, tenant_id, org_unit_id, doc_name, extracted.page_count)
        else:
            logger.info(
                "Document %s ('%s') has no text elements — image/table-only document, "
                "skipping text indexing/summarisation, proceeding to image processing.",
                document_id, doc_name,
            )

        # Images embedded WITHIN a document — non-fatal, only for PDFs.
        # Not to be confused with the standalone-image-upload branch above.
        # extracted.image_elements comes from the SAME partition_pdf() call
        # that already ran above (extractor.py requests
        # extract_image_block_types=["Image"] for PDFs) — no second file
        # open/re-parse needed, unlike the old PyMuPDF-based approach.
        total_images_indexed = 0
        image_chunk_offset = 0   # see table-image block below — avoids chunk_index collisions between the two extract_images_from_elements() calls

        if file_path.lower().endswith(".pdf") and extracted.image_elements:
            try:
                images = extract_images_from_elements(
                    all_elements=extracted.elements, image_elements=extracted.image_elements,
                    doc_name=doc_name, source_path=extracted.source_path,
                    hash_exists_fn=lambda h: vector_store.image_hash_exists(tenant_id, org_unit_id, h),
                    # 2026-08-23: crops are now persisted for EVERY image
                    # region (charts, diagrams — not just the table
                    # fallback below). Without this, a chart's only
                    # representation at answer time was its Vision caption
                    # — prose written once here at ingestion, before the
                    # question existed — so exact axis values/labels were
                    # already gone by the time an answer needed them.
                    # api/routes/chat.py re-attaches these crops to the
                    # final LLM call so the model reads the real pixels.
                    save_crops_dir=settings.VISUAL_CROPS_DIR,
                )
                indexed = _index_image_chunks(
                    images, str(document_id), tenant_id, org_unit_id, doc_hash, doc_name, is_ground_truth, metadata,
                    effective_from=effective_from, effective_to=effective_to,
                )
                total_images_indexed += indexed
                image_chunk_offset = len(images)
            except Exception as e:
                logger.warning("Image extraction failed for document %s (non-fatal): %s", document_id, e)

        # Table regions, captioned via GPT-4o Vision — but ONLY as a
        # fallback for tables whose structure-inference HTML
        # (el.metadata.text_as_html, what becomes table_html on the text
        # chunk — see pipeline/chunker.py) is missing or looks broken.
        # 2026-08-21 (v2): sending EVERY table through Vision regardless
        # was redundant AND lossy for tables that already extracted
        # cleanly — a Vision caption necessarily compresses a table into
        # prose ("key elements"), which can misread or drop exact values
        # a clean structured extraction already has right. table_html
        # (sent verbatim, un-summarized, in the POSSIBLY HELPFUL TABLES
        # prompt section — pipeline/memory.py) is strictly higher-fidelity
        # for any table where it came out reliable
        # (image_processor.py::table_html_is_reliable()); Vision is kept
        # as the fallback specifically for the failure mode confirmed live
        # tonight — a multi-column PDF layout bleeding unrelated text into
        # a table, where text_as_html itself comes out garbled/wrong. A
        # crop is read directly by Vision, unaffected by that defect.
        # Stored as role="image" chunks; Vision classifies these as
        # image_type="table" itself, no separate role needed.
        tables_needing_vision = [
            el for el in extracted.table_elements
            if not table_html_is_reliable(
                getattr(el.metadata, "text_as_html", None),
                # None when the page is unknown/unmeasured — treated as
                # "assume fine" by table_html_is_reliable().
                extracted.page_text_chars.get(getattr(el.metadata, "page_number", None)),
            )
        ]
        if extracted.table_elements:
            logger.info(
                "Table Vision fallback: %d/%d table(s) need it (rest have reliable text_as_html)",
                len(tables_needing_vision), len(extracted.table_elements),
            )
        if file_path.lower().endswith(".pdf") and tables_needing_vision:
            try:
                table_images = extract_images_from_elements(
                    all_elements=extracted.elements, image_elements=tables_needing_vision,
                    doc_name=doc_name, source_path=extracted.source_path,
                    hash_exists_fn=lambda h: vector_store.image_hash_exists(tenant_id, org_unit_id, h),
                    # 2026-08-22: persist the crop so it can be re-attached
                    # as an actual image to the final LLM call at answer
                    # time (api/routes/chat.py) — these are specifically
                    # the tables whose table_html came out unreliable, so
                    # a one-time Vision caption made here isn't the only
                    # signal available at answer time.
                    save_crops_dir=settings.VISUAL_CROPS_DIR,
                )
                # extract_images_from_elements() renumbers chunk_index from
                # 0 for whatever list it's given — called separately here
                # from the image_elements call above, so both would
                # otherwise start at chunk_index=0. Qdrant point IDs are
                # deterministic on (document_id, role, chunk_index)
                # (pipeline/vector_store.py::chunk_point_id()), so without
                # this offset the first table crop would silently
                # overwrite the first regular image's point instead of
                # creating its own.
                for img in table_images:
                    img["chunk_index"] += image_chunk_offset
                    img["total_chunks"] = image_chunk_offset + len(table_images)
                indexed = _index_image_chunks(
                    table_images, str(document_id), tenant_id, org_unit_id, doc_hash, doc_name, is_ground_truth, metadata,
                    effective_from=effective_from, effective_to=effective_to,
                )
                total_images_indexed += indexed
            except Exception as e:
                logger.warning("Table-image extraction failed for document %s (non-fatal): %s", document_id, e)

        if total_images_indexed:
            storage.set_image_count(document_id, tenant_id, org_unit_id, total_images_indexed)

        storage.set_status(document_id, tenant_id, org_unit_id, "READY")
        webhook.notify_status_change(document_id, tenant_id, org_unit_id, "READY")
        logger.info("Ingestion complete for document %s ('%s') — %d chunks", document_id, doc_name, len(chunks))

    except Exception as e:
        # Full traceback goes to the server log (exc_info=True) for live
        # debugging; status_detail (what the UI shows via the webhook/
        # check-status path) gets the shorter but still stage-tagged
        # message from _run_stage — real diagnostic value without dumping
        # a raw traceback into the document detail page.
        logger.error("Ingestion FAILED for document %s ('%s'): %s", document_id, doc_name, e, exc_info=True)
        error_detail = str(e)[:2000]
        storage.set_status(document_id, tenant_id, org_unit_id, "FAILED", status_detail=error_detail)
        webhook.notify_status_change(document_id, tenant_id, org_unit_id, "FAILED", status_detail=error_detail)

    finally:
        # 2026-08-23 - release the local copy the moment this job is done,
        # success or failure. Deleting when the WORK finishes rather than on
        # a timer is deliberate: measured ingest wall time for a 40-page PDF
        # is 8-11.5 minutes and a 200-page document can exceed 40, so any
        # short fixed TTL would delete a source file mid-parse. This way a
        # small document frees its file in seconds and a huge one is never
        # cut off.
        #
        # Guarded on the CALLER's flag, never on global config - see the
        # delete_local_when_done docstring. A caller sets it only for a file
        # it created as a throwaway (a scratch upload S3 already holds, or a
        # temp downloaded from S3), never for a document's only copy.
        if delete_local_when_done:
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.info("Released disposable local copy %s", file_path)
            except Exception as cleanup_err:
                # Non-fatal: the startup sweeper catches any leftovers.
                logger.warning("Could not remove local copy %s: %s", file_path, cleanup_err)


def process_web_document(
    document_id: str,
    tenant_id: str,
    org_unit_id: str,
    markdown_text: str,
    raw_html: Optional[str],
    doc_name: str,
    doc_hash: str,
    source_url: str,
    is_ground_truth: bool = False,
    metadata: Optional[dict] = None,
    effective_from: Optional[str] = None,
    effective_to: Optional[str] = None,
    caption_images: bool = True,
) -> None:
    """Web-scrape ingestion — pipeline/scraper.py already fetched + converted the page to markdown (+ kept raw_html for image extraction). effective_from/effective_to: see process_document()'s docstring."""
    metadata = metadata or {}
    storage.set_status(document_id, tenant_id, org_unit_id, "PROCESSING")
    webhook.notify_status_change(document_id, tenant_id, org_unit_id, "PROCESSING")

    try:
        _run_stage(
            "Clearing previous index", _clear_document_index,
            tenant_id=tenant_id, document_id=str(document_id),
        )

        extracted = _run_stage(
            "Text extraction", extractor.extract_web_markdown, markdown_text, doc_name, source_url,
        )
        chunks = _run_stage("Chunking", chunker.chunk_document, extracted)
        if not chunks:
            raise RuntimeError("No meaningful content extracted from this page")

        _run_stage(
            "Chunk embedding/indexing", _index_text_chunks,
            chunks, str(document_id), tenant_id, org_unit_id, doc_hash, doc_name,
            is_ground_truth, source_url, metadata,
            effective_from=effective_from, effective_to=effective_to,
        )

        _summarise_and_apply(chunks, document_id, tenant_id, org_unit_id, doc_name, extracted.page_count)

        # caption_images: per-request override of Vision captioning, on top of
        # the global settings.VISION_ENABLED switch. A crawl fans out into one
        # document per page, each captioning up to MAX_IMAGES_PER_DOCUMENT, so
        # the worst case is max_pages x that cap - this is the lever for
        # turning that off for one bulk run without editing config.
        if raw_html and caption_images:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_html, "html.parser")
                images = extract_images_from_soup(
                    soup=soup, base_url=source_url, doc_name=doc_name,
                    summary_id=document_id, doc_hash=doc_hash, source_path=source_url,
                    page_text=markdown_text[:2000],
                    hash_exists_fn=lambda h: vector_store.image_hash_exists(tenant_id, org_unit_id, h),
                )
                indexed = _index_image_chunks(
                    images, str(document_id), tenant_id, org_unit_id, doc_hash, doc_name, is_ground_truth, metadata,
                    effective_from=effective_from, effective_to=effective_to,
                )
                if indexed:
                    storage.set_image_count(document_id, tenant_id, org_unit_id, indexed)
            except Exception as e:
                logger.warning("Web image extraction failed for document %s (non-fatal): %s", document_id, e)

        elif raw_html:
            logger.info("Skipping image captioning for '%s' (caption_images=false)", doc_name)

        storage.set_status(document_id, tenant_id, org_unit_id, "READY")
        webhook.notify_status_change(document_id, tenant_id, org_unit_id, "READY")
        logger.info("Web ingestion complete for document %s ('%s') — %d chunks", document_id, doc_name, len(chunks))

    except Exception as e:
        logger.error("Web ingestion FAILED for document %s ('%s'): %s", document_id, doc_name, e, exc_info=True)
        error_detail = str(e)[:2000]
        storage.set_status(document_id, tenant_id, org_unit_id, "FAILED", status_detail=error_detail)
        webhook.notify_status_change(document_id, tenant_id, org_unit_id, "FAILED", status_detail=error_detail)
