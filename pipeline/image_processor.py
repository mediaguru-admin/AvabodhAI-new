"""
pipeline/image_processor.py
───────────────────────────
Core image understanding pipeline for AvabodhAI.

All three sources (PDF upload, web scraping, chat image upload)
call this single module. It:

  1. Filters noise (too small, UI icons, duplicates)
  2. Resizes to max 1024px for GPT-4o Vision (saves tokens)
  3. Calls GPT-4o Vision with structured Pydantic output
  4. Returns ImageCaptionOutput ready for embedding + storage

GPT-4o Vision SEES the actual pixels — not just alt text.
It reads chart values, understands diagrams, transcribes
embedded text in infographics, and describes photos.
"""

import base64
import hashlib
import io
import re
from typing import Optional

import httpx
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────────────

# Images smaller than this are noise (icons, tracking pixels, spacers)
# Sourced from settings so it's configurable via .env — falls back to sane defaults.
_MIN_SIZE_BYTES  = getattr(settings, "IMAGE_MIN_SIZE_BYTES", 5_000)   # 5KB
_MIN_DIMENSION   = 50          # 50px width or height

# Resize large images before Vision API call — saves tokens
# GPT-4o detail="auto": tiles ≤2048px; detail="low": fixed 512px
_MAX_DIMENSION   = getattr(settings, "IMAGE_MAX_DIMENSION", 1024)     # resize to this if larger

# CSS class / ID patterns that indicate UI chrome to skip
_NOISE_PATTERNS  = re.compile(
    r"logo|icon|avatar|emoji|badge|favicon|spinner|loader"
    r"|social|share|arrow|bullet|checkmark|star|rating|thumbnail",
    re.IGNORECASE,
)

VISION_MODEL = getattr(settings, "VISION_MODEL", "gpt-4o")   # sees actual pixels, reads charts, diagrams, text


# ── Pydantic output schema ───────────────────────────────────────────────────

class ImageCaptionOutput(BaseModel):
    """
    Structured output from GPT-4o Vision.
    Every field is populated from the model's understanding of the image —
    not from alt text or filename.
    """
    caption:          str        = Field(description="Detailed description of what the image shows")
    # 2026-08-22: bounded to exactly these three (image-side) — chunk type
    # overall is bounded to {text, table, chart, diagram}, "text" being
    # role="text" chunks, which never carry image_type at all. photo/
    # screenshot/infographic/other were dropped: this app's real documents
    # (financial/real-estate PDFs) don't produce genuine object photos —
    # every image encountered is one of these three. Any value Vision
    # returns outside this set falls back to "diagram" (the most general
    # of the three, not a distinct "unknown" bucket like "other" was).
    image_type:       str        = Field(description="chart|table|diagram")
    contains_chart:   bool       = Field(default=False)
    contains_table:   bool       = Field(default=False)
    contains_text:    bool       = Field(default=False, description="True if image has embedded text (infographic etc)")
    key_elements:     list[str]  = Field(default_factory=list, description="Key visual elements: axis labels, column headers, objects")
    suggested_alt_text: str      = Field(default="", description="Short accessible description max 125 chars")
    confidence:       float      = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("image_type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        # "NA" is distinct from {chart, table, diagram} — reserved for
        # _fallback_caption_from_alt_text(), where no actual Vision
        # classification happened at all (bare alt text only), so forcing
        # it into one of the three real categories would misrepresent it
        # as a real classification. Vision's own output must always be one
        # of the three; only this fallback path is allowed to say "NA".
        allowed = {"chart", "table", "diagram", "NA"}
        v = str(v).strip()
        v_lower = v.lower()
        if v_lower in {"chart", "table", "diagram"}:
            return v_lower
        if v == "NA":
            return v
        return "diagram"

    @field_validator("caption", "suggested_alt_text", mode="before")
    @classmethod
    def clean_str(cls, v) -> str:
        return str(v).strip() if v else ""


# ── Noise filter ─────────────────────────────────────────────────────────────

def should_skip_image(
    image_bytes: bytes,
    width:       int,
    height:      int,
    src:         str = "",
    css_classes: str = "",
    image_format: str = "",
) -> tuple[bool, str]:
    """
    Returns (True, reason) if image should be skipped, (False, "") otherwise.
    """
    size = len(image_bytes)

    if size < _MIN_SIZE_BYTES:
        return True, f"too small ({size} bytes < {_MIN_SIZE_BYTES})"

    if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
        return True, f"too small dimensions ({width}x{height})"

    if src.startswith("data:image"):
        return True, "inline data URI — UI element"

    if image_format.lower() == "svg" and size < 10_000:
        return True, "small SVG — likely UI icon"

    combined = f"{src} {css_classes}".lower()
    if _NOISE_PATTERNS.search(combined):
        return True, f"noise pattern detected in src/class: {combined[:60]}"

    return False, ""


def compute_image_hash(image_bytes: bytes) -> str:
    """SHA-256 of raw image bytes — for dedup."""
    return hashlib.sha256(image_bytes).hexdigest()


def table_html_is_reliable(
    table_html: Optional[str],
    page_text_chars: Optional[int] = None,
) -> bool:
    """
    Heuristic: is this table's structure-inference HTML (unstructured's
    infer_table_structure=True, el.metadata.text_as_html) trustworthy
    enough that captioning the table visually via GPT-4o Vision would be
    redundant AND lossy — Vision necessarily compresses a table into a
    prose caption ("key elements"), which can misread or drop values a
    clean structured extraction already has exactly right.

    Used by pipeline/ingest.py to skip the Vision-crop-caption call for
    tables that already extracted cleanly, and only fall back to it for
    tables where the structured extraction came out missing or clearly
    broken — the same garbled-table failure mode found live in testing
    (a multi-column PDF layout bleeding unrelated text into a table).

    Deliberately simple/conservative — false negatives (sending a
    perfectly fine table to Vision anyway) just cost an extra API call;
    false positives (skipping Vision for a genuinely broken table) lose
    real accuracy, so the bar for "reliable" is kept low on purpose.

    page_text_chars: how many characters of real, extractable text the
    SOURCE PAGE carries (pipeline/extractor.py::_pdf_page_text_chars()).
    None means unknown (non-PDF, or the measurement failed) and is treated
    as "assume fine", preserving the original behaviour.

    2026-09-23 — the text-layer check below is the fix for a confirmed
    live wrong answer. A page whose table is drawn as vector outlines (or
    scanned) has no text layer, so unstructured's hi_res path OCRs it, and
    OCR of a dense numeric table drops rows while still emitting
    structurally valid <table> markup. Both structural checks above pass,
    the table looks clean, and the missing rows are invisible — chat
    reported 21.79 million sq ft for a figure that is actually 34.01.
    Detecting the garbling in the OUTPUT was measured and rejected: those
    tables score perfectly uniform on cell-count consistency and carry a
    lower junk-character ratio than several genuinely good tables in the
    same file. Whether the source page had a text layer at all is the
    signal that actually separates them, and it's deterministic.
    """
    if not table_html or len(table_html.strip()) < 40:
        return False
    if table_html.count("<tr") < 2:   # need at least a header + one data row
        return False
    if (
        page_text_chars is not None
        and page_text_chars < settings.TABLE_TEXT_LAYER_MIN_CHARS
    ):
        # OCR-derived: structure may look fine while cell values/rows are
        # silently wrong. Vision reads the rendered crop directly instead.
        return False
    return True


# ── Image preparation ────────────────────────────────────────────────────────

def prepare_image_for_vision(image_bytes: bytes) -> tuple[str, str, int, int, str]:
    """
    Resize if needed, convert to JPEG for consistent encoding.
    Returns (base64_string, format, width, height, media_type).
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB — Vision API doesn't like RGBA/P mode JPEGs
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size

    # Resize if larger than max dimension (preserves aspect ratio)
    if max(width, height) > _MAX_DIMENSION:
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
        width, height = img.size

    # Encode as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    jpeg_bytes = buffer.read()

    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return b64, "jpeg", width, height, "image/jpeg"


# ── GPT-4o Vision call ───────────────────────────────────────────────────────

# 2026-08-22: simplified — this used to also carry the detailed per-type
# reading guidance ("if chart: describe axes...", "Read chart axes and
# approximate values...", etc). That guidance now lives in
# pipeline/memory.py's system prompt (rule 7a) instead, governing how the
# FINAL answer LLM extracts/uses a caption's data — captioning itself only
# needs to produce a comprehensive, structured description; how precisely
# that description gets mined for an answer is a downstream concern, not
# an ingestion-time one.
_VISION_PROMPT = """Generate a comprehensive, searchable description abiding the instructions below only if its a image. Otherwise if it feels like a table, please recosturct it in a llm understandable structure keeping in mind it can also be used to perfrom mathametical operations.


Return ONLY valid JSON with these exact fields:
{
  "caption": "<detailed description of what the image shows>",
  "image_type": "<one of: chart, table, diagram>",
  "contains_chart": <true or false>,
  "contains_table": <true or false>,
  "contains_text": <true if image contains readable embedded text>,
  "key_elements": ["<element1>", "<element2>", ...],
  "suggested_alt_text": "<max 125 chars accessible description>",
  "confidence": <0.0 to 1.0 — your confidence in this analysis>
}"""


def _fallback_caption_from_alt_text(alt_text: str) -> Optional[ImageCaptionOutput]:
    """
    Build a low-confidence ImageCaptionOutput from HTML alt text when
    GPT-4o Vision is unavailable or the call fails. Better than dropping
    the image entirely — it still gets embedded and is findable by search,
    just with less rich signal than an actual Vision caption.
    """
    alt_text = (alt_text or "").strip()
    if not alt_text:
        return None
    try:
        return ImageCaptionOutput(
            caption=alt_text,
            image_type="NA",   # no real Vision classification happened here — bare alt text only, see ImageCaptionOutput.validate_type
            contains_chart=False,
            contains_table=False,
            contains_text=False,
            key_elements=[],
            suggested_alt_text=alt_text[:125],
            confidence=0.3,   # low confidence — this is alt text, not actual Vision analysis
        )
    except Exception:
        return None


def caption_image_with_vision(
    image_bytes:    bytes,
    surrounding_text: str = "",
    alt_text:       str = "",
    force:          bool = False,
    model:          Optional[str] = None,
) -> Optional[ImageCaptionOutput]:
    """
    Call GPT-4o Vision and return structured ImageCaptionOutput.
    If the Vision call fails and alt_text is available, falls back to a
    low-confidence caption built from alt_text instead of dropping the
    image entirely. Returns None only if there's nothing usable at all.

    force: bypass the settings.VISION_ENABLED kill switch. Only the chat
    path where a USER attaches an image to their question passes this —
    that isn't document ingestion, and the toggle exists to control
    ingestion spend, not to disable a live feature mid-conversation.
    Every ingestion call site (PDF image regions, fallback table crops,
    standalone image uploads, scraped web images) leaves it False.

    model: override the captioning model for this one call. Defaults to
    settings.VISION_MODEL (gpt-4o), which is what every INGESTION call
    site uses. The chat path — a user attaching an image to their own
    question — passes settings.MAP_MODEL instead, so live chat traffic
    stays on the cheap model while document ingestion keeps the stronger
    reader.
    """
    if not force and not settings.VISION_ENABLED:
        # Same return path a genuine API failure takes, on purpose: the
        # alt-text fallback and the "no alt text -> None -> caller skips
        # this image" behavior are already the tested, understood shape of
        # "no caption available". Reusing it means the switch introduces no
        # new downstream branch to reason about.
        logger.info(
            "VISION_ENABLED=false — skipping GPT-4o Vision captioning (%s)",
            "using alt text instead" if (alt_text or "").strip() else "no alt text, image will be dropped",
        )
        return _fallback_caption_from_alt_text(alt_text)

    try:
        b64, fmt, width, height, media_type = prepare_image_for_vision(image_bytes)
    except Exception as e:
        logger.warning("Image preparation failed: %s", e)
        return _fallback_caption_from_alt_text(alt_text)

    # Build context hint if surrounding text provided
    context_hint = ""
    if surrounding_text and len(surrounding_text.strip()) > 10:
        context_hint = f"\n\nContext from surrounding document text:\n{surrounding_text[:500]}"

    prompt = _VISION_PROMPT + context_hint

    try:
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.LLM_REQUEST_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )
        response = client.chat.completions.create(
            model=model or VISION_MODEL,
            max_tokens=800,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":  "image_url",
                            "image_url": {
                                "url":    f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw_json = response.choices[0].message.content
        import json
        data = json.loads(raw_json)
        result = ImageCaptionOutput(**data)

        logger.info(
            "Vision captioned image: type=%s confidence=%.2f elements=%d",
            result.image_type, result.confidence, len(result.key_elements),
        )
        return result

    except Exception as e:
        logger.warning("GPT-4o Vision call failed: %s", e)
        return _fallback_caption_from_alt_text(alt_text)


# ── Download image from URL (web scraping mode) ──────────────────────────────

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AvabodhAI/1.0; +https://github.com/bapurba893/AvabodhAI) "
        "ImageFetcher/1.0"
    )
}


def download_image(url: str, timeout: int = 15) -> Optional[bytes]:
    """
    Download image bytes from a URL.
    Sends a browser-like User-Agent — some sites 403 on the default httpx UA.
    Returns None on failure (non-fatal).
    """
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers=_DOWNLOAD_HEADERS)
        if resp.status_code == 200:
            return resp.content
        logger.warning("Image download failed %s: HTTP %d", url, resp.status_code)
        return None
    except Exception as e:
        logger.warning("Image download error %s: %s", url, e)
        return None


# ── Build embedding text from caption ────────────────────────────────────────

def build_image_embedding_text(
    caption:      ImageCaptionOutput,
    doc_name:     str,
    surrounding_text: str = "",
) -> str:
    """
    Construct the composite text that gets embedded into pgvector.
    Rich signal = better semantic search results.
    The [IMAGE] prefix distinguishes image chunks from text chunks.
    """
    parts = [
        "[IMAGE]",
        f"Caption: {caption.caption}",
        f"Type: {caption.image_type}",
    ]
    if caption.key_elements:
        parts.append(f"Key elements: {', '.join(caption.key_elements)}")
    if caption.suggested_alt_text:
        parts.append(f"Alt text: {caption.suggested_alt_text}")
    if surrounding_text and len(surrounding_text.strip()) > 10:
        parts.append(f"Context: {surrounding_text[:400]}")
    parts.append(f"Source document: {doc_name}")

    return "\n".join(parts)


# ── Extract images from HTML soup (web scraping mode) ─────────────────────────

def extract_images_from_soup(
    soup,
    base_url:   str,
    doc_name:   str,
    summary_id,
    doc_hash:   str,
    source_path: str = "",
    page_text:  str = "",
    hash_exists_fn=None,
) -> list:
    """
    Extract, download, caption, and prepare images found in BeautifulSoup HTML.
    Called from scraper.py after page is fetched.
    Returns list[ImageEmbeddingInput] ready for embed_and_store_images().

    Import ImageEmbeddingInput from pipeline.embedder where needed to avoid
    circular imports — this function returns plain dicts that the caller
    converts to ImageEmbeddingInput.

    hash_exists_fn: optional callable(image_bytes_hash) -> bool. When provided
    (pass pipeline.embedder.check_image_hash_exists from the caller — kept as
    an injected callable here to avoid a circular import with embedder.py),
    images already stored in the DB are skipped BEFORE the Vision API call,
    saving cost on pages that reuse the same logo/banner/diagram repeatedly.
    """
    from urllib.parse import urljoin, urlparse
    results = []
    image_index = 0

    img_tags = soup.find_all("img")
    _cap = getattr(settings, "MAX_IMAGES_PER_DOCUMENT", 20) or 20
    if len(img_tags) > _cap:
        logger.warning(
            "'%s' has %d <img> tags - capping Vision captioning at %d "
            "(MAX_IMAGES_PER_DOCUMENT). The rest are skipped; text and tables are unaffected.",
            doc_name, len(img_tags), _cap,
        )
        img_tags = img_tags[:_cap]
    logger.info("Processing %d <img> tag(s) in '%s'", len(img_tags), doc_name)

    for img_tag in img_tags:
        try:
            src = img_tag.get("src", "").strip()
            if not src:
                continue

            # Make absolute URL
            if not src.startswith(("http://", "https://")):
                src = urljoin(base_url, src)

            # Skip data URIs
            if src.startswith("data:image"):
                continue

            alt_text   = img_tag.get("alt", "")
            css_class  = " ".join(img_tag.get("class", []))

            # Get surrounding context — parent figure/section text
            parent = img_tag.find_parent(["figure", "section", "article", "div", "p"])
            surrounding = parent.get_text(" ", strip=True)[:600] if parent else page_text[:400]

            # Download image bytes
            image_bytes = download_image(src)
            if image_bytes is None:
                continue

            # Detect format + get dimensions
            fmt = _detect_image_format_bytes(image_bytes)
            try:
                from PIL import Image as PILImage
                import io as _io
                pil_img = PILImage.open(_io.BytesIO(image_bytes))
                width, height = pil_img.size
            except Exception:
                width, height = 0, 0

            size = len(image_bytes)

            # Noise filter
            skip, reason = should_skip_image(
                image_bytes=image_bytes,
                width=width,
                height=height,
                src=src,
                css_classes=css_class,
                image_format=fmt,
            )
            if skip:
                logger.debug("Skipping web image %s: %s", src[:60], reason)
                continue

            img_hash = compute_image_hash(image_bytes)

            # Early dedup — skip the Vision API call entirely for images
            # already stored (e.g. a site header logo repeated on every page)
            if hash_exists_fn is not None:
                try:
                    if hash_exists_fn(img_hash):
                        logger.debug("Skipping already-embedded image (hash=%s)", img_hash[:12])
                        continue
                except Exception as e:
                    logger.debug("hash_exists_fn check failed, continuing anyway: %s", e)

            # Caption with GPT-4o Vision (falls back to alt_text if the call fails)
            caption = caption_image_with_vision(
                image_bytes=image_bytes,
                surrounding_text=surrounding,
                alt_text=alt_text,
            )
            if caption is None:
                continue

            if caption.image_type == "NA" and caption.confidence < 0.5:
                continue

            embedding_text = build_image_embedding_text(
                caption=caption,
                doc_name=doc_name,
                surrounding_text=surrounding,
            )

            results.append({
                "summary_id":        summary_id,
                "doc_hash":          doc_hash,
                "doc_name":          doc_name,
                "source_path":       source_path,
                "page_number":       None,
                "image_bytes_hash":  img_hash,
                "image_url":         src,
                "image_format":      fmt,
                "image_width":       width,
                "image_height":      height,
                "image_size_bytes":  size,
                "image_caption":     caption.caption,
                "image_type":        caption.image_type,
                "image_alt_text":    alt_text or caption.suggested_alt_text,
                "image_context":     surrounding[:500],
                "contains_chart":    caption.contains_chart,
                "contains_table":    caption.contains_table,
                "contains_text_img": caption.contains_text,
                "key_elements":      caption.key_elements,
                "vision_model_used": "gpt-4o",
                "vision_confidence": caption.confidence,
                "embedding_text":    embedding_text,
                "chunk_index":       image_index,
                "total_chunks":      1,
            })
            image_index += 1
            logger.info(
                "Web image %d captured from %s: type=%s",
                image_index, src[:50], caption.image_type,
            )

        except Exception as e:
            logger.warning("Failed to process web image: %s", e)
            continue

    # Update total_chunks
    total = len(results)
    for i, r in enumerate(results):
        r["chunk_index"]  = i
        r["total_chunks"] = total

    logger.info("Extracted %d web images from '%s'", total, doc_name)
    return results


# ── Extract images from a PDF (upload mode) ────────────────────────────────

def extract_images_from_elements(
    all_elements: list,
    image_elements: list,
    doc_name: str,
    source_path: str = "",
    hash_exists_fn=None,
    save_crops_dir: Optional[str] = None,
) -> list[dict]:
    """
    Extract, caption, and prepare images found by unstructured's own
    hi_res layout detection (pipeline/extractor.py::_partition_pdf,
    extract_image_block_types=["Image"], extract_image_block_to_payload=True)
    — replaces the earlier PyMuPDF-based extract_images_from_pdf()
    (2026-08-21). Returns the same plain-dict shape that function did.

    Why the switch: hi_res layout detection runs on the RENDERED page, so
    it also catches vector-drawn charts/figures that are just shapes+text
    on the page, not an embedded raster object at all. The PyMuPDF
    approach (page.get_images(), an XObject-table lookup) silently missed
    those entirely — confirmed against a real chart-heavy document, not
    a hypothetical gap.

    all_elements: the FULL originally-partitioned element list (before
    extractor.py splits out the image elements) — used to build each
    image's surrounding-text context from same-page text elements, the
    same role page.get_text() played in the old PyMuPDF version.
    image_elements: just the "Image"-category elements to process.

    hash_exists_fn: optional callable(image_bytes_hash) -> bool, same
    pattern as extract_images_from_soup() — pass a Qdrant-backed dedup
    check from the caller (pipeline/ingest.py) to skip the Vision API call
    for images already indexed (e.g. a repeated letterhead/watermark).

    save_crops_dir: when set, each element's crop is ALSO written to disk
    (under save_crops_dir/{content hash}.{fmt}) and the resulting path is
    added to the result dict as "crop_image_path".

    2026-08-23 — BOTH of pipeline/ingest.py's call sites now pass this
    (charts/diagrams as well as fallback table crops); it used to be the
    table-fallback site only, on the theory that a plain image's caption
    was its primary signal with no higher-fidelity alternative to fall
    back to. That theory was wrong in practice: the caption is written
    HERE, at ingestion, before any question exists — so a chart's axis
    values, legend entries and labels get compressed into prose and are
    simply gone by the time an answer needs an exact figure. The crop IS
    the higher-fidelity alternative, and api/routes/chat.py re-attaches it
    to the final LLM call so the answering model reads the pixels itself.
    The caption still earns its keep as the EMBEDDED text (retrieval has
    to match on something searchable); it just stops being the only thing
    the answering model ever sees.
    """
    import base64
    import io as _io

    results: list[dict] = []
    image_index = 0
    # Scoped across the WHOLE document — some PDFs (e.g. a repeated
    # infographic/template design) show the exact same image on every
    # page. Confirmed necessary against a real document during testing:
    # without this, the same ~12 images came through 4x (once per page),
    # badly skewing retrieval toward whichever caption repeated most.
    seen_hashes = set()

    # Per-page text context, built from the non-image elements — same
    # role page.get_text() played for the old PyMuPDF version.
    page_text_parts: dict[int, list[str]] = {}
    for el in all_elements:
        pg = getattr(el.metadata, "page_number", None)
        if pg is None:
            continue
        page_text_parts.setdefault(pg, []).append(str(el))
    page_text = {pg: "\n".join(parts) for pg, parts in page_text_parts.items()}

    _cap = getattr(settings, "MAX_IMAGES_PER_DOCUMENT", 20) or 20
    if len(image_elements) > _cap:
        logger.warning(
            "'%s' has %d image regions - capping Vision captioning at %d "
            "(MAX_IMAGES_PER_DOCUMENT). The rest are skipped; text and tables are unaffected.",
            doc_name, len(image_elements), _cap,
        )
        image_elements = image_elements[:_cap]
    logger.info("Processing %d image region(s) found in '%s'", len(image_elements), doc_name)

    for el in image_elements:
        try:
            b64 = getattr(el.metadata, "image_base64", None)
            if not b64:
                continue
            image_bytes = base64.b64decode(b64)
            mime = getattr(el.metadata, "image_mime_type", None) or "image/jpeg"
            image_format = mime.split("/")[-1] if "/" in mime else "jpeg"
            page_num = getattr(el.metadata, "page_number", None)
            context_text = page_text.get(page_num, "")

            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(_io.BytesIO(image_bytes))
                width, height = pil_img.size
            except Exception:
                width, height = 0, 0
            size = len(image_bytes)

            skip, reason = should_skip_image(
                image_bytes=image_bytes, width=width, height=height,
                image_format=image_format,
            )
            if skip:
                logger.debug("Skipping image region page=%s: %s", page_num, reason)
                continue

            img_hash = compute_image_hash(image_bytes)
            if img_hash in seen_hashes:
                logger.debug(
                    "Skipping duplicate image within this document (page %s, hash=%s)",
                    page_num, img_hash[:12],
                )
                continue
            seen_hashes.add(img_hash)

            if hash_exists_fn is not None:
                try:
                    if hash_exists_fn(img_hash):
                        logger.debug(
                            "Skipping already-embedded image on page %s (hash=%s)",
                            page_num, img_hash[:12],
                        )
                        continue
                except Exception as e:
                    logger.debug("Dedup check failed, continuing anyway: %s", e)

            caption = caption_image_with_vision(
                image_bytes=image_bytes, surrounding_text=context_text[:800],
            )
            if caption is None:
                logger.warning("Vision captioning failed for image region on page %s", page_num)
                continue
            if caption.image_type == "NA" and caption.confidence < 0.5:
                logger.debug("Skipping low-confidence image on page %s", page_num)
                continue

            embedding_text = build_image_embedding_text(
                caption=caption, doc_name=doc_name, surrounding_text=context_text[:400],
            )

            crop_image_path = None
            if save_crops_dir:
                try:
                    import os
                    os.makedirs(save_crops_dir, exist_ok=True)
                    crop_image_path = os.path.join(save_crops_dir, f"{img_hash}.{image_format}")
                    if not os.path.exists(crop_image_path):   # content-hash filename — already-saved crop needs no rewrite
                        with open(crop_image_path, "wb") as f:
                            f.write(image_bytes)
                except Exception as e:
                    logger.warning("Failed to save table crop to disk (non-fatal, caption still used): %s", e)
                    crop_image_path = None

            results.append({
                "doc_name": doc_name,
                "source_path": source_path,
                "page_number": page_num,
                "image_bytes_hash": img_hash,
                "image_url": None,   # PDF-embedded, no URL
                "image_format": image_format,
                "image_width": width,
                "image_height": height,
                "image_size_bytes": size,
                "image_caption": caption.caption,
                "image_type": caption.image_type,
                "image_alt_text": caption.suggested_alt_text,
                "image_context": context_text[:500],
                "contains_chart": caption.contains_chart,
                "contains_table": caption.contains_table,
                "contains_text_img": caption.contains_text,
                "key_elements": caption.key_elements,
                "vision_model_used": "gpt-4o",
                "vision_confidence": caption.confidence,
                "embedding_text": embedding_text,
                "crop_image_path": crop_image_path,
                "chunk_index": image_index,
                "total_chunks": 1,
            })
            image_index += 1
            logger.info(
                "Image %d captured: page=%s type=%s confidence=%.2f",
                image_index, page_num, caption.image_type, caption.confidence,
            )
        except Exception as e:
            logger.warning("Failed to process image region on page %s: %s", getattr(el.metadata, "page_number", "?"), e)
            continue

    total = len(results)
    for i, r in enumerate(results):
        r["chunk_index"] = i
        r["total_chunks"] = total

    logger.info("Extracted %d image(s) from '%s'", total, doc_name)
    return results


def _detect_image_format_bytes(image_bytes: bytes) -> str:
    if image_bytes[:4] == b"\x89PNG":
        return "png"
    if image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    if image_bytes[:4] in (b"GIF8", b"GIF9"):
        return "gif"
    if image_bytes[:4] == b"RIFF" and len(image_bytes) > 12 and image_bytes[8:12] == b"WEBP":
        return "webp"
    try:
        import io as _io
        from PIL import Image as PILImage
        img = PILImage.open(_io.BytesIO(image_bytes))
        return (img.format or "jpeg").lower()
    except Exception:
        return "jpeg"