"""
config/settings.py
------------------
Central configuration loaded from environment variables.
Never hardcode secrets — use .env file locally, secrets manager in production.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    # Ollama runs locally — no API key needed
    HUGGINGFACEHUB_API_TOKEN: str = ""   # keep empty, not used anymore

    # Fixed 2026-08-21: these previously defaulted to Ollama-only model names
    # ("llama3.2"/"nomic-embed-text") while EMBEDDING_DIMENSIONS below assumed
    # an OpenAI model — a fresh checkout with no .env silently misconfigured
    # itself. Ollama remains supported (see use_ollama below), these are just
    # the OpenAI-first defaults matching what every real deployment overrides.
    MAP_MODEL: str = "gpt-4o-mini"
    REDUCE_MODEL: str = "gpt-4o-mini"
    # 2026-08-23: the ANSWERING model for chat, split out from MAP_MODEL.
    # Both used to be MAP_MODEL, which meant the model chosen for cheap
    # per-chunk ingestion summaries was also the one reading chart/table
    # crops at answer time. That is backwards: the multimodal path (a user
    # attaching their own image) already used gpt-4o, so an image the USER
    # sent got the strong reader while a chart already inside their
    # documents got the weak one. Ingestion stays on MAP_MODEL — nothing
    # about summarisation cost changes.
    CHAT_MODEL: str = "gpt-4o-mini"
    # Answer-time sampling temperature. 0.1, not the previous hardcoded 0.3
    # — answers here are exact-figure extraction and a fixed response
    # schema, both of which tighten as temperature drops. Not 0.0: a little
    # headroom is kept deliberately.
    CHAT_TEMPERATURE: float = 0.1
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # ── Dense embedding dimensions (OpenAI text-embedding-3-small = 1536,
    # text-embedding-3-large = 3072). Used for both the OpenAI embedding
    # calls and the Qdrant `dense` vector size in vector_store.py.
    EMBEDDING_DIMENSIONS: int = 1536
    EMBEDDING_BATCH_SIZE: int = 100     # chunks per batch to OpenAI
    # If a dense-embedding batch fails, retry it this many times (backoff
    # below) before giving up. Exhausting retries stops the whole document
    # rather than skipping the batch and continuing — a skipped batch means
    # those chunks are just silently absent from the vector store with no
    # indication anything is missing, which is worse than an explicit
    # failure the user can retry.
    EMBEDDING_MAX_RETRIES: int = 2

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DOCKER_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_DIMENSIONS: int = 768

    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_PROVIDER: str = "auto"
    PYTHON_KB_INTERNAL_URL: str = "http://localhost:8000"

    MAP_MAX_TOKENS: int = 512
    REDUCE_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0
    LLM_REQUEST_TIMEOUT: int = 120
    LLM_MAX_RETRIES: int = 3
    # Chat answers are user-facing — somebody is watching a spinner — so
    # they fail fast rather than using the ingestion budget above. With
    # LLM_REQUEST_TIMEOUT=120 and LLM_MAX_RETRIES=3 a wedged endpoint could
    # burn ~8 minutes before giving up, which reads as "broken" long before
    # it reads as "slow". 45s x 1 retry caps a chat turn's LLM time at ~90s.
    # Ingestion keeps the longer budget: it runs in the background where
    # nobody is waiting, and a large document genuinely can take minutes.
    CHAT_REQUEST_TIMEOUT: int = 45
    CHAT_MAX_RETRIES: int = 1

    # ── Chunking ───────────────────────────────────────────────────────────
    # CHUNK_BREAKPOINT_THRESHOLD removed 2026-08-21 — was specific to the old
    # SemanticChunker (pipeline/splitter.py, being replaced by chunker.py's
    # unstructured chunk_by_title, which has no equivalent knob).
    MAX_CHUNK_SIZE: int = 3000
    MIN_CHUNK_SIZE: int = 100
    # Only applies when chunk_by_title has to hard-split an oversized
    # section (text-splitting) — trailing characters from the prior
    # chunk get prefixed onto the next one, so a forced split doesn't
    # lose the sentence context right at the cut. Normal (non-split)
    # chunks are untouched.
    CHUNK_OVERLAP: int = 200

    # Below this many characters of extractable text on a PDF page, any
    # table unstructured reports for that page must have come from OCR
    # (the page is vector-drawn or scanned), so its text_as_html is not
    # trusted and the table is routed to the GPT-4o Vision crop path
    # instead — see pipeline/image_processor.py::table_html_is_reliable().
    # 200 sits in a wide empirical gap: on the document that exposed this
    # bug, vector-drawn annexure pages measured 76-94 chars while every
    # text-layer page with a table measured 477+.
    TABLE_TEXT_LAYER_MIN_CHARS: int = 200

    # ── Summary ────────────────────────────────────────────────────────────
    SUMMARY_LANGUAGE: str = "English"

    # ── Chat conversation memory (pipeline/memory.py) ─────────────────────
    # 2026-09-23: replaces the plain fixed-window-only memory. Three layers,
    # combined — the standard production pattern (recency window + rolling
    # summary + semantic recall over the full history), not a knowledge
    # graph: a graph is the right tool for tracking entities/relationships
    # across sessions, not for "what did we just discuss in this thread."
    #   1. Recency window (MEMORY_WINDOW_SIZE turns, unchanged) — last N
    #      turns sent verbatim, for immediate coherence.
    #   2. Rolling summary (ChatThread.rolling_summary, db/models.py) —
    #      every turn that ages OUT of the window gets folded into a
    #      running summary instead of being dropped, so a long thread's
    #      earlier topics are never silently forgotten, without resending
    #      the whole transcript every request. Reuses the same
    #      MAP_MODEL/REDUCE_MODEL LLM plumbing as document summarisation.
    #   3. Semantic recall — the current query is embedded and matched
    #      against every past message in this thread (already indexed in
    #      Qdrant's avabodh_chat_messages collection for GET /chat/search,
    #      pipeline/chat_storage.py::_index_message_in_qdrant), so a
    #      specific fact from turn 3 can resurface on turn 40 even though
    #      it long ago fell out of both the window and the summary's level
    #      of detail.
    # How many turns (1 turn = 1 human + 1 AI message) stay verbatim in the
    # recency window before aging out into the rolling summary. Moved here
    # from a hardcoded pipeline/memory.py constant so it's tunable per
    # deployment without a code change.
    MEMORY_WINDOW_SIZE: int = 5
    MEMORY_SEMANTIC_RECALL_LIMIT: int = 4
    MEMORY_SUMMARY_MAX_TOKENS: int = 400

    # ── Qdrant (vector search) ────────────────────────────────────────────
    # Self-hosted, on-prem Qdrant instance — NOT Docker-managed by this
    # project's own docker-compose.yml (deliberate: permanent data stores run
    # on host storage the operator provisions themselves, provisioned via
    # scripts/init_qdrant.py). Must point at an instance separate from
    # clariona-core's own Postgres/vector store — never shared, even if
    # reachable on the same network.
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""   # optional — self-hosted Qdrant without auth is a valid local setup
    QDRANT_COLLECTION: str = "avabodh_chunks"
    QDRANT_CHAT_COLLECTION: str = "avabodh_chat_messages"
    # One point per document (its summary, embedded once at ingestion) —
    # backs the document-shortlist stage in pipeline/retriever.py so a
    # query first narrows to DOC_SHORTLIST_LIMIT relevant documents before
    # chunk-level search runs inside just those, instead of every query
    # competing across the whole corpus's chunks regardless of how many
    # documents exist. Keeps retrieval quality from degrading as the
    # knowledge base grows past a handful of documents.
    QDRANT_DOC_SUMMARY_COLLECTION: str = "avabodh_document_summaries"
    # If a chunk-upsert batch fails, retry it this many times (backoff
    # below) before declaring the whole upload/document failed.
    QDRANT_UPSERT_MAX_RETRIES: int = 5
    # How many documents the shortlist stage narrows to before chunk-level
    # search runs. Only used when the caller hasn't already scoped the
    # query to a specific document/set of documents.
    DOC_SHORTLIST_LIMIT: int = 5
    # MMR (Maximal Marginal Relevance) diversity weight for the final
    # chunk selection: 1.0 = pure relevance (no diversity penalty, old
    # behavior), 0.0 = pure diversity (ignores relevance).
    # 2026-09-23: started at 0.5 (equal weighting), raised to 0.85, then
    # to 1.0 — confirmed live that even a small diversity weight (0.15)
    # was enough to make chat answers give wrong values on a precision-
    # sensitive question, on the same day this was introduced. 1.0 makes
    # _mmr_select() mathematically identical to plain top-K-by-relevance
    # (the pre-MMR behavior) — diversity re-ranking is fully OFF by
    # default now. The system prompt (pipeline/memory.py) demands exact,
    # non-approximate figures; that is fundamentally in tension with
    # optimizing for diversity, which can deprioritize the one chunk with
    # the precise number in favor of a topically-different-but-less-exact
    # one. shortlist_documents() above already solves the cross-document
    # dilution problem MMR was partly meant to help with, so there's very
    # little upside left to offset that risk. Only lower this from 1.0 if
    # you specifically need broader-coverage answers, you've confirmed the
    # documents/queries in play aren't precision-sensitive, and you test
    # the change against real numeric-lookup questions before trusting it.
    MMR_LAMBDA: float = 1.0

    # ── Sparse vectors + reranking (fastembed, ONNX — no torch needed) ─────
    SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"
    # 2026-08-21: "BAAI/bge-reranker-v2-m3" (the originally-decided model)
    # is not supported by fastembed's TextCrossEncoder under any name
    # (confirmed via TextCrossEncoder.list_supported_models()) — every
    # rerank call silently fell back to un-reranked fusion order while
    # pointed at it. Tried loading it via BAAI's own FlagEmbedding library
    # instead (real torch weights) — too heavy for this dev box: a
    # Windows page-file failure on first load, a transformers version
    # conflict needing a pin, and a confirmed-live 392s stall on the first
    # real request once it was working. Reverted to fastembed's
    # TextCrossEncoder with a model it actually supports — ONNX runtime,
    # no torch, same dependency family as SPARSE_MODEL above.
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    # 2026-08-21: lowered from 50 — reranking is O(candidates) CPU cost,
    # confirmed live at 65s for 50 candidates on this box (severe memory
    # pressure, ~800MB free, causing heavy disk-swap during ONNX
    # inference). 20 cuts that roughly proportionally; raise this back up
    # once running on a machine with real headroom, since more candidates
    # pre-rerank generally means better final recall.
    SEARCH_CANDIDATES: int = 20   # candidates fetched pre-rerank — fallback only, see SEARCH_CANDIDATES_MAX
    # 2026-08-22: SEARCH_CANDIDATES above is now a FALLBACK, not the live
    # value — retriever.py queries Qdrant for how many chunks the target
    # document(s) actually have and fetches min(that count, this cap) as
    # candidates, instead of always fetching a fixed 20. Confirmed live:
    # a fixed 20 covers a 4-page document's ~15-20 chunks completely (full
    # recall by accident), but a 60-page document can have 200-300+
    # chunks — the actual best-matching chunk can rank 25th-50th in the
    # initial fusion and never reach the reranker at all, since it's
    # truncated before reranking runs. This cap bounds the worst case
    # (a huge document) so rerank cost can't blow back up to the ~65s
    # stalls a bare SEARCH_CANDIDATES=50 caused earlier tonight — pick a
    # bigger number here once running with real memory headroom.
    SEARCH_CANDIDATES_MAX: int = 80
    # Minimum fused (post-RRF) score a hit must clear to survive the
    # Qdrant query — None (default, blank in .env) means no floor, matching
    # Qdrant's own default. Tune from .env, not a code change: raise it to
    # cut low-relevance noise before rerank ever sees it (fewer candidates
    # -> also faster rerank, see SEARCH_CANDIDATES above); leave unset
    # while nothing's been tuned yet, since the "right" fused-RRF-score
    # floor depends on data actually observed for this deployment, not a
    # guessable constant.
    SEARCH_SCORE_THRESHOLD: Optional[float] = None
    # 2026-08-21: SPLADE's forward pass allocates a (batch, seq_len, vocab)
    # float32 array — at vocab~30522 that's ~15.6MB PER document in the
    # batch. A 50-chunk batch (a real large-PDF ingestion) demanded a single
    # 745MiB allocation and crashed with a numpy MemoryError. Small batches
    # keep any single allocation bounded regardless of document size.
    SPARSE_BATCH_SIZE: int = 8

    # ── Uploaded-file storage (tenant/org-scoped on disk — see Phase H #2) ──
    UPLOAD_DIR: str = "./uploaded_files"
    # 2026-08-23: on-disk crops of EVERY visual element a document yields —
    # charts, diagrams, and table regions alike (was tables-only, and only
    # the unreliable-text_as_html fallback tables at that). Reason for the
    # widening: a chart/diagram was only ever reaching the answering LLM as
    # the prose Vision caption written once at ingestion, before the
    # question was known — so exact axis values/labels were already lost by
    # the time an answer needed them, and no amount of prompt tightening
    # could recover them. Persisting the crop lets api/routes/chat.py
    # re-attach the ORIGINAL pixels to the final LLM call, so the model
    # reads the values itself at answer time.
    VISUAL_CROPS_DIR: str = "./visual_crops"
    # Hard cap on how many crops get attached to one answering call —
    # each attachment is a real image at detail="high" (multi-tile, so a
    # real token cost per image), and retrieval can legitimately return
    # several visual chunks for one question. Bounded here rather than
    # left to top_k so cost/latency stay predictable.
    MAX_ATTACHED_CROPS: int = 6

    # ── Content extraction (unstructured) ───────────────────────────────────
    # "hi_res" runs the layout-detection model + OCR fallback (accurate,
    # slower); "fast" skips both (quick, native-text-layer PDFs only).
    UNSTRUCTURED_STRATEGY: str = "hi_res"

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "Avabodh"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres123"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    APP_DB_USER: str = "avabodh_app"
    APP_DB_PASSWORD: str = "CHANGE-ME-app-role-password"

    # FTS_LANGUAGE / HYBRID_RRF_K removed 2026-08-21 — the Postgres
    # full-text-search trigger/column they configured is gone (chunk
    # storage + hybrid search moved to Qdrant, see pipeline/retriever.py).
    # SUPPORTED_EXTENSIONS removed too — pipeline/extractor.py now owns
    # that list (it covers far more than the old 4 formats).

    DOCUMENTS_DIR: str = "./documents"

    VISION_MODEL: str = "gpt-4o"
    # Master switch for INGESTION-TIME GPT-4o Vision transcription of image
    # regions (charts/diagrams) and of the fallback table crops. Default True
    # keeps today's exact behavior. Set false to ingest documents with zero
    # Vision spend — text, tables and their table_html still index normally.
    #
    # What false actually costs you: a Vision caption is the ONLY text an
    # image chunk has to embed (pipeline/image_processor.py::
    # build_image_embedding_text), so with no caption a PDF image region has
    # nothing searchable and is dropped rather than indexed — those charts
    # and diagrams simply won't exist in Qdrant, and no crop of them can be
    # attached at answer time either. Web-scraped images with real HTML alt
    # text still index off that alt text (low confidence), same as they
    # already do whenever a Vision call fails. Tables whose text_as_html
    # came out unreliable lose their crop fallback and are represented only
    # by that unreliable HTML.
    #
    # Deliberately does NOT gate the chat path where a USER attaches an
    # image to their question (api/routes/chat.py::_retrieve_with_image) —
    # that isn't document ingestion, and silently disabling it would break
    # the feature rather than save ingestion cost. That call site passes
    # force=True.
    VISION_ENABLED: bool = True
    # Hard cap on how many images from ONE document get sent to Vision.
    #
    # 2026-08-23: there was no cap. extract_images_from_soup() did
    # soup.find_all("img") and captioned every result, so a single scraped
    # page could fire dozens of billed gpt-4o calls - and a full-site crawl
    # multiplies that by max_pages. A real 30-page e-commerce crawl queued
    # ~270 Vision calls before anyone noticed, because product photos pass
    # the icon/logo noise filter perfectly well.
    #
    # Images beyond this limit are skipped, not failed: the document still
    # ingests, its text and tables are unaffected, and the skip is logged
    # with a count so the cause is obvious rather than mysterious.
    #
    # NOTE for crawls: this bounds ONE document. Each crawled page is its
    # own document, so worst-case Vision calls for a crawl are
    # max_pages x this value - set max_pages deliberately, or turn
    # VISION_ENABLED off for bulk scraping runs.
    MAX_IMAGES_PER_DOCUMENT: int = 20
    IMAGE_MIN_SIZE_BYTES: int = 5000
    IMAGE_MAX_DIMENSION: int = 1024
    IMAGE_MAX_WORKERS: int = 4

    # ── S3 storage for uploaded documents ─────────────────────────────────
    # 2026-08-23. S3_BUCKET EMPTY (the default) = originals stay on local
    # disk exactly as before; nothing about storage changes until a bucket
    # is named here. This is the single on/off switch for the feature.
    #
    # Credentials are shared with clariona-core's AWS account. Left blank
    # they fall through to boto3's normal chain (instance profile / IAM
    # role / ~/.aws), which is what a real deployment should use.
    STORAGE_S3_BUCKET: str = ""
    STORAGE_S3_REGION: str = "ap-south-1"
    STORAGE_S3_KEY_PREFIX: str = "avabodh/documents"
    # Only for S3-compatible stores (MinIO, Ceph). Blank for real AWS.
    STORAGE_S3_ENDPOINT_URL: str = ""
    # Orphan sweeper ONLY - not a cache expiry. A local upload is normally
    # deleted the moment its ingestion job finishes; this exists solely to
    # clear files left behind when the process died mid-job. It must stay
    # comfortably longer than the slowest real ingest: measured wall time
    # for a 40-page PDF is 8-11.5 min, so a 200-page document can exceed
    # 40 min. Anything near 30 min would delete a source file mid-parse.
    STORAGE_LOCAL_TTL_HOURS: int = 4
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    SECRET_KEY: str = "dev-only-CHANGE-ME-in-production"
    FILE_LINK_TTL_SECONDS: Optional[int] = None
    PUBLIC_BASE_URL: str = ""

    # ── Core webhook (ingestion status push) ────────────────────────────────
    # Optional. When set, pipeline/webhook.py posts a status ping to Core the
    # moment a document reaches PROCESSING/READY/FAILED, instead of Core only
    # finding out on its next poll (api/routes/documents.py already reports
    # status synchronously on every GET, so this is purely a push-vs-pull
    # latency improvement, not a new source of truth). Left blank, nothing
    # changes — Core's existing poll loop is untouched and remains the
    # fallback if a ping is ever lost, so this is safe to leave unset.
    CORE_WEBHOOK_URL: str = ""
    # Shared secret sent as the X-Webhook-Secret header on every ping, so
    # Core can trust it actually came from this Avabodh instance. Must match
    # whatever Core's own webhook-receiving endpoint is configured to expect.
    CORE_WEBHOOK_SECRET: str = ""
    # Per-call timeout for the webhook POST — kept short deliberately. This
    # call happens inside the ingestion background task; a slow/hung Core
    # endpoint must never be able to stall ingestion itself.
    CORE_WEBHOOK_TIMEOUT_SECONDS: float = 5.0

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "avabodh.log"

    # ── Phase H hardening (backward-compatible defaults — see plan doc) ────
    # CORS_ALLOWED_ORIGINS: allow_origins=["*"] + allow_credentials=True is
    # spec-invalid and real browsers silently reject it — default stays "*"
    # (matches today's behavior for non-browser/server-to-server callers,
    # which CORS never applies to anyway), only tightens to a real allowlist
    # + allow_credentials=True when explicitly configured.
    CORS_ALLOWED_ORIGINS: str = "*"
    # SERVICE_API_KEY: empty by default = today's exact header-only trust
    # model (X-Tenant-ID/X-Org-Unit-ID, no other credential) — unchanged for
    # every current caller. Set this to additionally require X-Service-Key.
    SERVICE_API_KEY: str = ""
    # SCRAPER_ALLOW_PRIVATE_NETWORKS: SSRF guard escape hatch — default
    # false blocks /web/scrape targets resolving to loopback/private/
    # link-local ranges.
    SCRAPER_ALLOW_PRIVATE_NETWORKS: bool = False

    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "Avabodh Project"

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    def __init__(self, **data: Any):
        merged = self._load_env_file_data()
        # Real process environment variables always win over the .env file
        # — standard 12-factor precedence, and what lets a test fixture's
        # monkeypatch.setenv (or a real deployment env var) override a
        # value the checked-in .env also happens to define. Without this,
        # any key present in .env was passed to super().__init__() as an
        # explicit kwarg unconditionally, which silently shadowed the real
        # environment for that key no matter what actually set it.
        known = set(type(self).model_fields)
        merged.update({k: v for k, v in os.environ.items() if k in known})
        merged.update(data)

        # Drop keys this class doesn't declare BEFORE pydantic sees them.
        #
        # BaseSettings defaults to extra="forbid", and an unknown key raises
        # a ValidationError whose message embeds the offending VALUE as
        # input_value=... So adding any new credential to .env before its
        # field exists in code doesn't just fail to boot - it prints that
        # credential in plaintext into the startup traceback, which then
        # lands in container logs, CI output and error trackers. That
        # happened for real with AWS_SECRET_ACCESS_KEY during development.
        #
        # Ignoring unknown keys also matches how real environment variables
        # already behave here: the OS environment is full of unrelated
        # variables and none of them have ever been an error.
        dropped = [k for k in merged if k not in known]
        merged = {k: v for k, v in merged.items() if k in known}

        super().__init__(**merged)

        if dropped:
            # Names only, never values - the whole point of this block.
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "Ignoring %d unrecognised setting(s) from .env: %s",
                len(dropped), ", ".join(sorted(dropped)),
            )

    @classmethod
    def _load_env_file_data(cls) -> dict[str, Any]:
        paths = [
            Path(__file__).resolve().parents[1] / "local_demo" / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for env_path in paths:
            if env_path.exists():
                return cls._parse_env_file(env_path)
        return {}

    @staticmethod
    def _parse_env_file(env_path: Path) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip('"').strip("'")
        return data

    @field_validator(
        "DB_PORT",
        "DB_POOL_SIZE",
        "DB_MAX_OVERFLOW",
        "DB_POOL_TIMEOUT",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_MAX_RETRIES",
        "QDRANT_UPSERT_MAX_RETRIES",
        "MAP_MAX_TOKENS",
        "REDUCE_MAX_TOKENS",
        "LLM_REQUEST_TIMEOUT",
        "LLM_MAX_RETRIES",
        "CHAT_REQUEST_TIMEOUT",
        "CHAT_MAX_RETRIES",
        "MAX_CHUNK_SIZE",
        "MIN_CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "IMAGE_MIN_SIZE_BYTES",
        "IMAGE_MAX_DIMENSION",
        "IMAGE_MAX_WORKERS",
        "FILE_LINK_TTL_SECONDS",
        "SEARCH_CANDIDATES",
        "SEARCH_CANDIDATES_MAX",
        "SEARCH_SCORE_THRESHOLD",
        "MAX_ATTACHED_CROPS",
        "STORAGE_LOCAL_TTL_HOURS",
        "MAX_IMAGES_PER_DOCUMENT",
        "DOC_SHORTLIST_LIMIT",
        "MMR_LAMBDA",
        "MEMORY_WINDOW_SIZE",
        "MEMORY_SEMANTIC_RECALL_LIMIT",
        "MEMORY_SUMMARY_MAX_TOKENS",
        "TABLE_TEXT_LAYER_MIN_CHARS",
        mode="before",
    )
    @classmethod
    def _blank_int_to_default(cls, value: Any, info) -> Any:
        """
        A blank value in .env ("SETTING=" with nothing after it) means "use
        the default", not "None" — leaving a setting blank is the normal way
        to say you have no opinion about it.

        2026-08-23: this returned a bare None, which is only valid for the
        Optional[...] fields in the list below. For a plain int field
        (MAX_ATTACHED_CROPS, STORAGE_LOCAL_TTL_HOURS) None fails validation
        and the app refuses to START — so blanking one line in .env, which
        .env.example itself shows as normal, was a hard boot failure.
        Returning the field's declared default keeps the Optional fields
        behaving exactly as before (their default IS None) while making the
        plain ints fall back correctly.
        """
        if value in ("", None):
            field = cls.model_fields.get(info.field_name)
            return field.default if field is not None else None
        return value

    def _ollama_health(self, base_url: str) -> bool:
        try:
            with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2) as response:
                return response.status == 200
        except (HTTPError, URLError, TimeoutError, OSError):
            return False

    @property
    def use_ollama(self) -> bool:
        if self.OPENAI_API_KEY:
            return False
        return self._ollama_health(self.ollama_url)

    @property
    def ollama_url(self) -> str:
        if os.path.exists("/.dockerenv"):
            return self.OLLAMA_DOCKER_BASE_URL
        return self.OLLAMA_BASE_URL

    @property
    def db_url(self) -> str:
        """Admin/superuser connection — used ONLY for schema bootstrap in init_db(). Not for regular queries."""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def app_db_url(self) -> str:
        """Restricted app-role connection — used for all normal request-serving queries. RLS actually applies to this one."""
        return (
            f"postgresql+psycopg2://{self.APP_DB_USER}:{self.APP_DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def async_db_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings loaded once at startup."""
    return Settings()
