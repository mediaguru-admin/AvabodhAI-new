import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.routes import documents, health
from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.tenant_guard_middleware import TenantGuardMiddleware
from db.database import init_db, sweep_orphaned_uploads
from pipeline.vector_store import ensure_collections
from pipeline.llm_log import flush_llm_log_producer
from config.settings import get_settings
from utils.logger import get_logger
from api.routes import search
from api.routes import chat
from api.routes.web import router as web_router
from api.routes import kb
from api.routes import internal
from api.routes import llm_logs
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

_DEFAULT_SECRET_KEY = "dev-only-CHANGE-ME-in-production"
_DEFAULT_APP_DB_PASSWORD = "CHANGE-ME-app-role-password"


def _apply_langsmith_env() -> None:
    """
    Push LangSmith + OpenAI settings into os.environ so that LangChain
    picks them up at import time regardless of load order.
    """
    s = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", s.OPENAI_API_KEY)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", s.LANGCHAIN_TRACING_V2)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", s.LANGCHAIN_ENDPOINT)
    os.environ.setdefault("LANGCHAIN_PROJECT", s.LANGCHAIN_PROJECT)
    if s.LANGCHAIN_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", s.LANGCHAIN_API_KEY)


_apply_langsmith_env()


def _warm_table_model() -> None:
    """
    Load unstructured's table-structure model, without the wasted work its
    own loader does.

    THE PROBLEM. unstructured_inference calls
    TableTransformerForObjectDetection.from_pretrained(model) with no
    arguments. That checkpoint's config carries use_timm_backbone=True and
    use_pretrained_backbone=True, so during __init__ transformers asks timm
    to build a ResNet-18 and fill it with ImageNet weights
    (transformers/models/table_transformer/modeling_table_transformer.py
    line 225) — BEFORE the actual table checkpoint is applied over the top
    of it. Every one of those ImageNet weights is then overwritten.

    Worse, transformers builds the model under its init-empty-weights
    context, so those parameters are meta tensors at that moment. timm's
    load_state_dict copies real weights into placeholders, which does
    nothing, and torch says so — 60 times:

        UserWarning: for conv1.weight: copying from a non-meta parameter in
        the checkpoint to a meta parameter in the current model, which is a
        no-op.

    So the warning is accurate: that copy genuinely does nothing. It is
    reporting wasted work, not a broken model.

    THE FIX. use_pretrained_backbone=False skips the ImageNet fetch
    entirely; the backbone is built empty and the table checkpoint fills it
    as it always did. Verified: warnings 60 -> 0, and all 367 state-dict
    tensors are bit-identical to the default load, so this changes nothing
    about what the model computes.

    Applied here rather than by editing site-packages because
    unstructured_inference's own load_agent() short-circuits when
    tables_agent.model is already set — so pre-loading it correctly means
    every later caller transparently gets this instance. The attributes
    below mirror exactly what its initialize() sets.
    """
    import torch
    from transformers import DetrImageProcessor, TableTransformerForObjectDetection
    from transformers.utils import logging as hf_logging
    from unstructured_inference.config import inference_config
    from unstructured_inference.models.tables import DEFAULT_MODEL, tables_agent

    if getattr(tables_agent, "model", None) is not None:
        return

    with tables_agent._lock:
        if getattr(tables_agent, "model", None) is not None:
            return

        tables_agent.device = "cuda" if torch.cuda.is_available() else "cpu"

        tables_agent.feature_extractor = DetrImageProcessor.from_pretrained(DEFAULT_MODEL)
        # Not set in the checkpoint config; required by newer models.
        tables_agent.feature_extractor.size["shortest_edge"] = inference_config.IMG_PROCESSOR_SHORTEST_EDGE
        tables_agent.feature_extractor.size["longest_edge"] = inference_config.IMG_PROCESSOR_LONGEST_EDGE

        # Silence transformers' "Some weights ... were not used" notice for
        # the duration of the load, exactly as unstructured_inference's own
        # initialize() does. The three keys it names are
        # BatchNorm num_batches_tracked counters: this architecture replaces
        # nn.BatchNorm2d with TableTransformerFrozenBatchNorm2d, which has no
        # such counter, so dropping them is correct and has no effect on
        # inference. Restored here because replacing their loader would
        # otherwise surface a message they had deliberately hidden.
        cached_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            model = TableTransformerForObjectDetection.from_pretrained(
                DEFAULT_MODEL,
                use_pretrained_backbone=False,   # the fix - see docstring
            )
        finally:
            hf_logging.set_verbosity(cached_verbosity)

        tables_agent.model = model.to(tables_agent.device, dtype=torch.float32)
        tables_agent.model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Avabodh API...")
    init_db()
    logger.info("Database tables verified.")

    # Disk-side counterpart to mark_interrupted_jobs() (called inside
    # init_db above): that repairs Postgres rows left in PROCESSING by a
    # dead process, this removes the local upload files those jobs left
    # behind. No-ops entirely when S3 is not configured, because then the
    # local file is the only copy of the document. Non-fatal.
    try:
        sweep_orphaned_uploads()
    except Exception as e:
        logger.error("Orphaned-upload sweep failed (continuing startup): %s", e)

    try:
        ensure_collections()
        logger.info("Qdrant collections verified.")
    except Exception as e:
        # Non-fatal at startup — a Qdrant instance that's slow to come up
        # (or briefly unreachable during a rolling restart) shouldn't
        # crash-loop the whole API; requests that actually need Qdrant
        # will surface the real error at call time instead.
        logger.error("Qdrant collection setup failed (continuing startup): %s", e)

    # 2026-08-21: SPLADE (sparse) and the reranker (fastembed
    # TextCrossEncoder, both ONNX) are lazily constructed on first use via
    # @lru_cache (pipeline/embedder.py) — loading either from disk into
    # memory is real, multi-second work. Left lazy, that cost lands
    # entirely on whichever user's chat/search request happens to be
    # first after every server start — confirmed live: a first-after-
    # restart /chat/message request took 392s (that was while briefly
    # using a heavier torch-based reranker; still worth warming up now
    # regardless, since fastembed's own model load is non-trivial too).
    # Warming both up here means that cost is paid once at boot (visible
    # in this log, not blocking a real request) instead of unpredictably
    # stalling an arbitrary user request. Non-fatal: a warmup failure here
    # just means the lazy-load-on-first-use fallback still applies, same
    # as before this existed.
    try:
        from pipeline import embedder as _embedder
        _embedder._sparse_model()
        _embedder._rerank_model()
        _embedder._dense_client()
        logger.info("Sparse + rerank + dense embedding clients warmed up.")
    except Exception as e:
        logger.error("Embedder warmup failed (continuing startup, first real request pays this cost instead): %s", e)

    # 2026-08-23: same reasoning as the embedder warmup above, applied to the
    # INGESTION side. unstructured's hi_res strategy (settings.UNSTRUCTURED_
    # STRATEGY, the default) loads a YOLOX page-layout model and a table-
    # structure transformer lazily, on the first PDF it ever sees. Left lazy,
    # that is a multi-hundred-MB download plus a multi-second load charged to
    # whichever upload happens to be first after a restart. Warmed here so a
    # deploy pays it once, visibly, in this log.
    #
    # Deliberately skipped when the strategy is not hi_res: "fast" never
    # touches either model, so loading them would waste boot time and memory
    # for nothing. Non-fatal either way - a failure here just restores the
    # old lazy-load-on-first-use behaviour.
    # unstructured tokenises with NLTK when parsing DOCX and some text
    # formats. Missing data is fetched from the network ON FIRST USE, i.e.
    # inside a real upload. Checked (and fetched) here instead, so the cost
    # and any network failure land at boot where they are visible.
    try:
        import nltk
        for pkg in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                logger.info("NLTK '%s' missing - downloading at startup rather than mid-request.", pkg)
                nltk.download(pkg, quiet=True)
        logger.info("NLTK tokenizer data verified.")
    except Exception as e:
        logger.error("NLTK warmup failed (first DOCX upload pays this cost instead): %s", e)

    if str(getattr(settings, "UNSTRUCTURED_STRATEGY", "hi_res")).lower() == "hi_res":
        try:
            from unstructured_inference.models.base import get_model
            get_model()
            logger.info("PDF layout model warmed up.")
        except Exception as e:
            logger.error("Layout model warmup failed (first PDF upload pays this cost instead): %s", e)

        try:
            _warm_table_model()
            logger.info("Table-structure model warmed up.")
        except Exception as e:
            logger.error("Table model warmup failed (first PDF upload pays this cost instead): %s", e)

    # Phase H #10 — refuse to start with no OPENAI_API_KEY at all; nothing
    # in this app (embeddings, summarisation, chat, vision) works without
    # it, so failing at boot beats failing three requests deep. Any
    # environment that already sets this (every currently-working
    # deployment) is unaffected.
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — this app cannot function without it "
            "(embeddings, summarisation, chat, and image captioning all require it). "
            "Set it in your environment before starting."
        )

    # If this fires in a real deployment, the signed preview file links
    # (GET /documents/{id}/file) are forgeable by anyone who's read this
    # codebase — the default is deliberately public knowledge, not a
    # real secret. This doesn't block startup (a broken preview link
    # feature is less bad than the app refusing to boot), but it's loud
    # on purpose. Set SECRET_KEY in your environment before going live.
    if settings.SECRET_KEY == _DEFAULT_SECRET_KEY:
        logger.warning(
            "SECURITY WARNING: SECRET_KEY is still the default dev value. "
            "Signed document preview links can be forged by anyone. "
            "Set a real SECRET_KEY in your environment before production use."
        )

    # If this fires, the app is connecting to Postgres as a role whose
    # password is a publicly-documented placeholder — anyone who's read
    # this codebase (or this repo, if it's ever public) could log in as
    # avabodh_app directly. Row-Level Security still wouldn't let that
    # login see other tenants' data (RLS applies to this role by design),
    # but it would still let someone impersonate the application itself
    # against whichever tenant/org-unit they choose to set. Set a real
    # APP_DB_PASSWORD in your environment before production use.
    if settings.APP_DB_PASSWORD == _DEFAULT_APP_DB_PASSWORD:
        logger.warning(
            "SECURITY WARNING: APP_DB_PASSWORD is still the default placeholder value. "
            "Set a real APP_DB_PASSWORD in your environment before production use."
        )

    # Phase H #4 — loud but non-fatal, matching the pattern above: an
    # empty SERVICE_API_KEY means TenantGuardMiddleware trusts
    # X-Tenant-ID/X-Org-Unit-ID headers alone, same as clariona-core's
    # current calls already assume. That's fine behind a trusted gateway,
    # but worth flagging loudly for anyone running this standalone on a
    # reachable network without one.
    if not settings.SERVICE_API_KEY:
        logger.warning(
            "SERVICE_API_KEY is not set. Requests are authorized by X-Tenant-ID/"
            "X-Org-Unit-ID headers alone, with no verification that the caller is "
            "entitled to claim them. Safe behind a trusted gateway (e.g. clariona-core); "
            "set SERVICE_API_KEY if this API is ever reachable directly."
        )

    yield
    logger.info("Shutting down Avabodh API...")
    flush_llm_log_producer()


app = FastAPI(
    title="Avabodh API",
    description="Document Intelligence API — Upload documents, get AI summaries.",
    version="1.0.0",
    lifespan=lifespan,
)

# Phase H #5 — allow_origins=["*"] + allow_credentials=True is an invalid
# combination per the CORS spec; real browsers reject it outright, so the
# old config wasn't actually granting the openness it looked like it was.
# CORS only governs browser JS callers — server-to-server callers like
# clariona-core are never subject to CORS at all, so this cannot affect
# them either way. Default stays wide open ("*") unless explicitly
# tightened to a real origin list via CORS_ALLOWED_ORIGINS.
_cors_origins = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
_cors_wildcard = _cors_origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order matters here: LoggingMiddleware must be added LAST so it ends up
# outermost — that way every request, including ones TenantGuardMiddleware
# rejects with a 400, still shows up in the standard access log with its
# real status code, not just the guard's own warning line.
app.add_middleware(TenantGuardMiddleware)
app.add_middleware(LoggingMiddleware)

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """
    Phase H #6 safety net — catches anything that escapes a route's own
    try/except (most routes already catch and return a generic message
    themselves; this is the backstop for whatever doesn't). Logs the full
    exception server-side, returns a generic message client-side.

    Note: exception handlers registered this way run in Starlette's
    ServerErrorMiddleware, which sits OUTSIDE LoggingMiddleware — an
    exception that reaches here skips LoggingMiddleware's normal
    "attach X-Request-ID to the response" step entirely (call_next raised
    instead of returning), so the header is set explicitly below instead
    of relying on that middleware.
    """
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled exception [%s] %s %s: %s", request_id, request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Contact support with the X-Request-ID response header if this persists."},
        headers={"X-Request-ID": request_id},
    )


app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(web_router, prefix="/web", tags=["Web Scraping"])
app.include_router(kb.router, prefix="/kb", tags=["Knowledge Base"])
app.include_router(internal.router, prefix="/internal", tags=["Internal"])
app.include_router(llm_logs.router, prefix="/llm-logs", tags=["LLM Logs"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "Avabodh API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import os
    import uvicorn

    # 2026-08-21: --reload has caused two SEPARATE real failures tonight,
    # not one:
    #   1. Watching the whole project dir (incl. avabodh.log, written on
    #      every request) meant every request's own log line triggered a
    #      full reload, sometimes mid-request — confirmed live as
    #      multi-minute stalls and a "only responds after I stop the
    #      server" symptom. reload_excludes (below) fixes this part.
    #   2. Independently, Windows' multiprocessing.spawn (what WatchFiles
    #      uses to launch the reloaded worker) can deadlock against an
    #      import lock still held in the parent process when a reload
    #      fires while a heavy import graph (openai/qdrant_client/
    #      fastembed/sqlalchemy) is mid-load — confirmed live via a
    #      traceback stuck in frozen importlib._bootstrap's `acquire`
    #      during `import _uuid`. This is a known category of flakiness
    #      with reload+spawn on Windows, not something reload_excludes
    #      can fix — the only reliable answer is not auto-reloading.
    # Given both, reload defaults OFF now — restart manually
    # (Ctrl+C, `python main.py` again) after edits. Set
    # AVABODH_DEV_RELOAD=1 to opt back into --reload if you want to trade
    # reliability for auto-restart convenience anyway.
    reload_enabled = os.environ.get("AVABODH_DEV_RELOAD", "").strip() == "1"
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8009,
        reload=reload_enabled,
        reload_excludes=[
            "*.log",
            "uploaded_files/*",
            "kb_uploaded_files/*",
            "documents/*",
        ] if reload_enabled else None,
    )