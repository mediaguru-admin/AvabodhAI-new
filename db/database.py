"""
db/database.py
--------------
SQLAlchemy engines + session factories.

TWO engines, deliberately, not one:

  _admin_engine — connects using DB_USER/DB_PASSWORD, an externally
  provisioned Postgres instance (not Docker-managed — see
  scripts/init_schema.py). Used ONLY inside init_db(), ONLY for one-time
  schema bootstrap: creating tables, the restricted app role, and the
  Row-Level Security policies. Never used to serve a real request.

  engine — connects using APP_DB_USER/APP_DB_PASSWORD (a restricted,
  non-superuser role created by init_db()). This is what every real
  request in this app actually queries through.

Why the split exists: Postgres Row-Level Security CANNOT restrict a
superuser's queries — that's a fixed Postgres rule, no policy setting
changes it. If the app queried using the superuser for everything, the
RLS policies created below would provide zero real protection for the
app's own traffic. This split is what makes RLS meaningful at all.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from fastapi import Request
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from config.settings import get_settings
from db.models import Base
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Admin/bootstrap engine — schema DDL only, never used for requests ──────
_admin_engine = create_engine(settings.db_url, pool_pre_ping=True, echo=False)

# ── App engine — everything else uses this. RLS actually applies here. ─────
engine = create_engine(
    settings.app_db_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=False,
)

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# The tables that carry tenant_id/org_unit_id and need RLS. document_chunks
# is gone as of 2026-08-21 — chunk data (and its tenant/org tags) now lives
# in Qdrant, which has no RLS equivalent; pipeline/retriever.py::build_filter()
# is the sole isolation control for chunks. See CLAUDE.md / the plan doc.
_ISOLATED_TABLES = ["documents", "chat_threads", "chat_messages", "llm_execution_logs"]


def _bootstrap_app_role(conn) -> None:
    """
    Create (or update the password of) the restricted application role
    that `engine` above connects as. Idempotent — safe to run on every
    startup. Not a superuser, not BYPASSRLS, so RLS genuinely applies to
    it — this is the role teammates should also use in pgAdmin if they
    want tenant/org-unit isolation to actually bound their own queries.

    APP_DB_PASSWORD is operator-controlled config (an env var), not
    end-user input, so direct (escaped) interpolation into DDL here is
    an accepted, deliberate exception to "always use bind parameters" —
    CREATE ROLE / ALTER ROLE cannot take bind parameters at all.
    """
    escaped_password = settings.APP_DB_PASSWORD.replace("'", "''")
    conn.execute(text(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{settings.APP_DB_USER}') THEN
                CREATE ROLE {settings.APP_DB_USER} LOGIN PASSWORD '{escaped_password}';
            ELSE
                ALTER ROLE {settings.APP_DB_USER} LOGIN PASSWORD '{escaped_password}';
            END IF;
        END
        $$;
    """))
    conn.execute(text(f'GRANT CONNECT ON DATABASE "{settings.DB_NAME}" TO {settings.APP_DB_USER}'))
    conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {settings.APP_DB_USER}"))
    # Runs AFTER create_all(), so this covers every table that exists at
    # this point — including ones added by future schema changes, since
    # create_all() always runs first in init_db() below.
    conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {settings.APP_DB_USER}"))
    logger.info("App role '%s' verified/updated.", settings.APP_DB_USER)


def _bootstrap_rls(conn) -> None:
    """
    Enable + FORCE Row-Level Security on the isolated tables, with a
    policy that only allows a row to be SEEN if it matches the
    tenant_id/org_unit_id set on the current session (via set_config()
    in get_db_session_fastapi/get_db_session_context below).

    FORCE (not just ENABLE) is required — without it, RLS is silently
    skipped for the table's OWNER, and in a simple setup like this the
    app role often ends up owning the tables it creates. FORCE closes
    that gap. It does NOT and cannot restrict an actual Postgres
    superuser — nothing can, by Postgres design — which is the whole
    reason the app connects as the restricted role above, not as
    DB_USER.

    WITH CHECK (true) deliberately does NOT restrict INSERT/UPDATE the
    same way USING restricts SELECT. Every write path in this codebase
    already sets tenant_id/org_unit_id explicitly and correctly at the
    Python level (see pipeline/storage.py, embedder.py, chat_storage.py)
    — RLS's job here is to be the backstop for ad-hoc READS (a teammate
    in pgAdmin, a future report, a bug in some other service), not to
    re-validate writes that are already independently guaranteed correct.
    Making writes fail-closed too would mean any request whose session
    variables aren't set for any reason (a bug, a forgotten header on a
    brand-new endpoint) fails LOUD on the write path, which is arguably
    safer — but it also means this table becomes unwritable the moment
    that wiring has any bug at all, anywhere. Read-side enforcement gets
    most of the real-world protection (this is what stops the "someone
    ran an unscoped SELECT in pgAdmin" scenario) at a much smaller blast
    radius. Revisit this trade-off if write-path correctness is ever in
    doubt.
    """
    for table in _ISOLATED_TABLES:
        conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        conn.execute(text(f"DROP POLICY IF EXISTS {table}_isolation ON {table}"))
        conn.execute(text(f"""
            CREATE POLICY {table}_isolation ON {table}
            USING (
                tenant_id = current_setting('app.tenant_id', true)
                AND org_unit_id = current_setting('app.org_unit_id', true)
            )
            WITH CHECK (true)
        """))
    logger.info("Row-Level Security enabled + forced on: %s", ", ".join(_ISOLATED_TABLES))


def _bootstrap_document_registry_rename(conn) -> None:
    """
    2026-08-21 architecture change (chunk storage moved to Qdrant, see
    db/models.py's module docstring and pipeline/vector_store.py):
    document_summaries -> documents. MUST run before create_all() below —
    create_all() only creates tables that don't exist yet, so if it ran
    first on a pre-migration database it would silently create a brand
    new, EMPTY `documents` table (since ORM Document.__tablename__ is
    "documents"), leaving the old, populated document_summaries sitting
    orphaned right next to it rather than actually being renamed.

    Guarded so this is a true no-op on both a fresh database (never had
    document_summaries — create_all() alone builds `documents` correctly)
    and an already-migrated one (idempotent, matching every other
    bootstrap step in this file).
    """
    old_exists = conn.execute(text(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'document_summaries'"
    )).first()
    new_exists = conn.execute(text(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'documents'"
    )).first()
    if not old_exists or new_exists:
        return

    logger.warning(
        "Migrating legacy document_summaries table to documents "
        "(one-time, first boot after the Qdrant chunk-storage architecture change)"
    )
    conn.execute(text("ALTER TABLE document_summaries RENAME TO documents"))


def _bootstrap_document_registry_columns(conn) -> None:
    """
    New Document lifecycle/metadata columns from the same 2026-08-21
    change as _bootstrap_document_registry_rename() above — split into
    its own step because it must run AFTER create_all(), the opposite
    ordering constraint: create_all() can't add these to a table that
    doesn't exist yet on a database that still needs the rename above.

    Existing rows are presumed already-ingested documents from before
    this column existed, so they're backfilled to 'READY' rather than
    the model's own default of 'UPLOADED' (which means "ingestion not
    finished yet") — being new to just this column is not the same as
    being mid-processing.
    """
    conn.execute(text("""
        ALTER TABLE documents
          ADD COLUMN IF NOT EXISTS status        VARCHAR(16) NOT NULL DEFAULT 'READY',
          ADD COLUMN IF NOT EXISTS status_detail TEXT,
          ADD COLUMN IF NOT EXISTS processed_at  TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS file_type     VARCHAR(16),
          ADD COLUMN IF NOT EXISTS file_size     INTEGER,
          ADD COLUMN IF NOT EXISTS source        VARCHAR(16) DEFAULT 'upload',
          ADD COLUMN IF NOT EXISTS stored_path   VARCHAR(1024),
          ADD COLUMN IF NOT EXISTS metadata      JSONB
    """))
    conn.execute(text("UPDATE documents SET status = 'READY' WHERE status = 'UPLOADED'"))


def _bootstrap_chat_memory_columns(conn) -> None:
    """
    create_all() only creates tables that don't exist yet — it cannot ALTER
    an existing chat_threads table to add the rolling-summary columns
    (db/models.py::ChatThread.rolling_summary / rolling_summary_through,
    added for pipeline/memory.py's rolling-summary conversation memory).
    IF NOT EXISTS makes this idempotent, matching the rest of init_db().
    """
    conn.execute(text("ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS rolling_summary TEXT"))
    conn.execute(text(
        "ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS rolling_summary_through INTEGER NOT NULL DEFAULT 0"
    ))


def _bootstrap_chat_degraded_column(conn) -> None:
    """
    Same reason/pattern as _bootstrap_chat_memory_columns() above — adds
    db/models.py::ChatMessage.degraded to an already-existing chat_messages
    table. Internal-only (see that column's docstring): which retrieval
    components silently fell back for a given turn, so a bad answer can be
    explained after the fact without re-running anything. Never exposed on
    ChatMessageResponse or any other API surface.
    """
    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS degraded JSON"))


def init_db() -> None:
    """
    Schema bootstrap — runs on every app startup, using the ADMIN engine
    only. Idempotent: safe to run repeatedly against an already-set-up
    database, which matters since this runs on every container start,
    not just the first one.

    2026-08-21: no longer enables the pgvector extension or the FTS
    trigger — chunk storage (and its embeddings/tsvector) moved to
    Qdrant (pipeline/vector_store.py, scripts/init_qdrant.py). Postgres
    now only needs the plain tables below.
    """
    try:
        with _admin_engine.begin() as conn:
            _bootstrap_document_registry_rename(conn)

        Base.metadata.create_all(bind=_admin_engine)
        logger.info("Database tables verified / created successfully.")

        with _admin_engine.begin() as conn:
            _bootstrap_document_registry_columns(conn)
            _bootstrap_chat_memory_columns(conn)
            _bootstrap_chat_degraded_column(conn)
            _bootstrap_app_role(conn)
            _bootstrap_rls(conn)

        mark_interrupted_jobs()

    except OperationalError as e:
        logger.critical("Cannot connect to PostgreSQL: %s", e)
        raise


def sweep_orphaned_uploads() -> int:
    """
    Startup cleanup for local upload files left behind by a dead process.

    2026-08-23, added alongside mark_interrupted_jobs() below and for the
    same reason: ingestion is in-process (BackgroundTasks, no job queue), so
    a crash/OOM/deploy-restart kills the job mid-flight. mark_interrupted_
    jobs() repairs the Postgres row; this repairs the disk. Normally
    pipeline/ingest.py deletes the local copy in its own finally block the
    moment a job ends, so in steady state this finds nothing.

    SAFETY RULE — a file is deleted ONLY when a document row proves S3
    already holds it: source_path still records where the local scratch
    file was written, while stored_path is rewritten to the s3:// URI once
    the object is durably stored. Both must line up.

    The first version of this deleted purely on age and destroyed five real
    files on its first run: documents ingested before S3 existed still have
    a LOCAL stored_path, so their upload was the only copy in the world and
    nothing had ever put it in a bucket. Age alone can never tell "abandoned
    scratch file" apart from "the user's only copy" — the S3 URI is the only
    honest proof, so that is what is checked now. Anything unmatched is left
    on disk: a stray file costs a few MB, and deleting the wrong one is
    unrecoverable.

    The age cutoff (settings.STORAGE_LOCAL_TTL_HOURS, default 4) is a second
    guard on top of that, not the primary one. It is NOT a cache expiry: it
    must stay well above the slowest real ingest, since measured wall time
    for a 40-page PDF is 8-11.5 minutes and a 200-page document can exceed
    40. Anything near 30 minutes could catch a job still parsing.

    Returns the number of files removed.
    """
    from pipeline import object_store

    if not object_store.is_enabled():
        return 0

    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        return 0

    # Every local path that a document has SINCE migrated to S3. Only these
    # are safe to remove. Admin engine: this must see across all tenants,
    # not be scoped by RLS the way a real request is.
    try:
        with _admin_engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT source_path FROM documents
                WHERE stored_path LIKE 's3://%'
                  AND source_path IS NOT NULL
            """)).fetchall()
    except Exception as e:
        logger.warning("Orphan sweep skipped - could not read documents: %s", e)
        return 0

    safe_to_delete = {os.path.normcase(os.path.abspath(r[0])) for r in rows}
    if not safe_to_delete:
        return 0

    cutoff = time.time() - (settings.STORAGE_LOCAL_TTL_HOURS * 3600)
    removed = 0
    # Iterate the KNOWN-SAFE paths, not the whole upload tree. rglob("*")
    # made startup cost scale with everything on disk while only ever
    # acting on this set, which is bounded by how many documents have
    # actually migrated to S3.
    for candidate in safe_to_delete:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue  # too recent to assume abandoned
            path.unlink()
            removed += 1
        except OSError as e:
            logger.warning("Could not sweep orphaned upload %s: %s", path, e)

    if removed:
        logger.info(
            "Swept %d orphaned upload file(s) older than %dh whose original is in S3.",
            removed, settings.STORAGE_LOCAL_TTL_HOURS,
        )
    return removed


def mark_interrupted_jobs() -> int:
    """
    Startup crash recovery: any Document still in PROCESSING when the app
    starts was mid-ingest when the previous process died (crash, OOM kill,
    deploy restart) — background ingestion is in-process (BackgroundTasks,
    no persistent job queue), so a dead process means that job is gone for
    good, not resumable. Mark those FAILED so they show up as retryable
    (POST /documents/{id}/reprocess) instead of hanging in PROCESSING
    forever with no worker ever going to finish them.

    Uses the ADMIN engine — this must see across every tenant/department,
    not scoped by RLS the way a real request is.
    """
    with _admin_engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE documents
            SET status = 'FAILED', status_detail = 'interrupted by restart'
            WHERE status = 'PROCESSING'
        """))
        if result.rowcount:
            logger.warning("Marked %d interrupted document(s) as FAILED on startup.", result.rowcount)
        return result.rowcount


def check_db_connection() -> bool:
    """Checks connectivity using the APP engine — this is what actually matters for serving traffic."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return False


def _apply_rls_context(session: Session, tenant_id: Optional[str], org_unit_id: Optional[str]) -> None:
    """
    Sets the session-local variables the RLS policies in _bootstrap_rls()
    check against. Uses set_config(..., true) — the `true` means
    "local to this transaction", equivalent to SET LOCAL, so it can
    never leak into a later request that reuses this pooled connection.

    Uses set_config() as a normal parameterized function call (bind
    params work here, unlike the raw SET command), NOT string
    interpolation — tenant_id/org_unit_id originate from HTTP headers,
    which are untrusted input, so this is the one place in this file
    where safe parameter binding is non-negotiable, not a nicety.

    2026-08-23 — bound to the session's "after_begin" event rather than
    executed once here, because "local to this transaction" cuts BOTH
    ways: a commit ends that transaction, and every statement after it
    runs in a NEW one where app.tenant_id/app.org_unit_id are back to
    empty. The RLS policies then match zero rows, so any read after a
    commit on the same session fails — confirmed live as
    "InvalidRequestError: Could not refresh instance '<Document ...>'"
    from PATCH /documents/{id} (api/routes/documents.py::update_document),
    where the UPDATE committed fine and the db.refresh() right after it
    came back empty. Note SQLAlchemy's expire_on_commit defaults to True,
    so this is NOT limited to explicit refresh() calls — merely READING
    an attribute off a committed instance triggers the same reload, and
    would have hit the same wall. Re-applying on every transaction start
    makes the context hold for the session's whole lifetime.

    The listener takes the connection handed to it by the event and uses
    that directly, rather than calling back into session.execute() (which
    would re-enter transaction begin from inside the begin handler).
    """
    if tenant_id is None and org_unit_id is None:
        return

    def _set(connection) -> None:
        if tenant_id is not None:
            connection.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": tenant_id})
        if org_unit_id is not None:
            connection.execute(text("SELECT set_config('app.org_unit_id', :v, true)"), {"v": org_unit_id})

    @event.listens_for(session, "after_begin")
    def _reapply_rls_on_new_transaction(sess, transaction, connection) -> None:
        _set(connection)


@contextmanager
def get_db_session(tenant_id: Optional[str] = None, org_unit_id: Optional[str] = None) -> Generator[Session, None, None]:
    session: Session = SessionFactory()
    try:
        _apply_rls_context(session, tenant_id, org_unit_id)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_session_context(tenant_id: Optional[str] = None, org_unit_id: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager for use outside FastAPI routes (pipeline/storage.py,
    embedder.py, chat_storage.py). Every call site in this codebase
    already has tenant_id/org_unit_id in scope as its own function
    parameters — pass them through here so RLS-protected SELECTs inside
    these functions (dedup checks especially) don't silently start
    returning zero rows once RLS is enabled.
    """
    session: Session = SessionFactory()
    try:
        _apply_rls_context(session, tenant_id, org_unit_id)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session_fastapi(request: Request) -> Generator[Session, None, None]:
    """
    FastAPI dependency injection version. Reads tenant_id/org_unit_id
    from request.state (set by TenantGuardMiddleware from the
    X-Tenant-ID/X-Org-Unit-ID headers) and applies them automatically —
    every route using `db: Session = Depends(get_db_session_fastapi)`
    gets RLS context for free, with no per-route change needed.

    Routes that don't receive those headers (health checks, and the
    signed preview-file endpoint, which is authorized via a token
    instead) simply get no RLS context set here — request.state.tenant_id
    is None for those, and _apply_rls_context() no-ops on None. The
    preview-file endpoint sets its own RLS context manually from the
    verified token payload — see api/routes/documents.py.
    """
    session: Session = SessionFactory()
    try:
        tenant_id = getattr(request.state, "tenant_id", None)
        org_unit_id = getattr(request.state, "org_unit_id", None)
        _apply_rls_context(session, tenant_id, org_unit_id)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from typing import AsyncGenerator

async_engine = create_async_engine(
    settings.async_db_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionFactory = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    autocommit=False,
    class_=AsyncSession,
)


async def get_async_db_session_fastapi() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency injection version for async routes."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
