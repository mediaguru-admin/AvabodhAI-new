"""
scripts/reset_data.py
-----------------------
Dev/test reset tool — deletes documents (Postgres rows + their Qdrant
points + on-disk files), and optionally chat threads/messages, scoped by
tenant/org and/or age. NOT a production retention policy — this is meant
to be run by hand (or scheduled yourself via Windows Task Scheduler/cron
if you want it recurring) to reset a dev/test environment between runs.

Safety model, matching this repo's other operator scripts
(scripts/init_schema.py --create-database, cli.py):
  - DRY RUN BY DEFAULT. Without --yes, this only lists what it WOULD
    delete and exits — nothing is touched.
  - At least one scope flag is required (--tenant-id+--org-unit-id,
    --older-than-days, or the explicit --all-tenants escape hatch) — no
    accidental unscoped wipe from a bare invocation.

Usage:
    # Preview only (no --yes) — always safe to run
    python scripts/reset_data.py --tenant-id t1 --org-unit-id o1
    python scripts/reset_data.py --older-than-days 7
    python scripts/reset_data.py --all-tenants

    # Actually delete
    python scripts/reset_data.py --tenant-id t1 --org-unit-id o1 --yes
    python scripts/reset_data.py --older-than-days 7 --include-chat --yes
    python scripts/reset_data.py --all-tenants --include-chat --yes
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from db.database import _admin_engine
from pipeline import vector_store
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_where(tenant_id, org_unit_id, older_than_days, created_column="created_at"):
    clauses, params = [], {}
    if tenant_id:
        clauses.append("tenant_id = :tenant_id")
        params["tenant_id"] = tenant_id
    if org_unit_id:
        clauses.append("org_unit_id = :org_unit_id")
        params["org_unit_id"] = org_unit_id
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        clauses.append(f"{created_column} < :cutoff")
        params["cutoff"] = cutoff
    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def _reset_documents(where_sql, params, execute: bool) -> int:
    with _admin_engine.begin() as conn:
        rows = conn.execute(
            text(f"SELECT id, tenant_id, org_unit_id, doc_name, stored_path, created_at FROM documents {where_sql} ORDER BY created_at"),
            params,
        ).fetchall()

    if not rows:
        print("Documents: nothing matches this scope.")
        return 0

    print(f"Documents: {len(rows)} match this scope" + (" (DRY RUN — nothing deleted):" if not execute else ":"))
    for r in rows[:20]:
        print(f"  - {r.doc_name}  (id={r.id}, tenant={r.tenant_id}, org_unit={r.org_unit_id}, created={r.created_at})")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")

    if not execute:
        return len(rows)

    for r in rows:
        try:
            vector_store.delete_document_points(tenant_id=r.tenant_id, document_id=str(r.id))
            vector_store.delete_document_summary(tenant_id=r.tenant_id, document_id=str(r.id))
        except Exception as e:
            logger.warning("Qdrant delete failed for document %s (continuing): %s", r.id, e)
        if r.stored_path and os.path.exists(r.stored_path):
            try:
                os.remove(r.stored_path)
            except OSError as e:
                logger.warning("File delete failed for %s (continuing): %s", r.stored_path, e)

    with _admin_engine.begin() as conn:
        conn.execute(text(f"DELETE FROM documents {where_sql}"), params)

    print(f"✓ Deleted {len(rows)} document(s) — Postgres rows, Qdrant points, and on-disk files.")
    return len(rows)


def _reset_chat(where_sql, params, execute: bool) -> int:
    with _admin_engine.begin() as conn:
        rows = conn.execute(
            text(f"SELECT id, tenant_id, org_unit_id, title, created_at FROM chat_threads {where_sql} ORDER BY created_at"),
            params,
        ).fetchall()

    if not rows:
        print("Chat threads: nothing matches this scope.")
        return 0

    print(f"Chat threads: {len(rows)} match this scope" + (" (DRY RUN — nothing deleted):" if not execute else ":"))
    for r in rows[:20]:
        print(f"  - {r.title or '(untitled)'}  (id={r.id}, tenant={r.tenant_id}, org_unit={r.org_unit_id}, created={r.created_at})")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")

    if not execute:
        return len(rows)

    for r in rows:
        try:
            vector_store.delete_thread_messages(tenant_id=r.tenant_id, thread_id=str(r.id))
        except Exception as e:
            logger.warning("Qdrant delete failed for thread %s (continuing): %s", r.id, e)

    with _admin_engine.begin() as conn:
        # chat_messages cascade-deletes via ON DELETE CASCADE on thread_id.
        conn.execute(text(f"DELETE FROM chat_threads {where_sql}"), params)

    print(f"✓ Deleted {len(rows)} chat thread(s) — Postgres rows (+cascaded messages) and Qdrant points.")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev/test reset — delete documents (and optionally chat) from Postgres + Qdrant, scoped by tenant/org and/or age.")
    parser.add_argument("--tenant-id", type=str, default=None)
    parser.add_argument("--org-unit-id", type=str, default=None)
    parser.add_argument("--older-than-days", type=int, default=None, help="Only delete rows created before this many days ago.")
    parser.add_argument("--all-tenants", action="store_true", help="Explicit escape hatch — required if neither --tenant-id nor --older-than-days is given, to prevent an accidental unscoped wipe.")
    parser.add_argument("--include-chat", action="store_true", help="Also delete matching chat_threads/chat_messages (same scope).")
    parser.add_argument("--yes", action="store_true", help="Actually delete. Without this, only a dry-run preview is printed.")
    args = parser.parse_args()

    if args.tenant_id and not args.org_unit_id:
        print("Error: --tenant-id requires --org-unit-id too (both headers are always required together in this app).")
        sys.exit(1)
    if not args.tenant_id and not args.older_than_days and not args.all_tenants:
        print("Error: no scope given. Pass --tenant-id + --org-unit-id, --older-than-days N, or --all-tenants to confirm you really want an unscoped wipe.")
        sys.exit(1)

    where_sql, params = _build_where(args.tenant_id, args.org_unit_id, args.older_than_days)
    scope_desc = where_sql if where_sql else "(no filter — ALL tenants)"
    print(f"Scope: {scope_desc}")
    if not args.yes:
        print("\n--- DRY RUN (pass --yes to actually delete) ---\n")

    doc_count = _reset_documents(where_sql, params, execute=args.yes)
    chat_count = 0
    if args.include_chat:
        chat_count = _reset_chat(where_sql, params, execute=args.yes)

    if not args.yes and (doc_count or chat_count):
        print(f"\nRe-run with --yes to actually delete {doc_count} document(s)" + (f" and {chat_count} chat thread(s)" if args.include_chat else "") + ".")


if __name__ == "__main__":
    main()
