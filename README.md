# AvabodhAI

A standalone, multi-tenant document ingestion and retrieval service. Upload
documents (or scrape web pages), get AI summaries and structured metadata,
and search across everything with hybrid (semantic + keyword) retrieval —
plus a RAG chat interface with multimodal (image-aware) support.

This is the knowledge-base backend `clariona-core` runs as `avabodh_api`
(`KNOWLEDGE_BASE_API_URL`), and it's also consumed by a misinformation-
detection engine. **The external API contract (paths, request/response
shapes) does not change without a deliberate, additive-only decision** —
see "Design principles" below.

Rewritten 2026-08-21 around a TL-authored architecture: PostgreSQL holds
document-level registry/lifecycle state, **Qdrant** holds the searchable
chunk index (dense + sparse vectors, filterable metadata, chunk text
itself). See `CLAUDE.md` for a deep technical walkthrough of the codebase.

---

## Architecture at a glance

```
Upload/Scrape → register (Postgres, status=UPLOADED)
              → background job (pipeline/ingest.py):
                  extract (unstructured) → chunk (chunk_by_title)
                  → embed (dense: OpenAI, sparse: SPLADE)
                  → index into Qdrant
                  → summarise + extract metadata (non-fatal)
                  → caption/index images (non-fatal)
              → status=READY | FAILED

Search/Chat → build_filter() [tenant/org/doc/metadata/effective-date]
            → Qdrant hybrid search (dense+sparse, RRF fusion)
            → rerank (cross-encoder)
            → results
```

| | PostgreSQL | Qdrant |
|---|---|---|
| Owns | Document registry, lifecycle status, chat threads/messages | Chunk text, dense + sparse vectors, filterable metadata, chat-message vectors |
| Isolation | Row-Level Security (RLS), forced, restricted app role | `build_filter()` — the sole isolation control (Qdrant has no RLS equivalent) |
| Managed by | You — **not Docker** (`scripts/init_schema.py`) | You — **not Docker** (`scripts/init_qdrant.py`) |

Both Postgres and Qdrant are **externally provisioned, plug-and-play
instances** — this repo's `docker-compose.yml` does not create or own
either one, and neither is shared with `clariona-core`'s own database, even
if reachable on the same network. Point this app at your own instances via
env vars (see `.env.example`).

---

## Multi-tenancy

Every table, every Qdrant point, every query is scoped by **both**
`tenant_id` (from `X-Tenant-ID`) and `org_unit_id` (from `X-Org-Unit-ID`),
always applied together, never independently — a department code that
happens to collide across two different tenants must never leak across
companies. Enforced in layers:

1. `TenantGuardMiddleware` — rejects any non-exempt request missing either header.
2. `api/dependencies.py` — threads both values into every route.
3. Postgres: forced Row-Level Security, via a restricted, non-superuser app role.
4. Qdrant: `pipeline/retriever.py::build_filter()` — the single chokepoint every read goes through.

`AvabodhAI` does **not** do user authentication/authorization — it trusts
`X-Tenant-ID`/`X-Org-Unit-ID` as coming from a trusted upstream caller (a
gateway, or an internal caller like `clariona-core`). If this service is
ever exposed directly to end users, that header must instead be derived
from a verified token, not trusted as plain input.

---

## Getting started

### 1. Provision your infrastructure

Postgres and Qdrant are **not** started by `docker compose up` — they're
real, permanent instances you provision yourself (local install, a managed
service, whatever your team runs), then point this app at via env vars.

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, DB_HOST/DB_PASSWORD, QDRANT_URL, SECRET_KEY

python scripts/init_schema.py    # creates Postgres tables, app role, RLS
python scripts/init_qdrant.py    # creates Qdrant collections + indexes
```

Both scripts are idempotent — safe to re-run against an already-provisioned
instance.

### 2. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium       # for /web/scrape
```

### 3. Run

```bash
uvicorn main:app --reload          # http://127.0.0.1:8000, docs at /docs
```

or with Docker (only the API itself is containerized — Postgres/Qdrant are
external):

```bash
docker compose up --build
```

### 4. Verify

```bash
curl http://localhost:8000/health/       # basic
curl http://localhost:8000/health/db     # Postgres reachable
curl http://localhost:8000/health/qdrant # Qdrant reachable
```

---

## Core API surface

All routes below require `X-Tenant-ID` + `X-Org-Unit-ID` headers, except
`/`, `/health*`, `/docs`, `/redoc`, `/openapi.json`, and
`GET /documents/{id}/file` (signed-token authorized instead).

Optional `X-Client-ID` names the calling project (e.g. `clariona-core`); it is
recorded on LLM execution logs as sent, not authenticated. With
`LLM_LOG_KAFKA_ENABLED=true` each log is also published as an
`llm.execution.logged.v1` event (topic from `LLM_LOG_KAFKA_TOPIC_TEMPLATE`) —
see `pipeline/llm_log.py`.

| Method | Path | Notes |
|---|---|---|
| POST | `/documents/upload` | Async — registers immediately (`status=UPLOADED`), ingests in the background. Poll `GET /documents/{id}` for `status`. |
| GET | `/documents/` | List, filterable by type/domain/category/status. |
| GET | `/documents/{id}` | Full detail + preview link + ingestion status. |
| GET | `/documents/{id}/preview-url` | Fresh signed preview link. |
| GET | `/documents/{id}/file` | Stream original file (signed link only). |
| PATCH / DELETE | `/documents/{id}` | Rename / delete (Postgres row + Qdrant points + file). |
| POST | `/documents/{id}/reprocess` | Retry ingestion for a FAILED (or any) document. |
| POST | `/search/` | Hybrid/semantic/keyword search — `org_ids`, `document_ids`, `filters` (allowlisted metadata), `as_of` (effective-date), `is_ground_truth` all optional. |
| GET | `/search/chunks/{doc_id}` | All chunks for a document, from Qdrant. |
| POST | `/chat/message` | Full JSON chat response, optional image attachment (multimodal). |
| POST/GET/PATCH/DELETE | `/chat/threads*` | Thread CRUD. |
| GET | `/chat/search` | Semantic search across chat history (Qdrant-backed). |
| POST | `/web/scrape` | Single page / batch / full-site crawl. Registers inline, ingests in the background. |
| GET | `/llm-logs/` | LLM execution log for this tenant/org unit: one row per chat-completion / vision call (purpose, model, tokens, latency, status). Filters: `purpose`, `status`, `client_id`, `model_name`, `entity_type`, `entity_id`, `created_from`/`created_to`; `include_text=true` adds prompt/response text. |
| GET | `/llm-logs/{id}` | One log row with prompt/response text. |
| GET | `/health/`, `/health/db`, `/health/qdrant` | Health checks. |

There's also a separate `/kb/*` subsystem (`api/routes/kb.py`) — an
independent, out-of-scope feature with its own tables and vector store, not
part of the migration described here.

### CLI

```bash
python cli.py --file doc.pdf --tenant-id t1 --org-unit-id d1
python cli.py --dir ./docs --tenant-id t1 --org-unit-id d1
python cli.py --check-db
python cli.py --check-qdrant
```

---

## Design principles

- **External contract is frozen; internals aren't.** Endpoint paths and
  response shapes only change additively (new optional fields) — two real
  systems (`clariona-core`'s knowledge-base integration, a
  misinformation-detection engine) depend on this contract today.
- **No permanent data store is Docker-managed.** Postgres and Qdrant are
  provisioned by the operator and connected to via env vars — this keeps
  the project genuinely plug-and-play: another team installs their own
  instances and points this app at them, nothing shared, nothing implicit.
- **Async ingestion, in-process.** `POST /documents/upload` and
  `POST /web/scrape` return immediately; the real work (extract, chunk,
  embed, index, summarise, caption) runs via FastAPI `BackgroundTasks` —
  no Celery/Redis for v1. `pipeline/ingest.py`'s job functions are already
  shaped so a real queue could be dropped in later without an API change.
- **Isolation is enforced at every layer, redundantly on purpose.**
  Middleware header check → per-route dependency → Postgres RLS →
  Qdrant's `build_filter()` chokepoint. No single layer is trusted alone.
- **Non-fatal failure boundaries.** A summarisation or image-captioning
  failure doesn't fail the whole ingest job — chunk indexing (the
  retrieval-critical part) succeeding is what matters; everything else
  degrades gracefully and is logged.

---

## Testing

```bash
pytest tests/
```

## Known gaps

- No automated end-to-end test suite yet against real Postgres/Qdrant —
  the plan's verification checklist (upload → poll → search modes →
  `as_of` filtering → chat → delete → restart-mid-ingest recovery) still
  needs to be run manually against real infrastructure.
- `unstructured`'s handling of messy/nested document structure (multi-
  column layouts, complex tables) is weaker than LlamaParse's — accepted
  trade-off for on-prem capability, flagged as a later quality-tuning
  target, not a blocker.
