"""
db/models.py
------------
Postgres now holds document-level registry/lifecycle state only — per the
2026-08-21 architecture (see AvabodhAIDocumentRetrievalArchitecture.md and
IMPLEMENTATION_PLAN (2).md): chunk text, dense vectors, sparse vectors, and
per-chunk metadata all moved to Qdrant (pipeline/vector_store.py). There is
no ORM model for chunks here any more — DocumentChunk and pgvector are gone.

Tables:
1. DocumentSummaryOutput — Pydantic LLM output validator
2. Document               — document registry (was DocumentSummary; renamed,
                             no data to migrate — see IMPLEMENTATION_PLAN (2)
                             §2.1). Adds status/status_detail lifecycle.
3. ChatThread              — chat conversation threads
4. ChatMessage             — individual chat messages (embedding columns
                             removed — chat-message vectors now live in
                             Qdrant's avabodh_chat_messages collection)
5. LLMExecutionLog         — one row per LLM chat/vision call (pipeline/llm_log.py)
6. KBDocument / KBChatHistory — the SEPARATE /kb subsystem (api/routes/kb.py,
                             pipeline/kb_pipeline.py). Out of scope for this
                             migration — left untouched, do not modify
                             without a separate decision from Vijay.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Float,
    Index, ForeignKey, JSON, Boolean, Enum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic — LLM output validation
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummaryOutput(BaseModel):
    doc_name:         str   = Field(default="unknown")
    summary_text:     str   = Field(min_length=10)
    key_topics:       list[str] = Field(default_factory=list)
    page_count:       int   = Field(default=0, ge=0)
    chunk_count:      int   = Field(default=0, ge=0)
    source_path:      str   = Field(default="")
    language:         str   = Field(default="English")
    model_used:       str   = Field(default="")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("summary_text", mode="before")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        return " ".join(v.split()) if isinstance(v, str) else v

    @field_validator("doc_name", mode="before")
    @classmethod
    def sanitise_doc_name(cls, v: str) -> str:
        import os
        return os.path.basename(str(v)) if v else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# NEW — Pydantic schemas for metadata extraction (LLM structured output)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadataOutput(BaseModel):
    """
    Document-level metadata — extracted once per document via LLM,
    using structured output during/after the Reduce step.
    """
    title:                str  = Field(default="", description="Real document title, not filename")
    author:               Optional[str] = Field(default=None, description="Author if mentioned in document")
    document_type:        str  = Field(default="other",
                                       description="resume, research_paper, contract, report, invoice, manual, article, other")
    domain:                str  = Field(default="general",
                                       description="legal, technical, financial, academic, medical, general")
    detected_language:    str  = Field(default="English")
    key_entities:          list[str] = Field(default_factory=list, description="Organizations, people, locations mentioned")
    mentioned_dates:       list[str] = Field(default_factory=list, description="Any dates referenced in the document")
    target_audience:       str  = Field(default="general", description="technical, general, executive")
    sentiment:              str  = Field(default="neutral", description="positive, negative, neutral, critical")
    confidentiality_level: str  = Field(default="public", description="public, internal, confidential")

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v) -> str:
        return v.strip()[:512] if isinstance(v, str) else ""

    @field_validator("document_type", "domain", "target_audience", "sentiment", "confidentiality_level", mode="before")
    @classmethod
    def lowercase_enum(cls, v) -> str:
        return str(v).strip().lower() if v else "other"


class ChunkMetadataOutput(BaseModel):
    """
    Chunk-level metadata — extracted during the Map step alongside
    the existing summary, using the SAME LLM call (no extra cost).
    """
    summary:          str  = Field(min_length=1, description="Concise summary of this section")
    section_heading:  Optional[str] = Field(default=None, description="Section/heading this chunk likely belongs to")
    chunk_type:       str  = Field(default="paragraph", description="paragraph, table, list, heading, code")
    topic:             str  = Field(default="general", description="2-3 word topic label for this chunk")
    entities:           list[str] = Field(default_factory=list, description="Named entities mentioned in this chunk")
    contains_data:      bool = Field(default=False, description="True if chunk has numbers, statistics, or tabular data")
    confidence_score:   float = Field(default=0.8, ge=0.0, le=1.0, description="LLM confidence in this extraction")

    @field_validator("chunk_type", mode="before")
    @classmethod
    def validate_chunk_type(cls, v) -> str:
        allowed = {"paragraph", "table", "list", "heading", "code"}
        v = str(v).strip().lower() if v else "paragraph"
        return v if v in allowed else "paragraph"

    @field_validator("summary", "topic", mode="before")
    @classmethod
    def clean_text_field(cls, v) -> str:
        return " ".join(str(v).split()) if v else ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLAlchemy Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Document table — document-level registry + lifecycle + metadata.
#    Chunk data (text, dense/sparse vectors, filterable metadata) lives in
#    Qdrant now — see pipeline/vector_store.py — not in a Postgres table.
# ─────────────────────────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — REQUIRED on every row, set from the X-Tenant-ID
    # header at write time. Every read in this app must filter on this. ──
    tenant_id           = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation WITHIN a tenant — REQUIRED, set from
    # X-Org-Unit-ID header, same trust model as tenant_id. ALWAYS filtered
    # together with tenant_id, never alone (see composite indexes below —
    # a standalone org_unit_id match would leak across tenants if two
    # different companies happen to use the same department code). ──────
    org_unit_id          = Column(String(128), nullable=False, index=True)
    doc_name            = Column(String(512), nullable=False, index=True)
    summary_text        = Column(Text, nullable=False)
    key_topics          = Column(Text, nullable=True)
    page_count          = Column(Integer, default=0)
    chunk_count         = Column(Integer, default=0)
    source_path         = Column(String(1024), nullable=True)
    language            = Column(String(64), default="English")
    model_used          = Column(String(128), nullable=True)
    confidence          = Column(Float, nullable=True)
    doc_hash            = Column(String(64), nullable=True, index=True)
    avg_chunk_size      = Column(Integer, nullable=True)
    embedding_model     = Column(String(128), nullable=True)

    # ── Ingestion lifecycle (replaces the old embedding_status/
    # embedding_stored_at pair — those were chunk-embedding-specific;
    # this tracks the whole background job: extract → chunk → embed →
    # index into Qdrant, per pipeline/ingest.py). ─────────────────────────
    status          = Column(String(16), nullable=False, default="UPLOADED", index=True)
    # UPLOADED | PROCESSING | READY | FAILED
    status_detail   = Column(Text, nullable=True)   # error message when status=FAILED
    processed_at    = Column(DateTime(timezone=True), nullable=True)
    file_type       = Column(String(16), nullable=True)   # extension, e.g. "pdf"
    file_size       = Column(Integer, nullable=True)
    source          = Column(String(16), nullable=True)   # "upload" | "web"
    stored_path     = Column(String(1024), nullable=True)  # file the ingest job reads

    # ── Client-supplied metadata — full payload kept here (authoritative);
    # allowlisted keys (FILTERABLE_METADATA_KEYS, pipeline/retriever.py)
    # also get copied into Qdrant's chunk payloads as meta_<key> fields so
    # search filtering never needs a Postgres lookup. ─────────────────────
    metadata_json   = Column("metadata", JSONB, nullable=True)

    # ── NEW — Document-level extracted metadata ───────────────────────────────
    title                  = Column(String(512), nullable=True, index=True)
    author                 = Column(String(256), nullable=True)
    document_type          = Column(String(64), nullable=True, index=True)   # resume/contract/report/etc
    domain                 = Column(String(64), nullable=True, index=True)   # legal/technical/financial/etc
    detected_language      = Column(String(64), nullable=True)
    key_entities            = Column(ARRAY(String), nullable=True)            # ["Avik Bhattacharya", "IIT Bombay"]
    mentioned_dates         = Column(ARRAY(String), nullable=True)
    target_audience         = Column(String(64), nullable=True)
    sentiment                = Column(String(32), nullable=True)
    confidentiality_level   = Column(String(32), nullable=True, default="public")
    metadata_status          = Column(String(32), default="pending")          # pending/completed/failed
    metadata_extracted_at   = Column(DateTime(timezone=True), nullable=True)

    # ── Image tracking ────────────────────────────────────────────────────────
    image_count             = Column(Integer, default=0)   # how many images extracted and embedded

    # ── NEW — client-supplied document fields (Upload Knowledge Document form) ──
    category       = Column(String(128), nullable=True, index=True)   # client-defined, free text: "Policy", "Guide"
    # effective_from/effective_to: authoritative copy here; also duplicated
    # onto every Qdrant chunk payload so `as_of` temporal filtering happens
    # directly during retrieval (architecture doc §11/§12/§27 Rule 7) — no
    # longer metadata-only as it was under the old pgvector design.
    effective_from = Column(DateTime(timezone=True), nullable=True)
    effective_to   = Column(DateTime(timezone=True), nullable=True)
    # is_ground_truth: 2026-08-21 — downgraded from a hard retrieval gate to
    # an OPTIONAL filter (IMPLEMENTATION_PLAN (2).md decisions table). A
    # search/chat request now opts into is_ground_truth-only results rather
    # than every query being silently restricted to ground-truth documents.
    is_ground_truth = Column(Boolean, nullable=False, default=False)

    created_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc),
                                 onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_documents_source_path", "source_path"),
        Index("ix_documents_doc_type_domain", "document_type", "domain"),
        # Tenant-scoped dedup lookups (check_duplicate / same_name upsert
        # in pipeline/storage.py) hit these directly.
        Index("ix_documents_tenant_hash", "tenant_id", "doc_hash"),
        Index("ix_documents_tenant_name", "tenant_id", "doc_name"),
        # org_unit_id is a hard boundary WITHIN a tenant, so every lookup
        # that used to be (tenant_id, X) becomes (tenant_id, org_unit_id,
        # X) — never org_unit_id alone (see column comment).
        Index("ix_documents_tenant_org_hash", "tenant_id", "org_unit_id", "doc_hash"),
        Index("ix_documents_tenant_org_name", "tenant_id", "org_unit_id", "doc_name"),
        Index("ix_documents_tenant_org_status", "tenant_id", "org_unit_id", "status"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. ChatThread
# ─────────────────────────────────────────────────────────────────────────────

class ChatThread(Base):
    __tablename__ = "chat_threads"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — same rule as documents: every thread belongs to
    # exactly one tenant, set from X-Tenant-ID when the thread is created. ──
    tenant_id  = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation, same rule as documents. ─────────────
    org_unit_id = Column(String(128), nullable=False, index=True)
    title      = Column(String(512), nullable=True)
    user_id    = Column(String(256), nullable=True)
    doc_filter = Column(String(512), nullable=True)
    message_count = Column(Integer, default=0)

    # ── Rolling conversation summary (pipeline/memory.py) ───────────────
    # Incrementally-updated summary of every turn that has aged OUT of the
    # recency window (MEMORY_WINDOW_SIZE turns, still sent verbatim). This
    # is what lets the chat prompt "remember" a long thread's earlier
    # topics without sending the whole transcript on every request.
    # rolling_summary_through counts how many ChatMessage rows (chronological,
    # human+ai both counted) are already folded into rolling_summary, so
    # pipeline/chat_storage.py::update_rolling_summary() only ever
    # summarises the NEW slice of aged-out messages, never re-summarising
    # ones already folded in.
    rolling_summary         = Column(Text, nullable=True)
    rolling_summary_through = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="thread",
                            cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")

    __table_args__ = (
        Index("ix_chat_threads_user_id", "user_id"),
        # Thread list/get/patch/delete all filter WHERE tenant_id = ...
        # AND org_unit_id = ... together.
        Index("ix_chat_threads_tenant_org_user", "tenant_id", "org_unit_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatThread id={self.id} title={self.title}>"


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChatMessage
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — denormalized from the parent ChatThread. Matters
    # most for the /chat/search semantic-search-over-history endpoint,
    # which queries chat_messages directly and must not cross tenants. ────
    tenant_id = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation, denormalized same as tenant_id. ─────
    org_unit_id = Column(String(128), nullable=False, index=True)
    thread_id = Column(UUID(as_uuid=True),
                       ForeignKey("chat_threads.id", ondelete="CASCADE"),
                       nullable=False, index=True)

    role      = Column(String(16), nullable=False)
    content   = Column(Text, nullable=False)

    # embedding/embedding_model columns removed 2026-08-21 — chat-message
    # vectors now live in Qdrant's avabodh_chat_messages collection
    # (pipeline/vector_store.py), point id = this row's id. GET /chat/search
    # queries Qdrant directly instead of this table's old pgvector column.

    sources   = Column(JSON, nullable=True)

    # ── Retrieval degradation, internal only (2026-09-23) ───────────────
    # Which components silently fell back to a degraded path while
    # answering THIS turn — e.g. ["sparse_embedding"] (hybrid search ran
    # dense-only, no keyword matching) or ["reranking"] (results were not
    # re-sorted by relevance). None/empty when nothing degraded.
    # Deliberately NOT on ChatMessageResponse or any API surface — this
    # exists so degraded answers can be found and explained after the
    # fact (e.g. "show me every turn from last week where reranking
    # failed"), not to be shown to end users, who have no useful action to
    # take on "the keyword search leg failed." See
    # pipeline/retriever.py::search()'s degraded out-param, which is what
    # actually observes each fallback as it happens.
    degraded  = Column(JSON, nullable=True)

    # ── NEW — multimodal chat (image attached to this turn) ────────────────
    has_image     = Column(Boolean, nullable=False, default=False)
    image_caption = Column(Text, nullable=True)   # GPT-4o Vision caption of the attached image

    prompt_tokens     = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)

    thread = relationship("ChatThread", back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_thread_role", "thread_id", "role"),
        # GET /chat/search filters WHERE tenant_id = ... AND org_unit_id = ...
        Index("ix_chat_messages_tenant_org", "tenant_id", "org_unit_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage role={self.role} thread={self.thread_id}>"


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLMExecutionLog — one row per chat-completion / vision call
# ─────────────────────────────────────────────────────────────────────────────

class LLMExecutionLog(Base):
    """
    Written by pipeline/llm_log.py, read via GET /llm-logs. `id` is also
    the execution_id of the optional Kafka event, so a consumer's copy can
    be matched back to this row. client_id is the caller's X-Client-ID
    header, recorded as sent (not authenticated).
    """
    __tablename__ = "llm_execution_logs"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id   = Column(String(128), nullable=False)
    org_unit_id = Column(String(128), nullable=False)
    client_id   = Column(String(128), nullable=True)
    request_id  = Column(String(128), nullable=True)

    purpose     = Column(String(64), nullable=False, index=True)
    provider    = Column(String(32), nullable=False)
    model_name  = Column(String(128), nullable=False)
    status      = Column(String(32), nullable=False)
    error_message = Column(Text, nullable=True)

    system_prompt = Column(Text, nullable=True)
    user_prompt   = Column(Text, nullable=True)
    response_text = Column(Text, nullable=True)

    prompt_tokens     = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens      = Column(Integer, nullable=True)
    execution_time_ms = Column(Integer, nullable=False)

    entity_type = Column(String(32), nullable=True)
    entity_id   = Column(String(128), nullable=True)

    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("ix_llm_execution_logs_tenant_org_created",
              "tenant_id", "org_unit_id", created_at.desc()),
        Index("ix_llm_execution_logs_entity", "entity_type", "entity_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Knowledge Base (KB) Tables — SEPARATE subsystem, out of scope, untouched
# ─────────────────────────────────────────────────────────────────────────────

class KBDocument(Base):
    __tablename__ = "kb_documents"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id      = Column(String(256), nullable=False, index=True)
    file_name     = Column(String(512), nullable=False)
    status        = Column(
        Enum("processing", "ready", "failed", name="kb_document_status"),
        nullable=False,
        default="processing",
    )
    error_message = Column(Text, nullable=True)
    created_at    = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc), nullable=False)


class KBChatHistory(Base):
    __tablename__ = "kb_chat_history"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(256), nullable=False, index=True)
    owner_id        = Column(String(256), nullable=False, index=True)
    role            = Column(
        Enum("human", "ai", name="kb_chat_role"),
        nullable=False,
    )
    content         = Column(Text, nullable=False)
    created_at      = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc), nullable=False)
