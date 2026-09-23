"""
api/schemas/chat.py
-------------------
Pydantic schemas for all chat endpoints.

NEW: ChatMessageRequest now accepts an optional base64-encoded image
     for multimodal chat (user attaches image + text query together).
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    """
    Validated on every incoming message.

    NEW — image support:
      image_base64  : base64-encoded image bytes (any format)
      image_media_type : "image/jpeg" | "image/png" | "image/webp" | "image/gif"

    When image_base64 is provided:
      - GPT-4o Vision captions the image
      - Caption is embedded and used for pgvector similarity search
      - Final LLM call receives BOTH the image AND retrieved context
        (GPT-4o literally sees the image pixels, not just the caption)
    """
    query:      str  = Field(min_length=1, max_length=4000, description="User question")
    thread_id:  Optional[UUID] = Field(default=None, description="Existing thread ID — null for new thread")
    doc_filter: Optional[str]  = Field(default=None, max_length=512, description="Limit search to specific document name")
    top_k:      int  = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")

    # ── NEW — optional image attachment ──────────────────────────────────────
    image_base64:     Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded image bytes. When provided, GPT-4o Vision "
            "sees the actual image pixels and uses retrieved document "
            "context to answer about it."
        ),
    )
    image_media_type: Optional[str] = Field(
        default="image/jpeg",
        description="MIME type of the attached image: image/jpeg | image/png | image/webp | image/gif",
    )

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("image_media_type", mode="before")
    @classmethod
    def validate_media_type(cls, v) -> str:
        allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if v and v not in allowed:
            return "image/jpeg"
        return v or "image/jpeg"


class ThreadCreateRequest(BaseModel):
    title:      Optional[str] = Field(default=None, max_length=512)
    user_id:    Optional[str] = Field(default=None, max_length=256)
    doc_filter: Optional[str] = Field(default=None, max_length=512)


class ThreadUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    """One source chunk used in AI response — can be text or image chunk."""
    doc_name:    str
    chunk_index: int
    chunk_text:  Optional[str]  = None
    similarity:  Optional[float] = None
    role:        Optional[str]  = "text"   # NEW — "text" or "image"
    image_type:  Optional[str]  = None     # NEW — chart/diagram/photo etc (image chunks only)
    image_url:   Optional[str]  = None     # NEW — original image URL (image chunks only)
    page_number:     Optional[int] = None   # NEW 2026-08-21 — which page this chunk came from
    section_heading: Optional[str] = None   # NEW 2026-08-21 — which document section this chunk came from
    table_html:      Optional[str] = None   # NEW 2026-08-21 — original <table> markup, when this source is a table
    quoted_text:     Optional[str] = None   # NEW 2026-09-23 — verbatim sentence the LLM cited from this source
    # (pipeline/memory.py's QUOTED EVIDENCE prompt rule; None when the model
    # skipped it, or the quote failed verbatim verification against the
    # source's own text — see api/routes/chat.py::_verified_quote_for_chunk).


class ChatMessageResponse(BaseModel):
    message_id:   UUID
    thread_id:    UUID
    role:         str  = "ai"
    content:      str  = Field(min_length=1)
    sources:      list[SourceReference] = Field(default_factory=list)
    thread_title: Optional[str] = None
    created_at:   datetime
    # NEW — True when user sent an image and GPT-4o Vision was used
    image_understood: Optional[bool] = None
    # NEW — GPT-4o Vision's caption of the attached image, if any
    image_caption:    Optional[str]  = None

    @field_validator("content", mode="before")
    @classmethod
    def clean_content(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("human", "ai", "system"):
            raise ValueError("role must be human, ai or system")
        return v

    class Config:
        from_attributes = True


class ThreadResponse(BaseModel):
    id:            UUID
    title:         Optional[str]
    user_id:       Optional[str]
    doc_filter:    Optional[str]
    message_count: int
    created_at:    datetime
    updated_at:    Optional[datetime]
    tenant_id:     Optional[str] = None   # which tenant this thread belongs to
    org_unit_id:   Optional[str] = None   # NEW — which department, within that tenant

    class Config:
        from_attributes = True


class ThreadListResponse(BaseModel):
    total:   int
    threads: list[ThreadResponse]


class MessageListResponse(BaseModel):
    thread_id: UUID
    total:     int
    messages:  list[dict]


class DeleteResponse(BaseModel):
    id:      UUID
    message: str = "Deleted successfully"