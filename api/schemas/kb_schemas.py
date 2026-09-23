from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from uuid import UUID


class KBUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kbDocumentId: UUID
    status: str


class KBStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kbDocumentId: UUID
    status: str
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class KBRetrieveRequest(BaseModel):
    owner_id: str
    query: str
    filters: Optional[Dict[str, Any]] = None


class KBChatRequest(BaseModel):
    owner_id: str
    message: str
    conversation_id: Optional[str] = None


class KBSourceReference(BaseModel):
    document_id: Optional[str] = None
    doc_name: str
    file_name: Optional[str] = None
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_text: Optional[str] = None
    source_path: Optional[str] = None


class KBChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    sources: List[KBSourceReference] = Field(default_factory=list)
