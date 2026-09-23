from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_tenant_id
from pipeline import attribute_store

router = APIRouter()


def optional_org_unit_id(x_org_unit_id: Optional[str] = Header(default=None, alias="X-Org-Unit-ID")) -> Optional[str]:
    return (x_org_unit_id or "").strip() or None


class AttributeDefinitionRequest(BaseModel):
    key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    description: str = Field(min_length=1, max_length=2000)
    type: str = Field(default="choice", pattern=r"^(choice|score|noul)$")
    allowed_values: list[str] = Field(default_factory=list, max_length=100)
    active: bool = True


@router.get("")
def get_attribute_definitions(
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: Optional[str] = Depends(optional_org_unit_id),
):
    return {"definitions": attribute_store.list_definitions(tenant_id, org_unit_id)}


@router.put("/{key}")
def put_attribute_definition(
    key: str,
    request: AttributeDefinitionRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: Optional[str] = Depends(optional_org_unit_id),
):
    if key != request.key:
        raise HTTPException(status_code=400, detail="Path key and body key must match")
    try:
        return attribute_store.upsert_definition(tenant_id, org_unit_id, request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Attribute store unavailable: {exc}")


@router.delete("/{key}", status_code=204)
def remove_attribute_definition(
    key: str,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: Optional[str] = Depends(optional_org_unit_id),
):
    try:
        attribute_store.delete_definition(tenant_id, org_unit_id, key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Attribute store unavailable: {exc}")


@router.get("/chunks/{chunk_id}")
def get_chunk_attributes(
    chunk_id: str,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: Optional[str] = Depends(optional_org_unit_id),
):
    try:
        row = attribute_store.get_chunk_attributes(tenant_id, org_unit_id, chunk_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Attribute store unavailable: {exc}")
    return row or {"chunk_id": chunk_id, "attributes": {}}
