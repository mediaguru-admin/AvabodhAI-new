"""Mongo-backed attribute definitions and Jev chunk extraction.

The module is deliberately fail-closed: Mongo/Jev failures never invent
attributes and never make an upload fail. Definitions are tenant-owned; an
organisation definition overrides a tenant definition with the same key.
"""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import requests
from openai import OpenAI
from pymongo import ASCENDING, MongoClient

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _client() -> MongoClient:
    client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=2000)
    db = client[settings.MONGODB_DATABASE]
    db.attribute_definitions.create_index(
        [("tenant_id", ASCENDING), ("org_unit_id", ASCENDING), ("key", ASCENDING)], unique=True
    )
    db.chunk_attributes.create_index(
        [("tenant_id", ASCENDING), ("org_unit_id", ASCENDING), ("document_id", ASCENDING), ("chunk_id", ASCENDING)],
        unique=True,
    )
    db.chunk_attributes.create_index([("tenant_id", ASCENDING), ("org_unit_id", ASCENDING), ("attributes", ASCENDING)])
    return client


def _db():
    return _client()[settings.MONGODB_DATABASE]


def list_definitions(tenant_id: str, org_unit_id: str | None = None) -> list[dict[str, Any]]:
    """Return effective definitions, with organisation values overriding tenant values."""
    try:
        rows = list(_db().attribute_definitions.find({
            "tenant_id": tenant_id,
            "org_unit_id": {"$in": [None, "", org_unit_id]},
            "active": True,
        }))
    except Exception as exc:
        logger.warning("Attribute definitions unavailable: %s", exc)
        return []
    effective: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("key", "")).strip()
        if not key:
            continue
        row = _normalise_definition(row)
        is_org = bool(org_unit_id and row.get("org_unit_id") == org_unit_id)
        if key not in effective or is_org:
            row.pop("_id", None)
            effective[key] = row
    return list(effective.values())


def upsert_definition(tenant_id: str, org_unit_id: str | None, definition: dict[str, Any]) -> dict[str, Any]:
    definition = _normalise_definition(definition)
    key = str(definition["key"]).strip()
    document = {
        "tenant_id": tenant_id,
        "org_unit_id": org_unit_id or None,
        "key": key,
        "description": str(definition["description"]).strip(),
        "type": definition.get("type", "choice"),
        "allowed_values": definition.get("allowed_values", []),
        "active": bool(definition.get("active", True)),
        "updated_at": datetime.now(timezone.utc),
    }
    _db().attribute_definitions.replace_one(
        {"tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "key": key}, document, upsert=True
    )
    document.pop("_id", None)
    return document


def _normalise_definition(definition: dict[str, Any]) -> dict[str, Any]:
    """Allow the UI's description-only definitions to carry a typed contract."""
    result = dict(definition)
    description = str(result.get("description", ""))
    type_match = re.search(r"(?:type|Type)\s*:\s*(choice|score|noul)", description, re.IGNORECASE)
    kind = str(result.get("type") or (type_match.group(1).lower() if type_match else "noul")).lower()
    result["type"] = {"choice": "choice", "score": "score", "noul": "noul"}.get(kind, "noul")
    if result["type"] == "choice" and not result.get("allowed_values"):
        result["allowed_values"] = re.findall(r"(?:^|\s)-\s*([a-zA-Z][\w-]*)\s*:", description)
    if result["type"] == "score" and not result.get("allowed_values"):
        scale = re.search(r"Scale\s*\(\s*1\s*to\s*(\d+)\s*\)", description, re.IGNORECASE)
        result["allowed_values"] = list(range(1, int(scale.group(1)) + 1)) if scale else [1, 2, 3, 4]
    return result


def delete_definition(tenant_id: str, org_unit_id: str | None, key: str) -> None:
    _db().attribute_definitions.update_one(
        {"tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "key": key},
        {"$set": {"active": False, "updated_at": datetime.now(timezone.utc)}},
    )


def get_chunk_attributes(tenant_id: str, org_unit_id: str | None, chunk_id: str) -> dict[str, Any] | None:
    row = _db().chunk_attributes.find_one({
        "tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "chunk_id": chunk_id,
    })
    if row:
        row.pop("_id", None)
    return row

def get_document_attributes(tenant_id: str, org_unit_id: str | None, document_id: str) -> dict[str, Any]:
    """Flatten successfully extracted chunk attributes for status/UI consumers."""
    try:
        rows = _db().chunk_attributes.find({"tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "document_id": document_id})
        collected: dict[str, list[Any]] = {}
        for row in rows:
            for key, value in (row.get("attributes") or {}).items():
                values = collected.setdefault(key, [])
                candidate = value.get("value") if isinstance(value, dict) else value
                existing = [item.get("value") if isinstance(item, dict) else item for item in values]
                if candidate not in existing:
                    values.append(value)
        return {key: values[0] if len(values) == 1 else values for key, values in collected.items()}
    except Exception as exc:
        logger.warning("Could not read document attributes: %s", exc)
        return {}


def _questions(definitions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    questions = {}
    for d in definitions:
        kind = d.get("type", "choice")
        if kind == "choice":
            options = [str(x) for x in d.get("allowed_values", [])]
            if "__NOT_FOUND__" not in options:
                options.append("__NOT_FOUND__")
            if not options or len(options) < 2:
                continue
            questions[d["key"]] = {"type": "choice", "options": options, "description": d["description"]}
        elif kind == "score":
            criteria = [str(x) for x in d.get("allowed_values", [])]
            if criteria:
                questions[d["key"]] = {"type": "score", "criteria": criteria, "description": d["description"]}
        elif kind == "noul":
            questions[d["key"]] = {"type": "noul", "description": d["description"]}
    return questions


def _validate_answers(answers: Any, definitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(answers, dict):
        raise ValueError("Attribute model returned a non-object JSON response")
    valid = {}
    definitions_by_key = {d["key"]: d for d in definitions}
    for key, answer in answers.items():
        if key not in definitions_by_key:
            continue
        definition = definitions_by_key[key]
        candidates = answer if isinstance(answer, list) else [answer]
        accepted: list[dict[str, Any]] = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                value = candidate.get("value")
                confidence = float(candidate.get("confidence", candidate.get("probability", 1.0)))
            else:
                value, confidence = candidate, 1.0
            if value in (None, "", "__NOT_FOUND__") or confidence < settings.JEV_MIN_CONFIDENCE:
                continue
            if definition.get("type") == "choice":
                options = definition.get("allowed_values", [])
                normalized = str(value).strip().lower()
                match = next((option for option in options if str(option).lower() == normalized), None)
                if match is None:
                    continue
                value = match
            if not any(existing["value"] == value for existing in accepted):
                accepted.append({"value": value, "confidence": confidence})
        if accepted:
            valid[key] = accepted if len(accepted) > 1 else accepted[0]
    return valid


def _extract_with_jev(chunk_text: str, definitions: list[dict[str, Any]]) -> dict[str, Any]:
    questions = _questions(definitions)
    response = requests.post(
        settings.JEV_API_URL,
        headers={"Authorization": f"Bearer {settings.JEV_API_KEY}", "Content-Type": "application/json"},
        json={"state": chunk_text, "questions": questions},
        timeout=settings.JEV_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return _validate_answers(payload.get("answers", payload.get("results", payload.get("decisions", payload))), definitions)


def _extract_with_openai(chunk_text: str, definitions: list[dict[str, Any]]) -> dict[str, Any]:
    client = OpenAI(api_key=settings.OPENAI_ATTRIBUTE_API_KEY)
    schema = {d["key"]: {"value": "value from text or __NOT_FOUND__", "confidence": 0.0} for d in definitions}
    prompt = "Return ONLY JSON matching this exact object shape. Use __NOT_FOUND__ when absent and confidence 0..1. If a field has multiple distinct values in the text, return an array of {value, confidence} objects for that field.\n" + json.dumps(schema) + "\nDefinitions:\n" + json.dumps(definitions, default=str) + "\nText:\n" + chunk_text
    result = client.chat.completions.create(model=settings.CHAT_MODEL, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
    return _validate_answers(json.loads(result.choices[0].message.content or "{}"), definitions)


def _extract_with_ollama(chunk_text: str, definitions: list[dict[str, Any]]) -> dict[str, Any]:
    schema = {d["key"]: {"value": "value from text or __NOT_FOUND__", "confidence": 0.0} for d in definitions}
    prompt = "Return ONLY valid JSON matching this exact object shape. Use __NOT_FOUND__ when absent and confidence 0..1. If a field has multiple distinct values in the text, return an array of {value, confidence} objects for that field.\n" + json.dumps(schema) + "\nDefinitions:\n" + json.dumps(definitions, default=str) + "\nText:\n" + chunk_text
    response = requests.post(
        f"{settings.ollama_url.rstrip('/')}/api/chat",
        json={"model": settings.OLLAMA_CHAT_MODEL, "messages": [{"role": "user", "content": prompt}], "format": "json", "stream": False},
        timeout=settings.LLM_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    content = (payload.get("message") or {}).get("content", "{}")
    return _validate_answers(json.loads(content), definitions)


def extract_chunk_attributes(chunk_text: str, definitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Use Jev when available; fall back to OpenAI with the same JSON contract."""
    if not definitions:
        return {}
    if settings.JEV_API_KEY:
        try:
            return _extract_with_jev(chunk_text, definitions)
        except Exception as exc:
            logger.warning("Jev extraction unavailable; falling back to OpenAI: %s", exc)
    if settings.OPENAI_ATTRIBUTE_API_KEY:
        try:
            return _extract_with_openai(chunk_text, definitions)
        except Exception as exc:
            logger.warning("OpenAI attribute extraction failed: %s", exc)
    if settings.use_ollama:
        # Small local models occasionally return an empty/partially invalid
        # JSON object. Retry a few times so a transient response cannot make
        # the completed upload look as if no attributes were extracted.
        for attempt in range(3):
            try:
                answers = _extract_with_ollama(chunk_text, definitions)
                if answers:
                    return answers
            except Exception as exc:
                logger.warning("Ollama attribute extraction failed (attempt %s): %s", attempt + 1, exc)
    return {}


def extract_and_store_chunks(chunks, tenant_id: str, org_unit_id: str | None, document_id: str) -> int:
    definitions = list_definitions(tenant_id, org_unit_id)
    try:
        _db().chunk_attributes.delete_many({
            "tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "document_id": document_id,
        })
    except Exception as exc:
        logger.warning("Could not clear prior attributes for document=%s: %s", document_id, exc)
    if not definitions or (not settings.JEV_API_KEY and not settings.OPENAI_ATTRIBUTE_API_KEY and not settings.use_ollama):
        return 0
    stored = 0
    collection = _db().chunk_attributes
    from pipeline.vector_store import chunk_point_id
    for chunk in chunks:
        try:
            attrs = extract_chunk_attributes(getattr(chunk, "page_content", ""), definitions)
            if not attrs:
                continue
            chunk_index = int(chunk.metadata.get("chunk_index", 0))
            chunk_id = chunk_point_id(document_id, "text", chunk_index)
            collection.replace_one(
                {"tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "document_id": document_id, "chunk_id": chunk_id},
                {"tenant_id": tenant_id, "org_unit_id": org_unit_id or None, "document_id": document_id,
                 "chunk_id": chunk_id, "chunk_index": chunk_index, "attributes": attrs,
                 "extractor": "jev", "created_at": datetime.now(timezone.utc)},
                upsert=True,
            )
            stored += 1
        except Exception as exc:
            logger.warning("Attribute extraction failed for document=%s chunk=%s: %s", document_id, chunk.metadata.get("chunk_index", "?"), exc)
    return stored
