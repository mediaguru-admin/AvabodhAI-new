"""
pipeline/llm_log.py
-------------------
LLM execution log — one row per chat-completion / vision call.

How calls get here:
  - LangChain models (ChatOpenAI / ChatOllama) are built with
    `callbacks=[LLM_LOG_CALLBACK], metadata={"llm_purpose": ...}` — see
    llm_log_kwargs(). The callback sees prompt, usage, model, latency and
    errors for every invoke, including with_structured_output chains.
  - The two raw `openai` SDK calls (multimodal chat, vision captioning)
    call record_openai_sdk_call() themselves.

Who the call is for (tenant, org unit, X-Client-ID, request id, the
document/thread it belongs to) lives in a ContextVar, set by
TenantGuardMiddleware per request and by ingestion / chat code for the
entity. ThreadPoolExecutor does NOT copy contextvars — submit via
contextvars.copy_context().run when the worker makes LLM calls.

Each row goes to llm_execution_logs (GET /llm-logs) and, when
LLM_LOG_KAFKA_ENABLED, is also published as an llm.execution.logged.v1
event (clariona-core's envelope). Recording never raises: a logging
failure must never fail the LLM call it describes.
"""

import json
import re
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

EVENT_TYPE = "llm.execution.logged.v1"
_DATA_URL_RE = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")

_ctx: ContextVar[dict] = ContextVar("llm_log_ctx", default={})


def set_llm_context(**fields: Any) -> None:
    """Merge fields (tenant_id, org_unit_id, client_id, request_id,
    entity_type, entity_id) into the current context. None values are ignored."""
    _ctx.set({**_ctx.get(), **{k: str(v) for k, v in fields.items() if v is not None}})


def get_llm_context() -> dict:
    return dict(_ctx.get())


def llm_log_kwargs(purpose: str) -> dict:
    """Constructor kwargs that make a LangChain chat model log its calls."""
    return {"callbacks": [LLM_LOG_CALLBACK], "metadata": {"llm_purpose": purpose}}


# ── Prompt rendering ─────────────────────────────────────────────────────────

def _cap(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = _DATA_URL_RE.sub("[image]", text)
    limit = settings.LLM_LOG_MAX_TEXT_CHARS
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def _content_text(content: Any) -> str:
    """Message content is a string or a list of multimodal parts."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
        else:
            parts.append("[image]")
    return "\n".join(parts)


def _split_prompts(messages: list) -> tuple[Optional[str], Optional[str]]:
    """(system_prompt, user_prompt) from LangChain messages or OpenAI dicts."""
    system, other = [], []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else m.type
        content = m.get("content") if isinstance(m, dict) else m.content
        text = _content_text(content)
        if role == "system":
            system.append(text)
        else:
            other.append((role, text))
    user = other[0][1] if len(other) == 1 else "\n\n".join(f"[{r}] {t}" for r, t in other)
    return ("\n\n".join(system) or None), (user or None)


def _status_for(error: BaseException) -> str:
    import openai
    if isinstance(error, openai.APITimeoutError):
        return "timeout"
    if isinstance(error, openai.RateLimitError):
        return "rate_limited"
    return "failed"


# ── Recording ────────────────────────────────────────────────────────────────

def record_llm_call(
    *,
    purpose: str,
    provider: str,
    model_name: str,
    messages: list,
    elapsed_ms: int,
    response_text: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    error: Optional[BaseException] = None,
    ctx: Optional[dict] = None,
) -> None:
    """Persist one LLM call (+ optional Kafka event). Never raises."""
    if not settings.LLM_LOG_ENABLED:
        return
    try:
        ctx = ctx if ctx is not None else get_llm_context()
        if not ctx.get("tenant_id"):
            logger.debug("LLM log skipped — no tenant in context (purpose=%s)", purpose)
            return
        system_prompt, user_prompt = _split_prompts(messages)
        total = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens is not None else None
        row = {
            "id": uuid.uuid4(),
            "tenant_id": ctx["tenant_id"],
            "org_unit_id": ctx.get("org_unit_id") or "none",
            "client_id": ctx.get("client_id"),
            "request_id": ctx.get("request_id"),
            "purpose": purpose,
            "provider": provider,
            "model_name": model_name or "unknown",
            "status": _status_for(error) if error else "success",
            "error_message": _cap(str(error)) if error else None,
            "system_prompt": _cap(system_prompt),
            "user_prompt": _cap(user_prompt),
            "response_text": _cap(response_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "execution_time_ms": max(int(elapsed_ms), 0),
            "entity_type": ctx.get("entity_type"),
            "entity_id": ctx.get("entity_id"),
            "created_at": datetime.now(timezone.utc),
        }
    except Exception as e:
        logger.warning("LLM log build failed (purpose=%s): %s", purpose, e)
        return

    try:
        _insert_row(row)
    except Exception as e:
        logger.warning("LLM log insert failed (purpose=%s): %s", purpose, e)

    if settings.LLM_LOG_KAFKA_ENABLED:
        try:
            _publish(build_event(row))
        except Exception as e:
            logger.warning("LLM log Kafka publish failed (purpose=%s): %s", purpose, e)


def record_openai_sdk_call(
    *,
    purpose: str,
    model_name: str,
    messages: list,
    started: float,
    response: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    """record_llm_call for a raw `openai` SDK chat.completions response.
    `started` is a time.monotonic() taken just before the call."""
    usage = getattr(response, "usage", None)
    text = None
    if response is not None and getattr(response, "choices", None):
        text = response.choices[0].message.content
    record_llm_call(
        purpose=purpose,
        provider="openai",
        model_name=getattr(response, "model", None) or model_name,
        messages=messages,
        elapsed_ms=(time.monotonic() - started) * 1000,
        response_text=text,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        error=error,
    )


def _insert_row(row: dict) -> None:
    from db.database import get_db_session_context
    from db.models import LLMExecutionLog

    with get_db_session_context(row["tenant_id"], row["org_unit_id"]) as session:
        session.add(LLMExecutionLog(**row))


# ── Kafka (optional) ─────────────────────────────────────────────────────────

def build_event(row: dict) -> dict:
    """llm.execution.logged.v1 envelope — same shape clariona-core's
    src/llm_execution_log/event_builder.py produces and its consumer stores."""
    occurred = row["created_at"].replace(microsecond=0).isoformat().replace("+00:00", "Z")
    execution_id = str(row["id"])
    return {
        "event_id": execution_id,
        "event_type": EVENT_TYPE,
        "event_version": 1,
        "tenant_id": row["tenant_id"],
        "domain": "llm",
        "producer": "avabodh-ai",
        "correlation_id": row["request_id"],
        "occurred_at": occurred,
        "effective_at": occurred,
        "replayable": False,
        "data": {
            "execution_id": execution_id,
            "purpose": row["purpose"],
            "llm_provider": row["provider"],
            "model_name": row["model_name"],
            "system_prompt_sent": row["system_prompt"],
            "user_prompt_sent": row["user_prompt"],
            "response_text": row["response_text"],
            "execution_time_ms": row["execution_time_ms"],
            "status": row["status"],
            "error_message": row["error_message"],
            "error_code": None,
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["total_tokens"],
            "estimated_cost_usd": None,
            "template_code": None,
            "template_id": None,
            "template_version": None,
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "executed_by": None,
            "request_parameters": {
                "source": "avabodh-ai",
                "client_id": row["client_id"],
                "org_unit_id": row["org_unit_id"],
                "request_id": row["request_id"],
            },
            "response_json": None,
            "response_parsed_successfully": None,
            "validation_errors": None,
            "variables_used": None,
        },
        "meta": {"schema_uri": "internal://events/llm/llm.execution.logged.v1"},
    }


_producer = None
_producer_lock = threading.Lock()


def _get_producer():
    global _producer
    if _producer is None:
        with _producer_lock:
            if _producer is None:
                from confluent_kafka import Producer
                _producer = Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})
    return _producer


def _on_delivery(err, msg) -> None:
    if err is not None:
        logger.warning("LLM log Kafka delivery failed: %s", err)


def _publish(event: dict) -> None:
    producer = _get_producer()
    producer.produce(
        settings.LLM_LOG_KAFKA_TOPIC_TEMPLATE.format(tenant_id=event["tenant_id"]),
        key=event["tenant_id"].encode("utf-8"),
        value=json.dumps(event, default=str).encode("utf-8"),
        on_delivery=_on_delivery,
    )
    producer.poll(0)


def flush_llm_log_producer(timeout: float = 5.0) -> None:
    if _producer is not None:
        _producer.flush(timeout)


# ── LangChain callback ───────────────────────────────────────────────────────

class LLMLogCallback(BaseCallbackHandler):
    """Records every LangChain chat-model call it is attached to. The context
    is snapshotted at start — end/error can arrive on another thread."""

    def __init__(self) -> None:
        self._runs: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._runs[run_id] = (time.monotonic(), messages[0] if messages else [], metadata or {}, get_llm_context())

    def _finish(self, run_id, response=None, error=None) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        started, messages, md, ctx = run
        try:
            gen = response.generations[0][0] if response and response.generations and response.generations[0] else None
            msg = getattr(gen, "message", None)
            usage = getattr(msg, "usage_metadata", None) or {}
            text = gen.text if gen else None
            if not text and getattr(msg, "tool_calls", None):
                # with_structured_output answers via a tool call, not text
                text = json.dumps([tc.get("args") for tc in msg.tool_calls], default=str)
            model = (getattr(msg, "response_metadata", None) or {}).get("model_name") or md.get("ls_model_name")
        except Exception as e:
            logger.warning("LLM log could not read response: %s", e)
            usage, text, model = {}, None, md.get("ls_model_name")
        record_llm_call(
            purpose=md.get("llm_purpose") or "unspecified",
            provider="openai" if md.get("ls_provider") == "openai" else "other",
            model_name=model,
            messages=messages,
            elapsed_ms=(time.monotonic() - started) * 1000,
            response_text=text,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            error=error,
            ctx=ctx,
        )

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._finish(run_id, response=response)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._finish(run_id, error=error)


LLM_LOG_CALLBACK = LLMLogCallback()
