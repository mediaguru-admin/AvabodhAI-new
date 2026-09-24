"""
tests/test_llm_log.py
---------------------
pipeline/llm_log.py: every LangChain / raw-SDK LLM call becomes one log row
(+ an llm.execution.logged.v1 event when Kafka is on) carrying the caller's
context, and recording can never break the call it describes.
"""

import contextvars
import threading
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pipeline import llm_log

# Same required-field checks clariona-core's consumer applies
# (core/src/llm_execution_log/consumer.py::validate_llm_execution_logged_event).
_CORE_STATUSES = {"success", "failed", "timeout", "rate_limited", "invalid_response"}


def _core_accepts(event: dict) -> bool:
    data = event.get("data") or {}
    return bool(
        event.get("event_id") and event.get("tenant_id")
        and event.get("event_type") == "llm.execution.logged.v1"
        and data.get("purpose") and data.get("llm_provider") and data.get("model_name")
        and data.get("status") in _CORE_STATUSES
        and data.get("execution_time_ms") is not None
    )


@pytest.fixture
def captured(monkeypatch):
    rows, events = [], []
    monkeypatch.setattr(llm_log, "_insert_row", rows.append)
    monkeypatch.setattr(llm_log, "_publish", events.append)
    monkeypatch.setattr(llm_log.settings, "LLM_LOG_ENABLED", True)
    monkeypatch.setattr(llm_log.settings, "LLM_LOG_KAFKA_ENABLED", True)
    monkeypatch.setattr(llm_log.settings, "LLM_LOG_MAX_TEXT_CHARS", 50000)
    return rows, events


def _run_in_fresh_context(fn):
    # Empty Context, not copy_context(): other tests (TestClient + the tenant
    # middleware) can leave LLM-log context set on the main thread.
    return contextvars.Context().run(fn)


class _FailingModel(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_langchain_call_logs_purpose_tokens_and_context(captured):
    rows, events = captured

    def run():
        llm_log.set_llm_context(tenant_id="t1", org_unit_id="o1", client_id="clariona-core",
                                request_id="req-1", entity_type="chat_thread", entity_id="th-1")
        model = GenericFakeChatModel(
            messages=iter([AIMessage(
                content="hello",
                usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                response_metadata={"model_name": "gpt-4o-mini"},
            )]),
            **llm_log.llm_log_kwargs("chat_answer"),
        )
        model.invoke([SystemMessage(content="be brief"), HumanMessage(content="hi")])

    _run_in_fresh_context(run)

    assert len(rows) == 1
    row = rows[0]
    assert row["purpose"] == "chat_answer"
    assert row["status"] == "success"
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 4, 15)
    assert row["model_name"] == "gpt-4o-mini"
    assert row["system_prompt"] == "be brief" and row["user_prompt"] == "hi"
    assert row["response_text"] == "hello"
    assert (row["tenant_id"], row["org_unit_id"], row["client_id"], row["request_id"]) == ("t1", "o1", "clariona-core", "req-1")
    assert (row["entity_type"], row["entity_id"]) == ("chat_thread", "th-1")

    assert len(events) == 1 and _core_accepts(events[0])
    assert events[0]["event_id"] == str(row["id"])
    assert events[0]["data"]["request_parameters"]["client_id"] == "clariona-core"


def test_langchain_error_is_logged_as_failed_and_still_raises(captured):
    rows, _ = captured

    def run():
        llm_log.set_llm_context(tenant_id="t1", org_unit_id="o1")
        model = _FailingModel(messages=iter([]), **llm_log.llm_log_kwargs("doc_summary"))
        with pytest.raises(RuntimeError):
            model.invoke("x")

    _run_in_fresh_context(run)
    assert rows[0]["status"] == "failed"
    assert rows[0]["purpose"] == "doc_summary"
    assert "boom" in rows[0]["error_message"]


def test_raw_sdk_call_strips_base64_images(captured):
    rows, _ = captured
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDREVGRw=="}},
        {"type": "text", "text": "describe data:image/png;base64,QUJDREVGRw=="},
    ]}]
    response = SimpleNamespace(
        model="gpt-4o",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"caption": "x"}'))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
    )

    def run():
        llm_log.set_llm_context(tenant_id="t1", org_unit_id="o1", entity_type="document", entity_id="d1")
        llm_log.record_openai_sdk_call(purpose="image_caption", model_name="gpt-4o",
                                       messages=messages, started=0.0, response=response)

    _run_in_fresh_context(run)
    row = rows[0]
    assert "base64" not in row["user_prompt"]
    assert row["user_prompt"] == "[image]\ndescribe [image]"
    assert (row["prompt_tokens"], row["total_tokens"]) == (100, 105)


def test_context_reaches_worker_threads_via_copy_context(captured):
    rows, _ = captured

    def run():
        llm_log.set_llm_context(tenant_id="t1", org_unit_id="o1", entity_id="d1")
        ctx = contextvars.copy_context()
        t = threading.Thread(target=ctx.run, args=(llm_log.record_llm_call,), kwargs=dict(
            purpose="doc_chunk_metadata", provider="openai", model_name="m",
            messages=[], elapsed_ms=1,
        ))
        t.start()
        t.join()

    _run_in_fresh_context(run)
    assert rows[0]["entity_id"] == "d1"


def test_no_tenant_skips_and_storage_failure_never_raises(captured, monkeypatch):
    rows, _ = captured
    _run_in_fresh_context(lambda: llm_log.record_llm_call(
        purpose="p", provider="openai", model_name="m", messages=[], elapsed_ms=1))
    assert rows == []

    def explode(_):
        raise RuntimeError("db down")
    monkeypatch.setattr(llm_log, "_insert_row", explode)
    monkeypatch.setattr(llm_log, "_publish", explode)

    def run():
        llm_log.set_llm_context(tenant_id="t1", org_unit_id="o1")
        llm_log.record_llm_call(purpose="p", provider="openai", model_name="m", messages=[], elapsed_ms=1)

    _run_in_fresh_context(run)  # must not raise
