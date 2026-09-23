"""
tests/test_webhook.py
----------------------
pipeline/webhook.py's one hard contract: it must NEVER raise, regardless
of what goes wrong, because it runs inside the ingestion background task
and a raised exception there would corrupt ingestion's own error handling
(pipeline/ingest.py's except/finally blocks are written assuming only
ingestion-related calls can fail there). Every test below either asserts
correct behavior on the happy path, or asserts "no exception propagates"
on a failure path — that second category is the one worth being paranoid
about, since a regression there is invisible until it takes ingestion
down with it in production.
"""

import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pipeline import webhook
from config.settings import get_settings


@pytest.fixture
def configured_settings(monkeypatch):
    """A settings object with the webhook actually turned on."""
    get_settings.cache_clear()
    monkeypatch.setenv("CORE_WEBHOOK_URL", "http://core.test/v1/knowledge-base/webhooks/status")
    monkeypatch.setenv("CORE_WEBHOOK_SECRET", "test-secret-123")
    monkeypatch.setenv("CORE_WEBHOOK_TIMEOUT_SECONDS", "5.0")
    settings = get_settings()
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def unconfigured_settings(monkeypatch):
    """A settings object with the webhook left off (the default, supported state)."""
    get_settings.cache_clear()
    monkeypatch.setenv("CORE_WEBHOOK_URL", "")
    yield get_settings()
    get_settings.cache_clear()


class TestNoOpWhenUnconfigured:
    def test_does_not_call_httpx_when_url_unset(self, unconfigured_settings):
        with patch("pipeline.webhook.httpx.post") as post:
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "READY")
            post.assert_not_called()

    def test_returns_none_and_does_not_raise(self, unconfigured_settings):
        result = webhook.notify_status_change("doc-1", "tenant-1", "org-1", "FAILED", status_detail="boom")
        assert result is None


class TestHappyPath:
    def test_sends_correct_payload_and_header(self, configured_settings):
        with patch("pipeline.webhook.httpx.post") as post:
            post.return_value = MagicMock(status_code=200)

            webhook.notify_status_change("doc-42", "tenant-9", "org-7", "READY")

            assert post.call_count == 1
            args, kwargs = post.call_args
            assert args[0] == "http://core.test/v1/knowledge-base/webhooks/status"
            assert kwargs["json"] == {
                "document_id": "doc-42",
                "tenant_id": "tenant-9",
                "org_unit_id": "org-7",
                "status": "READY",
                "status_detail": None,
            }
            assert kwargs["headers"] == {"X-Webhook-Secret": "test-secret-123"}
            assert kwargs["timeout"] == 5.0

    def test_includes_status_detail_when_given(self, configured_settings):
        with patch("pipeline.webhook.httpx.post") as post:
            post.return_value = MagicMock(status_code=200)

            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "FAILED", status_detail="extraction failed")

            _, kwargs = post.call_args
            assert kwargs["json"]["status_detail"] == "extraction failed"

    def test_document_id_is_stringified(self, configured_settings):
        """document_id can arrive as a UUID object from call sites — must serialize to str for JSON."""
        with patch("pipeline.webhook.httpx.post") as post:
            post.return_value = MagicMock(status_code=200)
            doc_id = uuid.uuid4()

            webhook.notify_status_change(doc_id, "tenant-1", "org-1", "PROCESSING")

            _, kwargs = post.call_args
            assert kwargs["json"]["document_id"] == str(doc_id)
            assert isinstance(kwargs["json"]["document_id"], str)

    def test_logs_attempt_and_successful_delivery(self, configured_settings, caplog):
        """Every emission must be visible in Avabodh's own logs — both the
        attempt (so you can see a webhook was even tried) and confirmed
        delivery (so you can tell it actually reached Core, not just that
        it was sent)."""
        with caplog.at_level("INFO", logger="pipeline.webhook"):
            with patch("pipeline.webhook.httpx.post") as post:
                post.return_value = MagicMock(status_code=200)
                webhook.notify_status_change("doc-42", "tenant-9", "org-7", "READY")

        messages = [r.message for r in caplog.records]
        assert any("Sending Core webhook" in m and "doc-42" in m and "READY" in m for m in messages)
        assert any("Core webhook delivered" in m and "doc-42" in m for m in messages)

    def test_does_not_log_delivered_on_4xx_response(self, configured_settings, caplog):
        with caplog.at_level("INFO", logger="pipeline.webhook"):
            with patch("pipeline.webhook.httpx.post") as post:
                post.return_value = MagicMock(status_code=403, text="bad secret")
                webhook.notify_status_change("doc-42", "tenant-9", "org-7", "READY")

        messages = [r.message for r in caplog.records]
        assert not any("Core webhook delivered" in m for m in messages)


class TestInvalidStatusRejected:
    def test_unknown_status_is_not_sent(self, configured_settings):
        with patch("pipeline.webhook.httpx.post") as post:
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "UPLOADED")
            post.assert_not_called()

    def test_unknown_status_does_not_raise(self, configured_settings):
        webhook.notify_status_change("doc-1", "tenant-1", "org-1", "not-a-real-status")

    @pytest.mark.parametrize("status", ["PROCESSING", "READY", "FAILED"])
    def test_all_three_real_statuses_are_accepted(self, configured_settings, status):
        with patch("pipeline.webhook.httpx.post") as post:
            post.return_value = MagicMock(status_code=200)

            webhook.notify_status_change("doc-1", "tenant-1", "org-1", status)

            post.assert_called_once()


class TestFailureModesNeverRaise:
    """The paranoid section — this is the actual contract worth guarding."""

    def test_timeout_does_not_raise(self, configured_settings):
        with patch("pipeline.webhook.httpx.post", side_effect=httpx.TimeoutException("timed out")):
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "READY")

    def test_connect_error_does_not_raise(self, configured_settings):
        with patch("pipeline.webhook.httpx.post", side_effect=httpx.ConnectError("connection refused")):
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "READY")

    def test_generic_http_error_does_not_raise(self, configured_settings):
        with patch("pipeline.webhook.httpx.post", side_effect=httpx.HTTPError("generic failure")):
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "FAILED", status_detail="x")

    def test_completely_unexpected_exception_does_not_raise(self, configured_settings):
        """DNS failures, TLS errors, etc. don't always surface as httpx.HTTPError subclasses."""
        with patch("pipeline.webhook.httpx.post", side_effect=OSError("network unreachable")):
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "PROCESSING")

    def test_4xx_response_does_not_raise(self, configured_settings):
        """Wrong secret, unknown document on Core's side, etc. — Core returns 4xx, not a transport error."""
        with patch("pipeline.webhook.httpx.post") as post:
            post.return_value = MagicMock(status_code=403, text="Forbidden: bad secret")
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "READY")

    def test_5xx_response_does_not_raise(self, configured_settings):
        with patch("pipeline.webhook.httpx.post") as post:
            post.return_value = MagicMock(status_code=500, text="Internal Server Error")
            webhook.notify_status_change("doc-1", "tenant-1", "org-1", "READY")

    def test_all_failure_modes_return_none(self, configured_settings):
        with patch("pipeline.webhook.httpx.post", side_effect=httpx.TimeoutException("x")):
            result = webhook.notify_status_change("doc-1", "tenant-1", "org-1", "READY")
            assert result is None
