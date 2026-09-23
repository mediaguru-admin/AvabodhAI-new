"""
pipeline/webhook.py
--------------------
Fire-and-forget status-change notification to Core, so Core learns a
document reached PROCESSING/READY/FAILED the moment it happens. Core has no
background polling anymore — this webhook is the primary path; a user
clicking "Check status" in tenant-ui is the only fallback (see
clariona-core's KBService.check_and_apply_status).

Hard rule: a failure here must NEVER propagate to the caller. This runs
inside the ingestion background task (pipeline/ingest.py) — a network
blip, a misconfigured URL, or a slow/down Core must not be able to affect
ingestion itself. Every failure mode is caught, logged, and swallowed.
"""

import httpx

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Statuses this module will actually send. Anything else is a programming
# error at the call site, not a value worth silently forwarding.
_VALID_STATUSES = {"PROCESSING", "READY", "FAILED"}


def notify_status_change(
    document_id: str,
    tenant_id: str,
    org_unit_id: str,
    status: str,
    status_detail: str | None = None,
) -> None:
    """
    Best-effort POST to settings.CORE_WEBHOOK_URL. Returns None always —
    there is nothing for a caller to do with success/failure here, since
    ingestion must proceed identically either way. Call sites should not
    wrap this in their own try/except; every exception is already handled.

    No-ops silently (does not even attempt the call) when CORE_WEBHOOK_URL
    is unset — this is the expected, supported state for any deployment
    that hasn't opted into the webhook yet, not a misconfiguration.
    """
    if status not in _VALID_STATUSES:
        logger.error(
            "webhook.notify_status_change called with invalid status %r for document %s — "
            "not sending (this is a caller bug, fix the call site, not this function)",
            status, document_id,
        )
        return

    settings = get_settings()
    if not settings.CORE_WEBHOOK_URL:
        return

    payload = {
        "document_id": str(document_id),
        "tenant_id": tenant_id,
        "org_unit_id": org_unit_id,
        "status": status,
        "status_detail": status_detail,
    }
    headers = {"X-Webhook-Secret": settings.CORE_WEBHOOK_SECRET}

    logger.info(
        "Sending Core webhook: document %s status=%s -> %s",
        document_id, status, settings.CORE_WEBHOOK_URL,
    )

    try:
        response = httpx.post(
            settings.CORE_WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=settings.CORE_WEBHOOK_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            # Not raised further — a 4xx/5xx from Core (bad secret, Core
            # temporarily down, unknown document) is exactly the situation
            # the manual "Check status" fallback exists for. Logged so it's
            # visible, not silent, but never fatal to ingestion.
            logger.warning(
                "Core webhook returned %d for document %s status=%s (user can still Check status manually): %s",
                response.status_code, document_id, status, response.text[:500],
            )
        else:
            logger.info(
                "Core webhook delivered: document %s status=%s (%d)",
                document_id, status, response.status_code,
            )
    except httpx.TimeoutException:
        logger.warning(
            "Core webhook timed out after %.1fs for document %s status=%s (user can still Check status manually)",
            settings.CORE_WEBHOOK_TIMEOUT_SECONDS, document_id, status,
        )
    except httpx.HTTPError as e:
        logger.warning(
            "Core webhook request failed for document %s status=%s (user can still Check status manually): %s",
            document_id, status, e,
        )
    except Exception as e:
        # Deliberately broad as the last line of defense — this function's
        # one hard contract is "never raise", and a webhook call is exactly
        # the kind of I/O that can fail in ways the two blocks above don't
        # anticipate (DNS resolution, TLS, etc.).
        logger.warning(
            "Unexpected error sending Core webhook for document %s status=%s (user can still Check status manually): %s",
            document_id, status, e,
        )
