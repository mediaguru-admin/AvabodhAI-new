"""
tests/test_internal_evidence.py
--------------------------------
api/routes/internal.py::retrieve_evidence — the endpoint replacing MDS's
old direct SQL against document_chunks/document_summaries (see that
route's module docstring). Mocks vector_store.search so these tests
don't need real Qdrant, matching tests/test_api.py's TestClient pattern.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from db.database import get_db_session_fastapi

client = TestClient(app)
HEADERS = {"X-Tenant-ID": "test-tenant", "X-Org-Unit-ID": "test-org"}
EMBEDDING = [0.1] * 1536


@pytest.fixture(autouse=True)
def _override_db_session():
    app.dependency_overrides[get_db_session_fastapi] = lambda: MagicMock()
    yield
    app.dependency_overrides.pop(get_db_session_fastapi, None)


def _payload(**overrides):
    body = {"query_embedding": EMBEDDING, "top_k": 5, "similarity_threshold": 0.6}
    body.update(overrides)
    return body


def test_missing_tenant_header_rejected():
    resp = client.post("/internal/retrieve-evidence", json=_payload(), headers={"X-Org-Unit-ID": "test-org"})
    assert resp.status_code == 400


def test_wrong_embedding_dimension_rejected():
    resp = client.post(
        "/internal/retrieve-evidence",
        json=_payload(query_embedding=[0.1] * 10),
        headers=HEADERS,
    )
    assert resp.status_code == 422


def test_filters_to_ground_truth_and_as_of():
    """The whole point of this endpoint: reproduce the old SQL's
    is_ground_truth=TRUE + effective_from/effective_to window, now via
    build_filter()'s is_ground_truth + as_of params."""
    with patch("api.routes.internal.build_filter") as mock_build_filter, \
         patch("api.routes.internal.vector_store.search", return_value=[]):
        mock_build_filter.return_value = MagicMock()
        resp = client.post("/internal/retrieve-evidence", json=_payload(), headers=HEADERS)

    assert resp.status_code == 200
    _, kwargs = mock_build_filter.call_args
    assert kwargs["is_ground_truth"] is True
    assert kwargs["as_of"] is not None
    assert kwargs["tenant_id"] == "test-tenant"
    assert kwargs["org_unit_id"] == "test-org"


def test_missing_query_embedding_rejected():
    body = _payload()
    del body["query_embedding"]
    resp = client.post("/internal/retrieve-evidence", json=body, headers=HEADERS)
    assert resp.status_code == 422


def test_metadata_filter_forwarded_to_build_filter():
    with patch("api.routes.internal.build_filter") as mock_build_filter, \
         patch("api.routes.internal.vector_store.search", return_value=[]):
        mock_build_filter.return_value = MagicMock()
        client.post(
            "/internal/retrieve-evidence",
            json=_payload(metadata={"country": ["India"]}),
            headers=HEADERS,
        )
    _, kwargs = mock_build_filter.call_args
    assert kwargs["metadata"] == {"country": ["India"]}


def test_unknown_metadata_key_rejected_with_422():
    with patch("api.routes.internal.build_filter", side_effect=ValueError("'bogus' is not a filterable metadata key")):
        resp = client.post(
            "/internal/retrieve-evidence",
            json=_payload(metadata={"bogus": ["x"]}),
            headers=HEADERS,
        )
    assert resp.status_code == 422


def test_org_ids_widens_scope():
    with patch("api.routes.internal.build_filter") as mock_build_filter, \
         patch("api.routes.internal.vector_store.search", return_value=[]):
        mock_build_filter.return_value = MagicMock()
        client.post(
            "/internal/retrieve-evidence",
            json=_payload(org_ids=["org-a", "org-b"]),
            headers=HEADERS,
        )
    _, kwargs = mock_build_filter.call_args
    assert kwargs["org_ids"] == ["org-a", "org-b"]


def test_search_is_vector_only_semantic_mode():
    with patch("api.routes.internal.build_filter", return_value=MagicMock()), \
         patch("api.routes.internal.vector_store.search", return_value=[]) as mock_search:
        client.post("/internal/retrieve-evidence", json=_payload(top_k=7, similarity_threshold=0.75), headers=HEADERS)

    _, kwargs = mock_search.call_args
    assert kwargs["mode"] == "semantic"
    assert kwargs["dense_vector"] == EMBEDDING
    assert kwargs["limit"] == 7
    assert kwargs["score_threshold"] == 0.75


def test_response_shape_maps_id_and_score():
    fake_hits = [
        {"id": "chunk-1", "chunk_text": "evidence one", "score": 0.91},
        {"id": "chunk-2", "chunk_text": "evidence two", "score": 0.83},
    ]
    with patch("api.routes.internal.build_filter", return_value=MagicMock()), \
         patch("api.routes.internal.vector_store.search", return_value=fake_hits):
        resp = client.post("/internal/retrieve-evidence", json=_payload(), headers=HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == [
        {"chunk_id": "chunk-1", "chunk_text": "evidence one", "similarity_score": 0.91},
        {"chunk_id": "chunk-2", "chunk_text": "evidence two", "similarity_score": 0.83},
    ]


def test_image_chunk_match_uses_caption_not_blank_text():
    """An image-role chunk has no chunk_text — its content lives in
    image_caption. A match on one is real evidence, not something to
    silently return as an empty string."""
    fake_hits = [
        {"id": "chunk-img", "role": "image", "image_caption": "Bar chart showing 40% increase in Q3 revenue", "score": 0.88},
    ]
    with patch("api.routes.internal.build_filter", return_value=MagicMock()), \
         patch("api.routes.internal.vector_store.search", return_value=fake_hits):
        resp = client.post("/internal/retrieve-evidence", json=_payload(), headers=HEADERS)

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["chunk_text"] == "Bar chart showing 40% increase in Q3 revenue"


def test_search_failure_returns_500_not_raw_exception():
    with patch("api.routes.internal.build_filter", return_value=MagicMock()), \
         patch("api.routes.internal.vector_store.search", side_effect=RuntimeError("qdrant down")):
        resp = client.post("/internal/retrieve-evidence", json=_payload(), headers=HEADERS)
    assert resp.status_code == 500
