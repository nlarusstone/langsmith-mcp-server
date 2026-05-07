"""Tests for session credential persistence in the middleware."""

import time
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from langsmith_mcp_server.middleware import (
    APIKeyMiddleware,
    SessionCredentials,
    _session_store,
    remove_session,
)


def _make_app(include_session_header: bool = True):
    """Create a test Starlette app with the APIKeyMiddleware.

    If include_session_header is True, the /mcp endpoint echoes back an
    mcp-session-id response header (simulating the MCP transport behavior
    on the initialize response).
    """

    async def mcp_endpoint(request: Request) -> Response:
        headers = {}
        if include_session_header:
            headers["mcp-session-id"] = "test-session-123"
        return JSONResponse(
            {"api_key_seen": request.state.api_key},
            headers=headers,
        )

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["POST", "DELETE"]),
            Route("/health", health, methods=["GET"]),
        ],
        middleware=[Middleware(APIKeyMiddleware)],
    )
    return app


@pytest.fixture(autouse=True)
def _clear_session_store():
    """Clear the session store before and after each test."""
    _session_store.clear()
    yield
    _session_store.clear()


class TestSessionCredentialPersistence:
    """Test that credentials are stored on initialize and reused on subsequent requests."""

    def test_first_request_stores_credentials(self):
        client = TestClient(_make_app())
        resp = client.post(
            "/mcp",
            headers={"LANGSMITH-API-KEY": "lsv2_pt_abc123"},
        )
        assert resp.status_code == 200
        assert "test-session-123" in _session_store
        assert _session_store["test-session-123"].api_key == "lsv2_pt_abc123"

    def test_subsequent_request_uses_stored_credentials(self):
        client = TestClient(_make_app())
        # First request — provides the API key
        resp1 = client.post(
            "/mcp",
            headers={"LANGSMITH-API-KEY": "lsv2_pt_abc123"},
        )
        assert resp1.status_code == 200

        # Second request — only provides mcp-session-id
        resp2 = client.post(
            "/mcp",
            headers={"mcp-session-id": "test-session-123"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["api_key_seen"] == "lsv2_pt_abc123"

    def test_request_without_key_or_session_returns_401(self):
        client = TestClient(_make_app())
        resp = client.post("/mcp")
        assert resp.status_code == 401
        assert "Missing LANGSMITH-API-KEY" in resp.json()["error"]

    def test_request_with_invalid_session_returns_401(self):
        client = TestClient(_make_app())
        resp = client.post(
            "/mcp",
            headers={"mcp-session-id": "nonexistent-session"},
        )
        assert resp.status_code == 401

    def test_workspace_and_endpoint_persisted(self):
        client = TestClient(_make_app())
        client.post(
            "/mcp",
            headers={
                "LANGSMITH-API-KEY": "lsv2_pt_abc123",
                "LANGSMITH-WORKSPACE-ID": "ws-42",
                "LANGSMITH-ENDPOINT": "https://custom.langsmith.dev",
            },
        )
        stored = _session_store["test-session-123"]
        assert stored.workspace_id == "ws-42"
        assert stored.endpoint == "https://custom.langsmith.dev"

    def test_delete_removes_session(self):
        _session_store["sess-to-delete"] = SessionCredentials(
            api_key="lsv2_pt_x",
            workspace_id="",
            endpoint="",
            created_at=time.monotonic(),
        )
        client = TestClient(_make_app())
        resp = client.delete(
            "/mcp",
            headers={
                "LANGSMITH-API-KEY": "lsv2_pt_x",
                "mcp-session-id": "sess-to-delete",
            },
        )
        assert resp.status_code == 200
        assert "sess-to-delete" not in _session_store

    def test_fresh_key_updates_stored_credentials(self):
        """Sending a new API key with an existing session updates the store (key rotation)."""
        _session_store["test-session-123"] = SessionCredentials(
            api_key="old-key",
            workspace_id="",
            endpoint="",
            created_at=time.monotonic(),
        )
        client = TestClient(_make_app())
        client.post(
            "/mcp",
            headers={
                "LANGSMITH-API-KEY": "new-key",
                "mcp-session-id": "test-session-123",
            },
        )
        assert _session_store["test-session-123"].api_key == "new-key"

    def test_health_check_skips_auth(self):
        client = TestClient(_make_app())
        resp = client.get("/health")
        assert resp.status_code == 200


class TestSessionEviction:
    """Test that expired sessions are cleaned up."""

    def test_expired_sessions_evicted(self):
        old_time = time.monotonic() - 90000  # > 24h ago
        for i in range(105):
            _session_store[f"session-{i}"] = SessionCredentials(
                api_key=f"key-{i}",
                workspace_id="",
                endpoint="",
                created_at=old_time if i < 50 else time.monotonic(),
            )

        client = TestClient(_make_app())
        # This request triggers eviction (store > 100 entries)
        client.post(
            "/mcp",
            headers={"LANGSMITH-API-KEY": "lsv2_pt_trigger"},
        )
        # Old sessions should be evicted
        assert "session-0" not in _session_store
        # Recent sessions remain
        assert "session-99" in _session_store


class TestRemoveSession:
    """Test the remove_session helper."""

    def test_remove_existing(self):
        _session_store["x"] = SessionCredentials(
            api_key="k", workspace_id="", endpoint="", created_at=time.monotonic()
        )
        remove_session("x")
        assert "x" not in _session_store

    def test_remove_nonexistent_is_noop(self):
        remove_session("does-not-exist")
