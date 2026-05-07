"""Simple API key authentication middleware for MCP HTTP."""

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_405_METHOD_NOT_ALLOWED

from langsmith_mcp_server.monitoring import set_monitoring_session_id

logger = logging.getLogger(__name__)

# Context variables to store LangSmith config for current request
# These are used to pass config from middleware to FastMCP's session state
api_key_context: ContextVar[str] = ContextVar("api_key", default="")
workspace_id_context: ContextVar[str] = ContextVar("workspace_id", default="")
endpoint_context: ContextVar[str] = ContextVar("endpoint", default="")


def get_api_key() -> str:
    """Get the API key from the current request context."""
    return api_key_context.get("")


@dataclass
class SessionCredentials:
    api_key: str
    workspace_id: str
    endpoint: str
    created_at: float


# In-memory store mapping mcp-session-id → credentials from the initialize request.
# MCP clients send custom headers only on the first request; subsequent requests
# include only the mcp-session-id header.
_session_store: dict[str, SessionCredentials] = {}

# Sessions older than this are evicted on next cleanup pass (24 hours)
_SESSION_TTL_SECONDS = 86400


def _evict_expired_sessions() -> None:
    """Remove sessions that have exceeded their TTL."""
    now = time.monotonic()
    expired = [
        sid
        for sid, creds in _session_store.items()
        if now - creds.created_at > _SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _session_store[sid]


def remove_session(session_id: str) -> None:
    """Remove a session's stored credentials (e.g. on session teardown)."""
    _session_store.pop(session_id, None)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts LANGSMITH-API-KEY header and optional config headers.

    On the first request (initialize), the client sends LANGSMITH-API-KEY and the
    server responds with an mcp-session-id. This middleware persists the credentials
    keyed by that session ID so that subsequent requests (which only carry
    mcp-session-id) can authenticate without re-sending the API key.
    """

    # MCP endpoint only accepts POST (and DELETE for session teardown); GET returns 405
    _MCP_PATH = "/mcp"
    _MCP_ALLOWED_METHODS = frozenset({"POST", "DELETE"})

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for health check and OAuth discovery (we use API key, not OAuth)
        if request.url.path in ("/health", "/.well-known/oauth-authorization-server"):
            return await call_next(request)

        # GET (e.g. opening /mcp in a browser) → 405 Method Not Allowed (standard for MCP HTTP)
        path = request.url.path.rstrip("/") or "/"
        if path == self._MCP_PATH and request.method not in self._MCP_ALLOWED_METHODS:
            return JSONResponse(
                status_code=HTTP_405_METHOD_NOT_ALLOWED,
                content={"error": "Method Not Allowed", "allow": list(self._MCP_ALLOWED_METHODS)},
                headers={"Allow": ", ".join(sorted(self._MCP_ALLOWED_METHODS))},
            )

        # Try to resolve credentials: from request headers first, then session store
        api_key = request.headers.get("LANGSMITH-API-KEY")
        workspace_id = request.headers.get("LANGSMITH-WORKSPACE-ID", "")
        endpoint = request.headers.get("LANGSMITH-ENDPOINT", "")

        request_session_id = request.headers.get("mcp-session-id")

        if not api_key and request_session_id:
            # Look up credentials from a previously initialized session
            stored = _session_store.get(request_session_id)
            if stored:
                api_key = stored.api_key
                workspace_id = workspace_id or stored.workspace_id
                endpoint = endpoint or stored.endpoint
                logger.debug("Resolved credentials from session %s", request_session_id)

        if not api_key:
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"error": "Missing LANGSMITH-API-KEY header"},
            )

        # Handle session teardown — clean up stored credentials
        if request.method == "DELETE" and request_session_id:
            remove_session(request_session_id)

        # Store in request state and context variables
        request.state.api_key = api_key
        request.state.workspace_id = workspace_id
        request.state.endpoint = endpoint

        api_key_context.set(api_key)
        workspace_id_context.set(workspace_id)
        endpoint_context.set(endpoint)

        # Session/thread id for monitoring: from headers or per-request id
        session_id = (
            request_session_id
            or request.headers.get("x-session-id")
            or request.headers.get("x-request-id")
            or str(uuid4())
        )
        set_monitoring_session_id(session_id)

        try:
            response = await call_next(request)

            # After the response is generated, persist credentials keyed by the
            # mcp-session-id returned in the response (set by the MCP transport
            # on the initialize response). Also store if the request already had
            # a session ID and provided fresh credentials (allows key rotation).
            response_session_id = response.headers.get("mcp-session-id")
            effective_session_id = response_session_id or request_session_id
            if effective_session_id and api_key:
                # Only store/update if we got the key from the request headers
                # (not from a session lookup) OR the response is issuing a new session
                has_fresh_key = bool(request.headers.get("LANGSMITH-API-KEY"))
                if has_fresh_key or response_session_id:
                    _session_store[effective_session_id] = SessionCredentials(
                        api_key=api_key,
                        workspace_id=workspace_id,
                        endpoint=endpoint,
                        created_at=time.monotonic(),
                    )
                    logger.debug("Stored credentials for session %s", effective_session_id)

            # Periodically evict expired sessions (cheap check)
            if len(_session_store) > 100:
                _evict_expired_sessions()

            return response
        finally:
            # Clear context after request
            api_key_context.set("")
            workspace_id_context.set("")
            endpoint_context.set("")
