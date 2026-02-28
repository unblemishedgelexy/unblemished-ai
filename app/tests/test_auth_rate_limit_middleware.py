from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.core.auth_middleware import AuthRateLimitMiddleware
from app.core.config import Settings
from app.core.logger import StructuredLogger, setup_logger


def test_rate_limit_blocks_after_threshold() -> None:
    middleware = _build_middleware(
        settings=Settings(
            auth_enabled=False,
            rate_limit_enabled=True,
            requests_per_minute=1,
        ),
    )

    first = asyncio.run(
        middleware.dispatch(
            _build_request(
                path="/v1/chat/reason",
                method="POST",
                body={"input_text": "hello", "user_id": "rate-user-1"},
            ),
            _allow_request,
        ),
    )
    second = asyncio.run(
        middleware.dispatch(
            _build_request(
                path="/v1/chat/reason",
                method="POST",
                body={"input_text": "hello", "user_id": "rate-user-1"},
            ),
            _allow_request,
        ),
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert json.loads(second.body.decode("utf-8"))["detail"] == "rate_limit_exceeded"


def test_auth_required_when_enabled() -> None:
    middleware = _build_middleware(
        settings=Settings(
            auth_enabled=True,
            auth_mode="api_key",
            auth_api_key_header="x-api-key",
            auth_api_key="super-secret-key",
            rate_limit_enabled=False,
            requests_per_minute=10,
        ),
    )

    unauthorized = asyncio.run(
        middleware.dispatch(
            _build_request(
                path="/v1/chat/reason",
                method="POST",
                body={"input_text": "secure request"},
            ),
            _allow_request,
        ),
    )
    authorized = asyncio.run(
        middleware.dispatch(
            _build_request(
                path="/v1/chat/reason",
                method="POST",
                body={"input_text": "secure request"},
                headers={"x-api-key": "super-secret-key"},
            ),
            _allow_request,
        ),
    )
    # Non-chat path should bypass auth middleware checks.
    health = asyncio.run(
        middleware.dispatch(
            _build_request(path="/health", method="GET"),
            _allow_request,
        ),
    )

    assert unauthorized.status_code == 401
    assert json.loads(unauthorized.body.decode("utf-8"))["detail"].startswith("missing_api_key_header")
    assert authorized.status_code == 200
    assert health.status_code == 200


def test_system_endpoints_are_auth_protected() -> None:
    middleware = _build_middleware(
        settings=Settings(
            auth_enabled=True,
            auth_mode="api_key",
            auth_api_key_header="x-api-key",
            auth_api_key="super-secret-key",
            rate_limit_enabled=False,
            requests_per_minute=10,
        ),
    )

    unauthorized = asyncio.run(
        middleware.dispatch(
            _build_request(
                path="/v1/system/status",
                method="GET",
            ),
            _allow_request,
        ),
    )
    authorized = asyncio.run(
        middleware.dispatch(
            _build_request(
                path="/v1/system/status",
                method="GET",
                headers={"x-api-key": "super-secret-key"},
            ),
            _allow_request,
        ),
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def _build_middleware(*, settings: Settings) -> AuthRateLimitMiddleware:
    logger = StructuredLogger(setup_logger(level="INFO"))
    return AuthRateLimitMiddleware(app=FastAPI(), settings=settings, logger=logger)


def _build_request(
    *,
    path: str,
    method: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
) -> Request:
    raw_body = b""
    header_payload: list[tuple[bytes, bytes]] = []
    if headers:
        header_payload.extend((key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items())
    if body is not None:
        raw_body = json.dumps(body).encode("utf-8")
        header_payload.append((b"content-type", b"application/json"))

    consumed = False

    async def receive() -> dict[str, Any]:
        nonlocal consumed
        if consumed:
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed = True
        return {"type": "http.request", "body": raw_body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": header_payload,
        "client": (client_host, 40000),
        "server": ("testserver", 80),
    }
    return Request(scope=scope, receive=receive)


async def _allow_request(request: Request) -> Response:
    return JSONResponse({"ok": True}, status_code=200)
