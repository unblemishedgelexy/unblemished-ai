from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Settings
from app.core.logger import StructuredLogger

_PROTECTED_PREFIXES = ("/v1/chat", "/v1/system")


@dataclass(slots=True)
class _AuthResult:
    allowed: bool
    subject: str | None
    status_code: int
    error: str | None


@dataclass(slots=True)
class _RateDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Applies auth + rate limiting on protected API endpoints.

    JWT mode is intentionally future-ready: this middleware validates token shape and basic claims,
    but does not perform signature verification.
    """

    def __init__(
        self,
        app: Any,
        *,
        settings: Settings,
        logger: StructuredLogger,
    ) -> None:
        super().__init__(app)
        self._settings = settings
        self._logger = logger
        self._limiter = _InMemoryRateLimiter(requests_per_minute=settings.requests_per_minute)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not _is_protected_path(request.url.path):
            return await call_next(request)

        auth_subject: str | None = None
        if self._settings.auth_enabled:
            auth_result = await self._authenticate(request)
            if not auth_result.allowed:
                self._logger.warning(
                    "auth.denied",
                    trace_id="middleware",
                    user_id="unknown",
                    memory_id="n/a",
                    retrieval_count=0,
                    path=request.url.path,
                    mode=self._settings.auth_mode,
                    reason=auth_result.error or "unauthorized",
                )
                return JSONResponse(
                    status_code=auth_result.status_code,
                    content={"detail": auth_result.error or "unauthorized"},
                )
            auth_subject = auth_result.subject
            request.state.auth_subject = auth_subject

        rate_key = ""
        rate_decision = _RateDecision(allowed=True, remaining=self._settings.requests_per_minute, retry_after_seconds=0)
        if self._settings.rate_limit_enabled:
            rate_key = await self._resolve_rate_key(request=request, auth_subject=auth_subject)
            rate_decision = await self._limiter.evaluate(rate_key)
            if not rate_decision.allowed:
                self._logger.warning(
                    "rate_limit.blocked",
                    trace_id="middleware",
                    user_id=rate_key,
                    memory_id="n/a",
                    retrieval_count=0,
                    path=request.url.path,
                    retry_after_seconds=rate_decision.retry_after_seconds,
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate_limit_exceeded"},
                    headers={
                        "Retry-After": str(rate_decision.retry_after_seconds),
                        "X-RateLimit-Limit": str(self._settings.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        response = await call_next(request)
        if self._settings.rate_limit_enabled:
            response.headers["X-RateLimit-Limit"] = str(self._settings.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(rate_decision.remaining)
            if rate_key:
                response.headers["X-RateLimit-Key"] = rate_key
        return response

    async def _authenticate(self, request: Request) -> _AuthResult:
        if self._settings.auth_mode == "jwt":
            return await self._authenticate_jwt(request)
        return await self._authenticate_api_key(request)

    async def _authenticate_api_key(self, request: Request) -> _AuthResult:
        header_name = self._settings.auth_api_key_header
        provided = request.headers.get(header_name)
        if not provided:
            return _AuthResult(
                allowed=False,
                subject=None,
                status_code=401,
                error=f"missing_api_key_header:{header_name}",
            )
        if provided != self._settings.auth_api_key:
            return _AuthResult(
                allowed=False,
                subject=None,
                status_code=401,
                error="invalid_api_key",
            )
        return _AuthResult(
            allowed=True,
            subject="api-key-user",
            status_code=200,
            error=None,
        )

    async def _authenticate_jwt(self, request: Request) -> _AuthResult:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return _AuthResult(
                allowed=False,
                subject=None,
                status_code=401,
                error="missing_bearer_token",
            )

        token = auth_header.split(" ", 1)[1].strip()
        payload = _decode_unverified_jwt_payload(token)
        if payload is None:
            return _AuthResult(
                allowed=False,
                subject=None,
                status_code=401,
                error="invalid_jwt_format",
            )

        expires_at = payload.get("exp")
        if isinstance(expires_at, (int, float)):
            now_ts = datetime.now(timezone.utc).timestamp()
            if float(expires_at) < now_ts:
                return _AuthResult(
                    allowed=False,
                    subject=None,
                    status_code=401,
                    error="jwt_expired",
                )

        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            subject = "jwt-user"
        return _AuthResult(
            allowed=True,
            subject=subject,
            status_code=200,
            error=None,
        )

    async def _resolve_rate_key(self, *, request: Request, auth_subject: str | None) -> str:
        request_user_id = await _extract_user_id_from_body(request)
        if request_user_id:
            return f"user:{request_user_id}"
        if auth_subject:
            return f"subject:{auth_subject}"

        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            ip = forwarded_for.split(",", 1)[0].strip()
            if ip:
                return f"ip:{ip}"
        if request.client and request.client.host:
            return f"ip:{request.client.host}"
        return "ip:unknown"


class _InMemoryRateLimiter:
    def __init__(self, *, requests_per_minute: int) -> None:
        self._requests_per_minute = requests_per_minute
        self._window_seconds = 60.0
        self._buckets: dict[str, deque[float]] = {}
        self._clock = time.monotonic
        self._mutex = asyncio.Lock()

    async def evaluate(self, key: str) -> _RateDecision:
        async with self._mutex:
            now = self._clock()
            bucket = self._buckets.setdefault(key, deque())
            self._prune(bucket=bucket, now=now)

            if len(bucket) >= self._requests_per_minute:
                retry_after = int(max(1, math.ceil(self._window_seconds - (now - bucket[0]))))
                return _RateDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            bucket.append(now)
            remaining = max(0, self._requests_per_minute - len(bucket))
            return _RateDecision(
                allowed=True,
                remaining=remaining,
                retry_after_seconds=0,
            )

    def _prune(self, *, bucket: deque[float], now: float) -> None:
        while bucket and (now - bucket[0]) >= self._window_seconds:
            bucket.popleft()


async def _extract_user_id_from_body(request: Request) -> str | None:
    if request.method.upper() not in {"POST", "PUT", "PATCH"}:
        return None
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return None
    raw_body = await request.body()
    if not raw_body:
        return None
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    user_id = payload.get("user_id")
    if not isinstance(user_id, str):
        return None
    cleaned = user_id.strip()
    return cleaned or None


def _decode_unverified_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_protected_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES)
