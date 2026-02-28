from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_trace_id() -> str:
    return str(uuid4())


def generate_anonymous_user_id(trace_id: str | None = None) -> str:
    if trace_id:
        return f"anon-{trace_id[:12]}"
    return f"anon-{uuid4().hex[:12]}"
