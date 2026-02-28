from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, constr


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input_text: constr(min_length=1, strip_whitespace=True)
    context: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    user_id: constr(min_length=1, strip_whitespace=True) | None = None
