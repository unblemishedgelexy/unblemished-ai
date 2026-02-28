from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, constr

from app.core.logger import StructuredLogger

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ContextDigestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context: dict[str, Any] = Field(default_factory=dict)


class KeywordExtractArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: constr(min_length=1, strip_whitespace=True)
    max_keywords: int = Field(default=5, ge=1, le=12)


class PriorityEstimateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: constr(min_length=1, strip_whitespace=True)
    priority_hint: str | None = None


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self, logger: StructuredLogger) -> None:
        self._logger = logger
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtin_tools()

    def is_registered(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition
        self._logger.info(
            "tool.registry.registered",
            trace_id="system",
            user_id="system",
            memory_id="n/a",
            retrieval_count=0,
            tool_name=definition.name,
        )

    def get_metadata(self, tool_name: str) -> dict[str, Any] | None:
        definition = self._tools.get(tool_name)
        if definition is None:
            return None
        return {
            "name": definition.name,
            "description": definition.description,
            "schema": definition.args_schema.model_json_schema(),
        }

    def list_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "schema": definition.args_schema.model_json_schema(),
            }
            for definition in self._tools.values()
        ]

    def validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        definition = self._tools.get(tool_name)
        if definition is None:
            raise KeyError(tool_name)
        try:
            validated = definition.args_schema.model_validate(arguments)
        except ValidationError:
            raise
        return validated.model_dump()

    def get_handler(self, tool_name: str) -> ToolHandler:
        definition = self._tools.get(tool_name)
        if definition is None:
            raise KeyError(tool_name)
        return definition.handler

    def _register_builtin_tools(self) -> None:
        self.register(
            ToolDefinition(
                name="context_digest",
                description="Summarize context keys and value types.",
                args_schema=ContextDigestArgs,
                handler=_context_digest_handler,
            ),
        )
        self.register(
            ToolDefinition(
                name="keyword_extract",
                description="Extract top keywords from text.",
                args_schema=KeywordExtractArgs,
                handler=_keyword_extract_handler,
            ),
        )
        self.register(
            ToolDefinition(
                name="priority_estimator",
                description="Estimate priority for a goal statement.",
                args_schema=PriorityEstimateArgs,
                handler=_priority_estimator_handler,
            ),
        )


async def _context_digest_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    context = arguments.get("context", {})
    if not isinstance(context, dict) or not context:
        return {"summary": "No context provided.", "keys": []}
    keys = sorted(context.keys())
    type_map = {key: type(context[key]).__name__ for key in keys}
    return {
        "summary": f"Context contains {len(keys)} keys.",
        "keys": keys,
        "types": type_map,
    }


async def _keyword_extract_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    text = str(arguments.get("text", ""))
    max_keywords = int(arguments.get("max_keywords", 5))
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    counts: dict[str, int] = {}
    for token in tokens:
        if len(token) <= 2:
            continue
        counts[token] = counts.get(token, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    keywords = [token for token, _ in sorted_items[:max_keywords]]
    return {
        "summary": f"Extracted {len(keywords)} keywords.",
        "keywords": keywords,
    }


async def _priority_estimator_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    goal = str(arguments.get("goal", ""))
    hint = str(arguments.get("priority_hint") or "").lower()
    lowered = goal.lower()
    if any(token in lowered for token in ("urgent", "critical", "must", "blocker")):
        priority = "high"
    elif any(token in lowered for token in ("later", "optional", "nice")):
        priority = "low"
    elif hint in {"high", "critical"}:
        priority = "high"
    elif hint in {"low"}:
        priority = "low"
    else:
        priority = "medium"

    return {
        "summary": f"Estimated priority: {priority}.",
        "priority": priority,
    }

