from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.core.logger import StructuredLogger
from app.services.tools.tool_executor import ToolExecutor
from app.services.tools.tool_registry import ToolRegistry


class ToolInterface:
    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        logger: StructuredLogger,
        enabled: bool = True,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._logger = logger
        self._enabled = enabled

    async def invoke(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        trace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        if not self._enabled:
            return _failed_invocation(
                tool_name=tool_name,
                arguments=arguments,
                result_summary="Tool execution disabled.",
            )

        if not self._registry.is_registered(tool_name):
            self._logger.warning(
                "tool.invoke.rejected",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                tool_name=tool_name,
                reason="not_whitelisted",
            )
            return _failed_invocation(
                tool_name=tool_name,
                arguments=arguments,
                result_summary="Tool not allowed by whitelist.",
            )

        try:
            validated_arguments = self._registry.validate_arguments(tool_name, arguments)
        except ValidationError:
            self._logger.warning(
                "tool.invoke.invalid_arguments",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                tool_name=tool_name,
            )
            return _failed_invocation(
                tool_name=tool_name,
                arguments=arguments,
                result_summary="Invalid tool arguments.",
            )

        handler = self._registry.get_handler(tool_name)
        outcome = await self._executor.execute(
            tool_name=tool_name,
            arguments=validated_arguments,
            handler=handler,
            trace_id=trace_id,
            user_id=user_id,
        )
        if not outcome.success:
            return {
                "tool_name": tool_name,
                "arguments": validated_arguments,
                "success": False,
                "result_summary": f"Tool execution failed: {outcome.error or 'unknown'}.",
                "duration_ms": max(outcome.duration_ms, 0),
            }

        result_summary = str(outcome.output.get("summary", "Tool executed successfully."))
        return {
            "tool_name": tool_name,
            "arguments": validated_arguments,
            "success": True,
            "result_summary": result_summary,
            "duration_ms": max(outcome.duration_ms, 0),
        }

    async def invoke_many(
        self,
        *,
        invocations: list[dict[str, Any]],
        trace_id: str,
        user_id: str,
        max_calls: int = 2,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for request in invocations[:max_calls]:
            name = str(request.get("tool_name", "")).strip()
            arguments = request.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            result = await self.invoke(
                tool_name=name,
                arguments=arguments,
                trace_id=trace_id,
                user_id=user_id,
            )
            results.append(result)
        return results

    def is_enabled(self) -> bool:
        return self._enabled

    def is_tool_allowed(self, tool_name: str) -> bool:
        if not self._enabled:
            return False
        return self._registry.is_registered(tool_name)

    def list_allowed_tools(self) -> list[str]:
        return [
            str(item.get("name"))
            for item in self._registry.list_metadata()
            if isinstance(item.get("name"), str)
        ]

    async def is_ready(self) -> bool:
        metadata = self._registry.list_metadata()
        return bool(metadata)


def _failed_invocation(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_summary: str,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "success": False,
        "result_summary": result_summary,
        "duration_ms": 0,
    }
