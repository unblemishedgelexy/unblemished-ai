from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.core.logger import StructuredLogger

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ToolExecutionOutcome:
    tool_name: str
    success: bool
    output: dict[str, Any]
    error: str | None
    duration_ms: int


class ToolExecutor:
    def __init__(self, logger: StructuredLogger, max_timeout_seconds: float) -> None:
        self._logger = logger
        self._max_timeout_seconds = max_timeout_seconds

    async def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        handler: ToolHandler,
        trace_id: str,
        user_id: str,
    ) -> ToolExecutionOutcome:
        started = asyncio.get_running_loop().time()
        try:
            output = await asyncio.wait_for(
                handler(arguments),
                timeout=self._max_timeout_seconds,
            )
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            self._logger.info(
                "tool.execute.completed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                tool_name=tool_name,
                duration_ms=duration_ms,
            )
            return ToolExecutionOutcome(
                tool_name=tool_name,
                success=True,
                output=output,
                error=None,
                duration_ms=duration_ms,
            )
        except asyncio.TimeoutError:
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            self._logger.error(
                "tool.execute.timeout",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                tool_name=tool_name,
                timeout_seconds=self._max_timeout_seconds,
            )
            return ToolExecutionOutcome(
                tool_name=tool_name,
                success=False,
                output={},
                error="timeout",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            self._logger.error(
                "tool.execute.failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                tool_name=tool_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return ToolExecutionOutcome(
                tool_name=tool_name,
                success=False,
                output={},
                error=type(exc).__name__,
                duration_ms=duration_ms,
            )

