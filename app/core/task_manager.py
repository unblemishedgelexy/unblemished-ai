from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.core.logger import StructuredLogger

TaskFactory = Callable[[], Awaitable[None]]


class TaskManager:
    def __init__(self, logger: StructuredLogger) -> None:
        self._logger = logger
        self._tasks: set[asyncio.Task[None]] = set()

    def register_task(
        self,
        *,
        task_name: str,
        trace_id: str,
        user_id: str,
        task_factory: TaskFactory,
        retry_once: bool = True,
    ) -> None:
        task = asyncio.create_task(
            self._run_with_retry(
                task_name=task_name,
                trace_id=trace_id,
                user_id=user_id,
                task_factory=task_factory,
                retry_once=retry_once,
            ),
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def graceful_shutdown(self, timeout_seconds: float = 5.0) -> None:
        if not self._tasks:
            return

        pending = list(self._tasks)
        done, not_done = await asyncio.wait(pending, timeout=timeout_seconds)
        for task in done:
            try:
                await task
            except Exception as exc:
                self._logger.error(
                    "task_manager.shutdown.task_failed",
                    trace_id="system",
                    user_id="system",
                    memory_id="n/a",
                    retrieval_count=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        for task in not_done:
            task.cancel()
            self._logger.warning(
                "task_manager.shutdown.task_cancelled",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                task_name=getattr(task.get_coro(), "__name__", "unknown"),
            )

    async def _run_with_retry(
        self,
        *,
        task_name: str,
        trace_id: str,
        user_id: str,
        task_factory: TaskFactory,
        retry_once: bool,
    ) -> None:
        attempts = 2 if retry_once else 1
        for attempt in range(1, attempts + 1):
            try:
                await task_factory()
                self._logger.info(
                    "task_manager.task.completed",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=0,
                    task_name=task_name,
                    attempt=attempt,
                )
                return
            except Exception as exc:
                self._logger.error(
                    "task_manager.task.failed",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=0,
                    task_name=task_name,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if attempt == attempts:
                    return

