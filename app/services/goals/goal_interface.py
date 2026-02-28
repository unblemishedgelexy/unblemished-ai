from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.logger import StructuredLogger
from app.services.goals.goal_store import GoalRecord, GoalStore
from app.utils.helpers import utc_now


@dataclass(slots=True)
class GoalSnapshot:
    user_id: str
    active_goal: str
    sub_tasks: list[str]
    completion_status: str
    goal_priority: str


class GoalInterface:
    def __init__(
        self,
        store: GoalStore | None,
        logger: StructuredLogger,
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._logger = logger
        self._enabled = enabled and store is not None

    @classmethod
    def create_disabled(cls, logger: StructuredLogger) -> GoalInterface:
        return cls(store=None, logger=logger, enabled=False)

    async def get_snapshot(self, user_id: str, trace_id: str) -> GoalSnapshot:
        if not self._enabled or self._store is None:
            return GoalSnapshot(
                user_id=user_id,
                active_goal="none",
                sub_tasks=[],
                completion_status="not-started",
                goal_priority="medium",
            )
        record = await self._store.get_goal(user_id=user_id, trace_id=trace_id)
        snapshot = _to_snapshot(record)
        self._logger.info(
            "goal.snapshot.loaded",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            active_goal=snapshot.active_goal,
            completion_status=snapshot.completion_status,
        )
        return snapshot

    async def update_after_interaction(
        self,
        user_id: str,
        trace_id: str,
        user_input: str,
        execution_plan: list[dict[str, Any]],
        tool_invocations: list[dict[str, Any]],
        fallback_triggered: bool,
    ) -> GoalSnapshot:
        current = await self.get_snapshot(user_id=user_id, trace_id=trace_id)

        active_goal = _extract_goal(user_input=user_input) or current.active_goal
        sub_tasks = _extract_sub_tasks(execution_plan=execution_plan)
        completion_status = _derive_completion_status(
            fallback_triggered=fallback_triggered,
            tool_invocations=tool_invocations,
            has_steps=bool(sub_tasks),
            current=current.completion_status,
        )
        goal_priority = _derive_priority(user_input=user_input, current=current.goal_priority)

        updated = GoalSnapshot(
            user_id=user_id,
            active_goal=active_goal,
            sub_tasks=sub_tasks,
            completion_status=completion_status,
            goal_priority=goal_priority,
        )
        if self._enabled and self._store is not None:
            await self._store.upsert_goal(
                goal=GoalRecord(
                    user_id=user_id,
                    active_goal=updated.active_goal,
                    sub_tasks=updated.sub_tasks,
                    completion_status=updated.completion_status,
                    goal_priority=updated.goal_priority,
                    updated_at=utc_now(),
                ),
                trace_id=trace_id,
            )

        self._logger.info(
            "goal.snapshot.updated",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            active_goal=updated.active_goal,
            completion_status=updated.completion_status,
            goal_priority=updated.goal_priority,
        )
        return updated

    async def is_ready(self) -> bool:
        if not self._enabled or self._store is None:
            return True
        return await self._store.is_ready()


def _to_snapshot(record: GoalRecord) -> GoalSnapshot:
    return GoalSnapshot(
        user_id=record.user_id,
        active_goal=record.active_goal,
        sub_tasks=record.sub_tasks,
        completion_status=record.completion_status,
        goal_priority=record.goal_priority,
    )


def _extract_goal(user_input: str) -> str | None:
    lowered = user_input.lower()
    marker = "goal:"
    if marker not in lowered:
        return None
    start = lowered.index(marker) + len(marker)
    extracted = user_input[start:].strip()
    if not extracted:
        return None
    return extracted[:220]


def _extract_sub_tasks(execution_plan: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for item in execution_plan[:8]:
        step = item.get("step")
        if isinstance(step, str) and step:
            output.append(step)
    return output


def _derive_completion_status(
    fallback_triggered: bool,
    tool_invocations: list[dict[str, Any]],
    has_steps: bool,
    current: str,
) -> str:
    if fallback_triggered:
        return "blocked"
    if any(bool(item.get("success")) for item in tool_invocations):
        return "in-progress"
    if has_steps:
        return "in-progress"
    return current or "not-started"


def _derive_priority(user_input: str, current: str) -> str:
    lowered = user_input.lower()
    if re.search(r"\b(urgent|asap|critical|highest|must)\b", lowered):
        return "high"
    if re.search(r"\b(low|later|optional|nice to have)\b", lowered):
        return "low"
    return current or "medium"

