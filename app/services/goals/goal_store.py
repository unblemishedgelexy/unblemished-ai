from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.core.logger import StructuredLogger
from app.repositories.goal_repository import GoalRepository, GoalRow
from app.utils.helpers import utc_now


@dataclass(slots=True)
class GoalRecord:
    user_id: str
    active_goal: str
    sub_tasks: list[str]
    completion_status: str
    goal_priority: str
    updated_at: datetime


class GoalStore:
    def __init__(self, repository: GoalRepository, logger: StructuredLogger) -> None:
        self._repository = repository
        self._logger = logger

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def get_goal(self, user_id: str, trace_id: str) -> GoalRecord:
        await self.initialize()
        row = await self._repository.get_goal(user_id=user_id)
        if row is None:
            default = GoalRecord(
                user_id=user_id,
                active_goal="none",
                sub_tasks=[],
                completion_status="not-started",
                goal_priority="medium",
                updated_at=utc_now(),
            )
            await self.upsert_goal(goal=default, trace_id=trace_id)
            return default

        return GoalRecord(
            user_id=row.user_id,
            active_goal=row.active_goal,
            sub_tasks=row.sub_tasks,
            completion_status=row.completion_status,
            goal_priority=row.goal_priority,
            updated_at=row.updated_at,
        )

    async def upsert_goal(self, goal: GoalRecord, trace_id: str) -> None:
        await self.initialize()
        await self._repository.upsert_goal(
            GoalRow(
                user_id=goal.user_id,
                active_goal=goal.active_goal,
                sub_tasks=goal.sub_tasks,
                completion_status=goal.completion_status,
                goal_priority=goal.goal_priority,
                updated_at=goal.updated_at,
            ),
        )
        self._logger.info(
            "goal.store.upsert.completed",
            trace_id=trace_id,
            user_id=goal.user_id,
            memory_id="n/a",
            retrieval_count=0,
            active_goal=goal.active_goal,
            completion_status=goal.completion_status,
            goal_priority=goal.goal_priority,
        )

    async def is_ready(self) -> bool:
        return await self._repository.is_ready()

