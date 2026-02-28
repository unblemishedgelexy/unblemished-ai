from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.core.config import Settings
from app.core.dependencies import create_container
from app.repositories.goal_repository import GoalRow
from app.repositories.memory_repository import MemoryRow
from app.repositories.profile_repository import ProfileRow
from app.repositories.skill_repository import SkillRow
from app.schemas.request_schema import ChatRequest


class _FakePostgresMemoryRepository:
    def __init__(self, dsn: str, logger) -> None:
        self._dsn = dsn
        self._logger = logger
        self._initialized = False
        self._rows: list[MemoryRow] = []

    async def initialize(self) -> None:
        self._initialized = True

    async def insert_memory(
        self,
        *,
        memory_id: str,
        user_id: str,
        trace_id: str,
        summary_text: str,
        created_at: datetime,
        embedding_json: str | None,
        importance_score: float,
        action_type: str | None,
        action_result_summary: str | None,
    ) -> None:
        await self.initialize()
        self._rows.append(
            MemoryRow(
                memory_id=memory_id,
                user_id=user_id,
                trace_id=trace_id,
                summary_text=summary_text,
                created_at=created_at,
                embedding=embedding_json,
                importance_score=importance_score,
                action_type=action_type,
                action_result_summary=action_result_summary,
            ),
        )

    async def fetch_user_memories(self, *, user_id: str, limit: int) -> list[MemoryRow]:
        await self.initialize()
        matches = [row for row in self._rows if row.user_id == user_id]
        matches.sort(key=lambda row: row.created_at, reverse=True)
        return matches[:limit]

    async def count_entries(self) -> int:
        await self.initialize()
        return len(self._rows)

    async def is_ready(self) -> bool:
        return self._initialized


class _FakePostgresProfileRepository:
    def __init__(self, dsn: str, logger) -> None:
        self._dsn = dsn
        self._logger = logger
        self._initialized = False
        self._rows: dict[str, ProfileRow] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def get_profile(self, user_id: str) -> ProfileRow | None:
        await self.initialize()
        return self._rows.get(user_id)

    async def upsert_profile(self, row: ProfileRow) -> None:
        await self.initialize()
        self._rows[row.user_id] = row

    async def is_ready(self) -> bool:
        return self._initialized


class _FakePostgresGoalRepository:
    def __init__(self, dsn: str, logger) -> None:
        self._dsn = dsn
        self._logger = logger
        self._initialized = False
        self._rows: dict[str, GoalRow] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def get_goal(self, user_id: str) -> GoalRow | None:
        await self.initialize()
        return self._rows.get(user_id)

    async def upsert_goal(self, row: GoalRow) -> None:
        await self.initialize()
        self._rows[row.user_id] = row

    async def is_ready(self) -> bool:
        return self._initialized


class _FakePostgresSkillRepository:
    def __init__(self, dsn: str, logger) -> None:
        self._dsn = dsn
        self._logger = logger
        self._initialized = False
        self._rows: dict[tuple[str, str, str], SkillRow] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def get_skill(self, *, user_id: str, trigger_text: str, tool_name: str) -> SkillRow | None:
        await self.initialize()
        return self._rows.get((user_id, trigger_text, tool_name))

    async def get_skill_by_id(self, *, skill_id: str) -> SkillRow | None:
        await self.initialize()
        for row in self._rows.values():
            if row.skill_id == skill_id:
                return row
        return None

    async def upsert_skill(self, row: SkillRow) -> None:
        await self.initialize()
        self._rows[(row.user_id, row.trigger_text, row.tool_name)] = row

    async def list_skills(self, *, user_id: str, include_inactive: bool, limit: int) -> list[SkillRow]:
        await self.initialize()
        rows = [row for row in self._rows.values() if row.user_id == user_id]
        if not include_inactive:
            rows = [row for row in rows if row.active]
        rows.sort(key=lambda item: (item.correction_count, item.updated_at), reverse=True)
        return rows[:limit]

    async def list_active_skills(self, *, user_id: str, limit: int) -> list[SkillRow]:
        await self.initialize()
        rows = [row for row in self._rows.values() if row.user_id == user_id and row.active]
        rows.sort(key=lambda item: (item.correction_count, item.updated_at), reverse=True)
        return rows[:limit]

    async def delete_skill(self, *, skill_id: str) -> bool:
        await self.initialize()
        target_key: tuple[str, str, str] | None = None
        for key, row in self._rows.items():
            if row.skill_id == skill_id:
                target_key = key
                break
        if target_key is None:
            return False
        del self._rows[target_key]
        return True

    async def is_ready(self) -> bool:
        return self._initialized


def test_postgres_driver_uses_mocked_postgres_repositories() -> None:
    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        settings = Settings(
            memory_db_path=str(Path(temp_dir) / "memory.db"),
            database_driver="postgres",
            postgres_dsn="postgresql://mocked-user:mocked-pass@localhost:5432/humoniod_test",
        )
        with (
            patch("app.core.dependencies.PostgresMemoryRepository", _FakePostgresMemoryRepository),
            patch("app.core.dependencies.PostgresProfileRepository", _FakePostgresProfileRepository),
            patch("app.core.dependencies.PostgresGoalRepository", _FakePostgresGoalRepository),
            patch("app.core.dependencies.PostgresSkillRepository", _FakePostgresSkillRepository),
        ):
            container = create_container(settings=settings)
            request = ChatRequest(
                input_text="Plan a clean architecture upgrade with persistence isolation.",
                trace_id="trace-pg-1",
                user_id="pg-user-1",
            )
            response = asyncio.run(container.brain.reason(request))
            asyncio.run(
                container.memory_store.store_memory(
                    user_id="pg-user-1",
                    trace_id="trace-pg-1",
                    summary_text="Postgres repository integration smoke memory.",
                ),
            )
            asyncio.run(container.task_manager.graceful_shutdown())

            memory_repo = container.memory_store._repository
            profile_repo = container.profile_store._repository
            goal_repo = container.goal_store._repository
            skill_repo = container.skill_interface._repository

            assert isinstance(memory_repo, _FakePostgresMemoryRepository)
            assert isinstance(profile_repo, _FakePostgresProfileRepository)
            assert isinstance(goal_repo, _FakePostgresGoalRepository)
            assert isinstance(skill_repo, _FakePostgresSkillRepository)
            assert response.trace_id == "trace-pg-1"
            assert response.user_profile_snapshot is not None
            assert asyncio.run(memory_repo.count_entries()) >= 1
            assert "pg-user-1" in profile_repo._rows
            assert "pg-user-1" in goal_repo._rows
