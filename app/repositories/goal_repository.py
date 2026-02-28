from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from app.core.logger import StructuredLogger
from app.utils.helpers import utc_now


@dataclass(slots=True)
class GoalRow:
    user_id: str
    active_goal: str
    sub_tasks: list[str]
    completion_status: str
    goal_priority: str
    updated_at: datetime


class GoalRepository(Protocol):
    async def initialize(self) -> None: ...

    async def get_goal(self, user_id: str) -> GoalRow | None: ...

    async def upsert_goal(self, row: GoalRow) -> None: ...

    async def is_ready(self) -> bool: ...


class SQLiteGoalRepository:
    def __init__(self, db_path: str, logger: StructuredLogger) -> None:
        self._db_path = db_path
        self._logger = logger
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True
            self._logger.info(
                "repository.goal.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path=self._db_path,
            )

    async def get_goal(self, user_id: str) -> GoalRow | None:
        await self.initialize()
        row = await asyncio.to_thread(self._get_goal_sync, user_id)
        if row is None:
            return None
        return GoalRow(
            user_id=row[0],
            active_goal=row[1],
            sub_tasks=_parse_sub_tasks(row[2]),
            completion_status=row[3],
            goal_priority=row[4],
            updated_at=_parse_datetime(row[5]),
        )

    async def upsert_goal(self, row: GoalRow) -> None:
        await self.initialize()
        await asyncio.to_thread(self._upsert_goal_sync, row)

    async def is_ready(self) -> bool:
        try:
            await self.initialize()
            return True
        except Exception:
            return False

    def _initialize_sync(self) -> None:
        db_file = Path(self._db_path)
        if db_file.parent and str(db_file.parent) not in {".", ""}:
            db_file.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_table (
                    user_id TEXT PRIMARY KEY,
                    active_goal TEXT NOT NULL,
                    sub_tasks TEXT NOT NULL,
                    completion_status TEXT NOT NULL,
                    goal_priority TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()

    def _get_goal_sync(self, user_id: str) -> tuple[str, str, str, str, str, str] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT user_id, active_goal, sub_tasks, completion_status, goal_priority, updated_at
                FROM goal_table
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,),
            )
            return cursor.fetchone()

    def _upsert_goal_sync(self, row: GoalRow) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO goal_table (user_id, active_goal, sub_tasks, completion_status, goal_priority, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    active_goal = excluded.active_goal,
                    sub_tasks = excluded.sub_tasks,
                    completion_status = excluded.completion_status,
                    goal_priority = excluded.goal_priority,
                    updated_at = excluded.updated_at
                """,
                (
                    row.user_id,
                    row.active_goal,
                    json.dumps(row.sub_tasks),
                    row.completion_status,
                    row.goal_priority,
                    row.updated_at.isoformat(),
                ),
            )
            conn.commit()


def _parse_sub_tasks(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, str)]


def _parse_datetime(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return utc_now()


class PostgresGoalRepository:
    def __init__(self, dsn: str, logger: StructuredLogger) -> None:
        self._dsn = dsn
        self._logger = logger
        self._pool: Any | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            asyncpg = _import_asyncpg()
            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=4)
            await self._initialize_schema()
            self._initialized = True
            self._logger.info(
                "repository.goal.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path="postgres",
            )

    async def get_goal(self, user_id: str) -> GoalRow | None:
        await self.initialize()
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT
                user_id,
                active_goal,
                sub_tasks,
                completion_status,
                goal_priority,
                updated_at
            FROM goal_table
            WHERE user_id = $1
            LIMIT 1
            """,
            user_id,
        )
        if row is None:
            return None
        return GoalRow(
            user_id=row["user_id"],
            active_goal=row["active_goal"],
            sub_tasks=_parse_postgres_sub_tasks(row["sub_tasks"]),
            completion_status=row["completion_status"],
            goal_priority=row["goal_priority"],
            updated_at=_coerce_datetime(row["updated_at"]),
        )

    async def upsert_goal(self, row: GoalRow) -> None:
        await self.initialize()
        assert self._pool is not None
        await self._pool.execute(
            """
            INSERT INTO goal_table (
                user_id,
                active_goal,
                sub_tasks,
                completion_status,
                goal_priority,
                updated_at
            )
            VALUES ($1, $2, $3::jsonb, $4, $5, $6)
            ON CONFLICT(user_id) DO UPDATE SET
                active_goal = excluded.active_goal,
                sub_tasks = excluded.sub_tasks,
                completion_status = excluded.completion_status,
                goal_priority = excluded.goal_priority,
                updated_at = excluded.updated_at
            """,
            row.user_id,
            row.active_goal,
            json.dumps(row.sub_tasks),
            row.completion_status,
            row.goal_priority,
            row.updated_at,
        )

    async def is_ready(self) -> bool:
        try:
            await self.initialize()
            assert self._pool is not None
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def _initialize_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_table (
                    user_id TEXT PRIMARY KEY,
                    active_goal TEXT NOT NULL,
                    sub_tasks JSONB NOT NULL,
                    completion_status TEXT NOT NULL,
                    goal_priority TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """,
            )


def _import_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - exercised in deployment environments.
        raise RuntimeError("asyncpg is required when DATABASE_DRIVER=postgres") from exc
    return asyncpg


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return _parse_datetime(value)
    return utc_now()


def _parse_postgres_sub_tasks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_sub_tasks(value)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []
