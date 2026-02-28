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
class SkillRow:
    skill_id: str
    user_id: str
    trigger_text: str
    trigger_type: str
    tool_name: str
    tool_arguments: dict[str, Any]
    correction_count: int
    source: str
    active: bool
    created_at: datetime
    updated_at: datetime


class SkillRepository(Protocol):
    async def initialize(self) -> None: ...

    async def get_skill_by_id(self, *, skill_id: str) -> SkillRow | None: ...

    async def get_skill(self, *, user_id: str, trigger_text: str, tool_name: str) -> SkillRow | None: ...

    async def upsert_skill(self, row: SkillRow) -> None: ...

    async def list_skills(self, *, user_id: str, include_inactive: bool, limit: int) -> list[SkillRow]: ...

    async def list_active_skills(self, *, user_id: str, limit: int) -> list[SkillRow]: ...

    async def delete_skill(self, *, skill_id: str) -> bool: ...

    async def is_ready(self) -> bool: ...


class SQLiteSkillRepository:
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
                "repository.skill.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path=self._db_path,
            )

    async def get_skill(self, *, user_id: str, trigger_text: str, tool_name: str) -> SkillRow | None:
        await self.initialize()
        row = await asyncio.to_thread(self._get_skill_sync, user_id, trigger_text, tool_name)
        return _to_sqlite_skill_row(row)

    async def get_skill_by_id(self, *, skill_id: str) -> SkillRow | None:
        await self.initialize()
        row = await asyncio.to_thread(self._get_skill_by_id_sync, skill_id)
        return _to_sqlite_skill_row(row)

    async def upsert_skill(self, row: SkillRow) -> None:
        await self.initialize()
        await asyncio.to_thread(self._upsert_skill_sync, row)

    async def list_skills(self, *, user_id: str, include_inactive: bool, limit: int) -> list[SkillRow]:
        await self.initialize()
        rows = await asyncio.to_thread(self._list_skills_sync, user_id, include_inactive, limit)
        output: list[SkillRow] = []
        for row in rows:
            converted = _to_sqlite_skill_row(row)
            if converted is not None:
                output.append(converted)
        return output

    async def list_active_skills(self, *, user_id: str, limit: int) -> list[SkillRow]:
        await self.initialize()
        rows = await asyncio.to_thread(self._list_active_skills_sync, user_id, limit)
        output: list[SkillRow] = []
        for row in rows:
            converted = _to_sqlite_skill_row(row)
            if converted is not None:
                output.append(converted)
        return output

    async def delete_skill(self, *, skill_id: str) -> bool:
        await self.initialize()
        return await asyncio.to_thread(self._delete_skill_sync, skill_id)

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
                CREATE TABLE IF NOT EXISTS skill_registry (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    trigger_text TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_arguments TEXT NOT NULL DEFAULT '{}',
                    correction_count INTEGER NOT NULL DEFAULT 1,
                    source TEXT NOT NULL DEFAULT 'user_correction',
                    active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_user_trigger_tool
                ON skill_registry (user_id, trigger_text, tool_name)
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_skill_active_lookup
                ON skill_registry (user_id, active, updated_at DESC)
                """,
            )
            conn.commit()

    def _get_skill_sync(
        self,
        user_id: str,
        trigger_text: str,
        tool_name: str,
    ) -> tuple[Any, ...] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    id,
                    user_id,
                    trigger_text,
                    trigger_type,
                    tool_name,
                    tool_arguments,
                    correction_count,
                    source,
                    active,
                    created_at,
                    updated_at
                FROM skill_registry
                WHERE user_id = ? AND trigger_text = ? AND tool_name = ?
                LIMIT 1
                """,
                (user_id, trigger_text, tool_name),
            )
            return cursor.fetchone()

    def _get_skill_by_id_sync(self, skill_id: str) -> tuple[Any, ...] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    id,
                    user_id,
                    trigger_text,
                    trigger_type,
                    tool_name,
                    tool_arguments,
                    correction_count,
                    source,
                    active,
                    created_at,
                    updated_at
                FROM skill_registry
                WHERE id = ?
                LIMIT 1
                """,
                (skill_id,),
            )
            return cursor.fetchone()

    def _upsert_skill_sync(self, row: SkillRow) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO skill_registry (
                    id,
                    user_id,
                    trigger_text,
                    trigger_type,
                    tool_name,
                    tool_arguments,
                    correction_count,
                    source,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, trigger_text, tool_name) DO UPDATE SET
                    trigger_type = excluded.trigger_type,
                    tool_arguments = excluded.tool_arguments,
                    correction_count = excluded.correction_count,
                    source = excluded.source,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (
                    row.skill_id,
                    row.user_id,
                    row.trigger_text,
                    row.trigger_type,
                    row.tool_name,
                    json.dumps(row.tool_arguments),
                    row.correction_count,
                    row.source,
                    1 if row.active else 0,
                    row.created_at.isoformat(),
                    row.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def _list_skills_sync(
        self,
        user_id: str,
        include_inactive: bool,
        limit: int,
    ) -> list[tuple[Any, ...]]:
        with sqlite3.connect(self._db_path) as conn:
            if include_inactive:
                cursor = conn.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        trigger_text,
                        trigger_type,
                        tool_name,
                        tool_arguments,
                        correction_count,
                        source,
                        active,
                        created_at,
                        updated_at
                    FROM skill_registry
                    WHERE user_id = ?
                    ORDER BY correction_count DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, max(limit, 1)),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        trigger_text,
                        trigger_type,
                        tool_name,
                        tool_arguments,
                        correction_count,
                        source,
                        active,
                        created_at,
                        updated_at
                    FROM skill_registry
                    WHERE user_id = ? AND active = 1
                    ORDER BY correction_count DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, max(limit, 1)),
                )
            return cursor.fetchall()

    def _list_active_skills_sync(
        self,
        user_id: str,
        limit: int,
    ) -> list[tuple[Any, ...]]:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    id,
                    user_id,
                    trigger_text,
                    trigger_type,
                    tool_name,
                    tool_arguments,
                    correction_count,
                    source,
                    active,
                    created_at,
                    updated_at
                FROM skill_registry
                WHERE user_id = ? AND active = 1
                ORDER BY correction_count DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, max(limit, 1)),
            )
            return cursor.fetchall()

    def _delete_skill_sync(self, skill_id: str) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM skill_registry
                WHERE id = ?
                """,
                (skill_id,),
            )
            conn.commit()
            return cursor.rowcount > 0


class PostgresSkillRepository:
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
                "repository.skill.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path="postgres",
            )

    async def get_skill(self, *, user_id: str, trigger_text: str, tool_name: str) -> SkillRow | None:
        await self.initialize()
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT
                id,
                user_id,
                trigger_text,
                trigger_type,
                tool_name,
                tool_arguments,
                correction_count,
                source,
                active,
                created_at,
                updated_at
            FROM skill_registry
            WHERE user_id = $1 AND trigger_text = $2 AND tool_name = $3
            LIMIT 1
            """,
            user_id,
            trigger_text,
            tool_name,
        )
        return _to_postgres_skill_row(row)

    async def get_skill_by_id(self, *, skill_id: str) -> SkillRow | None:
        await self.initialize()
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT
                id,
                user_id,
                trigger_text,
                trigger_type,
                tool_name,
                tool_arguments,
                correction_count,
                source,
                active,
                created_at,
                updated_at
            FROM skill_registry
            WHERE id = $1
            LIMIT 1
            """,
            skill_id,
        )
        return _to_postgres_skill_row(row)

    async def upsert_skill(self, row: SkillRow) -> None:
        await self.initialize()
        assert self._pool is not None
        await self._pool.execute(
            """
            INSERT INTO skill_registry (
                id,
                user_id,
                trigger_text,
                trigger_type,
                tool_name,
                tool_arguments,
                correction_count,
                source,
                active,
                created_at,
                updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
            ON CONFLICT(user_id, trigger_text, tool_name) DO UPDATE SET
                trigger_type = excluded.trigger_type,
                tool_arguments = excluded.tool_arguments,
                correction_count = excluded.correction_count,
                source = excluded.source,
                active = excluded.active,
                updated_at = excluded.updated_at
            """,
            row.skill_id,
            row.user_id,
            row.trigger_text,
            row.trigger_type,
            row.tool_name,
            json.dumps(row.tool_arguments),
            row.correction_count,
            row.source,
            row.active,
            row.created_at,
            row.updated_at,
        )

    async def list_skills(self, *, user_id: str, include_inactive: bool, limit: int) -> list[SkillRow]:
        await self.initialize()
        assert self._pool is not None
        query = """
            SELECT
                id,
                user_id,
                trigger_text,
                trigger_type,
                tool_name,
                tool_arguments,
                correction_count,
                source,
                active,
                created_at,
                updated_at
            FROM skill_registry
            WHERE user_id = $1
        """
        if not include_inactive:
            query += " AND active = TRUE"
        query += " ORDER BY correction_count DESC, updated_at DESC LIMIT $2"
        rows = await self._pool.fetch(query, user_id, max(limit, 1))
        output: list[SkillRow] = []
        for row in rows:
            converted = _to_postgres_skill_row(row)
            if converted is not None:
                output.append(converted)
        return output

    async def list_active_skills(self, *, user_id: str, limit: int) -> list[SkillRow]:
        await self.initialize()
        assert self._pool is not None
        rows = await self._pool.fetch(
            """
            SELECT
                id,
                user_id,
                trigger_text,
                trigger_type,
                tool_name,
                tool_arguments,
                correction_count,
                source,
                active,
                created_at,
                updated_at
            FROM skill_registry
            WHERE user_id = $1 AND active = TRUE
            ORDER BY correction_count DESC, updated_at DESC
            LIMIT $2
            """,
            user_id,
            max(limit, 1),
        )
        output: list[SkillRow] = []
        for row in rows:
            converted = _to_postgres_skill_row(row)
            if converted is not None:
                output.append(converted)
        return output

    async def delete_skill(self, *, skill_id: str) -> bool:
        await self.initialize()
        assert self._pool is not None
        result = await self._pool.execute(
            """
            DELETE FROM skill_registry
            WHERE id = $1
            """,
            skill_id,
        )
        return result.endswith("1")

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
                CREATE TABLE IF NOT EXISTS skill_registry (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    trigger_text TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
                    correction_count INTEGER NOT NULL DEFAULT 1,
                    source TEXT NOT NULL DEFAULT 'user_correction',
                    active BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """,
            )
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_user_trigger_tool
                ON skill_registry (user_id, trigger_text, tool_name)
                """,
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_skill_active_lookup
                ON skill_registry (user_id, active, updated_at DESC)
                """,
            )


def _import_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - exercised in deployment environments.
        raise RuntimeError("asyncpg is required when DATABASE_DRIVER=postgres") from exc
    return asyncpg


def _to_sqlite_skill_row(row: tuple[Any, ...] | None) -> SkillRow | None:
    if row is None:
        return None
    return SkillRow(
        skill_id=str(row[0]),
        user_id=str(row[1]),
        trigger_text=str(row[2]),
        trigger_type=str(row[3]),
        tool_name=str(row[4]),
        tool_arguments=_parse_tool_arguments(row[5]),
        correction_count=int(row[6]) if row[6] is not None else 1,
        source=str(row[7]) if row[7] is not None else "user_correction",
        active=bool(row[8]),
        created_at=_coerce_datetime(row[9]),
        updated_at=_coerce_datetime(row[10]),
    )


def _to_postgres_skill_row(row: Any) -> SkillRow | None:
    if row is None:
        return None
    return SkillRow(
        skill_id=str(row["id"]),
        user_id=str(row["user_id"]),
        trigger_text=str(row["trigger_text"]),
        trigger_type=str(row["trigger_type"]),
        tool_name=str(row["tool_name"]),
        tool_arguments=_parse_tool_arguments(row["tool_arguments"]),
        correction_count=int(row["correction_count"]) if row["correction_count"] is not None else 1,
        source=str(row["source"]) if row["source"] is not None else "user_correction",
        active=bool(row["active"]),
        created_at=_coerce_datetime(row["created_at"]),
        updated_at=_coerce_datetime(row["updated_at"]),
    )


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return utc_now()
    return utc_now()
