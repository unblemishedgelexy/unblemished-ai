from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from app.core.logger import StructuredLogger
from app.utils.helpers import utc_now


@dataclass(slots=True)
class ProfileRow:
    user_id: str
    preferred_tone: str
    dominant_intent_type: str
    conversation_depth_preference: str
    emotional_baseline: str
    updated_at: datetime


class ProfileRepository(Protocol):
    async def initialize(self) -> None: ...

    async def get_profile(self, user_id: str) -> ProfileRow | None: ...

    async def upsert_profile(self, row: ProfileRow) -> None: ...

    async def is_ready(self) -> bool: ...


class SQLiteProfileRepository:
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
                "repository.profile.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path=self._db_path,
            )

    async def get_profile(self, user_id: str) -> ProfileRow | None:
        await self.initialize()
        row = await asyncio.to_thread(self._get_profile_sync, user_id)
        if row is None:
            return None
        return ProfileRow(
            user_id=row[0],
            preferred_tone=row[1],
            dominant_intent_type=row[2],
            conversation_depth_preference=row[3],
            emotional_baseline=row[4],
            updated_at=_parse_datetime(row[5]),
        )

    async def upsert_profile(self, row: ProfileRow) -> None:
        await self.initialize()
        await asyncio.to_thread(self._upsert_profile_sync, row)

    async def is_ready(self) -> bool:
        try:
            await self.initialize()
            return True
        except Exception:
            return False

    def _initialize_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile_table (
                    user_id TEXT PRIMARY KEY,
                    preferred_tone TEXT NOT NULL,
                    dominant_intent_type TEXT NOT NULL,
                    conversation_depth_preference TEXT NOT NULL,
                    emotional_baseline TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()

    def _get_profile_sync(self, user_id: str) -> tuple[str, str, str, str, str, str] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT user_id, preferred_tone, dominant_intent_type, conversation_depth_preference, emotional_baseline, updated_at
                FROM user_profile_table
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,),
            )
            return cursor.fetchone()

    def _upsert_profile_sync(self, row: ProfileRow) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_profile_table (user_id, preferred_tone, dominant_intent_type, conversation_depth_preference, emotional_baseline, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferred_tone = excluded.preferred_tone,
                    dominant_intent_type = excluded.dominant_intent_type,
                    conversation_depth_preference = excluded.conversation_depth_preference,
                    emotional_baseline = excluded.emotional_baseline,
                    updated_at = excluded.updated_at
                """,
                (
                    row.user_id,
                    row.preferred_tone,
                    row.dominant_intent_type,
                    row.conversation_depth_preference,
                    row.emotional_baseline,
                    row.updated_at.isoformat(),
                ),
            )
            conn.commit()


def _parse_datetime(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return utc_now()


class PostgresProfileRepository:
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
                "repository.profile.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path="postgres",
            )

    async def get_profile(self, user_id: str) -> ProfileRow | None:
        await self.initialize()
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT
                user_id,
                preferred_tone,
                dominant_intent_type,
                conversation_depth_preference,
                emotional_baseline,
                updated_at
            FROM user_profile_table
            WHERE user_id = $1
            LIMIT 1
            """,
            user_id,
        )
        if row is None:
            return None
        return ProfileRow(
            user_id=row["user_id"],
            preferred_tone=row["preferred_tone"],
            dominant_intent_type=row["dominant_intent_type"],
            conversation_depth_preference=row["conversation_depth_preference"],
            emotional_baseline=row["emotional_baseline"],
            updated_at=_coerce_datetime(row["updated_at"]),
        )

    async def upsert_profile(self, row: ProfileRow) -> None:
        await self.initialize()
        assert self._pool is not None
        await self._pool.execute(
            """
            INSERT INTO user_profile_table (
                user_id,
                preferred_tone,
                dominant_intent_type,
                conversation_depth_preference,
                emotional_baseline,
                updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT(user_id) DO UPDATE SET
                preferred_tone = excluded.preferred_tone,
                dominant_intent_type = excluded.dominant_intent_type,
                conversation_depth_preference = excluded.conversation_depth_preference,
                emotional_baseline = excluded.emotional_baseline,
                updated_at = excluded.updated_at
            """,
            row.user_id,
            row.preferred_tone,
            row.dominant_intent_type,
            row.conversation_depth_preference,
            row.emotional_baseline,
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
                CREATE TABLE IF NOT EXISTS user_profile_table (
                    user_id TEXT PRIMARY KEY,
                    preferred_tone TEXT NOT NULL,
                    dominant_intent_type TEXT NOT NULL,
                    conversation_depth_preference TEXT NOT NULL,
                    emotional_baseline TEXT NOT NULL,
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
