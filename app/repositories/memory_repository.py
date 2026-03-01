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
class MemoryRow:
    memory_id: str
    user_id: str
    trace_id: str
    summary_text: str
    created_at: datetime
    embedding: str | None
    importance_score: float
    action_type: str | None
    action_result_summary: str | None


class MemoryRepository(Protocol):
    async def initialize(self) -> None: ...

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
    ) -> None: ...

    async def fetch_user_memories(self, *, user_id: str, limit: int) -> list[MemoryRow]: ...

    async def count_entries(self) -> int: ...

    async def is_ready(self) -> bool: ...


class SQLiteMemoryRepository:
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
                "repository.memory.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path=self._db_path,
            )

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
        await asyncio.to_thread(
            self._insert_memory_sync,
            memory_id,
            user_id,
            trace_id,
            summary_text,
            created_at.isoformat(),
            embedding_json,
            importance_score,
            action_type,
            action_result_summary,
        )

    async def fetch_user_memories(self, *, user_id: str, limit: int) -> list[MemoryRow]:
        await self.initialize()
        rows = await asyncio.to_thread(self._fetch_user_memories_sync, user_id, limit)
        return [
            MemoryRow(
                memory_id=row[0],
                user_id=row[1],
                trace_id=row[2],
                summary_text=row[3],
                created_at=_parse_datetime(row[4]),
                embedding=row[5],
                importance_score=float(row[6]) if row[6] is not None else 0.5,
                action_type=row[7],
                action_result_summary=row[8],
            )
            for row in rows
        ]

    async def count_entries(self) -> int:
        await self.initialize()
        return await asyncio.to_thread(self._count_entries_sync)

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
                CREATE TABLE IF NOT EXISTS memory_table (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    embedding TEXT,
                    importance_score REAL NOT NULL DEFAULT 0.5,
                    action_type TEXT,
                    action_result_summary TEXT
                )
                """,
            )
            _ensure_column(
                conn=conn,
                table_name="memory_table",
                column_name="importance_score",
                definition="REAL NOT NULL DEFAULT 0.5",
            )
            _ensure_column(
                conn=conn,
                table_name="memory_table",
                column_name="action_type",
                definition="TEXT",
            )
            _ensure_column(
                conn=conn,
                table_name="memory_table",
                column_name="action_result_summary",
                definition="TEXT",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_user_created ON memory_table (user_id, created_at DESC)",
            )
            conn.commit()

    def _insert_memory_sync(
        self,
        memory_id: str,
        user_id: str,
        trace_id: str,
        summary_text: str,
        created_at: str,
        embedding_json: str | None,
        importance_score: float,
        action_type: str | None,
        action_result_summary: str | None,
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO memory_table (
                    id,
                    user_id,
                    trace_id,
                    summary_text,
                    created_at,
                    embedding,
                    importance_score,
                    action_type,
                    action_result_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    trace_id,
                    summary_text,
                    created_at,
                    embedding_json,
                    importance_score,
                    action_type,
                    action_result_summary,
                ),
            )
            conn.commit()

    def _fetch_user_memories_sync(
        self,
        user_id: str,
        limit: int,
    ) -> list[tuple[str, str, str, str, str, str | None, float, str | None, str | None]]:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, user_id, trace_id, summary_text, created_at, embedding, importance_score, action_type, action_result_summary
                FROM memory_table
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return cursor.fetchall()

    def _count_entries_sync(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_table")
            row = cursor.fetchone()
            return int(row[0]) if row else 0


def _parse_datetime(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return utc_now()


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


class PostgresMemoryRepository:
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
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=6,
                statement_cache_size=0,
                command_timeout=20,
            )
            await self._initialize_schema()
            self._initialized = True
            self._logger.info(
                "repository.memory.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path="postgres",
            )

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
        assert self._pool is not None
        await self._pool.execute(
            """
            INSERT INTO memory_table (
                id,
                user_id,
                trace_id,
                summary_text,
                created_at,
                embedding,
                importance_score,
                action_type,
                action_result_summary
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
            """,
            memory_id,
            user_id,
            trace_id,
            summary_text,
            created_at,
            embedding_json,
            importance_score,
            action_type,
            action_result_summary,
        )

    async def fetch_user_memories(self, *, user_id: str, limit: int) -> list[MemoryRow]:
        await self.initialize()
        assert self._pool is not None
        rows = await self._pool.fetch(
            """
            SELECT
                id,
                user_id,
                trace_id,
                summary_text,
                created_at,
                embedding,
                importance_score,
                action_type,
                action_result_summary
            FROM memory_table
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
        return [
            MemoryRow(
                memory_id=row["id"],
                user_id=row["user_id"],
                trace_id=row["trace_id"],
                summary_text=row["summary_text"],
                created_at=_coerce_datetime(row["created_at"]),
                embedding=_normalize_embedding_json(row["embedding"]),
                importance_score=float(row["importance_score"]) if row["importance_score"] is not None else 0.5,
                action_type=row["action_type"],
                action_result_summary=row["action_result_summary"],
            )
            for row in rows
        ]

    async def count_entries(self) -> int:
        await self.initialize()
        assert self._pool is not None
        value = await self._pool.fetchval("SELECT COUNT(*) FROM memory_table")
        return int(value) if value is not None else 0

    async def is_ready(self) -> bool:
        try:
            await self.initialize()
            assert self._pool is not None
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception as exc:
            self._logger.error(
                "repository.memory.not_ready",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return False

    async def _initialize_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_table (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    embedding JSONB NULL,
                    importance_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                    action_type TEXT NULL,
                    action_result_summary TEXT NULL
                )
                """,
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_user_created
                ON memory_table (user_id, created_at DESC)
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


def _normalize_embedding_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return None
