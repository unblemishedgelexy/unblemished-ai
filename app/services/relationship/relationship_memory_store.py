from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.core.logger import StructuredLogger
from app.utils.helpers import utc_now


@dataclass(slots=True)
class RelationshipMemory:
    user_id: str
    first_interaction_timestamp: str
    important_dates: list[str]
    shared_memories: list[str]
    conflict_history: list[str]
    inside_jokes: list[str]


class RelationshipMemoryStore:
    def __init__(self, *, db_path: str, logger: StructuredLogger) -> None:
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
                "relationship.memory.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                db_path=self._db_path,
            )

    async def get_or_create(self, *, user_id: str, trace_id: str) -> RelationshipMemory:
        await self.initialize()
        row = await asyncio.to_thread(self._get_row_sync, user_id)
        if row is None:
            created = RelationshipMemory(
                user_id=user_id,
                first_interaction_timestamp=utc_now().isoformat(),
                important_dates=[],
                shared_memories=[],
                conflict_history=[],
                inside_jokes=[],
            )
            await asyncio.to_thread(self._insert_row_sync, created)
            self._logger.info(
                "relationship.memory.created",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
            )
            return created
        return RelationshipMemory(
            user_id=row[0],
            first_interaction_timestamp=row[1],
            important_dates=_parse_json_list(row[2]),
            shared_memories=_parse_json_list(row[3]),
            conflict_history=_parse_json_list(row[4]),
            inside_jokes=_parse_json_list(row[5]),
        )

    async def update(
        self,
        *,
        memory: RelationshipMemory,
        trace_id: str,
    ) -> None:
        await self.initialize()
        await asyncio.to_thread(self._update_row_sync, memory)
        self._logger.info(
            "relationship.memory.updated",
            trace_id=trace_id,
            user_id=memory.user_id,
            memory_id="n/a",
            retrieval_count=0,
            shared_memories=len(memory.shared_memories),
            conflicts=len(memory.conflict_history),
            inside_jokes=len(memory.inside_jokes),
        )

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
                CREATE TABLE IF NOT EXISTS relationship_memory (
                    user_id TEXT PRIMARY KEY,
                    first_interaction_timestamp TEXT NOT NULL,
                    important_dates TEXT NOT NULL,
                    shared_memories TEXT NOT NULL,
                    conflict_history TEXT NOT NULL,
                    inside_jokes TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()

    def _get_row_sync(self, user_id: str) -> tuple[str, str, str, str, str, str] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT user_id, first_interaction_timestamp, important_dates, shared_memories, conflict_history, inside_jokes
                FROM relationship_memory
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,),
            )
            return cursor.fetchone()

    def _insert_row_sync(self, memory: RelationshipMemory) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO relationship_memory (
                    user_id,
                    first_interaction_timestamp,
                    important_dates,
                    shared_memories,
                    conflict_history,
                    inside_jokes,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.user_id,
                    memory.first_interaction_timestamp,
                    json.dumps(memory.important_dates),
                    json.dumps(memory.shared_memories),
                    json.dumps(memory.conflict_history),
                    json.dumps(memory.inside_jokes),
                    utc_now().isoformat(),
                ),
            )
            conn.commit()

    def _update_row_sync(self, memory: RelationshipMemory) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE relationship_memory
                SET
                    important_dates = ?,
                    shared_memories = ?,
                    conflict_history = ?,
                    inside_jokes = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    json.dumps(memory.important_dates),
                    json.dumps(memory.shared_memories),
                    json.dumps(memory.conflict_history),
                    json.dumps(memory.inside_jokes),
                    utc_now().isoformat(),
                    memory.user_id,
                ),
            )
            conn.commit()


def _parse_json_list(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, str)]

