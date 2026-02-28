from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from app.core.logger import StructuredLogger
from app.repositories.memory_repository import MemoryRepository
from app.services.embeddings.embedding_interface import EmbeddingInterface
from app.services.memory.policies.memory_decay_policy import MemoryDecayPolicy
from app.services.memory.policies.memory_importance_policy import MemoryImportancePolicy
from app.utils.helpers import utc_now


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    trace_id: str
    summary_text: str
    created_at: datetime
    embedding: list[float] | None = None
    importance_score: float = 0.5
    action_type: str | None = None
    action_result_summary: str | None = None


class MemoryStore:
    def __init__(
        self,
        repository: MemoryRepository,
        logger: StructuredLogger,
        embedding_interface: EmbeddingInterface,
        importance_policy: MemoryImportancePolicy | None = None,
        decay_policy: MemoryDecayPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._logger = logger
        self._embedding_interface = embedding_interface
        self._importance_policy = importance_policy or MemoryImportancePolicy()
        self._decay_policy = decay_policy or MemoryDecayPolicy()

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def store_memory(
        self,
        user_id: str,
        trace_id: str,
        summary_text: str,
        retrieval_count: int = 0,
        action_type: str | None = None,
        action_result_summary: str | None = None,
        importance_override: float | None = None,
    ) -> MemoryRecord:
        await self.initialize()

        memory_id = str(uuid4())
        created_at = utc_now()
        embedding_vector = await self._embedding_interface.generate_embedding(
            text=summary_text,
            trace_id=trace_id,
            user_id=user_id,
        )
        prior_memories = await self.fetch_user_memories(
            user_id=user_id,
            limit=25,
            trace_id=trace_id,
        )
        importance_score = self._importance_policy.score(
            summary_text=summary_text,
            prior_memories=prior_memories,
        )
        if importance_override is not None:
            importance_score = _clamp_importance(importance_override)

        await self._repository.insert_memory(
            memory_id=memory_id,
            user_id=user_id,
            trace_id=trace_id,
            summary_text=summary_text,
            created_at=created_at,
            embedding_json=json.dumps(embedding_vector) if embedding_vector is not None else None,
            importance_score=importance_score,
            action_type=action_type,
            action_result_summary=action_result_summary,
        )
        self._logger.info(
            "memory.store.completed",
            trace_id=trace_id,
            user_id=user_id,
            memory_id=memory_id,
            retrieval_count=retrieval_count,
            importance_score=importance_score,
            action_type=action_type or "conversation-summary",
        )
        return MemoryRecord(
            memory_id=memory_id,
            user_id=user_id,
            trace_id=trace_id,
            summary_text=summary_text,
            created_at=created_at,
            embedding=embedding_vector,
            importance_score=importance_score,
            action_type=action_type,
            action_result_summary=action_result_summary,
        )

    async def fetch_user_memories(
        self,
        user_id: str,
        limit: int = 100,
        trace_id: str = "system",
    ) -> list[MemoryRecord]:
        await self.initialize()
        rows = await self._repository.fetch_user_memories(user_id=user_id, limit=limit)
        records: list[MemoryRecord] = []
        for row in rows:
            decayed_importance = self._decay_policy.apply(
                base_importance=row.importance_score,
                created_at=row.created_at,
            )
            records.append(
                MemoryRecord(
                    memory_id=row.memory_id,
                    user_id=row.user_id,
                    trace_id=row.trace_id,
                    summary_text=row.summary_text,
                    created_at=row.created_at,
                    embedding=_load_embedding(row.embedding),
                    importance_score=decayed_importance,
                    action_type=row.action_type,
                    action_result_summary=row.action_result_summary,
                ),
            )
        self._logger.info(
            "memory.store.fetch.completed",
            trace_id=trace_id,
            user_id=user_id,
            memory_id=records[0].memory_id if records else "none",
            retrieval_count=len(records),
            limit=limit,
        )
        return records

    async def count_entries(self) -> int:
        await self.initialize()
        return await self._repository.count_entries()

    async def is_ready(self) -> bool:
        return await self._repository.is_ready()


def _load_embedding(raw_embedding: str | None) -> list[float] | None:
    if raw_embedding is None:
        return None
    try:
        payload = json.loads(raw_embedding)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    output: list[float] = []
    for item in payload:
        if isinstance(item, (int, float)):
            output.append(float(item))
    return output if output else None


def _clamp_importance(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 4)
