from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from app.core.logger import StructuredLogger
from app.services.embeddings.embedding_interface import EmbeddingInterface
from app.services.memory.memory_store import MemoryRecord, MemoryStore
from app.services.memory.policies.memory_ranking_policy import MemoryRankingPolicy


@dataclass(slots=True)
class RetrievedMemory:
    memory_id: str
    summary_text: str
    relevance_score: float
    created_at: datetime
    importance_score: float
    action_type: str | None = None


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        logger: StructuredLogger,
        embedding_interface: EmbeddingInterface,
        ranking_policy: MemoryRankingPolicy | None = None,
    ) -> None:
        self._store = store
        self._logger = logger
        self._embedding_interface = embedding_interface
        self._ranking_policy = ranking_policy or MemoryRankingPolicy()

    async def retrieve(
        self,
        user_id: str,
        trace_id: str,
        query_text: str,
        top_k: int,
    ) -> list[RetrievedMemory]:
        candidate_limit = max(top_k * 8, 20)
        candidates = await self._store.fetch_user_memories(
            user_id=user_id,
            limit=candidate_limit,
            trace_id=trace_id,
        )
        if not candidates:
            self._logger.info(
                "memory.retrieve.completed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="none",
                retrieval_count=0,
                top_k=top_k,
            )
            return []

        query_tokens = _tokenize(query_text)
        query_embedding = await self._embedding_interface.generate_embedding(
            text=query_text,
            trace_id=trace_id,
            user_id=user_id,
        )
        faiss_semantic_scores, faiss_indexed_count = self._semantic_scores_with_faiss(
            query_embedding=query_embedding,
            candidates=candidates,
            top_n=candidate_limit,
        )
        if faiss_indexed_count > 0:
            self._logger.info(
                "memory.retrieve.faiss.semantic",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="none",
                retrieval_count=faiss_indexed_count,
                top_k=top_k,
            )

        scored = [
            self._score_memory(
                record=record,
                query_tokens=query_tokens,
                query_embedding=query_embedding,
                semantic_override=faiss_semantic_scores.get(record.memory_id),
            )
            for record in candidates
        ]
        scored.sort(key=lambda item: (item.relevance_score, item.importance_score, item.created_at), reverse=True)

        selected = scored[:top_k]
        self._logger.info(
            "memory.retrieve.completed",
            trace_id=trace_id,
            user_id=user_id,
            memory_id=",".join(item.memory_id for item in selected) if selected else "none",
            retrieval_count=len(selected),
            top_k=top_k,
        )
        return selected

    def _score_memory(
        self,
        record: MemoryRecord,
        query_tokens: set[str],
        query_embedding: list[float] | None,
        semantic_override: float | None,
    ) -> RetrievedMemory:
        memory_tokens = _tokenize(record.summary_text)
        keyword_score = self._ranking_policy.keyword_overlap_score(query_tokens, memory_tokens)
        semantic_similarity = (
            semantic_override
            if semantic_override is not None
            else self._ranking_policy.semantic_similarity(
                query_embedding=query_embedding,
                memory_embedding=record.embedding,
            )
        )
        recency_boost = self._ranking_policy.recentness_boost(record.created_at)
        final_score = self._ranking_policy.combine(
            keyword_score=keyword_score,
            semantic_similarity=semantic_similarity,
            recency_boost=recency_boost,
            importance_score=record.importance_score,
        )

        return RetrievedMemory(
            memory_id=record.memory_id,
            summary_text=record.summary_text,
            relevance_score=final_score,
            created_at=record.created_at,
            importance_score=record.importance_score,
            action_type=record.action_type,
        )

    def _semantic_scores_with_faiss(
        self,
        *,
        query_embedding: list[float] | None,
        candidates: list[MemoryRecord],
        top_n: int,
    ) -> tuple[dict[str, float], int]:
        if not query_embedding:
            return {}, 0
        query_dim = len(query_embedding)
        if query_dim == 0:
            return {}, 0

        memory_ids: list[str] = []
        vectors: list[list[float]] = []
        for record in candidates:
            if not record.embedding:
                continue
            if len(record.embedding) != query_dim:
                continue
            memory_ids.append(record.memory_id)
            vectors.append(record.embedding)
        if not vectors:
            return {}, 0

        try:
            import faiss  # type: ignore
            import numpy as np
        except Exception:
            return {}, 0

        matrix = np.asarray(vectors, dtype="float32")
        query = np.asarray([query_embedding], dtype="float32")
        if matrix.ndim != 2 or query.ndim != 2:
            return {}, 0

        faiss.normalize_L2(matrix)
        faiss.normalize_L2(query)
        index = faiss.IndexFlatIP(query_dim)
        index.add(matrix)
        top_n = min(max(top_n, 1), len(memory_ids))
        distances, indices = index.search(query, top_n)

        output: dict[str, float] = {}
        for idx, distance in zip(indices[0], distances[0]):
            index_id = int(idx)
            if index_id < 0 or index_id >= len(memory_ids):
                continue
            output[memory_ids[index_id]] = round(_cosine_to_unit(float(distance)), 4)
        return output, len(memory_ids)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine_to_unit(value: float) -> float:
    # FAISS IP over normalized vectors approximates cosine in [-1, 1].
    return max(0.0, min((value + 1.0) / 2.0, 1.0))
