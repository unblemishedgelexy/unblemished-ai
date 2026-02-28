from __future__ import annotations

import math
from datetime import datetime

from app.utils.helpers import utc_now


class MemoryRankingPolicy:
    def keyword_overlap_score(self, query_tokens: set[str], memory_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & memory_tokens)
        return overlap / max(len(query_tokens), 1)

    def semantic_similarity(
        self,
        query_embedding: list[float] | None,
        memory_embedding: list[float] | None,
    ) -> float:
        if not query_embedding or not memory_embedding:
            return 0.0

        length = min(len(query_embedding), len(memory_embedding))
        if length == 0:
            return 0.0

        dot = 0.0
        query_norm = 0.0
        memory_norm = 0.0
        for idx in range(length):
            q = query_embedding[idx]
            m = memory_embedding[idx]
            dot += q * m
            query_norm += q * q
            memory_norm += m * m

        denom = math.sqrt(query_norm) * math.sqrt(memory_norm)
        if denom == 0:
            return 0.0

        cosine = dot / denom
        return max(0.0, min((cosine + 1.0) / 2.0, 1.0))

    def recentness_boost(self, created_at: datetime) -> float:
        age_seconds = max((utc_now() - created_at).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        return 1.0 / (1.0 + age_days)

    def combine(
        self,
        *,
        keyword_score: float,
        semantic_similarity: float,
        recency_boost: float,
        importance_score: float,
    ) -> float:
        base_score = keyword_score * 0.4 + semantic_similarity * 0.5 + recency_boost * 0.1
        importance_factor = 0.75 + 0.25 * max(0.0, min(importance_score, 1.0))
        return round(min(1.0, base_score * importance_factor), 4)

