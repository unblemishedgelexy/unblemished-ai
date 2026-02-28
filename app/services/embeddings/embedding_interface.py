from __future__ import annotations

from app.core.logger import StructuredLogger
from app.services.embeddings.embedding_adapter import EmbeddingAdapter


class EmbeddingInterface:
    def __init__(
        self,
        adapter: EmbeddingAdapter,
        logger: StructuredLogger,
        enabled: bool,
    ) -> None:
        self._adapter = adapter
        self._logger = logger
        self._enabled = enabled

    async def generate_embedding(
        self,
        text: str,
        trace_id: str,
        user_id: str,
    ) -> list[float] | None:
        if not self._enabled:
            self._logger.info(
                "embedding.generate.skipped",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                reason="embedding_disabled",
            )
            return None

        embedding = await self._adapter.embed(text)
        self._logger.info(
            "embedding.generate.completed",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            embedding_dim=len(embedding),
        )
        return embedding

    def is_enabled(self) -> bool:
        return self._enabled

    async def is_ready(self) -> bool:
        return True

