from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta

from app.services.memory.memory_retriever import MemoryRetriever
from app.utils.helpers import utc_now


def test_importance_score_influences_ranking_with_decay(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory()
        retriever = MemoryRetriever(
            store=harness.memory_store,
            logger=harness.logger,
            embedding_interface=harness.embedding_interface,
        )

        high = await harness.memory_store.store_memory(
            user_id="user-importance",
            trace_id="trace-importance-high",
            summary_text="important critical architecture decision focus",
        )
        await harness.memory_store.store_memory(
            user_id="user-importance",
            trace_id="trace-importance-low",
            summary_text="general update",
        )

        old_ts = (utc_now() - timedelta(days=20)).isoformat()
        with sqlite3.connect(harness.settings.memory_db_path) as conn:
            conn.execute(
                "UPDATE memory_table SET created_at = ? WHERE id = ?",
                (old_ts, high.memory_id),
            )
            conn.commit()

        ranked = await retriever.retrieve(
            user_id="user-importance",
            trace_id="trace-importance-query",
            query_text="architecture decision",
            top_k=1,
        )
        assert len(ranked) == 1
        assert ranked[0].memory_id == high.memory_id
        assert ranked[0].relevance_score >= 0.0

    asyncio.run(run())

