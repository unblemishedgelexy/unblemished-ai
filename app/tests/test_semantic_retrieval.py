from __future__ import annotations

import asyncio

from app.services.memory.memory_retriever import MemoryRetriever


def test_semantic_similarity_ranking_works(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"embedding_dim": 48, "embedding_enabled": True})
        retriever = MemoryRetriever(
            store=harness.memory_store,
            logger=harness.logger,
            embedding_interface=harness.embedding_interface,
        )

        exact = await harness.memory_store.store_memory(
            user_id="user-semantic",
            trace_id="trace-semantic-seed-1",
            summary_text="kubernetes autoscaling hpa policy",
        )
        await harness.memory_store.store_memory(
            user_id="user-semantic",
            trace_id="trace-semantic-seed-2",
            summary_text="kubernetes autoscaling hpa policy with cost guardrails",
        )

        ranked = await retriever.retrieve(
            user_id="user-semantic",
            trace_id="trace-semantic-query",
            query_text="kubernetes autoscaling hpa policy",
            top_k=1,
        )

        assert len(ranked) == 1
        assert ranked[0].memory_id == exact.memory_id

    asyncio.run(run())

