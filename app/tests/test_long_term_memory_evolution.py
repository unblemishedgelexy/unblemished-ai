from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_long_term_summary_trigger_and_format(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={
                "memory_long_term_every_n_messages": 2,
                "memory_top_k": 4,
            },
        )
        user_id = "user-long-term-trigger"

        await harness.brain.reason(
            ChatRequest(
                input_text="I want a migration plan for payment retries and timeout policies.",
                trace_id="trace-long-term-1",
                user_id=user_id,
            ),
        )
        await harness.brain.reason(
            ChatRequest(
                input_text="Also keep rollback constraints and observability checkpoints.",
                trace_id="trace-long-term-2",
                user_id=user_id,
            ),
        )

        long_term = await _wait_for_long_term_memories(
            harness=harness,
            user_id=user_id,
            minimum_count=1,
        )
        assert len(long_term) >= 1
        latest = long_term[0]
        assert latest.action_type == "long_term:summary"
        assert "MemoryType: long_term_summary" in latest.summary_text
        assert "KeyKnowledge:" in latest.summary_text
        assert len(latest.summary_text.split()) <= harness.settings.memory_max_summary_tokens

    asyncio.run(run())


def test_long_term_summary_deduplicates(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={
                "memory_long_term_every_n_messages": 2,
                "memory_top_k": 4,
            },
        )
        user_id = "user-long-term-dedup"
        first = "Need API gateway reliability with retry budget and circuit breaker."
        second = "Need API gateway reliability with retry budget and circuit breaker."

        await harness.brain.reason(
            ChatRequest(
                input_text=first,
                trace_id="trace-long-term-dedup-1",
                user_id=user_id,
            ),
        )
        await harness.brain.reason(
            ChatRequest(
                input_text=second,
                trace_id="trace-long-term-dedup-2",
                user_id=user_id,
            ),
        )
        first_batch = await _wait_for_long_term_memories(
            harness=harness,
            user_id=user_id,
            minimum_count=1,
        )
        first_count = len(first_batch)

        await harness.brain.reason(
            ChatRequest(
                input_text=first,
                trace_id="trace-long-term-dedup-3",
                user_id=user_id,
            ),
        )
        await harness.brain.reason(
            ChatRequest(
                input_text=second,
                trace_id="trace-long-term-dedup-4",
                user_id=user_id,
            ),
        )
        await asyncio.sleep(0.2)
        second_batch = await _long_term_memories(harness=harness, user_id=user_id)
        assert len(second_batch) == first_count

    asyncio.run(run())


def test_long_term_memory_injected_into_prompt(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={
                "memory_long_term_every_n_messages": 2,
                "memory_top_k": 4,
                "expose_full_prompt_in_response": True,
            },
        )
        user_id = "user-long-term-injection"

        await harness.brain.reason(
            ChatRequest(
                input_text="My preference: always include rollback and audit logs in deployment plans.",
                trace_id="trace-long-term-inject-1",
                user_id=user_id,
            ),
        )
        await harness.brain.reason(
            ChatRequest(
                input_text="My constraint: production changes must include staged rollout checkpoints.",
                trace_id="trace-long-term-inject-2",
                user_id=user_id,
            ),
        )

        long_term = await _wait_for_long_term_memories(
            harness=harness,
            user_id=user_id,
            minimum_count=1,
        )
        long_term_ids = {item.memory_id for item in long_term}

        response = await harness.brain.reason(
            ChatRequest(
                input_text="Create production deployment plan with safe rollout.",
                trace_id="trace-long-term-inject-3",
                user_id=user_id,
            ),
        )
        used_ids = {item.memory_id for item in response.context_used}
        assert bool(long_term_ids & used_ids)
        assert "[LONG_TERM]" in response.prompt

    asyncio.run(run())


async def _wait_for_long_term_memories(
    *,
    harness,
    user_id: str,
    minimum_count: int,
) -> list:
    for _ in range(50):
        rows = await _long_term_memories(harness=harness, user_id=user_id)
        if len(rows) >= minimum_count:
            return rows
        await asyncio.sleep(0.05)
    return await _long_term_memories(harness=harness, user_id=user_id)


async def _long_term_memories(*, harness, user_id: str) -> list:
    rows = await harness.memory_store.fetch_user_memories(
        user_id=user_id,
        limit=100,
        trace_id="trace-long-term-read",
    )
    return [item for item in rows if item.action_type == "long_term:summary"]
