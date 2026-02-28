from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_different_user_memory_isolated(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"memory_top_k": 3})

        await harness.brain.reason(
            ChatRequest(
                input_text="Design deployment rollback policy for blue green systems.",
                context={"domain": "platform"},
                trace_id="trace-mem-a-seed",
                user_id="user-a",
            ),
        )
        await asyncio.sleep(0.08)

        user_b_response = await harness.brain.reason(
            ChatRequest(
                input_text="Refine deployment rollback policy.",
                context={"domain": "platform"},
                trace_id="trace-mem-b-query",
                user_id="user-b",
            ),
        )
        assert len(user_b_response.context_used) == 0

    asyncio.run(run())


def test_top_k_limit_respected(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"memory_top_k": 2})
        user_id = "user-top-k"

        for index in range(4):
            await harness.brain.reason(
                ChatRequest(
                    input_text=f"Store topic {index} for deployment memory context",
                    context={"index": index},
                    trace_id=f"trace-topk-seed-{index}",
                    user_id=user_id,
                ),
            )
            await asyncio.sleep(0.05)

        response = await harness.brain.reason(
            ChatRequest(
                input_text="retrieve deployment memory context for topk test",
                context={},
                trace_id="trace-topk-query",
                user_id=user_id,
            ),
        )
        assert len(response.context_used) <= 2

    asyncio.run(run())

