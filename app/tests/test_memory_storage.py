from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_same_user_remembers_prior_topic(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"memory_top_k": 3})
        user_id = "user-memory-1"

        first_response = await harness.brain.reason(
            ChatRequest(
                input_text="Design retry strategy for payment gateway.",
                context={"domain": "payments"},
                trace_id="trace-memory-seed",
                user_id=user_id,
            ),
        )
        assert len(first_response.context_used) == 0

        await asyncio.sleep(0.08)

        second_response = await harness.brain.reason(
            ChatRequest(
                input_text="Refine retry strategy with circuit breaker.",
                context={"domain": "payments"},
                trace_id="trace-memory-query",
                user_id=user_id,
            ),
        )
        assert len(second_response.context_used) >= 1

    asyncio.run(run())

