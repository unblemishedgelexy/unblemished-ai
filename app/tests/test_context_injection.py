from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_context_injected_in_prompt(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={
                "memory_top_k": 3,
                "expose_full_prompt_in_response": True,
            },
        )
        user_id = "user-context-injection"

        await harness.brain.reason(
            ChatRequest(
                input_text="Create payment retry policy with exponential backoff.",
                context={"domain": "payments"},
                trace_id="trace-context-seed",
                user_id=user_id,
            ),
        )
        await asyncio.sleep(0.08)

        response = await harness.brain.reason(
            ChatRequest(
                input_text="Refine payment retry policy and include circuit breaker.",
                context={"domain": "payments"},
                trace_id="trace-context-query",
                user_id=user_id,
            ),
        )

        assert "Relevant Past Context:" in response.prompt
        assert len(response.context_used) >= 1
        assert "score=" in response.prompt

    asyncio.run(run())
