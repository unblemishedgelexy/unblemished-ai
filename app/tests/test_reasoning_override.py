from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_adaptive_reasoning_mode_override_to_deep(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"reasoning_mode": "balanced"})
        user_id = "user-override"

        profile = await harness.profile_store.get_profile(user_id=user_id, trace_id="trace-profile-seed")
        profile.conversation_depth_preference = "deep"
        await harness.profile_store.upsert_profile(profile=profile, trace_id="trace-profile-seed")

        response = await harness.brain.reason(
            ChatRequest(
                input_text=(
                    "Design and compare a modular distributed architecture and orchestrate fallback paths "
                    "and error taxonomy, then compare parallel execution strategies and cost envelopes, "
                    "because multiple teams will share ownership while strict contracts must hold and "
                    "cross-region recovery and capacity planning and release governance and audit controls "
                    "must be analyzed together."
                ),
                context={
                    "phase": "4",
                    "scope": "platform",
                    "priority": "high",
                    "traffic": "global",
                    "compliance": "strict",
                    "multi_region": True,
                    "budget": "sensitive",
                    "team_count": 5,
                },
                trace_id="trace-override-1",
                user_id=user_id,
            ),
        )

        assert response.complexity_score is not None
        assert response.complexity_score > 0.7
        assert response.reasoning_mode == "deep"

    asyncio.run(run())
