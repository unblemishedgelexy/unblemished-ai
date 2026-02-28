from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_profile_updates_after_interaction(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"reasoning_mode": "balanced"})
        response = await harness.brain.reason(
            ChatRequest(
                input_text="bhai help me design rollout strategy for kubernetes",
                context={"domain": "platform"},
                trace_id="trace-profile-1",
                user_id="user-profile-1",
            ),
        )

        assert response.user_profile_snapshot is not None
        assert response.user_profile_snapshot.preferred_tone == "casual"
        assert response.user_profile_snapshot.conversation_depth_preference == "balanced"

    asyncio.run(run())

