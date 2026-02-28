from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_goal_persists_for_same_user(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"tool_execution_enabled": True})
        user_id = "user-goal-1"

        first = await harness.brain.reason(
            ChatRequest(
                input_text="goal: build release automation pipeline with canary rollback",
                context={"phase": "5"},
                trace_id="trace-goal-1",
                user_id=user_id,
            ),
        )
        second = await harness.brain.reason(
            ChatRequest(
                input_text="Refine deployment checks for reliability.",
                context={"phase": "5"},
                trace_id="trace-goal-2",
                user_id=user_id,
            ),
        )

        assert first.goal_snapshot is not None
        assert second.goal_snapshot is not None
        assert second.goal_snapshot.active_goal.startswith("build release automation pipeline")
        assert second.goal_snapshot.completion_status in {"in-progress", "blocked"}

    asyncio.run(run())

