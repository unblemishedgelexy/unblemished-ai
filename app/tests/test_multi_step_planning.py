from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_multi_step_plan_generated(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"tool_execution_enabled": True})
        response = await harness.brain.reason(
            ChatRequest(
                input_text="tool: context_digest build a modular execution plan",
                context={"phase": "5"},
                trace_id="trace-plan-1",
                user_id="user-plan-1",
            ),
        )

        assert response.execution_plan is not None
        assert len(response.execution_plan) >= 3
        assert any(step.requires_tool for step in response.execution_plan)
        assert any(
            step.required_tool_name == "context_digest"
            for step in response.execution_plan
            if step.requires_tool
        )

    asyncio.run(run())

