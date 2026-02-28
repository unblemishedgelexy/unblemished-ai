from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_tool_sandbox_blocks_unauthorized_tool(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={"tool_execution_enabled": True, "self_evaluation_enabled": False},
        )
        response = await harness.brain.reason(
            ChatRequest(
                input_text="tool: shell_exec run arbitrary command",
                context={"phase": "5"},
                trace_id="trace-sandbox-1",
                user_id="user-sandbox-1",
            ),
        )

        assert response.tool_invocations is not None
        assert len(response.tool_invocations) >= 1
        first = response.tool_invocations[0]
        assert first.success is False
        assert "whitelist" in first.result_summary.lower()
        assert response.fallback_triggered is False

    asyncio.run(run())

