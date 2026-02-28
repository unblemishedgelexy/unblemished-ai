from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_tool_invoked_correctly(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={"tool_execution_enabled": True, "tool_max_timeout_seconds": 2.0},
        )
        response = await harness.brain.reason(
            ChatRequest(
                input_text="tool: keyword_extract please extract key terms for modular architecture planning",
                context={"phase": "5"},
                trace_id="trace-tool-1",
                user_id="user-tool-1",
            ),
        )

        assert response.tool_invocations is not None
        assert len(response.tool_invocations) >= 1
        first = response.tool_invocations[0]
        assert first.tool_name == "keyword_extract"
        assert first.success is True
        assert "Tool Outputs:" in response.reflection.final_answer

    asyncio.run(run())

