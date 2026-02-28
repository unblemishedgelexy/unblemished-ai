from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_repeated_correction_learns_skill_and_executes_before_model(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={"tool_execution_enabled": True, "self_evaluation_enabled": False},
        )
        user_id = "user-skill-1"

        correction = 'learn skill: "flash-scan" => context_digest'
        await harness.brain.reason(
            ChatRequest(
                input_text=correction,
                context={"phase": "learn"},
                trace_id="trace-skill-learn-1",
                user_id=user_id,
            ),
        )
        await harness.brain.reason(
            ChatRequest(
                input_text=correction,
                context={"phase": "learn"},
                trace_id="trace-skill-learn-2",
                user_id=user_id,
            ),
        )

        learned = await harness.skill_interface._repository.get_skill(  # noqa: SLF001
            user_id=user_id,
            trigger_text="flash-scan",
            tool_name="context_digest",
        )
        assert learned is not None
        assert learned.active is True
        assert learned.correction_count >= 2

        response = await harness.brain.reason(
            ChatRequest(
                input_text="please flash-scan this request now",
                context={"region": "us"},
                trace_id="trace-skill-run-1",
                user_id=user_id,
            ),
        )
        assert response.tool_invocations is not None
        assert len(response.tool_invocations) >= 1
        assert response.tool_invocations[0].tool_name == "context_digest"
        assert response.model_latency_ms == 0
        assert "skill trigger match" in response.reflection.final_answer.lower()

    asyncio.run(run())


def test_skill_learning_blocks_unsafe_tool_name(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(
            overrides={"tool_execution_enabled": True},
        )
        user_id = "user-skill-unsafe"
        correction = 'learn skill: "boom-cmd" => shell_exec'

        await harness.brain.reason(
            ChatRequest(
                input_text=correction,
                trace_id="trace-skill-unsafe-1",
                user_id=user_id,
            ),
        )
        await harness.brain.reason(
            ChatRequest(
                input_text=correction,
                trace_id="trace-skill-unsafe-2",
                user_id=user_id,
            ),
        )

        learned = await harness.skill_interface._repository.get_skill(  # noqa: SLF001
            user_id=user_id,
            trigger_text="boom-cmd",
            tool_name="shell_exec",
        )
        assert learned is None

    asyncio.run(run())


def test_no_skill_match_continues_to_model_path(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"self_evaluation_enabled": False})
        response = await harness.brain.reason(
            ChatRequest(
                input_text="Explain throughput tuning basics in distributed systems.",
                trace_id="trace-skill-fallback-1",
                user_id="user-skill-fallback",
            ),
        )
        assert response.model_latency_ms > 0

    asyncio.run(run())
