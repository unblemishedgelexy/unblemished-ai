from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_same_input_consistency(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    request = ChatRequest(
        input_text="Design modular AI architecture with service boundaries",
        context={"phase": 1},
        trace_id="trace-consistency-1",
        user_id="user-consistency",
    )
    first = asyncio.run(harness.brain.reason(request))
    second = asyncio.run(harness.brain.reason(request))

    assert first.intent == second.intent
    assert first.prompt_version == second.prompt_version
    assert first.reflection.final_answer.startswith("Structured Reasoning Output")
    assert second.reflection.final_answer.startswith("Structured Reasoning Output")


def test_deep_vs_fast_mode_difference(brain_factory) -> None:
    fast_harness = brain_factory(overrides={"reasoning_mode": "fast"})
    deep_harness = brain_factory(overrides={"reasoning_mode": "deep"})

    fast_response = asyncio.run(
        fast_harness.brain.reason(
            ChatRequest(
                input_text="Design modular AI architecture",
                context={"phase": 1},
                trace_id="trace-fast",
                user_id="user-depth",
            ),
        ),
    )
    deep_response = asyncio.run(
        deep_harness.brain.reason(
            ChatRequest(
                input_text="Design modular AI architecture",
                context={"phase": 1},
                trace_id="trace-deep",
                user_id="user-depth",
            ),
        ),
    )

    assert fast_response.reasoning_mode == "fast"
    assert deep_response.reasoning_mode == "deep"
    assert len(deep_response.reasoning_steps) > len(fast_response.reasoning_steps)
    assert deep_response.prompt != fast_response.prompt


def test_balanced_mode_has_reflection(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    response = asyncio.run(
        harness.brain.reason(
            ChatRequest(
                input_text="Design modular AI architecture",
                context={"phase": 1},
                trace_id="trace-balanced",
                user_id="user-balanced",
            ),
        ),
    )

    assert response.reasoning_mode == "balanced"
    assert response.reflection.metadata.reflection_pass_enabled is True
    assert 0.0 <= response.reflection.metadata.logical_consistency_score <= 1.0

