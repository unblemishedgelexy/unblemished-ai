from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_brain_returns_structured_response(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    request = ChatRequest(
        input_text="Design a modular AI service with clean architecture",
        context={"phase": 1},
        trace_id="trace-test-1",
        user_id="user-a",
    )
    response = asyncio.run(harness.brain.reason(request))

    assert response.trace_id == "trace-test-1"
    assert response.intent == "solution-design"
    assert response.prompt_version
    assert response.reasoning_mode == "balanced"
    assert response.processing_time_ms >= response.model_latency_ms
    assert len(response.reasoning_steps) >= 3
    assert 0.0 <= response.reflection.confidence <= 1.0
    assert 0.0 <= response.reflection.metadata.clarity_score <= 1.0
    assert isinstance(response.context_used, list)
    assert response.user_profile_snapshot is not None
    assert response.reflection.final_answer.startswith("Structured Reasoning Output")


def test_brain_greeting_is_not_stale_architecture_response(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    request = ChatRequest(
        input_text="hello",
        context={"phase": 1},
        trace_id="trace-test-greeting-1",
        user_id="user-greeting",
    )
    response = asyncio.run(harness.brain.reason(request))
    answer = response.reflection.final_answer.lower()

    assert "hello" in answer
    assert "build a modular fastapi architecture with transport-only routes" not in answer


def test_brain_small_talk_is_not_stale_architecture_response(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    request = ChatRequest(
        input_text="kya chal raha hai",
        context={"phase": 1},
        trace_id="trace-test-smalltalk-1",
        user_id="user-smalltalk",
    )
    response = asyncio.run(harness.brain.reason(request))
    answer = response.reflection.final_answer.lower()

    assert "running fine" in answer
    assert response.fallback_triggered is False
    assert "build a modular fastapi architecture with transport-only routes" not in answer


def test_brain_flirty_mode_and_name_hint_are_respected(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    request = ChatRequest(
        input_text="kya chal raha hai",
        context={
            "answer_style": "flirty",
            "name": "Kanchana",
            "phase": 1,
        },
        trace_id="trace-test-flirty-1",
        user_id="user-flirty",
    )
    response = asyncio.run(harness.brain.reason(request))
    answer = response.reflection.final_answer.lower()

    assert "kanchana" in answer
    assert "playful" in answer
    assert response.fallback_triggered is False
