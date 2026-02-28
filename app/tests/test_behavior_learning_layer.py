from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest
from app.services.brain.orchestrator import _extract_behavior_correction


def test_correction_phrase_detection() -> None:
    cases = [
        ("wrong: use exponential backoff with jitter", "wrong", "use exponential backoff with jitter"),
        ("not that, prioritize canary rollback steps", "not that", "prioritize canary rollback steps"),
        ("I meant add retry budget and timeout caps", "I meant", "add retry budget and timeout caps"),
        (
            "no, do this instead: generate security checklist with rbac and audit logs",
            "no, do this instead",
            "generate security checklist with rbac and audit logs",
        ),
    ]
    for text, expected_phrase, expected_instruction in cases:
        event = _extract_behavior_correction(text)
        assert event is not None
        assert event.trigger_phrase == expected_phrase
        assert event.corrected_instruction == expected_instruction


def test_behavior_correction_saved_high_priority_and_retrieved(brain_factory) -> None:
    async def run() -> None:
        harness = brain_factory(overrides={"memory_top_k": 3})
        user_id = "user-behavior-correction-1"

        await harness.brain.reason(
            ChatRequest(
                input_text="Generate onboarding checklist for Kubernetes cluster rollout.",
                trace_id="trace-behavior-seed",
                user_id=user_id,
            ),
        )

        corrected_instruction = "generate onboarding checklist focused on security hardening and rbac"
        await harness.brain.reason(
            ChatRequest(
                input_text=f"no, do this instead: {corrected_instruction}",
                trace_id="trace-behavior-correction",
                user_id=user_id,
            ),
        )

        correction_memory = await _wait_for_correction_memory(
            harness=harness,
            user_id=user_id,
        )
        assert correction_memory is not None
        assert correction_memory.importance_score >= 0.99
        assert correction_memory.action_type == "behavior:correction"
        assert "OriginalInput: Generate onboarding checklist for Kubernetes cluster rollout." in correction_memory.summary_text
        assert "CorrectedInstruction: generate onboarding checklist focused on security hardening and rbac" in correction_memory.summary_text

        follow_up = await harness.brain.reason(
            ChatRequest(
                input_text="Generate onboarding checklist for Kubernetes cluster rollout.",
                trace_id="trace-behavior-followup",
                user_id=user_id,
            ),
        )
        used_ids = {item.memory_id for item in follow_up.context_used}
        assert correction_memory.memory_id in used_ids

    asyncio.run(run())


async def _wait_for_correction_memory(*, harness, user_id: str):
    for _ in range(40):
        memories = await harness.memory_store.fetch_user_memories(
            user_id=user_id,
            limit=40,
            trace_id="trace-behavior-wait",
        )
        for memory in memories:
            if memory.action_type == "behavior:correction":
                return memory
        await asyncio.sleep(0.05)
    return None
