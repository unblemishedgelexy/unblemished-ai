from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_latency_metrics_are_present(brain_factory) -> None:
    harness = brain_factory(overrides={"reasoning_mode": "balanced"})
    request = ChatRequest(
        input_text="Design a modular architecture with clear service boundaries",
        context={"phase": 1},
        trace_id="trace-latency-1",
        user_id="user-latency-1",
    )
    response = asyncio.run(harness.brain.reason(request))

    assert response.processing_time_ms >= 0
    assert response.model_latency_ms >= 0
    assert response.processing_time_ms >= response.model_latency_ms

