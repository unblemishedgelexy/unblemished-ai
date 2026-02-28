from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_cost_metrics_populated(brain_factory) -> None:
    harness = brain_factory(
        overrides={
            "reasoning_mode": "balanced",
            "model_routing_enabled": True,
            "self_evaluation_enabled": True,
        },
    )
    request = ChatRequest(
        input_text="Design a scalable modular architecture with memory and evaluation loops",
        context={"phase": "4", "traffic": "high"},
        trace_id="trace-cost",
        user_id="user-cost",
    )
    response = asyncio.run(harness.brain.reason(request))

    assert response.cost_estimate is not None
    assert response.cost_estimate.estimated_token_cost > 0
    assert response.cost_estimate.estimated_inference_cost >= response.cost_estimate.estimated_token_cost
    assert response.cost_estimate.total_latency_breakdown.get("model_ms", 0) >= 0
    assert response.cost_estimate.total_latency_breakdown.get("total_ms", 0) >= response.cost_estimate.total_latency_breakdown.get("model_ms", 0)

