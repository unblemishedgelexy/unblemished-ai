from __future__ import annotations

import asyncio

from app.schemas.request_schema import ChatRequest


def test_model_routing_changes_selected_model(brain_factory) -> None:
    harness = brain_factory(overrides={"model_routing_enabled": True})

    fast_response = asyncio.run(
        harness.brain.reason(
            ChatRequest(
                input_text="What is modular architecture?",
                context={},
                trace_id="trace-routing-fast",
                user_id="user-routing",
            ),
        ),
    )
    creative_response = asyncio.run(
        harness.brain.reason(
            ChatRequest(
                input_text="Design a modular architecture for a cognitive AI platform",
                context={"phase": "4"},
                trace_id="trace-routing-creative",
                user_id="user-routing",
            ),
        ),
    )

    assert fast_response.cost_estimate is not None
    assert creative_response.cost_estimate is not None
    assert fast_response.cost_estimate.routed_model == "fast_model"
    assert creative_response.cost_estimate.routed_model == "creative_model"

