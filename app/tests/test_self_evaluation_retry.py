from __future__ import annotations

import asyncio
from typing import Any

from app.core.logger import StructuredLogger, setup_logger
from app.schemas.request_schema import ChatRequest
from app.services.brain.model_adapter import ModelAdapter, ModelResult


class FlakyModelAdapter(ModelAdapter):
    def __init__(self, logger: StructuredLogger) -> None:
        super().__init__(logger=logger)
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, Any] | None = None,
        model_name: str = "creative_model",
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_seconds: float = 4.0,
        max_retries: int = 2,
    ) -> ModelResult:
        self.calls += 1
        if self.calls == 1:
            return ModelResult(
                text="Structured Reasoning Output\n- Short draft.",
                latency_ms=1,
                fallback_used=False,
                model_name=model_name,
            )
        return ModelResult(
            text=(
                "Structured Reasoning Output\n"
                "- Core Analysis: Break the problem into intent parsing, memory context assembly, and response guarantees.\n"
                "- Proposed Action: Run modular services with strict schema validation and fallback handling.\n"
                "- Implementation Note: Keep transport and orchestration separated."
            ),
            latency_ms=1,
            fallback_used=False,
            model_name=model_name,
        )


def test_self_evaluation_retry_happens_once(brain_factory) -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    flaky_adapter = FlakyModelAdapter(logger=logger)
    harness = brain_factory(
        overrides={"self_evaluation_enabled": True, "model_routing_enabled": True},
        model_adapter=flaky_adapter,
    )
    request = ChatRequest(
        input_text="Design an architecture plan with strict contracts",
        context={"phase": "4"},
        trace_id="trace-self-eval",
        user_id="user-self-eval",
    )

    response = asyncio.run(harness.brain.reason(request))

    assert response.evaluation_scores is not None
    assert response.evaluation_scores.retry_count == 1
    assert flaky_adapter.calls == 2
    assert response.fallback_triggered is False

