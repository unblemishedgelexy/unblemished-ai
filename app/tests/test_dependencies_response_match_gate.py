from __future__ import annotations

import asyncio

from app.core.logger import StructuredLogger, setup_logger
from app.services.brain.model_adapter import ModelAdapter, ModelResult


class _JudgeReadyAdapter(ModelAdapter):
    def __init__(self, *, judge_ready: bool) -> None:
        super().__init__(
            logger=StructuredLogger(setup_logger(level="INFO")),
            backend="local_llama",
            local_model_path="dummy.gguf",
        )
        self._judge_ready = judge_ready
        self.calls = 0

    def can_use_model_judge(self) -> bool:
        return self._judge_ready

    async def generate(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, object] | None = None,
        model_name: str = "fast_model",
        max_tokens: int = 96,
        temperature: float = 0.0,
        timeout_seconds: float = 1.2,
        max_retries: int = 0,
    ) -> ModelResult:
        del prompt, trace_id, context, model_name, max_tokens, temperature, timeout_seconds, max_retries
        self.calls += 1
        return ModelResult(
            text='{"match_score": 0.9, "is_match": true, "reason": "judge_override"}',
            latency_ms=1,
            fallback_used=False,
            model_name="fast_model",
            backend="local_llama",
        )


def test_response_match_judge_enabled_for_hybrid_backend(brain_factory) -> None:
    adapter = _JudgeReadyAdapter(judge_ready=True)
    harness = brain_factory(
        overrides={
            "model_backend": "hybrid",
            "response_match_model_enabled": True,
        },
        model_adapter=adapter,
    )
    predictor = harness.container.response_match_predictor
    report = asyncio.run(
        predictor.predict(
            user_input="Need account access recovery steps",
            answer_text="Account issue maybe contact support.",
            intent="troubleshooting",
            trace_id="trace-gate-hybrid",
            user_id="user-gate-hybrid",
        ),
    )

    assert adapter.calls == 1
    assert any(reason.startswith("model_judge:") for reason in report.reasons)


def test_response_match_judge_enabled_for_api_backend(brain_factory) -> None:
    adapter = _JudgeReadyAdapter(judge_ready=True)
    harness = brain_factory(
        overrides={
            "model_backend": "api_llm",
            "response_match_model_enabled": True,
        },
        model_adapter=adapter,
    )
    predictor = harness.container.response_match_predictor
    report = asyncio.run(
        predictor.predict(
            user_input="Need account access recovery steps",
            answer_text="Account issue maybe contact support.",
            intent="troubleshooting",
            trace_id="trace-gate-api",
            user_id="user-gate-api",
        ),
    )

    assert adapter.calls == 1
    assert any(reason.startswith("model_judge:") for reason in report.reasons)


def test_response_match_judge_disabled_when_config_false(brain_factory) -> None:
    adapter = _JudgeReadyAdapter(judge_ready=True)
    harness = brain_factory(
        overrides={
            "model_backend": "hybrid",
            "response_match_model_enabled": False,
        },
        model_adapter=adapter,
    )
    predictor = harness.container.response_match_predictor
    report = asyncio.run(
        predictor.predict(
            user_input="Need account access recovery steps",
            answer_text="Account issue maybe contact support.",
            intent="troubleshooting",
            trace_id="trace-gate-disabled",
            user_id="user-gate-disabled",
        ),
    )

    assert adapter.calls == 0
    assert not any(reason.startswith("model_judge:") for reason in report.reasons)
