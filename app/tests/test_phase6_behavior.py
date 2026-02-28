from __future__ import annotations

import asyncio
import re
from typing import Any

from app.core.logger import StructuredLogger, setup_logger
from app.schemas.request_schema import ChatRequest
from app.services.behavior.directness_enforcer import DirectnessEnforcer
from app.services.behavior.human_coherence_scorer import HumanCoherenceScorer
from app.services.behavior.response_match_predictor import ResponseMatchPredictor
from app.services.behavior.response_style_engine import ResponseStyleEngine
from app.services.behavior.topic_alignment_checker import TopicAlignmentChecker
from app.services.behavior.vagueness_detector import VaguenessDetector
from app.services.brain.model_adapter import ModelAdapter, ModelResult
from app.services.brain.response_post_processor import ResponsePostProcessor
from app.services.intent.intent_analyzer import IntentAnalyzer


class VagueThenClearModelAdapter(ModelAdapter):
    def __init__(self, logger) -> None:
        super().__init__(logger=logger)
        self.calls = 0
        self.prompts: list[str] = []

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
        self.prompts.append(prompt)
        if self.calls == 1:
            return ModelResult(
                text=(
                    "Structured Reasoning Output\n"
                    "- Direct Answer: It depends.\n"
                    "- Core Analysis: Maybe many options are possible.\n"
                    "- Proposed Action: Let me know and I can continue in the next message."
                ),
                latency_ms=2,
                fallback_used=False,
                model_name=model_name,
            )
        return ModelResult(
            text=(
                "Structured Reasoning Output\n"
                "- Direct Answer: Use dependency injection and service boundaries.\n"
                "- Core Analysis: This keeps routes thin, avoids tight coupling, and preserves testability.\n"
                "- Proposed Action: Create interfaces per service and wire concrete implementations in one composition root."
            ),
            latency_ms=2,
            fallback_used=False,
            model_name=model_name,
        )


def test_no_continuation_phrases() -> None:
    processor = DirectnessEnforcer()
    report = processor.inspect(
        answer_text=(
            "It depends. Let me know if you want details. "
            "I can continue in the next message."
        ),
        user_input="How should I structure service modules?",
        strict_response_mode=True,
    )

    assert report.requires_regeneration is True
    assert "next message" not in report.cleaned_text.lower()
    assert "let me know" not in report.cleaned_text.lower()


def test_direct_question_addressing() -> None:
    enforcer = DirectnessEnforcer()
    report = enforcer.inspect(
        answer_text="The weather is pleasant today and people like walking outside.",
        user_input="How do I design a modular FastAPI architecture?",
        strict_response_mode=True,
    )
    assert report.explicit_question_addressed is False
    assert report.requires_regeneration is True


def test_regeneration_on_vagueness(brain_factory) -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = VagueThenClearModelAdapter(logger=logger)
    harness = brain_factory(
        overrides={
            "strict_response_mode": True,
            "self_evaluation_enabled": True,
        },
        model_adapter=adapter,
    )
    request = ChatRequest(
        input_text="How should I design a clean architecture for FastAPI?",
        context={"project": "phase6"},
        trace_id="trace-phase6-vague",
        user_id="user-phase6-vague",
    )
    response = asyncio.run(harness.brain.reason(request))

    assert response.evaluation_scores is not None
    assert response.evaluation_scores.retry_count == 1
    assert adapter.calls == 2
    assert len(adapter.prompts) == 2
    assert "RegenerationDirective:" in adapter.prompts[1]


def test_coherence_scoring() -> None:
    enforcer = DirectnessEnforcer()
    vagueness = VaguenessDetector()
    alignment_checker = TopicAlignmentChecker()
    scorer = HumanCoherenceScorer()

    directness = enforcer.inspect(
        answer_text=(
            "Structured Reasoning Output\n"
            "Answer: Use separate services and DI wiring.\n"
            "Reasoning: This keeps APIs stable and testable."
        ),
        user_input="How should I structure services in FastAPI?",
        strict_response_mode=True,
    )
    vague_report = vagueness.inspect(answer_text=directness.cleaned_text)
    alignment = alignment_checker.inspect(
        user_input="How should I structure services in FastAPI?",
        answer_text=directness.cleaned_text,
        intent="solution-design",
    )
    report = scorer.score(
        answer_text=directness.cleaned_text,
        companion_mode=False,
        directness_report=directness,
        vagueness_report=vague_report,
        topic_alignment_report=alignment,
    )

    assert 0.0 <= report.coherence_score <= 1.0
    assert report.requires_regeneration is False


def test_intent_style_mapping() -> None:
    analyzer = IntentAnalyzer()
    style_engine = ResponseStyleEngine()

    factual_intent = analyzer.analyze("What is clean architecture?")
    technical_intent = analyzer.analyze("Design a modular API service layer")
    emotional_intent = analyzer.analyze("I feel overwhelmed and anxious")
    relational_intent = analyzer.analyze("How can this companion stay grounded in relationship context?")
    strategic_intent = analyzer.analyze("Create a strategy roadmap with priorities")

    assert analyzer.answer_style(factual_intent) == "factual"
    assert analyzer.answer_style(technical_intent) == "technical"
    assert analyzer.answer_style(emotional_intent) == "emotional"
    assert analyzer.answer_style(relational_intent) == "relational"
    assert analyzer.answer_style(strategic_intent) == "strategic"

    style = style_engine.resolve(
        intent=technical_intent,
        user_input="Design a modular API service layer",
        companion_mode=False,
    )
    assert style.style_key == "technical"
    assert style.requires_example is True


def test_requested_style_override_respected() -> None:
    style_engine = ResponseStyleEngine()
    style = style_engine.resolve(
        intent="general-reasoning",
        user_input="hello there",
        companion_mode=True,
        requested_style="flirty",
    )
    assert style.style_key == "flirty"


def test_response_match_predictor_detects_mismatch() -> None:
    predictor = ResponseMatchPredictor()
    report = asyncio.run(
        predictor.predict(
            user_input="What is capital of France?",
            answer_text=(
                "Answer: Build a modular FastAPI architecture with transport-only routes and service engines."
            ),
            intent="question-answering",
        ),
    )

    assert report.is_match is False
    assert report.score < 0.45


class StubJudgeModelAdapter(ModelAdapter):
    async def generate(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, Any] | None = None,
        model_name: str = "fast_model",
        max_tokens: int = 96,
        temperature: float = 0.0,
        timeout_seconds: float = 1.2,
        max_retries: int = 0,
    ) -> ModelResult:
        return ModelResult(
            text='{"match_score": 0.92, "is_match": true, "reason": "paraphrase_alignment"}',
            latency_ms=2,
            fallback_used=False,
            model_name=model_name,
        )


def test_response_match_predictor_can_use_model_judge() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    predictor = ResponseMatchPredictor(
        model_adapter=StubJudgeModelAdapter(logger=logger),
        logger=logger,
        model_judge_enabled=True,
    )
    report = asyncio.run(
        predictor.predict(
            user_input="Need account access recovery steps",
            answer_text="Use forgot password flow from sign in page and reset credentials.",
            intent="troubleshooting",
            trace_id="trace-predictor-model-judge",
            user_id="user-predictor-model-judge",
        ),
    )

    assert report.is_match is True
    assert report.score >= 0.45
    assert any(reason.startswith("model_judge:") for reason in report.reasons)


def test_strict_post_processor_keeps_non_empty_answer_and_reasoning() -> None:
    processor = ResponsePostProcessor()
    output = processor.process(
        text="",
        strict_response_mode=True,
        require_example=False,
    )

    assert output.startswith("Structured Reasoning Output")
    assert re.search(r"Answer:\n\S", output) is not None
    assert re.search(r"Reasoning:\n\S", output) is not None


def test_post_processor_enforces_flirty_prefix_when_missing() -> None:
    processor = ResponsePostProcessor()
    output = processor.process(
        text="Main tumhari madad ke liye ready hoon.",
        strict_response_mode=False,
        answer_style="flirty",
    )
    assert output != "Main tumhari madad ke liye ready hoon."
    assert any(token in output.lower() for token in ("cutie", "sweet", "jaan", "lovely"))


def test_post_processor_removes_hashtag_noise_in_relaxed_mode() -> None:
    processor = ResponsePostProcessor()
    output = processor.process(
        text="Hey there #chatwithme #randomtag",
        strict_response_mode=False,
        answer_style="flirty",
    )
    assert "#" not in output


def test_post_processor_flirty_rewrites_robotic_lines() -> None:
    processor = ResponsePostProcessor()
    output = processor.process(
        text="main aapki madad karne ke liye yahaan hoon. kya madad chahiye?",
        strict_response_mode=False,
        answer_style="flirty",
    )
    lowered = output.lower()
    assert "madad karne ke liye yahaan hoon" not in lowered
    assert "kya madad chahiye" not in lowered
    assert any(token in lowered for token in ("vibe", "mood", "baat"))


def test_post_processor_flirty_collapses_repetition() -> None:
    processor = ResponsePostProcessor()
    output = processor.process(
        text="kya kahaan hai kya kahaan hai kya kahaan hai",
        strict_response_mode=False,
        answer_style="flirty",
    )
    assert output.lower().count("kya kahaan hai") <= 1


class CountingJudgeModelAdapter(ModelAdapter):
    def __init__(self, logger) -> None:
        super().__init__(logger=logger, backend="local_llama", local_model_path="dummy.gguf")
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, Any] | None = None,
        model_name: str = "fast_model",
        max_tokens: int = 96,
        temperature: float = 0.0,
        timeout_seconds: float = 1.2,
        max_retries: int = 0,
    ) -> ModelResult:
        del prompt, trace_id, context, model_name, max_tokens, temperature, timeout_seconds, max_retries
        self.calls += 1
        return ModelResult(
            text='{"match_score": 0.9, "is_match": true, "reason": "judge_used"}',
            latency_ms=1,
            fallback_used=False,
            model_name="fast_model",
        )


def test_response_match_predictor_smart_gate_only_calls_judge_for_ambiguous_case() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = CountingJudgeModelAdapter(logger=logger)
    predictor = ResponseMatchPredictor(
        model_adapter=adapter,
        logger=logger,
        model_judge_enabled=True,
    )

    _ = asyncio.run(
        predictor.predict(
            user_input="please run context_digest for this request",
            answer_text="Tool run complete with summary.",
            intent="tool-directive",
            trace_id="trace-smart-gate-obvious",
            user_id="user-smart-gate-obvious",
        ),
    )
    _ = asyncio.run(
        predictor.predict(
            user_input="Need account access recovery steps",
            answer_text="Account issue maybe contact support.",
            intent="troubleshooting",
            trace_id="trace-smart-gate-ambiguous",
            user_id="user-smart-gate-ambiguous",
        ),
    )

    assert adapter.calls == 1
