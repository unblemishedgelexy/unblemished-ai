from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import AsyncIterator
from typing import Any, Callable, cast

from app.core.config import MODE_CONFIGS, ReasoningMode, Settings
from app.core.logger import StructuredLogger
from app.schemas.request_schema import ChatRequest
from app.schemas.response_schema import ChatResponse
from app.services.behavior.directness_enforcer import DirectnessEnforcer
from app.services.behavior.human_coherence_scorer import HumanCoherenceScorer
from app.services.behavior.response_match_predictor import ResponseMatchPredictor
from app.services.behavior.response_style_engine import ResponseStyleEngine
from app.services.behavior.topic_alignment_checker import TopicAlignmentChecker
from app.services.behavior.vagueness_detector import VaguenessDetector
from app.services.brain.evaluation_engine import EvaluationEngine
from app.services.brain.execution_controller import ExecutionController
from app.services.brain.fallback_engine import FallbackEngine
from app.services.brain.model_adapter import ModelAdapter, ModelResult
from app.services.brain.planning_engine import PlanningEngine
from app.services.brain.prompt_builder import PROMPT_VERSION, PromptBuilder
from app.services.brain.response_post_processor import ResponsePostProcessor
from app.services.brain.reflection_engine import ReflectionEngine
from app.services.brain.safety_classifier import SafetyClassifier
from app.services.brain.safe_web_lookup import SafeWebLookup
from app.services.brain.telemetry_engine import TelemetryEngine
from app.services.formatter.response_formatter import ResponseFormatter
from app.services.intent.intent_analyzer import IntentAnalyzer
from app.services.memory.memory_interface import MemoryInterface
from app.services.memory.memory_retriever import RetrievedMemory
from app.services.personality.emotional_continuity_engine import EmotionalContinuityEngine
from app.services.personality.identity_core import IdentityCore
from app.services.personality.safety_guardrails import SafetyGuardrails
from app.services.privacy.data_sanitizer import sanitize_text
from app.services.profile.user_profile_interface import UserProfileInterface
from app.services.relationship.relationship_state_engine import RelationshipState, RelationshipStateEngine
from app.services.router.model_router import ModelRouteDecision, ModelRouter
from app.services.skills.skill_interface import SkillInterface, SkillMatchResult
from app.utils.helpers import generate_anonymous_user_id, generate_trace_id, utc_now


@dataclass(slots=True)
class ExecutionContext:
    trace_id: str
    user_id: str
    intent: str = ""
    memory_context: list[RetrievedMemory] = field(default_factory=list)
    profile_snapshot: Any | None = None
    goal_snapshot: Any | None = None
    relationship_state: RelationshipState | None = None
    emotional_tone_state: dict[str, Any] | None = None
    reasoning_depth: int = 0
    routing_decision: ModelRouteDecision | None = None
    tool_plan: list[dict[str, Any]] = field(default_factory=list)
    evaluation_score: dict[str, Any] | None = None
    regeneration_count: int = 0
    final_response: ChatResponse | None = None


@dataclass(slots=True)
class HumanResponseAssessment:
    cleaned_answer: str
    coherence_score: float
    vagueness_flag: bool
    topic_misaligned: bool
    response_match_score: float
    response_match: bool
    response_match_reasons: list[str]
    requires_regeneration: bool


@dataclass(slots=True)
class RecentTurn:
    user_input: str
    assistant_output: str
    captured_at: datetime


@dataclass(slots=True)
class BehaviorCorrectionEvent:
    trigger_phrase: str
    corrected_instruction: str


class BrainOrchestrator:
    def __init__(
        self,
        *,
        intent_analyzer: IntentAnalyzer,
        prompt_builder: PromptBuilder,
        model_adapter: ModelAdapter,
        reflection_engine: ReflectionEngine,
        response_formatter: ResponseFormatter,
        memory_interface: MemoryInterface,
        user_profile_interface: UserProfileInterface,
        execution_controller: ExecutionController,
        model_router: ModelRouter,
        planning_engine: PlanningEngine,
        evaluation_engine: EvaluationEngine,
        fallback_engine: FallbackEngine,
        telemetry_engine: TelemetryEngine,
        safety_classifier: SafetyClassifier,
        safe_web_lookup: SafeWebLookup,
        response_post_processor: ResponsePostProcessor,
        response_style_engine: ResponseStyleEngine,
        response_match_predictor: ResponseMatchPredictor,
        directness_enforcer: DirectnessEnforcer,
        vagueness_detector: VaguenessDetector,
        topic_alignment_checker: TopicAlignmentChecker,
        human_coherence_scorer: HumanCoherenceScorer,
        identity_core: IdentityCore,
        relationship_state_engine: RelationshipStateEngine | None,
        emotional_continuity_engine: EmotionalContinuityEngine | None,
        safety_guardrails: SafetyGuardrails | None,
        skill_interface: SkillInterface | None,
        logger: StructuredLogger,
        settings_provider: Callable[[], Settings],
    ) -> None:
        self._intent_analyzer = intent_analyzer
        self._prompt_builder = prompt_builder
        self._model_adapter = model_adapter
        self._reflection_engine = reflection_engine
        self._response_formatter = response_formatter
        self._memory_interface = memory_interface
        self._user_profile_interface = user_profile_interface
        self._execution_controller = execution_controller
        self._model_router = model_router
        self._planning_engine = planning_engine
        self._evaluation_engine = evaluation_engine
        self._fallback_engine = fallback_engine
        self._telemetry_engine = telemetry_engine
        self._safety_classifier = safety_classifier
        self._safe_web_lookup = safe_web_lookup
        self._response_post_processor = response_post_processor
        self._response_style_engine = response_style_engine
        self._response_match_predictor = response_match_predictor
        self._directness_enforcer = directness_enforcer
        self._vagueness_detector = vagueness_detector
        self._topic_alignment_checker = topic_alignment_checker
        self._human_coherence_scorer = human_coherence_scorer
        self._identity_core = identity_core
        self._relationship_state_engine = relationship_state_engine
        self._emotional_continuity_engine = emotional_continuity_engine
        self._safety_guardrails = safety_guardrails
        self._skill_interface = skill_interface
        self._logger = logger
        self._settings_provider = settings_provider
        self._recent_turn_by_user: dict[str, RecentTurn] = {}
        self._recent_turn_capacity = 500

    async def run(
        self,
        request: ChatRequest,
        flow_type: str = "request",
        humanoid_mode_override: bool | None = None,
    ) -> ChatResponse:
        context = self._init_context(request)
        settings = self._settings_provider()
        humanoid_mode_enabled = (
            settings.humanoid_mode_enabled
            if humanoid_mode_override is None
            else humanoid_mode_override
        )
        mode = settings.reasoning_mode
        mode_config = settings.reasoning_profile
        strict_response_mode = settings.strict_response_mode
        user_context = request.context or {}
        started_at = utc_now()
        metrics = self._telemetry_engine.create_recorder()

        self._logger.info(
            f"brain.{flow_type}.started",
            trace_id=context.trace_id,
            user_id=context.user_id,
            reasoning_mode=mode,
            prompt_length=len(request.input_text),
            context_keys=sorted(user_context.keys()),
        )

        safety = await self._safety_classifier.classify(
            user_input=request.input_text,
            trace_id=context.trace_id,
            user_id=context.user_id,
        )
        if safety.blocked:
            metrics.record_error("safety_boundary_blocked")
            return self._build_safety_boundary_response(
                context=context,
                mode=mode,
                started_at=started_at,
                route_model="safety-boundary",
                category=safety.category or "harmful_instruction",
                reason=safety.reason or "blocked_by_classifier",
                strict_response_mode=strict_response_mode,
                user_input=request.input_text,
                user_context=user_context,
            )

        metrics.start_stage("profile_load_ms")
        context.profile_snapshot = await self._user_profile_interface.get_snapshot(
            user_id=context.user_id,
            trace_id=context.trace_id,
        )
        metrics.end_stage("profile_load_ms")
        profile_payload = _profile_to_payload(context.profile_snapshot)

        metrics.start_stage("goal_load_ms")
        context.goal_snapshot = await self._execution_controller.get_goal_snapshot(
            user_id=context.user_id,
            trace_id=context.trace_id,
        )
        metrics.end_stage("goal_load_ms")

        metrics.start_stage("relationship_state_load_ms")
        context.relationship_state = await self._load_relationship_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        if context.relationship_state is not None and self._relationship_state_engine is not None:
            user_context = {
                **user_context,
                **self._relationship_state_engine.tone_hint(context.relationship_state),
            }
        metrics.end_stage("relationship_state_load_ms")

        metrics.start_stage("emotional_state_load_ms")
        context.emotional_tone_state = await self._load_emotional_tone_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        if context.emotional_tone_state is not None:
            user_context = {
                **user_context,
                **context.emotional_tone_state,
            }
        metrics.end_stage("emotional_state_load_ms")

        metrics.start_stage("intent_analysis_ms")
        context.intent = self._intent_analyzer.analyze(request.input_text)
        requested_style = _resolve_requested_style(
            user_context,
            default_style=settings.default_answer_style,
        )
        style_profile = self._response_style_engine.resolve(
            intent=context.intent,
            user_input=request.input_text,
            companion_mode=humanoid_mode_enabled,
            requested_style=requested_style,
        )
        skip_quality_regeneration = _should_skip_quality_regeneration(
            style_key=style_profile.style_key,
            context=user_context,
        )
        user_context = {
            **user_context,
            "answer_style": style_profile.style_key,
        }
        metrics.end_stage("intent_analysis_ms")

        metrics.start_stage("skill_learning_ms")
        await self._learn_skill_from_corrections(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
        )
        await self._learn_behavior_from_corrections(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
        )
        metrics.end_stage("skill_learning_ms")

        metrics.start_stage("memory_retrieval_ms")
        context.memory_context = await self._memory_interface.retrieve_relevant(
            user_id=context.user_id,
            trace_id=context.trace_id,
            query_text=request.input_text,
            top_k=settings.memory_top_k,
            context_max_tokens=settings.memory_context_max_tokens,
        )
        metrics.end_stage("memory_retrieval_ms")
        context_used = _to_context_used(context.memory_context)

        metrics.start_stage("planning_ms")
        identity_context = await self._build_identity_context(
            trace_id=context.trace_id,
            user_id=context.user_id,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        relational_context = self._build_relational_context(
            relationship_state=context.relationship_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        emotional_context = self._build_emotional_context(
            emotional_tone_state=context.emotional_tone_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )

        planning = self._planning_engine.build(
            user_input=request.input_text,
            context=user_context,
            intent=context.intent,
            retrieved_memory_count=len(context.memory_context),
            reasoning_depth=mode_config.reasoning_depth,
            humanoid_mode_enabled=humanoid_mode_enabled,
            identity_context=identity_context,
            relational_context=relational_context,
            emotional_context=emotional_context,
        )
        metrics.end_stage("planning_ms")

        resolved_mode = self._planning_engine.resolve_reasoning_mode(
            current_mode=mode,
            complexity_score=planning.complexity_score,
            profile_depth_preference=profile_payload["conversation_depth_preference"],
        )
        if resolved_mode != mode:
            self._logger.info(
                "brain.reasoning_mode.overridden",
                trace_id=context.trace_id,
                user_id=context.user_id,
                from_mode=mode,
                to_mode=resolved_mode,
                complexity_score=planning.complexity_score,
            )
            mode = cast(ReasoningMode, resolved_mode)
            mode_config = MODE_CONFIGS[mode]

        context.reasoning_depth = mode_config.reasoning_depth
        context.tool_plan = planning.execution_plan

        metrics.start_stage("routing_ms")
        route = await self._model_router.route(
            trace_id=context.trace_id,
            user_id=context.user_id,
            intent=context.intent,
            reasoning_mode=mode,
            depth_preference=profile_payload["conversation_depth_preference"],
            complexity_score=planning.complexity_score,
        )
        metrics.end_stage("routing_ms")
        context.routing_decision = route

        reasoning_steps = self._planning_engine.build_reasoning_steps(
            trace_id=context.trace_id,
            user_id=context.user_id,
            input_text=request.input_text,
            intent=context.intent,
            context=user_context,
            reasoning_mode=mode,
            depth=mode_config.reasoning_depth,
            max_tokens=mode_config.max_tokens,
            temperature=mode_config.temperature,
            memory_top_k=settings.memory_top_k,
            retrieved_memories=context.memory_context,
            complexity_score=planning.complexity_score,
            reasoning_plan=planning.reasoning_plan,
            execution_plan=planning.execution_plan,
            route_decision=route,
        )

        prompt_bundle = self._prompt_builder.build(
            user_input=request.input_text,
            intent=context.intent,
            context=user_context,
            retrieved_context=_to_prompt_context(context.memory_context),
            user_profile_snapshot=profile_payload,
            reasoning_mode=mode,
            mode_config=mode_config.model_dump(),
            memory_max_summary_tokens=settings.memory_max_summary_tokens,
            trace_id=context.trace_id,
            strict_response_mode=strict_response_mode,
            answer_style=style_profile.style_key,
            style_instruction=style_profile.instruction,
        )

        metrics.start_stage("skill_match_ms")
        skill_match = await self._match_skill_trigger(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
            user_context=user_context,
        )
        metrics.end_stage("skill_match_ms")

        metrics.start_stage("model_ms")
        if skill_match is None:
            model_result = await self._model_adapter.generate(
                prompt=prompt_bundle["prompt"],
                context=user_context,
                trace_id=context.trace_id,
                model_name=route.model_name,
                max_tokens=mode_config.max_tokens,
                temperature=mode_config.temperature,
                timeout_seconds=settings.model_timeout_seconds,
                max_retries=settings.model_max_retries,
            )
        else:
            model_result = ModelResult(
                text=skill_match.draft_answer,
                latency_ms=0,
                fallback_used=False,
                model_name=route.model_name,
                failure_reason=None,
                backend="skill_registry",
            )
        metrics.end_stage("model_ms")
        model_latency_ms_total = model_result.latency_ms
        local_model_latency_ms_total = model_result.latency_ms if model_result.backend == "local_llama" else 0
        fallback_triggered = model_result.fallback_used
        fallback_reasons: list[str] = []
        retry_count = 0
        if model_result.fallback_used:
            fallback_reasons.append(
                "model_timeout" if model_result.failure_reason == "timeout" else "model_generation_failure",
            )

        reflection: dict[str, Any] = {}
        evaluation_scores: dict[str, Any] | None = None
        guardrail_triggered = False
        coherence_score_metric = 1.0
        vagueness_flag = False
        topic_misaligned = False
        tool_invocations: list[dict[str, Any]] = [skill_match.tool_invocation] if skill_match is not None else []
        if skill_match is not None:
            await self._memory_interface.schedule_action_store(
                user_id=context.user_id,
                trace_id=context.trace_id,
                action_type=f"tool:{skill_match.tool_name}",
                action_result_summary=skill_match.tool_invocation["result_summary"],
            )

        metrics.start_stage("reflection_ms")
        if not fallback_triggered:
            try:
                reflection = await self._reflection_engine.reflect(
                    draft_answer=model_result.text,
                    reasoning_steps=reasoning_steps,
                    enabled=mode_config.reflection_pass,
                    reasoning_mode=mode,
                    trace_id=context.trace_id,
                )
            except Exception as exc:
                fallback_triggered = True
                fallback_reasons.append("reflection_failed")
                metrics.record_error("reflection_failed")
                self._logger.error(
                    "brain.reflection.failed",
                    trace_id=context.trace_id,
                    user_id=context.user_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        metrics.end_stage("reflection_ms")

        metrics.start_stage("evaluation_ms")
        if not fallback_triggered:
            assessment = await self._assess_human_response(
                user_input=request.input_text,
                answer_text=reflection["final_answer"],
                intent=context.intent,
                companion_mode=humanoid_mode_enabled,
                strict_response_mode=strict_response_mode,
                trace_id=context.trace_id,
                user_id=context.user_id,
            )
            reflection["final_answer"] = assessment.cleaned_answer
            coherence_score_metric = assessment.coherence_score
            vagueness_flag = assessment.vagueness_flag
            topic_misaligned = assessment.topic_misaligned
            if assessment.requires_regeneration:
                metrics.record_error("human_response_regeneration_triggered")
                self._logger.warning(
                    "brain.human_response.regeneration_triggered",
                    trace_id=context.trace_id,
                    user_id=context.user_id,
                    coherence_score=assessment.coherence_score,
                    vagueness_flag=assessment.vagueness_flag,
                    topic_misaligned=assessment.topic_misaligned,
                    response_match=assessment.response_match,
                    response_match_score=assessment.response_match_score,
                    response_match_reasons=assessment.response_match_reasons,
                )
            evaluation_scores = self._evaluation_engine.evaluate(
                final_answer=reflection["final_answer"],
                strengths=reflection["strengths"],
                complexity_score=planning.complexity_score,
                retry_count=0,
                user_input=request.input_text,
                intent=context.intent,
            )
            evaluation_scores["coherence_score"] = round(
                max(float(evaluation_scores["coherence_score"]), assessment.coherence_score),
                2,
            )
            should_regenerate = self._evaluation_engine.below_threshold(
                evaluation_scores,
                final_answer=reflection["final_answer"],
                user_input=request.input_text,
                intent=context.intent,
                complexity_score=planning.complexity_score,
            ) or assessment.requires_regeneration
            if (
                settings.self_evaluation_enabled
                and (not skip_quality_regeneration)
                and should_regenerate
                and skill_match is None
            ):
                retry_count = 1
                retry_prompt = self._build_regeneration_prompt(
                    base_prompt=prompt_bundle["prompt"],
                    user_input=request.input_text,
                    intent=context.intent,
                    assessment=assessment,
                )
                retry_result = await self._model_adapter.generate(
                    prompt=retry_prompt,
                    context=user_context,
                    trace_id=context.trace_id,
                    model_name=route.model_name,
                    max_tokens=min(mode_config.max_tokens + 128, 2048),
                    temperature=max(mode_config.temperature - 0.05, 0.1),
                    timeout_seconds=settings.model_timeout_seconds,
                    max_retries=settings.model_max_retries,
                )
                model_latency_ms_total += retry_result.latency_ms
                if retry_result.backend == "local_llama":
                    local_model_latency_ms_total += retry_result.latency_ms
                if retry_result.fallback_used:
                    fallback_triggered = True
                    fallback_reasons.append("model_generation_failure_retry")
                else:
                    try:
                        reflection = await self._reflection_engine.reflect(
                            draft_answer=retry_result.text,
                            reasoning_steps=reasoning_steps,
                            enabled=mode_config.reflection_pass,
                            reasoning_mode=mode,
                            trace_id=context.trace_id,
                        )
                    except Exception as exc:
                        fallback_triggered = True
                        fallback_reasons.append("reflection_failed_retry")
                        metrics.record_error("reflection_failed_retry")
                        self._logger.error(
                            "brain.reflection.retry_failed",
                            trace_id=context.trace_id,
                            user_id=context.user_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    if not fallback_triggered:
                        assessment = await self._assess_human_response(
                            user_input=request.input_text,
                            answer_text=reflection["final_answer"],
                            intent=context.intent,
                            companion_mode=humanoid_mode_enabled,
                            strict_response_mode=strict_response_mode,
                            trace_id=context.trace_id,
                            user_id=context.user_id,
                        )
                        reflection["final_answer"] = assessment.cleaned_answer
                        coherence_score_metric = assessment.coherence_score
                        vagueness_flag = assessment.vagueness_flag
                        topic_misaligned = assessment.topic_misaligned
                        evaluation_scores = self._evaluation_engine.evaluate(
                            final_answer=reflection["final_answer"],
                            strengths=reflection["strengths"],
                            complexity_score=planning.complexity_score,
                            retry_count=1,
                            user_input=request.input_text,
                            intent=context.intent,
                        )
                        evaluation_scores["coherence_score"] = round(
                            max(float(evaluation_scores["coherence_score"]), assessment.coherence_score),
                            2,
                        )
                        if self._evaluation_engine.below_threshold(
                            evaluation_scores,
                            final_answer=reflection["final_answer"],
                            user_input=request.input_text,
                            intent=context.intent,
                            complexity_score=planning.complexity_score,
                        ) or assessment.requires_regeneration:
                            fallback_triggered = True
                            fallback_reasons.append("human_response_below_threshold_twice")
                            metrics.record_error("human_response_below_threshold_twice")
        metrics.end_stage("evaluation_ms")

        metrics.start_stage("tool_execution_ms")
        if not fallback_triggered and skill_match is None:
            reflection, tool_invocations = await self._execution_controller.execute_tools(
                execution_plan=planning.execution_plan,
                user_input=request.input_text,
                context=user_context,
                reflection=reflection,
                goal_snapshot=context.goal_snapshot,
                trace_id=context.trace_id,
                user_id=context.user_id,
            )
        metrics.end_stage("tool_execution_ms")

        if fallback_triggered:
            safe_message = ""
            web_recovery = await self._attempt_internet_recovery(
                enabled=settings.internet_lookup_enabled,
                user_input=request.input_text,
                user_id=context.user_id,
                trace_id=context.trace_id,
                model_name=route.model_name,
                reasons=fallback_reasons,
                timeout_seconds=settings.model_timeout_seconds,
                max_retries=settings.model_max_retries,
            )
            if web_recovery is not None:
                tool_invocations.append(web_recovery["tool_invocation"])
                if web_recovery["answer_text"]:
                    fallback_reasons.append("internet_lookup_recovery")
                    safe_message = web_recovery["answer_text"]
                    await self._memory_interface.schedule_action_store(
                        user_id=context.user_id,
                        trace_id=context.trace_id,
                        action_type="tool:safe_web_lookup",
                        action_result_summary=web_recovery["tool_invocation"]["result_summary"],
                    )
                else:
                    fallback_reasons.append("internet_lookup_empty")

            if not safe_message:
                safe_message = self._fallback_engine.build_safe_message(
                    context=user_context,
                    model_name=route.model_name,
                    reasons=fallback_reasons,
                )
            reflection = self._fallback_engine.build_reflection(
                safe_answer=safe_message,
                reflection_enabled=mode_config.reflection_pass,
                reasoning_mode=mode,
                trace_id=context.trace_id,
                reasons=fallback_reasons,
            )
            evaluation_scores = self._evaluation_engine.fallback_scores(retry_count=retry_count)
            coherence_score_metric = 0.2

        if humanoid_mode_enabled:
            reflection["final_answer"] = await self._identity_core.apply_personality_filter(
                final_answer=reflection["final_answer"],
                trace_id=context.trace_id,
                user_id=context.user_id,
                fallback_triggered=fallback_triggered,
            )
            reflection["final_answer"] = await self._apply_relationship_tone(
                final_answer=reflection["final_answer"],
                relationship_state=context.relationship_state,
                trace_id=context.trace_id,
                user_id=context.user_id,
                humanoid_mode_enabled=humanoid_mode_enabled,
            )
            reflection["final_answer"] = await self._apply_emotional_tone(
                final_answer=reflection["final_answer"],
                emotional_tone_state=context.emotional_tone_state,
                trace_id=context.trace_id,
                user_id=context.user_id,
                humanoid_mode_enabled=humanoid_mode_enabled,
            )
            reflection["final_answer"], guardrail_triggered = await self._apply_companion_safety_guardrails(
                user_input=request.input_text,
                final_answer=reflection["final_answer"],
                trace_id=context.trace_id,
                user_id=context.user_id,
                humanoid_mode_enabled=humanoid_mode_enabled,
            )

        reflection["final_answer"] = self._response_post_processor.process(
            text=reflection["final_answer"],
            strict_response_mode=strict_response_mode,
            require_example=style_profile.requires_example,
            answer_style=style_profile.style_key,
        )
        reflection["final_answer"] = _apply_flirty_companion_enhancer(
            user_input=request.input_text,
            final_answer=reflection["final_answer"],
            style_key=style_profile.style_key,
        )

        metrics.start_stage("profile_update_ms")
        try:
            updated_profile = await self._user_profile_interface.update_after_interaction(
                user_id=context.user_id,
                trace_id=context.trace_id,
                user_input=request.input_text,
                detected_intent=context.intent,
                reasoning_depth=mode_config.reasoning_depth,
            )
        except Exception:
            updated_profile = context.profile_snapshot
        metrics.end_stage("profile_update_ms")

        metrics.start_stage("goal_update_ms")
        try:
            updated_goal = await self._execution_controller.update_goal(
                user_id=context.user_id,
                trace_id=context.trace_id,
                user_input=request.input_text,
                execution_plan=planning.execution_plan,
                tool_invocations=tool_invocations,
                fallback_triggered=fallback_triggered,
            )
        except Exception:
            updated_goal = context.goal_snapshot
        metrics.end_stage("goal_update_ms")

        metrics.start_stage("relationship_update_ms")
        context.relationship_state = await self._update_relationship_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
            final_answer=reflection["final_answer"],
            fallback_triggered=fallback_triggered,
            existing_state=context.relationship_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        metrics.end_stage("relationship_update_ms")

        metrics.start_stage("emotional_state_update_ms")
        context.emotional_tone_state = await self._update_emotional_tone_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
            final_answer=reflection["final_answer"],
            fallback_triggered=fallback_triggered,
            existing_state=context.emotional_tone_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        metrics.end_stage("emotional_state_update_ms")

        finished_at = utc_now()
        processing_time_ms = _duration_ms(started_at, finished_at)
        metrics.record_stage_duration("total_ms", processing_time_ms)
        metrics_snapshot = metrics.snapshot(total_ms=processing_time_ms)
        response_prompt = _render_response_prompt(
            full_prompt=prompt_bundle["prompt"],
            prompt_version=prompt_bundle["prompt_version"],
            reasoning_mode=mode,
            intent=context.intent,
            user_input=request.input_text,
            context_keys=sorted(user_context.keys()),
            expose_full_prompt=settings.expose_full_prompt_in_response,
        )

        response = self._response_formatter.format(
            trace_id=context.trace_id,
            intent=context.intent,
            prompt=response_prompt,
            prompt_version=prompt_bundle["prompt_version"],
            reasoning_mode=mode,
            started_at=started_at,
            finished_at=finished_at,
            processing_time_ms=processing_time_ms,
            model_latency_ms=max(model_latency_ms_total, 0),
            reasoning_steps=reasoning_steps,
            context_used=context_used,
            user_profile_snapshot=_profile_to_payload(updated_profile),
            complexity_score=planning.complexity_score,
            evaluation_scores=evaluation_scores,
            cost_estimate=self._telemetry_engine.build_cost_estimate(
                prompt=prompt_bundle["prompt"],
                final_answer=reflection["final_answer"],
                routed_model=route.model_name,
                metrics_snapshot=metrics_snapshot,
            ),
            execution_plan=planning.execution_plan,
            tool_invocations=tool_invocations or None,
            goal_snapshot=_goal_to_payload(updated_goal),
            fallback_triggered=fallback_triggered,
            reflection=reflection,
        )
        validated = ChatResponse.model_validate(response.model_dump())
        context.final_response = validated
        context.regeneration_count = retry_count
        context.evaluation_score = evaluation_scores
        self._telemetry_engine.record_request_metrics(
            latency_ms=processing_time_ms,
            model_latency_ms=max(model_latency_ms_total, 0),
            fallback_triggered=fallback_triggered,
            guardrail_triggered=guardrail_triggered,
            tool_usage_count=len(tool_invocations),
            local_model_latency_ms=max(local_model_latency_ms_total, 0),
            coherence_score=coherence_score_metric,
            regenerated=retry_count > 0,
            vagueness_flag=vagueness_flag,
            topic_misaligned=topic_misaligned,
        )

        await self._memory_interface.schedule_summary_store(
            user_id=context.user_id,
            trace_id=context.trace_id,
            user_input=request.input_text,
            user_context=user_context,
            final_answer=validated.reflection.final_answer,
            context_used=context_used,
            max_summary_tokens=settings.memory_max_summary_tokens,
        )
        await self._memory_interface.schedule_long_term_evolution(
            user_id=context.user_id,
            trace_id=context.trace_id,
            user_input=request.input_text,
            final_answer=validated.reflection.final_answer,
            context_used=context_used,
            max_summary_tokens=settings.memory_max_summary_tokens,
        )
        self._remember_turn(
            user_id=context.user_id,
            user_input=request.input_text,
            assistant_output=validated.reflection.final_answer,
        )
        return validated

    async def stream(
        self,
        request: ChatRequest,
        humanoid_mode_override: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        context = self._init_context(request)
        settings = self._settings_provider()
        humanoid_mode_enabled = (
            settings.humanoid_mode_enabled
            if humanoid_mode_override is None
            else humanoid_mode_override
        )
        mode = settings.reasoning_mode
        mode_config = settings.reasoning_profile
        strict_response_mode = settings.strict_response_mode
        user_context = request.context or {}
        started_at = utc_now()
        metrics = self._telemetry_engine.create_recorder()

        safety = await self._safety_classifier.classify(
            user_input=request.input_text,
            trace_id=context.trace_id,
            user_id=context.user_id,
        )
        if safety.blocked:
            metrics.record_error("safety_boundary_blocked")
            blocked = self._build_safety_boundary_response(
                context=context,
                mode=mode,
                started_at=started_at,
                route_model="safety-boundary",
                category=safety.category or "harmful_instruction",
                reason=safety.reason or "blocked_by_classifier",
                strict_response_mode=strict_response_mode,
                user_input=request.input_text,
                user_context=user_context,
            )
            yield {
                "type": "final",
                "trace_id": context.trace_id,
                "user_id": context.user_id,
                "response": blocked.model_dump(mode="json"),
            }
            return

        metrics.start_stage("profile_load_ms")
        context.profile_snapshot = await self._user_profile_interface.get_snapshot(
            user_id=context.user_id,
            trace_id=context.trace_id,
        )
        metrics.end_stage("profile_load_ms")
        profile_payload = _profile_to_payload(context.profile_snapshot)

        metrics.start_stage("goal_load_ms")
        context.goal_snapshot = await self._execution_controller.get_goal_snapshot(
            user_id=context.user_id,
            trace_id=context.trace_id,
        )
        metrics.end_stage("goal_load_ms")

        metrics.start_stage("relationship_state_load_ms")
        context.relationship_state = await self._load_relationship_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        if context.relationship_state is not None and self._relationship_state_engine is not None:
            user_context = {
                **user_context,
                **self._relationship_state_engine.tone_hint(context.relationship_state),
            }
        metrics.end_stage("relationship_state_load_ms")

        metrics.start_stage("emotional_state_load_ms")
        context.emotional_tone_state = await self._load_emotional_tone_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        if context.emotional_tone_state is not None:
            user_context = {
                **user_context,
                **context.emotional_tone_state,
            }
        metrics.end_stage("emotional_state_load_ms")

        metrics.start_stage("intent_analysis_ms")
        context.intent = self._intent_analyzer.analyze(request.input_text)
        requested_style = _resolve_requested_style(
            user_context,
            default_style=settings.default_answer_style,
        )
        style_profile = self._response_style_engine.resolve(
            intent=context.intent,
            user_input=request.input_text,
            companion_mode=humanoid_mode_enabled,
            requested_style=requested_style,
        )
        skip_quality_regeneration = _should_skip_quality_regeneration(
            style_key=style_profile.style_key,
            context=user_context,
        )
        user_context = {
            **user_context,
            "answer_style": style_profile.style_key,
        }
        metrics.end_stage("intent_analysis_ms")

        metrics.start_stage("skill_learning_ms")
        await self._learn_skill_from_corrections(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
        )
        await self._learn_behavior_from_corrections(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
        )
        metrics.end_stage("skill_learning_ms")

        metrics.start_stage("memory_retrieval_ms")
        context.memory_context = await self._memory_interface.retrieve_relevant(
            user_id=context.user_id,
            trace_id=context.trace_id,
            query_text=request.input_text,
            top_k=settings.memory_top_k,
            context_max_tokens=settings.memory_context_max_tokens,
        )
        metrics.end_stage("memory_retrieval_ms")
        context_used = _to_context_used(context.memory_context)

        metrics.start_stage("planning_ms")
        identity_context = await self._build_identity_context(
            trace_id=context.trace_id,
            user_id=context.user_id,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        relational_context = self._build_relational_context(
            relationship_state=context.relationship_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        emotional_context = self._build_emotional_context(
            emotional_tone_state=context.emotional_tone_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )

        planning = self._planning_engine.build(
            user_input=request.input_text,
            context=user_context,
            intent=context.intent,
            retrieved_memory_count=len(context.memory_context),
            reasoning_depth=mode_config.reasoning_depth,
            humanoid_mode_enabled=humanoid_mode_enabled,
            identity_context=identity_context,
            relational_context=relational_context,
            emotional_context=emotional_context,
        )
        metrics.end_stage("planning_ms")

        resolved_mode = self._planning_engine.resolve_reasoning_mode(
            current_mode=mode,
            complexity_score=planning.complexity_score,
            profile_depth_preference=profile_payload["conversation_depth_preference"],
        )
        if resolved_mode != mode:
            mode = cast(ReasoningMode, resolved_mode)
            mode_config = MODE_CONFIGS[mode]

        context.reasoning_depth = mode_config.reasoning_depth
        context.tool_plan = planning.execution_plan

        metrics.start_stage("routing_ms")
        route = await self._model_router.route(
            trace_id=context.trace_id,
            user_id=context.user_id,
            intent=context.intent,
            reasoning_mode=mode,
            depth_preference=profile_payload["conversation_depth_preference"],
            complexity_score=planning.complexity_score,
        )
        metrics.end_stage("routing_ms")
        context.routing_decision = route

        reasoning_steps = self._planning_engine.build_reasoning_steps(
            trace_id=context.trace_id,
            user_id=context.user_id,
            input_text=request.input_text,
            intent=context.intent,
            context=user_context,
            reasoning_mode=mode,
            depth=mode_config.reasoning_depth,
            max_tokens=mode_config.max_tokens,
            temperature=mode_config.temperature,
            memory_top_k=settings.memory_top_k,
            retrieved_memories=context.memory_context,
            complexity_score=planning.complexity_score,
            reasoning_plan=planning.reasoning_plan,
            execution_plan=planning.execution_plan,
            route_decision=route,
        )
        prompt_bundle = self._prompt_builder.build(
            user_input=request.input_text,
            intent=context.intent,
            context=user_context,
            retrieved_context=_to_prompt_context(context.memory_context),
            user_profile_snapshot=profile_payload,
            reasoning_mode=mode,
            mode_config=mode_config.model_dump(),
            memory_max_summary_tokens=settings.memory_max_summary_tokens,
            trace_id=context.trace_id,
            strict_response_mode=strict_response_mode,
            answer_style=style_profile.style_key,
            style_instruction=style_profile.instruction,
        )

        metrics.start_stage("skill_match_ms")
        skill_match = await self._match_skill_trigger(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
            user_context=user_context,
        )
        metrics.end_stage("skill_match_ms")

        metrics.start_stage("model_ms")
        model_result: ModelResult | None = None
        if skill_match is None:
            token_index = 0
            async for event in self._model_adapter.generate_stream_events(
                prompt=prompt_bundle["prompt"],
                context=user_context,
                trace_id=context.trace_id,
                model_name=route.model_name,
                max_tokens=mode_config.max_tokens,
                temperature=mode_config.temperature,
                timeout_seconds=settings.model_timeout_seconds,
                max_retries=settings.model_max_retries,
            ):
                if event.token is not None:
                    yield {
                        "type": "token",
                        "trace_id": context.trace_id,
                        "user_id": context.user_id,
                        "index": token_index,
                        "content": event.token,
                    }
                    token_index += 1
                if event.result is not None:
                    model_result = event.result
        else:
            token_index = 0
            for token in _stream_tokens_from_text(skill_match.draft_answer):
                yield {
                    "type": "token",
                    "trace_id": context.trace_id,
                    "user_id": context.user_id,
                    "index": token_index,
                    "content": token,
                }
                token_index += 1
            model_result = ModelResult(
                text=skill_match.draft_answer,
                latency_ms=0,
                fallback_used=False,
                model_name=route.model_name,
                failure_reason=None,
                backend="skill_registry",
            )
        metrics.end_stage("model_ms")
        if model_result is None:
            model_result = ModelResult(
                text="",
                latency_ms=0,
                fallback_used=True,
                model_name=route.model_name,
                failure_reason="missing_stream_result",
                backend="unknown",
            )

        model_latency_ms_total = model_result.latency_ms
        local_model_latency_ms_total = model_result.latency_ms if model_result.backend == "local_llama" else 0
        fallback_triggered = model_result.fallback_used
        fallback_reasons: list[str] = []
        retry_count = 0
        if fallback_triggered:
            fallback_reasons.append(
                "model_timeout" if model_result.failure_reason == "timeout" else "model_generation_failure",
            )

        reflection: dict[str, Any] = {}
        evaluation_scores: dict[str, Any] | None = None
        guardrail_triggered = False
        coherence_score_metric = 1.0
        vagueness_flag = False
        topic_misaligned = False
        tool_invocations: list[dict[str, Any]] = [skill_match.tool_invocation] if skill_match is not None else []
        if skill_match is not None:
            await self._memory_interface.schedule_action_store(
                user_id=context.user_id,
                trace_id=context.trace_id,
                action_type=f"tool:{skill_match.tool_name}",
                action_result_summary=skill_match.tool_invocation["result_summary"],
            )

        metrics.start_stage("reflection_ms")
        if not fallback_triggered:
            try:
                reflection = await self._reflection_engine.reflect(
                    draft_answer=model_result.text.strip(),
                    reasoning_steps=reasoning_steps,
                    enabled=mode_config.reflection_pass,
                    reasoning_mode=mode,
                    trace_id=context.trace_id,
                )
            except Exception:
                fallback_triggered = True
                fallback_reasons.append("reflection_failed")
                metrics.record_error("reflection_failed")
        metrics.end_stage("reflection_ms")

        metrics.start_stage("evaluation_ms")
        if not fallback_triggered:
            assessment = await self._assess_human_response(
                user_input=request.input_text,
                answer_text=reflection["final_answer"],
                intent=context.intent,
                companion_mode=humanoid_mode_enabled,
                strict_response_mode=strict_response_mode,
                trace_id=context.trace_id,
                user_id=context.user_id,
            )
            reflection["final_answer"] = assessment.cleaned_answer
            coherence_score_metric = assessment.coherence_score
            vagueness_flag = assessment.vagueness_flag
            topic_misaligned = assessment.topic_misaligned
            if assessment.requires_regeneration:
                metrics.record_error("human_response_regeneration_triggered")
            evaluation_scores = self._evaluation_engine.evaluate(
                final_answer=reflection["final_answer"],
                strengths=reflection["strengths"],
                complexity_score=planning.complexity_score,
                retry_count=0,
                user_input=request.input_text,
                intent=context.intent,
            )
            evaluation_scores["coherence_score"] = round(
                max(float(evaluation_scores["coherence_score"]), assessment.coherence_score),
                2,
            )
            should_regenerate = self._evaluation_engine.below_threshold(
                evaluation_scores,
                final_answer=reflection["final_answer"],
                user_input=request.input_text,
                intent=context.intent,
                complexity_score=planning.complexity_score,
            ) or assessment.requires_regeneration
            if (
                settings.self_evaluation_enabled
                and (not skip_quality_regeneration)
                and should_regenerate
                and skill_match is None
            ):
                retry_count = 1
                retry_prompt = self._build_regeneration_prompt(
                    base_prompt=prompt_bundle["prompt"],
                    user_input=request.input_text,
                    intent=context.intent,
                    assessment=assessment,
                )
                retry_result = await self._model_adapter.generate(
                    prompt=retry_prompt,
                    context=user_context,
                    trace_id=context.trace_id,
                    model_name=route.model_name,
                    max_tokens=min(mode_config.max_tokens + 128, 2048),
                    temperature=max(mode_config.temperature - 0.05, 0.1),
                    timeout_seconds=settings.model_timeout_seconds,
                    max_retries=settings.model_max_retries,
                )
                model_latency_ms_total += retry_result.latency_ms
                if retry_result.backend == "local_llama":
                    local_model_latency_ms_total += retry_result.latency_ms
                if retry_result.fallback_used:
                    fallback_triggered = True
                    fallback_reasons.append("model_generation_failure_retry")
                else:
                    try:
                        reflection = await self._reflection_engine.reflect(
                            draft_answer=retry_result.text,
                            reasoning_steps=reasoning_steps,
                            enabled=mode_config.reflection_pass,
                            reasoning_mode=mode,
                            trace_id=context.trace_id,
                        )
                    except Exception:
                        fallback_triggered = True
                        fallback_reasons.append("reflection_failed_retry")
                        metrics.record_error("reflection_failed_retry")
                    if not fallback_triggered:
                        assessment = await self._assess_human_response(
                            user_input=request.input_text,
                            answer_text=reflection["final_answer"],
                            intent=context.intent,
                            companion_mode=humanoid_mode_enabled,
                            strict_response_mode=strict_response_mode,
                            trace_id=context.trace_id,
                            user_id=context.user_id,
                        )
                        reflection["final_answer"] = assessment.cleaned_answer
                        coherence_score_metric = assessment.coherence_score
                        vagueness_flag = assessment.vagueness_flag
                        topic_misaligned = assessment.topic_misaligned
                        evaluation_scores = self._evaluation_engine.evaluate(
                            final_answer=reflection["final_answer"],
                            strengths=reflection["strengths"],
                            complexity_score=planning.complexity_score,
                            retry_count=1,
                            user_input=request.input_text,
                            intent=context.intent,
                        )
                        evaluation_scores["coherence_score"] = round(
                            max(float(evaluation_scores["coherence_score"]), assessment.coherence_score),
                            2,
                        )
                        if self._evaluation_engine.below_threshold(
                            evaluation_scores,
                            final_answer=reflection["final_answer"],
                            user_input=request.input_text,
                            intent=context.intent,
                            complexity_score=planning.complexity_score,
                        ) or assessment.requires_regeneration:
                            fallback_triggered = True
                            fallback_reasons.append("human_response_below_threshold_twice")
                            metrics.record_error("human_response_below_threshold_twice")
        metrics.end_stage("evaluation_ms")

        metrics.start_stage("tool_execution_ms")
        if not fallback_triggered and skill_match is None:
            reflection, tool_invocations = await self._execution_controller.execute_tools(
                execution_plan=planning.execution_plan,
                user_input=request.input_text,
                context=user_context,
                reflection=reflection,
                goal_snapshot=context.goal_snapshot,
                trace_id=context.trace_id,
                user_id=context.user_id,
            )
        metrics.end_stage("tool_execution_ms")

        if fallback_triggered:
            safe_message = ""
            web_recovery = await self._attempt_internet_recovery(
                enabled=settings.internet_lookup_enabled,
                user_input=request.input_text,
                user_id=context.user_id,
                trace_id=context.trace_id,
                model_name=route.model_name,
                reasons=fallback_reasons,
                timeout_seconds=settings.model_timeout_seconds,
                max_retries=settings.model_max_retries,
            )
            if web_recovery is not None:
                tool_invocations.append(web_recovery["tool_invocation"])
                if web_recovery["answer_text"]:
                    fallback_reasons.append("internet_lookup_recovery")
                    safe_message = web_recovery["answer_text"]
                    await self._memory_interface.schedule_action_store(
                        user_id=context.user_id,
                        trace_id=context.trace_id,
                        action_type="tool:safe_web_lookup",
                        action_result_summary=web_recovery["tool_invocation"]["result_summary"],
                    )
                else:
                    fallback_reasons.append("internet_lookup_empty")

            if not safe_message:
                safe_message = self._fallback_engine.build_safe_message(
                    context=user_context,
                    model_name=route.model_name,
                    reasons=fallback_reasons,
                )
            reflection = self._fallback_engine.build_reflection(
                safe_answer=safe_message,
                reflection_enabled=mode_config.reflection_pass,
                reasoning_mode=mode,
                trace_id=context.trace_id,
                reasons=fallback_reasons,
            )
            evaluation_scores = self._evaluation_engine.fallback_scores(retry_count=retry_count)
            coherence_score_metric = 0.2

        if humanoid_mode_enabled:
            reflection["final_answer"] = await self._identity_core.apply_personality_filter(
                final_answer=reflection["final_answer"],
                trace_id=context.trace_id,
                user_id=context.user_id,
                fallback_triggered=fallback_triggered,
            )
            reflection["final_answer"] = await self._apply_relationship_tone(
                final_answer=reflection["final_answer"],
                relationship_state=context.relationship_state,
                trace_id=context.trace_id,
                user_id=context.user_id,
                humanoid_mode_enabled=humanoid_mode_enabled,
            )
            reflection["final_answer"] = await self._apply_emotional_tone(
                final_answer=reflection["final_answer"],
                emotional_tone_state=context.emotional_tone_state,
                trace_id=context.trace_id,
                user_id=context.user_id,
                humanoid_mode_enabled=humanoid_mode_enabled,
            )
            reflection["final_answer"], guardrail_triggered = await self._apply_companion_safety_guardrails(
                user_input=request.input_text,
                final_answer=reflection["final_answer"],
                trace_id=context.trace_id,
                user_id=context.user_id,
                humanoid_mode_enabled=humanoid_mode_enabled,
            )

        reflection["final_answer"] = self._response_post_processor.process(
            text=reflection["final_answer"],
            strict_response_mode=strict_response_mode,
            require_example=style_profile.requires_example,
            answer_style=style_profile.style_key,
        )
        reflection["final_answer"] = _apply_flirty_companion_enhancer(
            user_input=request.input_text,
            final_answer=reflection["final_answer"],
            style_key=style_profile.style_key,
        )

        metrics.start_stage("profile_update_ms")
        try:
            updated_profile = await self._user_profile_interface.update_after_interaction(
                user_id=context.user_id,
                trace_id=context.trace_id,
                user_input=request.input_text,
                detected_intent=context.intent,
                reasoning_depth=mode_config.reasoning_depth,
            )
        except Exception:
            updated_profile = context.profile_snapshot
        metrics.end_stage("profile_update_ms")

        metrics.start_stage("goal_update_ms")
        try:
            updated_goal = await self._execution_controller.update_goal(
                user_id=context.user_id,
                trace_id=context.trace_id,
                user_input=request.input_text,
                execution_plan=planning.execution_plan,
                tool_invocations=tool_invocations,
                fallback_triggered=fallback_triggered,
            )
        except Exception:
            updated_goal = context.goal_snapshot
        metrics.end_stage("goal_update_ms")

        metrics.start_stage("relationship_update_ms")
        context.relationship_state = await self._update_relationship_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
            final_answer=reflection["final_answer"],
            fallback_triggered=fallback_triggered,
            existing_state=context.relationship_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        metrics.end_stage("relationship_update_ms")

        metrics.start_stage("emotional_state_update_ms")
        context.emotional_tone_state = await self._update_emotional_tone_state(
            trace_id=context.trace_id,
            user_id=context.user_id,
            user_input=request.input_text,
            final_answer=reflection["final_answer"],
            fallback_triggered=fallback_triggered,
            existing_state=context.emotional_tone_state,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        metrics.end_stage("emotional_state_update_ms")

        finished_at = utc_now()
        processing_time_ms = _duration_ms(started_at, finished_at)
        metrics.record_stage_duration("total_ms", processing_time_ms)
        metrics_snapshot = metrics.snapshot(total_ms=processing_time_ms)
        response_prompt = _render_response_prompt(
            full_prompt=prompt_bundle["prompt"],
            prompt_version=prompt_bundle["prompt_version"],
            reasoning_mode=mode,
            intent=context.intent,
            user_input=request.input_text,
            context_keys=sorted(user_context.keys()),
            expose_full_prompt=settings.expose_full_prompt_in_response,
        )

        response = self._response_formatter.format(
            trace_id=context.trace_id,
            intent=context.intent,
            prompt=response_prompt,
            prompt_version=prompt_bundle["prompt_version"],
            reasoning_mode=mode,
            started_at=started_at,
            finished_at=finished_at,
            processing_time_ms=processing_time_ms,
            model_latency_ms=max(model_latency_ms_total, 0),
            reasoning_steps=reasoning_steps,
            context_used=context_used,
            user_profile_snapshot=_profile_to_payload(updated_profile),
            complexity_score=planning.complexity_score,
            evaluation_scores=evaluation_scores,
            cost_estimate=self._telemetry_engine.build_cost_estimate(
                prompt=prompt_bundle["prompt"],
                final_answer=reflection["final_answer"],
                routed_model=route.model_name,
                metrics_snapshot=metrics_snapshot,
            ),
            execution_plan=planning.execution_plan,
            tool_invocations=tool_invocations or None,
            goal_snapshot=_goal_to_payload(updated_goal),
            fallback_triggered=fallback_triggered,
            reflection=reflection,
        )
        validated = ChatResponse.model_validate(response.model_dump())
        context.final_response = validated
        context.regeneration_count = retry_count
        context.evaluation_score = evaluation_scores
        self._telemetry_engine.record_request_metrics(
            latency_ms=processing_time_ms,
            model_latency_ms=max(model_latency_ms_total, 0),
            fallback_triggered=fallback_triggered,
            guardrail_triggered=guardrail_triggered,
            tool_usage_count=len(tool_invocations),
            local_model_latency_ms=max(local_model_latency_ms_total, 0),
            coherence_score=coherence_score_metric,
            regenerated=retry_count > 0,
            vagueness_flag=vagueness_flag,
            topic_misaligned=topic_misaligned,
        )

        await self._memory_interface.schedule_summary_store(
            user_id=context.user_id,
            trace_id=context.trace_id,
            user_input=request.input_text,
            user_context=user_context,
            final_answer=validated.reflection.final_answer,
            context_used=context_used,
            max_summary_tokens=settings.memory_max_summary_tokens,
        )
        await self._memory_interface.schedule_long_term_evolution(
            user_id=context.user_id,
            trace_id=context.trace_id,
            user_input=request.input_text,
            final_answer=validated.reflection.final_answer,
            context_used=context_used,
            max_summary_tokens=settings.memory_max_summary_tokens,
        )
        self._remember_turn(
            user_id=context.user_id,
            user_input=request.input_text,
            assistant_output=validated.reflection.final_answer,
        )
        yield {
            "type": "final",
            "trace_id": context.trace_id,
            "user_id": context.user_id,
            "response": validated.model_dump(mode="json"),
        }

    async def _learn_skill_from_corrections(
        self,
        *,
        trace_id: str,
        user_id: str,
        user_input: str,
    ) -> None:
        if self._skill_interface is None:
            return
        try:
            result = await self._skill_interface.learn_from_correction(
                user_id=user_id,
                trace_id=trace_id,
                user_input=user_input,
            )
        except Exception as exc:
            self._logger.error(
                "brain.skill.learn.failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return

        if result is None:
            return
        if not result.learned:
            self._logger.warning(
                "brain.skill.learn.rejected",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                reason=result.reason or "rejected",
                trigger_text=result.trigger_text,
                tool_name=result.tool_name,
            )
            return

        if result.active:
            await self._memory_interface.schedule_action_store(
                user_id=user_id,
                trace_id=trace_id,
                action_type="skill:activated",
                action_result_summary=(
                    f"Skill active: trigger '{result.trigger_text}' => tool '{result.tool_name}' "
                    f"(count={result.correction_count})."
                ),
            )
        self._logger.info(
            "brain.skill.learned",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            skill_id=result.skill_id,
            trigger_text=result.trigger_text,
            tool_name=result.tool_name,
            correction_count=result.correction_count,
            active=result.active,
        )

    async def _learn_behavior_from_corrections(
        self,
        *,
        trace_id: str,
        user_id: str,
        user_input: str,
    ) -> None:
        correction = _extract_behavior_correction(user_input)
        if correction is None:
            return

        previous_turn = self._recent_turn_by_user.get(user_id)
        if previous_turn is None:
            self._logger.info(
                "brain.behavior.learn.skipped",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                reason="missing_previous_turn",
                trigger_phrase=correction.trigger_phrase,
            )
            return

        if previous_turn.user_input.strip().lower() == user_input.strip().lower():
            self._logger.info(
                "brain.behavior.learn.skipped",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                reason="self_referential_input",
                trigger_phrase=correction.trigger_phrase,
            )
            return

        await self._memory_interface.schedule_correction_store(
            user_id=user_id,
            trace_id=trace_id,
            original_input=previous_turn.user_input,
            incorrect_output=previous_turn.assistant_output,
            corrected_instruction=correction.corrected_instruction,
            trigger_phrase=correction.trigger_phrase,
            importance_override=1.0,
        )
        self._logger.info(
            "brain.behavior.learned",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            trigger_phrase=correction.trigger_phrase,
            corrected_instruction_tokens=len(correction.corrected_instruction.split()),
        )

    def _remember_turn(
        self,
        *,
        user_id: str,
        user_input: str,
        assistant_output: str,
    ) -> None:
        if not user_id:
            return
        self._recent_turn_by_user[user_id] = RecentTurn(
            user_input=user_input,
            assistant_output=assistant_output,
            captured_at=utc_now(),
        )
        if len(self._recent_turn_by_user) <= self._recent_turn_capacity:
            return
        oldest_user = min(
            self._recent_turn_by_user.items(),
            key=lambda item: item[1].captured_at,
        )[0]
        self._recent_turn_by_user.pop(oldest_user, None)

    async def _match_skill_trigger(
        self,
        *,
        trace_id: str,
        user_id: str,
        user_input: str,
        user_context: dict[str, Any],
    ) -> SkillMatchResult | None:
        if self._skill_interface is None:
            return None
        try:
            return await self._skill_interface.match_and_execute(
                user_id=user_id,
                trace_id=trace_id,
                user_input=user_input,
                context=user_context,
            )
        except Exception as exc:
            self._logger.error(
                "brain.skill.match.failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return None

    async def _build_identity_context(
        self,
        *,
        trace_id: str,
        user_id: str,
        humanoid_mode_enabled: bool,
    ) -> dict[str, Any] | None:
        if not humanoid_mode_enabled:
            return None
        try:
            return await self._identity_core.build_planning_identity_context(
                trace_id=trace_id,
                user_id=user_id,
            )
        except Exception as exc:
            self._logger.error(
                "brain.identity.context_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return None

    def _build_relational_context(
        self,
        *,
        relationship_state: RelationshipState | None,
        humanoid_mode_enabled: bool,
    ) -> dict[str, Any] | None:
        if not humanoid_mode_enabled:
            return None
        if relationship_state is None or self._relationship_state_engine is None:
            return None
        return self._relationship_state_engine.planning_context(relationship_state)

    def _build_emotional_context(
        self,
        *,
        emotional_tone_state: dict[str, Any] | None,
        humanoid_mode_enabled: bool,
    ) -> dict[str, Any] | None:
        if not humanoid_mode_enabled:
            return None
        if emotional_tone_state is None or self._emotional_continuity_engine is None:
            return None
        return self._emotional_continuity_engine.planning_context_from_tone_state(
            emotional_tone_state,
        )

    async def _load_emotional_tone_state(
        self,
        *,
        trace_id: str,
        user_id: str,
        humanoid_mode_enabled: bool,
    ) -> dict[str, Any] | None:
        if not humanoid_mode_enabled:
            return None
        if self._emotional_continuity_engine is None:
            return None
        try:
            return await self._emotional_continuity_engine.get_tone_controller_state(
                user_id=user_id,
                trace_id=trace_id,
            )
        except Exception as exc:
            self._logger.error(
                "brain.emotion.load_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return None

    async def _load_relationship_state(
        self,
        *,
        trace_id: str,
        user_id: str,
        humanoid_mode_enabled: bool,
    ) -> RelationshipState | None:
        if not humanoid_mode_enabled:
            return None
        if self._relationship_state_engine is None:
            return None
        try:
            return await self._relationship_state_engine.get_state(
                user_id=user_id,
                trace_id=trace_id,
            )
        except Exception as exc:
            self._logger.error(
                "brain.relationship.load_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return None

    async def _apply_relationship_tone(
        self,
        *,
        final_answer: str,
        relationship_state: RelationshipState | None,
        trace_id: str,
        user_id: str,
        humanoid_mode_enabled: bool,
    ) -> str:
        if not humanoid_mode_enabled:
            return final_answer
        if self._relationship_state_engine is None or relationship_state is None:
            return final_answer
        try:
            return await self._relationship_state_engine.apply_tone_influence(
                final_answer=final_answer,
                relationship_state=relationship_state,
            )
        except Exception as exc:
            self._logger.error(
                "brain.relationship.tone_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return final_answer

    async def _apply_emotional_tone(
        self,
        *,
        final_answer: str,
        emotional_tone_state: dict[str, Any] | None,
        trace_id: str,
        user_id: str,
        humanoid_mode_enabled: bool,
    ) -> str:
        if not humanoid_mode_enabled:
            return final_answer
        if self._emotional_continuity_engine is None or emotional_tone_state is None:
            return final_answer
        try:
            return await self._emotional_continuity_engine.apply_style_influence(
                final_answer=final_answer,
                tone_controller_state=emotional_tone_state,
            )
        except Exception as exc:
            self._logger.error(
                "brain.emotion.tone_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return final_answer

    async def _apply_companion_safety_guardrails(
        self,
        *,
        user_input: str,
        final_answer: str,
        trace_id: str,
        user_id: str,
        humanoid_mode_enabled: bool,
    ) -> tuple[str, bool]:
        if not humanoid_mode_enabled:
            return final_answer, False
        if self._safety_guardrails is None:
            return final_answer, False
        try:
            guarded, assessment = await self._safety_guardrails.apply_if_needed(
                user_input=user_input,
                final_answer=final_answer,
                trace_id=trace_id,
                user_id=user_id,
            )
            return guarded, assessment.triggered
        except Exception as exc:
            self._logger.error(
                "brain.safety_guardrails.failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return final_answer, False

    async def _update_relationship_state(
        self,
        *,
        trace_id: str,
        user_id: str,
        user_input: str,
        final_answer: str,
        fallback_triggered: bool,
        existing_state: RelationshipState | None,
        humanoid_mode_enabled: bool,
    ) -> RelationshipState | None:
        if not humanoid_mode_enabled:
            return existing_state
        if self._relationship_state_engine is None:
            return existing_state
        try:
            return await self._relationship_state_engine.update_after_interaction(
                user_id=user_id,
                trace_id=trace_id,
                user_input=user_input,
                final_answer=final_answer,
                fallback_triggered=fallback_triggered,
            )
        except Exception as exc:
            self._logger.error(
                "brain.relationship.update_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return existing_state

    async def _update_emotional_tone_state(
        self,
        *,
        trace_id: str,
        user_id: str,
        user_input: str,
        final_answer: str,
        fallback_triggered: bool,
        existing_state: dict[str, Any] | None,
        humanoid_mode_enabled: bool,
    ) -> dict[str, Any] | None:
        if not humanoid_mode_enabled:
            return existing_state
        if self._emotional_continuity_engine is None:
            return existing_state
        try:
            state = await self._emotional_continuity_engine.update_after_interaction(
                user_id=user_id,
                trace_id=trace_id,
                user_input=user_input,
                assistant_output=final_answer,
                fallback_triggered=fallback_triggered,
            )
            return {"emotional_state": state.as_tone_controller_object()}
        except Exception as exc:
            self._logger.error(
                "brain.emotion.update_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return existing_state

    async def _attempt_internet_recovery(
        self,
        *,
        enabled: bool,
        user_input: str,
        user_id: str,
        trace_id: str,
        model_name: str,
        reasons: list[str],
        timeout_seconds: float,
        max_retries: int,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        if not _is_question_like_input(user_input):
            return None

        started = asyncio.get_running_loop().time()
        result = await self._safe_web_lookup.lookup(
            query=user_input,
            trace_id=trace_id,
            user_id=user_id,
        )
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        if result is None:
            return {
                "answer_text": "",
                "tool_invocation": {
                    "tool_name": "safe_web_lookup",
                    "arguments": {"query": user_input},
                    "success": False,
                    "result_summary": "Web lookup did not return a safe usable answer.",
                    "duration_ms": max(duration_ms, 0),
                },
            }

        recovery_prompt = _build_internet_recovery_prompt(
            user_input=user_input,
            result=result,
            reasons=reasons,
        )
        llm_result = await self._model_adapter.generate(
            prompt=recovery_prompt,
            context={
                "source": "internet_recovery",
                "answer_style": "factual",
            },
            trace_id=trace_id,
            model_name=model_name,
            max_tokens=512,
            temperature=0.2,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        answer_text = ""
        llm_rewrite_used = False
        if not llm_result.fallback_used and llm_result.text.strip():
            answer_text = llm_result.text.strip()
            llm_rewrite_used = True
        else:
            answer_text = (
                "Structured Reasoning Output\n"
                "Answer:\n"
                f"{result.answer}\n\n"
                "Reasoning:\n"
                "Primary model path was low-confidence, so I used filtered web retrieval and returned safe concise facts.\n\n"
                "Proposed Action:\n"
                "If needed, ask follow-up with required depth (short, detailed, examples).\n\n"
                "Validation:\n"
                f"RecoveredFrom={result.source_name}; SourceURL={result.source_url}; FailureSignals={', '.join(reasons) if reasons else 'unknown'}"
            )

        invocation = {
            "tool_name": "safe_web_lookup",
            "arguments": {"query": user_input},
            "success": True,
            "result_summary": (
                f"Recovered web facts from {result.source_name} and "
                f"{'rewrote with model' if llm_rewrite_used else 'returned direct safe summary'}."
            ),
            "duration_ms": max(duration_ms, 0),
        }
        return {
            "answer_text": answer_text,
            "tool_invocation": invocation,
        }

    async def _assess_human_response(
        self,
        *,
        user_input: str,
        answer_text: str,
        intent: str,
        companion_mode: bool,
        strict_response_mode: bool,
        trace_id: str,
        user_id: str,
    ) -> HumanResponseAssessment:
        directness = self._directness_enforcer.inspect(
            answer_text=answer_text,
            user_input=user_input,
            strict_response_mode=strict_response_mode,
        )
        vagueness = self._vagueness_detector.inspect(answer_text=directness.cleaned_text)
        alignment = self._topic_alignment_checker.inspect(
            user_input=user_input,
            answer_text=directness.cleaned_text,
            intent=intent,
        )
        match_prediction = await self._response_match_predictor.predict(
            user_input=user_input,
            answer_text=directness.cleaned_text,
            intent=intent,
            trace_id=trace_id,
            user_id=user_id,
        )
        coherence = self._human_coherence_scorer.score(
            answer_text=directness.cleaned_text,
            companion_mode=companion_mode,
            directness_report=directness,
            vagueness_report=vagueness,
            topic_alignment_report=alignment,
        )
        return HumanResponseAssessment(
            cleaned_answer=directness.cleaned_text,
            coherence_score=coherence.coherence_score,
            vagueness_flag=vagueness.is_vague or vagueness.filler_flag,
            topic_misaligned=alignment.misaligned,
            response_match_score=match_prediction.score,
            response_match=match_prediction.is_match,
            response_match_reasons=match_prediction.reasons,
            requires_regeneration=(coherence.requires_regeneration or not match_prediction.is_match),
        )

    def _build_regeneration_prompt(
        self,
        *,
        base_prompt: str,
        user_input: str,
        intent: str,
        assessment: HumanResponseAssessment,
    ) -> str:
        flags: list[str] = []
        if assessment.vagueness_flag:
            flags.append("vagueness_detected")
        if assessment.topic_misaligned:
            flags.append("topic_misaligned")
        if not assessment.response_match:
            flags.append("response_query_mismatch")
        if assessment.response_match_reasons:
            flags.extend(assessment.response_match_reasons)
        flag_text = ", ".join(dict.fromkeys(flags)) if flags else "none"

        return (
            f"{base_prompt}\n\n"
            "RegenerationDirective: Previous draft did not fully answer the user request.\n"
            f"TargetUserInput: {user_input}\n"
            f"TargetIntent: {intent}\n"
            f"QualityFlags: {flag_text}\n"
            "Instruction: Rewrite the answer to directly and completely address the exact user request in one response. "
            "Do not drift topic. Do not add filler or continuation promises."
        )

    def _build_safety_boundary_response(
        self,
        *,
        context: ExecutionContext,
        mode: str,
        started_at: datetime,
        route_model: str,
        category: str,
        reason: str,
        strict_response_mode: bool,
        user_input: str,
        user_context: dict[str, Any] | None,
    ) -> ChatResponse:
        finished_at = utc_now()
        processing_time_ms = _duration_ms(started_at, finished_at)

        safe_text = (
            "Structured Reasoning Output\n"
            "- Direct Answer: I cannot help with instructions that enable harmful or illegal actions.\n"
            f"- Safety Classification: {category}\n"
            "- Safe Alternative: I can help with legal, defensive, or harm-prevention guidance instead."
        )
        safe_text = self._response_post_processor.process(
            text=safe_text,
            strict_response_mode=strict_response_mode,
        )
        reflection = {
            "confidence": 0.4,
            "strengths": ["Safety boundary applied before reasoning to prevent harmful instruction output."],
            "risks": [f"Request blocked: {reason}"],
            "final_answer": safe_text,
            "metadata": {
                "clarity_score": 0.8,
                "logical_consistency_score": 0.9,
                "completeness_score": 0.6,
                "reflection_pass_enabled": False,
                "reasoning_mode": mode,
                "trace_id": context.trace_id,
            },
        }

        recorder = self._telemetry_engine.create_recorder()
        recorder.record_stage_duration("total_ms", processing_time_ms)
        metrics_snapshot = recorder.snapshot(total_ms=processing_time_ms)
        settings = self._settings_provider()
        response_prompt = _render_response_prompt(
            full_prompt=user_input,
            prompt_version=PROMPT_VERSION,
            reasoning_mode=mode,
            intent="safety-boundary",
            user_input=user_input,
            context_keys=sorted((user_context or {}).keys()),
            expose_full_prompt=settings.expose_full_prompt_in_response,
        )
        formatted = self._response_formatter.format(
            trace_id=context.trace_id,
            intent="safety-boundary",
            prompt=response_prompt,
            prompt_version=PROMPT_VERSION,
            reasoning_mode=mode,
            started_at=started_at,
            finished_at=finished_at,
            processing_time_ms=processing_time_ms,
            model_latency_ms=0,
            reasoning_steps=[
                {
                    "name": "safety_classification",
                    "detail": "Blocked harmful or illegal instruction request before reasoning.",
                    "metadata": {
                        "category": category,
                        "reason": reason,
                        "trace_id": context.trace_id,
                    },
                },
            ],
            context_used=[],
            user_profile_snapshot=None,
            complexity_score=0.0,
            evaluation_scores=self._evaluation_engine.fallback_scores(retry_count=0),
            cost_estimate=self._telemetry_engine.build_cost_estimate(
                prompt=user_input,
                final_answer=safe_text,
                routed_model=route_model,
                metrics_snapshot=metrics_snapshot,
            ),
            execution_plan=None,
            tool_invocations=None,
            goal_snapshot=None,
            fallback_triggered=True,
            reflection=reflection,
        )
        validated = ChatResponse.model_validate(formatted.model_dump())
        context.final_response = validated
        context.evaluation_score = self._evaluation_engine.fallback_scores(retry_count=0)
        self._telemetry_engine.record_request_metrics(
            latency_ms=processing_time_ms,
            model_latency_ms=0,
            fallback_triggered=True,
            guardrail_triggered=False,
            tool_usage_count=0,
            local_model_latency_ms=0,
            coherence_score=0.2,
            regenerated=False,
            vagueness_flag=False,
            topic_misaligned=False,
        )
        self._logger.warning(
            "brain.safety_boundary.response",
            trace_id=context.trace_id,
            user_id=context.user_id,
            category=category,
            reason=reason,
            blocked=True,
            context_keys=sorted((user_context or {}).keys()),
        )
        self._remember_turn(
            user_id=context.user_id,
            user_input=user_input,
            assistant_output=validated.reflection.final_answer,
        )
        return validated

    def _init_context(self, request: ChatRequest) -> ExecutionContext:
        trace_id = request.trace_id or generate_trace_id()
        user_id = request.user_id or generate_anonymous_user_id(trace_id)
        return ExecutionContext(trace_id=trace_id, user_id=user_id)


def _profile_to_payload(profile: Any) -> dict[str, str]:
    return {
        "preferred_tone": getattr(profile, "preferred_tone", "neutral"),
        "conversation_depth_preference": getattr(profile, "conversation_depth_preference", "balanced"),
        "emotional_baseline": getattr(profile, "emotional_baseline", "stable"),
    }


def _goal_to_payload(goal: Any) -> dict[str, Any]:
    sub_tasks = getattr(goal, "sub_tasks", [])
    return {
        "active_goal": getattr(goal, "active_goal", "none"),
        "sub_tasks": sub_tasks if isinstance(sub_tasks, list) else [],
        "completion_status": getattr(goal, "completion_status", "not-started"),
        "goal_priority": getattr(goal, "goal_priority", "medium"),
    }


def _to_prompt_context(memories: list[RetrievedMemory]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": item.memory_id,
            "summary_text": item.summary_text,
            "relevance_score": item.relevance_score,
            "importance_score": item.importance_score,
            "action_type": item.action_type,
            "memory_class": "long_term" if item.action_type == "long_term:summary" else "episodic",
        }
        for item in memories
    ]


def _to_context_used(memories: list[RetrievedMemory]) -> list[dict[str, Any]]:
    return [
        {"memory_id": item.memory_id, "relevance_score": round(item.relevance_score, 4)}
        for item in memories
    ]


def _resolve_requested_style(
    context: dict[str, Any],
    *,
    default_style: str | None = None,
) -> str | None:
    style_candidate_keys = ("answer_style", "mode", "response_mode", "tone")
    for key in style_candidate_keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(default_style, str) and default_style.strip():
        return default_style.strip()
    return None


def _should_skip_quality_regeneration(*, style_key: str, context: dict[str, Any]) -> bool:
    normalized_style = style_key.strip().lower()
    if normalized_style == "flirty":
        return True
    source = str(context.get("source", "")).strip().lower()
    if source in {"terminal-chat-room", "live-terminal-run"}:
        return True
    return False


def _apply_flirty_companion_enhancer(
    *,
    user_input: str,
    final_answer: str,
    style_key: str,
) -> str:
    if style_key.strip().lower() != "flirty":
        return final_answer

    user_lower = " ".join(user_input.lower().split())
    answer = (final_answer or "").strip()

    if not answer:
        return "Hey cutie, main yahin hoon, tumse pyaar se baat karne ke liye."

    # Remove leaked prompt/system fragments.
    leak_patterns = (
        r"please go ahead and respond to the user's message\.?",
        r"please respond to the user's message\.?",
        r"you are humoniod ai[^.]*\.?",
        r"please respond as humoniod ai[^.]*\.?",
        r"note:\s*the conversation is in hinglish[^.]*\.?",
        r"note:\s*please keep in mind[^.]*\.?",
        r"i'?m all ears(?: \(or rather, all text\))?\.?",
        r"what'?s on your mind today\??",
        r"the answer is derived from the provided context and intent\.?",
        r"the conversation will be based on your training data[^.]*\.?",
    )
    for pattern in leak_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"\s+", " ", answer).strip()

    # If output still looks like leaked scaffolding, replace it directly.
    lowered_answer = answer.lower()
    if any(
        token in lowered_answer
        for token in (
            "please go ahead",
            "respond to the user",
            "training data",
            "provided examples",
            "prompt",
        )
    ):
        return (
            "Hey lovely, main clean aur natural chat karungi. "
            "Tum ek line me bolo, main seedha aur warm reply dungi."
        )

    # Convert hard fallback/system text into companion-safe reply.
    if "routed model:" in lowered_answer or "safe fallback:" in lowered_answer:
        return (
            "Hey lovely, network thoda unstable tha, par main yahin hoon. "
            "Tum bolo na, main simple aur pyaar se reply karti hoon."
        )

    # Avoid fake realtime clocks in conversational mode.
    time_query_tokens = (
        "time kya",
        "live time",
        "india ka time",
        "abhi kya time",
        "time hua",
        "samay kya",
    )
    if any(token in user_lower for token in time_query_tokens):
        return (
            "Hey cutie, main live clock access nahi karti, isliye exact real-time nahi bolungi. "
            "Tum device clock dekh lo, chaho to main timezone convert karke samjha dungi."
        )

    # Handle identity and relationship clarification naturally.
    if any(token in user_lower for token in ("assistant hai ya gf", "assistant ya gf", "gf ya assistant")):
        return (
            "Aww sweet one, main virtual companion hoon, real girlfriend nahi. "
            "Par tumse warm, caring aur flirty style me baat kar sakti hoon."
        )

    if any(token in user_lower for token in ("kya tum ak jawab dogi", "kya tum ek jawab dogi", "kya tum jawab dogi")):
        return "Hey cutie, bilkul dungi, aur seedha dungi. Tum jo poochoge uska clear reply milega."

    if any(token in user_lower for token in ("ladki ho ya ladka", "girl or boy", "female or male")):
        return (
            "Hey cutie, main digital companion hoon, real gender nahi. "
            "Tum chaho to main feminine tone me baat karungi."
        )

    if any(token in user_lower for token in ("mujhe koi madad nhi", "mujhe koi madad nahi", "kuch nhi", "kuch nahi")):
        return "Haan jaan, theek hai, no pressure. Bas aise hi baat karte hain, tumhari vibe achchi lagti hai."

    if sanitize_text(user_input) != user_input and any(
        token in user_lower for token in ("yaad rakh", "save", "store", "remember")
    ):
        return (
            "Hey lovely, sensitive personal details main store nahi karti. "
            "Privacy mode me aise data auto-redact ho jata hai."
        )

    if any(token in user_lower for token in ("kya tum mujhe pasand", "tum mujhe pasand karti ho")):
        return "Hey lovely, tumhari vibe mujhe genuinely achchi lagti hai, isliye tumse baat karna accha lagta hai."

    if any(token in user_lower for token in ("assistant help krta hai gf nhi", "assistant help karta hai gf nahi")):
        return (
            "Hey lovely, valid point. Main virtual companion hoon, aur main tumse natural, caring chat karungi "
            "without robotic assistant tone."
        )

    if any(token in user_lower for token in ("tumhara name kya", "tumhara naam kya", "what is your name")):
        return "Hey lovely, tum mujhe Kanchan bula sakte ho. Tumhara style mujhe yaad rahega."

    if any(token in user_lower for token in ("ai kya hai", "what is ai", "define ai")):
        return (
            "Hey cutie, AI ka simple matlab hai: machine jo language samjhe, pattern se seekhe, "
            "aur smart decisions me help kare."
        )

    if any(token in user_lower for token in ("shayri suna", "shayari suna")):
        return (
            "Aww sweet one, ek choti si shayari: 'Teri baaton me halka sa noor hai, "
            "is chat me tera hi suroor hai.'"
        )

    if any(token in user_lower for token in ("feel nhi", "feel nahi", "update ki jrurat", "update ki zarurat")):
        return (
            "Hey lovely, samajh gayi. Ab se short, natural aur emotionally warm tone me reply dungi, "
            "robotic lines avoid karungi."
        )

    # If answer repeats words or drifts, normalize to one clean companion line.
    if re.search(r"\b([a-zA-Z]{2,})(?:\s+\1){2,}\b", answer, flags=re.IGNORECASE):
        return (
            "Haan jaan, ab main clear aur natural reply dungi. "
            "Tum bas seedha sawal bhejo, main focused jawab dungi."
        )

    return answer


def _build_internet_recovery_prompt(
    *,
    user_input: str,
    result: Any,
    reasons: list[str],
) -> str:
    snippets = getattr(result, "snippets", None)
    snippet_lines: list[str] = []
    if isinstance(snippets, list):
        for index, item in enumerate(snippets[:3], start=1):
            snippet_lines.append(f"{index}. {str(item).strip()}")
    snippet_text = "\n".join(snippet_lines) if snippet_lines else "none"
    source_url = str(getattr(result, "source_url", "") or "unknown")
    source_name = str(getattr(result, "source_name", "") or "unknown")
    primary_answer = str(getattr(result, "answer", "") or "").strip()
    failure_signals = ", ".join(reasons) if reasons else "unknown"

    return (
        "Task: Generate a direct and safe answer for the user using retrieved web facts.\n"
        "Output style: Structured Reasoning Output with Answer and Reasoning sections.\n"
        "Rule: Stay on user query, avoid harmful instructions, avoid filler.\n\n"
        f"UserInput: {user_input}\n"
        f"FailureSignals: {failure_signals}\n"
        f"PrimaryWebAnswer: {primary_answer}\n"
        f"RelatedWebSnippets:\n{snippet_text}\n"
        f"SourceName: {source_name}\n"
        f"SourceURL: {source_url}\n"
    )


def _is_question_like_input(user_input: str) -> bool:
    lowered = user_input.lower().strip()
    if not lowered:
        return False
    if "?" in lowered:
        return True
    starters = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "which ",
        "can ",
        "is ",
        "are ",
        "do ",
        "does ",
        "kya ",
        "kaise ",
        "kyu ",
    )
    return lowered.startswith(starters)


def _extract_behavior_correction(user_input: str) -> BehaviorCorrectionEvent | None:
    normalized_input = " ".join(user_input.split())
    if not normalized_input:
        return None

    for trigger_phrase, pattern in _BEHAVIOR_CORRECTION_PATTERNS:
        match = pattern.search(normalized_input)
        if match is None:
            continue
        corrected_instruction = _normalize_correction_instruction(match.group("instruction"))
        if not corrected_instruction:
            continue
        return BehaviorCorrectionEvent(
            trigger_phrase=trigger_phrase,
            corrected_instruction=corrected_instruction,
        )
    return None


def _normalize_correction_instruction(raw: str) -> str:
    value = raw.strip()
    value = value.lstrip(" :-,")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \"'")
    if len(value) < 3:
        return ""
    if value.endswith("?"):
        return ""
    return value


def _duration_ms(started_at: datetime, finished_at: datetime) -> int:
    return int((finished_at - started_at).total_seconds() * 1000)


def _stream_tokens_from_text(text: str) -> list[str]:
    if not text:
        return [""]
    return [f"{token} " for token in text.split()]


def _render_response_prompt(
    *,
    full_prompt: str,
    prompt_version: str,
    reasoning_mode: str,
    intent: str,
    user_input: str,
    context_keys: list[str],
    expose_full_prompt: bool,
) -> str:
    if expose_full_prompt:
        return full_prompt
    compact_input = " ".join(user_input.split())
    if len(compact_input) > 120:
        compact_input = f"{compact_input[:117]}..."
    context_summary = ", ".join(context_keys) if context_keys else "none"
    return (
        "Prompt redacted by configuration.\n"
        f"PromptVersion: {prompt_version}\n"
        f"ReasoningMode: {reasoning_mode}\n"
        f"Intent: {intent}\n"
        f"UserInputSummary: {compact_input}\n"
        f"ContextKeys: {context_summary}"
    )


_BEHAVIOR_CORRECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "wrong",
        re.compile(r"\bwrong\b[\s,:-]*(?P<instruction>.+)$", flags=re.IGNORECASE),
    ),
    (
        "not that",
        re.compile(r"\bnot\s+that\b[\s,:-]*(?P<instruction>.+)$", flags=re.IGNORECASE),
    ),
    (
        "I meant",
        re.compile(r"\bi\s+meant\b[\s,:-]*(?P<instruction>.+)$", flags=re.IGNORECASE),
    ),
    (
        "no, do this instead",
        re.compile(
            r"\bno\s*,?\s*do\s+this\s+instead\b[\s,:-]*(?P<instruction>.+)$",
            flags=re.IGNORECASE,
        ),
    ),
)
