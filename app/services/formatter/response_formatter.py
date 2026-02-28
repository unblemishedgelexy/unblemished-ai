from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas.response_schema import (
    ChatResponse,
    CostEstimateResponse,
    ContextUsedResponse,
    ExecutionPlanStepResponse,
    EvaluationScoresResponse,
    GoalSnapshotResponse,
    ReasoningStepResponse,
    ReflectionResponse,
    ToolInvocationResponse,
    UserProfileSnapshotResponse,
)


class ResponseFormatter:
    def format(
        self,
        trace_id: str,
        intent: str,
        prompt: str,
        prompt_version: str,
        reasoning_mode: str,
        started_at: datetime,
        finished_at: datetime,
        processing_time_ms: int,
        model_latency_ms: int,
        reasoning_steps: list[dict[str, Any]],
        context_used: list[dict[str, Any]],
        user_profile_snapshot: dict[str, Any] | None,
        complexity_score: float | None,
        evaluation_scores: dict[str, Any] | None,
        cost_estimate: dict[str, Any] | None,
        execution_plan: list[dict[str, Any]] | None,
        tool_invocations: list[dict[str, Any]] | None,
        goal_snapshot: dict[str, Any] | None,
        fallback_triggered: bool,
        reflection: dict[str, Any],
    ) -> ChatResponse:
        return ChatResponse(
            trace_id=trace_id,
            intent=intent,
            prompt=prompt,
            prompt_version=prompt_version,
            reasoning_mode=reasoning_mode,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=processing_time_ms,
            processing_time_ms=processing_time_ms,
            model_latency_ms=model_latency_ms,
            reasoning_steps=[ReasoningStepResponse(**step) for step in reasoning_steps],
            context_used=[ContextUsedResponse(**item) for item in context_used],
            user_profile_snapshot=(
                UserProfileSnapshotResponse(**user_profile_snapshot)
                if user_profile_snapshot is not None
                else None
            ),
            complexity_score=complexity_score,
            evaluation_scores=(
                EvaluationScoresResponse(**evaluation_scores)
                if evaluation_scores is not None
                else None
            ),
            cost_estimate=(
                CostEstimateResponse(**cost_estimate)
                if cost_estimate is not None
                else None
            ),
            execution_plan=(
                [ExecutionPlanStepResponse(**item) for item in execution_plan]
                if execution_plan is not None
                else None
            ),
            tool_invocations=(
                [ToolInvocationResponse(**item) for item in tool_invocations]
                if tool_invocations is not None
                else None
            ),
            goal_snapshot=(
                GoalSnapshotResponse(**goal_snapshot)
                if goal_snapshot is not None
                else None
            ),
            fallback_triggered=fallback_triggered,
            reflection=ReflectionResponse(**reflection),
        )
