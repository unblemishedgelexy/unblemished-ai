from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, confloat


class ReasoningStepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    detail: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    clarity_score: confloat(ge=0.0, le=1.0)
    logical_consistency_score: confloat(ge=0.0, le=1.0)
    completeness_score: confloat(ge=0.0, le=1.0)
    reflection_pass_enabled: bool
    reasoning_mode: str
    trace_id: str


class ReflectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confidence: confloat(ge=0.0, le=1.0)
    strengths: list[str]
    risks: list[str]
    final_answer: str
    metadata: ReflectionMetadataResponse


class ContextUsedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    memory_id: str
    relevance_score: confloat(ge=0.0, le=1.0)


class UserProfileSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    preferred_tone: str
    conversation_depth_preference: str
    emotional_baseline: str


class EvaluationScoresResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer_quality_score: confloat(ge=0.0, le=1.0)
    coherence_score: confloat(ge=0.0, le=1.0)
    usefulness_score: confloat(ge=0.0, le=1.0)
    retry_count: int = Field(ge=0, le=1, default=0)


class CostEstimateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    estimated_token_cost: confloat(ge=0.0)
    estimated_inference_cost: confloat(ge=0.0)
    total_latency_breakdown: dict[str, int] = Field(default_factory=dict)
    routed_model: str


class ExecutionPlanStepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    step: str
    requires_tool: bool
    required_tool_name: str | None = None


class ToolInvocationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result_summary: str
    duration_ms: int = Field(ge=0)


class GoalSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    active_goal: str
    sub_tasks: list[str]
    completion_status: str
    goal_priority: str


class FinalAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    final_answer: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    trace_id: str
    intent: str
    prompt: str
    prompt_version: str
    reasoning_mode: str
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    processing_time_ms: int = Field(ge=0)
    model_latency_ms: int = Field(ge=0)
    reasoning_steps: list[ReasoningStepResponse]
    context_used: list[ContextUsedResponse]
    user_profile_snapshot: UserProfileSnapshotResponse | None = None
    complexity_score: confloat(ge=0.0, le=1.0) | None = None
    evaluation_scores: EvaluationScoresResponse | None = None
    cost_estimate: CostEstimateResponse | None = None
    execution_plan: list[ExecutionPlanStepResponse] | None = None
    tool_invocations: list[ToolInvocationResponse] | None = None
    goal_snapshot: GoalSnapshotResponse | None = None
    fallback_triggered: bool = False
    reflection: ReflectionResponse
