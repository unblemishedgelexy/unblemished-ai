from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, confloat


class RuntimeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    app_name: str
    app_version: str
    reasoning_mode: str
    reflection_enabled: bool
    regeneration_enabled: bool
    model_backend_configured: str
    model_backend_effective: str
    response_match_model_enabled: bool
    response_match_mode: str
    model_ready: bool
    model_routing_enabled: bool
    self_evaluation_enabled: bool
    tool_execution_enabled: bool
    embedding_provider: str
    embedding_dim: int = Field(ge=1)
    embedding_enabled: bool
    privacy_redaction_enabled: bool
    privacy_remove_sensitive_context_keys: bool
    relationship_memory_text_enabled: bool
    database_driver: str
    telemetry_enabled: bool
    auth_enabled: bool
    auth_mode: str
    auth_header: str
    rate_limit_enabled: bool
    requests_per_minute: int = Field(ge=1)
    uptime_seconds: int = Field(ge=0)
    server_time_utc: datetime


class MemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    database_driver: str
    memory_ready: bool
    memory_entries_count: int = Field(ge=0)
    embedding_enabled: bool
    embedding_provider: str
    embedding_dim: int = Field(ge=1)


class SkillItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    skill_id: str
    user_id: str
    trigger_text: str
    trigger_type: str
    tool_name: str
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    correction_count: int = Field(ge=0)
    source: str
    active: bool
    created_at: datetime
    updated_at: datetime


class SkillsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str
    total_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    skills: list[SkillItemResponse]


class SkillUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str = Field(min_length=1)
    trigger_text: str = Field(min_length=3)
    trigger_type: str = Field(default="contains")
    tool_name: str = Field(min_length=2)
    tool_arguments: dict[str, Any] | None = None
    active: bool = True


class SkillUpsertResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    skill: SkillItemResponse


class SkillDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    deleted: bool
    skill_id: str


class TelemetrySnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_count: int = Field(ge=0)
    latency_avg_ms: confloat(ge=0.0)
    model_latency_avg_ms: confloat(ge=0.0)
    local_model_latency_avg_ms: confloat(ge=0.0)
    fallback_rate: confloat(ge=0.0, le=1.0)
    coherence_score_avg: confloat(ge=0.0, le=1.0)
    regeneration_rate: confloat(ge=0.0, le=1.0)
    guardrail_trigger_count: int = Field(ge=0)
    tool_usage_count: int = Field(ge=0)
    vagueness_flag_count: int = Field(ge=0)
    topic_misalignment_rate: confloat(ge=0.0, le=1.0)
    telemetry_enabled: bool
    dropped_event_count: int = Field(ge=0)


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    model_ready: bool
    memory_ready: bool
    profile_engine_ready: bool
    tool_engine_ready: bool
    goal_engine_ready: bool
    skill_engine_ready: bool
    telemetry_enabled: bool
    uptime_seconds: int = Field(ge=0)
    memory_entries_count: int = Field(ge=0)
    telemetry: TelemetrySnapshotResponse


class AnalyzerEventResponse(BaseModel):
    model_config = ConfigDict(extra="allow", strict=False)

    event: str
    timestamp: datetime
    reasons: list[str] = Field(default_factory=list)
    status: str
    model_ready: bool
    memory_ready: bool
    skill_engine_ready: bool
    tool_engine_ready: bool
    memory_entries_count: int = Field(ge=0)
    request_count: int = Field(ge=0)
    tool_usage_count: int = Field(ge=0)
    fallback_rate: confloat(ge=0.0, le=1.0)
    latency_avg_ms: confloat(ge=0.0)
    dropped_event_count: int = Field(ge=0)
    reasoning_mode: str
    backend: str


class AnalyzerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    analyzer_enabled: bool
    analyzer_running: bool
    file_path: str
    entry_count: int = Field(ge=0)
    records: list[AnalyzerEventResponse]
