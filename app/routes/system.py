from __future__ import annotations

import time
from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import HTMLResponse

from app.core.config import get_settings
from app.core.dependencies import get_brain, get_runtime_analyzer, get_skill_interface, get_telemetry_exporter
from app.core.runtime_analyzer import RuntimeAnalyzer
from app.core.telemetry_exporter import TelemetryExporter
from app.schemas.system_schema import (
    AnalyzerResponse,
    MemoryResponse,
    RuntimeResponse,
    SkillDeleteResponse,
    SkillItemResponse,
    SkillsResponse,
    SkillUpsertRequest,
    SkillUpsertResponse,
    StatusResponse,
    TelemetrySnapshotResponse,
)
from app.services.brain.brain_interface import BrainInterface
from app.services.skills.skill_interface import SkillDefinition, SkillInterface
from app.utils.helpers import utc_now

router = APIRouter(prefix="/system", tags=["system"])
_SYSTEM_STARTED_MONOTONIC = time.monotonic()
_DASHBOARD_HTML = FilePath(__file__).resolve().parents[1] / "static" / "system_dashboard.html"


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def system_dashboard() -> HTMLResponse:
    if not _DASHBOARD_HTML.exists():
        raise HTTPException(status_code=404, detail="dashboard_not_found")
    return HTMLResponse(_DASHBOARD_HTML.read_text(encoding="utf-8"))


@router.get("/runtime", response_model=RuntimeResponse)
async def get_runtime(
    brain: BrainInterface = Depends(get_brain),
) -> RuntimeResponse:
    settings = get_settings()
    return RuntimeResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        reasoning_mode=settings.reasoning_mode,
        reflection_enabled=settings.reasoning_profile.reflection_pass,
        regeneration_enabled=settings.self_evaluation_enabled,
        model_backend_configured=settings.model_backend,
        model_backend_effective=brain.model_backend_effective(),
        response_match_model_enabled=(
            settings.response_match_model_enabled and brain.can_use_model_judge()
        ),
        response_match_mode="smart_gate",
        model_ready=brain.is_model_ready(),
        model_routing_enabled=settings.model_routing_enabled,
        self_evaluation_enabled=settings.self_evaluation_enabled,
        tool_execution_enabled=settings.tool_execution_enabled,
        embedding_provider=settings.embedding_provider,
        embedding_dim=settings.embedding_dim,
        embedding_enabled=settings.embedding_enabled,
        privacy_redaction_enabled=settings.privacy_redaction_enabled,
        privacy_remove_sensitive_context_keys=settings.privacy_remove_sensitive_context_keys,
        relationship_memory_text_enabled=settings.relationship_memory_text_enabled,
        database_driver=settings.database_driver,
        telemetry_enabled=settings.telemetry_enabled,
        auth_enabled=settings.auth_enabled,
        auth_mode=settings.auth_mode,
        auth_header=settings.auth_api_key_header,
        rate_limit_enabled=settings.rate_limit_enabled,
        requests_per_minute=settings.requests_per_minute,
        uptime_seconds=int(time.monotonic() - _SYSTEM_STARTED_MONOTONIC),
        server_time_utc=utc_now(),
    )


@router.get("/memory", response_model=MemoryResponse)
async def get_memory(
    brain: BrainInterface = Depends(get_brain),
) -> MemoryResponse:
    settings = get_settings()
    memory_ready = await brain.is_memory_ready()
    memory_entries_count = await brain.memory_entries_count() if memory_ready else 0
    return MemoryResponse(
        status="ok" if memory_ready else "degraded",
        database_driver=settings.database_driver,
        memory_ready=memory_ready,
        memory_entries_count=memory_entries_count,
        embedding_enabled=settings.embedding_enabled,
        embedding_provider=settings.embedding_provider,
        embedding_dim=settings.embedding_dim,
    )


@router.get("/skills", response_model=SkillsResponse)
async def get_skills(
    user_id: str = Query(..., min_length=1),
    include_inactive: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
    skill_interface: SkillInterface = Depends(get_skill_interface),
) -> SkillsResponse:
    skills = await skill_interface.list_skills(
        user_id=user_id,
        include_inactive=include_inactive,
        limit=limit,
    )
    active_count = len([item for item in skills if item.active])
    return SkillsResponse(
        user_id=user_id,
        total_count=len(skills),
        active_count=active_count,
        skills=[_to_skill_item(item) for item in skills],
    )


@router.post("/skill", response_model=SkillUpsertResponse)
async def upsert_skill(
    payload: SkillUpsertRequest,
    skill_interface: SkillInterface = Depends(get_skill_interface),
) -> SkillUpsertResponse:
    try:
        created = await skill_interface.upsert_skill(
            user_id=payload.user_id,
            trigger_text=payload.trigger_text,
            trigger_type=payload.trigger_type,
            tool_name=payload.tool_name,
            tool_arguments=payload.tool_arguments,
            active=payload.active,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SkillUpsertResponse(
        status="ok",
        skill=_to_skill_item(created),
    )


@router.delete("/skill/{skill_id}", response_model=SkillDeleteResponse)
async def delete_skill(
    skill_id: str = Path(..., min_length=1),
    user_id: str | None = Query(default=None),
    skill_interface: SkillInterface = Depends(get_skill_interface),
) -> SkillDeleteResponse:
    deleted = await skill_interface.delete_skill(skill_id=skill_id, user_id=user_id)
    return SkillDeleteResponse(
        status="deleted" if deleted else "not_found",
        deleted=deleted,
        skill_id=skill_id,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status(
    brain: BrainInterface = Depends(get_brain),
    exporter: TelemetryExporter = Depends(get_telemetry_exporter),
    skill_interface: SkillInterface = Depends(get_skill_interface),
) -> StatusResponse:
    settings = get_settings()
    memory_ready = await brain.is_memory_ready()
    profile_ready = await brain.is_profile_engine_ready()
    tool_ready = await brain.is_tool_engine_ready()
    goal_ready = await brain.is_goal_engine_ready()
    skill_ready = await skill_interface.is_ready()
    memory_entries_count = await brain.memory_entries_count() if memory_ready else 0
    snapshot = exporter.snapshot()
    telemetry = TelemetrySnapshotResponse(
        request_count=snapshot.request_count,
        latency_avg_ms=snapshot.latency_avg_ms,
        model_latency_avg_ms=snapshot.model_latency_avg_ms,
        local_model_latency_avg_ms=snapshot.local_model_latency_avg_ms,
        fallback_rate=snapshot.fallback_rate,
        coherence_score_avg=snapshot.coherence_score_avg,
        regeneration_rate=snapshot.regeneration_rate,
        guardrail_trigger_count=snapshot.guardrail_trigger_count,
        tool_usage_count=snapshot.tool_usage_count,
        vagueness_flag_count=snapshot.vagueness_flag_count,
        topic_misalignment_rate=snapshot.topic_misalignment_rate,
        telemetry_enabled=snapshot.telemetry_enabled,
        dropped_event_count=snapshot.dropped_event_count,
    )
    return StatusResponse(
        status="ok" if (brain.is_model_ready() and memory_ready and skill_ready) else "degraded",
        model_ready=brain.is_model_ready(),
        memory_ready=memory_ready,
        profile_engine_ready=profile_ready,
        tool_engine_ready=tool_ready,
        goal_engine_ready=goal_ready,
        skill_engine_ready=skill_ready,
        telemetry_enabled=settings.telemetry_enabled,
        uptime_seconds=int(time.monotonic() - _SYSTEM_STARTED_MONOTONIC),
        memory_entries_count=memory_entries_count,
        telemetry=telemetry,
    )


def _to_skill_item(skill: SkillDefinition) -> SkillItemResponse:
    return SkillItemResponse(
        skill_id=skill.skill_id,
        user_id=skill.user_id,
        trigger_text=skill.trigger_text,
        trigger_type=skill.trigger_type,
        tool_name=skill.tool_name,
        tool_arguments=dict(skill.tool_arguments),
        correction_count=skill.correction_count,
        source=skill.source,
        active=skill.active,
        created_at=skill.created_at,
        updated_at=skill.updated_at,
    )


@router.get("/analyzer", response_model=AnalyzerResponse)
async def get_analyzer(
    limit: int = Query(60, ge=1, le=500),
    analyzer: RuntimeAnalyzer = Depends(get_runtime_analyzer),
) -> AnalyzerResponse:
    records = await analyzer.tail(limit=limit)
    entry_count = await analyzer.entry_count()
    return AnalyzerResponse(
        status="ok" if analyzer.is_enabled else "disabled",
        analyzer_enabled=analyzer.is_enabled,
        analyzer_running=analyzer.is_running,
        file_path=analyzer.file_path,
        entry_count=entry_count,
        records=[record for record in records if isinstance(record, dict)],
    )
