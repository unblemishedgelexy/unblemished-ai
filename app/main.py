from __future__ import annotations

import time

from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.core.auth_middleware import AuthRateLimitMiddleware
from app.core.config import get_settings
from app.core.dependencies import get_brain, get_runtime_analyzer, get_task_manager, get_telemetry_exporter
from app.core.logger import StructuredLogger, setup_logger
from app.core.runtime_analyzer import RuntimeAnalyzer
from app.core.telemetry_exporter import TelemetryExporter
from app.routes.chat import router as chat_router
from app.routes.system import router as system_router
from app.services.brain.brain_interface import BrainInterface

_STARTED_MONOTONIC = time.monotonic()


def create_app() -> FastAPI:
    settings = get_settings()
    logger = StructuredLogger(setup_logger(level=settings.log_level))

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Phase 5 Text Brain Engine with Agent Mode tool and goal orchestration",
    )
    app.add_middleware(AuthRateLimitMiddleware, settings=settings, logger=logger)
    app.include_router(chat_router, prefix="/v1")
    app.include_router(system_router, prefix="/v1")

    @app.on_event("startup")
    async def on_startup() -> None:
        await get_telemetry_exporter().start()
        await get_runtime_analyzer().start()

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await get_runtime_analyzer().shutdown()
        await get_telemetry_exporter().shutdown()
        await get_task_manager().graceful_shutdown()

    @app.get("/health", tags=["system"])
    async def health(
        brain: BrainInterface = Depends(get_brain),
    ) -> dict[str, object]:
        runtime_settings = get_settings()
        memory_ready = await brain.is_memory_ready()
        profile_ready = await brain.is_profile_engine_ready()
        tool_ready = await brain.is_tool_engine_ready()
        goal_ready = await brain.is_goal_engine_ready()
        memory_entries_count = await brain.memory_entries_count() if memory_ready else 0
        return {
            "status": "ok",
            "reasoning_mode": runtime_settings.reasoning_mode,
            "model_ready": brain.is_model_ready(),
            "memory_ready": memory_ready,
            "embedding_enabled": runtime_settings.embedding_enabled,
            "profile_engine_ready": profile_ready,
            "model_router_enabled": runtime_settings.model_routing_enabled,
            "self_evaluation_enabled": runtime_settings.self_evaluation_enabled,
            "tool_engine_ready": tool_ready,
            "goal_engine_ready": goal_ready,
            "memory_entries_count": memory_entries_count,
            "uptime_seconds": int(time.monotonic() - _STARTED_MONOTONIC),
        }

    @app.get("/metrics", tags=["system"], include_in_schema=False)
    async def metrics(
        exporter: TelemetryExporter = Depends(get_telemetry_exporter),
    ) -> PlainTextResponse:
        return PlainTextResponse(
            exporter.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/analyzer/health", tags=["system"], include_in_schema=False)
    async def analyzer_health(
        analyzer: RuntimeAnalyzer = Depends(get_runtime_analyzer),
    ) -> dict[str, object]:
        return {
            "enabled": analyzer.is_enabled,
            "running": analyzer.is_running,
            "file_path": analyzer.file_path,
            "entry_count": await analyzer.entry_count(),
        }

    @app.get("/", include_in_schema=False)
    async def root_dashboard() -> RedirectResponse:
        return RedirectResponse(url="/v1/system/dashboard", status_code=307)

    return app


app = create_app()
