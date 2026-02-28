from __future__ import annotations

import asyncio

from app.core.logger import StructuredLogger, setup_logger
from app.core.telemetry_exporter import TelemetryExporter
from app.main import create_app


def test_telemetry_exporter_aggregates_prometheus_metrics() -> None:
    async def _exercise() -> str:
        exporter = TelemetryExporter(
            logger=StructuredLogger(setup_logger(level="INFO")),
            enabled=True,
            queue_size=32,
        )
        await exporter.start()
        exporter.record_request(
            latency_ms=100,
            model_latency_ms=70,
            fallback_triggered=False,
            guardrail_triggered=False,
            tool_usage_count=1,
            local_model_latency_ms=80,
            coherence_score=0.8,
            regenerated=False,
            vagueness_flag=False,
            topic_misaligned=False,
        )
        exporter.record_request(
            latency_ms=300,
            model_latency_ms=130,
            fallback_triggered=True,
            guardrail_triggered=True,
            tool_usage_count=2,
            local_model_latency_ms=0,
            coherence_score=0.4,
            regenerated=True,
            vagueness_flag=True,
            topic_misaligned=True,
        )
        await asyncio.sleep(0)
        await exporter.shutdown()
        return exporter.render_prometheus()

    output = asyncio.run(_exercise())
    assert "request_count 2" in output
    assert "latency 200.000000" in output
    assert "model_latency 100.000000" in output
    assert "local_model_latency 80.000000" in output
    assert "fallback_rate 0.500000" in output
    assert "coherence_score_avg 0.600000" in output
    assert "regeneration_rate 0.500000" in output
    assert "guardrail_trigger_count 1" in output
    assert "tool_usage_count 3" in output
    assert "vagueness_flag_count 1" in output
    assert "topic_misalignment_rate 0.500000" in output


def test_metrics_route_registered() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/" in paths
    assert "/metrics" in paths
    assert "/v1/system/runtime" in paths
    assert "/v1/system/memory" in paths
    assert "/v1/system/skills" in paths
    assert "/v1/system/skill" in paths
    assert "/v1/system/skill/{skill_id}" in paths
    assert "/v1/system/status" in paths
    assert "/v1/system/analyzer" in paths
    assert "/v1/system/dashboard" in paths
    assert "/analyzer/health" in paths
