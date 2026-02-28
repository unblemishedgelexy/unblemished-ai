from __future__ import annotations

from app.core.metrics import MetricsRecorder, MetricsSnapshot
from app.core.telemetry_exporter import TelemetryExporter


class TelemetryEngine:
    def __init__(self, exporter: TelemetryExporter | None = None) -> None:
        self._exporter = exporter

    def create_recorder(self) -> MetricsRecorder:
        return MetricsRecorder()

    def record_request_metrics(
        self,
        *,
        latency_ms: int,
        model_latency_ms: int,
        fallback_triggered: bool,
        guardrail_triggered: bool,
        tool_usage_count: int,
        local_model_latency_ms: int = 0,
        coherence_score: float = 1.0,
        regenerated: bool = False,
        vagueness_flag: bool = False,
        topic_misaligned: bool = False,
    ) -> None:
        if self._exporter is None:
            return
        self._exporter.record_request(
            latency_ms=latency_ms,
            model_latency_ms=model_latency_ms,
            local_model_latency_ms=local_model_latency_ms,
            fallback_triggered=fallback_triggered,
            guardrail_triggered=guardrail_triggered,
            tool_usage_count=tool_usage_count,
            coherence_score=coherence_score,
            regenerated=regenerated,
            vagueness_flag=vagueness_flag,
            topic_misaligned=topic_misaligned,
        )

    def build_cost_estimate(
        self,
        *,
        prompt: str,
        final_answer: str,
        routed_model: str,
        metrics_snapshot: MetricsSnapshot,
    ) -> dict[str, object]:
        token_count = len(prompt.split()) + len(final_answer.split())
        token_price = {
            "fast_model": 0.0000025,
            "creative_model": 0.0000038,
            "deep_model": 0.0000052,
        }.get(routed_model, 0.0000035)
        infer_rate = {
            "fast_model": 0.000018,
            "creative_model": 0.000026,
            "deep_model": 0.000037,
        }.get(routed_model, 0.000025)

        model_ms = max(metrics_snapshot.stage_breakdown_ms.get("model_ms", 0), 0)
        token_cost = token_count * token_price
        return {
            "estimated_token_cost": round(token_cost, 6),
            "estimated_inference_cost": round(token_cost + (model_ms / 1000.0) * infer_rate, 6),
            "total_latency_breakdown": {
                key: max(int(value), 0) for key, value in metrics_snapshot.stage_breakdown_ms.items()
            },
            "routed_model": routed_model,
        }
