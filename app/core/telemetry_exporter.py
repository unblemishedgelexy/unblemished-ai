from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from app.core.logger import StructuredLogger


@dataclass(slots=True, frozen=True)
class TelemetryEvent:
    latency_ms: int
    model_latency_ms: int
    local_model_latency_ms: int
    fallback_triggered: bool
    guardrail_triggered: bool
    tool_usage_count: int
    coherence_score: float
    regenerated: bool
    vagueness_flag: bool
    topic_misaligned: bool


@dataclass(slots=True)
class TelemetrySnapshot:
    request_count: int
    latency_avg_ms: float
    model_latency_avg_ms: float
    local_model_latency_avg_ms: float
    fallback_rate: float
    coherence_score_avg: float
    regeneration_rate: float
    guardrail_trigger_count: int
    tool_usage_count: int
    vagueness_flag_count: int
    topic_misalignment_rate: float
    telemetry_enabled: bool
    dropped_event_count: int


class TelemetryExporter:
    """
    Non-blocking telemetry aggregation with optional OpenTelemetry metric emitters.
    Export surface:
    - request_count
    - latency
    - model_latency
    - local_model_latency
    - fallback_rate
    - coherence_score_avg
    - regeneration_rate
    - guardrail_trigger_count
    - tool_usage_count
    - vagueness_flag_count
    - topic_misalignment_rate
    """

    def __init__(
        self,
        *,
        logger: StructuredLogger,
        enabled: bool,
        queue_size: int = 4096,
    ) -> None:
        self._logger = logger
        self._enabled = enabled
        self._queue_size = max(queue_size, 64)
        self._queue: asyncio.Queue[TelemetryEvent | None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._started = False

        self._lock = threading.Lock()
        self._request_count = 0
        self._latency_ms_sum = 0
        self._model_latency_ms_sum = 0
        self._local_model_latency_ms_sum = 0
        self._local_model_request_count = 0
        self._fallback_count = 0
        self._guardrail_trigger_count = 0
        self._tool_usage_count = 0
        self._coherence_score_sum = 0.0
        self._regeneration_count = 0
        self._vagueness_flag_count = 0
        self._topic_misalignment_count = 0
        self._dropped_event_count = 0

        self._otel_ready = False
        self._otel_request_counter: Any | None = None
        self._otel_latency_histogram: Any | None = None
        self._otel_model_latency_histogram: Any | None = None
        self._otel_local_model_latency_histogram: Any | None = None
        self._otel_fallback_rate_histogram: Any | None = None
        self._otel_guardrail_counter: Any | None = None
        self._otel_tool_usage_counter: Any | None = None
        self._otel_coherence_histogram: Any | None = None
        self._otel_regeneration_rate_histogram: Any | None = None
        self._otel_vagueness_counter: Any | None = None
        self._otel_topic_misalignment_rate_histogram: Any | None = None

    async def start(self) -> None:
        if not self._enabled or self._started:
            return
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._worker_task = asyncio.create_task(self._run_worker())
        self._started = True
        self._setup_otel_instruments()
        self._logger.info(
            "telemetry.exporter.started",
            trace_id="system",
            user_id="system",
            memory_id="n/a",
            retrieval_count=0,
            telemetry_enabled=self._enabled,
            otel_ready=self._otel_ready,
            queue_size=self._queue_size,
        )

    async def shutdown(self) -> None:
        if not self._enabled:
            return
        if self._queue is None or self._worker_task is None:
            return
        if self._worker_task.done():
            return

        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            await self._queue.put(None)

        await asyncio.wait({self._worker_task}, timeout=2.0)

    def record_request(
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
        if not self._enabled:
            return

        event = TelemetryEvent(
            latency_ms=max(int(latency_ms), 0),
            model_latency_ms=max(int(model_latency_ms), 0),
            local_model_latency_ms=max(int(local_model_latency_ms), 0),
            fallback_triggered=bool(fallback_triggered),
            guardrail_triggered=bool(guardrail_triggered),
            tool_usage_count=max(int(tool_usage_count), 0),
            coherence_score=max(min(float(coherence_score), 1.0), 0.0),
            regenerated=bool(regenerated),
            vagueness_flag=bool(vagueness_flag),
            topic_misaligned=bool(topic_misaligned),
        )

        if not self._started or self._queue is None:
            # Best-effort lazy start in active loop, while keeping call non-blocking.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.start())
            except RuntimeError:
                pass
            self._apply_event(event)
            return

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            with self._lock:
                self._dropped_event_count += 1
            self._logger.warning(
                "telemetry.exporter.queue_full",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
            )

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            request_count = self._request_count
            latency_avg_ms = (self._latency_ms_sum / request_count) if request_count else 0.0
            model_latency_avg_ms = (self._model_latency_ms_sum / request_count) if request_count else 0.0
            local_model_latency_avg_ms = (
                self._local_model_latency_ms_sum / self._local_model_request_count
            ) if self._local_model_request_count else 0.0
            fallback_rate = (self._fallback_count / request_count) if request_count else 0.0
            coherence_score_avg = (self._coherence_score_sum / request_count) if request_count else 0.0
            regeneration_rate = (self._regeneration_count / request_count) if request_count else 0.0
            topic_misalignment_rate = (self._topic_misalignment_count / request_count) if request_count else 0.0
            return TelemetrySnapshot(
                request_count=request_count,
                latency_avg_ms=latency_avg_ms,
                model_latency_avg_ms=model_latency_avg_ms,
                local_model_latency_avg_ms=local_model_latency_avg_ms,
                fallback_rate=fallback_rate,
                coherence_score_avg=coherence_score_avg,
                regeneration_rate=regeneration_rate,
                guardrail_trigger_count=self._guardrail_trigger_count,
                tool_usage_count=self._tool_usage_count,
                vagueness_flag_count=self._vagueness_flag_count,
                topic_misalignment_rate=topic_misalignment_rate,
                telemetry_enabled=self._enabled,
                dropped_event_count=self._dropped_event_count,
            )

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP request_count Total processed request count.",
            "# TYPE request_count counter",
            f"request_count {snapshot.request_count}",
            "# HELP latency Average end-to-end latency in milliseconds.",
            "# TYPE latency gauge",
            f"latency {snapshot.latency_avg_ms:.6f}",
            "# HELP model_latency Average model latency in milliseconds.",
            "# TYPE model_latency gauge",
            f"model_latency {snapshot.model_latency_avg_ms:.6f}",
            "# HELP local_model_latency Average local backend model latency in milliseconds.",
            "# TYPE local_model_latency gauge",
            f"local_model_latency {snapshot.local_model_latency_avg_ms:.6f}",
            "# HELP fallback_rate Fraction of requests that triggered fallback.",
            "# TYPE fallback_rate gauge",
            f"fallback_rate {snapshot.fallback_rate:.6f}",
            "# HELP coherence_score_avg Average human coherence score.",
            "# TYPE coherence_score_avg gauge",
            f"coherence_score_avg {snapshot.coherence_score_avg:.6f}",
            "# HELP regeneration_rate Fraction of requests that required one regeneration.",
            "# TYPE regeneration_rate gauge",
            f"regeneration_rate {snapshot.regeneration_rate:.6f}",
            "# HELP guardrail_trigger_count Total guardrail trigger count.",
            "# TYPE guardrail_trigger_count counter",
            f"guardrail_trigger_count {snapshot.guardrail_trigger_count}",
            "# HELP tool_usage_count Total tool invocation count.",
            "# TYPE tool_usage_count counter",
            f"tool_usage_count {snapshot.tool_usage_count}",
            "# HELP vagueness_flag_count Total vagueness flag count.",
            "# TYPE vagueness_flag_count counter",
            f"vagueness_flag_count {snapshot.vagueness_flag_count}",
            "# HELP topic_misalignment_rate Fraction of requests with topic misalignment.",
            "# TYPE topic_misalignment_rate gauge",
            f"topic_misalignment_rate {snapshot.topic_misalignment_rate:.6f}",
            "# HELP telemetry_enabled Telemetry subsystem enabled flag.",
            "# TYPE telemetry_enabled gauge",
            f"telemetry_enabled {1 if snapshot.telemetry_enabled else 0}",
            "# HELP telemetry_dropped_event_count Telemetry events dropped due to backpressure.",
            "# TYPE telemetry_dropped_event_count counter",
            f"telemetry_dropped_event_count {snapshot.dropped_event_count}",
        ]
        return "\n".join(lines) + "\n"

    async def _run_worker(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            if event is None:
                self._queue.task_done()
                break
            self._apply_event(event)
            self._queue.task_done()

    def _apply_event(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._request_count += 1
            self._latency_ms_sum += event.latency_ms
            self._model_latency_ms_sum += event.model_latency_ms
            if event.local_model_latency_ms > 0:
                self._local_model_latency_ms_sum += event.local_model_latency_ms
                self._local_model_request_count += 1
            if event.fallback_triggered:
                self._fallback_count += 1
            if event.guardrail_triggered:
                self._guardrail_trigger_count += 1
            self._tool_usage_count += event.tool_usage_count
            self._coherence_score_sum += event.coherence_score
            if event.regenerated:
                self._regeneration_count += 1
            if event.vagueness_flag:
                self._vagueness_flag_count += 1
            if event.topic_misaligned:
                self._topic_misalignment_count += 1

            request_count = self._request_count
            fallback_rate = (self._fallback_count / request_count) if request_count else 0.0
            regeneration_rate = (self._regeneration_count / request_count) if request_count else 0.0
            topic_misalignment_rate = (
                self._topic_misalignment_count / request_count
            ) if request_count else 0.0

        self._emit_to_otel(
            event=event,
            fallback_rate=fallback_rate,
            regeneration_rate=regeneration_rate,
            topic_misalignment_rate=topic_misalignment_rate,
        )

    def _setup_otel_instruments(self) -> None:
        try:
            from opentelemetry import metrics
        except Exception:
            self._otel_ready = False
            return

        try:
            meter = metrics.get_meter("humoniod.telemetry.exporter")
            self._otel_request_counter = meter.create_counter(
                "request_count",
                unit="1",
                description="Total processed request count",
            )
            self._otel_latency_histogram = meter.create_histogram(
                "latency",
                unit="ms",
                description="Request latency in milliseconds",
            )
            self._otel_model_latency_histogram = meter.create_histogram(
                "model_latency",
                unit="ms",
                description="Model latency in milliseconds",
            )
            self._otel_local_model_latency_histogram = meter.create_histogram(
                "local_model_latency",
                unit="ms",
                description="Local backend model latency in milliseconds",
            )
            self._otel_fallback_rate_histogram = meter.create_histogram(
                "fallback_rate",
                unit="1",
                description="Fallback rate",
            )
            self._otel_guardrail_counter = meter.create_counter(
                "guardrail_trigger_count",
                unit="1",
                description="Guardrail trigger count",
            )
            self._otel_tool_usage_counter = meter.create_counter(
                "tool_usage_count",
                unit="1",
                description="Tool usage count",
            )
            self._otel_coherence_histogram = meter.create_histogram(
                "coherence_score_avg",
                unit="1",
                description="Human coherence score",
            )
            self._otel_regeneration_rate_histogram = meter.create_histogram(
                "regeneration_rate",
                unit="1",
                description="Regeneration rate",
            )
            self._otel_vagueness_counter = meter.create_counter(
                "vagueness_flag_count",
                unit="1",
                description="Vagueness flag count",
            )
            self._otel_topic_misalignment_rate_histogram = meter.create_histogram(
                "topic_misalignment_rate",
                unit="1",
                description="Topic misalignment rate",
            )
            self._otel_ready = True
        except Exception:
            self._otel_ready = False

    def _emit_to_otel(
        self,
        *,
        event: TelemetryEvent,
        fallback_rate: float,
        regeneration_rate: float,
        topic_misalignment_rate: float,
    ) -> None:
        if not self._otel_ready:
            return
        try:
            if self._otel_request_counter is not None:
                self._otel_request_counter.add(1)
            if self._otel_latency_histogram is not None:
                self._otel_latency_histogram.record(event.latency_ms)
            if self._otel_model_latency_histogram is not None:
                self._otel_model_latency_histogram.record(event.model_latency_ms)
            if self._otel_local_model_latency_histogram is not None and event.local_model_latency_ms > 0:
                self._otel_local_model_latency_histogram.record(event.local_model_latency_ms)
            if self._otel_fallback_rate_histogram is not None:
                self._otel_fallback_rate_histogram.record(fallback_rate)
            if event.guardrail_triggered and self._otel_guardrail_counter is not None:
                self._otel_guardrail_counter.add(1)
            if event.tool_usage_count > 0 and self._otel_tool_usage_counter is not None:
                self._otel_tool_usage_counter.add(event.tool_usage_count)
            if self._otel_coherence_histogram is not None:
                self._otel_coherence_histogram.record(event.coherence_score)
            if self._otel_regeneration_rate_histogram is not None:
                self._otel_regeneration_rate_histogram.record(regeneration_rate)
            if event.vagueness_flag and self._otel_vagueness_counter is not None:
                self._otel_vagueness_counter.add(1)
            if self._otel_topic_misalignment_rate_histogram is not None:
                self._otel_topic_misalignment_rate_histogram.record(topic_misalignment_rate)
        except Exception:
            # Export failures should never affect request execution.
            return
