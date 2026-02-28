from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings
from app.core.logger import StructuredLogger
from app.core.telemetry_exporter import TelemetryExporter
from app.services.brain.brain_interface import BrainInterface
from app.services.skills.skill_interface import SkillInterface
from app.utils.helpers import utc_now


@dataclass(slots=True)
class AnalyzerSnapshot:
    status: str
    model_ready: bool
    memory_ready: bool
    skill_engine_ready: bool
    tool_engine_ready: bool
    memory_entries_count: int
    request_count: int
    tool_usage_count: int
    fallback_rate: float
    latency_avg_ms: float
    dropped_event_count: int
    reasoning_mode: str
    backend: str


class RuntimeAnalyzer:
    def __init__(
        self,
        *,
        logger: StructuredLogger,
        enabled: bool,
        file_path: str,
        poll_interval_seconds: float,
        request_delta_threshold: int,
        latency_delta_ms: int,
        max_file_bytes: int,
        max_records: int,
        settings_provider: Callable[[], Settings],
        brain_provider: Callable[[], BrainInterface],
        telemetry_exporter_provider: Callable[[], TelemetryExporter],
        skill_interface_provider: Callable[[], SkillInterface],
    ) -> None:
        self._logger = logger
        self._enabled = enabled
        self._file_path = Path(file_path)
        self._poll_interval_seconds = max(0.25, float(poll_interval_seconds))
        self._request_delta_threshold = max(1, int(request_delta_threshold))
        self._latency_delta_ms = max(1, int(latency_delta_ms))
        self._max_file_bytes = max(10240, int(max_file_bytes))
        self._max_records = max(20, int(max_records))
        self._settings_provider = settings_provider
        self._brain_provider = brain_provider
        self._telemetry_exporter_provider = telemetry_exporter_provider
        self._skill_interface_provider = skill_interface_provider

        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._last_recorded_snapshot: AnalyzerSnapshot | None = None
        self._entry_count = 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def file_path(self) -> str:
        return str(self._file_path)

    async def start(self) -> None:
        if not self._enabled:
            return
        if self.is_running:
            return

        await asyncio.to_thread(self._ensure_parent_dir_sync)
        self._entry_count = await asyncio.to_thread(self._count_entries_sync)
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop())
        self._logger.info(
            "runtime.analyzer.started",
            trace_id="system",
            user_id="system",
            memory_id="n/a",
            retrieval_count=0,
            file_path=str(self._file_path),
            poll_interval_seconds=self._poll_interval_seconds,
            max_file_bytes=self._max_file_bytes,
            max_records=self._max_records,
        )

    async def shutdown(self) -> None:
        if self._task is None:
            return
        if self._task.done():
            return

        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=3.0)
        except asyncio.TimeoutError:
            self._task.cancel()
        self._logger.info(
            "runtime.analyzer.stopped",
            trace_id="system",
            user_id="system",
            memory_id="n/a",
            retrieval_count=0,
            file_path=str(self._file_path),
            entry_count=self._entry_count,
        )

    async def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 2000))
        return await asyncio.to_thread(self._tail_sync, bounded)

    async def entry_count(self) -> int:
        self._entry_count = await asyncio.to_thread(self._count_entries_sync)
        return self._entry_count

    async def _run_loop(self) -> None:
        await self._capture_and_maybe_record(force_reason="startup")
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval_seconds)
            except asyncio.TimeoutError:
                pass
            if self._stop_event.is_set():
                break
            await self._capture_and_maybe_record(force_reason=None)

    async def _capture_and_maybe_record(self, force_reason: str | None) -> None:
        try:
            current = await self._build_snapshot()
        except Exception as exc:
            self._logger.error(
                "runtime.analyzer.snapshot_failed",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return

        reasons = self._diff_reasons(previous=self._last_recorded_snapshot, current=current)
        if force_reason:
            reasons.insert(0, force_reason)
        if not reasons:
            return

        payload = {
            "event": "runtime_analyzer.state_change",
            "timestamp": utc_now().isoformat(),
            "reasons": reasons,
            "status": current.status,
            "model_ready": current.model_ready,
            "memory_ready": current.memory_ready,
            "skill_engine_ready": current.skill_engine_ready,
            "tool_engine_ready": current.tool_engine_ready,
            "memory_entries_count": current.memory_entries_count,
            "request_count": current.request_count,
            "tool_usage_count": current.tool_usage_count,
            "fallback_rate": round(current.fallback_rate, 4),
            "latency_avg_ms": round(current.latency_avg_ms, 2),
            "dropped_event_count": current.dropped_event_count,
            "reasoning_mode": current.reasoning_mode,
            "backend": current.backend,
        }

        try:
            self._entry_count = await asyncio.to_thread(self._append_record_sync, payload)
            self._last_recorded_snapshot = current
        except Exception as exc:
            self._logger.error(
                "runtime.analyzer.write_failed",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
                file_path=str(self._file_path),
            )

    async def _build_snapshot(self) -> AnalyzerSnapshot:
        settings = self._settings_provider()
        brain = self._brain_provider()
        exporter = self._telemetry_exporter_provider()
        skill_interface = self._skill_interface_provider()

        memory_ready = await brain.is_memory_ready()
        skill_ready = await skill_interface.is_ready()
        tool_ready = await brain.is_tool_engine_ready()
        memory_entries_count = await brain.memory_entries_count() if memory_ready else 0
        telemetry = exporter.snapshot()

        model_ready = brain.is_model_ready()
        status = "ok" if (model_ready and memory_ready and skill_ready) else "degraded"
        return AnalyzerSnapshot(
            status=status,
            model_ready=model_ready,
            memory_ready=memory_ready,
            skill_engine_ready=skill_ready,
            tool_engine_ready=tool_ready,
            memory_entries_count=memory_entries_count,
            request_count=telemetry.request_count,
            tool_usage_count=telemetry.tool_usage_count,
            fallback_rate=telemetry.fallback_rate,
            latency_avg_ms=telemetry.latency_avg_ms,
            dropped_event_count=telemetry.dropped_event_count,
            reasoning_mode=settings.reasoning_mode,
            backend=settings.model_backend,
        )

    def _diff_reasons(self, *, previous: AnalyzerSnapshot | None, current: AnalyzerSnapshot) -> list[str]:
        if previous is None:
            return ["initial_snapshot"]

        reasons: list[str] = []
        if current.status != previous.status:
            reasons.append("status_changed")
        if current.model_ready != previous.model_ready:
            reasons.append("model_readiness_changed")
        if current.memory_ready != previous.memory_ready:
            reasons.append("memory_readiness_changed")
        if current.skill_engine_ready != previous.skill_engine_ready:
            reasons.append("skill_engine_readiness_changed")
        if current.tool_engine_ready != previous.tool_engine_ready:
            reasons.append("tool_engine_readiness_changed")
        if current.reasoning_mode != previous.reasoning_mode:
            reasons.append("reasoning_mode_changed")
        if current.backend != previous.backend:
            reasons.append("backend_changed")
        if current.memory_entries_count != previous.memory_entries_count:
            reasons.append("memory_entries_changed")
        if (current.request_count - previous.request_count) >= self._request_delta_threshold:
            reasons.append("request_volume_delta")
        if current.tool_usage_count != previous.tool_usage_count:
            reasons.append("tool_usage_changed")
        if abs(current.latency_avg_ms - previous.latency_avg_ms) >= float(self._latency_delta_ms):
            reasons.append("latency_shift")
        if abs(current.fallback_rate - previous.fallback_rate) >= 0.05:
            reasons.append("fallback_rate_shift")
        if current.dropped_event_count != previous.dropped_event_count:
            reasons.append("telemetry_drop_detected")
        return reasons

    def _ensure_parent_dir_sync(self) -> None:
        parent = self._file_path.parent
        parent.mkdir(parents=True, exist_ok=True)

    def _count_entries_sync(self) -> int:
        if not self._file_path.exists():
            return 0
        with self._file_path.open("r", encoding="utf-8") as file_obj:
            return sum(1 for _ in file_obj)

    def _tail_sync(self, limit: int) -> list[dict[str, Any]]:
        if not self._file_path.exists():
            return []
        with self._file_path.open("r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()
        records: list[dict[str, Any]] = []
        for raw_line in lines[-limit:]:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _append_record_sync(self, payload: dict[str, Any]) -> int:
        self._ensure_parent_dir_sync()
        line = json.dumps(payload, default=str, separators=(",", ":"))
        with self._file_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(line)
            file_obj.write("\n")

        with self._file_path.open("r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()

        if len(lines) > self._max_records or self._file_path.stat().st_size > self._max_file_bytes:
            lines = self._trim_lines(lines)
            with self._file_path.open("w", encoding="utf-8") as file_obj:
                file_obj.writelines(lines)
        return len(lines)

    def _trim_lines(self, lines: list[str]) -> list[str]:
        selected: list[str] = []
        total_size = 0

        for raw_line in reversed(lines):
            line_size = len(raw_line.encode("utf-8"))
            if len(selected) >= self._max_records:
                break
            if selected and (total_size + line_size) > self._max_file_bytes:
                break
            selected.append(raw_line)
            total_size += line_size

        selected.reverse()
        return selected
