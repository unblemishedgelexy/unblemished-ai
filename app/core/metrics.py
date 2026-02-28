from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class MetricsSnapshot:
    stage_breakdown_ms: dict[str, int]
    error_categories: dict[str, int]
    total_ms: int


class MetricsRecorder:
    def __init__(self) -> None:
        self._active_stages: dict[str, float] = {}
        self._stage_breakdown_ms: dict[str, int] = {}
        self._error_categories: dict[str, int] = {}
        self._started_at = monotonic()

    def start_stage(self, stage_name: str) -> None:
        self._active_stages[stage_name] = monotonic()

    def end_stage(self, stage_name: str) -> int:
        started = self._active_stages.pop(stage_name, None)
        if started is None:
            return 0
        duration_ms = max(int((monotonic() - started) * 1000), 0)
        self._stage_breakdown_ms[stage_name] = duration_ms
        return duration_ms

    def record_stage_duration(self, stage_name: str, duration_ms: int) -> None:
        self._stage_breakdown_ms[stage_name] = max(duration_ms, 0)

    def record_error(self, category: str) -> None:
        self._error_categories[category] = self._error_categories.get(category, 0) + 1

    def snapshot(self, total_ms: int | None = None) -> MetricsSnapshot:
        computed_total = total_ms
        if computed_total is None:
            computed_total = max(int((monotonic() - self._started_at) * 1000), 0)
        return MetricsSnapshot(
            stage_breakdown_ms=dict(self._stage_breakdown_ms),
            error_categories=dict(self._error_categories),
            total_ms=max(computed_total, 0),
        )

