from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import Settings
from app.core.logger import StructuredLogger, setup_logger
from app.core.runtime_analyzer import RuntimeAnalyzer
from app.core.telemetry_exporter import TelemetrySnapshot


class _FakeBrain:
    def __init__(self) -> None:
        self.model_ready = True
        self.memory_ready = True
        self.tool_ready = True
        self.memory_entries = 0

    def is_model_ready(self) -> bool:
        return self.model_ready

    async def is_memory_ready(self) -> bool:
        return self.memory_ready

    async def memory_entries_count(self) -> int:
        return self.memory_entries

    async def is_tool_engine_ready(self) -> bool:
        return self.tool_ready


class _FakeSkillInterface:
    def __init__(self) -> None:
        self.ready = True

    async def is_ready(self) -> bool:
        return self.ready


class _FakeTelemetryExporter:
    def __init__(self) -> None:
        self._snapshot = TelemetrySnapshot(
            request_count=0,
            latency_avg_ms=0.0,
            model_latency_avg_ms=0.0,
            local_model_latency_avg_ms=0.0,
            fallback_rate=0.0,
            coherence_score_avg=1.0,
            regeneration_rate=0.0,
            guardrail_trigger_count=0,
            tool_usage_count=0,
            vagueness_flag_count=0,
            topic_misalignment_rate=0.0,
            telemetry_enabled=True,
            dropped_event_count=0,
        )

    def snapshot(self) -> TelemetrySnapshot:
        return self._snapshot


def test_runtime_analyzer_records_only_on_delta(tmp_path: Path) -> None:
    async def run() -> None:
        file_path = tmp_path / "analyzer.jsonl"
        brain = _FakeBrain()
        skill = _FakeSkillInterface()
        telemetry = _FakeTelemetryExporter()
        settings = Settings(memory_db_path=str(tmp_path / "memory.db"))

        analyzer = RuntimeAnalyzer(
            logger=StructuredLogger(setup_logger(level="INFO")),
            enabled=True,
            file_path=str(file_path),
            poll_interval_seconds=1.0,
            request_delta_threshold=3,
            latency_delta_ms=40,
            max_file_bytes=204800,
            max_records=500,
            settings_provider=lambda: settings,
            brain_provider=lambda: brain,  # type: ignore[arg-type]
            telemetry_exporter_provider=lambda: telemetry,  # type: ignore[arg-type]
            skill_interface_provider=lambda: skill,  # type: ignore[arg-type]
        )

        await analyzer._capture_and_maybe_record(force_reason="startup")  # noqa: SLF001
        assert await analyzer.entry_count() == 1

        telemetry._snapshot.request_count = 1  # noqa: SLF001
        await analyzer._capture_and_maybe_record(force_reason=None)  # noqa: SLF001
        assert await analyzer.entry_count() == 1

        telemetry._snapshot.request_count = 4  # noqa: SLF001
        await analyzer._capture_and_maybe_record(force_reason=None)  # noqa: SLF001
        assert await analyzer.entry_count() == 2

        tail = await analyzer.tail(limit=5)
        assert tail[-1]["event"] == "runtime_analyzer.state_change"
        assert "request_volume_delta" in tail[-1].get("reasons", [])

    asyncio.run(run())


def test_runtime_analyzer_compacts_file(tmp_path: Path) -> None:
    async def run() -> None:
        file_path = tmp_path / "analyzer_compact.jsonl"
        brain = _FakeBrain()
        skill = _FakeSkillInterface()
        telemetry = _FakeTelemetryExporter()
        settings = Settings(memory_db_path=str(tmp_path / "memory.db"))

        analyzer = RuntimeAnalyzer(
            logger=StructuredLogger(setup_logger(level="INFO")),
            enabled=True,
            file_path=str(file_path),
            poll_interval_seconds=1.0,
            request_delta_threshold=1,
            latency_delta_ms=1,
            max_file_bytes=10240,
            max_records=20,
            settings_provider=lambda: settings,
            brain_provider=lambda: brain,  # type: ignore[arg-type]
            telemetry_exporter_provider=lambda: telemetry,  # type: ignore[arg-type]
            skill_interface_provider=lambda: skill,  # type: ignore[arg-type]
        )

        await analyzer._capture_and_maybe_record(force_reason="startup")  # noqa: SLF001
        for index in range(60):
            telemetry._snapshot.request_count = index + 1  # noqa: SLF001
            telemetry._snapshot.tool_usage_count = index  # noqa: SLF001
            brain.memory_entries = index
            await analyzer._capture_and_maybe_record(force_reason=None)  # noqa: SLF001

        entry_count = await analyzer.entry_count()
        assert entry_count <= 20
        assert file_path.exists()
        assert file_path.stat().st_size <= 10240

    asyncio.run(run())
