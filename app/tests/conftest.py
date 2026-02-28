from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from app.core.config import Settings
from app.core.dependencies import ServiceContainer, create_container
from app.services.brain.model_adapter import ModelAdapter


@dataclass(slots=True)
class TestHarness:
    container: ServiceContainer
    temp_dir: TemporaryDirectory[str]

    @property
    def brain(self):
        return self.container.brain

    @property
    def settings(self) -> Settings:
        return self.container.settings

    @property
    def memory_store(self):
        return self.container.memory_store

    @property
    def profile_store(self):
        return self.container.profile_store

    @property
    def goal_store(self):
        return self.container.goal_store

    @property
    def embedding_interface(self):
        return self.container.embedding_interface

    @property
    def logger(self):
        return self.container.logger

    @property
    def task_manager(self):
        return self.container.task_manager

    @property
    def model_router(self):
        return self.container.model_router

    @property
    def skill_interface(self):
        return self.container.skill_interface


def _build_settings(
    *,
    db_path: str,
    overrides: dict[str, Any] | None,
) -> Settings:
    payload = Settings(memory_db_path=db_path).model_dump()
    if overrides:
        payload.update(overrides)
    return Settings(**payload)


@pytest.fixture
def brain_factory():
    harnesses: list[TestHarness] = []

    def _factory(
        *,
        overrides: dict[str, Any] | None = None,
        model_adapter: ModelAdapter | None = None,
    ) -> TestHarness:
        temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        db_path = str(Path(temp_dir.name) / "memory.db")
        settings = _build_settings(db_path=db_path, overrides=overrides)
        container = create_container(settings=settings, model_adapter=model_adapter)
        harness = TestHarness(container=container, temp_dir=temp_dir)
        harnesses.append(harness)
        return harness

    yield _factory

    for harness in harnesses:
        try:
            asyncio.run(harness.task_manager.graceful_shutdown())
        except RuntimeError:
            # If an event loop is already active during fixture teardown, skip explicit drain.
            pass
        harness.temp_dir.cleanup()
