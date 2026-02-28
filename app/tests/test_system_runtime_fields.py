from __future__ import annotations

import asyncio

from app.routes.system import get_runtime


class _FakeBrain:
    def __init__(self, *, effective_backend: str, judge_ready: bool) -> None:
        self._effective_backend = effective_backend
        self._judge_ready = judge_ready

    def is_model_ready(self) -> bool:
        return True

    def model_backend_effective(self) -> str:
        return self._effective_backend

    def can_use_model_judge(self) -> bool:
        return self._judge_ready


def test_runtime_exposes_backend_and_judge_fields(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_BACKEND", "hybrid")
    monkeypatch.setenv("RESPONSE_MATCH_MODEL_ENABLED", "true")

    runtime = asyncio.run(
        get_runtime(
            brain=_FakeBrain(
                effective_backend="local_llama",
                judge_ready=True,
            ),
        ),
    )

    assert runtime.model_backend_configured == "hybrid"
    assert runtime.model_backend_effective == "local_llama"
    assert runtime.response_match_model_enabled is True
    assert runtime.response_match_mode == "smart_gate"


def test_runtime_marks_judge_disabled_when_model_judge_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_BACKEND", "api_llm")
    monkeypatch.setenv("RESPONSE_MATCH_MODEL_ENABLED", "true")

    runtime = asyncio.run(
        get_runtime(
            brain=_FakeBrain(
                effective_backend="heuristic",
                judge_ready=False,
            ),
        ),
    )

    assert runtime.model_backend_configured == "api_llm"
    assert runtime.model_backend_effective == "heuristic"
    assert runtime.response_match_model_enabled is False
    assert runtime.response_match_mode == "smart_gate"
