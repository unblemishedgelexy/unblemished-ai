from __future__ import annotations

import asyncio
from typing import Any

from app.core.logger import StructuredLogger, setup_logger
from app.services.brain.model_adapter import ModelAdapter


def test_local_backend_selection() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(
        logger=logger,
        backend="local_llama",
        local_model_path="dummy.gguf",
    )

    async def _fake_local_call(*, prompt: str, max_tokens: int, temperature: float) -> str:
        return f"local raw answer for: {prompt}"

    adapter._call_local_llama = _fake_local_call  # type: ignore[method-assign]

    result = asyncio.run(
        adapter.generate(
            prompt="Explain modular architecture",
            trace_id="trace-local-select",
            context={},
            model_name="creative_model",
            max_tokens=128,
            max_retries=0,
        ),
    )

    assert result.backend == "local_llama"
    assert result.fallback_used is False
    assert "local raw answer" in result.text


def test_local_backend_fallback_behavior() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(
        logger=logger,
        backend="local_llama",
        local_model_path="dummy.gguf",
    )

    async def _failing_local_call(*, prompt: str, max_tokens: int, temperature: float) -> str:
        raise RuntimeError("local backend failure")

    adapter._call_local_llama = _failing_local_call  # type: ignore[method-assign]

    result = asyncio.run(
        adapter.generate(
            prompt="Explain modular architecture",
            trace_id="trace-local-fallback",
            context={},
            model_name="creative_model",
            max_tokens=128,
            max_retries=1,
        ),
    )

    assert result.backend == "local_llama"
    assert result.fallback_used is True
    assert result.failure_reason == "RuntimeError"


def test_local_backend_streaming_path() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(
        logger=logger,
        backend="local_llama",
        local_model_path="dummy.gguf",
    )

    async def _fake_local_stream(
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
    ):
        del prompt, max_tokens, temperature, timeout_seconds
        for token in ("hello ", "world"):
            yield token

    adapter._call_local_llama_stream = _fake_local_stream  # type: ignore[method-assign]

    async def _collect() -> tuple[list[str], Any]:
        tokens: list[str] = []
        final = None
        async for event in adapter.generate_stream_events(
            prompt="test prompt",
            trace_id="trace-local-stream",
            context={},
            model_name="creative_model",
            max_tokens=64,
            max_retries=0,
        ):
            if event.token is not None:
                tokens.append(event.token)
            if event.result is not None:
                final = event.result
        return tokens, final

    tokens, final = asyncio.run(_collect())
    assert "".join(tokens) == "hello world"
    assert final is not None
    assert final.backend == "local_llama"
    assert final.fallback_used is False

