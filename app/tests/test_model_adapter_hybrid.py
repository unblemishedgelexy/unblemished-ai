from __future__ import annotations

import asyncio
from typing import Any

from app.core.logger import StructuredLogger, setup_logger
from app.services.brain.model_adapter import ModelAdapter


def test_hybrid_backend_prefers_api_when_available() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(logger=logger, backend="hybrid", local_model_path="dummy.gguf")
    adapter._api_enabled = True  # noqa: SLF001
    adapter._local_backend_ready = lambda: True  # type: ignore[method-assign]  # noqa: SLF001

    async def _fake_api(
        *,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        del prompt, model_name, max_tokens, temperature
        return "api_llm answer"

    async def _should_not_run_local(*, prompt: str, max_tokens: int, temperature: float) -> str:
        del prompt, max_tokens, temperature
        raise AssertionError("local backend should not run when api_llm succeeds")

    adapter._call_openai_compatible = _fake_api  # type: ignore[method-assign]  # noqa: SLF001
    adapter._call_local_llama = _should_not_run_local  # type: ignore[method-assign]  # noqa: SLF001

    result = asyncio.run(
        adapter.generate(
            prompt="UserInput: test hybrid api path",
            trace_id="trace-hybrid-api",
            model_name="creative_model",
            max_retries=0,
        ),
    )

    assert result.backend == "api_llm"
    assert result.fallback_used is False
    assert "api_llm answer" in result.text
    assert adapter.effective_backend() == "api_llm"


def test_hybrid_backend_falls_back_to_local_on_api_failure() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(logger=logger, backend="hybrid", local_model_path="dummy.gguf")
    adapter._api_enabled = True  # noqa: SLF001
    adapter._local_backend_ready = lambda: True  # type: ignore[method-assign]  # noqa: SLF001

    async def _failing_api(
        *,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        del prompt, model_name, max_tokens, temperature
        raise RuntimeError("api down")

    async def _fake_local(*, prompt: str, max_tokens: int, temperature: float) -> str:
        del prompt, max_tokens, temperature
        return "local_llama answer"

    adapter._call_openai_compatible = _failing_api  # type: ignore[method-assign]  # noqa: SLF001
    adapter._call_local_llama = _fake_local  # type: ignore[method-assign]  # noqa: SLF001

    result = asyncio.run(
        adapter.generate(
            prompt="UserInput: test hybrid local fallback path",
            trace_id="trace-hybrid-local-fallback",
            model_name="creative_model",
            max_retries=0,
        ),
    )

    assert result.backend == "local_llama"
    assert result.fallback_used is False
    assert "local_llama answer" in result.text
    assert adapter.effective_backend() == "local_llama"


def test_hybrid_backend_falls_back_to_heuristic_when_api_and_local_fail() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(logger=logger, backend="hybrid", local_model_path="dummy.gguf")
    adapter._api_enabled = True  # noqa: SLF001
    adapter._local_backend_ready = lambda: True  # type: ignore[method-assign]  # noqa: SLF001

    async def _failing_api(
        *,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        del prompt, model_name, max_tokens, temperature
        raise RuntimeError("api down")

    async def _failing_local(*, prompt: str, max_tokens: int, temperature: float) -> str:
        del prompt, max_tokens, temperature
        raise RuntimeError("local down")

    adapter._call_openai_compatible = _failing_api  # type: ignore[method-assign]  # noqa: SLF001
    adapter._call_local_llama = _failing_local  # type: ignore[method-assign]  # noqa: SLF001

    result = asyncio.run(
        adapter.generate(
            prompt="UserInput: explain retry strategy",
            trace_id="trace-hybrid-heuristic-fallback",
            model_name="creative_model",
            max_retries=0,
        ),
    )

    assert result.backend == "heuristic"
    assert result.fallback_used is False
    assert "Answer:" in result.text
    assert "Reasoning:" in result.text


def test_hybrid_stream_path_fails_over_to_local_llama() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    adapter = ModelAdapter(logger=logger, backend="hybrid", local_model_path="dummy.gguf")
    adapter._api_enabled = True  # noqa: SLF001
    adapter._local_backend_ready = lambda: True  # type: ignore[method-assign]  # noqa: SLF001

    async def _failing_api(
        *,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        del prompt, model_name, max_tokens, temperature
        raise RuntimeError("api stream fail")

    async def _fake_local_stream(
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
    ):
        del prompt, max_tokens, temperature, timeout_seconds
        for token in ("local ", "stream ", "ok"):
            yield token

    adapter._call_openai_compatible = _failing_api  # type: ignore[method-assign]  # noqa: SLF001
    adapter._call_local_llama_stream = _fake_local_stream  # type: ignore[method-assign]  # noqa: SLF001

    async def _collect() -> tuple[list[str], Any]:
        tokens: list[str] = []
        final = None
        async for event in adapter.generate_stream_events(
            prompt="UserInput: stream fallback",
            trace_id="trace-hybrid-stream",
            model_name="creative_model",
            max_retries=0,
        ):
            if event.token is not None:
                tokens.append(event.token)
            if event.result is not None:
                final = event.result
        return tokens, final

    tokens, final = asyncio.run(_collect())
    assert "".join(tokens) == "local stream ok"
    assert final is not None
    assert final.backend == "local_llama"
    assert final.fallback_used is False
