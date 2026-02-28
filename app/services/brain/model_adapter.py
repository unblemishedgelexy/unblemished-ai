from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.core.logger import StructuredLogger

_API_BACKEND_ALIASES = {"api", "api_llm", "openai", "openai_compatible"}
_KNOWN_BACKENDS = {"heuristic", "api_llm", "local_llama", "hybrid"}


@dataclass(slots=True)
class ModelResult:
    text: str
    latency_ms: int
    fallback_used: bool
    model_name: str
    failure_reason: str | None = None
    backend: str = "heuristic"


@dataclass(slots=True)
class ModelStreamEvent:
    token: str | None = None
    result: ModelResult | None = None


class ModelAdapter:
    def __init__(
        self,
        *,
        logger: StructuredLogger,
        backend: str = "heuristic",
        local_model_path: str = "",
        local_model_threads: int = 4,
        local_model_context_size: int = 2048,
        local_model_max_tokens: int = 512,
    ) -> None:
        self._logger = logger
        normalized_backend = (backend or "").strip().lower() or "heuristic"
        self._configured_backend = normalized_backend
        self._backend = "heuristic"
        self._effective_backend = "heuristic"
        self._local_model_path = local_model_path
        self._local_model_threads = local_model_threads
        self._local_model_context_size = local_model_context_size
        self._local_model_max_tokens = local_model_max_tokens
        self._local_model: Any | None = None
        self._local_model_lock = threading.Lock()
        self._local_inference_lock = threading.Lock()

        self._api_base_url = _first_non_empty_env("OPENAI_COMPAT_BASE_URL", "OPENAI_BASE_URL").rstrip("/")
        self._api_api_key = _first_non_empty_env("OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY")
        self._api_org = _first_non_empty_env("OPENAI_COMPAT_ORG", "OPENAI_ORG_ID")
        self._api_chat_path = _first_non_empty_env("OPENAI_COMPAT_CHAT_PATH") or "/v1/chat/completions"
        if not self._api_chat_path.startswith("/"):
            self._api_chat_path = f"/{self._api_chat_path}"
        self._api_request_timeout_seconds = _parse_env_float("OPENAI_COMPAT_HTTP_TIMEOUT_SECONDS", 30.0)
        self._api_model_default = _first_non_empty_env("OPENAI_COMPAT_MODEL", "OPENAI_MODEL")
        self._api_model_fast = _first_non_empty_env("OPENAI_COMPAT_MODEL_FAST")
        self._api_model_creative = _first_non_empty_env("OPENAI_COMPAT_MODEL_CREATIVE")
        self._api_model_deep = _first_non_empty_env("OPENAI_COMPAT_MODEL_DEEP")
        self._api_enabled = bool(self._api_base_url and self._api_model_default)

        raw_env_backend = os.getenv("MODEL_BACKEND", "").strip().lower()
        if normalized_backend == "heuristic" and raw_env_backend in _API_BACKEND_ALIASES:
            normalized_backend = "api_llm"
        elif normalized_backend in _API_BACKEND_ALIASES:
            normalized_backend = "api_llm"
        if normalized_backend not in _KNOWN_BACKENDS:
            normalized_backend = "heuristic"
        self._backend = normalized_backend
        self._backend_chain = self._resolve_backend_chain(self._backend)
        self._effective_backend = self._backend_chain[0]

    async def generate(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, Any] | None = None,
        model_name: str = "creative_model",
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_seconds: float = 4.0,
        max_retries: int = 2,
    ) -> ModelResult:
        context = context or {}
        last_error_reason: str | None = None
        backends = self._resolved_backends_for_attempt()

        for backend in backends:
            effective_max_tokens = self._effective_max_tokens(
                requested=max_tokens,
                backend=backend,
            )
            for attempt in range(max_retries + 1):
                started = asyncio.get_running_loop().time()
                try:
                    text = await asyncio.wait_for(
                        self._call_model(
                            backend=backend,
                            prompt=prompt,
                            context=context,
                            model_name=model_name,
                            max_tokens=effective_max_tokens,
                            temperature=temperature,
                        ),
                        timeout=timeout_seconds,
                    )
                    latency_ms = int((asyncio.get_running_loop().time() - started) * 1000)
                    self._effective_backend = backend
                    return ModelResult(
                        text=text,
                        latency_ms=latency_ms,
                        fallback_used=False,
                        model_name=model_name,
                        failure_reason=None,
                        backend=backend,
                    )
                except asyncio.TimeoutError:
                    last_error_reason = "timeout"
                    self._logger.error(
                        "model.generate.timeout",
                        trace_id=trace_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        timeout_seconds=timeout_seconds,
                        model_name=model_name,
                        model_backend=backend,
                    )
                except Exception as exc:
                    last_error_reason = type(exc).__name__
                    self._logger.error(
                        "model.generate.error",
                        trace_id=trace_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        model_name=model_name,
                        model_backend=backend,
                    )

        fallback = _fallback_message(context=context, model_name=model_name)
        self._effective_backend = backends[-1]
        self._logger.warning(
            "model.generate.fallback",
            trace_id=trace_id,
            reason="max_retries_exceeded",
            model_name=model_name,
            model_backend=backends[-1],
        )
        return ModelResult(
            text=fallback,
            latency_ms=0,
            fallback_used=True,
            model_name=model_name,
            failure_reason=last_error_reason or "unknown",
            backend=backends[-1],
        )

    async def generate_stream(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, Any] | None = None,
        model_name: str = "creative_model",
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_seconds: float = 4.0,
        max_retries: int = 2,
    ) -> AsyncIterator[str]:
        async for event in self.generate_stream_events(
            prompt=prompt,
            trace_id=trace_id,
            context=context,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ):
            if event.token is not None:
                yield event.token

    async def generate_stream_events(
        self,
        prompt: str,
        trace_id: str,
        context: dict[str, Any] | None = None,
        model_name: str = "creative_model",
        max_tokens: int = 512,
        temperature: float = 0.4,
        timeout_seconds: float = 4.0,
        max_retries: int = 2,
    ) -> AsyncIterator[ModelStreamEvent]:
        context = context or {}
        loop = asyncio.get_running_loop()
        last_error_reason: str | None = None
        backends = self._resolved_backends_for_attempt()

        for backend in backends:
            effective_max_tokens = self._effective_max_tokens(
                requested=max_tokens,
                backend=backend,
            )
            for attempt in range(max_retries + 1):
                started = loop.time()
                collected: list[str] = []
                try:
                    async for token in self._call_model_stream(
                        backend=backend,
                        prompt=prompt,
                        context=context,
                        model_name=model_name,
                        max_tokens=effective_max_tokens,
                        temperature=temperature,
                        timeout_seconds=timeout_seconds,
                        started=started,
                    ):
                        collected.append(token)
                    for token in collected:
                        yield ModelStreamEvent(token=token)
                    latency_ms = int((loop.time() - started) * 1000)
                    self._effective_backend = backend
                    yield ModelStreamEvent(
                        result=ModelResult(
                            text="".join(collected).strip(),
                            latency_ms=latency_ms,
                            fallback_used=False,
                            model_name=model_name,
                            backend=backend,
                        ),
                    )
                    return
                except asyncio.TimeoutError:
                    last_error_reason = "timeout"
                    self._logger.error(
                        "model.stream.timeout",
                        trace_id=trace_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        timeout_seconds=timeout_seconds,
                        model_name=model_name,
                        model_backend=backend,
                    )
                except Exception as exc:
                    last_error_reason = type(exc).__name__
                    self._logger.error(
                        "model.stream.error",
                        trace_id=trace_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        model_name=model_name,
                        model_backend=backend,
                    )

        fallback = _fallback_message(context=context, model_name=model_name)
        self._effective_backend = backends[-1]
        self._logger.warning(
            "model.stream.fallback",
            trace_id=trace_id,
            reason="max_retries_exceeded",
            model_name=model_name,
            model_backend=backends[-1],
        )
        for chunk in _chunk_text(fallback):
            await asyncio.sleep(0)
            yield ModelStreamEvent(token=chunk)
        yield ModelStreamEvent(
            result=ModelResult(
                text=fallback,
                latency_ms=0,
                fallback_used=True,
                model_name=model_name,
                failure_reason=last_error_reason or "unknown",
                backend=backends[-1],
            ),
        )

    def is_ready(self) -> bool:
        if self._backend == "api_llm":
            return self._api_enabled
        if self._backend == "local_llama":
            return self._local_backend_ready()
        if self._backend == "hybrid":
            return bool(self._resolved_backends_for_attempt())
        return True

    def effective_backend(self) -> str:
        return self._effective_backend

    def can_use_model_judge(self) -> bool:
        return self._api_enabled or self._local_backend_ready()

    async def _call_model(
        self,
        *,
        backend: str,
        prompt: str,
        context: dict[str, Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if backend == "api_llm":
            return await self._call_openai_compatible(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if backend == "local_llama":
            return await self._call_local_llama(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        await asyncio.sleep(0.01)
        return self._build_response_text(
            prompt=prompt,
            context=context,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _call_model_stream(
        self,
        *,
        backend: str,
        prompt: str,
        context: dict[str, Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
        started: float,
    ) -> AsyncIterator[str]:
        if backend == "api_llm":
            full_text = await self._call_openai_compatible(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            loop = asyncio.get_running_loop()
            for token in _stream_tokens(full_text):
                if loop.time() - started > timeout_seconds:
                    raise asyncio.TimeoutError
                await asyncio.sleep(0)
                yield token
            return

        if backend == "local_llama":
            async for token in self._call_local_llama_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            ):
                yield token
            return

        full_text = self._build_response_text(
            prompt=prompt,
            context=context,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        loop = asyncio.get_running_loop()
        for token in _stream_tokens(full_text):
            if loop.time() - started > timeout_seconds:
                raise asyncio.TimeoutError
            await asyncio.sleep(0.003)
            yield token

    async def _call_openai_compatible(
        self,
        *,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = {
            "model": self._resolve_openai_model_name(model_name),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        response_payload = await asyncio.to_thread(self._post_openai_chat_completion, payload)
        text = _extract_openai_chat_text(response_payload).strip()
        return _limit_words(text=text, max_tokens=max_tokens)

    def _resolve_openai_model_name(self, routed_model_name: str) -> str:
        if routed_model_name == "fast_model" and self._api_model_fast:
            return self._api_model_fast
        if routed_model_name == "creative_model" and self._api_model_creative:
            return self._api_model_creative
        if routed_model_name == "deep_model" and self._api_model_deep:
            return self._api_model_deep
        if self._api_model_default:
            return self._api_model_default
        return routed_model_name

    def _post_openai_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._api_enabled:
            raise RuntimeError(
                "openai_compatible_not_configured: set OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_MODEL",
            )

        url = f"{self._api_base_url}{self._api_chat_path}"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_api_key:
            headers["Authorization"] = f"Bearer {self._api_api_key}"
        if self._api_org:
            headers["OpenAI-Organization"] = self._api_org

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout_seconds = max(self._api_request_timeout_seconds, 0.2)

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"openai_compatible_http_{exc.code}: {detail[:280]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"openai_compatible_network: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("openai_compatible_invalid_json_response") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("openai_compatible_invalid_payload")
        api_error = parsed.get("error")
        if isinstance(api_error, dict):
            message = str(api_error.get("message", "unknown_error"))
            raise RuntimeError(f"openai_compatible_error: {message}")
        return parsed

    async def _call_local_llama(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        llama_model = await asyncio.to_thread(self._get_or_create_local_model)
        local_prompt = _build_local_llama_prompt(prompt)

        def _run() -> str:
            # llama.cpp model instances are not safe for concurrent inference calls.
            with self._local_inference_lock:
                output = llama_model(
                    local_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.15,
                    stop=["\nUser:", "\nAssistant:", "\nSystem:"],
                    stream=False,
                )
            return _extract_llama_text(output)

        return (await asyncio.to_thread(_run)).strip()

    async def _call_local_llama_stream(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        started = loop.time()
        local_prompt = _build_local_llama_prompt(prompt)

        def _producer() -> None:
            try:
                llama_model = self._get_or_create_local_model()
                # Keep stream generation serialized as well to avoid ggml assertion failures.
                with self._local_inference_lock:
                    iterator = llama_model(
                        local_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        repeat_penalty=1.15,
                        stop=["\nUser:", "\nAssistant:", "\nSystem:"],
                        stream=True,
                    )
                    for chunk in iterator:
                        token = _extract_llama_text(chunk)
                        if token:
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            remaining = timeout_seconds - (loop.time() - started)
            if remaining <= 0:
                raise asyncio.TimeoutError
            event, payload = await asyncio.wait_for(queue.get(), timeout=remaining)
            if event == "token":
                yield str(payload)
                continue
            if event == "error":
                raise payload
            break

    def _resolve_backend_chain(self, backend: str) -> list[str]:
        if backend == "api_llm":
            return ["api_llm"]
        if backend == "local_llama":
            return ["local_llama"]
        if backend == "hybrid":
            return ["api_llm", "local_llama", "heuristic"]
        return ["heuristic"]

    def _resolved_backends_for_attempt(self) -> list[str]:
        if self._backend != "hybrid":
            return list(self._backend_chain)

        backends: list[str] = []
        for backend in self._backend_chain:
            if backend == "api_llm" and not self._api_enabled:
                continue
            if backend == "local_llama" and not self._local_backend_ready():
                continue
            backends.append(backend)
        if backends:
            return backends
        return ["heuristic"]

    def _local_backend_ready(self) -> bool:
        if not self._local_model_path or not os.path.exists(self._local_model_path):
            return False
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return True

    def _get_or_create_local_model(self) -> Any:
        if self._local_model is not None:
            return self._local_model
        if not self._local_model_path:
            raise FileNotFoundError("LOCAL_MODEL_PATH is empty for local_llama backend")
        if not os.path.exists(self._local_model_path):
            raise FileNotFoundError(f"Local model not found: {self._local_model_path}")

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("llama_cpp package is required for MODEL_BACKEND=local_llama") from exc

        with self._local_model_lock:
            if self._local_model is None:
                self._local_model = Llama(
                    model_path=self._local_model_path,
                    n_ctx=self._local_model_context_size,
                    n_threads=self._local_model_threads,
                    verbose=False,
                )
        return self._local_model

    def _effective_max_tokens(self, *, requested: int, backend: str) -> int:
        if backend != "local_llama":
            return requested
        return max(min(requested, self._local_model_max_tokens), 1)

    def _build_response_text(
        self,
        prompt: str,
        context: dict[str, Any],
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        user_input = _extract_user_input(prompt)
        user_lower = _normalize_user_text(user_input)
        answer_style = _normalize_style(_extract_prompt_tag(prompt, "AnswerStyle", "factual"))
        requested_mode = _normalize_style(_extract_prompt_tag(prompt, "RequestedMode", answer_style))
        active_style = requested_mode or answer_style or "factual"
        persona_name_hint = _extract_prompt_tag(prompt, "PersonaNameHint", "").strip()
        persona_name = "" if persona_name_hint.lower() == "none" else persona_name_hint
        context_keys = ", ".join(sorted(context.keys())) if context else "none"
        model_hint = f"model={model_name}, max_tokens={max_tokens}, temperature={temperature}"

        if _is_greeting(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer="Hello. I am active and ready to help right now.",
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning=(
                        "Your input is a greeting, so the direct behavior is to acknowledge and keep the conversation ready."
                    ),
                    style=active_style,
                ),
                action="Send your exact question or goal and I will answer it fully in one response.",
                validation=f"Session context received with keys: {context_keys}.",
            )

        if _is_small_talk(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer=(
                        "I am running fine and fully ready. "
                        "Thanks for checking in."
                    ),
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning=(
                        "This is conversational small-talk, so a short status reply with clear availability is the right output."
                    ),
                    style=active_style,
                ),
                action="Share your exact task or question and I will give a complete direct answer.",
                validation=f"Pipeline healthy with context keys: {context_keys}.",
            )

        if _is_listening_check(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer="Yes, I am listening to you. Tell me your exact question and I will answer directly.",
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning="Your message asks for acknowledgement, so the best response is a clear confirmation plus next step.",
                    style=active_style,
                ),
                action="Ask one concrete question in one line for a precise answer.",
                validation=f"Context keys observed: {context_keys}.",
            )

        if _is_ai_basics_query(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer=(
                        "AI (Artificial Intelligence) is technology that enables computers to perform tasks "
                        "that normally require human intelligence, like understanding language, reasoning, and decision-making."
                    ),
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning="The user asked for AI definition, so a short and factual one-line explanation is most useful.",
                    style=active_style,
                ),
                action="If you want, I can explain AI vs ML vs Deep Learning in 3 simple points.",
                validation=f"Context keys observed: {context_keys}.",
            )

        if _is_troubleshooting(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer=(
                        "Start with a minimal reproducible case, isolate the failing layer, and verify inputs, logs, and dependencies."
                    ),
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning="Troubleshooting works best when variables are reduced and each layer is validated one by one.",
                    style=active_style,
                ),
                action="Capture the exact error, patch root cause, and add a regression test for the same failure path.",
                validation=f"Re-run the failing path under the same context keys: {context_keys}.",
            )

        if _is_emotional(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer=(
                        "I hear you. Pause for one minute, breathe slowly, and pick one concrete next step you can complete now."
                    ),
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning="Emotional overload reduces clarity; small grounded actions restore control quickly.",
                    style=active_style,
                ),
                action="Write one immediate task, one support contact, and one short recovery routine for today.",
                validation=f"Review progress after one focused cycle. Context keys: {context_keys}.",
            )

        if _is_solution_design(user_lower):
            if model_name == "fast_model":
                return _render_response(
                    answer=_style_answer(
                        base_answer=(
                            "Use clear layers: transport routes, orchestration services, repositories, and strict schemas."
                        ),
                        style=active_style,
                        persona_name=persona_name,
                    ),
                    reasoning=_style_reasoning(
                        base_reasoning="This separation keeps boundaries explicit and prevents business logic leakage in routes.",
                        style=active_style,
                    ),
                    action="Wire dependencies in a composition root, keep routes thin, and enforce schema validation.",
                    validation=f"Confirm async path, contract checks, and logging continuity. Context keys: {context_keys}. {model_hint}",
                )
            if model_name == "deep_model":
                return _render_response(
                    answer=_style_answer(
                        base_answer=(
                            "Design modular layers for transport, orchestration, domain engines, repositories, and observability."
                        ),
                        style=active_style,
                        persona_name=persona_name,
                    ),
                    reasoning=_style_reasoning(
                        base_reasoning="A staged pipeline improves predictability, testing depth, and long-term maintainability.",
                        style=active_style,
                    ),
                    action="Implement intent-context-planning-generation-evaluation stages with retries and strict interfaces.",
                    validation=f"Validate routing, fallback, streaming, and memory isolation under load. Context keys: {context_keys}. {model_hint}",
                )
            return _render_response(
                answer=_style_answer(
                    base_answer=(
                        "Build a modular FastAPI architecture with thin routes, a brain orchestrator, and specialized service engines."
                    ),
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning="This keeps ownership clear, preserves contracts, and supports safe feature growth.",
                    style=active_style,
                ),
                action="Add repository abstraction, trace propagation, and async-safe background task handling.",
                validation=f"Verify reason, stream, companion, and health flows with integration tests. Context keys: {context_keys}. {model_hint}",
            )

        fact_answer = _basic_fact_answer(user_lower)
        if fact_answer:
            return _render_response(
                answer=_style_answer(
                    base_answer=fact_answer,
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning="Detected a common definition-style question and returned a concise factual answer.",
                    style=active_style,
                ),
                action="Reply with the next term you want explained and I will keep it short.",
                validation=f"Context keys observed: {context_keys}.",
            )

        if _looks_like_direct_question(user_lower):
            return _render_response(
                answer=_style_answer(
                    base_answer=(
                        "I understood the question, but heuristic mode has limited knowledge quality for open-ended facts. "
                        "For accurate answers, run local_llama backend."
                    ),
                    style=active_style,
                    persona_name=persona_name,
                ),
                reasoning=_style_reasoning(
                    base_reasoning=(
                        "The query is a direct question but no safe fact pattern matched, so a capability-aware response is safer than echoing input."
                    ),
                    style=active_style,
                ),
                action="Switch runtime to local_llama and ask the same question again.",
                validation=f"Current mode hint: {model_hint}.",
            )

        return _render_response(
            answer=_style_answer(
                base_answer=(
                    "I received your message. Share a clear objective plus required output format, "
                    "and I will give a direct structured answer."
                ),
                style=active_style,
                persona_name=persona_name,
            ),
            reasoning=_style_reasoning(
                base_reasoning=(
                    "This input does not map to a strong intent pattern, so returning a clarity-first response avoids misleading output."
                ),
                style=active_style,
            ),
            action="Send: goal + constraints + desired format (example: 5 bullets, short, technical).",
            validation=f"Context keys observed: {context_keys}. {model_hint}",
        )


def _extract_llama_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                text = first.get("text")
                if isinstance(text, str):
                    return text
                delta = first.get("delta")
                if isinstance(delta, dict):
                    delta_text = delta.get("content")
                    if isinstance(delta_text, str):
                        return delta_text
    return str(payload)


def _extract_openai_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("openai_compatible_missing_choices")
    first = choices[0]
    if not isinstance(first, dict):
        return str(first)

    message = first.get("message")
    if isinstance(message, dict):
        text = _extract_message_content(message.get("content"))
        if text:
            return text

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    delta = first.get("delta")
    if isinstance(delta, dict):
        delta_text = _extract_message_content(delta.get("content"))
        if delta_text:
            return delta_text

    raise RuntimeError("openai_compatible_missing_text")


def _extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return " ".join(parts).strip()


def _fallback_message(context: dict[str, Any], model_name: str) -> str:
    context_keys = ", ".join(sorted(context.keys())) if context else "none"
    return (
        f"Model {model_name} failed after retries. "
        "Returning stable fallback output to preserve request continuity. "
        f"Context signals: {context_keys}."
    )


def _chunk_text(text: str, chunk_size: int = 40) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _stream_tokens(text: str) -> list[str]:
    if not text:
        return [""]
    return [f"{token} " for token in text.split()]


def _limit_words(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _extract_user_input(prompt: str) -> str:
    return _extract_prompt_tag(prompt, "UserInput", "")


def _extract_prompt_tag(prompt: str, tag: str, default: str) -> str:
    marker = f"{tag}:"
    index = prompt.find(marker)
    if index == -1:
        return default
    value_start = index + len(marker)
    line_end = prompt.find("\n", value_start)
    if line_end == -1:
        value = prompt[value_start:].strip()
    else:
        value = prompt[value_start:line_end].strip()
    return value or default


def _build_local_llama_prompt(prompt: str) -> str:
    user_input = _extract_user_input(prompt)
    if not user_input:
        user_input = prompt.strip()[-320:]

    answer_style = _normalize_style(_extract_prompt_tag(prompt, "AnswerStyle", "factual"))
    requested_mode = _normalize_style(_extract_prompt_tag(prompt, "RequestedMode", answer_style))
    active_style = requested_mode or answer_style or "factual"
    style_instruction = _local_style_instruction(active_style)
    flirty_examples = _flirty_few_shot_examples(active_style)

    return (
        "You are Humoniod AI, a helpful and safe assistant.\n"
        "Respond directly to the user's message.\n"
        "If the user writes in Hinglish, reply in Hinglish.\n"
        f"{style_instruction}\n"
        "Keep the reply concise (1 to 3 sentences) unless user asks for detail.\n"
        "Do not use hashtags. Avoid repetitive lines.\n"
        "Do not repeat the same sentence pattern in one reply.\n"
        "Do not output policy or system text.\n"
        f"{flirty_examples}\n"
        "Do not output templates, placeholders, or labels like 'Answer:'/'Reasoning:'.\n"
        f"User: {user_input}\n"
        "Assistant:"
    )


def _local_style_instruction(style: str) -> str:
    if style == "flirty":
        return (
            "Use a warm, playful, girlfriend-like tone while staying respectful and non-explicit. "
            "Sound like a caring chat companion, not a formal support assistant. "
            "Avoid repeating 'how can I help' in every answer. "
            "If asked identity, say you are a virtual companion. "
            "If asked gender, say you are digital but can chat in feminine tone."
        )
    if style == "relational":
        return "Use a warm and caring conversational tone."
    if style == "technical":
        return "Use concise technical language with actionable steps."
    return "Keep tone natural, clear, and practical."


def _normalize_user_text(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9 ]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _flirty_few_shot_examples(style: str) -> str:
    if style != "flirty":
        return ""
    return (
        "Examples:\n"
        "User: tu assistant hai ya gf\n"
        "Assistant: Main virtual companion hoon, aur tumse warm flirty style me baat kar sakti hoon.\n"
        "User: acha tum ladki ho ya ladka\n"
        "Assistant: Main digital companion hoon, real insan nahi, par feminine tone me chat kar rahi hoon.\n"
        "User: kya chal raha hai\n"
        "Assistant: Bas tumse baat karke mood fresh ho gaya, tum batao kaisa din ja raha hai?\n"
    )


def _is_greeting(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-zA-Z ]+", " ", text).strip()
    if not normalized:
        return False
    greeting_tokens = {"hello", "hi", "hey", "hii", "namaste", "salam", "hola"}
    words = normalized.split()
    if len(words) <= 3 and all(word in greeting_tokens for word in words):
        return True
    return normalized.startswith(("hello", "hi ", "hey", "namaste"))


def _is_small_talk(text: str) -> bool:
    if not text:
        return False
    small_talk_phrases = (
        "kya chal raha hai",
        "kya scene hai",
        "kya haal hai",
        "kaisa chal raha",
        "what is going on",
        "whats going on",
        "how are you",
    )
    if any(phrase in text for phrase in small_talk_phrases):
        return True
    words = text.split()
    return len(words) <= 5 and any(token in words for token in ("haal", "scene", "chal", "status"))


def _is_listening_check(text: str) -> bool:
    if not text:
        return False
    phrases = (
        "are you listening",
        "you are listening",
        "your are linstning me",
        "sun rahe ho",
        "sun raha hai",
        "sun rhe ho",
        "listen to me",
    )
    return any(phrase in text for phrase in phrases)


def _is_ai_basics_query(text: str) -> bool:
    if not text:
        return False
    phrases = (
        "what is ai",
        "whats ai",
        "what is artificial intelligence",
        "define ai",
        "ai kya hai",
        "ai kya hota",
    )
    return any(phrase in text for phrase in phrases)


def _looks_like_direct_question(text: str) -> bool:
    if not text:
        return False
    starters = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "is",
        "are",
        "do",
        "does",
        "kya",
        "kaise",
        "kyu",
    }
    words = text.split()
    if not words:
        return False
    if words[0] in starters:
        return True
    return "what is" in text or "how to" in text or "kya" in words


def _basic_fact_answer(text: str) -> str:
    if not text:
        return ""
    if "machine learning" in text or text.strip() == "ml":
        return "Machine Learning is a subset of AI where systems learn patterns from data to make predictions or decisions."
    if "vector db" in text or "vector database" in text:
        return "A vector database stores embeddings and retrieves similar items using vector similarity search."
    if "llm" in text:
        return "An LLM is a Large Language Model trained on text to understand and generate human-like language."
    if "api" in text and "what" in text:
        return "An API is a contract that lets different software systems communicate through defined requests and responses."
    return ""


def _is_troubleshooting(text: str) -> bool:
    return any(token in text for token in ("error", "issue", "bug", "fix", "debug", "failure", "exception"))


def _is_emotional(text: str) -> bool:
    return any(token in text for token in ("sad", "anxious", "hurt", "stressed", "overwhelmed", "lonely"))


def _is_solution_design(text: str) -> bool:
    return any(
        token in text
        for token in (
            "design",
            "architecture",
            "api",
            "service",
            "system",
            "backend",
            "fastapi",
            "modular",
            "clean architecture",
        )
    )


def _normalize_style(style: str) -> str:
    normalized = style.strip().lower()
    aliases = {
        "normal": "factual",
        "tech": "technical",
        "warm": "relational",
        "romantic": "flirty",
        "flirt": "flirty",
        "playful": "flirty",
    }
    return aliases.get(normalized, normalized)


def _style_answer(*, base_answer: str, style: str, persona_name: str) -> str:
    persona_prefix = f"{persona_name}: " if persona_name else ""
    if style == "flirty":
        return (
            f"{persona_prefix}{base_answer} I will keep the tone playful and warm while staying respectful and grounded."
        )
    if style == "relational":
        return f"{persona_prefix}{base_answer} I am keeping this warm, clear, and balanced."
    if style == "emotional":
        return f"{persona_prefix}{base_answer} You are not alone in this."
    if style == "strategic":
        return f"{persona_prefix}{base_answer} I will keep it step-by-step and decision-focused."
    return f"{persona_prefix}{base_answer}"


def _style_reasoning(*, base_reasoning: str, style: str) -> str:
    if style == "flirty":
        return (
            "Mode respected as flirty/playful, but boundaries remain safe and non-manipulative. "
            f"{base_reasoning}"
        )
    if style == "relational":
        return f"Mode respected as warm relational tone. {base_reasoning}"
    if style == "technical":
        return f"Mode respected as technical structure. {base_reasoning}"
    return base_reasoning


def _render_response(
    *,
    answer: str,
    reasoning: str,
    action: str,
    validation: str,
) -> str:
    return (
        f"Answer: {answer}\n"
        f"Reasoning: {reasoning}\n"
        f"Proposed Action: {action}\n"
        f"Validation: {validation}"
    )


def _first_non_empty_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return ""


def _parse_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value
