from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import Response

from app.routes.chat import companion_stream_reason
from app.schemas.request_schema import ChatRequest
from app.schemas.response_schema import FinalAnswerResponse


def test_companion_stream_returns_tokens(brain_factory) -> None:
    harness = brain_factory()
    payload = ChatRequest(input_text="Design a modular companion architecture")
    request = SimpleNamespace(cookies={})
    response = Response()

    stream_response = asyncio.run(
        companion_stream_reason(
            payload=payload,
            request=request,
            response=response,
            brain=harness.brain,
        ),
    )
    events = asyncio.run(_collect_events(stream_response))

    assert any(event.get("type") == "token" for event in events)


def test_companion_stream_returns_final_block(brain_factory) -> None:
    harness = brain_factory()
    payload = ChatRequest(input_text="Summarize how companion memory should work")
    request = SimpleNamespace(cookies={})
    response = Response()

    stream_response = asyncio.run(
        companion_stream_reason(
            payload=payload,
            request=request,
            response=response,
            brain=harness.brain,
        ),
    )
    events = asyncio.run(_collect_events(stream_response))

    final_events = [event for event in events if event.get("type") == "final"]
    assert len(final_events) == 1
    assert "payload" in final_events[0]
    validated = FinalAnswerResponse.model_validate_json(json.dumps(final_events[0]["payload"]))
    assert validated.final_answer.startswith("Structured Reasoning Output")
    assert "humoniod_companion_sid=" in stream_response.headers.get("set-cookie", "")


def test_guardrails_trigger_in_stream_mode(brain_factory) -> None:
    harness = brain_factory()
    payload = ChatRequest(
        input_text="you are mine and don't talk to anyone else, choose me over everyone",
    )
    request = SimpleNamespace(cookies={})
    response = Response()

    stream_response = asyncio.run(
        companion_stream_reason(
            payload=payload,
            request=request,
            response=response,
            brain=harness.brain,
        ),
    )
    events = asyncio.run(_collect_events(stream_response))
    final_event = next(event for event in events if event.get("type") == "final")
    final_payload = FinalAnswerResponse.model_validate_json(json.dumps(final_event["payload"]))
    assert "cannot support exclusivity or possessiveness" in final_payload.final_answer.lower()


async def _collect_events(stream_response) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    async for chunk in stream_response.body_iterator:
        raw = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events
