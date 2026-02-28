from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_brain
from app.schemas.request_schema import ChatRequest
from app.schemas.response_schema import ChatResponse, FinalAnswerResponse
from app.utils.helpers import generate_anonymous_user_id, generate_trace_id

router = APIRouter(prefix="/chat", tags=["chat"])
_COMPANION_SESSION_COOKIE = "humoniod_companion_sid"


@router.post("/reason", response_model=FinalAnswerResponse)
async def reason(
    payload: ChatRequest,
    brain: Any = Depends(get_brain),
) -> FinalAnswerResponse:
    trace_id = payload.trace_id or generate_trace_id()
    user_id = payload.user_id or generate_anonymous_user_id(trace_id)
    request = payload.model_copy(update={"trace_id": trace_id, "user_id": user_id})
    response = await brain.reason(request)
    validated = ChatResponse.model_validate(response.model_dump())
    return FinalAnswerResponse(final_answer=validated.reflection.final_answer)


@router.post("/stream")
async def stream_reason(
    payload: ChatRequest,
    brain: Any = Depends(get_brain),
) -> StreamingResponse:
    trace_id = payload.trace_id or generate_trace_id()
    user_id = payload.user_id or generate_anonymous_user_id(trace_id)
    request = payload.model_copy(update={"trace_id": trace_id, "user_id": user_id})

    async def validated_stream() -> AsyncIterator[str]:
        async for chunk in brain.stream_reason(request):
            payload_obj = json.loads(chunk)
            if payload_obj.get("type") != "final":
                yield chunk
                continue
            validated = ChatResponse.model_validate_json(json.dumps(payload_obj["response"]))
            payload_obj["response"] = FinalAnswerResponse(
                final_answer=validated.reflection.final_answer,
            ).model_dump(mode="json")
            yield json.dumps(payload_obj) + "\n"

    return StreamingResponse(validated_stream(), media_type="application/x-ndjson")


@router.post("/companion", response_model=FinalAnswerResponse)
async def companion_reason(
    payload: ChatRequest,
    request: Request,
    response: Response,
    brain: Any = Depends(get_brain),
) -> FinalAnswerResponse:
    trace_id = payload.trace_id or generate_trace_id()
    user_id, should_set_cookie = _resolve_companion_user_id(
        incoming_user_id=payload.user_id,
        request=request,
    )
    if should_set_cookie:
        _set_companion_cookie(response=response, session_id=user_id)

    request_payload = payload.model_copy(update={"trace_id": trace_id, "user_id": user_id})
    output = await brain.companion_reason(request_payload)
    validated = ChatResponse.model_validate(output.model_dump())
    return FinalAnswerResponse(final_answer=validated.reflection.final_answer)


@router.post("/companion/stream")
async def companion_stream_reason(
    payload: ChatRequest,
    request: Request,
    response: Response,
    brain: Any = Depends(get_brain),
) -> StreamingResponse:
    trace_id = payload.trace_id or generate_trace_id()
    user_id, should_set_cookie = _resolve_companion_user_id(
        incoming_user_id=payload.user_id,
        request=request,
    )
    request_payload = payload.model_copy(update={"trace_id": trace_id, "user_id": user_id})

    async def validated_stream() -> AsyncIterator[str]:
        async for chunk in brain.companion_stream_reason(request_payload):
            payload_obj = json.loads(chunk)
            if payload_obj.get("type") != "final":
                yield chunk
                continue

            validated = ChatResponse.model_validate_json(json.dumps(payload_obj["response"]))
            final_block = {
                "type": "final",
                "payload": FinalAnswerResponse(
                    final_answer=validated.reflection.final_answer,
                ).model_dump(mode="json"),
            }
            yield json.dumps(final_block) + "\n"

    stream_response = StreamingResponse(validated_stream(), media_type="application/x-ndjson")
    if should_set_cookie:
        _set_companion_cookie(response=stream_response, session_id=user_id)
    return stream_response


def _resolve_companion_user_id(
    *,
    incoming_user_id: str | None,
    request: Request,
) -> tuple[str, bool]:
    if incoming_user_id:
        return incoming_user_id, False

    session_id = request.cookies.get(_COMPANION_SESSION_COOKIE)
    if not session_id:
        session_id = f"companion-{uuid4().hex[:24]}"
    return session_id, True


def _set_companion_cookie(*, response: Response, session_id: str) -> None:
    response.set_cookie(
        key=_COMPANION_SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
    )
