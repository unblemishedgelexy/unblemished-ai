from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.schemas.request_schema import ChatRequest
from app.schemas.response_schema import ChatResponse
from app.services.brain.model_adapter import ModelAdapter
from app.services.brain.orchestrator import BrainOrchestrator
from app.services.goals.goal_interface import GoalInterface
from app.services.memory.memory_interface import MemoryInterface
from app.services.profile.user_profile_interface import UserProfileInterface
from app.services.router.model_router import ModelRouter
from app.services.tools.tool_interface import ToolInterface


class BrainInterface:
    def __init__(
        self,
        *,
        orchestrator: BrainOrchestrator,
        model_adapter: ModelAdapter,
        memory_interface: MemoryInterface,
        user_profile_interface: UserProfileInterface,
        model_router: ModelRouter,
        tool_interface: ToolInterface,
        goal_interface: GoalInterface,
    ) -> None:
        self._orchestrator = orchestrator
        self._model_adapter = model_adapter
        self._memory_interface = memory_interface
        self._user_profile_interface = user_profile_interface
        self._model_router = model_router
        self._tool_interface = tool_interface
        self._goal_interface = goal_interface

    async def reason(self, request: ChatRequest) -> ChatResponse:
        return await self._orchestrator.run(
            request=request,
            flow_type="request",
            humanoid_mode_override=False,
        )

    async def companion_reason(self, request: ChatRequest) -> ChatResponse:
        return await self._orchestrator.run(
            request=request,
            flow_type="companion",
            humanoid_mode_override=True,
        )

    async def stream_reason(self, request: ChatRequest) -> AsyncIterator[str]:
        async for event in self._orchestrator.stream(
            request=request,
            humanoid_mode_override=False,
        ):
            yield json.dumps(event) + "\n"

    async def companion_stream_reason(self, request: ChatRequest) -> AsyncIterator[str]:
        async for event in self._orchestrator.stream(
            request=request,
            humanoid_mode_override=True,
        ):
            yield json.dumps(event) + "\n"

    def is_model_ready(self) -> bool:
        return self._model_adapter.is_ready()

    def model_backend_effective(self) -> str:
        return self._model_adapter.effective_backend()

    def can_use_model_judge(self) -> bool:
        return self._model_adapter.can_use_model_judge()

    async def is_memory_ready(self) -> bool:
        return await self._memory_interface.is_ready()

    async def memory_entries_count(self) -> int:
        return await self._memory_interface.count_entries()

    async def is_profile_engine_ready(self) -> bool:
        return await self._user_profile_interface.is_ready()

    def is_model_router_enabled(self) -> bool:
        return self._model_router.is_enabled()

    async def is_tool_engine_ready(self) -> bool:
        return await self._tool_interface.is_ready()

    async def is_goal_engine_ready(self) -> bool:
        return await self._goal_interface.is_ready()
