from __future__ import annotations

from typing import Any

from app.services.goals.goal_interface import GoalInterface, GoalSnapshot
from app.services.memory.memory_interface import MemoryInterface
from app.services.tools.tool_interface import ToolInterface


class ExecutionController:
    def __init__(
        self,
        *,
        tool_interface: ToolInterface,
        goal_interface: GoalInterface,
        memory_interface: MemoryInterface,
        max_tool_calls: int = 2,
    ) -> None:
        self._tool_interface = tool_interface
        self._goal_interface = goal_interface
        self._memory_interface = memory_interface
        self._max_tool_calls = max_tool_calls

    async def execute_tools(
        self,
        *,
        execution_plan: list[dict[str, Any]],
        user_input: str,
        context: dict[str, Any],
        reflection: dict[str, Any],
        goal_snapshot: GoalSnapshot,
        trace_id: str,
        user_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        tool_requests = self._build_tool_requests(
            execution_plan=execution_plan,
            user_input=user_input,
            context=context,
            reflection_final_answer=reflection["final_answer"],
            goal_snapshot=goal_snapshot,
        )
        if not tool_requests:
            return reflection, []

        invocations = await self._tool_interface.invoke_many(
            invocations=tool_requests,
            trace_id=trace_id,
            user_id=user_id,
            max_calls=self._max_tool_calls,
        )
        for item in invocations:
            await self._memory_interface.schedule_action_store(
                user_id=user_id,
                trace_id=trace_id,
                action_type=f"tool:{item['tool_name']}",
                action_result_summary=item["result_summary"],
            )

        updated_reflection = self._inject_tool_results(reflection, invocations)
        return updated_reflection, invocations

    async def get_goal_snapshot(self, *, user_id: str, trace_id: str) -> GoalSnapshot:
        return await self._goal_interface.get_snapshot(user_id=user_id, trace_id=trace_id)

    async def update_goal(
        self,
        *,
        user_id: str,
        trace_id: str,
        user_input: str,
        execution_plan: list[dict[str, Any]],
        tool_invocations: list[dict[str, Any]],
        fallback_triggered: bool,
    ) -> GoalSnapshot:
        return await self._goal_interface.update_after_interaction(
            user_id=user_id,
            trace_id=trace_id,
            user_input=user_input,
            execution_plan=execution_plan,
            tool_invocations=tool_invocations,
            fallback_triggered=fallback_triggered,
        )

    def _build_tool_requests(
        self,
        *,
        execution_plan: list[dict[str, Any]],
        user_input: str,
        context: dict[str, Any],
        reflection_final_answer: str,
        goal_snapshot: GoalSnapshot,
    ) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        for step in execution_plan:
            if not step.get("requires_tool"):
                continue
            tool_name = step.get("required_tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            requests.append(
                {
                    "tool_name": tool_name,
                    "arguments": self._build_tool_args(
                        tool_name=tool_name,
                        user_input=user_input,
                        context=context,
                        reflection_final_answer=reflection_final_answer,
                        goal_snapshot=goal_snapshot,
                    ),
                },
            )
            if len(requests) >= self._max_tool_calls:
                break
        return requests

    def _build_tool_args(
        self,
        *,
        tool_name: str,
        user_input: str,
        context: dict[str, Any],
        reflection_final_answer: str,
        goal_snapshot: GoalSnapshot,
    ) -> dict[str, Any]:
        if tool_name == "keyword_extract":
            return {"text": reflection_final_answer or user_input, "max_keywords": 6}
        if tool_name == "context_digest":
            return {"context": context}
        if tool_name == "priority_estimator":
            hint = context.get("priority") if isinstance(context.get("priority"), str) else None
            goal = goal_snapshot.active_goal if goal_snapshot.active_goal != "none" else user_input
            return {"goal": goal, "priority_hint": hint}
        return {}

    def _inject_tool_results(
        self,
        reflection: dict[str, Any],
        invocations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        success = [item for item in invocations if item.get("success")]
        failed = [item for item in invocations if not item.get("success")]

        updated = dict(reflection)
        strengths = list(updated.get("strengths", []))
        risks = list(updated.get("risks", []))
        if success:
            lines = "\n".join(f"- {item['tool_name']}: {item['result_summary']}" for item in success)
            updated["final_answer"] = f"{updated['final_answer']}\n- Tool Outputs:\n{lines}"
            updated["confidence"] = _bounded_score(float(updated.get("confidence", 0.7)) + 0.03)
            strengths.append("Tool evidence integrated into final answer.")
        if failed:
            names = ", ".join(item["tool_name"] for item in failed)
            risks.append(f"Some tool calls were rejected or failed: {names}.")
        updated["strengths"] = strengths
        updated["risks"] = risks
        return updated


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(value, 1.0)), 2)
