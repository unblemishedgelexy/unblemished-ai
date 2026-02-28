from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.memory.memory_retriever import RetrievedMemory
from app.services.router.model_router import ModelRouteDecision

MAX_TOOL_CALLS_PER_REQUEST = 2


@dataclass(slots=True)
class PlanningBundle:
    complexity_score: float
    reasoning_plan: list[str]
    execution_plan: list[dict[str, Any]]


class PlanningEngine:
    def build(
        self,
        *,
        user_input: str,
        context: dict[str, Any],
        intent: str,
        retrieved_memory_count: int,
        reasoning_depth: int,
        humanoid_mode_enabled: bool = False,
        identity_context: dict[str, Any] | None = None,
        relational_context: dict[str, Any] | None = None,
        emotional_context: dict[str, Any] | None = None,
    ) -> PlanningBundle:
        complexity_score = self._estimate_complexity(
            user_input=user_input,
            context=context,
            intent=intent,
            retrieved_memory_count=retrieved_memory_count,
        )
        reasoning_plan = self._build_reasoning_plan(
            intent=intent,
            complexity_score=complexity_score,
            retrieved_memory_count=retrieved_memory_count,
            reasoning_depth=reasoning_depth,
            humanoid_mode_enabled=humanoid_mode_enabled,
            identity_context=identity_context,
            relational_context=relational_context,
            emotional_context=emotional_context,
        )
        execution_plan = self._build_execution_plan(
            user_input=user_input,
            intent=intent,
            complexity_score=complexity_score,
            humanoid_mode_enabled=humanoid_mode_enabled,
        )
        return PlanningBundle(
            complexity_score=complexity_score,
            reasoning_plan=reasoning_plan,
            execution_plan=execution_plan,
        )

    def build_reasoning_steps(
        self,
        *,
        trace_id: str,
        user_id: str,
        input_text: str,
        intent: str,
        context: dict[str, Any],
        reasoning_mode: str,
        depth: int,
        max_tokens: int,
        temperature: float,
        memory_top_k: int,
        retrieved_memories: list[RetrievedMemory],
        complexity_score: float,
        reasoning_plan: list[str],
        execution_plan: list[dict[str, Any]],
        route_decision: ModelRouteDecision,
    ) -> list[dict[str, Any]]:
        steps = [
            {
                "name": "intent-analysis",
                "detail": f"Inferred intent: {intent}.",
                "metadata": {
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "prompt_length": len(input_text),
                    "reasoning_mode": reasoning_mode,
                },
            },
            {
                "name": "planning-scratchpad",
                "detail": "Prepared internal reasoning plan and complexity estimate.",
                "metadata": {
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "complexity_score": complexity_score,
                    "reasoning_plan": reasoning_plan,
                    "step_plan": execution_plan,
                },
            },
            {
                "name": "memory-retrieval",
                "detail": "Retrieved user-scoped relevant memory context.",
                "metadata": {
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "top_k": memory_top_k,
                    "retrieval_count": len(retrieved_memories),
                    "memory_ids": [m.memory_id for m in retrieved_memories],
                },
            },
            {
                "name": "context-binding",
                "detail": "Bound request context signals into reasoning state.",
                "metadata": {
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "context_keys": sorted(context.keys()),
                    "reasoning_mode": reasoning_mode,
                },
            },
            {
                "name": "model-routing",
                "detail": "Selected routed model for this request.",
                "metadata": {
                    "trace_id": trace_id,
                    "user_id": user_id,
                    "routed_model": route_decision.model_name,
                    "routing_reason": route_decision.reason,
                    "routing_enabled": route_decision.routing_enabled,
                },
            },
        ]
        for idx in range(depth):
            steps.append(
                {
                    "name": f"reasoning-pass-{idx + 1}",
                    "detail": "Expanded structured reasoning depth for the active mode.",
                    "metadata": {
                        "trace_id": trace_id,
                        "user_id": user_id,
                        "pass_index": idx + 1,
                        "target_depth": depth,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "reasoning_mode": reasoning_mode,
                    },
                },
            )
        return steps

    def resolve_reasoning_mode(
        self,
        *,
        current_mode: str,
        complexity_score: float,
        profile_depth_preference: str,
    ) -> str:
        if complexity_score > 0.7 and profile_depth_preference == "deep" and current_mode != "deep":
            return "deep"
        return current_mode

    def _estimate_complexity(
        self,
        *,
        user_input: str,
        context: dict[str, Any],
        intent: str,
        retrieved_memory_count: int,
    ) -> float:
        tokens = len(user_input.split())
        token_factor = min(tokens / 45.0, 1.0)
        context_factor = min(len(context) / 8.0, 1.0)
        memory_factor = min(retrieved_memory_count / 5.0, 1.0)
        intent_factor = {
            "question-answering": 0.25,
            "general-reasoning": 0.35,
            "solution-design": 0.5,
            "troubleshooting": 0.55,
        }.get(intent, 0.35)
        structure = sum(
            user_input.lower().count(token)
            for token in (" and ", " or ", " then ", " because ", " while ", " compare ")
        )
        score = (
            0.35 * token_factor
            + 0.2 * context_factor
            + 0.2 * intent_factor
            + 0.15 * min(structure / 4.0, 1.0)
            + 0.1 * memory_factor
        )
        return round(max(0.0, min(score, 1.0)), 4)

    def _build_reasoning_plan(
        self,
        *,
        intent: str,
        complexity_score: float,
        retrieved_memory_count: int,
        reasoning_depth: int,
        humanoid_mode_enabled: bool,
        identity_context: dict[str, Any] | None,
        relational_context: dict[str, Any] | None,
        emotional_context: dict[str, Any] | None,
    ) -> list[str]:
        plan: list[str] = []
        if humanoid_mode_enabled:
            plan.extend(
                self._build_humanoid_prefix_plan(
                    identity_context=identity_context,
                    relational_context=relational_context,
                    emotional_context=emotional_context,
                ),
            )

        plan.extend(
            [
            f"classify-intent:{intent}",
            f"bind-memory:{retrieved_memory_count}",
            f"reasoning-depth:{reasoning_depth}",
            ],
        )
        plan.append("enable-deep-analysis" if complexity_score > 0.7 else "keep-standard-analysis")
        plan.append("run-reflection-and-evaluation")
        return plan

    def _build_execution_plan(
        self,
        *,
        user_input: str,
        intent: str,
        complexity_score: float,
        humanoid_mode_enabled: bool,
    ) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = [
            {"step": f"analyze-intent:{intent}", "requires_tool": False, "required_tool_name": None},
            {"step": "bind-user-memory-context", "requires_tool": False, "required_tool_name": None},
        ]
        if humanoid_mode_enabled:
            plan.append(
                {
                    "step": "apply-humanoid-style-continuity",
                    "requires_tool": False,
                    "required_tool_name": None,
                },
            )
        requested = self._extract_explicit_tool_names(user_input) or (
            [self._infer_tool_name(user_input)] if self._infer_tool_name(user_input) else []
        )
        for name in requested[:MAX_TOOL_CALLS_PER_REQUEST]:
            plan.append({"step": f"execute-tool:{name}", "requires_tool": True, "required_tool_name": name})
        if complexity_score > 0.7:
            plan.append(
                {
                    "step": "perform-deep-reasoning-review",
                    "requires_tool": False,
                    "required_tool_name": None,
                },
            )
        plan.append({"step": "finalize-structured-answer", "requires_tool": False, "required_tool_name": None})
        return plan

    def _extract_explicit_tool_names(self, user_input: str) -> list[str]:
        match = re.search(r"tool:\s*([a-zA-Z0-9_,.\-\s]+)", user_input, flags=re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1).strip()
        if not raw:
            return []
        items = [x.strip() for x in raw.split(",")] if "," in raw else [raw.split()[0]]
        return [item.strip(".") for item in items if item][:MAX_TOOL_CALLS_PER_REQUEST]

    def _infer_tool_name(self, user_input: str) -> str | None:
        lowered = user_input.lower()
        if "keyword" in lowered or ("extract" in lowered and "term" in lowered):
            return "keyword_extract"
        if "priority" in lowered or "prioritize" in lowered:
            return "priority_estimator"
        if "context" in lowered and ("summary" in lowered or "digest" in lowered):
            return "context_digest"
        return None

    def _build_humanoid_prefix_plan(
        self,
        *,
        identity_context: dict[str, Any] | None,
        relational_context: dict[str, Any] | None,
        emotional_context: dict[str, Any] | None,
    ) -> list[str]:
        plan = ["humanoid-mode:enabled"]
        if identity_context is not None and bool(identity_context.get("personality_consistency", False)):
            plan.append("personality-consistency:enabled")
        if relational_context is not None:
            relational_depth = str(relational_context.get("relational_depth", "growing"))
            plan.append(f"relational-depth:{relational_depth}")
        if emotional_context is not None and bool(emotional_context.get("continuity_enabled", False)):
            plan.append("conversational-continuity:enabled")
        return plan
