from __future__ import annotations

from dataclasses import dataclass

from app.core.logger import StructuredLogger


@dataclass(slots=True)
class ModelRouteDecision:
    model_name: str
    reason: str
    routing_enabled: bool


class ModelRouter:
    def __init__(self, logger: StructuredLogger, enabled: bool) -> None:
        self._logger = logger
        self._enabled = enabled

    async def route(
        self,
        *,
        trace_id: str,
        user_id: str,
        intent: str,
        reasoning_mode: str,
        depth_preference: str,
        complexity_score: float,
    ) -> ModelRouteDecision:
        if not self._enabled:
            model_name = _default_model(reasoning_mode=reasoning_mode)
            decision = ModelRouteDecision(
                model_name=model_name,
                reason="routing_disabled",
                routing_enabled=False,
            )
            self._log(trace_id=trace_id, user_id=user_id, intent=intent, complexity_score=complexity_score, decision=decision)
            return decision

        model_name = _default_model(reasoning_mode=reasoning_mode)
        reason = "default_mode_mapping"

        if reasoning_mode == "deep" or complexity_score >= 0.78:
            model_name = "deep_model"
            reason = "high_complexity_or_deep_mode"
        elif depth_preference == "deep" and complexity_score >= 0.62:
            model_name = "deep_model"
            reason = "profile_prefers_depth"
        elif intent == "solution-design" and complexity_score >= 0.45:
            model_name = "creative_model"
            reason = "solution_design_path"
        elif intent in {"question-answering", "troubleshooting"} and complexity_score <= 0.55:
            model_name = "fast_model"
            reason = "low_latency_path"

        decision = ModelRouteDecision(
            model_name=model_name,
            reason=reason,
            routing_enabled=True,
        )
        self._log(trace_id=trace_id, user_id=user_id, intent=intent, complexity_score=complexity_score, decision=decision)
        return decision

    def is_enabled(self) -> bool:
        return self._enabled

    def _log(
        self,
        *,
        trace_id: str,
        user_id: str,
        intent: str,
        complexity_score: float,
        decision: ModelRouteDecision,
    ) -> None:
        self._logger.info(
            "router.model.selected",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            intent=intent,
            complexity_score=round(complexity_score, 4),
            model_name=decision.model_name,
            reason=decision.reason,
            routing_enabled=decision.routing_enabled,
        )


def _default_model(reasoning_mode: str) -> str:
    if reasoning_mode == "deep":
        return "deep_model"
    if reasoning_mode == "fast":
        return "fast_model"
    return "creative_model"

