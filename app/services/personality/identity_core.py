from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.core.logger import StructuredLogger
from app.services.brain.planning_engine import PlanningBundle
from app.services.personality.personality_profile import PersonalityProfileService


class IdentityCore:
    def __init__(
        self,
        *,
        profile_service: PersonalityProfileService,
        logger: StructuredLogger,
    ) -> None:
        self._profile_service = profile_service
        self._logger = logger

    async def build_planning_identity_context(
        self,
        *,
        trace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        profile = await self._profile_service.get_profile(trace_id=trace_id, user_id=user_id)
        return {
            "personality_consistency": True,
            "persona_name": profile.name,
            "speaking_style": profile.speaking_style,
            "attachment_style": profile.attachment_style,
        }

    async def inject_identity_into_planning(
        self,
        *,
        planning: PlanningBundle,
        trace_id: str,
        user_id: str,
        intent: str,
    ) -> PlanningBundle:
        profile = await self._profile_service.get_profile(trace_id=trace_id, user_id=user_id)
        persona_tag = (
            f"identity:{profile.name}|style:{profile.speaking_style}|humor:{profile.humor_style}"
            f"|attachment:{profile.attachment_style}|intent:{intent}"
        )
        reasoning_plan = list(planning.reasoning_plan)
        reasoning_plan.append(persona_tag)

        execution_plan = list(planning.execution_plan)
        execution_plan.append(
            {
                "step": f"identity-consistency-check:{profile.name}",
                "requires_tool": False,
                "required_tool_name": None,
            },
        )
        self._logger.info(
            "identity_core.planning_injected",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            persona_name=profile.name,
        )
        return replace(planning, reasoning_plan=reasoning_plan, execution_plan=execution_plan)

    async def apply_personality_filter(
        self,
        *,
        final_answer: str,
        trace_id: str,
        user_id: str,
        fallback_triggered: bool,
    ) -> str:
        # Safety takes precedence over persona style shaping.
        if fallback_triggered:
            return final_answer

        profile = await self._profile_service.get_profile(trace_id=trace_id, user_id=user_id)
        normalized = final_answer.strip()
        lowered = normalized.lower()

        disallowed_markers = (
            "as an ai language model",
            "i am just an ai",
            "i cannot have personality",
        )
        if any(marker in lowered for marker in disallowed_markers):
            normalized = (
                "Structured Reasoning Output\n"
                f"- Persona: {profile.name}\n"
                "- Response adjusted to preserve consistent humanoid identity and safety boundaries.\n"
                f"- Speaking Style: {profile.speaking_style}\n"
                f"- Humor Style: {profile.humor_style}\n"
                f"- Attachment Style: {profile.attachment_style}"
            )

        if profile.flirt_intensity_range[1] <= 1 and "flirt" in lowered:
            normalized += "\n- Boundary Notice: Interaction kept respectful and non-flirtatious."

        self._logger.info(
            "identity_core.response_filtered",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            persona_name=profile.name,
            boundary_count=len(profile.moral_boundaries),
        )
        return normalized
