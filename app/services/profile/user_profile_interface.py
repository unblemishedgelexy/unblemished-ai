from __future__ import annotations

import re

from app.core.logger import StructuredLogger
from app.services.profile.user_profile_store import UserProfileRecord, UserProfileStore
from app.utils.helpers import utc_now


class UserProfileInterface:
    def __init__(self, store: UserProfileStore, logger: StructuredLogger) -> None:
        self._store = store
        self._logger = logger

    async def get_snapshot(self, user_id: str, trace_id: str) -> UserProfileRecord:
        profile = await self._store.get_profile(user_id=user_id, trace_id=trace_id)
        self._logger.info(
            "profile.snapshot.loaded",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
        )
        return profile

    async def update_after_interaction(
        self,
        user_id: str,
        trace_id: str,
        user_input: str,
        detected_intent: str,
        reasoning_depth: int,
    ) -> UserProfileRecord:
        profile = await self._store.get_profile(user_id=user_id, trace_id=trace_id)

        updated = UserProfileRecord(
            user_id=user_id,
            preferred_tone=_infer_preferred_tone(user_input, profile.preferred_tone),
            dominant_intent_type=_infer_dominant_intent(detected_intent, profile.dominant_intent_type),
            conversation_depth_preference=_infer_depth_preference(reasoning_depth),
            emotional_baseline=_infer_emotional_baseline(user_input, profile.emotional_baseline),
            updated_at=utc_now(),
        )
        await self._store.upsert_profile(profile=updated, trace_id=trace_id)
        return updated

    async def is_ready(self) -> bool:
        return await self._store.is_ready()


def _infer_preferred_tone(user_input: str, current: str) -> str:
    lowered = user_input.lower()
    if any(token in lowered for token in ("please", "kindly", "would you", "could you")):
        return "formal"
    if any(token in lowered for token in ("bro", "bhai", "yaar", "dude")):
        return "casual"
    return current or "neutral"


def _infer_dominant_intent(new_intent: str, current: str) -> str:
    if current == "general-reasoning":
        return new_intent
    if new_intent == current:
        return new_intent
    return current


def _infer_depth_preference(reasoning_depth: int) -> str:
    if reasoning_depth <= 2:
        return "shallow"
    if reasoning_depth <= 4:
        return "balanced"
    return "deep"


def _infer_emotional_baseline(user_input: str, current: str) -> str:
    lowered = user_input.lower()
    intensity_terms = len(re.findall(r"\b(urgent|critical|blocked|asap|important|must)\b", lowered))
    punctuation_intensity = user_input.count("!") + user_input.count("??")
    if intensity_terms + punctuation_intensity >= 2:
        return "high-intensity"
    if any(token in lowered for token in ("thanks", "great", "good", "nice")):
        return "positive"
    return current or "stable"

