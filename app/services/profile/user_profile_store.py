from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.core.logger import StructuredLogger
from app.repositories.profile_repository import ProfileRepository, ProfileRow
from app.utils.helpers import utc_now


@dataclass(slots=True)
class UserProfileRecord:
    user_id: str
    preferred_tone: str
    dominant_intent_type: str
    conversation_depth_preference: str
    emotional_baseline: str
    updated_at: datetime


class UserProfileStore:
    def __init__(self, repository: ProfileRepository, logger: StructuredLogger) -> None:
        self._repository = repository
        self._logger = logger

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def get_profile(
        self,
        user_id: str,
        trace_id: str,
    ) -> UserProfileRecord:
        await self.initialize()
        row = await self._repository.get_profile(user_id=user_id)
        if row is None:
            profile = UserProfileRecord(
                user_id=user_id,
                preferred_tone="neutral",
                dominant_intent_type="general-reasoning",
                conversation_depth_preference="balanced",
                emotional_baseline="stable",
                updated_at=utc_now(),
            )
            await self.upsert_profile(profile=profile, trace_id=trace_id)
            return profile

        return UserProfileRecord(
            user_id=row.user_id,
            preferred_tone=row.preferred_tone,
            dominant_intent_type=row.dominant_intent_type,
            conversation_depth_preference=row.conversation_depth_preference,
            emotional_baseline=row.emotional_baseline,
            updated_at=row.updated_at,
        )

    async def upsert_profile(
        self,
        profile: UserProfileRecord,
        trace_id: str,
    ) -> None:
        await self.initialize()
        await self._repository.upsert_profile(
            ProfileRow(
                user_id=profile.user_id,
                preferred_tone=profile.preferred_tone,
                dominant_intent_type=profile.dominant_intent_type,
                conversation_depth_preference=profile.conversation_depth_preference,
                emotional_baseline=profile.emotional_baseline,
                updated_at=profile.updated_at,
            ),
        )
        self._logger.info(
            "profile.store.upsert.completed",
            trace_id=trace_id,
            user_id=profile.user_id,
            memory_id="n/a",
            retrieval_count=0,
            preferred_tone=profile.preferred_tone,
            dominant_intent_type=profile.dominant_intent_type,
        )

    async def is_ready(self) -> bool:
        return await self._repository.is_ready()

