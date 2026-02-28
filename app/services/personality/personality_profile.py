from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.core.logger import StructuredLogger
from app.utils.helpers import utc_now


@dataclass(slots=True, frozen=True)
class PersonalityProfile:
    name: str
    speaking_style: str
    humor_style: str
    emotional_sensitivity_level: int
    attachment_style: str
    moral_boundaries: tuple[str, ...]
    flirt_intensity_range: tuple[int, int]


class PersonalityProfileService:
    def __init__(
        self,
        *,
        db_path: str,
        logger: StructuredLogger,
    ) -> None:
        self._db_path = db_path
        self._logger = logger
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def get_profile(self, *, trace_id: str, user_id: str) -> PersonalityProfile:
        await self._initialize()
        persona_name = os.getenv("PERSONA_NAME", "Kanchana").strip() or "Kanchana"
        row = await asyncio.to_thread(self._get_profile_sync, persona_name)
        if row is None:
            defaults = _default_profile(persona_name)
            await asyncio.to_thread(self._insert_profile_sync, defaults)
            self._logger.info(
                "personality.profile.created",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                persona_name=defaults.name,
            )
            return defaults

        profile = PersonalityProfile(
            name=row[0],
            speaking_style=row[1],
            humor_style=row[2],
            emotional_sensitivity_level=int(row[3]),
            attachment_style=row[4],
            moral_boundaries=_parse_boundaries(row[5]),
            flirt_intensity_range=(int(row[6]), int(row[7])),
        )
        self._logger.info(
            "personality.profile.loaded",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            persona_name=profile.name,
        )
        return profile

    async def _initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _initialize_sync(self) -> None:
        db_file = Path(self._db_path)
        if db_file.parent and str(db_file.parent) not in {".", ""}:
            db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personality_profile (
                    name TEXT PRIMARY KEY,
                    speaking_style TEXT NOT NULL,
                    humor_style TEXT NOT NULL,
                    emotional_sensitivity_level INTEGER NOT NULL,
                    attachment_style TEXT NOT NULL,
                    moral_boundaries TEXT NOT NULL,
                    flirt_intensity_min INTEGER NOT NULL,
                    flirt_intensity_max INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()

    def _get_profile_sync(
        self,
        persona_name: str,
    ) -> tuple[str, str, str, int, str, str, int, int] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT name, speaking_style, humor_style, emotional_sensitivity_level, attachment_style,
                       moral_boundaries, flirt_intensity_min, flirt_intensity_max
                FROM personality_profile
                WHERE name = ?
                LIMIT 1
                """,
                (persona_name,),
            )
            return cursor.fetchone()

    def _insert_profile_sync(self, profile: PersonalityProfile) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO personality_profile (
                    name,
                    speaking_style,
                    humor_style,
                    emotional_sensitivity_level,
                    attachment_style,
                    moral_boundaries,
                    flirt_intensity_min,
                    flirt_intensity_max,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.name,
                    profile.speaking_style,
                    profile.humor_style,
                    profile.emotional_sensitivity_level,
                    profile.attachment_style,
                    json.dumps(list(profile.moral_boundaries)),
                    profile.flirt_intensity_range[0],
                    profile.flirt_intensity_range[1],
                    utc_now().isoformat(),
                ),
            )
            conn.commit()


def _default_profile(persona_name: str) -> PersonalityProfile:
    speaking_style = os.getenv("PERSONA_SPEAKING_STYLE", "warm-structured")
    humor_style = os.getenv("PERSONA_HUMOR_STYLE", "light-playful")
    sensitivity = _to_int(os.getenv("PERSONA_EMOTIONAL_SENSITIVITY_LEVEL"), default=7, minimum=1, maximum=10)
    attachment_style = os.getenv("PERSONA_ATTACHMENT_STYLE", "secure-supportive")
    boundaries_raw = os.getenv(
        "PERSONA_MORAL_BOUNDARIES",
        "no-harm,no-coercion,no-deception,respect-consent",
    )
    boundaries = tuple(item.strip() for item in boundaries_raw.split(",") if item.strip())
    flirt_min = _to_int(os.getenv("PERSONA_FLIRT_INTENSITY_MIN"), default=0, minimum=0, maximum=10)
    flirt_max = _to_int(os.getenv("PERSONA_FLIRT_INTENSITY_MAX"), default=3, minimum=0, maximum=10)
    if flirt_min > flirt_max:
        flirt_min, flirt_max = flirt_max, flirt_min
    return PersonalityProfile(
        name=persona_name,
        speaking_style=speaking_style,
        humor_style=humor_style,
        emotional_sensitivity_level=sensitivity,
        attachment_style=attachment_style,
        moral_boundaries=boundaries,
        flirt_intensity_range=(flirt_min, flirt_max),
    )


def _parse_boundaries(raw: str) -> tuple[str, ...]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = []
    if not isinstance(payload, list):
        return tuple()
    return tuple(item for item in payload if isinstance(item, str) and item.strip())


def _to_int(raw: str | None, *, default: int, minimum: int, maximum: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))

