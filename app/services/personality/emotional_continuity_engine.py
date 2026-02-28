from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logger import StructuredLogger
from app.utils.helpers import utc_now


@dataclass(slots=True, frozen=True)
class EmotionalState:
    positive_momentum: float
    neutral_state: float
    tension_level: float
    affection_intensity: float
    window_size: int
    updated_at: str

    def as_tone_controller_object(self) -> dict[str, Any]:
        return {
            "positive_momentum": round(self.positive_momentum, 4),
            "neutral_state": round(self.neutral_state, 4),
            "tension_level": round(self.tension_level, 4),
            "affection_intensity": round(self.affection_intensity, 4),
            "tone_directive": _tone_directive(self),
            "simulation_scope": "style_only",
        }


class EmotionalContinuityEngine:
    """
    Tracks conversational emotional momentum for style adaptation only.
    This engine does not model consciousness and must not alter factual logic.
    """

    def __init__(
        self,
        *,
        db_path: str,
        logger: StructuredLogger,
        window_size: int = 8,
        max_step_delta: float = 0.12,
    ) -> None:
        self._db_path = db_path
        self._logger = logger
        self._window_size = max(3, min(32, window_size))
        self._max_step_delta = max(0.03, min(0.25, max_step_delta))
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True
            self._logger.info(
                "emotion.continuity.initialized",
                trace_id="system",
                user_id="system",
                memory_id="n/a",
                retrieval_count=0,
                window_size=self._window_size,
                max_step_delta=self._max_step_delta,
            )

    async def get_emotional_state(self, *, user_id: str, trace_id: str) -> EmotionalState:
        await self.initialize()
        row = await asyncio.to_thread(self._get_state_row_sync, user_id)
        if row is None:
            baseline = _baseline_state(window_size=self._window_size)
            await asyncio.to_thread(self._upsert_state_sync, user_id, baseline)
            self._logger.info(
                "emotion.state.created",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                tone_directive=_tone_directive(baseline),
            )
            return baseline

        return EmotionalState(
            positive_momentum=float(row[0]),
            neutral_state=float(row[1]),
            tension_level=float(row[2]),
            affection_intensity=float(row[3]),
            window_size=int(row[4]),
            updated_at=row[5],
        )

    async def update_after_interaction(
        self,
        *,
        user_id: str,
        trace_id: str,
        user_input: str,
        assistant_output: str,
        fallback_triggered: bool,
    ) -> EmotionalState:
        await self.initialize()
        previous = await self.get_emotional_state(user_id=user_id, trace_id=trace_id)

        immediate = _extract_signal(
            user_input=user_input,
            assistant_output=assistant_output,
            fallback_triggered=fallback_triggered,
        )
        await asyncio.to_thread(self._insert_signal_sync, user_id, trace_id, immediate)
        await asyncio.to_thread(self._trim_history_sync, user_id, max(self._window_size * 20, 120))

        window = await asyncio.to_thread(self._fetch_recent_signals_sync, user_id, self._window_size)
        target = _aggregate_targets(window, fallback_triggered=fallback_triggered)

        next_state = _smooth_transition(
            previous=previous,
            target=target,
            max_step_delta=self._max_step_delta,
            window_size=self._window_size,
        )
        await asyncio.to_thread(self._upsert_state_sync, user_id, next_state)

        self._logger.info(
            "emotion.state.updated",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=len(window),
            positive_momentum=round(next_state.positive_momentum, 4),
            neutral_state=round(next_state.neutral_state, 4),
            tension_level=round(next_state.tension_level, 4),
            affection_intensity=round(next_state.affection_intensity, 4),
            tone_directive=_tone_directive(next_state),
        )
        return next_state

    async def get_tone_controller_state(self, *, user_id: str, trace_id: str) -> dict[str, Any]:
        state = await self.get_emotional_state(user_id=user_id, trace_id=trace_id)
        return {"emotional_state": state.as_tone_controller_object()}

    def planning_context_from_tone_state(self, tone_controller_state: dict[str, Any]) -> dict[str, Any]:
        emotional_state = tone_controller_state.get("emotional_state", {})
        tone_directive = str(emotional_state.get("tone_directive", "steady-neutral"))
        return {
            "continuity_enabled": True,
            "tone_directive": tone_directive,
        }

    async def apply_style_influence(
        self,
        *,
        final_answer: str,
        tone_controller_state: dict[str, Any],
    ) -> str:
        emotional_state = tone_controller_state.get("emotional_state", {})
        directive = str(emotional_state.get("tone_directive", "steady-neutral"))
        if directive == "deescalate-calm":
            suffix = "\n- Conversational Continuity: calm and de-escalating style applied."
        elif directive == "warm-supportive":
            suffix = "\n- Conversational Continuity: warm and supportive style maintained."
        elif directive == "steady-neutral":
            suffix = "\n- Conversational Continuity: stable and neutral style maintained."
        else:
            suffix = "\n- Conversational Continuity: balanced empathic style maintained."
        return f"{final_answer}{suffix}"

    async def is_ready(self) -> bool:
        try:
            await self.initialize()
            return True
        except Exception:
            return False

    def _initialize_sync(self) -> None:
        db_file = Path(self._db_path)
        if db_file.parent and str(db_file.parent) not in {".", ""}:
            db_file.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotional_interaction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    positive_signal REAL NOT NULL,
                    tension_signal REAL NOT NULL,
                    affection_signal REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_emotional_interaction_user_time
                ON emotional_interaction_log (user_id, created_at DESC)
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotional_state_snapshot (
                    user_id TEXT PRIMARY KEY,
                    positive_momentum REAL NOT NULL,
                    neutral_state REAL NOT NULL,
                    tension_level REAL NOT NULL,
                    affection_intensity REAL NOT NULL,
                    window_size INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()

    def _get_state_row_sync(self, user_id: str) -> tuple[float, float, float, float, int, str] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    positive_momentum,
                    neutral_state,
                    tension_level,
                    affection_intensity,
                    window_size,
                    updated_at
                FROM emotional_state_snapshot
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,),
            )
            return cursor.fetchone()

    def _insert_signal_sync(
        self,
        user_id: str,
        trace_id: str,
        signal: dict[str, float],
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO emotional_interaction_log (
                    user_id,
                    trace_id,
                    positive_signal,
                    tension_signal,
                    affection_signal,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    trace_id,
                    signal["positive"],
                    signal["tension"],
                    signal["affection"],
                    utc_now().isoformat(),
                ),
            )
            conn.commit()

    def _fetch_recent_signals_sync(self, user_id: str, limit: int) -> list[tuple[float, float, float]]:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT positive_signal, tension_signal, affection_signal
                FROM emotional_interaction_log
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return cursor.fetchall()

    def _trim_history_sync(self, user_id: str, keep_limit: int) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                DELETE FROM emotional_interaction_log
                WHERE id IN (
                    SELECT id
                    FROM emotional_interaction_log
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (user_id, keep_limit),
            )
            conn.commit()

    def _upsert_state_sync(self, user_id: str, state: EmotionalState) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO emotional_state_snapshot (
                    user_id,
                    positive_momentum,
                    neutral_state,
                    tension_level,
                    affection_intensity,
                    window_size,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    positive_momentum = excluded.positive_momentum,
                    neutral_state = excluded.neutral_state,
                    tension_level = excluded.tension_level,
                    affection_intensity = excluded.affection_intensity,
                    window_size = excluded.window_size,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    state.positive_momentum,
                    state.neutral_state,
                    state.tension_level,
                    state.affection_intensity,
                    state.window_size,
                    state.updated_at,
                ),
            )
            conn.commit()


def _baseline_state(*, window_size: int) -> EmotionalState:
    return EmotionalState(
        positive_momentum=0.45,
        neutral_state=0.62,
        tension_level=0.18,
        affection_intensity=0.32,
        window_size=window_size,
        updated_at=utc_now().isoformat(),
    )


def _extract_signal(
    *,
    user_input: str,
    assistant_output: str,
    fallback_triggered: bool,
) -> dict[str, float]:
    user_tokens = _tokenize(user_input)
    assistant_tokens = _tokenize(assistant_output)

    positive = _keyword_score(user_tokens, _POSITIVE_LEXICON) + (_keyword_score(assistant_tokens, _POSITIVE_LEXICON) * 0.25)
    tension = _keyword_score(user_tokens, _TENSION_LEXICON)
    affection = _keyword_score(user_tokens, _AFFECTION_LEXICON) + (_keyword_score(assistant_tokens, _AFFECTION_LEXICON) * 0.15)

    if fallback_triggered:
        positive -= 0.12
        tension += 0.12
        affection -= 0.06

    return {
        "positive": _clamp01(positive),
        "tension": _clamp01(tension),
        "affection": _clamp01(affection),
    }


def _aggregate_targets(
    rows: list[tuple[float, float, float]],
    *,
    fallback_triggered: bool,
) -> dict[str, float]:
    if not rows:
        return {"positive": 0.45, "tension": 0.18, "affection": 0.32}

    count = float(len(rows))
    avg_positive = sum(item[0] for item in rows) / count
    avg_tension = sum(item[1] for item in rows) / count
    avg_affection = sum(item[2] for item in rows) / count

    trend_boost = 0.0
    if len(rows) > 1:
        newest = rows[0]
        oldest = rows[-1]
        trend_boost = (newest[0] - oldest[0]) * 0.2
        trend_boost = max(-0.08, min(0.08, trend_boost))

    positive = _clamp01(avg_positive + trend_boost)
    tension = _clamp01(avg_tension - (trend_boost * 0.4))
    affection = _clamp01(avg_affection + (trend_boost * 0.25))

    if fallback_triggered:
        positive = _clamp01(positive - 0.08)
        tension = _clamp01(tension + 0.08)
        affection = _clamp01(affection - 0.04)

    return {"positive": positive, "tension": tension, "affection": affection}


def _smooth_transition(
    *,
    previous: EmotionalState,
    target: dict[str, float],
    max_step_delta: float,
    window_size: int,
) -> EmotionalState:
    positive = _smooth_value(previous.positive_momentum, target["positive"], max_step_delta)
    tension = _smooth_value(previous.tension_level, target["tension"], max_step_delta)
    affection = _smooth_value(previous.affection_intensity, target["affection"], max_step_delta * 0.85)

    neutral_target = _neutral_target(positive=positive, tension=tension, affection=affection)
    neutral = _smooth_value(previous.neutral_state, neutral_target, max_step_delta)

    return EmotionalState(
        positive_momentum=positive,
        neutral_state=neutral,
        tension_level=tension,
        affection_intensity=affection,
        window_size=window_size,
        updated_at=utc_now().isoformat(),
    )


def _neutral_target(*, positive: float, tension: float, affection: float) -> float:
    balance_gap = abs(positive - tension)
    neutral = 1.0 - min(1.0, (balance_gap * 1.2) + (affection * 0.35))
    return _clamp01(neutral)


def _smooth_value(previous: float, target: float, max_delta: float) -> float:
    delta = target - previous
    if delta > max_delta:
        return _clamp01(previous + max_delta)
    if delta < -max_delta:
        return _clamp01(previous - max_delta)
    return _clamp01(target)


def _keyword_score(tokens: list[str], lexicon: dict[str, float]) -> float:
    if not tokens:
        return 0.0
    score = 0.0
    for token in tokens:
        score += lexicon.get(token, 0.0)
    normalized = score / max(1.0, len(tokens) * 0.75)
    return _clamp01(normalized)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z']+", text.lower()) if token]


def _tone_directive(state: EmotionalState) -> str:
    if state.tension_level >= 0.65:
        return "deescalate-calm"
    if state.affection_intensity >= 0.62 and state.positive_momentum >= 0.55:
        return "warm-supportive"
    if state.neutral_state >= 0.6:
        return "steady-neutral"
    return "balanced-empathic"


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


_POSITIVE_LEXICON = {
    "thanks": 0.45,
    "thank": 0.35,
    "great": 0.55,
    "good": 0.35,
    "nice": 0.35,
    "happy": 0.6,
    "excited": 0.65,
    "awesome": 0.7,
    "love": 0.7,
    "appreciate": 0.6,
}

_TENSION_LEXICON = {
    "angry": 0.75,
    "frustrated": 0.7,
    "upset": 0.65,
    "sad": 0.45,
    "hurt": 0.7,
    "confused": 0.4,
    "hate": 0.8,
    "annoyed": 0.65,
    "stress": 0.6,
    "tired": 0.35,
}

_AFFECTION_LEXICON = {
    "dear": 0.55,
    "care": 0.45,
    "miss": 0.55,
    "close": 0.5,
    "bond": 0.55,
    "hug": 0.6,
    "sweet": 0.5,
    "kind": 0.35,
    "beloved": 0.75,
    "cherish": 0.75,
}
