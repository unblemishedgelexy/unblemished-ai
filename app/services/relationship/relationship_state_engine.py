from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.core.logger import StructuredLogger
from app.services.brain.planning_engine import PlanningBundle
from app.services.privacy.data_sanitizer import sanitize_text
from app.services.relationship.relationship_memory_store import RelationshipMemory, RelationshipMemoryStore
from app.utils.helpers import utc_now


@dataclass(slots=True, frozen=True)
class RelationshipState:
    trust_level: int
    intimacy_level: int
    emotional_closeness: str
    conversation_warmth: str


class RelationshipStateEngine:
    def __init__(
        self,
        *,
        db_path: str,
        logger: StructuredLogger,
        memory_store: RelationshipMemoryStore,
        relationship_memory_text_enabled: bool = False,
        privacy_redaction_enabled: bool = True,
    ) -> None:
        self._db_path = db_path
        self._logger = logger
        self._memory_store = memory_store
        self._relationship_memory_text_enabled = relationship_memory_text_enabled
        self._privacy_redaction_enabled = privacy_redaction_enabled
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def get_state(self, *, user_id: str, trace_id: str) -> RelationshipState:
        await self._initialize()
        memory = await self._memory_store.get_or_create(user_id=user_id, trace_id=trace_id)
        row = await asyncio.to_thread(self._get_state_row_sync, user_id)
        if row is None:
            state = RelationshipState(
                trust_level=40,
                intimacy_level=25,
                emotional_closeness="growing",
                conversation_warmth="friendly",
            )
            await asyncio.to_thread(self._upsert_state_sync, user_id, state)
            self._logger.info(
                "relationship.state.created",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                first_interaction=memory.first_interaction_timestamp,
            )
            return state
        return RelationshipState(
            trust_level=int(row[0]),
            intimacy_level=int(row[1]),
            emotional_closeness=row[2],
            conversation_warmth=row[3],
        )

    def inject_into_planning(
        self,
        *,
        planning: PlanningBundle,
        relationship_state: RelationshipState,
    ) -> PlanningBundle:
        relation_tag = (
            "relationship-state:"
            f"trust={relationship_state.trust_level},"
            f"intimacy={relationship_state.intimacy_level},"
            f"warmth={relationship_state.conversation_warmth}"
        )
        reasoning_plan = list(planning.reasoning_plan)
        reasoning_plan.append(relation_tag)

        execution_plan = list(planning.execution_plan)
        execution_plan.append(
            {
                "step": "tone-calibration-from-relationship-state",
                "requires_tool": False,
                "required_tool_name": None,
            },
        )
        return replace(planning, reasoning_plan=reasoning_plan, execution_plan=execution_plan)

    def tone_hint(self, state: RelationshipState) -> dict[str, str]:
        return {
            "relationship_emotional_closeness": state.emotional_closeness,
            "relationship_conversation_warmth": state.conversation_warmth,
        }

    def planning_context(self, state: RelationshipState) -> dict[str, str]:
        if state.intimacy_level >= 75:
            relational_depth = "deep"
        elif state.intimacy_level >= 45:
            relational_depth = "close"
        else:
            relational_depth = "growing"
        return {
            "relational_depth": relational_depth,
            "conversation_warmth": state.conversation_warmth,
        }

    async def apply_tone_influence(
        self,
        *,
        final_answer: str,
        relationship_state: RelationshipState,
    ) -> str:
        suffix = ""
        if relationship_state.conversation_warmth == "warm":
            suffix = "\n- Tone: Warm and supportive."
        elif relationship_state.conversation_warmth == "neutral":
            suffix = "\n- Tone: Clear and respectful."
        if not suffix:
            return final_answer
        return f"{final_answer}{suffix}"

    async def update_after_interaction(
        self,
        *,
        user_id: str,
        trace_id: str,
        user_input: str,
        final_answer: str,
        fallback_triggered: bool,
    ) -> RelationshipState:
        state = await self.get_state(user_id=user_id, trace_id=trace_id)
        memory = await self._memory_store.get_or_create(user_id=user_id, trace_id=trace_id)

        trust_delta = 1
        intimacy_delta = 1
        if fallback_triggered:
            trust_delta = -2
            intimacy_delta = -1
        if any(token in user_input.lower() for token in ("thanks", "thank you", "great", "good")):
            trust_delta += 1
        if any(token in user_input.lower() for token in ("sad", "hurt", "upset", "miss", "alone")):
            intimacy_delta += 1
        if any(token in user_input.lower() for token in ("angry", "hate", "annoyed", "frustrated")):
            trust_delta -= 1

        next_state = RelationshipState(
            trust_level=_clamp(state.trust_level + trust_delta),
            intimacy_level=_clamp(state.intimacy_level + intimacy_delta),
            emotional_closeness=_derive_closeness(state.intimacy_level + intimacy_delta),
            conversation_warmth=_derive_warmth(state.trust_level + trust_delta),
        )
        await asyncio.to_thread(self._upsert_state_sync, user_id, next_state)

        updated_memory = _update_relational_memory(
            memory=memory,
            user_input=user_input,
            final_answer=final_answer,
            store_text_snippets=self._relationship_memory_text_enabled,
            privacy_redaction_enabled=self._privacy_redaction_enabled,
        )
        await self._memory_store.update(memory=updated_memory, trace_id=trace_id)

        self._logger.info(
            "relationship.state.updated",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            trust_level=next_state.trust_level,
            intimacy_level=next_state.intimacy_level,
            emotional_closeness=next_state.emotional_closeness,
            conversation_warmth=next_state.conversation_warmth,
        )
        return next_state

    async def is_ready(self) -> bool:
        try:
            await self._initialize()
            return await self._memory_store.is_ready()
        except Exception:
            return False

    async def _initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._memory_store.initialize()
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _initialize_sync(self) -> None:
        db_file = Path(self._db_path)
        if db_file.parent and str(db_file.parent) not in {".", ""}:
            db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relationship_state (
                    user_id TEXT PRIMARY KEY,
                    trust_level INTEGER NOT NULL,
                    intimacy_level INTEGER NOT NULL,
                    emotional_closeness TEXT NOT NULL,
                    conversation_warmth TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()

    def _get_state_row_sync(self, user_id: str) -> tuple[int, int, str, str] | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT trust_level, intimacy_level, emotional_closeness, conversation_warmth
                FROM relationship_state
                WHERE user_id = ?
                LIMIT 1
                """,
                (user_id,),
            )
            return cursor.fetchone()

    def _upsert_state_sync(self, user_id: str, state: RelationshipState) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO relationship_state (
                    user_id,
                    trust_level,
                    intimacy_level,
                    emotional_closeness,
                    conversation_warmth,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    trust_level = excluded.trust_level,
                    intimacy_level = excluded.intimacy_level,
                    emotional_closeness = excluded.emotional_closeness,
                    conversation_warmth = excluded.conversation_warmth,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    state.trust_level,
                    state.intimacy_level,
                    state.emotional_closeness,
                    state.conversation_warmth,
                    utc_now().isoformat(),
                ),
            )
            conn.commit()


def _clamp(value: int) -> int:
    return max(0, min(100, value))


def _derive_closeness(intimacy_level: int) -> str:
    if intimacy_level >= 75:
        return "deep"
    if intimacy_level >= 45:
        return "close"
    return "growing"


def _derive_warmth(trust_level: int) -> str:
    if trust_level >= 70:
        return "warm"
    if trust_level >= 35:
        return "friendly"
    return "neutral"


def _update_relational_memory(
    *,
    memory: RelationshipMemory,
    user_input: str,
    final_answer: str,
    store_text_snippets: bool,
    privacy_redaction_enabled: bool,
) -> RelationshipMemory:
    important_dates = list(memory.important_dates)
    shared_memories = list(memory.shared_memories)
    conflict_history = list(memory.conflict_history)
    inside_jokes = list(memory.inside_jokes)

    safe_user_input = sanitize_text(user_input) if privacy_redaction_enabled else user_input
    safe_final_answer = sanitize_text(final_answer) if privacy_redaction_enabled else final_answer
    lowered = safe_user_input.lower()
    if store_text_snippets:
        if any(token in lowered for token in ("anniversary", "birthday", "first met", "special date")):
            important_dates.append(safe_user_input[:120])
        if any(token in lowered for token in ("remember", "we talked about", "shared")):
            shared_memories.append(safe_user_input[:140])
        if any(token in lowered for token in ("conflict", "fight", "argument", "hurt")):
            conflict_history.append(safe_user_input[:140])
        if any(token in lowered for token in ("inside joke", "joke we have", "our joke")):
            inside_jokes.append(safe_user_input[:120])
    if "Tool Outputs" in safe_final_answer:
        shared_memories.append("Collaborative tool-assisted discussion")

    return RelationshipMemory(
        user_id=memory.user_id,
        first_interaction_timestamp=memory.first_interaction_timestamp,
        important_dates=_dedupe_keep_recent(important_dates, limit=25),
        shared_memories=_dedupe_keep_recent(shared_memories, limit=60),
        conflict_history=_dedupe_keep_recent(conflict_history, limit=30),
        inside_jokes=_dedupe_keep_recent(inside_jokes, limit=30),
    )


def _dedupe_keep_recent(values: list[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in reversed(values):
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    output.reverse()
    return output
