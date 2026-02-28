from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from app.core.task_manager import TaskManager
from app.core.logger import StructuredLogger
from app.services.memory.memory_retriever import MemoryRetriever, RetrievedMemory
from app.services.memory.memory_store import MemoryStore
from app.services.memory.memory_summarizer import MemorySummarizer
from app.services.privacy.data_sanitizer import sanitize_context, sanitize_text


@dataclass(slots=True)
class TurnSnapshot:
    user_input: str
    assistant_output: str
    context_used: list[dict[str, Any]]


class MemoryInterface:
    def __init__(
        self,
        store: MemoryStore,
        retriever: MemoryRetriever,
        summarizer: MemorySummarizer,
        task_manager: TaskManager,
        logger: StructuredLogger,
        long_term_every_n_messages: int = 5,
        privacy_redaction_enabled: bool = True,
        privacy_remove_sensitive_context_keys: bool = True,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._summarizer = summarizer
        self._task_manager = task_manager
        self._logger = logger
        self._long_term_every_n_messages = max(2, long_term_every_n_messages)
        self._privacy_redaction_enabled = privacy_redaction_enabled
        self._privacy_remove_sensitive_context_keys = privacy_remove_sensitive_context_keys
        self._pending_turns_by_user: dict[str, list[TurnSnapshot]] = {}
        self._pending_turns_lock = asyncio.Lock()

    async def retrieve_relevant(
        self,
        user_id: str,
        trace_id: str,
        query_text: str,
        top_k: int,
        context_max_tokens: int,
    ) -> list[RetrievedMemory]:
        raw = await self._retriever.retrieve(
            user_id=user_id,
            trace_id=trace_id,
            query_text=query_text,
            top_k=top_k,
        )
        compressed = _compress_for_context(raw=raw, max_tokens=context_max_tokens, top_k=top_k)
        self._logger.info(
            "memory.context.compressed",
            trace_id=trace_id,
            user_id=user_id,
            memory_id=",".join(item.memory_id for item in compressed) if compressed else "none",
            retrieval_count=len(compressed),
            source_count=len(raw),
            context_max_tokens=context_max_tokens,
        )
        return compressed

    async def schedule_summary_store(
        self,
        user_id: str,
        trace_id: str,
        user_input: str,
        user_context: dict[str, Any],
        final_answer: str,
        context_used: list[dict[str, Any]],
        max_summary_tokens: int,
    ) -> None:
        self._logger.info(
            "memory.summarizer.scheduled",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="pending",
            retrieval_count=len(context_used),
        )
        self._task_manager.register_task(
            task_name="memory.summary_store",
            trace_id=trace_id,
            user_id=user_id,
            task_factory=lambda: self._safe_summary_store(
                user_id=user_id,
                trace_id=trace_id,
                user_input=user_input,
                user_context=user_context,
                final_answer=final_answer,
                context_used=context_used,
                max_summary_tokens=max_summary_tokens,
            ),
            retry_once=True,
        )

    async def schedule_long_term_evolution(
        self,
        *,
        user_id: str,
        trace_id: str,
        user_input: str,
        final_answer: str,
        context_used: list[dict[str, Any]],
        max_summary_tokens: int,
    ) -> None:
        snapshot = TurnSnapshot(
            user_input=user_input,
            assistant_output=final_answer,
            context_used=context_used,
        )
        batch: list[TurnSnapshot] = []
        async with self._pending_turns_lock:
            queue = self._pending_turns_by_user.setdefault(user_id, [])
            queue.append(snapshot)
            if len(queue) < self._long_term_every_n_messages:
                self._logger.info(
                    "memory.long_term.trigger.deferred",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=len(queue),
                    threshold=self._long_term_every_n_messages,
                )
                return
            batch = queue[: self._long_term_every_n_messages]
            self._pending_turns_by_user[user_id] = queue[self._long_term_every_n_messages :]

        self._logger.info(
            "memory.long_term.summarizer.scheduled",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="pending",
            retrieval_count=len(batch),
            threshold=self._long_term_every_n_messages,
        )
        self._task_manager.register_task(
            task_name="memory.long_term_summary_store",
            trace_id=trace_id,
            user_id=user_id,
            task_factory=lambda: self._safe_long_term_summary_store(
                user_id=user_id,
                trace_id=trace_id,
                turns=batch,
                max_summary_tokens=max_summary_tokens,
            ),
            retry_once=True,
        )

    async def schedule_action_store(
        self,
        user_id: str,
        trace_id: str,
        action_type: str,
        action_result_summary: str,
        importance_override: float | None = None,
    ) -> None:
        self._logger.info(
            "memory.action.scheduled",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="pending",
            retrieval_count=0,
            action_type=action_type,
            importance_override=importance_override,
        )
        self._task_manager.register_task(
            task_name="memory.action_store",
            trace_id=trace_id,
            user_id=user_id,
            task_factory=lambda: self._safe_action_store(
                user_id=user_id,
                trace_id=trace_id,
                action_type=action_type,
                action_result_summary=action_result_summary,
                importance_override=importance_override,
            ),
            retry_once=True,
        )

    async def schedule_correction_store(
        self,
        *,
        user_id: str,
        trace_id: str,
        original_input: str,
        incorrect_output: str,
        corrected_instruction: str,
        trigger_phrase: str,
        importance_override: float = 1.0,
    ) -> None:
        self._logger.info(
            "memory.correction.scheduled",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="pending",
            retrieval_count=0,
            trigger_phrase=trigger_phrase,
        )
        self._task_manager.register_task(
            task_name="memory.correction_store",
            trace_id=trace_id,
            user_id=user_id,
            task_factory=lambda: self._safe_correction_store(
                user_id=user_id,
                trace_id=trace_id,
                original_input=original_input,
                incorrect_output=incorrect_output,
                corrected_instruction=corrected_instruction,
                trigger_phrase=trigger_phrase,
                importance_override=importance_override,
            ),
            retry_once=True,
        )

    async def is_ready(self) -> bool:
        return await self._store.is_ready()

    async def count_entries(self) -> int:
        return await self._store.count_entries()

    async def _safe_summary_store(
        self,
        user_id: str,
        trace_id: str,
        user_input: str,
        user_context: dict[str, Any],
        final_answer: str,
        context_used: list[dict[str, Any]],
        max_summary_tokens: int,
    ) -> None:
        try:
            safe_input = _sanitize_text_if_enabled(
                text=user_input,
                enabled=self._privacy_redaction_enabled,
            )
            safe_context = _sanitize_context_if_enabled(
                context=user_context,
                enabled=self._privacy_redaction_enabled,
                drop_sensitive_keys=self._privacy_remove_sensitive_context_keys,
            )
            safe_answer = _sanitize_text_if_enabled(
                text=final_answer,
                enabled=self._privacy_redaction_enabled,
            )
            await self._summarizer.summarize_and_store(
                user_id=user_id,
                trace_id=trace_id,
                user_input=safe_input,
                user_context=safe_context,
                final_answer=safe_answer,
                context_used=context_used,
                max_summary_tokens=max_summary_tokens,
            )
        except Exception as exc:
            self._logger.error(
                "memory.summarizer.failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=len(context_used),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def _safe_action_store(
        self,
        user_id: str,
        trace_id: str,
        action_type: str,
        action_result_summary: str,
        importance_override: float | None = None,
    ) -> None:
        try:
            safe_action_summary = _sanitize_text_if_enabled(
                text=action_result_summary,
                enabled=self._privacy_redaction_enabled,
            )
            record = await self._store.store_memory(
                user_id=user_id,
                trace_id=trace_id,
                summary_text=(
                    f"ActionType: {action_type}\n"
                    f"ActionResult: {safe_action_summary}"
                ),
                retrieval_count=0,
                action_type=action_type,
                action_result_summary=safe_action_summary,
                importance_override=importance_override,
            )
            self._logger.info(
                "memory.action.stored",
                trace_id=trace_id,
                user_id=user_id,
                memory_id=record.memory_id,
                retrieval_count=0,
                action_type=action_type,
            )
        except Exception as exc:
            self._logger.error(
                "memory.action.store_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
                action_type=action_type,
            )

    async def _safe_correction_store(
        self,
        *,
        user_id: str,
        trace_id: str,
        original_input: str,
        incorrect_output: str,
        corrected_instruction: str,
        trigger_phrase: str,
        importance_override: float,
    ) -> None:
        safe_original_input = _sanitize_text_if_enabled(
            text=original_input,
            enabled=self._privacy_redaction_enabled,
        )
        safe_incorrect_output = _sanitize_text_if_enabled(
            text=incorrect_output,
            enabled=self._privacy_redaction_enabled,
        )
        safe_corrected_instruction = _sanitize_text_if_enabled(
            text=corrected_instruction,
            enabled=self._privacy_redaction_enabled,
        )
        safe_trigger_phrase = _sanitize_text_if_enabled(
            text=trigger_phrase,
            enabled=self._privacy_redaction_enabled,
        )
        summary_text = (
            "MemoryType: behavior_correction\n"
            f"TriggerPhrase: {safe_trigger_phrase}\n"
            f"OriginalInput: {safe_original_input}\n"
            f"IncorrectOutput: {safe_incorrect_output}\n"
            f"CorrectedInstruction: {safe_corrected_instruction}\n"
            "Rule: For similar future inputs, prefer corrected instruction."
        )
        try:
            record = await self._store.store_memory(
                user_id=user_id,
                trace_id=trace_id,
                summary_text=summary_text,
                retrieval_count=0,
                action_type="behavior:correction",
                action_result_summary=safe_corrected_instruction,
                importance_override=importance_override,
            )
            self._logger.info(
                "memory.correction.stored",
                trace_id=trace_id,
                user_id=user_id,
                memory_id=record.memory_id,
                retrieval_count=0,
                trigger_phrase=safe_trigger_phrase,
            )
        except Exception as exc:
            self._logger.error(
                "memory.correction.store_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
                trigger_phrase=safe_trigger_phrase,
            )

    async def _safe_long_term_summary_store(
        self,
        *,
        user_id: str,
        trace_id: str,
        turns: list[TurnSnapshot],
        max_summary_tokens: int,
    ) -> None:
        if not turns:
            return
        recent_turns = [
            {
                "user_input": _sanitize_text_if_enabled(
                    text=item.user_input,
                    enabled=self._privacy_redaction_enabled,
                ),
                "assistant_output": _sanitize_text_if_enabled(
                    text=item.assistant_output,
                    enabled=self._privacy_redaction_enabled,
                ),
                "context_used": item.context_used,
            }
            for item in turns
        ]
        try:
            summary_text = await self._summarizer.build_long_term_summary(
                recent_turns=recent_turns,
                max_summary_tokens=max_summary_tokens,
            )
            if not summary_text:
                self._logger.info(
                    "memory.long_term.skipped",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=len(turns),
                    reason="empty_summary",
                )
                return

            is_duplicate = await self._summarizer.is_duplicate_long_term_summary(
                user_id=user_id,
                trace_id=trace_id,
                summary_text=summary_text,
            )
            if is_duplicate:
                self._logger.info(
                    "memory.long_term.skipped",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=len(turns),
                    reason="duplicate_summary",
                )
                return

            record = await self._store.store_memory(
                user_id=user_id,
                trace_id=trace_id,
                summary_text=summary_text,
                retrieval_count=len(turns),
                action_type="long_term:summary",
                action_result_summary=summary_text,
                importance_override=0.96,
            )
            self._logger.info(
                "memory.long_term.stored",
                trace_id=trace_id,
                user_id=user_id,
                memory_id=record.memory_id,
                retrieval_count=len(turns),
                action_type="long_term:summary",
            )
        except Exception as exc:
            self._logger.error(
                "memory.long_term.store_failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=len(turns),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )


def _compress_for_context(
    raw: list[RetrievedMemory],
    max_tokens: int,
    top_k: int,
) -> list[RetrievedMemory]:
    if not raw:
        return []

    prioritized = sorted(raw, key=lambda item: (item.importance_score, item.relevance_score), reverse=True)
    merged: list[RetrievedMemory] = []

    for candidate in prioritized:
        merged_into_existing = False
        candidate_tokens = _tokenize(candidate.summary_text)
        for index, existing in enumerate(merged):
            if _should_keep_separate(existing=existing, candidate=candidate):
                continue
            overlap_ratio = _overlap_ratio(candidate_tokens, _tokenize(existing.summary_text))
            if overlap_ratio >= 0.6:
                merged[index] = RetrievedMemory(
                    memory_id=existing.memory_id,
                    summary_text=f"{existing.summary_text} | {candidate.summary_text}",
                    relevance_score=max(existing.relevance_score, candidate.relevance_score),
                    created_at=max(existing.created_at, candidate.created_at),
                    importance_score=max(existing.importance_score, candidate.importance_score),
                    action_type=existing.action_type or candidate.action_type,
                )
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(candidate)

    final: list[RetrievedMemory] = []
    used_tokens = 0
    for item in merged:
        item_tokens = item.summary_text.split()
        if used_tokens + len(item_tokens) > max_tokens:
            remaining = max_tokens - used_tokens
            if remaining <= 0:
                break
            item = RetrievedMemory(
                memory_id=item.memory_id,
                summary_text=" ".join(item_tokens[:remaining]),
                relevance_score=item.relevance_score,
                created_at=item.created_at,
                importance_score=item.importance_score,
                action_type=item.action_type,
            )
            final.append(item)
            break
        final.append(item)
        used_tokens += len(item_tokens)
        if len(final) >= top_k:
            break

    return final


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / max(len(left), 1)


def _should_keep_separate(*, existing: RetrievedMemory, candidate: RetrievedMemory) -> bool:
    return existing.action_type == "long_term:summary" or candidate.action_type == "long_term:summary"


def _sanitize_text_if_enabled(*, text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return sanitize_text(text)


def _sanitize_context_if_enabled(
    *,
    context: dict[str, Any],
    enabled: bool,
    drop_sensitive_keys: bool,
) -> dict[str, Any]:
    if not enabled:
        return dict(context)
    return sanitize_context(context, drop_sensitive_keys=drop_sensitive_keys)
