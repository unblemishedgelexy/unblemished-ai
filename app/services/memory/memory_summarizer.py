from __future__ import annotations

import re
from typing import Any

from app.core.logger import StructuredLogger
from app.services.memory.memory_store import MemoryStore


class MemorySummarizer:
    def __init__(self, store: MemoryStore, logger: StructuredLogger) -> None:
        self._store = store
        self._logger = logger

    async def summarize_and_store(
        self,
        user_id: str,
        trace_id: str,
        user_input: str,
        user_context: dict[str, Any],
        final_answer: str,
        context_used: list[dict[str, Any]],
        max_summary_tokens: int,
    ) -> str:
        summary_text = _build_summary(
            user_input=user_input,
            user_context=user_context,
            final_answer=final_answer,
            context_used=context_used,
        )
        compact_summary = _limit_tokens(summary_text, max_summary_tokens)

        record = await self._store.store_memory(
            user_id=user_id,
            trace_id=trace_id,
            summary_text=compact_summary,
            retrieval_count=len(context_used),
        )
        self._logger.info(
            "memory.summarizer.completed",
            trace_id=trace_id,
            user_id=user_id,
            memory_id=record.memory_id,
            retrieval_count=len(context_used),
        )
        return record.memory_id

    async def build_long_term_summary(
        self,
        *,
        recent_turns: list[dict[str, Any]],
        max_summary_tokens: int,
    ) -> str | None:
        summary_text = _build_long_term_summary(recent_turns)
        if not summary_text:
            return None
        return _limit_tokens(summary_text, max_summary_tokens)

    async def is_duplicate_long_term_summary(
        self,
        *,
        user_id: str,
        trace_id: str,
        summary_text: str,
        similarity_threshold: float = 0.82,
    ) -> bool:
        candidate_tokens = _tokenize(summary_text)
        if not candidate_tokens:
            return True

        memories = await self._store.fetch_user_memories(
            user_id=user_id,
            limit=120,
            trace_id=trace_id,
        )
        for memory in memories:
            if memory.action_type != "long_term:summary":
                continue
            score = _token_overlap(candidate_tokens, _tokenize(memory.summary_text))
            if score >= similarity_threshold:
                return True
        return False

    def long_term_prompt_template(self) -> str:
        return LONG_TERM_SUMMARY_PROMPT_TEMPLATE


def _build_summary(
    user_input: str,
    user_context: dict[str, Any],
    final_answer: str,
    context_used: list[dict[str, Any]],
) -> str:
    context_pairs = ", ".join(f"{key}={value}" for key, value in sorted(user_context.items()))
    context_pairs = context_pairs or "none"
    memory_refs = ", ".join(item["memory_id"] for item in context_used) or "none"
    compact_answer = _compact_answer(final_answer)

    return (
        f"UserInput: {user_input}\n"
        f"UserContext: {context_pairs}\n"
        f"AssistantAnswer: {compact_answer}\n"
        f"ReferencedMemories: {memory_refs}"
    )


def _build_long_term_summary(recent_turns: list[dict[str, Any]]) -> str | None:
    if not recent_turns:
        return None

    fact_lines: list[str] = []
    seen_fingerprints: set[str] = set()
    for turn in recent_turns:
        user_input = _compact_text(str(turn.get("user_input", "")))
        assistant_output = _compact_text(str(turn.get("assistant_output", "")))
        if not user_input:
            continue

        fact_type = _classify_fact_type(user_input)
        fact_body = _trim_words(user_input, max_words=18)
        if assistant_output:
            answer_hint = _trim_words(assistant_output, max_words=12)
            fact_line = f"- [{fact_type}] {fact_body} -> {answer_hint}"
        else:
            fact_line = f"- [{fact_type}] {fact_body}"

        fingerprint = _fingerprint_fact(fact_line)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        fact_lines.append(fact_line)
        if len(fact_lines) >= 4:
            break

    if not fact_lines:
        return None

    return (
        "MemoryType: long_term_summary\n"
        f"SummaryWindowMessages: {len(recent_turns)}\n"
        f"PromptTemplate: {LONG_TERM_TEMPLATE_NAME}\n"
        "KeyKnowledge:\n"
        f"{chr(10).join(fact_lines)}"
    )


def _limit_tokens(text: str, max_summary_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_summary_tokens:
        return text
    return " ".join(tokens[:max_summary_tokens])


def _compact_answer(text: str) -> str:
    if not text:
        return "none"

    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"structured reasoning output", "answer:", "reasoning:", "example:"}:
            continue
        if any(marker in lowered for marker in _SYSTEM_META_MARKERS):
            continue
        line = re.sub(r"\s+", " ", line)
        cleaned_lines.append(line)

    if not cleaned_lines:
        return "none"

    # Keep summary memory compact and high-signal.
    compact = " ".join(cleaned_lines[:2])
    tokens = compact.split()
    if len(tokens) > 48:
        compact = " ".join(tokens[:48])
    return compact.strip()


def _compact_text(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered in {"none", "n/a"}:
        return ""
    return normalized


def _classify_fact_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(always|never|prefer|avoid|instead|do not|don't)\b", lowered):
        return "preference"
    if re.search(r"\b(goal|objective|target|want|need|plan|roadmap)\b", lowered):
        return "goal"
    if re.search(r"\b(urgent|deadline|priority|must|blocked|risk|constraint)\b", lowered):
        return "constraint"
    return "fact"


def _trim_words(text: str, max_words: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_words:
        return text
    return " ".join(tokens[:max_words])


def _fingerprint_fact(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    tokens = lowered.split()
    if len(tokens) > 12:
        tokens = tokens[:12]
    return " ".join(tokens)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


_SYSTEM_META_MARKERS: tuple[str, ...] = (
    "promptversion:",
    "traceid:",
    "reasoningmode:",
    "requestedmode:",
    "personanamehint:",
    "styleinstruction:",
    "strictresponsemode:",
    "maxtokens:",
    "temperature:",
    "contextkeys:",
    "userprofile:",
    "relevant past context:",
)

LONG_TERM_TEMPLATE_NAME = "long_term_facts_v1"
LONG_TERM_SUMMARY_PROMPT_TEMPLATE = (
    "You are building long-term memory. Summarize only durable user knowledge.\n"
    "Rules:\n"
    "1) Keep 3-5 concise bullet points.\n"
    "2) Keep only goals, constraints, preferences, and stable facts.\n"
    "3) Remove conversational filler and temporary details.\n"
    "4) Avoid duplicates with existing long-term memory.\n"
    "Output Format:\n"
    "MemoryType: long_term_summary\n"
    "KeyKnowledge:\n"
    "- [goal|constraint|preference|fact] ...\n"
)
