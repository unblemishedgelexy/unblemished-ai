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
    compact_input = _compact_user_input(user_input)
    compact_answer = _compact_answer(final_answer)
    intent_type = _classify_intent_type(compact_input)
    extracted_goal = _extract_goal_phrase(compact_input)
    extracted_constraints = _extract_constraints(compact_input, context_pairs)
    extracted_preferences = _extract_preferences(compact_input, context_pairs)

    return (
        "MemoryType: turn_summary\n"
        f"IntentType: {intent_type}\n"
        f"UserInput: {compact_input}\n"
        f"UserGoal: {extracted_goal}\n"
        f"UserConstraints: {extracted_constraints}\n"
        f"UserPreferences: {extracted_preferences}\n"
        f"UserContext: {context_pairs}\n"
        f"AssistantCommitment: {compact_answer}\n"
        f"ReferencedMemories: {memory_refs}"
    )


def _build_long_term_summary(recent_turns: list[dict[str, Any]]) -> str | None:
    if not recent_turns:
        return None

    fact_lines: list[tuple[float, str]] = []
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
        durability = _durability_score(user_input)
        fact_lines.append((durability, fact_line))

    if not fact_lines:
        return None

    fact_lines.sort(key=lambda item: item[0], reverse=True)
    selected_lines = [line for _, line in fact_lines[:5]]

    if not selected_lines:
        return None

    return (
        "MemoryType: long_term_summary\n"
        f"SummaryWindowMessages: {len(recent_turns)}\n"
        f"PromptTemplate: {LONG_TERM_TEMPLATE_NAME}\n"
        "KeyKnowledge:\n"
        f"{chr(10).join(selected_lines)}"
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
    compact = " ".join(cleaned_lines[:3])
    tokens = compact.split()
    if len(tokens) > 64:
        compact = " ".join(tokens[:64])
    return compact.strip()


def _compact_user_input(text: str) -> str:
    compact = _compact_text(text)
    if not compact:
        return "none"
    compact = re.sub(r"\s+", " ", compact).strip()
    compact = re.sub(r"^(please|plz|kindly)\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\b(hey|hi|hello|bhai|bro)\b[:,]?\s*", "", compact, flags=re.IGNORECASE)
    compact = compact.strip(" -:")
    if not compact:
        return "none"
    return _trim_words(compact, max_words=28)


def _classify_intent_type(text: str) -> str:
    lowered = text.lower()
    if lowered == "none":
        return "unknown"
    if re.search(r"\b(design|build|plan|strategy|architecture|implement)\b", lowered):
        return "planning"
    if re.search(r"\b(fix|debug|error|issue|problem|not working|diagnose)\b", lowered):
        return "troubleshooting"
    if re.search(r"\b(explain|define|what is|how does|meaning)\b", lowered):
        return "explanation"
    if re.search(r"\b(wrong|not that|i meant|instead)\b", lowered):
        return "correction"
    return "conversation"


def _extract_goal_phrase(text: str) -> str:
    if not text or text == "none":
        return "none"
    lowered = text.lower()
    patterns = (
        r"\b(i need to|i want to|i want|need to|goal is to|objective is to)\s+(.+)",
        r"\b(please|kindly)\s+(.+)",
        r"\b(design|build|plan|create|explain|summarize|refine)\s+(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            phrase = match.group(match.lastindex or 1)
            return _trim_words(phrase.strip(), max_words=16)
    return _trim_words(text, max_words=16)


def _extract_constraints(user_input: str, context_pairs: str) -> str:
    lowered = user_input.lower()
    hits: list[str] = []
    keywords = ("must", "deadline", "urgent", "priority", "constraint", "limit", "cap", "no", "without", "avoid")
    for token in keywords:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            hits.append(token)
    if "priority=" in context_pairs.lower():
        hits.append("context.priority")
    if not hits:
        return "none"
    return ", ".join(dict.fromkeys(hits))


def _extract_preferences(user_input: str, context_pairs: str) -> str:
    lowered = user_input.lower()
    hits: list[str] = []
    markers = ("prefer", "instead", "use", "style", "format", "short", "detailed", "step-by-step")
    for marker in markers:
        if marker in lowered:
            hits.append(marker)
    if "answer_style=" in context_pairs.lower():
        hits.append("context.answer_style")
    if not hits:
        return "none"
    return ", ".join(dict.fromkeys(hits))


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


def _durability_score(text: str) -> float:
    lowered = text.lower()
    score = 0.25
    if re.search(r"\b(always|never|prefer|avoid|do not|don't)\b", lowered):
        score += 0.35
    if re.search(r"\b(goal|objective|target|want|need|plan|roadmap)\b", lowered):
        score += 0.25
    if re.search(r"\b(urgent|deadline|priority|must|blocked|risk|constraint)\b", lowered):
        score += 0.25
    if len(lowered.split()) >= 8:
        score += 0.1
    return round(min(score, 1.0), 4)


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
