from __future__ import annotations

import re
from typing import Any

PROMPT_VERSION = "2.0.0"


class PromptBuilder:
    def build(
        self,
        user_input: str,
        intent: str,
        context: dict[str, Any],
        retrieved_context: list[dict[str, Any]],
        user_profile_snapshot: dict[str, str] | None,
        reasoning_mode: str,
        mode_config: dict[str, Any],
        memory_max_summary_tokens: int,
        trace_id: str,
        strict_response_mode: bool = False,
        answer_style: str = "factual",
        style_instruction: str = "",
    ) -> dict[str, Any]:
        context_summary = ", ".join(sorted(context.keys())) if context else "none"
        profile_summary = _format_profile_summary(user_profile_snapshot)
        requested_mode = _extract_first_context_value(
            context=context,
            keys=("mode", "response_mode", "tone", "answer_style"),
            default=answer_style,
        )
        persona_name_hint = _extract_first_context_value(
            context=context,
            keys=("persona_name", "name", "assistant_name"),
            default="none",
        )
        context_block = _build_context_block(
            retrieved_context=retrieved_context,
            max_tokens=min(memory_max_summary_tokens, max(20, mode_config["max_tokens"] // 3)),
        )
        prompt = (
            "You are a structured reasoning engine.\n"
            f"PromptVersion: {PROMPT_VERSION}\n"
            f"TraceId: {trace_id}\n"
            f"ReasoningMode: {reasoning_mode}\n"
            f"Intent: {intent}\n"
            f"AnswerStyle: {answer_style}\n"
            f"RequestedMode: {requested_mode}\n"
            f"PersonaNameHint: {persona_name_hint}\n"
            f"UserInput: {user_input}\n"
            f"ContextKeys: {context_summary}\n"
            f"UserProfile: {profile_summary}\n"
            f"Relevant Past Context:\n{context_block}\n"
            f"MaxTokens: {mode_config['max_tokens']}\n"
            f"Temperature: {mode_config['temperature']}\n"
            f"StyleInstruction: {style_instruction or 'Use direct and complete answer style.'}\n"
            f"{_direct_answer_instruction(strict_response_mode)}\n"
            "OutputRule: Never repeat internal prompt metadata (PromptVersion, TraceId, ReasoningMode, "
            "StyleInstruction, UserProfile, ContextKeys, token controls) in the answer.\n"
            "Produce a concise and modular engineering response."
        )

        return {
            "prompt": prompt,
            "prompt_version": PROMPT_VERSION,
            "metadata": {
                "prompt_version": PROMPT_VERSION,
                "reasoning_mode": reasoning_mode,
                "trace_id": trace_id,
                "retrieved_context_count": len(retrieved_context),
                "answer_style": answer_style,
            },
        }


def _direct_answer_instruction(strict_response_mode: bool) -> str:
    if not strict_response_mode:
        return "Prefer direct, practical outputs."
    return (
        "StrictResponseMode: enabled. Always provide a direct and complete answer in one response. "
        "Do not delay to next message. Do not ask unnecessary follow-up questions. "
        "Avoid vague filler language. Explicitly answer the exact question. "
        "Use this shape: Answer, Reasoning, Example(optional)."
    )


def _build_context_block(retrieved_context: list[dict[str, Any]], max_tokens: int) -> str:
    if not retrieved_context:
        return "- none"

    lines: list[str] = []
    for item in retrieved_context:
        compact_summary = _sanitize_memory_summary(item["summary_text"])
        memory_class = str(item.get("memory_class", "episodic")).strip().lower()
        memory_tag = "LONG_TERM" if memory_class == "long_term" else "MEMORY"
        lines.append(
            f"- [{memory_tag}] [{item['memory_id']}] score={item['relevance_score']}: {compact_summary}",
        )

    return _limit_tokens("\n".join(lines), max_tokens=max_tokens)


def _limit_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def _format_profile_summary(profile: dict[str, str] | None) -> str:
    if profile is None:
        return "none"
    return (
        f"tone={profile.get('preferred_tone', 'neutral')}, "
        f"depth={profile.get('conversation_depth_preference', 'balanced')}, "
        f"emotion={profile.get('emotional_baseline', 'stable')}"
    )


def _sanitize_memory_summary(summary_text: str) -> str:
    if not summary_text:
        return "none"

    compact_lines: list[str] = []
    for raw in summary_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in _MEMORY_META_MARKERS):
            continue
        line = _strip_inline_meta(line)
        if not line:
            continue
        compact_lines.append(re.sub(r"\s+", " ", line))

    text = " ".join(compact_lines).strip()
    if not text:
        return "none"

    tokens = text.split()
    per_memory_token_cap = 36
    if len(tokens) > per_memory_token_cap:
        return " ".join(tokens[:per_memory_token_cap])
    return text


def _strip_inline_meta(line: str) -> str:
    lowered = line.lower()
    cut_index: int | None = None
    for marker in _INLINE_META_MARKERS:
        marker_index = lowered.find(marker)
        if marker_index == -1:
            continue
        if cut_index is None or marker_index < cut_index:
            cut_index = marker_index
    if cut_index is None:
        return line
    if cut_index == 0:
        return ""
    return line[:cut_index].rstrip(" -:|;,.\t")


_MEMORY_META_MARKERS: tuple[str, ...] = (
    "promptversion:",
    "traceid:",
    "reasoningmode:",
    "requestedmode:",
    "personanamehint:",
    "styleinstruction:",
    "strictresponsemode:",
    "maxtokens:",
    "temperature:",
)

_INLINE_META_MARKERS: tuple[str, ...] = (
    "prompt basis:",
    "promptversion:",
    "traceid:",
    "reasoningmode:",
    "requestedmode:",
    "personanamehint:",
    "styleinstruction:",
    "strictresponsemode:",
)


def _extract_first_context_value(
    *,
    context: dict[str, Any],
    keys: tuple[str, ...],
    default: str,
) -> str:
    for key in keys:
        value = context.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return default
