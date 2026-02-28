from __future__ import annotations

import re


class ResponsePostProcessor:
    def process(
        self,
        *,
        text: str,
        strict_response_mode: bool,
        require_example: bool = False,
        answer_style: str | None = None,
    ) -> str:
        normalized_style = _normalize_style(answer_style)
        if not strict_response_mode:
            cleaned = _clean_relaxed_output(text)
            return _apply_style_overlay(cleaned, normalized_style)

        cleaned = _strip_filler(text)
        cleaned = _normalize_spacing(cleaned)
        answer, reasoning, example = _extract_sections(cleaned)
        if _contains_uncertainty(cleaned) and "assumption:" not in reasoning.lower():
            reasoning = (
                f"{reasoning} Assumption: This answer is based on the most probable interpretation "
                "of the available context."
            ).strip()
        if "it depends" in answer.lower() and not _has_explanation(answer):
            reasoning = f"{reasoning} Clarification: The outcome depends on constraints such as scope, risk, and timeline.".strip()

        if require_example and not example:
            example = "Example: Apply the steps to one concrete scenario and validate the output against expected behavior."
        if not answer.strip():
            answer = "No answer generated."
        if not reasoning.strip():
            reasoning = "Reasoning is based on available context."
        answer = _apply_style_overlay(answer, normalized_style)
        return _build_structured_output(answer=answer, reasoning=reasoning, example=example)


def _strip_filler(text: str) -> str:
    result = text
    for pattern in _FILLER_REGEXES:
        result = pattern.sub("", result)
    for pattern in _CONTINUATION_REGEXES:
        result = pattern.sub("", result)
    # Remove repeated blank lines introduced by substitutions.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _normalize_spacing(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if not _is_placeholder_line(line)]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _extract_sections(text: str) -> tuple[str, str, str | None]:
    if not text:
        return "No content generated.", "Provide concrete implementation details.", None

    explicit = _extract_explicit_shape(text)
    if explicit is not None:
        return explicit

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _is_system_meta_line(line):
            continue
        if _is_placeholder_line(line):
            continue
        if line.lower().startswith("structured reasoning output"):
            continue
        line = re.sub(r"^-\s*", "", line)
        line = _strip_inline_system_meta(line)
        if _is_placeholder_line(line):
            continue
        if not line:
            continue
        lines.append(line)

    if not lines:
        return "No content generated.", "Provide concrete implementation details.", None

    direct_parts: list[str] = []
    reasoning_parts: list[str] = []
    example_parts: list[str] = []

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("direct answer:"):
            direct_parts.append(line.split(":", 1)[1].strip())
            continue
        if lowered.startswith("core analysis:") or lowered.startswith("proposed action:") or lowered.startswith("validation:"):
            reasoning_parts.append(line.split(":", 1)[1].strip())
            continue
        if lowered.startswith("tool outputs:"):
            reasoning_parts.append(line)
            continue
        if lowered.startswith("example:"):
            example_parts.append(line)
            continue
        if "example" in lowered and ":" in line:
            example_parts.append(line)
            continue
        if not direct_parts:
            direct_parts.append(line)
        else:
            reasoning_parts.append(line)

    answer = " ".join(part for part in direct_parts if part).strip() or lines[0]
    reasoning = " ".join(part for part in reasoning_parts if part).strip() or "The answer is derived from the provided context and intent."
    example = " ".join(part for part in example_parts if part).strip() or None
    return answer, reasoning, example


def _extract_explicit_shape(text: str) -> tuple[str, str, str | None] | None:
    answer_match = re.search(r"answer:\s*(.+?)(?:\nreasoning:|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    reasoning_match = re.search(r"reasoning:\s*(.+?)(?:\nexample:|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match is None or reasoning_match is None:
        return None

    example_match = re.search(r"example:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    answer = _squash_ws(_remove_system_meta_content(answer_match.group(1)))
    reasoning = _squash_ws(_remove_system_meta_content(reasoning_match.group(1)))
    example = _squash_ws(_remove_system_meta_content(example_match.group(1))) if example_match is not None else None
    if not answer or not reasoning:
        return None
    return answer, reasoning, example


def _squash_ws(value: str) -> str:
    squashed = " ".join(value.split()).strip()
    for pattern in _PLACEHOLDER_REGEXES:
        squashed = pattern.sub("", squashed).strip()
    return squashed


def _remove_system_meta_content(value: str) -> str:
    cleaned_lines: list[str] = []
    for raw in value.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _is_system_meta_line(line):
            continue
        if _is_placeholder_line(line):
            continue
        line = _strip_inline_system_meta(line)
        if _is_placeholder_line(line):
            continue
        if not line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _is_placeholder_line(line: str) -> bool:
    lowered = line.lower().strip()
    if not lowered:
        return False
    if lowered in {
        "[direct answer]",
        "[brief explanation of the reasoning]",
        "[optional example to illustrate the answer]",
    }:
        return True
    return bool(re.fullmatch(r"\[[^\]]+\]", lowered))


def _is_system_meta_line(line: str) -> bool:
    lowered = line.lower().strip()
    if not lowered:
        return False
    if any(lowered.startswith(marker) for marker in _SYSTEM_META_LINE_PREFIXES):
        return True
    meta_hits = sum(1 for marker in _SYSTEM_META_INLINE_MARKERS if marker in lowered)
    return meta_hits >= 2


def _strip_inline_system_meta(line: str) -> str:
    lowered = line.lower()
    first_index: int | None = None
    for marker in _SYSTEM_META_INLINE_MARKERS:
        index = lowered.find(marker)
        if index == -1:
            continue
        if first_index is None or index < first_index:
            first_index = index
    if first_index is None:
        return line.strip()
    if first_index == 0:
        return ""
    return line[:first_index].rstrip(" -:|;,.\t")


def _build_structured_output(*, answer: str, reasoning: str, example: str | None) -> str:
    blocks = [
        "Structured Reasoning Output",
        "Answer:",
        answer.strip() or "No answer generated.",
        "",
        "Reasoning:",
        reasoning.strip() or "Reasoning is based on available context.",
    ]
    if example:
        blocks.extend(
            [
                "",
                "Example:",
                example.strip(),
            ],
        )
    return "\n".join(blocks).strip()


def _clean_relaxed_output(text: str) -> str:
    cleaned = _strip_filler(text)
    cleaned = _MOJIBAKE_REGEX.sub("", cleaned)
    cleaned = _HASHTAG_REGEX.sub("", cleaned)
    cleaned = _MULTISPACE_REGEX.sub(" ", cleaned).strip()
    cleaned = _ANYSPACE_REGEX.sub(" ", cleaned).strip()
    return cleaned


def _normalize_style(style: str | None) -> str:
    if style is None:
        return ""
    normalized = style.strip().lower()
    aliases = {
        "romantic": "flirty",
        "flirt": "flirty",
        "playful": "flirty",
        "warm": "relational",
    }
    return aliases.get(normalized, normalized)


def _apply_style_overlay(text: str, style: str) -> str:
    value = text.strip()
    if not value:
        return value
    if style != "flirty":
        return value
    value = _rewrite_flirty_companion_text(value)
    value = _collapse_repeated_phrase(value)
    lowered = value.lower()
    if any(lowered.startswith(prefix.lower()) for prefix in _FLIRTY_PREFIXES):
        return value
    prefix = _FLIRTY_PREFIXES[len(value) % len(_FLIRTY_PREFIXES)]
    return f"{prefix} {value}"


def _rewrite_flirty_companion_text(text: str) -> str:
    rewrites = (
        (r"\bmain aapki madad karne ke liye yahaan hoon\b", "main yahin hoon, tumse pyaar se baat karne ke liye."),
        (r"\bhow can i help you today\??\b", "tum batao, aaj kya vibe chal rahi hai?"),
        (r"\bkya madad chahiye\??\b", "bolo, aaj kis baat ka mood hai?"),
        (r"\bmain assistant hoon\b", "main tumhari virtual companion hoon"),
        (r"\bmain ladki hoon\b", "main digital companion hoon, feminine tone me chat karti hoon"),
    )
    rewritten = text
    for pattern, replacement in rewrites:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    return rewritten.strip()


def _collapse_repeated_phrase(text: str) -> str:
    compact = _REPEATED_PHRASE_REGEX.sub(r"\1", text)
    compact = _PUNCT_REPEAT_REGEX.sub(r"\1", compact)
    return compact.strip()


def _contains_uncertainty(text: str) -> bool:
    return any(pattern.search(text) for pattern in _UNCERTAINTY_REGEXES)


def _has_explanation(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("because", "depends on", "if", "when", "given"))


_FILLER_PATTERNS: tuple[str, ...] = (
    r"\blet me know if you want (?:more|additional) (?:details|help)\b\.?",
    r"\bi can (?:also )?provide more (?:details|information) if needed\b\.?",
    r"\bin the next message\b",
    r"\bfeel free to ask follow[- ]up questions\b\.?",
    r"\bhope this helps\b\.?",
    r"\bi hope this helps\b\.?",
    r"\bas an ai language model\b[, ]*",
)

_CONTINUATION_PATTERNS: tuple[str, ...] = (
    r"\bin the next message\b",
    r"\bi will continue\b",
    r"\bto be continued\b",
)

_UNCERTAINTY_PATTERNS: tuple[str, ...] = (
    r"\bnot sure\b",
    r"\buncertain\b",
    r"\bcannot be certain\b",
    r"\bunknown\b",
)

_PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"\[\s*direct answer\s*\]",
    r"\[\s*brief explanation of the reasoning\s*\]",
    r"\[\s*optional example to illustrate the answer\s*\]",
)

_FLIRTY_PREFIXES: tuple[str, ...] = (
    "Hey cutie,",
    "Aww sweet one,",
    "Haan jaan,",
    "Hey lovely,",
)

_SYSTEM_META_INLINE_MARKERS: tuple[str, ...] = (
    "prompt basis:",
    "promptversion:",
    "traceid:",
    "reasoningmode:",
    "intent:",
    "answerstyle:",
    "requestedmode:",
    "personanamehint:",
    "userinput:",
    "contextkeys:",
    "userprofile:",
    "relevant past context:",
    "maxtokens:",
    "temperature:",
    "styleinstruction:",
    "strictresponsemode:",
)

_SYSTEM_META_LINE_PREFIXES: tuple[str, ...] = (
    "prompt basis:",
    "promptversion:",
    "traceid:",
    "reasoningmode:",
    "intent:",
    "answerstyle:",
    "requestedmode:",
    "personanamehint:",
    "userinput:",
    "contextkeys:",
    "userprofile:",
    "relevant past context:",
    "maxtokens:",
    "temperature:",
    "styleinstruction:",
    "strictresponsemode:",
)

_FILLER_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _FILLER_PATTERNS
)
_CONTINUATION_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _CONTINUATION_PATTERNS
)
_UNCERTAINTY_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _UNCERTAINTY_PATTERNS
)
_PLACEHOLDER_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _PLACEHOLDER_PATTERNS
)
_MOJIBAKE_REGEX = re.compile(r"Ã°[\w\-]*")
_HASHTAG_REGEX = re.compile(r"(?:^|\s)#[a-zA-Z0-9_]+")
_MULTISPACE_REGEX = re.compile(r"\s{2,}")
_ANYSPACE_REGEX = re.compile(r"\s+")
_REPEATED_PHRASE_REGEX = re.compile(r"(?i)\b([a-z0-9']+(?:\s+[a-z0-9']+){0,6})\b(?:\s+\1){1,}")
_PUNCT_REPEAT_REGEX = re.compile(r"([!?.,])\1+")
