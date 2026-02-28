from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DirectnessReport:
    cleaned_text: str
    has_continuation_promise: bool
    has_filler_ending: bool
    vague_depends_without_explanation: bool
    explicit_question_addressed: bool
    uncertainty_detected: bool
    requires_regeneration: bool


class DirectnessEnforcer:
    def inspect(
        self,
        *,
        answer_text: str,
        user_input: str,
        strict_response_mode: bool,
    ) -> DirectnessReport:
        continuation = _contains_any_pattern(answer_text, _CONTINUATION_PATTERNS)
        filler = _contains_any_pattern(answer_text, _FILLER_PATTERNS)
        vague_depends = self._has_vague_depends(answer_text)
        question_addressed = self._question_addressed(user_input=user_input, answer_text=answer_text)
        uncertainty_detected = _contains_any_pattern(answer_text, _UNCERTAINTY_PATTERNS)

        cleaned = _strip_patterns(answer_text, _CONTINUATION_PATTERNS + _FILLER_PATTERNS).strip()
        if uncertainty_detected and "assumption:" not in cleaned.lower():
            cleaned = (
                f"{cleaned}\n"
                "Assumption: Based on available context, this is the most probable answer. "
                "If key constraints differ, adjust the implementation details accordingly."
            ).strip()

        requires_regeneration = strict_response_mode and any(
            (
                continuation,
                filler,
                vague_depends,
                not question_addressed,
            ),
        )
        return DirectnessReport(
            cleaned_text=cleaned,
            has_continuation_promise=continuation,
            has_filler_ending=filler,
            vague_depends_without_explanation=vague_depends,
            explicit_question_addressed=question_addressed,
            uncertainty_detected=uncertainty_detected,
            requires_regeneration=requires_regeneration,
        )

    def _has_vague_depends(self, answer_text: str) -> bool:
        lowered = answer_text.lower()
        if "it depends" not in lowered:
            return False
        explanatory_tokens = ("because", "depends on", "based on", "if", "when", "given")
        return not any(token in lowered for token in explanatory_tokens)

    def _question_addressed(self, *, user_input: str, answer_text: str) -> bool:
        lowered_input = user_input.strip().lower()
        is_question = "?" in user_input or lowered_input.startswith(
            ("what", "why", "how", "when", "where", "who", "which", "can", "should", "could"),
        )
        if not is_question:
            return True

        input_keywords = _keywords(lowered_input)
        if not input_keywords:
            return True
        answer_keywords = _keywords(answer_text.lower())
        overlap = input_keywords & answer_keywords
        if len(overlap) >= 1:
            return True

        # Fallback for paraphrased but actionable answers to direct questions.
        lowered_answer = answer_text.lower()
        has_action_shape = any(
            marker in lowered_answer
            for marker in (
                "answer:",
                "reasoning:",
                "proposed action:",
                "validation:",
                "step",
                "steps",
                "start with",
                "use ",
                "check ",
                "verify ",
                "debug",
                "troubleshoot",
                "fix",
            )
        )
        return has_action_shape and len(answer_text.split()) >= 16


def _contains_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in patterns)


def _strip_patterns(text: str, patterns: tuple[str, ...]) -> str:
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 3 and token not in _STOP_WORDS}


_CONTINUATION_PATTERNS: tuple[str, ...] = (
    r"\b(in|on) (the )?next message\b",
    r"\bi(?:'| a)m going to continue\b",
    r"\bi will continue\b",
    r"\bto be continued\b",
)

_FILLER_PATTERNS: tuple[str, ...] = (
    r"\blet me know\b",
    r"\bfeel free to ask\b",
    r"\bhope this helps\b",
    r"\bi can provide more details\b",
    r"\bif you want more details\b",
)

_UNCERTAINTY_PATTERNS: tuple[str, ...] = (
    r"\bnot sure\b",
    r"\buncertain\b",
    r"\bcannot be certain\b",
    r"\bunknown\b",
    r"\bmight be wrong\b",
)

_STOP_WORDS: set[str] = {
    "that",
    "this",
    "with",
    "from",
    "have",
    "will",
    "your",
    "about",
    "into",
    "their",
    "them",
    "then",
    "there",
    "would",
    "could",
    "should",
    "what",
    "when",
    "where",
    "which",
    "while",
}
