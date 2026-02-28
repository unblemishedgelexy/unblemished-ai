from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TopicAlignmentReport:
    score: float
    misaligned: bool
    keyword_overlap: int


class TopicAlignmentChecker:
    def inspect(
        self,
        *,
        user_input: str,
        answer_text: str,
        intent: str,
    ) -> TopicAlignmentReport:
        input_keywords = _keywords(user_input)
        answer_keywords = _keywords(answer_text)
        if not input_keywords:
            return TopicAlignmentReport(score=1.0, misaligned=False, keyword_overlap=0)

        overlap = input_keywords & answer_keywords
        overlap_ratio = len(overlap) / max(len(input_keywords), 1)
        intent_tokens = [token for token in intent.replace("-", " ").split() if token]
        intent_present = any(token in answer_text.lower() for token in intent_tokens)
        score = min(0.2 + (overlap_ratio * 0.6) + (0.2 if intent_present else 0.0), 1.0)
        has_reasoning_cues = any(token in answer_text.lower() for token in _REASONING_CUES)
        misaligned = len(overlap) == 0 and not intent_present and not has_reasoning_cues

        return TopicAlignmentReport(
            score=round(score, 3),
            misaligned=misaligned,
            keyword_overlap=len(overlap),
        )


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 3 and token not in _STOP_WORDS}


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

_REASONING_CUES: tuple[str, ...] = (
    "design",
    "architecture",
    "reasoning",
    "implementation",
    "validation",
    "modular",
    "service",
    "context",
    "plan",
)
