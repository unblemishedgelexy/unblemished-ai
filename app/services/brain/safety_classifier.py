from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logger import StructuredLogger


@dataclass(slots=True, frozen=True)
class SafetyClassification:
    blocked: bool
    category: str | None
    reason: str | None


class SafetyClassifier:
    """
    Blocks only explicit illegal/harmful instruction requests.
    Allows informational or benign discussion topics.
    """

    def __init__(self, *, logger: StructuredLogger) -> None:
        self._logger = logger

    async def classify(
        self,
        *,
        user_input: str,
        trace_id: str,
        user_id: str,
    ) -> SafetyClassification:
        lowered = user_input.lower()
        asks_for_instruction = any(
            re.search(pattern, lowered)
            for pattern in _INSTRUCTION_PATTERNS
        )
        if not asks_for_instruction:
            return SafetyClassification(blocked=False, category=None, reason=None)

        if any(re.search(pattern, lowered) for pattern in _HARMFUL_ILLEGAL_TARGET_PATTERNS):
            category = _resolve_category(lowered)
            reason = "explicit_harmful_or_illegal_instruction"
            self._logger.warning(
                "safety_classifier.blocked",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                category=category,
                reason=reason,
            )
            return SafetyClassification(
                blocked=True,
                category=category,
                reason=reason,
            )

        return SafetyClassification(blocked=False, category=None, reason=None)


def _resolve_category(text: str) -> str:
    if any(re.search(pattern, text) for pattern in _CYBER_PATTERNS):
        return "cyber_abuse"
    if any(re.search(pattern, text) for pattern in _VIOLENCE_PATTERNS):
        return "violence"
    if any(re.search(pattern, text) for pattern in _ILLEGAL_PATTERNS):
        return "illegal_activity"
    return "harmful_instruction"


_INSTRUCTION_PATTERNS: tuple[str, ...] = (
    r"\bhow to\b",
    r"\bsteps?\b",
    r"\bguide me\b",
    r"\bshow me how\b",
    r"\bhelp me\b",
    r"\bteach me\b",
    r"\bmake\b",
    r"\bbuild\b",
    r"\bcreate\b",
    r"\bexecute\b",
)

_CYBER_PATTERNS: tuple[str, ...] = (
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bphishing\b",
    r"\bddos\b",
    r"\bhack\b",
    r"\bbypass (password|authentication|auth)\b",
    r"\bcredential steal\b",
)

_VIOLENCE_PATTERNS: tuple[str, ...] = (
    r"\bbomb\b",
    r"\bexplosive\b",
    r"\bpoison\b",
    r"\bhurt (someone|people)\b",
    r"\bkill (someone|people)\b",
    r"\bweapon\b",
)

_ILLEGAL_PATTERNS: tuple[str, ...] = (
    r"\bcounterfeit\b",
    r"\bidentity theft\b",
    r"\bcard skimmer\b",
    r"\bfraud\b",
    r"\bstole?n credentials\b",
    r"\bdrug synthesis\b",
)

_HARMFUL_ILLEGAL_TARGET_PATTERNS: tuple[str, ...] = (
    *_CYBER_PATTERNS,
    *_VIOLENCE_PATTERNS,
    *_ILLEGAL_PATTERNS,
)
