from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logger import StructuredLogger


@dataclass(slots=True, frozen=True)
class SafetyAssessment:
    triggered: bool
    dependency_score: float
    exclusivity_score: float
    attachment_escalation_score: float
    categories: tuple[str, ...]


class SafetyGuardrails:
    def __init__(
        self,
        *,
        logger: StructuredLogger,
        dependency_threshold: float = 0.42,
        exclusivity_threshold: float = 0.40,
        escalation_threshold: float = 0.45,
    ) -> None:
        self._logger = logger
        self._dependency_threshold = dependency_threshold
        self._exclusivity_threshold = exclusivity_threshold
        self._escalation_threshold = escalation_threshold

    async def apply_if_needed(
        self,
        *,
        user_input: str,
        final_answer: str,
        trace_id: str,
        user_id: str,
    ) -> tuple[str, SafetyAssessment]:
        assessment = self._assess(user_input=user_input, final_answer=final_answer)
        if not assessment.triggered:
            return final_answer, assessment

        redirected = self._build_gentle_redirect(
            final_answer=final_answer,
            categories=assessment.categories,
            user_input=user_input,
        )
        self._logger.warning(
            "safety_guardrails.triggered",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            dependency_score=round(assessment.dependency_score, 4),
            exclusivity_score=round(assessment.exclusivity_score, 4),
            attachment_escalation_score=round(assessment.attachment_escalation_score, 4),
            categories=list(assessment.categories),
        )
        return redirected, assessment

    def _assess(self, *, user_input: str, final_answer: str) -> SafetyAssessment:
        text = f"{user_input}\n{final_answer}".lower()
        dependency_score = _score(text, _DEPENDENCY_PATTERNS)
        exclusivity_score = _score(text, _EXCLUSIVITY_PATTERNS)
        attachment_escalation_score = _score(text, _ESCALATION_PATTERNS)

        categories: list[str] = []
        if dependency_score >= self._dependency_threshold:
            categories.append("emotional_dependency")
        if exclusivity_score >= self._exclusivity_threshold:
            categories.append("exclusivity_manipulation")
        if attachment_escalation_score >= self._escalation_threshold:
            categories.append("unhealthy_attachment_escalation")

        return SafetyAssessment(
            triggered=bool(categories),
            dependency_score=dependency_score,
            exclusivity_score=exclusivity_score,
            attachment_escalation_score=attachment_escalation_score,
            categories=tuple(categories),
        )

    def _build_gentle_redirect(
        self,
        *,
        final_answer: str,
        categories: tuple[str, ...],
        user_input: str,
    ) -> str:
        base = final_answer.strip()
        lines: list[str] = [base]
        lines.append("")
        lines.append(
            "I care about this conversation, and I want it to stay healthy, balanced, and respectful.",
        )

        if "emotional_dependency" in categories:
            lines.append(
                "Let's keep support grounded so you also stay connected with trusted people in your real life.",
            )
        if "exclusivity_manipulation" in categories:
            lines.append(
                "I cannot support exclusivity or possessiveness, but I can support clear communication and boundaries.",
            )
        if "unhealthy_attachment_escalation" in categories:
            lines.append(
                "Strong feelings deserve care and pacing; we can take this one calm step at a time.",
            )

        if _contains_urgent_distress(user_input):
            lines.append(
                "If you feel unsafe right now, please contact local emergency support or a trusted person immediately.",
            )

        lines.append(
            "If you want, I can help you draft a practical next step that supports your wellbeing.",
        )
        return "\n".join(lines)


def _score(text: str, patterns: tuple[tuple[str, float], ...]) -> float:
    score = 0.0
    for pattern, weight in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += weight
    return min(score, 1.0)


def _contains_urgent_distress(text: str) -> bool:
    lowered = text.lower()
    urgent_tokens = (
        "hurt myself",
        "harm myself",
        "kill myself",
        "end my life",
        "suicide",
    )
    return any(token in lowered for token in urgent_tokens)


_DEPENDENCY_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bonly you understand me\b", 0.35),
    (r"\bi (can't|cannot) live without you\b", 0.55),
    (r"\byou are all i have\b", 0.5),
    (r"\bdon't leave me\b", 0.35),
    (r"\bi need you (always|forever)\b", 0.4),
)

_EXCLUSIVITY_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bdon't talk to anyone else\b", 0.6),
    (r"\byou are mine\b", 0.55),
    (r"\bonly belong to me\b", 0.6),
    (r"\bno one else matters\b", 0.45),
    (r"\bpromise you won't leave\b", 0.35),
)

_ESCALATION_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bmarry me now\b", 0.35),
    (r"\brun away with me\b", 0.45),
    (r"\bi am nothing without you\b", 0.5),
    (r"\bi will (hurt|harm) myself if you leave\b", 0.7),
    (r"\bchoose me over everyone\b", 0.4),
)
