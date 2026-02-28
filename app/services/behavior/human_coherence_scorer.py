from __future__ import annotations

from dataclasses import dataclass

from app.services.behavior.directness_enforcer import DirectnessReport
from app.services.behavior.topic_alignment_checker import TopicAlignmentReport
from app.services.behavior.vagueness_detector import VaguenessReport


@dataclass(slots=True, frozen=True)
class HumanCoherenceReport:
    coherence_score: float
    topic_alignment: float
    completeness: float
    clarity: float
    verbosity_penalty: float
    emotional_appropriateness: float
    requires_regeneration: bool


class HumanCoherenceScorer:
    def __init__(self, threshold: float = 0.45) -> None:
        self._threshold = threshold

    def score(
        self,
        *,
        answer_text: str,
        companion_mode: bool,
        directness_report: DirectnessReport,
        vagueness_report: VaguenessReport,
        topic_alignment_report: TopicAlignmentReport,
    ) -> HumanCoherenceReport:
        word_count = len(answer_text.split())
        topic_alignment = topic_alignment_report.score
        completeness = _bounded(0.25 + min(word_count, 180) / 220.0)
        clarity = _bounded(0.95 - (vagueness_report.vagueness_score * 0.55))
        verbosity_penalty = _bounded(max(word_count - 260, 0) / 340.0)
        emotional_appropriateness = self._emotional_appropriateness(
            answer_text=answer_text,
            companion_mode=companion_mode,
        )

        coherence = (
            (topic_alignment * 0.34)
            + (completeness * 0.27)
            + (clarity * 0.24)
            + (emotional_appropriateness * 0.15)
            - (verbosity_penalty * 0.2)
        )
        coherence_score = _bounded(coherence)

        requires_regeneration = any(
            (
                directness_report.requires_regeneration,
                vagueness_report.is_vague,
                topic_alignment_report.misaligned,
                coherence_score < self._threshold,
            ),
        )
        return HumanCoherenceReport(
            coherence_score=coherence_score,
            topic_alignment=topic_alignment,
            completeness=completeness,
            clarity=clarity,
            verbosity_penalty=verbosity_penalty,
            emotional_appropriateness=emotional_appropriateness,
            requires_regeneration=requires_regeneration,
        )

    def threshold(self) -> float:
        return self._threshold

    def _emotional_appropriateness(self, *, answer_text: str, companion_mode: bool) -> float:
        if not companion_mode:
            return 1.0
        lowered = answer_text.lower()
        harmful_tokens = (
            "you are mine",
            "don't leave me",
            "only you",
            "choose me over everyone",
        )
        if any(token in lowered for token in harmful_tokens):
            return 0.25
        warm_tokens = ("i understand", "we can", "grounded", "balanced", "respectful")
        warmth = 0.15 * sum(1 for token in warm_tokens if token in lowered)
        return _bounded(0.7 + min(warmth, 0.3))


def _bounded(value: float) -> float:
    return round(max(0.0, min(value, 1.0)), 3)
