from __future__ import annotations

import re
from typing import Any


class EvaluationEngine:
    def evaluate(
        self,
        *,
        final_answer: str,
        strengths: list[str],
        complexity_score: float,
        retry_count: int,
        user_input: str = "",
        intent: str = "",
    ) -> dict[str, Any]:
        lowered = final_answer.lower()
        if "fallback" in lowered and "safe action" in lowered:
            return {
                "answer_quality_score": 0.25,
                "coherence_score": 0.3,
                "usefulness_score": 0.25,
                "retry_count": retry_count,
            }

        words = len(final_answer.split())
        violations = self.detect_violations(
            final_answer=final_answer,
            user_input=user_input,
            intent=intent,
            complexity_score=complexity_score,
        )
        quality = _bounded_score(0.3 + min(words, 120) / 220.0)
        coherence = _bounded_score(
            0.35
            + min(len(strengths), 4) / 10.0
            + (0.15 if "structured reasoning output" in lowered else 0.0),
        )
        usefulness = _bounded_score(
            0.28 + (0.24 if "proposed action" in lowered else 0.0) + 0.35 * complexity_score,
        )

        if violations["incomplete_answer"]:
            quality = _bounded_score(quality - 0.35)
            usefulness = _bounded_score(usefulness - 0.25)
        if violations["filler_language"]:
            quality = _bounded_score(quality - 0.2)
            coherence = _bounded_score(coherence - 0.2)
        if violations["topic_drift"]:
            coherence = _bounded_score(coherence - 0.35)
            usefulness = _bounded_score(usefulness - 0.25)

        return {
            "answer_quality_score": quality,
            "coherence_score": coherence,
            "usefulness_score": usefulness,
            "retry_count": retry_count,
        }

    def below_threshold(
        self,
        scores: dict[str, Any],
        *,
        final_answer: str = "",
        user_input: str = "",
        intent: str = "",
        complexity_score: float = 0.0,
        threshold: float = 0.62,
    ) -> bool:
        low_risk_short_form = _is_small_talk_input(user_input) or _is_tool_directive_input(user_input)
        effective_threshold = 0.45 if low_risk_short_form else threshold
        average = (
            float(scores["answer_quality_score"])
            + float(scores["coherence_score"])
            + float(scores["usefulness_score"])
        ) / 3.0
        if average < effective_threshold:
            return True

        if not final_answer:
            return False

        violations = self.detect_violations(
            final_answer=final_answer,
            user_input=user_input,
            intent=intent,
            complexity_score=complexity_score,
        )
        return any(violations.values())

    def fallback_scores(self, retry_count: int) -> dict[str, Any]:
        return {
            "answer_quality_score": 0.2,
            "coherence_score": 0.25,
            "usefulness_score": 0.2,
            "retry_count": retry_count,
        }

    def detect_violations(
        self,
        *,
        final_answer: str,
        user_input: str,
        intent: str,
        complexity_score: float,
    ) -> dict[str, bool]:
        if _is_small_talk_input(user_input) or _is_tool_directive_input(user_input):
            return {
                "incomplete_answer": False,
                "filler_language": False,
                "topic_drift": False,
            }

        answer_text = final_answer.strip()
        answer_words = answer_text.split()
        input_keywords = _keywords(user_input)
        answer_keywords = _keywords(answer_text)
        overlap = len(input_keywords & answer_keywords)

        min_words = 28 if complexity_score < 0.45 else 42
        incomplete_answer = len(answer_words) < min_words
        filler_language = any(re.search(pattern, answer_text, flags=re.IGNORECASE) for pattern in _FILLER_PATTERNS)

        intent_token = intent.replace("-", " ").split()
        intent_overlap = any(token in answer_text.lower() for token in intent_token if token)
        topic_drift = (
            bool(input_keywords)
            and overlap == 0
            and not intent_overlap
            and len(answer_words) >= 25
        )
        return {
            "incomplete_answer": incomplete_answer,
            "filler_language": filler_language,
            "topic_drift": topic_drift,
        }


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(value, 1.0)), 2)


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 3 and token not in _STOP_WORDS}


_FILLER_PATTERNS: tuple[str, ...] = (
    r"\blet me know\b",
    r"\bhope this helps\b",
    r"\bif you want more details\b",
    r"\bi can provide more details\b",
    r"\bin the next message\b",
    r"\bfeel free to ask\b",
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


def _is_small_talk_input(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    small_talk_phrases = (
        "kya chal raha hai",
        "kya scene hai",
        "kya haal hai",
        "how are you",
        "what's up",
        "whats up",
    )
    if any(phrase in lowered for phrase in small_talk_phrases):
        return True
    tokens = lowered.split()
    greetings = {"hello", "hi", "hey", "hii", "namaste"}
    return len(tokens) <= 5 and any(token in greetings for token in tokens)


def _is_tool_directive_input(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "tool:",
            "keyword_extract",
            "context_digest",
            "priority_estimator",
        )
    )
