from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.core.logger import StructuredLogger
from app.services.brain.model_adapter import ModelAdapter


@dataclass(slots=True, frozen=True)
class ResponseMatchReport:
    score: float
    is_match: bool
    reasons: list[str]


class ResponseMatchPredictor:
    """
    Lightweight correctness predictor for query-answer alignment.

    This is intentionally heuristic (not a trained model) so Humoniod can
    classify response quality as match / mismatch and trigger one regeneration.
    """

    def __init__(
        self,
        *,
        model_adapter: ModelAdapter | None = None,
        logger: StructuredLogger | None = None,
        model_judge_enabled: bool = False,
        model_judge_name: str = "fast_model",
        model_judge_timeout_seconds: float = 1.2,
        model_judge_max_retries: int = 0,
        model_match_threshold: float = 0.45,
    ) -> None:
        self._model_adapter = model_adapter
        self._logger = logger
        self._model_judge_enabled = model_judge_enabled
        self._model_judge_name = model_judge_name
        self._model_judge_timeout_seconds = model_judge_timeout_seconds
        self._model_judge_max_retries = model_judge_max_retries
        self._model_match_threshold = model_match_threshold

    async def predict(
        self,
        *,
        user_input: str,
        answer_text: str,
        intent: str,
        trace_id: str = "predictor",
        user_id: str = "predictor",
    ) -> ResponseMatchReport:
        heuristic = self._predict_heuristic(
            user_input=user_input,
            answer_text=answer_text,
            intent=intent,
        )
        if not self._should_use_model_judge(heuristic):
            return heuristic

        model_judge = await self._run_model_judge(
            user_input=user_input,
            answer_text=answer_text,
            intent=intent,
            trace_id=trace_id,
            user_id=user_id,
        )
        if model_judge is None:
            return heuristic

        model_score, model_is_match, model_reason = model_judge
        merged_score = max(0.0, min(1.0, (heuristic.score * 0.6) + (model_score * 0.4)))
        if model_score >= 0.78:
            merged_match = True
        elif model_score <= 0.22:
            merged_match = False
        else:
            merged_match = merged_score >= self._model_match_threshold
        merged_reasons = list(dict.fromkeys([*heuristic.reasons, f"model_judge:{model_reason}"]))
        return ResponseMatchReport(
            score=round(merged_score, 3),
            is_match=merged_match,
            reasons=merged_reasons,
        )

    def _predict_heuristic(
        self,
        *,
        user_input: str,
        answer_text: str,
        intent: str,
    ) -> ResponseMatchReport:
        normalized_input = _normalize(user_input)
        normalized_answer = _normalize(answer_text)
        reasons: list[str] = []

        if _is_tool_directive(normalized_input):
            return ResponseMatchReport(
                score=0.82,
                is_match=True,
                reasons=[],
            )

        if _is_small_talk(normalized_input):
            smalltalk_match = any(
                token in normalized_answer
                for token in ("hello", "hi", "ready", "running fine", "help", "active")
            )
            score = 0.78 if smalltalk_match else 0.32
            if not smalltalk_match:
                reasons.append("small_talk_mismatch")
            return ResponseMatchReport(
                score=round(score, 3),
                is_match=smalltalk_match,
                reasons=reasons,
            )

        input_keywords = _keywords(normalized_input)
        answer_keywords = _keywords(normalized_answer)
        overlap = input_keywords & answer_keywords
        overlap_ratio = len(overlap) / max(len(input_keywords), 1) if input_keywords else 0.0

        intent_tokens = [token for token in intent.replace("-", " ").split() if token]
        intent_present = any(token in normalized_answer for token in intent_tokens)
        concept_bonus = _concept_match_bonus(
            query_text=normalized_input,
            answer_text=normalized_answer,
        )
        structure_bonus = 0.08 if any(
            marker in normalized_answer
            for marker in ("answer:", "core analysis:", "proposed action:", "implementation note:")
        ) else 0.0

        off_topic_penalty = 0.0
        if _looks_architecture_template(normalized_answer) and not _is_architecture_query(normalized_input):
            off_topic_penalty = 0.35
            reasons.append("off_topic_template")

        score = (
            0.18
            + (overlap_ratio * 0.56)
            + (0.2 if intent_present else 0.0)
            + structure_bonus
            + concept_bonus
            - off_topic_penalty
        )
        score = max(0.0, min(score, 1.0))
        is_match = score >= 0.45

        if overlap_ratio < 0.12:
            reasons.append("low_keyword_overlap")
        if not intent_present:
            reasons.append("intent_signal_missing")
        if not is_match and not reasons:
            reasons.append("generic_mismatch")

        return ResponseMatchReport(
            score=round(score, 3),
            is_match=is_match,
            reasons=reasons,
        )

    def _should_use_model_judge(self, heuristic: ResponseMatchReport) -> bool:
        if not self._model_judge_enabled:
            return False
        if self._model_adapter is None:
            return False
        # Obvious cases do not need extra model round-trip.
        if heuristic.is_match and heuristic.score >= 0.8:
            return False
        if (not heuristic.is_match) and heuristic.score <= 0.08:
            return False
        return True

    async def _run_model_judge(
        self,
        *,
        user_input: str,
        answer_text: str,
        intent: str,
        trace_id: str,
        user_id: str,
    ) -> tuple[float, bool, str] | None:
        if self._model_adapter is None:
            return None
        prompt = _build_model_judge_prompt(
            user_input=user_input,
            answer_text=answer_text,
            intent=intent,
        )
        try:
            result = await self._model_adapter.generate(
                prompt=prompt,
                trace_id=f"{trace_id}-match-judge",
                context={"answer_style": "factual", "judge_task": "response_match"},
                model_name=self._model_judge_name,
                max_tokens=96,
                temperature=0.0,
                timeout_seconds=self._model_judge_timeout_seconds,
                max_retries=self._model_judge_max_retries,
            )
            if result.fallback_used:
                return None
            parsed = _parse_model_judge_output(result.text)
            if parsed is None:
                return None
            return parsed
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "predictor.model_judge.failed",
                    trace_id=trace_id,
                    user_id=user_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            return None


def _build_model_judge_prompt(
    *,
    user_input: str,
    answer_text: str,
    intent: str,
) -> str:
    return (
        "You are a strict response alignment judge.\n"
        "Task: score whether the assistant answer directly and correctly addresses user request.\n"
        "Return only JSON with keys: match_score (0..1), is_match (true/false), reason (short).\n"
        "No markdown. No extra text.\n\n"
        f"Intent: {intent}\n"
        f"UserInput: {user_input}\n"
        f"AssistantAnswer: {answer_text}\n"
    )


def _parse_model_judge_output(raw_text: str) -> tuple[float, bool, str] | None:
    candidate = raw_text.strip()
    if not candidate:
        return None
    json_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_match:
        parsed = _parse_json_candidate(json_match.group(0))
        if parsed is not None:
            return parsed

    score_match = re.search(
        r"(?:match[_ ]?score|score)\s*[:=]\s*([01](?:\.\d+)?)",
        candidate,
        flags=re.IGNORECASE,
    )
    bool_match = re.search(r"is[_ ]?match\s*[:=]\s*(true|false|yes|no|1|0)", candidate, flags=re.IGNORECASE)
    reason_match = re.search(r"reason\s*[:=]\s*([^\n]+)", candidate, flags=re.IGNORECASE)
    if score_match is None and bool_match is None:
        return None

    score = float(score_match.group(1)) if score_match is not None else 0.5
    if bool_match is not None:
        is_match = _parse_bool(bool_match.group(1))
    else:
        is_match = score >= 0.5
    reason = reason_match.group(1).strip() if reason_match is not None else "parsed_text_judge"
    return max(0.0, min(score, 1.0)), is_match, reason


def _parse_json_candidate(candidate: str) -> tuple[float, bool, str] | None:
    for raw in (candidate, candidate.replace("'", "\"")):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        score_raw: Any = payload.get("match_score", payload.get("score", 0.5))
        try:
            score = float(score_raw)
        except Exception:
            score = 0.5
        score = max(0.0, min(score, 1.0))
        raw_match = payload.get("is_match")
        if isinstance(raw_match, bool):
            is_match = raw_match
        elif raw_match is None:
            is_match = score >= 0.5
        else:
            is_match = _parse_bool(str(raw_match))
        reason = str(payload.get("reason", "json_judge")).strip() or "json_judge"
        return score, is_match, reason
    return None


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 3 and token not in _STOP_WORDS}


def _is_small_talk(text: str) -> bool:
    phrases = (
        "kya chal raha hai",
        "kya scene hai",
        "kya haal hai",
        "how are you",
        "what's up",
        "whats up",
    )
    if any(phrase in text for phrase in phrases):
        return True
    words = text.split()
    return len(words) <= 4 and any(word in {"hello", "hi", "hey", "namaste"} for word in words)


def _looks_architecture_template(text: str) -> bool:
    markers = (
        "modular fastapi architecture",
        "transport-only routes",
        "thin brain facade",
        "staged orchestrator flow",
    )
    return any(marker in text for marker in markers)


def _is_architecture_query(text: str) -> bool:
    return any(
        token in text
        for token in (
            "architecture",
            "fastapi",
            "service",
            "api",
            "modular",
            "system design",
            "backend",
        )
    )


def _is_tool_directive(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "tool:",
            "keyword_extract",
            "context_digest",
            "priority_estimator",
        )
    )


def _concept_match_bonus(*, query_text: str, answer_text: str) -> float:
    matched_groups = 0
    for group in _CONCEPT_GROUPS:
        if any(token in query_text for token in group) and any(token in answer_text for token in group):
            matched_groups += 1
    return min(0.14 * matched_groups, 0.28)


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

_CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "architecture",
        "modular",
        "service",
        "services",
        "orchestrator",
        "orchestration",
        "layer",
        "layers",
    ),
    (
        "schema",
        "contracts",
        "contract",
        "validation",
        "strict",
    ),
    (
        "memory",
        "context",
        "retrieval",
    ),
    (
        "fallback",
        "retry",
        "resilience",
        "reliability",
    ),
)
