from __future__ import annotations

from typing import Any


class ReflectionEngine:
    async def reflect(
        self,
        draft_answer: str,
        reasoning_steps: list[dict[str, Any]],
        enabled: bool,
        reasoning_mode: str,
        trace_id: str,
    ) -> dict[str, Any]:
        strengths: list[str] = []
        risks: list[str] = []

        if enabled:
            strengths.append("Reflection pass executed.")
        else:
            risks.append("Reflection pass skipped due to mode configuration.")

        if len(reasoning_steps) >= 3:
            strengths.append("Reasoning used explicit staged decomposition.")
        else:
            risks.append("Reasoning depth may be shallow for complex inputs.")

        if len(draft_answer.split()) >= 20:
            strengths.append("Draft answer has enough detail for implementation.")
        else:
            risks.append("Draft answer may need more detail for execution.")

        clarity_score = _bounded_score(
            0.45 + min(len(draft_answer.split()), 80) / 200.0,
        )
        logical_consistency_score = _bounded_score(
            0.45 + min(len(reasoning_steps), 10) / 18.0,
        )
        completeness_score = _bounded_score(
            0.4 + 0.3 * (1.0 if enabled else 0.0) + min(len(strengths), 4) / 10.0,
        )

        confidence = _bounded_score(
            (clarity_score + logical_consistency_score + completeness_score) / 3.0,
        )

        return {
            "confidence": confidence,
            "strengths": strengths,
            "risks": risks,
            "final_answer": draft_answer,
            "metadata": {
                "clarity_score": clarity_score,
                "logical_consistency_score": logical_consistency_score,
                "completeness_score": completeness_score,
                "reflection_pass_enabled": enabled,
                "reasoning_mode": reasoning_mode,
                "trace_id": trace_id,
            },
        }


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(value, 1.0)), 2)

