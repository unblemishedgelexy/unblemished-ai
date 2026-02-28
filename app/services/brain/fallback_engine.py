from __future__ import annotations

from typing import Any


class FallbackEngine:
    def build_safe_message(
        self,
        *,
        context: dict[str, Any],
        model_name: str,
        reasons: list[str],
    ) -> str:
        context_keys = ", ".join(sorted(context.keys())) if context else "none"
        return (
            "Structured Reasoning Output\n"
            f"- Routed Model: {model_name}\n"
            "- Safe Fallback: Returning stable structured response.\n"
            "- Reason: Failure grace strategy triggered.\n"
            f"- Failure Signals: {', '.join(reasons) if reasons else 'unknown'}\n"
            f"- Context Signals Used: {context_keys}"
        )

    def build_reflection(
        self,
        *,
        safe_answer: str,
        reflection_enabled: bool,
        reasoning_mode: str,
        trace_id: str,
        reasons: list[str],
    ) -> dict[str, Any]:
        return {
            "confidence": 0.35,
            "strengths": ["Failure grace strategy applied to prevent crash."],
            "risks": [f"Primary path failed: {', '.join(reasons) if reasons else 'unknown'}"],
            "final_answer": safe_answer,
            "metadata": {
                "clarity_score": 0.4,
                "logical_consistency_score": 0.42,
                "completeness_score": 0.32,
                "reflection_pass_enabled": reflection_enabled,
                "reasoning_mode": reasoning_mode,
                "trace_id": trace_id,
            },
        }

