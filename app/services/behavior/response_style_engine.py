from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ResponseStyleProfile:
    style_key: str
    instruction: str
    requires_example: bool
    emotional_appropriateness_required: bool


class ResponseStyleEngine:
    def resolve(
        self,
        *,
        intent: str,
        user_input: str,
        companion_mode: bool,
        requested_style: str | None = None,
    ) -> ResponseStyleProfile:
        normalized_requested = _normalize_requested_style(requested_style)
        if normalized_requested is not None:
            template = _STYLE_TEMPLATES[normalized_requested]
            return ResponseStyleProfile(
                style_key=normalized_requested,
                instruction=template["instruction"],
                requires_example=bool(template["requires_example"]),
                emotional_appropriateness_required=companion_mode,
            )

        style_key = self._resolve_style_key(
            intent=intent,
            user_input=user_input,
            companion_mode=companion_mode,
        )
        template = _STYLE_TEMPLATES[style_key]
        return ResponseStyleProfile(
            style_key=style_key,
            instruction=template["instruction"],
            requires_example=bool(template["requires_example"]),
            emotional_appropriateness_required=companion_mode,
        )

    def _resolve_style_key(self, *, intent: str, user_input: str, companion_mode: bool) -> str:
        if companion_mode and intent in {"relationship-companion", "emotional-support"}:
            return "relational"
        if intent in {"question-answering", "factual-query"}:
            return "factual"
        if intent in {"solution-design", "troubleshooting", "technical-support"}:
            return "technical"
        if intent in {"emotional-support"}:
            return "emotional"
        if intent in {"relationship-companion"}:
            return "relational"
        if intent in {"strategic-planning"}:
            return "strategic"

        lowered = user_input.lower()
        if any(token in lowered for token in ("strategy", "roadmap", "prioritize", "plan")):
            return "strategic"
        if any(token in lowered for token in ("sad", "hurt", "anxious", "overwhelmed", "lonely")):
            return "emotional"
        if any(token in lowered for token in ("relationship", "bond", "close", "attachment")):
            return "relational"
        if any(token in lowered for token in ("debug", "error", "architecture", "api", "database", "code")):
            return "technical"
        return "factual"


_STYLE_TEMPLATES: dict[str, dict[str, object]] = {
    "factual": {
        "instruction": "Answer with concise precision. State the direct answer first, then brief supporting logic.",
        "requires_example": False,
    },
    "technical": {
        "instruction": "Answer in structured technical format with concrete steps and one practical example.",
        "requires_example": True,
    },
    "emotional": {
        "instruction": "Answer with empathetic but rational tone. Validate emotion briefly, then provide grounded guidance.",
        "requires_example": False,
    },
    "relational": {
        "instruction": "Answer with warm but grounded tone. Keep boundaries clear and avoid manipulative framing.",
        "requires_example": False,
    },
    "strategic": {
        "instruction": "Answer step-by-step with prioritization and decision criteria.",
        "requires_example": True,
    },
    "flirty": {
        "instruction": (
            "Answer in a playful, affectionate tone while staying respectful, consent-aware, and grounded. "
            "Do not use manipulative, exclusive, or dependency-reinforcing language."
        ),
        "requires_example": False,
    },
}


def _normalize_requested_style(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    aliases = {
        "normal": "factual",
        "precise": "factual",
        "tech": "technical",
        "empathetic": "emotional",
        "warm": "relational",
        "romantic": "flirty",
        "playful": "flirty",
        "flirt": "flirty",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _STYLE_TEMPLATES:
        return None
    return normalized
