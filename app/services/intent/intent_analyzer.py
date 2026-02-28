from __future__ import annotations


class IntentAnalyzer:
    def analyze(self, user_input: str) -> str:
        normalized = user_input.lower()

        if any(token in normalized for token in ("sad", "anxious", "hurt", "overwhelmed", "stressed", "lonely")):
            return "emotional-support"
        if any(token in normalized for token in ("relationship", "bond", "companion", "attachment", "trust")):
            return "relationship-companion"
        if "?" in user_input or normalized.startswith(
            ("what", "why", "how", "when", "where", "who", "which", "can", "should", "could"),
        ):
            return "question-answering"
        if any(token in normalized for token in ("strategy", "roadmap", "prioritize", "tradeoff", "timeline")):
            return "strategic-planning"
        if any(token in normalized for token in ("error", "issue", "fix", "debug", "exception", "failure")):
            return "troubleshooting"
        if any(token in normalized for token in ("build", "design", "architecture", "plan", "create", "api", "service")):
            return "solution-design"
        return "general-reasoning"

    def answer_style(self, intent: str) -> str:
        mapping = {
            "question-answering": "factual",
            "solution-design": "technical",
            "troubleshooting": "technical",
            "emotional-support": "emotional",
            "relationship-companion": "relational",
            "strategic-planning": "strategic",
            "general-reasoning": "factual",
        }
        return mapping.get(intent, "factual")
