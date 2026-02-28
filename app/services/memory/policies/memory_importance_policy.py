from __future__ import annotations

import re
from typing import Any


class MemoryImportancePolicy:
    def score(self, summary_text: str, prior_memories: list[Any]) -> float:
        repetition = self._repetition_frequency(summary_text=summary_text, prior_memories=prior_memories)
        emotional = self._emotional_intensity(summary_text)
        emphasis = self._explicit_emphasis(summary_text)
        importance = 0.45 * repetition + 0.3 * emotional + 0.25 * emphasis
        return round(max(0.0, min(importance, 1.0)), 4)

    def _repetition_frequency(self, summary_text: str, prior_memories: list[Any]) -> float:
        if not prior_memories:
            return 0.1

        current_tokens = _tokenize(summary_text)
        if not current_tokens:
            return 0.0

        similar_count = 0
        for memory in prior_memories[:20]:
            summary = getattr(memory, "summary_text", "")
            overlap = len(current_tokens & _tokenize(summary))
            ratio = overlap / max(len(current_tokens), 1)
            if ratio >= 0.2:
                similar_count += 1
        return min(1.0, similar_count / 5.0)

    def _emotional_intensity(self, summary_text: str) -> float:
        lowered = summary_text.lower()
        intense_terms = len(
            re.findall(
                r"\b(urgent|critical|blocked|risk|frustrated|angry|happy|excited|stuck|pain)\b",
                lowered,
            ),
        )
        punctuation = summary_text.count("!") + summary_text.count("??")
        return min(1.0, 0.1 * intense_terms + 0.05 * punctuation)

    def _explicit_emphasis(self, summary_text: str) -> float:
        emphasis_terms = len(
            re.findall(
                r"\b(important|must|definitely|always|never|key|priority|focus)\b",
                summary_text.lower(),
            ),
        )
        uppercase_words = len(re.findall(r"\b[A-Z]{3,}\b", summary_text))
        return min(1.0, 0.12 * emphasis_terms + 0.06 * uppercase_words)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

