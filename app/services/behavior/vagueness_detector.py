from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class VaguenessReport:
    vagueness_score: float
    is_vague: bool
    filler_flag: bool
    depends_without_explanation: bool


class VaguenessDetector:
    def inspect(self, *, answer_text: str) -> VaguenessReport:
        lowered = answer_text.lower()
        marker_hits = sum(
            1 for pattern in _VAGUE_MARKERS if re.search(pattern, lowered, flags=re.IGNORECASE)
        )
        filler_hits = sum(
            1 for pattern in _FILLER_MARKERS if re.search(pattern, lowered, flags=re.IGNORECASE)
        )
        depends_without_explanation = "it depends" in lowered and not any(
            token in lowered for token in ("because", "depends on", "if", "when", "given")
        )

        score = min((marker_hits * 0.18) + (filler_hits * 0.2) + (0.25 if depends_without_explanation else 0.0), 1.0)
        return VaguenessReport(
            vagueness_score=round(score, 3),
            is_vague=score >= 0.45,
            filler_flag=filler_hits > 0,
            depends_without_explanation=depends_without_explanation,
        )


_VAGUE_MARKERS: tuple[str, ...] = (
    r"\bit depends\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bsomehow\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bmore or less\b",
)

_FILLER_MARKERS: tuple[str, ...] = (
    r"\blet me know\b",
    r"\bhope this helps\b",
    r"\bfeel free to ask\b",
    r"\bif needed\b",
)

