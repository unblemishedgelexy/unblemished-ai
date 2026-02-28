from __future__ import annotations

from datetime import datetime

from app.utils.helpers import utc_now


class MemoryDecayPolicy:
    def apply(self, base_importance: float, created_at: datetime) -> float:
        age_days = max((utc_now() - created_at).total_seconds() / 86400.0, 0.0)
        decay_strength = max(0.15, 1.0 - base_importance)
        decayed = base_importance * (1.0 / (1.0 + age_days * decay_strength * 0.35))
        return round(max(0.0, min(decayed, 1.0)), 4)

