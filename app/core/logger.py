from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def setup_logger(name: str = "humoniod.ai", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


class StructuredLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def info(self, event: str, **payload: Any) -> None:
        self._logger.info(json.dumps(self._record(event, payload), default=str))

    def warning(self, event: str, **payload: Any) -> None:
        self._logger.warning(json.dumps(self._record(event, payload), default=str))

    def error(self, event: str, **payload: Any) -> None:
        self._logger.error(json.dumps(self._record(event, payload), default=str))

    @staticmethod
    def _record(event: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
