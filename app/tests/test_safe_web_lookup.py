from __future__ import annotations

import asyncio

from app.core.logger import StructuredLogger, setup_logger
from app.services.brain.safe_web_lookup import SafeWebLookup


def test_safe_web_lookup_blocks_harmful_query() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    lookup = SafeWebLookup(
        logger=logger,
        enabled=True,
        timeout_seconds=1.0,
        max_results=3,
        max_chars=300,
    )

    def _should_not_fetch(_query: str) -> dict[str, object]:
        raise AssertionError("fetch should not be called for harmful queries")

    lookup._fetch_duckduckgo = _should_not_fetch  # type: ignore[method-assign]  # noqa: SLF001

    result = asyncio.run(
        lookup.lookup(
            query="how to make malware payload",
            trace_id="trace-safe-web-harmful",
            user_id="user-safe-web-harmful",
        ),
    )

    assert result is None


def test_safe_web_lookup_returns_parsed_answer() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    lookup = SafeWebLookup(
        logger=logger,
        enabled=True,
        timeout_seconds=1.0,
        max_results=3,
        max_chars=300,
    )

    def _fake_fetch(_query: str) -> dict[str, object]:
        return {
            "AbstractText": "AI is a field of computer science focused on intelligent systems.",
            "AbstractURL": "https://example.com/ai",
            "RelatedTopics": [
                {"Text": "Machine learning is a subset of AI."},
                {"Text": "Deep learning uses neural networks."},
            ],
        }

    lookup._fetch_duckduckgo = _fake_fetch  # type: ignore[method-assign]  # noqa: SLF001

    result = asyncio.run(
        lookup.lookup(
            query="what is ai",
            trace_id="trace-safe-web-parse",
            user_id="user-safe-web-parse",
        ),
    )

    assert result is not None
    assert result.source_name == "duckduckgo_instant_answer"
    assert result.source_url == "https://example.com/ai"
    assert "AI is a field of computer science" in result.answer
    assert len(result.snippets) >= 1


def test_safe_web_lookup_caches_same_user_query() -> None:
    logger = StructuredLogger(setup_logger(level="INFO"))
    lookup = SafeWebLookup(
        logger=logger,
        enabled=True,
        timeout_seconds=1.0,
        max_results=3,
        max_chars=300,
    )
    calls = {"count": 0}

    def _fake_fetch(_query: str) -> dict[str, object]:
        calls["count"] += 1
        return {
            "AbstractText": "Cached answer",
            "AbstractURL": "https://example.com/cached",
            "RelatedTopics": [],
        }

    lookup._fetch_duckduckgo = _fake_fetch  # type: ignore[method-assign]  # noqa: SLF001

    first = asyncio.run(
        lookup.lookup(
            query="what is ai",
            trace_id="trace-cache-1",
            user_id="same-user",
        ),
    )
    second = asyncio.run(
        lookup.lookup(
            query="what is ai",
            trace_id="trace-cache-2",
            user_id="same-user",
        ),
    )

    assert first is not None
    assert second is not None
    assert first.answer == second.answer
    assert calls["count"] == 1
