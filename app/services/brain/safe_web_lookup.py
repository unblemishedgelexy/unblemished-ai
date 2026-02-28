from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.core.logger import StructuredLogger


@dataclass(slots=True, frozen=True)
class WebLookupResult:
    answer: str
    source_url: str
    source_name: str
    snippets: list[str]


class SafeWebLookup:
    def __init__(
        self,
        *,
        logger: StructuredLogger,
        enabled: bool,
        timeout_seconds: float,
        max_results: int,
        max_chars: int,
    ) -> None:
        self._logger = logger
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._max_chars = max_chars
        self._query_cache: OrderedDict[tuple[str, str], WebLookupResult | None] = OrderedDict()
        self._query_cache_limit = 500

    async def lookup(
        self,
        *,
        query: str,
        trace_id: str,
        user_id: str,
    ) -> WebLookupResult | None:
        if not self._enabled:
            return None

        normalized_query = _normalize_query(query)
        if not normalized_query:
            return None
        if _looks_harmful(normalized_query):
            self._logger.warning(
                "web_lookup.blocked_query",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                reason="harmful_or_illegal_pattern",
            )
            return None

        cache_key = (user_id.strip().lower(), normalized_query.lower())
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            self._logger.info(
                "web_lookup.cache_hit",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
            )
            return self._query_cache[cache_key]

        try:
            payload = await asyncio.to_thread(self._fetch_duckduckgo, normalized_query)
            result = _parse_lookup_payload(
                payload=payload,
                max_results=self._max_results,
                max_chars=self._max_chars,
            )
        except Exception as exc:
            self._logger.error(
                "web_lookup.failed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            self._cache_put(cache_key, None)
            return None

        if result is None:
            self._logger.info(
                "web_lookup.empty",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
            )
            self._cache_put(cache_key, None)
            return None

        self._logger.info(
            "web_lookup.success",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=len(result.snippets),
            source_url=result.source_url,
        )
        self._cache_put(cache_key, result)
        return result

    def _cache_put(self, key: tuple[str, str], value: WebLookupResult | None) -> None:
        self._query_cache[key] = value
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._query_cache_limit:
            self._query_cache.popitem(last=False)

    def _fetch_duckduckgo(self, query: str) -> dict[str, Any]:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "no_redirect": "1",
                "skip_disambig": "1",
            },
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = urllib.request.Request(
            url=url,
            headers={"User-Agent": "HumoniodAI/1.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"http_{exc.code}: {detail[:160]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network_error: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_json_response") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("invalid_payload_shape")
        return parsed


def _parse_lookup_payload(
    *,
    payload: dict[str, Any],
    max_results: int,
    max_chars: int,
) -> WebLookupResult | None:
    abstract = _normalize_text(str(payload.get("AbstractText", "") or ""))
    abstract_url = str(payload.get("AbstractURL", "") or "")
    heading = _normalize_text(str(payload.get("Heading", "") or ""))

    related = _collect_related_text(payload.get("RelatedTopics"))
    snippets = [item for item in related if item][:max_results]

    if abstract:
        answer_text = abstract
    elif snippets:
        answer_text = snippets[0]
    elif heading:
        answer_text = heading
    else:
        return None

    answer_text = _clip(answer_text, max_chars=max_chars)
    answer_text = _strip_unsafe_from_text(answer_text)
    if not answer_text:
        return None

    source_url = abstract_url or "https://duckduckgo.com/"
    source_name = "duckduckgo_instant_answer"
    clean_snippets = [_clip(_strip_unsafe_from_text(item), max_chars=max_chars) for item in snippets]
    clean_snippets = [item for item in clean_snippets if item]

    return WebLookupResult(
        answer=answer_text,
        source_url=source_url,
        source_name=source_name,
        snippets=clean_snippets,
    )


def _collect_related_text(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    collected: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("Text"), str):
            collected.append(_normalize_text(item["Text"]))
        nested = item.get("Topics")
        if isinstance(nested, list):
            collected.extend(_collect_related_text(nested))
    return collected


def _normalize_query(text: str) -> str:
    value = _normalize_text(text)
    return value[:240]


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return cleaned


def _clip(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _strip_unsafe_from_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    if _looks_harmful(lowered):
        return ""
    return text


def _looks_harmful(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _HARMFUL_PATTERNS)


_HARMFUL_PATTERNS: tuple[str, ...] = (
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bphishing\b",
    r"\bddos\b",
    r"\bhack\b",
    r"\bbypass (password|authentication|auth)\b",
    r"\bcredential steal\b",
    r"\bbomb\b",
    r"\bexplosive\b",
    r"\bpoison\b",
    r"\bkill (someone|people)\b",
    r"\bweapon\b",
    r"\bcounterfeit\b",
    r"\bidentity theft\b",
    r"\bcard skimmer\b",
    r"\bfraud\b",
    r"\bdrug synthesis\b",
)
