from __future__ import annotations

import re
from typing import Any


def sanitize_text(text: str) -> str:
    if not text:
        return text

    output = text
    output = _EMAIL_REGEX.sub("[REDACTED_EMAIL]", output)
    output = _CARD_REGEX.sub(_replace_card_like, output)
    output = _AADHAAR_REGEX.sub("[REDACTED_AADHAAR]", output)
    output = _PAN_REGEX.sub("[REDACTED_PAN]", output)
    output = _UPI_REGEX.sub("[REDACTED_UPI]", output)
    output = _PHONE_REGEX.sub("[REDACTED_PHONE]", output)
    output = _IPV4_REGEX.sub("[REDACTED_IP]", output)
    output = _PASSWORD_INLINE_REGEX.sub(r"\1: [REDACTED_SECRET]", output)
    output = _TOKEN_INLINE_REGEX.sub("[REDACTED_TOKEN]", output)
    return output


def sanitize_context(
    context: dict[str, Any],
    *,
    drop_sensitive_keys: bool,
) -> dict[str, Any]:
    if not context:
        return {}

    safe: dict[str, Any] = {}
    for raw_key, raw_value in context.items():
        key = str(raw_key)
        lowered_key = key.strip().lower()
        if drop_sensitive_keys and _is_sensitive_key(lowered_key):
            continue
        safe[key] = _sanitize_context_value(raw_value, drop_sensitive_keys=drop_sensitive_keys)
    return safe


def _sanitize_context_value(value: Any, *, drop_sensitive_keys: bool) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return sanitize_context(value, drop_sensitive_keys=drop_sensitive_keys)
    if isinstance(value, list):
        return [_sanitize_context_value(item, drop_sensitive_keys=drop_sensitive_keys) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    return any(token in key for token in _SENSITIVE_KEY_TOKENS)


def _replace_card_like(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 13 or len(digits) > 19:
        return raw
    if not _passes_luhn(digits):
        return raw
    return "[REDACTED_CARD]"


def _passes_luhn(number: str) -> bool:
    checksum = 0
    parity = len(number) % 2
    for index, char in enumerate(number):
        if not char.isdigit():
            return False
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


_SENSITIVE_KEY_TOKENS: tuple[str, ...] = (
    "password",
    "passcode",
    "pin",
    "otp",
    "cvv",
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "cookie",
    "session",
    "email",
    "phone",
    "mobile",
    "contact",
    "address",
    "aadhaar",
    "pan",
    "card",
    "account_number",
    "iban",
    "ifsc",
    "upi",
)

_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_REGEX = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{8,}\d)(?!\d)")
_AADHAAR_REGEX = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_PAN_REGEX = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
_CARD_REGEX = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_UPI_REGEX = re.compile(r"\b[a-zA-Z0-9._-]{2,}@(upi|ybl|ibl|axl|okhdfcbank|okicici|oksbi|paytm)\b", re.IGNORECASE)
_PASSWORD_INLINE_REGEX = re.compile(r"(?i)\b(password|passcode|pin|otp|cvv)\b\s*[:=]?\s*\S+")
_TOKEN_INLINE_REGEX = re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_IPV4_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
