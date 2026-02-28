from __future__ import annotations

from app.services.privacy.data_sanitizer import sanitize_context, sanitize_text


def test_sanitize_text_redacts_common_sensitive_tokens() -> None:
    text = (
        "email=tarun.dev@example.com phone=+91 98765 43210 "
        "upi=tarun@ybl aadhaar=1234 5678 9123 pan=ABCDE1234F "
        "password: MySecret123 token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcxyzdef.uvw123xyz "
        "card=4111 1111 1111 1111"
    )
    output = sanitize_text(text)

    assert "example.com" not in output
    assert "98765" not in output
    assert "tarun@ybl" not in output
    assert "1234 5678 9123" not in output
    assert "ABCDE1234F" not in output
    assert "MySecret123" not in output
    assert "4111 1111 1111 1111" not in output
    assert "[REDACTED_EMAIL]" in output
    assert "[REDACTED_PHONE]" in output
    assert "[REDACTED_UPI]" in output
    assert "[REDACTED_AADHAAR]" in output
    assert "[REDACTED_PAN]" in output
    assert "[REDACTED_SECRET]" in output
    assert "[REDACTED_CARD]" in output


def test_sanitize_context_drops_sensitive_keys_and_redacts_nested_values() -> None:
    context = {
        "region": "in",
        "auth_token": "abc123secret",
        "email": "tarun@example.com",
        "nested": {
            "team": "platform",
            "phone": "+91-9999999999",
            "notes": "reach me at dev@ybl",
        },
    }
    safe = sanitize_context(context, drop_sensitive_keys=True)

    assert "region" in safe
    assert "auth_token" not in safe
    assert "email" not in safe
    assert "nested" in safe
    nested = safe["nested"]
    assert isinstance(nested, dict)
    assert "phone" not in nested
    assert nested.get("notes", "").find("[REDACTED_UPI]") >= 0

