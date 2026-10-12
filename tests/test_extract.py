from __future__ import annotations

import pytest

from src.extract import (
    parse_extraction_payload,
    resolve_api_key,
    resolve_model_name,
    validate_review_text,
)


def test_resolve_model_name_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    assert resolve_model_name() == "gemini-2.5-flash"


def test_resolve_model_name_rejects_blank_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "   ")
    with pytest.raises(ValueError, match="MODEL_NAME"):
        resolve_model_name()


def test_validate_review_text_rejects_blank_review() -> None:
    with pytest.raises(ValueError, match="review_text"):
        validate_review_text("   ")


def test_parse_extraction_payload_accepts_dict() -> None:
    payload = {
        "findings": [
            {
                "aspect": "Product effectiveness",
                "sentiment": "positive",
                "supporting_quote": "works well",
            }
        ]
    }
    extraction = parse_extraction_payload(payload)
    assert extraction.findings[0].aspect.value == "Product effectiveness"
    assert extraction.findings[0].sentiment.value == "positive"


def test_parse_extraction_payload_rejects_missing_findings() -> None:
    with pytest.raises(ValueError, match="findings"):
        parse_extraction_payload({"unexpected": "payload"})


def test_resolve_api_key_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("src.extract.load_environment", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        resolve_api_key()
