"""Gemini-based extraction helpers for review aspect analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from src.prompts import build_extraction_prompt
from src.schema import ReviewExtraction, validate_review_extraction

DEFAULT_MODEL_NAME = "gemini-3.8-flash"


def load_environment(env_file: str | os.PathLike[str] | None = None) -> None:
    """Load project environment variables from the repository .env file if present."""

    if env_file is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    else:
        env_path = Path(env_file)

    if env_path.exists():
        load_dotenv(env_path, override=False)


def resolve_api_key(api_key: str | None = None) -> str:
    """Return a configured Gemini API key or raise a clear validation error."""

    if api_key is not None:
        value = api_key.strip()
        if value:
            return value
        raise ValueError("api_key must be a non-empty string when explicitly provided.")

    load_environment()
    value = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not value:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or .env file."
        )
    return value


def resolve_model_name(model_name: str | None = None) -> str:
    """Resolve the Gemini model from an explicit argument, the environment, or a default."""

    if model_name is not None:
        value = model_name.strip()
        if not value:
            raise ValueError("MODEL_NAME cannot be blank.")
        return value

    load_environment()
    candidate = os.getenv("MODEL_NAME")
    if candidate is not None:
        value = candidate.strip()
        if not value:
            raise ValueError("MODEL_NAME cannot be blank.")
        return value

    # Default to the project’s configured model to keep the extraction path usable even
    # when the environment has not been loaded into the current process.
    if DEFAULT_MODEL_NAME.strip():
        return DEFAULT_MODEL_NAME

    raise ValueError("MODEL_NAME is required. Set MODEL_NAME in the environment or .env file.")


def validate_review_text(review_text: str) -> str:
    """Ensure the review contains usable text for extraction."""

    if not isinstance(review_text, str):
        raise TypeError("review_text must be a string")
    normalized = review_text.strip()
    if not normalized:
        raise ValueError("review_text must not be empty or whitespace only.")
    return review_text


def parse_extraction_payload(payload: ReviewExtraction | dict[str, Any] | str) -> ReviewExtraction:
    """Coerce model output into a validated ReviewExtraction model."""

    if isinstance(payload, ReviewExtraction):
        return payload

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("response payload is not valid JSON.") from exc
        payload = parsed

    if not isinstance(payload, dict):
        raise ValueError("response payload must be a JSON object or dictionary.")

    if "findings" not in payload:
        raise ValueError("response payload must contain a 'findings' list.")

    try:
        return ReviewExtraction.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"response payload failed schema validation: {exc}") from exc


def build_llm(model_name: str | None = None, api_key: str | None = None) -> ChatGoogleGenerativeAI:
    """Construct a configured ChatGoogleGenerativeAI client."""

    resolved_model = resolve_model_name(model_name)
    resolved_key = resolve_api_key(api_key)
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        api_key=resolved_key,
        temperature=0,
    )


def extract_review_aspects(
    review_text: str,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    llm: ChatGoogleGenerativeAI | None = None,
) -> ReviewExtraction:
    """Extract structured findings from a review using the configured Gemini model."""

    normalized_review = validate_review_text(review_text)
    if llm is None:
        llm = build_llm(model_name=model_name, api_key=api_key)

    response = llm.with_structured_output(ReviewExtraction).invoke(build_extraction_prompt(normalized_review))
    extraction = parse_extraction_payload(response)
    validate_review_extraction(extraction, normalized_review)
    return extraction
