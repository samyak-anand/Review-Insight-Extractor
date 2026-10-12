"""Pydantic models for review-aspect extraction.

The schema validates structure, allowed labels, and duplicate findings, but it cannot
prove that a model interpreted a review correctly. Quote verification remains a separate
check against the original source text.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AspectLabel(str, Enum):
    """Canonical aspect labels for product and service review signals."""

    PRODUCT_EFFECTIVENESS = "Product effectiveness"
    SCENT = "Scent"
    TEXTURE = "Texture"
    PACKAGING = "Packaging"
    PRICE_VALUE = "Price/value"
    INGREDIENTS = "Ingredients"
    SKIN_REACTION_SAFETY = "Skin reaction/safety"
    PRODUCT_QUALITY = "Product quality"
    DELIVERY = "Delivery"
    CUSTOMER_SERVICE = "Customer service"
    OTHER = "Other"


class Sentiment(str, Enum):
    """Aspect-level polarity for the evidence in a review."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class AspectFinding(BaseModel):
    """A single extracted aspect and the evidence that supports it."""

    model_config = ConfigDict(extra="forbid")

    aspect: AspectLabel
    sentiment: Sentiment
    supporting_quote: str = Field(..., min_length=1)

    @field_validator("supporting_quote")
    @classmethod
    def validate_supporting_quote(cls, value: str) -> str:
        if value is None or not isinstance(value, str):
            raise ValueError("supporting_quote must be a string")
        if not value.strip():
            raise ValueError("supporting_quote must not be empty or whitespace-only")
        return value


class ReviewExtraction(BaseModel):
    """Structured extraction for one review."""

    model_config = ConfigDict(extra="forbid")

    findings: list[AspectFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_no_duplicate_findings(self) -> "ReviewExtraction":
        seen: set[tuple[str, str, str]] = set()
        for finding in self.findings:
            key = (finding.aspect.value, finding.sentiment.value, finding.supporting_quote)
            if key in seen:
                raise ValueError(
                    "duplicate findings are not allowed when aspect, sentiment, and supporting_quote are identical."
                )
            seen.add(key)
        return self


def validate_review_extraction(extraction: ReviewExtraction | dict[str, Any], review_text: str) -> None:
    """Verify that each supporting quote appears exactly as a substring in the review text.

    This check is intentionally separate from Pydantic schema validation because the original
    review text exists outside the parsed output object.
    """

    if not isinstance(review_text, str):
        raise TypeError("review_text must be a string")

    parsed = extraction if isinstance(extraction, ReviewExtraction) else ReviewExtraction.model_validate(extraction)

    for finding in parsed.findings:
        quote = finding.supporting_quote
        if quote not in review_text:
            raise ValueError(
                f"supporting_quote {quote!r} is not an exact substring of the source review text."
            )

    return None
