#!/usr/bin/env python3
"""Run a controlled 10-review Gemini extraction smoke test against the sample CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract import extract_review_aspects, resolve_model_name
SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "all_beauty_sample_500.csv"


def load_validation_reviews(path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Sample CSV is missing the 'text' column.")

    cleaned = df[["review_id", "text"]].copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    return cleaned.head(10).to_dict(orient="records")


def main() -> None:
    reviews = load_validation_reviews(SAMPLE_PATH)
    print(f"Validating {len(reviews)} reviews with model: {resolve_model_name()}")

    success_count = 0
    failures: list[tuple[str, str]] = []

    for index, row in enumerate(reviews, start=1):
        review_id = str(row["review_id"])
        review_text = str(row["text"])
        try:
            extraction = extract_review_aspects(review_text)
            print(f"{index}. review_id={review_id} findings={len(extraction.findings)}")
            success_count += 1
        except Exception as exc:  # pragma: no cover - command-line diagnostics only
            print(f"{index}. review_id={review_id} ERROR={exc.__class__.__name__}: {exc}")
            failures.append((review_id, str(exc)))

    print(f"SUMMARY: success={success_count} failed={len(failures)} total={len(reviews)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
