#!/usr/bin/env python3
"""Prepare a reproducible 500-review sample from the Amazon Reviews 2023 All Beauty dataset.

This project uses the official public dataset repository published by McAuley-Lab.
The review data itself is distributed as JSONL, while the product metadata is published
as Parquet. We sample from the review slice because the review text, rating, timestamps,
and verified-purchase flags live there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REVIEW_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/All_Beauty.jsonl?download=1"
)
METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw_meta_All_Beauty/full-00000-of-00001.parquet?download=1"
)
REVIEW_PATH = RAW_DIR / "All_Beauty.jsonl"
METADATA_PATH = RAW_DIR / "raw_meta_All_Beauty" / "full-00000-of-00001.parquet"
OUTPUT_PATH = RAW_DIR / "all_beauty_sample_500.csv"
RANDOM_SEED = 42


def ensure_remote_file(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urlretrieve(url, target)
    return target


def make_review_id(row: pd.Series) -> str:
    key = (
        f"{row['asin']}|{row['user_id']}|"
        f"{row['timestamp'].isoformat() if pd.notna(row['timestamp']) else ''}|"
        f"{row['rating']}|{row['title']}|{row['text']}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def main() -> None:
    ensure_remote_file(REVIEW_URL, REVIEW_PATH)
    ensure_remote_file(METADATA_URL, METADATA_PATH)

    review_df = pd.read_json(REVIEW_PATH, lines=True)
    metadata_df = pd.read_parquet(METADATA_PATH)

    review_df = review_df.copy()
    review_df["timestamp"] = pd.to_datetime(review_df["timestamp"], errors="coerce")
    review_df["text_clean"] = review_df["text"].fillna("").astype(str).str.strip()
    review_df["review_id"] = review_df.apply(make_review_id, axis=1)

    usable = review_df[review_df["text_clean"].ne("")].copy()
    unique_usable = usable.drop_duplicates(subset=["review_id"]).copy()
    sample = unique_usable.sample(n=500, random_state=RANDOM_SEED, replace=False).copy()

    sample = sample[
        [
            "review_id",
            "rating",
            "title",
            "text",
            "asin",
            "parent_asin",
            "user_id",
            "timestamp",
            "verified_purchase",
            "helpful_vote",
            "images",
        ]
    ].reset_index(drop=True)

    sample.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "source_reviews": len(review_df),
        "usable_reviews": len(usable),
        "unique_usable_reviews": len(unique_usable),
        "sample_size": len(sample),
        "missing_text_count": int((review_df["text_clean"] == "").sum()),
        "duplicate_review_id_count": int(review_df["review_id"].duplicated().sum()),
        "sample_unique_identifiers": sample["review_id"].nunique(),
        "timestamp_missing_count": int(sample["timestamp"].isna().sum()),
        "timestamp_min": sample["timestamp"].min(),
        "timestamp_max": sample["timestamp"].max(),
        "metadata_rows": len(metadata_df),
        "metadata_columns": list(metadata_df.columns),
    }
    print(summary)
    print(f"Sample saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
