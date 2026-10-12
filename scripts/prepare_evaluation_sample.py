#!/usr/bin/env python3
"""Prepare a deterministic review set and blank human annotation template."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "all_beauty_sample_500.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
SEED = 42
REVIEW_COLUMNS = ["review_id", "rating", "title", "asin", "parent_asin", "timestamp", "text"]
ANNOTATION_COLUMNS = [
    "review_id",
    "annotation_status",
    "aspect",
    "sentiment",
    "supporting_quote",
    "rationale",
]


def prepare_evaluation_sample(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    sample_size: int = 50,
    seed: int = SEED,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    source = pd.read_csv(input_path)
    required = set(REVIEW_COLUMNS)
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Source sample is missing required columns: {missing}")
    if len(source) != 500:
        raise ValueError(f"Expected the Stage 1 population to contain 500 rows, found {len(source)}.")

    source["review_id"] = source["review_id"].fillna("").astype(str).str.strip()
    source["text"] = source["text"].fillna("").astype(str)
    if source["review_id"].eq("").any() or source["review_id"].duplicated().any():
        raise ValueError("Source review IDs must be non-empty and unique.")
    if source["text"].str.strip().eq("").any():
        raise ValueError("Source reviews must have non-empty text.")
    if not 1 <= sample_size <= len(source):
        raise ValueError(f"sample_size must be between 1 and {len(source)}.")

    output_path = Path(output_dir)
    reviews_path = output_path / "evaluation_reviews_50.csv"
    gold_path = output_path / "gold_annotations_template.csv"
    if not overwrite and (reviews_path.exists() or gold_path.exists()):
        raise FileExistsError(
            f"Evaluation files already exist in {output_path}; pass --overwrite to replace them."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    selected = source.sample(n=sample_size, random_state=seed).loc[:, REVIEW_COLUMNS]
    annotations = pd.DataFrame(
        {
            "review_id": selected["review_id"].tolist(),
            "annotation_status": "unannotated",
            "aspect": "",
            "sentiment": "",
            "supporting_quote": "",
            "rationale": "",
        },
        columns=ANNOTATION_COLUMNS,
    )

    for frame, path in [(selected, reviews_path), (annotations, gold_path)]:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temp_path, index=False, encoding="utf-8", na_rep="")
        temp_path.replace(path)
    return reviews_path, gold_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    reviews_path, gold_path = prepare_evaluation_sample(
        args.input,
        args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Prepared review set: {reviews_path}")
    print(f"Prepared blank annotation template: {gold_path}")
    print(f"Selection: {args.sample_size} unique reviews, pandas sample(random_state={args.seed}).")


if __name__ == "__main__":
    main()