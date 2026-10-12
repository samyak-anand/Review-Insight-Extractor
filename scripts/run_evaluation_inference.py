#!/usr/bin/env python3
"""Run live inference only for the prepared evaluation reviews, with provenance."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_batch_extraction import run_batch_extraction
from src.evaluation import extraction_fingerprint, load_evaluation_reviews, source_fingerprint
from src.extract import extract_review_aspects, resolve_model_name


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=PROJECT_ROOT / "data/evaluation/evaluation_reviews_50.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/evaluation/predictions")
    parser.add_argument("--max-retries", type=int, default=3, help="Transient retries per review (0-3).")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--confirm-live-calls", action="store_true", help="Explicitly approve live Gemini inference for these reviews.")
    args = parser.parse_args()

    reviews = load_evaluation_reviews(args.reviews)
    if len(reviews) > 50:
        raise ValueError("Evaluation inference is limited to at most 50 selected reviews.")
    if not 0 <= args.max_retries <= 3:
        raise ValueError("--max-retries must be between zero and three.")
    max_calls = len(reviews) * (args.max_retries + 1)
    if not args.confirm_live_calls:
        raise SystemExit(
            f"No API calls made. This run would make up to {max_calls} Gemini calls "
            f"for {len(reviews)} reviews. Re-run with --confirm-live-calls only after approval."
        )

    output_dir = args.output_dir
    manifest_path = output_dir / "prediction_manifest.json"
    extraction_version = extraction_fingerprint()
    selection_version = source_fingerprint(reviews)
    model_name = resolve_model_name()
    output_files = [
        output_dir / "aspect_sentiment_results.csv",
        output_dir / "review_checkpoints.csv",
        output_dir / "extraction_failures.csv",
    ]

    if not args.reset:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = (selection_version, extraction_version, model_name)
            actual = (existing.get("source_fingerprint"), existing.get("extraction_fingerprint"), existing.get("model_name"))
            if actual != expected:
                raise ValueError("Existing prediction state has different source or extraction provenance; use --reset intentionally.")
        elif any(path.exists() for path in output_files):
            raise ValueError("Prediction outputs exist without a manifest; use --reset intentionally before live inference.")

    manifest: dict[str, object] = {
        "prediction_source": "live",
        "status": "in_progress",
        "source_fingerprint": selection_version,
        "extraction_fingerprint": extraction_version,
        "model_name": model_name,
        "selected_reviews": len(reviews),
        "maximum_api_calls": max_calls,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path, manifest)

    staged = reviews.copy()
    staged["user_id"] = ""
    api_call_count = 0

    def counted_extractor(review_text: str):
        nonlocal api_call_count
        api_call_count += 1
        return extract_review_aspects(review_text)

    with tempfile.TemporaryDirectory(prefix="review-eval-") as temp_dir:
        staged_path = Path(temp_dir) / "evaluation_reviews.csv"
        staged.to_csv(staged_path, index=False, encoding="utf-8", na_rep="")
        run_batch_extraction(
            staged_path,
            output_dir,
            max_reviews=None,
            extractor=counted_extractor,
            reset=args.reset,
            max_retries=args.max_retries,
        )

    checkpoints_path = output_dir / "review_checkpoints.csv"
    checkpoints = pd.read_csv(checkpoints_path, dtype={"review_id": str})
    status_counts = checkpoints["status"].value_counts().to_dict() if not checkpoints.empty else {}
    manifest.update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "attempted_reviews": int(status_counts.get("completed", 0) + status_counts.get("failed", 0)),
            "successful_reviews": int(status_counts.get("completed", 0)),
            "failed_reviews": int(status_counts.get("failed", 0)),
            "api_calls_made": api_call_count,
        }
    )
    _write_manifest(manifest_path, manifest)
    print(f"Evaluation inference finished. Manifest: {manifest_path}")
    print(f"Maximum API calls allowed by this run: {max_calls}")


if __name__ == "__main__":
    main()