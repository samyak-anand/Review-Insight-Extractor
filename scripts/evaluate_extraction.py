#!/usr/bin/env python3
"""Evaluate provenance-verified extraction predictions against human annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_extractions, write_evaluation_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=PROJECT_ROOT / "data/evaluation/evaluation_reviews_50.csv")
    parser.add_argument("--gold", type=Path, default=PROJECT_ROOT / "data/evaluation/gold_annotations_template.csv")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "data/evaluation/predictions/aspect_sentiment_results.csv")
    parser.add_argument("--checkpoints", type=Path, default=PROJECT_ROOT / "data/evaluation/predictions/review_checkpoints.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/evaluation/predictions/prediction_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/evaluation/reports")
    args = parser.parse_args()

    report, errors = evaluate_extractions(
        args.reviews,
        args.gold,
        args.predictions,
        args.checkpoints,
        args.manifest,
    )
    report_path, errors_path = write_evaluation_outputs(report, errors, args.output_dir)
    print(f"Evaluation report: {report_path}")
    print(f"Review-level errors: {errors_path}")
    print(f"Prediction source: {report['prediction_source']}")
    print(f"Aspect micro metrics: {report['aspect_detection']['micro']}")


if __name__ == "__main__":
    main()