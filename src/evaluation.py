"""Validated evaluation utilities for aspect-level extraction outputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.extract import resolve_model_name
from src.schema import AspectLabel, Sentiment

REVIEW_COLUMNS = {"review_id", "text"}
GOLD_COLUMNS = {
    "review_id",
    "annotation_status",
    "aspect",
    "sentiment",
    "supporting_quote",
    "rationale",
}
PREDICTION_COLUMNS = {"review_id", "aspect", "sentiment", "supporting_quote"}
CHECKPOINT_COLUMNS = {"review_id", "status", "attempt_count", "aspect_count"}
ANNOTATION_STATUSES = {"unannotated", "annotated", "no_aspects", "needs_adjudication"}
ASPECTS = {label.value for label in AspectLabel}
SENTIMENTS = [label.value for label in Sentiment]
EXTRACTION_FILES = ["src/extract.py", "src/schema.py", "src/prompts.py"]
ERROR_COLUMNS = [
    "review_id",
    "error_type",
    "aspect",
    "gold_sentiment",
    "predicted_sentiment",
    "detail",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _nonblank(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be blank.")
    return result


def load_evaluation_reviews(path: str | Path) -> pd.DataFrame:
    reviews = _read_csv(path)
    _require_columns(reviews, REVIEW_COLUMNS, "Evaluation reviews")
    reviews = reviews.copy()
    reviews["review_id"] = reviews["review_id"].map(lambda value: _nonblank(value, "review_id"))
    if reviews["review_id"].duplicated().any():
        raise ValueError("Evaluation review IDs must be unique.")
    if reviews["text"].map(lambda value: not str(value).strip()).any():
        raise ValueError("Evaluation review text must not be blank.")
    return reviews


def source_fingerprint(reviews: pd.DataFrame) -> str:
    """Hash the selected IDs and exact source text, independent of row order."""

    pairs = sorted((str(row.review_id), str(row.text)) for row in reviews.itertuples())
    serialized = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extraction_fingerprint(project_root: str | Path | None = None) -> str:
    """Fingerprint the extraction, schema, and prompt source files (not secrets)."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative_path in EXTRACTION_FILES:
        digest.update(relative_path.encode("utf-8"))
        digest.update((root / relative_path).read_bytes())
    return "sha256:" + digest.hexdigest()


def _load_gold(path: str | Path, reviews: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], set[str]]:
    gold_df = _read_csv(path)
    _require_columns(gold_df, GOLD_COLUMNS, "Gold annotations")
    review_ids = set(reviews["review_id"])
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in gold_df.to_dict(orient="records"):
        review_id = _nonblank(row["review_id"], "gold review_id")
        if review_id not in review_ids:
            raise ValueError(f"Gold annotation references an unknown review_id: {review_id}")
        status = _nonblank(row["annotation_status"], "annotation_status")
        if status not in ANNOTATION_STATUSES:
            raise ValueError(f"Unknown annotation_status {status!r} for review {review_id}.")
        grouped.setdefault(review_id, []).append(row)

    missing_ids = review_ids - set(grouped)
    if missing_ids:
        raise ValueError(f"Gold annotations are missing selected review IDs: {sorted(missing_ids)}")

    review_text = dict(zip(reviews["review_id"], reviews["text"], strict=True))
    normalized: dict[str, dict[str, Any]] = {}
    for review_id, rows in grouped.items():
        statuses = {row["annotation_status"].strip() for row in rows}
        if len(statuses) != 1:
            raise ValueError(f"Review {review_id} has mixed annotation statuses.")
        status = statuses.pop()
        if status != "annotated" and len(rows) != 1:
            raise ValueError(f"Review {review_id} must have one row for status {status!r}.")

        findings: dict[str, dict[str, str]] = {}
        for row in rows:
            aspect = row["aspect"].strip()
            sentiment = row["sentiment"].strip()
            quote = row["supporting_quote"]
            if status == "annotated":
                if aspect not in ASPECTS:
                    raise ValueError(f"Unknown gold aspect {aspect!r} for review {review_id}.")
                if sentiment not in SENTIMENTS:
                    raise ValueError(f"Unknown gold sentiment {sentiment!r} for review {review_id}.")
                if not quote or quote not in review_text[review_id]:
                    raise ValueError(f"Gold quote for review {review_id} must be an exact source-text substring.")
                if aspect in findings:
                    raise ValueError(f"Gold annotations contain duplicate aspect {aspect!r} for review {review_id}.")
                findings[aspect] = {
                    "sentiment": sentiment,
                    "supporting_quote": quote,
                }
            elif any((aspect, sentiment, quote, row["rationale"].strip())):
                raise ValueError(f"Review {review_id} status {status!r} must not contain a finding.")
        normalized[review_id] = {"status": status, "findings": findings}
    return normalized, review_ids


def _load_predictions(
    path: str | Path,
    reviews: pd.DataFrame,
    gold: dict[str, dict[str, Any]],
    checkpoints: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, str]]], int, int]:
    predictions_df = _read_csv(path)
    _require_columns(predictions_df, PREDICTION_COLUMNS, "Predictions")
    review_ids = set(reviews["review_id"])
    unique_findings: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for row in predictions_df.to_dict(orient="records"):
        review_id = _nonblank(row["review_id"], "prediction review_id")
        if review_id not in review_ids:
            raise ValueError(f"Prediction references an unknown review_id: {review_id}")
        aspect = _nonblank(row["aspect"], "prediction aspect")
        sentiment = _nonblank(row["sentiment"], "prediction sentiment")
        quote = row["supporting_quote"]
        if aspect not in ASPECTS:
            raise ValueError(f"Unknown predicted aspect {aspect!r} for review {review_id}.")
        if sentiment not in SENTIMENTS:
            raise ValueError(f"Unknown predicted sentiment {sentiment!r} for review {review_id}.")
        if not quote:
            raise ValueError(f"Prediction quote must not be blank for review {review_id}.")
        checkpoint = checkpoints.get(review_id)
        if checkpoint is None or checkpoint["status"] != "completed":
            raise ValueError(f"Predictions exist for review {review_id} without a completed checkpoint.")
        key = (review_id, aspect, sentiment, quote)
        unique_findings[key] = {
            "review_id": review_id,
            "aspect": aspect,
            "sentiment": sentiment,
            "supporting_quote": quote,
        }

    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for finding in unique_findings.values():
        review_id = finding["review_id"]
        aspect = finding["aspect"]
        by_aspect = normalized.setdefault(review_id, {})
        if aspect in by_aspect:
            raise ValueError(
                f"Conflicting duplicate predictions for review {review_id}, aspect {aspect!r}; "
                "exact duplicate rows are collapsed, distinct rows for one aspect are rejected."
            )
        by_aspect[aspect] = finding

    total_unique = len(unique_findings)
    excluded_unannotated = sum(
        len(findings)
        for review_id, findings in normalized.items()
        if gold[review_id]["status"] in {"unannotated", "needs_adjudication"}
    )
    return normalized, total_unique, excluded_unannotated


def _load_checkpoints(path: str | Path, reviews: pd.DataFrame) -> dict[str, dict[str, Any]]:
    checkpoint_df = _read_csv(path)
    _require_columns(checkpoint_df, CHECKPOINT_COLUMNS, "Checkpoints")
    review_ids = set(reviews["review_id"])
    checkpoints: dict[str, dict[str, Any]] = {}
    for row in checkpoint_df.to_dict(orient="records"):
        review_id = _nonblank(row["review_id"], "checkpoint review_id")
        if review_id not in review_ids:
            raise ValueError(f"Checkpoint references an unknown review_id: {review_id}")
        if review_id in checkpoints:
            raise ValueError(f"Duplicate checkpoint review_id: {review_id}")
        status = row["status"].strip()
        if status not in {"completed", "failed"}:
            raise ValueError(f"Invalid checkpoint status {status!r} for review {review_id}.")
        try:
            attempt_count = int(row["attempt_count"])
            aspect_count = int(row["aspect_count"])
        except ValueError as exc:
            raise ValueError(f"Invalid checkpoint count for review {review_id}.") from exc
        if attempt_count < 0 or aspect_count < 0:
            raise ValueError(f"Negative checkpoint count for review {review_id}.")
        checkpoints[review_id] = {
            "status": status,
            "attempt_count": attempt_count,
            "aspect_count": aspect_count,
        }
    return checkpoints


def _validate_manifest(path: str | Path, reviews: pd.DataFrame) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Prediction manifest is missing or invalid: {path}") from exc
    required = {"prediction_source", "source_fingerprint", "extraction_fingerprint", "model_name"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Prediction manifest is missing fields: {sorted(missing)}")
    if manifest["prediction_source"] not in {"live", "synthetic"}:
        raise ValueError("prediction_source must be 'live' or 'synthetic'.")
    if manifest.get("status") != "completed":
        raise ValueError("Prediction manifest must have status 'completed'.")
    if manifest["source_fingerprint"] != source_fingerprint(reviews):
        raise ValueError("Predictions were not generated for the exact selected review IDs and text.")
    if manifest["extraction_fingerprint"] != extraction_fingerprint():
        raise ValueError("Prediction extraction fingerprint does not match the current Stage 3 code.")
    if manifest["model_name"] != resolve_model_name():
        raise ValueError("Prediction model name does not match the current configured model.")
    return manifest


def _score_counts(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_extractions(
    reviews_path: str | Path,
    gold_path: str | Path,
    predictions_path: str | Path,
    checkpoints_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate an exact, provenance-verified prediction run against human labels."""

    reviews = load_evaluation_reviews(reviews_path)
    gold, _ = _load_gold(gold_path, reviews)
    checkpoints = _load_checkpoints(checkpoints_path, reviews)
    manifest = _validate_manifest(manifest_path, reviews)
    predictions, unique_prediction_count, excluded_prediction_count = _load_predictions(
        predictions_path, reviews, gold, checkpoints
    )
    for review_id, checkpoint in checkpoints.items():
        if checkpoint["status"] == "completed":
            actual_count = len(predictions.get(review_id, {}))
            if actual_count != checkpoint["aspect_count"]:
                raise ValueError(
                    f"Checkpoint aspect_count mismatch for review {review_id}: "
                    f"expected {checkpoint['aspect_count']}, found {actual_count}."
                )

    annotated_ids = {
        review_id
        for review_id, entry in gold.items()
        if entry["status"] in {"annotated", "no_aspects"}
    }
    gold_pairs = {
        (review_id, aspect)
        for review_id in annotated_ids
        for aspect in gold[review_id]["findings"]
    }
    predicted_pairs = {
        (review_id, aspect)
        for review_id in annotated_ids
        for aspect in predictions.get(review_id, {})
    }
    matched_pairs = gold_pairs & predicted_pairs
    tp_pairs = matched_pairs
    fp_pairs = predicted_pairs - gold_pairs
    fn_pairs = gold_pairs - predicted_pairs
    aspect_metrics = _score_counts(len(tp_pairs), len(fp_pairs), len(fn_pairs))

    per_aspect: dict[str, dict[str, float | int | None]] = {}
    for aspect in sorted(ASPECTS):
        aspect_gold = {pair for pair in gold_pairs if pair[1] == aspect}
        aspect_pred = {pair for pair in predicted_pairs if pair[1] == aspect}
        per_aspect[aspect] = _score_counts(
            len(aspect_gold & aspect_pred), len(aspect_pred - aspect_gold), len(aspect_gold - aspect_pred)
        )

    sentiment_correct = 0
    confusion = {gold_label: {pred_label: 0 for pred_label in SENTIMENTS} for gold_label in SENTIMENTS}
    for review_id, aspect in matched_pairs:
        gold_sentiment = gold[review_id]["findings"][aspect]["sentiment"]
        predicted_sentiment = predictions[review_id][aspect]["sentiment"]
        confusion[gold_sentiment][predicted_sentiment] += 1
        sentiment_correct += gold_sentiment == predicted_sentiment
    sentiment_denominator = len(matched_pairs)
    sentiment_metrics: dict[str, Any] = {
        "correct": sentiment_correct,
        "denominator": sentiment_denominator,
        "accuracy": sentiment_correct / sentiment_denominator if sentiment_denominator else None,
        "confusion_matrix_gold_rows_predicted_columns": confusion,
        "per_class": {},
    }
    for label in SENTIMENTS:
        class_tp = confusion[label][label]
        class_fp = sum(confusion[other][label] for other in SENTIMENTS if other != label)
        class_fn = sum(confusion[label][other] for other in SENTIMENTS if other != label)
        sentiment_metrics["per_class"][label] = _score_counts(class_tp, class_fp, class_fn)

    source_text = dict(zip(reviews["review_id"], reviews["text"], strict=True))
    evaluated_predictions = [
        finding
        for review_id, findings in predictions.items()
        if review_id in annotated_ids
        for finding in findings.values()
    ]
    invalid_quotes = [
        finding for finding in evaluated_predictions if finding["supporting_quote"] not in source_text[finding["review_id"]]
    ]

    errors: list[dict[str, str]] = []
    for review_id, aspect in sorted(fp_pairs):
        errors.append({
            "review_id": review_id,
            "error_type": "false_positive_aspect",
            "aspect": aspect,
            "gold_sentiment": "",
            "predicted_sentiment": predictions[review_id][aspect]["sentiment"],
            "detail": "Predicted aspect is absent from completed gold annotations.",
        })
    for review_id, aspect in sorted(fn_pairs):
        errors.append({
            "review_id": review_id,
            "error_type": "missed_aspect",
            "aspect": aspect,
            "gold_sentiment": gold[review_id]["findings"][aspect]["sentiment"],
            "predicted_sentiment": "",
            "detail": "Gold aspect is absent from predictions.",
        })
    for review_id, aspect in sorted(matched_pairs):
        gold_sentiment = gold[review_id]["findings"][aspect]["sentiment"]
        predicted_sentiment = predictions[review_id][aspect]["sentiment"]
        if gold_sentiment != predicted_sentiment:
            errors.append({
                "review_id": review_id,
                "error_type": "sentiment_mismatch",
                "aspect": aspect,
                "gold_sentiment": gold_sentiment,
                "predicted_sentiment": predicted_sentiment,
                "detail": "Sentiment differs for a correctly matched review/aspect pair.",
            })
    for finding in invalid_quotes:
        errors.append({
            "review_id": finding["review_id"],
            "error_type": "invalid_quote",
            "aspect": finding["aspect"],
            "gold_sentiment": "",
            "predicted_sentiment": finding["sentiment"],
            "detail": "Predicted quote is not an exact substring of the original review text.",
        })
    for review_id, checkpoint in checkpoints.items():
        if checkpoint["status"] == "failed":
            errors.append({
                "review_id": review_id,
                "error_type": "extraction_failure",
                "aspect": "",
                "gold_sentiment": "",
                "predicted_sentiment": "",
                "detail": "Extraction checkpoint status is failed.",
            })

    error_counts = dict(Counter(row["error_type"] for row in errors))
    completed_ids = {key for key, value in checkpoints.items() if value["status"] == "completed"}
    failed_ids = {key for key, value in checkpoints.items() if value["status"] == "failed"}
    attempted_count = len(completed_ids | failed_ids)
    zero_finding_completed = sum(checkpoints[review_id]["aspect_count"] == 0 for review_id in completed_ids)
    operational = {
        "selected_reviews": len(reviews),
        "attempted_reviews": attempted_count,
        "successful_reviews": len(completed_ids),
        "failed_reviews": len(failed_ids),
        "unattempted_reviews": len(reviews) - attempted_count,
        "valid_zero_finding_completed_reviews": zero_finding_completed,
        "extraction_failure_rate": len(failed_ids) / attempted_count if attempted_count else None,
    }
    status_counts = Counter(entry["status"] for entry in gold.values())
    quote_denominator = len(evaluated_predictions)
    report = {
        "prediction_source": manifest["prediction_source"],
        "model_name": manifest["model_name"],
        "extraction_fingerprint": manifest["extraction_fingerprint"],
        "selection_fingerprint": manifest["source_fingerprint"],
        "gold_annotation_status_counts": dict(status_counts),
        "quality_population_reviews": len(annotated_ids),
        "aspect_detection": {"micro": aspect_metrics, "per_aspect": per_aspect},
        "sentiment_classification": sentiment_metrics,
        "quote_quality": {
            "exact_substring_valid": quote_denominator - len(invalid_quotes),
            "denominator": quote_denominator,
            "invalid_quote_count": len(invalid_quotes),
            "exact_substring_rate": (quote_denominator - len(invalid_quotes)) / quote_denominator if quote_denominator else None,
            "excluded_predictions_for_unannotated_or_adjudication_reviews": excluded_prediction_count,
        },
        "operational_reliability": operational,
        "prediction_findings_after_exact_duplicate_collapse": unique_prediction_count,
        "error_counts": error_counts,
    }
    return report, pd.DataFrame(errors, columns=ERROR_COLUMNS)


def write_evaluation_outputs(report: dict[str, Any], errors: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "evaluation_report.json"
    errors_path = output_path / "evaluation_errors.csv"
    for path, writer in [
        (report_path, lambda temp: temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")),
        (errors_path, lambda temp: errors.to_csv(temp, index=False, encoding="utf-8", na_rep="")),
    ]:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        writer(temp_path)
        temp_path.replace(path)
    return report_path, errors_path