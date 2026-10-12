#!/usr/bin/env python3
"""Run a resumable, retry-aware extraction batch over the 500-review sample."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract import extract_review_aspects
from src.schema import AspectFinding, AspectLabel, ReviewExtraction, Sentiment, validate_review_extraction

DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "all_beauty_sample_500.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED_COLUMNS = {"review_id", "rating", "title", "text", "asin", "parent_asin", "user_id", "timestamp"}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_review_id(value: Any) -> str:
    return _safe_str(value)


def _coerce_text(value: Any) -> str:
    return _safe_str(value)


def _determine_error_category(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "transient_network"
    if any(token in message for token in ["429", "rate limit", "too many requests", "temporar", "timeout", "timed out", "service unavailable", "internal error", "server issue", "connection reset"]):
        return "transient_api"
    if any(token in message for token in ["api key", "missing gemini", "permission denied", "forbidden", "authentication", "invalid api key", "unauthorized", "not found"]):
        return "permanent_auth_or_config"
    if any(token in message for token in ["schema", "validation", "quote", "invalid response", "json", "findings", "duplicate", "non-empty"]) or exc.__class__.__name__.lower().endswith("validationerror"):
        return "permanent_validation"
    return "unexpected_error"


def _is_retryable_error(exc: BaseException) -> bool:
    category = _determine_error_category(exc)
    return category in {"transient_network", "transient_api"}


def _safe_error_summary(exc: BaseException) -> str:
    text = str(exc)
    if not text:
        return exc.__class__.__name__
    return text[:200].replace("\n", " ")


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime validation path
        raise ValueError(f"Checkpoint or output file is malformed: {path}: {exc}") from exc


def _write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _checkpoint_columns() -> list[str]:
    return [
        "review_id",
        "status",
        "attempt_count",
        "aspect_count",
        "error_category",
        "error_summary",
        "updated_at",
    ]


def _failure_columns() -> list[str]:
    return [
        "review_id",
        "attempt_count",
        "error_category",
        "error_summary",
        "final_status",
        "updated_at",
    ]


def _ensure_checkpoint_shape(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_checkpoint_columns())
    expected = set(_checkpoint_columns())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Checkpoint file is malformed; missing columns: {sorted(missing)}")
    return df.loc[:, list(_checkpoint_columns())]


def _ensure_failure_shape(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_failure_columns())
    expected = set(_failure_columns())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Failure log is malformed; missing columns: {sorted(missing)}")
    return df.loc[:, list(_failure_columns())]


def _load_output_state(results_path: Path, checkpoint_path: Path, failure_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_df = _read_csv_if_present(results_path)
    checkpoint_df = _ensure_checkpoint_shape(_read_csv_if_present(checkpoint_path))
    failure_df = _ensure_failure_shape(_read_csv_if_present(failure_path))
    return results_df, checkpoint_df, failure_df


def load_review_rows(input_path: str | Path, *, max_reviews: int | None = None) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Load eligible review rows while tracking empty IDs, duplicate IDs, and blank texts."""

    sample_path = Path(input_path)
    df = pd.read_csv(sample_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input sample is missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    duplicate_ids = df["review_id"].fillna("").astype(str).str.strip()
    duplicate_ids = duplicate_ids[duplicate_ids.ne("") & duplicate_ids.duplicated(keep=False)]
    duplicate_set = set(duplicate_ids.tolist())

    for row in df.to_dict(orient="records"):
        review_id = _coerce_review_id(row.get("review_id", ""))
        raw_text = _coerce_text(row.get("text", ""))

        if not review_id:
            invalid_rows.append({
                "review_id": review_id,
                "error_category": "missing_review_id",
                "error_summary": "missing review identifier",
                "final_status": "failed",
            })
            continue

        if review_id in duplicate_set:
            invalid_rows.append({
                "review_id": review_id,
                "error_category": "duplicate_review_id",
                "error_summary": "duplicate review identifier encountered in input",
                "final_status": "failed",
            })
            continue

        if not raw_text:
            invalid_rows.append({
                "review_id": review_id,
                "error_category": "empty_review_text",
                "error_summary": "empty review text",
                "final_status": "failed",
            })
            continue

        records.append(
            {
                "review_id": review_id,
                "rating": row.get("rating"),
                "title": row.get("title"),
                "text": raw_text,
                "asin": _safe_str(row.get("asin", "")),
                "parent_asin": _safe_str(row.get("parent_asin", "")),
                "user_id": _safe_str(row.get("user_id", "")),
                "timestamp": row.get("timestamp"),
                "verified_purchase": row.get("verified_purchase"),
                "helpful_vote": row.get("helpful_vote"),
                "images": row.get("images"),
            }
        )

    if max_reviews is not None:
        records = records[: max_reviews]

    return records, len(records), invalid_rows


def _normalize_aspect_rows(record: dict[str, Any], extraction: ReviewExtraction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finding in extraction.findings:
        validate_review_extraction(extraction, record["text"])
        rows.append(
            {
                "review_id": record["review_id"],
                "asin": record.get("asin", ""),
                "parent_asin": record.get("parent_asin", ""),
                "rating": record.get("rating"),
                "timestamp": record.get("timestamp"),
                "aspect": finding.aspect.value,
                "sentiment": finding.sentiment.value,
                "supporting_quote": finding.supporting_quote,
            }
        )
    return rows


class RetryExhaustedError(RuntimeError):
    """Raised when a retryable error exhausts the configured retry budget."""

    def __init__(self, message: str, *, attempt_count: int):
        super().__init__(message)
        self.attempt_count = attempt_count


def _retry_extraction(
    review_text: str,
    *,
    extractor: Callable[[str], ReviewExtraction],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
) -> tuple[ReviewExtraction, int]:
    last_exc: BaseException | None = None
    for attempt in range(1, max_retries + 2):
        try:
            return extractor(review_text), attempt
        except Exception as exc:  # pragma: no cover - exercised through tests
            last_exc = exc
            if not _is_retryable_error(exc):
                raise
            if attempt > max_retries:
                raise RetryExhaustedError(f"Retry budget exceeded for review after {attempt} attempts.", attempt_count=attempt) from exc
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            time.sleep(delay)
    raise RuntimeError("retry loop exited without a result") from last_exc


def _merge_checkpoint_record(df: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    rows = df.to_dict(orient="records") if not df.empty else []
    filtered = [row for row in rows if str(row.get("review_id", "")) != str(record["review_id"])]
    filtered.append(record)
    return pd.DataFrame(filtered, columns=_checkpoint_columns())


def _merge_failure_record(df: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    rows = df.to_dict(orient="records") if not df.empty else []
    filtered = [row for row in rows if str(row.get("review_id", "")) != str(record["review_id"]) or str(row.get("final_status", "")) != str(record["final_status"])]
    filtered.append(record)
    return pd.DataFrame(filtered, columns=_failure_columns())


def run_batch_extraction(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    max_reviews: int | None = None,
    extractor: Callable[[str], ReviewExtraction] | None = None,
    reset: bool = False,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
) -> dict[str, int]:
    """Process a bounded set of reviews, persisting results and checkpoints safely."""

    sample_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "aspect_sentiment_results.csv"
    checkpoint_path = output_dir / "review_checkpoints.csv"
    failure_path = output_dir / "extraction_failures.csv"

    if reset:
        for candidate in [results_path, checkpoint_path, failure_path]:
            if candidate.exists():
                candidate.unlink()

    results_df, checkpoint_df, failure_df = _load_output_state(results_path, checkpoint_path, failure_path)
    existing_result_keys = set()
    if not results_df.empty:
        for _, row in results_df.iterrows():
            existing_result_keys.add(
                (
                    str(row.get("review_id", "")),
                    str(row.get("aspect", "")),
                    str(row.get("sentiment", "")),
                    str(row.get("supporting_quote", "")),
                )
            )

    checkpoint_by_review = {}
    if not checkpoint_df.empty:
        for _, row in checkpoint_df.iterrows():
            checkpoint_by_review[str(row.get("review_id", ""))] = row.to_dict()

    summary = {
        "eligible_review_count": 0,
        "processed_review_count": 0,
        "zero_finding_count": 0,
        "failed_review_count": 0,
        "skipped_review_count": 0,
        "aspect_row_count": 0,
    }

    records, eligible_count, invalid_rows = load_review_rows(sample_path, max_reviews=max_reviews)
    summary["eligible_review_count"] = eligible_count

    for failure_row in invalid_rows:
        review_id = str(failure_row["review_id"])
        failure_record = {
            "review_id": review_id,
            "attempt_count": 0,
            "error_category": failure_row["error_category"],
            "error_summary": failure_row["error_summary"],
            "final_status": "failed",
            "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        failure_df = _merge_failure_record(failure_df, failure_record)
        _write_csv_atomic(failure_path, failure_df)

        checkpoint_record = {
            "review_id": review_id,
            "status": "failed",
            "attempt_count": 0,
            "aspect_count": 0,
            "error_category": failure_row["error_category"],
            "error_summary": failure_row["error_summary"],
            "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        checkpoint_df = _merge_checkpoint_record(checkpoint_df, checkpoint_record)
        _write_csv_atomic(checkpoint_path, checkpoint_df)
        summary["failed_review_count"] += 1

    completed_ids = set()
    if not checkpoint_df.empty:
        completed_ids = set(str(v) for v in checkpoint_df.loc[checkpoint_df["status"] == "completed", "review_id"].tolist())

    for record in records:
        review_id = str(record["review_id"])
        if review_id in completed_ids:
            summary["skipped_review_count"] += 1
            continue

        attempts = int(checkpoint_by_review.get(review_id, {}).get("attempt_count", 0) or 0)
        current_extractor = extractor or extract_review_aspects
        try:
            extraction, attempts_used = _retry_extraction(
                record["text"],
                extractor=current_extractor,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )
            rows = _normalize_aspect_rows(record, extraction)
            deduped_rows = []
            for row in rows:
                key = (row["review_id"], row["aspect"], row["sentiment"], row["supporting_quote"])
                if key not in existing_result_keys:
                    deduped_rows.append(row)
                    existing_result_keys.add(key)
            if not results_df.empty:
                merged = pd.concat([results_df, pd.DataFrame(deduped_rows)], ignore_index=True)
            else:
                merged = pd.DataFrame(deduped_rows)
            if not merged.empty:
                results_df = merged.loc[:, ["review_id", "asin", "parent_asin", "rating", "timestamp", "aspect", "sentiment", "supporting_quote"]]
            else:
                results_df = pd.DataFrame(columns=["review_id", "asin", "parent_asin", "rating", "timestamp", "aspect", "sentiment", "supporting_quote"])
            _write_csv_atomic(results_path, results_df)

            summary["processed_review_count"] += 1
            summary["aspect_row_count"] = len(results_df)
            if len(rows) == 0:
                summary["zero_finding_count"] += 1

            checkpoint_record = {
                "review_id": review_id,
                "status": "completed",
                "attempt_count": attempts_used,
                "aspect_count": len(rows),
                "error_category": "",
                "error_summary": "",
                "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            checkpoint_df = _merge_checkpoint_record(checkpoint_df, checkpoint_record)
            _write_csv_atomic(checkpoint_path, checkpoint_df)
            completed_ids.add(review_id)
        except Exception as exc:  # pragma: no cover - exercised through tests
            category = _determine_error_category(exc)
            summary["failed_review_count"] += 1
            attempt_count = getattr(exc, "attempt_count", attempts + 1)
            failure_record = {
                "review_id": review_id,
                "attempt_count": int(attempt_count),
                "error_category": category,
                "error_summary": _safe_error_summary(exc),
                "final_status": "failed",
                "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            failure_df = _merge_failure_record(failure_df, failure_record)
            _write_csv_atomic(failure_path, failure_df)

            checkpoint_record = {
                "review_id": review_id,
                "status": "failed",
                "attempt_count": int(attempt_count),
                "aspect_count": 0,
                "error_category": category,
                "error_summary": _safe_error_summary(exc),
                "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            checkpoint_df = _merge_checkpoint_record(checkpoint_df, checkpoint_record)
            _write_csv_atomic(checkpoint_path, checkpoint_df)

    if results_df.empty:
        results_df = pd.DataFrame(columns=["review_id", "asin", "parent_asin", "rating", "timestamp", "aspect", "sentiment", "supporting_quote"])
        _write_csv_atomic(results_path, results_df)
    if checkpoint_df.empty:
        checkpoint_df = pd.DataFrame(columns=_checkpoint_columns())
        _write_csv_atomic(checkpoint_path, checkpoint_df)
    if failure_df.empty:
        failure_df = pd.DataFrame(columns=_failure_columns())
        _write_csv_atomic(failure_path, failure_df)

    summary["aspect_row_count"] = len(results_df)
    print(
        "eligible={eligible} processed={processed} zero_findings={zero} failed={failed} "
        "skipped={skipped} aspect_rows={aspect_rows}".format(
            eligible=summary["eligible_review_count"],
            processed=summary["processed_review_count"],
            zero=summary["zero_finding_count"],
            failed=summary["failed_review_count"],
            skipped=summary["skipped_review_count"],
            aspect_rows=summary["aspect_row_count"],
        )
    )
    return summary


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run a resumable batch extraction over the All Beauty sample.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_SAMPLE_PATH), help="CSV path to the review sample.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for results, checkpoints, and failures.")
    parser.add_argument("--max-reviews", type=int, default=10, help="Maximum number of eligible reviews to process. Use --all to process all 500.")
    parser.add_argument("--all", action="store_true", help="Process all eligible rows in the sample instead of the default safe limit.")
    parser.add_argument("--reset", action="store_true", help="Clear prior checkpoint, output, and failure state before starting.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for transient extraction failures.")
    parser.add_argument("--base-delay", type=float, default=1.0, help="Initial delay between retries in seconds.")
    parser.add_argument("--max-delay", type=float, default=8.0, help="Maximum backoff delay in seconds.")
    args = parser.parse_args()

    if args.all:
        limit = None
    else:
        limit = args.max_reviews

    run_batch_extraction(
        args.input,
        args.output_dir,
        max_reviews=limit,
        reset=args.reset,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
