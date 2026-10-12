"""Load, validate, filter, and summarize dashboard artifacts without UI dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import AspectLabel, Sentiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_COLUMNS = ["review_id", "asin", "parent_asin", "rating", "timestamp", "aspect", "sentiment", "supporting_quote"]
CHECKPOINT_COLUMNS = ["review_id", "status", "attempt_count", "aspect_count", "error_category", "error_summary", "updated_at"]
FAILURE_COLUMNS = ["review_id", "attempt_count", "error_category", "error_summary", "final_status", "updated_at"]
REVIEW_COLUMNS = ["review_id", "text"]
REVIEW_METADATA_COLUMNS = ["rating", "asin", "parent_asin", "timestamp"]
ASPECT_LABELS = [label.value for label in AspectLabel]
SENTIMENT_LABELS = [label.value for label in Sentiment]


class DashboardDataError(ValueError):
    """Raised when a present dashboard artifact violates its required schema."""


@dataclass(frozen=True)
class DashboardPaths:
    """Repository-relative artifact locations used by the dashboard."""

    results: Path = PROJECT_ROOT / "data/processed/aspect_sentiment_results.csv"
    checkpoints: Path = PROJECT_ROOT / "data/processed/review_checkpoints.csv"
    failures: Path = PROJECT_ROOT / "data/processed/extraction_failures.csv"
    reviews: Path = PROJECT_ROOT / "data/raw/all_beauty_sample_500.csv"
    evaluation_report: Path = PROJECT_ROOT / "data/evaluation/reports/evaluation_report.json"
    evaluation_errors: Path = PROJECT_ROOT / "data/evaluation/reports/evaluation_errors.csv"


@dataclass
class DashboardData:
    """Validated source and pipeline data, plus explicit loading diagnostics."""

    results: pd.DataFrame
    checkpoints: pd.DataFrame
    failures: pd.DataFrame
    reviews: pd.DataFrame
    evaluation_report: dict[str, Any] | None
    evaluation_errors: pd.DataFrame
    invalid_results: pd.DataFrame
    available: dict[str, bool]
    issues: list[str]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_csv(path: Path, name: str, required: set[str], issues: list[str], columns: list[str]) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        issues.append(f"{name} is not available at {path}.")
        return _empty(columns), False
    try:
        if path.stat().st_size == 0:
            issues.append(f"{name} exists but is empty.")
            return _empty(columns), True
        frame = pd.read_csv(path, dtype={"review_id": "string"}, encoding="utf-8")
    except pd.errors.EmptyDataError:
        issues.append(f"{name} exists but has no CSV header or rows.")
        return _empty(columns), True
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DashboardDataError(f"Could not read {name} at {path}: {exc}") from exc

    missing = sorted(required - set(frame.columns))
    if missing:
        raise DashboardDataError(f"{name} is missing required columns: {', '.join(missing)}")
    return frame, True


def _clean_ids(frame: pd.DataFrame, name: str, issues: list[str]) -> pd.DataFrame:
    if frame.empty or "review_id" not in frame:
        return frame
    cleaned = frame.copy()
    cleaned["review_id"] = cleaned["review_id"].fillna("").astype(str).str.strip()
    blank = cleaned["review_id"].eq("")
    if blank.any():
        issues.append(f"{name} contains {int(blank.sum())} rows with blank review IDs; those rows were excluded.")
        cleaned = cleaned.loc[~blank].copy()
    duplicates = cleaned["review_id"].duplicated(keep="last")
    if duplicates.any():
        issues.append(f"{name} contains duplicate review IDs; the last row per ID was retained for review-level counts.")
        cleaned = cleaned.loc[~duplicates].copy()
    return cleaned.reset_index(drop=True)


def _load_reviews(path: Path, issues: list[str]) -> tuple[pd.DataFrame, bool]:
    frame, available = _read_csv(path, "Review sample", {"review_id", "text"}, issues, REVIEW_COLUMNS)
    if frame.empty:
        return _empty(REVIEW_COLUMNS), available
    frame = _clean_ids(frame, "Review sample", issues)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame.loc[frame["text"].str.strip().ne("")].copy()
    retained_columns = [column for column in [*REVIEW_COLUMNS, *REVIEW_METADATA_COLUMNS] if column in frame]
    frame = frame.loc[:, retained_columns]
    return frame.reset_index(drop=True), available


def _load_results(path: Path, reviews: pd.DataFrame, issues: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    required = {"review_id", "aspect", "sentiment", "supporting_quote"}
    frame, available = _read_csv(path, "Aspect results", required, issues, RESULT_COLUMNS)
    if frame.empty:
        return _empty(RESULT_COLUMNS + ["review_text", "extraction_status"]), _empty(RESULT_COLUMNS), available

    frame = frame.copy()
    frame["review_id"] = frame["review_id"].fillna("").astype(str).str.strip()
    for column in RESULT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame["aspect"] = frame["aspect"].fillna("").astype(str).str.strip()
    frame["sentiment"] = frame["sentiment"].fillna("").astype(str).str.strip()
    frame["supporting_quote"] = frame["supporting_quote"].fillna("").astype(str)
    aspect_valid = frame["aspect"].isin(ASPECT_LABELS)
    sentiment_valid = frame["sentiment"].isin(SENTIMENT_LABELS)
    id_valid = frame["review_id"].ne("")
    quote_valid = frame["supporting_quote"].str.strip().ne("")
    valid = aspect_valid & sentiment_valid & id_valid & quote_valid
    invalid = frame.loc[~valid, RESULT_COLUMNS].copy()
    if not invalid.empty:
        issues.append(
            f"Aspect results contain {len(invalid)} invalid rows (blank IDs/quotes or non-canonical aspect/sentiment); they are excluded from charts."
        )
    frame = frame.loc[valid].copy()
    duplicate_key = ["review_id", "aspect", "sentiment", "supporting_quote"]
    duplicate_rows = frame.duplicated(duplicate_key, keep="first")
    if duplicate_rows.any():
        issues.append(f"Aspect results contain {int(duplicate_rows.sum())} exact duplicate findings; duplicates were collapsed.")
        frame = frame.loc[~duplicate_rows].copy()

    if not reviews.empty:
        source_text = reviews.set_index("review_id")["text"]
        frame["review_text"] = frame["review_id"].map(source_text).fillna("")
    else:
        frame["review_text"] = ""
    return frame.reset_index(drop=True), invalid.reset_index(drop=True), available


def _load_checkpoints(path: Path, issues: list[str]) -> tuple[pd.DataFrame, bool]:
    required = {"review_id", "status", "attempt_count", "aspect_count"}
    frame, available = _read_csv(path, "Review checkpoints", required, issues, CHECKPOINT_COLUMNS)
    if frame.empty:
        return _empty(CHECKPOINT_COLUMNS), available
    frame = _clean_ids(frame, "Review checkpoints", issues)
    for column in CHECKPOINT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame["status"] = frame["status"].fillna("unknown").astype(str).str.strip()
    frame["attempt_count"] = pd.to_numeric(frame["attempt_count"], errors="coerce")
    frame["aspect_count"] = pd.to_numeric(frame["aspect_count"], errors="coerce")
    invalid_status = ~frame["status"].isin(["completed", "failed"])
    if invalid_status.any():
        issues.append(f"Review checkpoints contain {int(invalid_status.sum())} unrecognized status values.")
    invalid_counts = frame["attempt_count"].isna() | frame["aspect_count"].isna()
    if invalid_counts.any():
        issues.append(f"Review checkpoints contain {int(invalid_counts.sum())} rows with invalid attempt/aspect counts.")
    return frame.reset_index(drop=True), available


def _load_failures(path: Path, issues: list[str]) -> tuple[pd.DataFrame, bool]:
    required = {"review_id", "final_status"}
    frame, available = _read_csv(path, "Extraction failure log", required, issues, FAILURE_COLUMNS)
    if frame.empty:
        return _empty(FAILURE_COLUMNS), available
    for column in FAILURE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return _clean_ids(frame, "Extraction failure log", issues), available


def _load_evaluation_report(path: Path, issues: list[str]) -> tuple[dict[str, Any] | None, bool]:
    if not path.exists():
        return None, False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(f"Evaluation report could not be read: {exc}")
        return None, True
    if not isinstance(value, dict):
        issues.append("Evaluation report must contain a JSON object; it was ignored.")
        return None, True
    return value, True


def load_dashboard_data(paths: DashboardPaths = DashboardPaths()) -> DashboardData:
    """Load available artifacts from explicit, repository-rooted paths."""

    issues: list[str] = []
    reviews, reviews_available = _load_reviews(paths.reviews, issues)
    results, invalid_results, results_available = _load_results(paths.results, reviews, issues)
    checkpoints, checkpoints_available = _load_checkpoints(paths.checkpoints, issues)
    failures, failures_available = _load_failures(paths.failures, issues)
    evaluation_report, evaluation_available = _load_evaluation_report(paths.evaluation_report, issues)
    evaluation_errors, evaluation_errors_available = _read_csv(
        paths.evaluation_errors,
        "Evaluation error analysis",
        {"review_id", "error_type"},
        issues,
        ["review_id", "error_type"],
    )
    available = {
        "reviews": reviews_available,
        "results": results_available,
        "checkpoints": checkpoints_available,
        "failures": failures_available,
        "evaluation": evaluation_available,
        "evaluation_errors": evaluation_errors_available,
    }

    if not results.empty:
        status_map = checkpoints.set_index("review_id")["status"] if not checkpoints.empty else pd.Series(dtype="object")
        results["extraction_status"] = results["review_id"].map(status_map).fillna("unknown")
        if not reviews.empty:
            source_fields = [column for column in ["review_id", "rating", "asin", "parent_asin", "timestamp"] if column in reviews]
            source = reviews.loc[:, source_fields].drop_duplicates("review_id", keep="last").set_index("review_id")
            for column in ["rating", "asin", "parent_asin", "timestamp"]:
                if column in source and (column not in results or results[column].isna().all()):
                    results[column] = results["review_id"].map(source[column])

    return DashboardData(
        results=results,
        checkpoints=checkpoints,
        failures=failures,
        reviews=reviews,
        evaluation_report=evaluation_report,
        evaluation_errors=evaluation_errors,
        invalid_results=invalid_results,
        available=available,
        issues=issues,
    )


def filter_findings(
    findings: pd.DataFrame,
    *,
    aspect: str | None = None,
    sentiment: str | None = None,
    asin: str | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
    extraction_status: str | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """Apply conjunctive filters to aspect rows; absent metadata filters are no-ops."""

    filtered = findings.copy()
    if aspect and aspect != "All":
        filtered = filtered.loc[filtered["aspect"].eq(aspect)]
    if sentiment and sentiment != "All":
        filtered = filtered.loc[filtered["sentiment"].eq(sentiment)]
    if asin and asin != "All" and "asin" in filtered:
        filtered = filtered.loc[filtered["asin"].fillna("").astype(str).eq(asin)]
    if rating_min is not None and "rating" in filtered:
        ratings = pd.to_numeric(filtered["rating"], errors="coerce")
        filtered = filtered.loc[ratings.ge(rating_min)]
    if rating_max is not None and "rating" in filtered:
        ratings = pd.to_numeric(filtered["rating"], errors="coerce")
        filtered = filtered.loc[ratings.le(rating_max)]
    if extraction_status and extraction_status != "All" and "extraction_status" in filtered:
        filtered = filtered.loc[filtered["extraction_status"].eq(extraction_status)]
    if search and search.strip():
        query = search.strip().casefold()
        searchable = [column for column in ["review_id", "supporting_quote", "review_text", "asin", "parent_asin"] if column in filtered]
        if searchable:
            matches = pd.Series(False, index=filtered.index)
            for column in searchable:
                matches |= filtered[column].fillna("").astype(str).str.casefold().str.contains(query, regex=False)
            filtered = filtered.loc[matches]
    return filtered.reset_index(drop=True)


def aspect_frequency(findings: pd.DataFrame) -> pd.DataFrame:
    """Count aspect-level finding rows, not reviews."""

    if findings.empty:
        return pd.DataFrame(columns=["aspect", "finding_count"])
    return (
        findings.groupby("aspect", as_index=False)
        .size()
        .rename(columns={"size": "finding_count"})
        .sort_values(["finding_count", "aspect"], ascending=[False, True], ignore_index=True)
    )


def sentiment_distribution(findings: pd.DataFrame) -> pd.DataFrame:
    """Count canonical sentiment labels across aspect finding rows."""

    counts = findings["sentiment"].value_counts() if not findings.empty else pd.Series(dtype="int64")
    return pd.DataFrame({"sentiment": SENTIMENT_LABELS, "finding_count": [int(counts.get(label, 0)) for label in SENTIMENT_LABELS]})


def aspect_sentiment_counts(findings: pd.DataFrame) -> pd.DataFrame:
    """Return finding counts for each observed aspect and canonical sentiment."""

    if findings.empty:
        return pd.DataFrame(columns=["aspect", *SENTIMENT_LABELS])
    counts = pd.crosstab(findings["aspect"], findings["sentiment"]).reindex(columns=SENTIMENT_LABELS, fill_value=0)
    counts.index.name = "aspect"
    return counts.reset_index()


def review_count(findings: pd.DataFrame) -> int:
    """Count unique reviews represented by finding rows, never aspect rows."""

    return int(findings["review_id"].nunique()) if "review_id" in findings else 0


def pipeline_summary(data: DashboardData) -> dict[str, int | str | None]:
    """Compute distinct-review operational counts from checkpoints when available."""

    sample_count = int(data.reviews["review_id"].nunique()) if "review_id" in data.reviews else 0
    if not data.available["checkpoints"] or data.checkpoints.empty:
        completed = failed = attempted = None
        progress = "Checkpoint status unavailable"
    else:
        completed = int(data.checkpoints["status"].eq("completed").sum())
        failed = int(data.checkpoints["status"].eq("failed").sum())
        attempted = int(data.checkpoints["status"].isin(["completed", "failed"]).sum())
        if sample_count and attempted >= sample_count:
            progress = "Checkpointed for all sample reviews"
        else:
            progress = f"{attempted} of {sample_count} sample reviews checkpointed"
    return {
        "sample_reviews": sample_count,
        "attempted_reviews": attempted,
        "successful_reviews": completed,
        "failed_reviews": failed,
        "aspect_finding_rows": len(data.results),
        "pipeline_progress": progress,
    }