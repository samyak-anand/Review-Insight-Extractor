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
        invalid["validation_issue"] = "invalid label, ID, or quote"
    if not invalid.empty:
        issues.append(
            f"Aspect results contain {len(invalid)} invalid rows (blank IDs/quotes or non-canonical aspect/sentiment); they are excluded from charts."
        )
    frame = frame.loc[valid].copy()

    if not reviews.empty and not frame.empty:
        known_ids = set(reviews["review_id"])
        orphan_mask = ~frame["review_id"].isin(known_ids)
        if orphan_mask.any():
            orphaned = frame.loc[orphan_mask, RESULT_COLUMNS].copy()
            orphaned["validation_issue"] = "review ID is not present in the source sample"
            invalid = pd.concat([invalid, orphaned], ignore_index=True)
            issues.append(f"Aspect results contain {int(orphan_mask.sum())} rows whose review IDs are not in the source sample; they are excluded.")
            frame = frame.loc[~orphan_mask].copy()

    duplicate_key = ["review_id", "aspect", "sentiment", "supporting_quote"]
    duplicate_rows = frame.duplicated(duplicate_key, keep="first")
    if duplicate_rows.any():
        issues.append(f"Aspect results contain {int(duplicate_rows.sum())} exact duplicate findings; duplicates were collapsed.")
        frame = frame.loc[~duplicate_rows].copy()

    if not reviews.empty:
        source_text = reviews.set_index("review_id")["text"]
        frame["review_text"] = frame["review_id"].map(source_text).fillna("")
        frame["quote_valid"] = frame.apply(
            lambda row: row["supporting_quote"] in row["review_text"], axis=1
        )
        invalid_quote_mask = ~frame["quote_valid"]
        invalid_quote_count = int(invalid_quote_mask.sum())
        if invalid_quote_count:
            invalid_quotes = frame.loc[invalid_quote_mask, RESULT_COLUMNS].copy()
            invalid_quotes["validation_issue"] = "supporting quote is not an exact source-text substring"
            invalid = pd.concat([invalid, invalid_quotes], ignore_index=True)
            issues.append(f"{invalid_quote_count} supporting quotes are not exact substrings of their source reviews.")
            frame = frame.loc[~invalid_quote_mask].copy()
    else:
        frame["review_text"] = ""
        frame["quote_valid"] = pd.NA
    output_columns = [*RESULT_COLUMNS, "review_text", "quote_valid"]
    return frame.reindex(columns=output_columns).reset_index(drop=True), invalid.reset_index(drop=True), available


def _load_checkpoints(
    path: Path,
    issues: list[str],
    known_review_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, bool]:
    required = {"review_id", "status", "attempt_count", "aspect_count"}
    frame, available = _read_csv(path, "Review checkpoints", required, issues, CHECKPOINT_COLUMNS)
    if frame.empty:
        return _empty(CHECKPOINT_COLUMNS), available
    frame = _clean_ids(frame, "Review checkpoints", issues)
    if known_review_ids is not None:
        unknown = ~frame["review_id"].isin(known_review_ids)
        if unknown.any():
            issues.append(f"Review checkpoints contain {int(unknown.sum())} IDs outside the source sample; those rows were excluded.")
            frame = frame.loc[~unknown].copy()
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


def _load_failures(
    path: Path,
    issues: list[str],
    known_review_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, bool]:
    required = {"review_id", "final_status"}
    frame, available = _read_csv(path, "Extraction failure log", required, issues, FAILURE_COLUMNS)
    if frame.empty:
        return _empty(FAILURE_COLUMNS), available
    for column in FAILURE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = _clean_ids(frame, "Extraction failure log", issues)
    if known_review_ids is not None:
        unknown = ~frame["review_id"].isin(known_review_ids)
        if unknown.any():
            issues.append(f"Failure log contains {int(unknown.sum())} IDs outside the source sample; those rows were excluded.")
            frame = frame.loc[~unknown].copy()
    return frame.reset_index(drop=True), available


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
    known_review_ids = set(reviews["review_id"]) if not reviews.empty else None
    checkpoints, checkpoints_available = _load_checkpoints(paths.checkpoints, issues, known_review_ids)
    failures, failures_available = _load_failures(paths.failures, issues, known_review_ids)
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

    if not checkpoints.empty:
        result_counts = data_result_counts(results)
        completed = checkpoints.loc[checkpoints["status"].eq("completed")]
        mismatches = [
            row.review_id
            for row in completed.itertuples()
            if pd.notna(row.aspect_count) and int(row.aspect_count) != result_counts.get(str(row.review_id), 0)
        ]
        if mismatches:
            issues.append(f"{len(mismatches)} completed checkpoints disagree with result-row aspect counts.")

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


def aspect_sentiment_matrix(findings: pd.DataFrame) -> pd.DataFrame:
    """Summarize aspect frequency and sentiment shares over unique review/aspect/sentiment pairs."""

    columns = [
        "aspect",
        "finding_rows",
        "review_count",
        "sentiment_pair_count",
        "positive_review_count",
        "neutral_review_count",
        "negative_review_count",
        "positive_proportion",
        "neutral_proportion",
        "negative_proportion",
    ]
    if findings.empty:
        return pd.DataFrame(columns=columns)

    pairs = findings.drop_duplicates(["review_id", "aspect", "sentiment"])
    rows: list[dict[str, Any]] = []
    for aspect, group in findings.groupby("aspect", sort=True):
        aspect_pairs = pairs.loc[pairs["aspect"].eq(aspect)]
        pair_count = len(aspect_pairs)
        review_total = group["review_id"].nunique()
        counts = aspect_pairs["sentiment"].value_counts()
        rows.append(
            {
                "aspect": aspect,
                "finding_rows": len(group),
                "review_count": int(review_total),
                "sentiment_pair_count": pair_count,
                "positive_review_count": int(aspect_pairs.loc[aspect_pairs["sentiment"].eq("positive"), "review_id"].nunique()),
                "neutral_review_count": int(aspect_pairs.loc[aspect_pairs["sentiment"].eq("neutral"), "review_id"].nunique()),
                "negative_review_count": int(aspect_pairs.loc[aspect_pairs["sentiment"].eq("negative"), "review_id"].nunique()),
                "positive_proportion": float(counts.get("positive", 0) / pair_count) if pair_count else 0.0,
                "neutral_proportion": float(counts.get("neutral", 0) / pair_count) if pair_count else 0.0,
                "negative_proportion": float(counts.get("negative", 0) / pair_count) if pair_count else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def representative_evidence(findings: pd.DataFrame) -> pd.DataFrame:
    """Select one stable source quote per observed aspect/sentiment pair."""

    columns = ["aspect", "sentiment", "review_id", "supporting_quote", "review_text"]
    if findings.empty:
        return pd.DataFrame(columns=columns)
    evidence_findings = findings
    if "quote_valid" in findings:
        evidence_findings = findings.loc[findings["quote_valid"].fillna(False).eq(True)]
    selected_columns = [column for column in columns if column in evidence_findings]
    candidates = evidence_findings.loc[:, selected_columns].drop_duplicates(["review_id", "aspect", "sentiment"])
    candidates = candidates.assign(
        _sentiment_order=candidates["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).fillna(3)
    ).sort_values(["aspect", "_sentiment_order", "review_id", "supporting_quote"])
    result = candidates.groupby(["aspect", "sentiment"], as_index=False, sort=True).head(1)
    return result.drop(columns="_sentiment_order").reset_index(drop=True).reindex(columns=columns)


def complaint_prioritization(
    findings: pd.DataFrame,
    *,
    frequency_weight: int = 60,
    negative_share_weight: int = 40,
) -> pd.DataFrame:
    """Rank aspects using normalized unique negative-review volume and negative sentiment share."""

    if frequency_weight < 0 or negative_share_weight < 0:
        raise ValueError("Complaint weights must be non-negative.")
    if frequency_weight + negative_share_weight == 0:
        raise ValueError("At least one complaint score weight must be greater than zero.")
    matrix = aspect_sentiment_matrix(findings)
    columns = [
        "aspect",
        "negative_review_count",
        "review_count",
        "negative_proportion",
        "normalized_frequency",
        "priority_score",
        "frequency_weight",
        "negative_share_weight",
        "severity_available",
    ]
    if matrix.empty:
        return pd.DataFrame(columns=columns)

    maximum_frequency = int(matrix["negative_review_count"].max())
    matrix["normalized_frequency"] = (
        matrix["negative_review_count"] / maximum_frequency if maximum_frequency else 0.0
    )
    total_weight = frequency_weight + negative_share_weight
    matrix["priority_score"] = 100 * (
        frequency_weight * matrix["normalized_frequency"]
        + negative_share_weight * matrix["negative_proportion"]
    ) / total_weight
    matrix["frequency_weight"] = frequency_weight
    matrix["negative_share_weight"] = negative_share_weight
    matrix["severity_available"] = False
    return matrix.loc[:, columns].sort_values(
        ["priority_score", "negative_review_count", "aspect"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def build_insight_report(priorities: pd.DataFrame) -> dict[str, Any]:
    """Separate measured review facts from cautious interpretation and proposed actions."""

    if priorities.empty:
        return {
            "observed_facts": [],
            "generated_interpretations": [],
            "proposed_recommendations": [],
            "method_note": "No aspect finding rows are available for the current filters.",
        }
    if int(priorities["negative_review_count"].max()) == 0:
        return {
            "observed_facts": ["No negative aspect findings are present in the current filtered rows."],
            "generated_interpretations": ["No aspect has complaint evidence to prioritize in this selection."],
            "proposed_recommendations": [],
            "method_note": "No complaint recommendation is generated without negative review evidence.",
        }
    top = priorities.iloc[0]
    aspect = str(top["aspect"])
    facts = [
        f"{aspect} has {int(top['negative_review_count'])} unique reviews with a negative finding out of {int(top['review_count'])} reviews mentioning the aspect.",
        f"Negative findings account for {float(top['negative_proportion']):.1%} of unique review/aspect/sentiment pairs for {aspect}.",
        f"The configurable frequency/sentiment priority score is {float(top['priority_score']):.1f}/100; no validated severity field is available.",
    ]
    return {
        "observed_facts": facts,
        "generated_interpretations": [
            f"{aspect} ranks highest under the current frequency and negative-share weights in this filtered sample; the score is a prioritization heuristic, not a causal finding."
        ],
        "proposed_recommendations": [
            f"Review the source quotes for {aspect} and check recurrence across products before deciding whether further investigation is warranted."
        ],
        "method_note": "Facts are computed from the currently filtered aspect rows. Interpretations and recommendations are deterministic text, not model-generated conclusions.",
    }


def insight_report_markdown(report: dict[str, Any], evidence: pd.DataFrame | None = None) -> str:
    """Format a labeled, reproducible insight report without adding unsupported claims."""

    sections = ["# Evidence-backed insight report"]
    for title, key in [
        ("Observed facts", "observed_facts"),
        ("Generated interpretations", "generated_interpretations"),
        ("Proposed recommendations", "proposed_recommendations"),
    ]:
        sections.append(f"\n## {title}")
        items = report.get(key, [])
        sections.extend(f"- {item}" for item in items)
        if not items:
            sections.append("- No supported findings for the current filters.")
    if evidence is not None and not evidence.empty:
        sections.append("\n## Representative source evidence")
        for row in evidence.itertuples():
            sections.append(f"- {row.aspect} ({row.sentiment}), review {row.review_id}: \"{row.supporting_quote}\"")
    sections.append(f"\n_Method: {report.get('method_note', 'Method not provided.')}_")
    return "\n".join(sections) + "\n"


def findings_csv_bytes(findings: pd.DataFrame) -> bytes:
    """Serialize current filtered finding rows for a safe CSV download."""

    return findings.to_csv(index=False, encoding="utf-8").encode("utf-8")


def data_result_counts(findings: pd.DataFrame) -> dict[str, int]:
    """Return aspect-row counts keyed by review ID for checkpoint reconciliation."""

    if findings.empty:
        return {}
    return findings.groupby("review_id").size().astype(int).to_dict()


def data_quality_summary(data: DashboardData) -> dict[str, Any]:
    """Summarize data validity and result/checkpoint consistency from loaded artifacts."""

    quote_mismatches = (
        int(
            data.invalid_results.get("validation_issue", pd.Series(dtype=str))
            .eq("supporting quote is not an exact source-text substring")
            .sum()
        )
        if data.available["results"]
        else None
    )
    checkpoint_mismatches: list[dict[str, Any]] = []
    result_counts = data_result_counts(data.results)
    if not data.checkpoints.empty:
        for row in data.checkpoints.loc[data.checkpoints["status"].eq("completed")].itertuples():
            if pd.isna(row.aspect_count):
                continue
            observed = result_counts.get(str(row.review_id), 0)
            if int(row.aspect_count) != observed:
                checkpoint_mismatches.append(
                    {"review_id": str(row.review_id), "checkpoint_aspect_count": int(row.aspect_count), "result_rows": observed}
                )
    return {
        "source_reviews": int(data.reviews["review_id"].nunique()) if "review_id" in data.reviews else 0,
        "valid_finding_rows": len(data.results),
        "invalid_result_rows": len(data.invalid_results),
        "invalid_exact_quotes": quote_mismatches,
        "checkpoint_result_mismatches": checkpoint_mismatches,
        "source_metadata_fields": [column for column in REVIEW_METADATA_COLUMNS if column in data.reviews],
        "validated_severity_available": False,
    }


def review_count(findings: pd.DataFrame) -> int:
    """Count unique reviews represented by finding rows, never aspect rows."""

    return int(findings["review_id"].nunique()) if "review_id" in findings else 0


def pipeline_summary(data: DashboardData) -> dict[str, int | str | None]:
    """Compute distinct-review operational counts from checkpoints when available."""

    sample_count = int(data.reviews["review_id"].nunique()) if "review_id" in data.reviews else 0
    if not data.available["checkpoints"] or data.checkpoints.empty:
        completed = failed = attempted = recorded_attempts = None
        progress = "Checkpoint status unavailable"
    else:
        completed = int(data.checkpoints["status"].eq("completed").sum())
        failed = int(data.checkpoints["status"].eq("failed").sum())
        attempted = int(data.checkpoints["status"].isin(["completed", "failed"]).sum())
        recorded_attempts = int(pd.to_numeric(data.checkpoints["attempt_count"], errors="coerce").fillna(0).sum())
        if sample_count and attempted >= sample_count:
            progress = "Checkpointed for all sample reviews"
        else:
            progress = f"{attempted} of {sample_count} sample reviews checkpointed"
    return {
        "sample_reviews": sample_count,
        "attempted_reviews": attempted,
        "recorded_attempts": recorded_attempts,
        "successful_reviews": completed,
        "failed_reviews": failed,
        "aspect_finding_rows": len(data.results),
        "pipeline_progress": progress,
    }