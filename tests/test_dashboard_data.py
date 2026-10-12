from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard_data import (
    DashboardDataError,
    DashboardPaths,
    aspect_frequency,
    aspect_sentiment_counts,
    aspect_sentiment_matrix,
    build_insight_report,
    complaint_prioritization,
    data_quality_summary,
    findings_csv_bytes,
    filter_findings,
    insight_report_markdown,
    load_dashboard_data,
    pipeline_summary,
    representative_evidence,
    review_count,
    sentiment_distribution,
)


def make_paths(root: Path) -> DashboardPaths:
    return DashboardPaths(
        results=root / "processed/aspect_sentiment_results.csv",
        checkpoints=root / "processed/review_checkpoints.csv",
        failures=root / "processed/extraction_failures.csv",
        reviews=root / "raw/reviews.csv",
        evaluation_report=root / "evaluation/evaluation_report.json",
        evaluation_errors=root / "evaluation/evaluation_errors.csv",
    )


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8")


def sample_reviews() -> list[dict[str, object]]:
    return [
        {"review_id": "r1", "text": "The scent is lovely and the cream feels smooth.", "rating": 5, "asin": "A1", "parent_asin": "P1"},
        {"review_id": "r2", "text": "Delivery was late.", "rating": 2, "asin": "A2", "parent_asin": "P2"},
    ]


def sample_results() -> list[dict[str, object]]:
    return [
        {"review_id": "r1", "asin": "A1", "parent_asin": "P1", "rating": 5, "timestamp": "2024-01-01", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "scent is lovely"},
        {"review_id": "r1", "asin": "A1", "parent_asin": "P1", "rating": 5, "timestamp": "2024-01-01", "aspect": "Texture", "sentiment": "positive", "supporting_quote": "feels smooth"},
        {"review_id": "r2", "asin": "A2", "parent_asin": "P2", "rating": 2, "timestamp": "2024-01-02", "aspect": "Delivery", "sentiment": "negative", "supporting_quote": "Delivery was late"},
    ]


def write_base_artifacts(paths: DashboardPaths) -> None:
    write_csv(paths.reviews, sample_reviews())
    write_csv(paths.results, sample_results())
    write_csv(
        paths.checkpoints,
        [
            {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 2, "error_category": "", "error_summary": "", "updated_at": "2024-01-01T00:00:00Z"},
            {"review_id": "r2", "status": "failed", "attempt_count": 4, "aspect_count": 0, "error_category": "transient_api", "error_summary": "rate limited", "updated_at": "2024-01-01T00:00:00Z"},
        ],
    )
    write_csv(paths.failures, [{"review_id": "r2", "attempt_count": 4, "error_category": "transient_api", "error_summary": "rate limited", "final_status": "failed", "updated_at": "2024-01-01T00:00:00Z"}])


def test_valid_artifacts_load_with_stage4_contract(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_base_artifacts(paths)
    paths.evaluation_report.parent.mkdir(parents=True, exist_ok=True)
    paths.evaluation_report.write_text(json.dumps({"prediction_source": "live", "quality_population_reviews": 7}), encoding="utf-8")
    write_csv(paths.evaluation_errors, [{"review_id": "r1", "error_type": "missed_aspect"}])

    data = load_dashboard_data(paths)
    assert len(data.results) == 3
    assert data.results["review_text"].notna().all()
    assert data.results["extraction_status"].tolist() == ["completed", "completed", "failed"]
    assert "user_id" not in data.reviews.columns
    assert data.evaluation_report["quality_population_reviews"] == 7
    assert len(data.evaluation_errors) == 1
    assert data.issues == []
    assert pipeline_summary(data)["recorded_attempts"] == 5


def test_missing_files_return_empty_artifacts_with_actionable_state(tmp_path: Path) -> None:
    data = load_dashboard_data(make_paths(tmp_path))
    assert data.results.empty
    assert data.checkpoints.empty
    assert data.evaluation_report is None
    assert not data.available["results"]
    assert any("Aspect results is not available" in issue for issue in data.issues)


def test_empty_artifact_files_do_not_crash(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    for path in [paths.reviews, paths.results, paths.checkpoints, paths.failures, paths.evaluation_errors]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    data = load_dashboard_data(paths)
    assert data.results.empty
    assert data.reviews.empty
    assert any("exists but is empty" in issue for issue in data.issues)


def test_missing_required_columns_raise_actionable_error(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_csv(paths.reviews, [{"review_id": "r1", "text": "Text"}])
    write_csv(paths.results, [{"review_id": "r1", "sentiment": "positive"}])
    with pytest.raises(DashboardDataError, match="missing required columns.*aspect"):
        load_dashboard_data(paths)


def test_unexpected_aspect_or_sentiment_is_excluded_and_reported(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_csv(paths.reviews, sample_reviews())
    rows = sample_results() + [
        {"review_id": "r1", "aspect": "Fragrance", "sentiment": "positive", "supporting_quote": "lovely"},
        {"review_id": "r2", "aspect": "Delivery", "sentiment": "mixed", "supporting_quote": "late"},
    ]
    write_csv(paths.results, rows)
    data = load_dashboard_data(paths)
    assert len(data.results) == 3
    assert len(data.invalid_results) == 2
    assert any("non-canonical aspect/sentiment" in issue for issue in data.issues)


def test_combined_filters_apply_conjunctively(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_base_artifacts(paths)
    findings = load_dashboard_data(paths).results
    filtered = filter_findings(
        findings,
        aspect="Scent",
        sentiment="positive",
        asin="A1",
        rating_min=4,
        rating_max=5,
        extraction_status="completed",
        search="lovely",
    )
    assert filtered["review_id"].tolist() == ["r1"]
    assert filtered["aspect"].tolist() == ["Scent"]
    assert filter_findings(findings, asin="A2", rating_min=4).empty


def test_aspect_sentiment_aggregates_count_finding_rows() -> None:
    findings = pd.DataFrame(sample_results())
    assert aspect_frequency(findings).to_dict("records") == [
        {"aspect": "Delivery", "finding_count": 1},
        {"aspect": "Scent", "finding_count": 1},
        {"aspect": "Texture", "finding_count": 1},
    ]
    assert sentiment_distribution(findings).set_index("sentiment")["finding_count"].to_dict() == {
        "positive": 2, "neutral": 0, "negative": 1
    }
    matrix = aspect_sentiment_counts(findings).set_index("aspect")
    assert matrix.loc["Scent", "positive"] == 1
    assert matrix.loc["Delivery", "negative"] == 1


def test_duplicate_review_ids_and_multi_aspect_rows_do_not_inflate_review_counts(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_csv(paths.reviews, sample_reviews() + [sample_reviews()[0]])
    write_csv(paths.results, sample_results())
    checkpoints = [
        {"review_id": "r1", "status": "failed", "attempt_count": 1, "aspect_count": 0},
        {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 2},
    ]
    write_csv(paths.checkpoints, checkpoints)

    data = load_dashboard_data(paths)
    summary = pipeline_summary(data)
    assert len(data.reviews) == 2
    assert len(data.results.loc[data.results["review_id"].eq("r1")]) == 2
    assert review_count(data.results) == 2
    assert summary["sample_reviews"] == 2
    assert summary["successful_reviews"] == 1
    assert summary["failed_reviews"] == 0
    assert any("duplicate review IDs" in issue for issue in data.issues)


def test_missing_optional_result_metadata_is_supported(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_csv(paths.reviews, sample_reviews())
    write_csv(paths.results, [{"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "lovely"}])
    data = load_dashboard_data(paths)
    assert data.results.iloc[0]["asin"] == "A1"
    assert data.results.iloc[0]["review_text"].startswith("The scent")
    assert data.results.iloc[0]["extraction_status"] == "unknown"


def test_missing_evaluation_artifacts_are_unavailable_not_fabricated(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_base_artifacts(paths)
    data = load_dashboard_data(paths)
    assert data.evaluation_report is None
    assert data.evaluation_errors.empty
    assert not data.available["evaluation"]
    assert not data.available["evaluation_errors"]


def priority_findings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"review_id": "r1", "aspect": "Scent", "sentiment": "negative", "supporting_quote": "bad scent", "review_text": "The bad scent is overwhelming."},
            {"review_id": "r1", "aspect": "Scent", "sentiment": "negative", "supporting_quote": "scent is overwhelming", "review_text": "The bad scent is overwhelming."},
            {"review_id": "r2", "aspect": "Scent", "sentiment": "negative", "supporting_quote": "scent burns", "review_text": "The scent burns."},
            {"review_id": "r3", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "nice scent", "review_text": "I like the nice scent."},
            {"review_id": "r4", "aspect": "Texture", "sentiment": "negative", "supporting_quote": "rough texture", "review_text": "The rough texture bothers me."},
        ]
    )


def test_aspect_matrix_uses_unique_review_sentiment_pairs() -> None:
    matrix = aspect_sentiment_matrix(priority_findings()).set_index("aspect")
    scent = matrix.loc["Scent"]
    assert scent["finding_rows"] == 4
    assert scent["review_count"] == 3
    assert scent["sentiment_pair_count"] == 3
    assert scent["negative_review_count"] == 2
    assert scent["positive_proportion"] == pytest.approx(1 / 3)
    assert scent["negative_proportion"] == pytest.approx(2 / 3)


def test_complaint_priority_is_weighted_transparent_and_unique_review_based() -> None:
    priorities = complaint_prioritization(priority_findings(), frequency_weight=60, negative_share_weight=40)
    assert priorities.iloc[0]["aspect"] == "Scent"
    assert priorities.iloc[0]["negative_review_count"] == 2
    assert priorities.iloc[0]["priority_score"] == pytest.approx(100 * (0.6 + 0.4 * (2 / 3)))
    assert not priorities["severity_available"].any()
    with pytest.raises(ValueError, match="At least one"):
        complaint_prioritization(priority_findings(), frequency_weight=0, negative_share_weight=0)


def test_representative_evidence_and_insight_report_keep_claim_types_separate() -> None:
    findings = priority_findings()
    evidence = representative_evidence(findings)
    scent_negative = evidence.loc[evidence["aspect"].eq("Scent") & evidence["sentiment"].eq("negative")].iloc[0]
    assert scent_negative["review_id"] == "r1"
    assert scent_negative["supporting_quote"] == "bad scent"
    assert scent_negative["supporting_quote"] in scent_negative["review_text"]

    report = build_insight_report(complaint_prioritization(findings))
    markdown = insight_report_markdown(report, evidence)
    assert "## Observed facts" in markdown
    assert "## Generated interpretations" in markdown
    assert "## Proposed recommendations" in markdown
    assert "not a causal finding" in markdown
    assert 'Scent (negative), review r1: "bad scent"' in markdown


def test_insight_report_does_not_infer_a_complaint_without_negative_evidence() -> None:
    findings = pd.DataFrame(
        [{"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "nice scent"}]
    )
    report = build_insight_report(complaint_prioritization(findings))
    assert report["proposed_recommendations"] == []
    assert "No negative aspect findings" in report["observed_facts"][0]


def test_filtered_findings_export_is_valid_utf8_csv() -> None:
    findings = priority_findings().iloc[:1]
    payload = findings_csv_bytes(findings)
    exported = pd.read_csv(pd.io.common.BytesIO(payload))
    assert exported.loc[0, "review_id"] == "r1"
    assert exported.loc[0, "supporting_quote"] == "bad scent"


def test_data_quality_reports_invalid_quotes_orphan_ids_and_count_mismatches(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    write_csv(paths.reviews, [{"review_id": "r1", "text": "The scent is nice."}])
    write_csv(
        paths.results,
        [
            {"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "not exact"},
            {"review_id": "outside", "aspect": "Texture", "sentiment": "negative", "supporting_quote": "rough"},
        ],
    )
    write_csv(paths.checkpoints, [
        {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 2},
        {"review_id": "outside", "status": "completed", "attempt_count": 1, "aspect_count": 1},
    ])
    data = load_dashboard_data(paths)
    quality = data_quality_summary(data)
    assert data.results.empty
    assert len(data.invalid_results) == 2
    assert quality["invalid_exact_quotes"] == 1
    assert quality["validated_severity_available"] is False
    assert quality["checkpoint_result_mismatches"] == [
        {"review_id": "r1", "checkpoint_aspect_count": 2, "result_rows": 0}
    ]
    assert any("outside the source sample" in issue for issue in data.issues)