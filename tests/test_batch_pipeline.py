from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_batch_extraction.py"
SPEC = importlib.util.spec_from_file_location("run_batch_extraction", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load batch extraction module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

run_batch_extraction = MODULE.run_batch_extraction
load_review_rows = MODULE.load_review_rows

def make_sample_csv(path: Path, records: list[dict[str, object]]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def test_successful_review_with_multiple_aspects(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {
                "review_id": "r1",
                "rating": 5,
                "title": "Great",
                "text": "I loved the scent and it worked well. The bottle leaked in transit.",
                "asin": "A1",
                "parent_asin": "A1",
                "user_id": "u1",
                "timestamp": "2024-01-01 00:00:00",
                "verified_purchase": True,
                "helpful_vote": 0,
                "images": "[]",
            }
        ],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction.model_validate(
            {
                "findings": [
                    {
                        "aspect": "Scent",
                        "sentiment": "positive",
                        "supporting_quote": "loved the scent",
                    },
                    {
                        "aspect": "Packaging",
                        "sentiment": "negative",
                        "supporting_quote": "bottle leaked in transit",
                    },
                ]
            }
        )

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True)

    assert result["processed_review_count"] == 1
    assert result["aspect_row_count"] == 2
    output = pd.read_csv(tmp_path / "out" / "aspect_sentiment_results.csv")
    assert output["review_id"].tolist() == ["r1", "r1"]
    assert set(output["aspect"].tolist()) == {"Scent", "Packaging"}


def test_successful_review_with_zero_findings(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {
                "review_id": "r1",
                "rating": 3,
                "title": "Okay",
                "text": "It was fine.",
                "asin": "A2",
                "parent_asin": "A2",
                "user_id": "u2",
                "timestamp": "2024-01-01 00:00:00",
                "verified_purchase": True,
                "helpful_vote": 0,
                "images": "[]",
            }
        ],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction(findings=[])

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True)
    assert result["zero_finding_count"] == 1
    assert result["processed_review_count"] == 1
    checkpoint = pd.read_csv(tmp_path / "out" / "review_checkpoints.csv")
    assert checkpoint.iloc[0]["status"] == "completed"


def test_transient_error_followed_by_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {
                "review_id": "r1",
                "rating": 4,
                "title": "Good",
                "text": "The product works well.",
                "asin": "A3",
                "parent_asin": "A3",
                "user_id": "u3",
                "timestamp": "2024-01-01 00:00:00",
                "verified_purchase": True,
                "helpful_vote": 0,
                "images": "[]",
            }
        ],
    )
    call_count = {"n": 0}

    def extractor(review_text: str):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("429 rate limit")
        return MODULE.ReviewExtraction.model_validate({
            "findings": [
                {"aspect": "Product effectiveness", "sentiment": "positive", "supporting_quote": "works well"}
            ]
        })

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True, max_retries=2, base_delay=0.0)
    assert result["processed_review_count"] == 1
    assert call_count["n"] == 2


def test_retry_exhaustion_marks_review_failed(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {
                "review_id": "r1",
                "rating": 2,
                "title": "Bad",
                "text": "The product failed.",
                "asin": "A4",
                "parent_asin": "A4",
                "user_id": "u4",
                "timestamp": "2024-01-01 00:00:00",
                "verified_purchase": True,
                "helpful_vote": 0,
                "images": "[]",
            }
        ],
    )

    def extractor(review_text: str):
        raise RuntimeError("503 temporary server issue")

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True, max_retries=2, base_delay=0.0)
    assert result["failed_review_count"] == 1
    failures = pd.read_csv(tmp_path / "out" / "extraction_failures.csv")
    assert failures.iloc[0]["final_status"] == "failed"
    assert int(failures.iloc[0]["attempt_count"]) >= 2


def test_permanent_error_not_retried(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {
                "review_id": "r1",
                "rating": 1,
                "title": "Nope",
                "text": "I hate this product.",
                "asin": "A5",
                "parent_asin": "A5",
                "user_id": "u5",
                "timestamp": "2024-01-01 00:00:00",
                "verified_purchase": True,
                "helpful_vote": 0,
                "images": "[]",
            }
        ],
    )
    call_count = {"n": 0}

    def extractor(review_text: str):
        call_count["n"] += 1
        raise ValueError("Missing Gemini API key")

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True, max_retries=3, base_delay=0.0)
    assert call_count["n"] == 1
    assert result["failed_review_count"] == 1


def test_invalid_quote_or_schema_output_not_marked_successful(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {
                "review_id": "r1",
                "rating": 3,
                "title": "Mixed",
                "text": "The scent is nice.",
                "asin": "A6",
                "parent_asin": "A6",
                "user_id": "u6",
                "timestamp": "2024-01-01 00:00:00",
                "verified_purchase": True,
                "helpful_vote": 0,
                "images": "[]",
            }
        ],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction.model_validate(
            {
                "findings": [
                    {
                        "aspect": "Scent",
                        "sentiment": "positive",
                        "supporting_quote": "definitely not in the review",
                    }
                ]
            }
        )

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True)
    assert result["failed_review_count"] == 1
    output = pd.read_csv(tmp_path / "out" / "aspect_sentiment_results.csv")
    assert len(output) == 0


def test_completed_reviews_are_skipped_after_restart(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {"review_id": "r1", "rating": 5, "title": "One", "text": "Great", "asin": "A7", "parent_asin": "A7", "user_id": "u7", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
            {"review_id": "r2", "rating": 4, "title": "Two", "text": "Works well", "asin": "A8", "parent_asin": "A8", "user_id": "u8", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
        ],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote=review_text.strip())])

    first = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=None, extractor=extractor, reset=True)
    assert first["processed_review_count"] == 2

    second = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=None, extractor=extractor, reset=False)
    assert second["processed_review_count"] == 0
    assert second["skipped_review_count"] == 2


def test_failed_reviews_can_be_retried_on_subsequent_run(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [{"review_id": "r1", "rating": 5, "title": "One", "text": "Great product.", "asin": "A9", "parent_asin": "A9", "user_id": "u9", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"}],
    )
    attempts = {"n": 0}

    def extractor(review_text: str):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("503 temporary server issue")
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote="Great product")])

    first = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True, max_retries=0, base_delay=0.0)
    assert first["failed_review_count"] == 1

    def extractor_success(review_text: str):
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote="Great product")])

    second = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor_success, reset=False, max_retries=1, base_delay=0.0)
    assert second["processed_review_count"] == 1


def test_no_duplicate_aspect_rows_after_restart(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [{"review_id": "r1", "rating": 5, "title": "One", "text": "The scent is great.", "asin": "A10", "parent_asin": "A10", "user_id": "u10", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"}],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.SCENT, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote="scent is great")])

    run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True)
    run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=False)
    output = pd.read_csv(tmp_path / "out" / "aspect_sentiment_results.csv")
    assert len(output) == 1


def test_partial_results_survive_later_failure(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {"review_id": "r1", "rating": 5, "title": "One", "text": "Great product.", "asin": "A11", "parent_asin": "A11", "user_id": "u11", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
            {"review_id": "r2", "rating": 1, "title": "Two", "text": "Terrible product.", "asin": "A12", "parent_asin": "A12", "user_id": "u12", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
        ],
    )

    calls = {"n": 0}

    def extractor(review_text: str):
        calls["n"] += 1
        if "Terrible" in review_text:
            raise RuntimeError("503 temporary server issue")
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote="Great product")])

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=None, extractor=extractor, reset=True, max_retries=1, base_delay=0.0)
    assert result["processed_review_count"] == 1
    assert result["failed_review_count"] == 1
    output = pd.read_csv(tmp_path / "out" / "aspect_sentiment_results.csv")
    assert output["review_id"].tolist() == ["r1"]


def test_csv_columns_and_source_ids_are_preserved(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [{"review_id": "r1", "rating": 5, "title": "One", "text": "Great product.", "asin": "A13", "parent_asin": "A13", "user_id": "u13", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"}],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote="Great product")])

    run_batch_extraction(sample_path, tmp_path / "out", max_reviews=1, extractor=extractor, reset=True)
    output = pd.read_csv(tmp_path / "out" / "aspect_sentiment_results.csv")
    assert set(output.columns) >= {"review_id", "asin", "parent_asin", "rating", "timestamp", "aspect", "sentiment", "supporting_quote"}
    assert output.iloc[0]["review_id"] == "r1"
    assert output.iloc[0]["asin"] == "A13"


def test_empty_and_duplicate_input_ids_are_handled_safely(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(
        sample_path,
        [
            {"review_id": "", "rating": 5, "title": "Empty", "text": "Great product.", "asin": "A14", "parent_asin": "A14", "user_id": "u14", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
            {"review_id": "dup", "rating": 4, "title": "One", "text": "Nice product.", "asin": "A15", "parent_asin": "A15", "user_id": "u15", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
            {"review_id": "dup", "rating": 4, "title": "One", "text": "Nice product.", "asin": "A15", "parent_asin": "A15", "user_id": "u15", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
            {"review_id": "r2", "rating": 5, "title": "Valid", "text": "works and smells good", "asin": "A16", "parent_asin": "A16", "user_id": "u16", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"},
        ],
    )

    def extractor(review_text: str):
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote=review_text[:len(review_text)//2])])

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=None, extractor=extractor, reset=True)
    assert result["eligible_review_count"] >= 2
    assert result["failed_review_count"] >= 2
    assert result["processed_review_count"] >= 1


def test_checkpoint_recovery_with_malformed_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "review_checkpoints.csv").write_text("review_id,status\nmalformed\n", encoding="utf-8")
    sample_path = tmp_path / "sample.csv"
    make_sample_csv(sample_path, [{"review_id": "r1", "rating": 5, "title": "One", "text": "Great product.", "asin": "A17", "parent_asin": "A17", "user_id": "u17", "timestamp": "2024-01-01 00:00:00", "verified_purchase": True, "helpful_vote": 0, "images": "[]"}])

    with pytest.raises(ValueError, match="malformed|checkpoint"):
        run_batch_extraction(sample_path, out_dir, max_reviews=1, extractor=lambda review_text: MODULE.ReviewExtraction(findings=[]), reset=False)


def test_small_test_run_never_processes_all_500_rows(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    records = []
    for i in range(500):
        records.append({
            "review_id": f"r{i}",
            "rating": 5,
            "title": f"T{i}",
            "text": f"This product works well {i}.",
            "asin": f"A{i}",
            "parent_asin": f"A{i}",
            "user_id": f"u{i}",
            "timestamp": "2024-01-01 00:00:00",
            "verified_purchase": True,
            "helpful_vote": 0,
            "images": "[]",
        })
    make_sample_csv(sample_path, records)

    def extractor(review_text: str):
        return MODULE.ReviewExtraction(findings=[MODULE.AspectFinding(aspect=MODULE.AspectLabel.PRODUCT_EFFECTIVENESS, sentiment=MODULE.Sentiment.POSITIVE, supporting_quote=review_text[:10])])

    result = run_batch_extraction(sample_path, tmp_path / "out", max_reviews=10, extractor=extractor, reset=True)
    assert result["processed_review_count"] <= 10
    assert result["eligible_review_count"] == 10
