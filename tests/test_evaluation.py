from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_evaluation_sample import prepare_evaluation_sample
from src.evaluation import (
    evaluate_extractions,
    extraction_fingerprint,
    load_evaluation_reviews,
    source_fingerprint,
    write_evaluation_outputs,
)
from src.extract import resolve_model_name


def write_case(
    root: Path,
    *,
    reviews: list[dict[str, str]] | None = None,
    gold: list[dict[str, str]] | None = None,
    predictions: list[dict[str, str]] | None = None,
    checkpoints: list[dict[str, str | int]] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    if reviews is None:
        reviews = [
            {"review_id": "r1", "text": "The scent is pleasant."},
            {"review_id": "r2", "text": "Shipping was late."},
        ]
    if gold is None:
        gold = [
        {
            "review_id": "r1",
            "annotation_status": "annotated",
            "aspect": "Scent",
            "sentiment": "positive",
            "supporting_quote": "scent is pleasant",
            "rationale": "",
        },
        {
            "review_id": "r2",
            "annotation_status": "annotated",
            "aspect": "Delivery",
            "sentiment": "negative",
            "supporting_quote": "Shipping was late",
            "rationale": "",
        },
        ]
    if predictions is None:
        predictions = [
            {"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "scent is pleasant"},
            {"review_id": "r2", "aspect": "Delivery", "sentiment": "negative", "supporting_quote": "Shipping was late"},
        ]
    if checkpoints is None:
        prediction_aspects = {review["review_id"]: set() for review in reviews}
        for prediction in predictions:
            if prediction["review_id"] in prediction_aspects:
                prediction_aspects[prediction["review_id"]].add(prediction["aspect"])
        checkpoints = [
            {"review_id": review["review_id"], "status": "completed", "attempt_count": 1,
             "aspect_count": len(prediction_aspects[review["review_id"]])}
            for review in reviews
        ]

    paths = {
        name: root / filename
        for name, filename in [
            ("reviews", "reviews.csv"),
            ("gold", "gold.csv"),
            ("predictions", "predictions.csv"),
            ("checkpoints", "checkpoints.csv"),
            ("manifest", "manifest.json"),
        ]
    }
    pd.DataFrame(reviews).to_csv(paths["reviews"], index=False, encoding="utf-8")
    pd.DataFrame(gold, columns=["review_id", "annotation_status", "aspect", "sentiment", "supporting_quote", "rationale"]).to_csv(paths["gold"], index=False, encoding="utf-8")
    pd.DataFrame(predictions, columns=["review_id", "aspect", "sentiment", "supporting_quote"]).to_csv(paths["predictions"], index=False, encoding="utf-8")
    pd.DataFrame(checkpoints, columns=["review_id", "status", "attempt_count", "aspect_count"]).to_csv(paths["checkpoints"], index=False, encoding="utf-8")
    reviews_frame = pd.read_csv(paths["reviews"], dtype=str, keep_default_na=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "prediction_source": "synthetic",
                "status": "completed",
                "source_fingerprint": source_fingerprint(reviews_frame),
                "extraction_fingerprint": extraction_fingerprint(),
                "model_name": resolve_model_name(),
            }
        ),
        encoding="utf-8",
    )
    return tuple(paths[name] for name in ("reviews", "gold", "predictions", "checkpoints", "manifest"))  # type: ignore[return-value]


def evaluate_case(root: Path, **kwargs):
    paths = write_case(root, **kwargs)
    return evaluate_extractions(*paths)


def test_perfect_predictions_produce_expected_metrics(tmp_path: Path) -> None:
    report, errors = evaluate_case(tmp_path)
    assert report["aspect_detection"]["micro"] == {
        "tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0
    }
    assert report["sentiment_classification"]["denominator"] == 2
    assert report["sentiment_classification"]["accuracy"] == 1.0
    assert report["quote_quality"]["denominator"] == 2
    assert report["quote_quality"]["invalid_quote_count"] == 0
    assert report["prediction_source"] == "synthetic"
    assert errors.empty


def test_missing_and_extra_aspects_count_false_negative_and_positive(tmp_path: Path) -> None:
    report, errors = evaluate_case(
        tmp_path,
        predictions=[
            {"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "scent is pleasant"},
            {"review_id": "r2", "aspect": "Packaging", "sentiment": "negative", "supporting_quote": "Shipping was late"},
        ],
    )
    metrics = report["aspect_detection"]["micro"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert set(errors["error_type"]) == {"false_positive_aspect", "missed_aspect"}


def test_defined_zero_f1_is_zero_not_undefined(tmp_path: Path) -> None:
    report, _ = evaluate_case(
        tmp_path,
        predictions=[
            {"review_id": "r1", "aspect": "Packaging", "sentiment": "positive", "supporting_quote": "scent is pleasant"},
            {"review_id": "r2", "aspect": "Other", "sentiment": "negative", "supporting_quote": "Shipping was late"},
        ],
    )
    metrics = report["aspect_detection"]["micro"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 2, 2)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_sentiment_only_uses_matched_aspects(tmp_path: Path) -> None:
    report, _ = evaluate_case(
        tmp_path,
        reviews=[{"review_id": "r1", "text": "The scent is pleasant and works well."}],
        gold=[{"review_id": "r1", "annotation_status": "annotated", "aspect": "Scent", "sentiment": "negative", "supporting_quote": "scent is pleasant", "rationale": ""}],
        predictions=[
            {"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "scent is pleasant"},
            {"review_id": "r1", "aspect": "Product effectiveness", "sentiment": "positive", "supporting_quote": "works well"},
        ],
    )
    assert report["aspect_detection"]["micro"]["tp"] == 1
    assert report["aspect_detection"]["micro"]["fp"] == 1
    assert report["sentiment_classification"]["denominator"] == 1
    assert report["sentiment_classification"]["accuracy"] == 0.0


def test_quote_validity_is_exact_substring_and_separate(tmp_path: Path) -> None:
    report, errors = evaluate_case(
        tmp_path,
        predictions=[
            {"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "The scent is pleasant"},
            {"review_id": "r2", "aspect": "Delivery", "sentiment": "negative", "supporting_quote": "late shipping"},
        ],
    )
    assert report["aspect_detection"]["micro"]["tp"] == 2
    assert report["quote_quality"]["denominator"] == 2
    assert report["quote_quality"]["invalid_quote_count"] == 1
    assert "invalid_quote" in set(errors["error_type"])


def test_valid_zero_finding_differs_from_failed_extraction(tmp_path: Path) -> None:
    report, errors = evaluate_case(
        tmp_path,
        reviews=[{"review_id": "r1", "text": "No relevant product opinion."}, {"review_id": "r2", "text": "No result."}],
        gold=[
            {"review_id": "r1", "annotation_status": "no_aspects", "aspect": "", "sentiment": "", "supporting_quote": "", "rationale": ""},
            {"review_id": "r2", "annotation_status": "no_aspects", "aspect": "", "sentiment": "", "supporting_quote": "", "rationale": ""},
        ],
        predictions=[],
        checkpoints=[
            {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 0},
            {"review_id": "r2", "status": "failed", "attempt_count": 4, "aspect_count": 0},
        ],
    )
    operational = report["operational_reliability"]
    assert operational["valid_zero_finding_completed_reviews"] == 1
    assert operational["successful_reviews"] == 1
    assert operational["failed_reviews"] == 1
    assert operational["extraction_failure_rate"] == 0.5
    assert list(errors["error_type"]) == ["extraction_failure"]


def test_exact_duplicate_predictions_collapse_and_conflicts_reject(tmp_path: Path) -> None:
    prediction = {"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "scent is pleasant"}
    report, _ = evaluate_case(tmp_path / "exact", predictions=[prediction, prediction])
    assert report["prediction_findings_after_exact_duplicate_collapse"] == 1

    with pytest.raises(ValueError, match="Conflicting duplicate predictions"):
        evaluate_case(
            tmp_path / "conflict",
            predictions=[
                prediction,
                {**prediction, "sentiment": "neutral", "supporting_quote": "scent is pleasant"},
            ],
        )


@pytest.mark.parametrize(
    ("reviews", "message"),
    [
        ([{"review_id": "", "text": "Text"}], "review_id must not be blank"),
        ([{"review_id": "r1", "text": "Text"}, {"review_id": "r1", "text": "More"}], "must be unique"),
    ],
)
def test_missing_or_duplicate_review_ids_are_rejected(tmp_path: Path, reviews, message: str) -> None:
    review_path = tmp_path / "reviews.csv"
    pd.DataFrame(reviews).to_csv(review_path, index=False)
    with pytest.raises(ValueError, match=message):
        load_evaluation_reviews(review_path)


def test_empty_gold_and_predictions_return_undefined_scores(tmp_path: Path) -> None:
    report, _ = evaluate_case(
        tmp_path,
        gold=[
            {"review_id": "r1", "annotation_status": "unannotated", "aspect": "", "sentiment": "", "supporting_quote": "", "rationale": ""},
            {"review_id": "r2", "annotation_status": "unannotated", "aspect": "", "sentiment": "", "supporting_quote": "", "rationale": ""},
        ],
        predictions=[],
        checkpoints=[
            {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 0},
            {"review_id": "r2", "status": "completed", "attempt_count": 1, "aspect_count": 0},
        ],
    )
    assert report["quality_population_reviews"] == 0
    assert report["aspect_detection"]["micro"]["precision"] is None
    assert report["aspect_detection"]["micro"]["recall"] is None
    assert report["sentiment_classification"]["accuracy"] is None


def test_predictions_for_unannotated_reviews_are_explicitly_excluded(tmp_path: Path) -> None:
    report, _ = evaluate_case(
        tmp_path,
        gold=[
            {"review_id": "r1", "annotation_status": "unannotated", "aspect": "", "sentiment": "", "supporting_quote": "", "rationale": ""},
            {"review_id": "r2", "annotation_status": "no_aspects", "aspect": "", "sentiment": "", "supporting_quote": "", "rationale": ""},
        ],
        predictions=[{"review_id": "r1", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "scent"}],
        checkpoints=[
            {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 1},
            {"review_id": "r2", "status": "completed", "attempt_count": 1, "aspect_count": 0},
        ],
    )
    assert report["quality_population_reviews"] == 1
    assert report["quote_quality"]["excluded_predictions_for_unannotated_or_adjudication_reviews"] == 1
    assert report["quote_quality"]["denominator"] == 0


def test_checkpoint_aspect_count_must_match_prediction_rows(tmp_path: Path) -> None:
    paths = write_case(
        tmp_path,
        predictions=[],
        checkpoints=[
            {"review_id": "r1", "status": "completed", "attempt_count": 1, "aspect_count": 1},
            {"review_id": "r2", "status": "completed", "attempt_count": 1, "aspect_count": 0},
        ],
    )
    with pytest.raises(ValueError, match="aspect_count"):
        evaluate_extractions(*paths)


def test_unknown_prediction_review_and_invalid_labels_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown review_id"):
        evaluate_case(
            tmp_path / "unknown",
            predictions=[{"review_id": "r3", "aspect": "Scent", "sentiment": "positive", "supporting_quote": "x"}],
        )
    with pytest.raises(ValueError, match="Unknown predicted aspect"):
        evaluate_case(
            tmp_path / "label",
            predictions=[{"review_id": "r1", "aspect": "Fragrance", "sentiment": "positive", "supporting_quote": "scent is pleasant"}],
        )


def test_provenance_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = write_case(tmp_path)
    reviews_path = paths[0]
    reviews = pd.read_csv(reviews_path)
    reviews.loc[0, "text"] = "The scent is pleasant. Edited source text."
    reviews.to_csv(reviews_path, index=False)
    with pytest.raises(ValueError, match="exact selected review IDs and text"):
        evaluate_extractions(*paths)


def test_synthetic_end_to_end_evaluation_writes_report_and_errors(tmp_path: Path) -> None:
    paths = write_case(tmp_path / "fixture")
    report, errors = evaluate_extractions(*paths)
    report_path, errors_path = write_evaluation_outputs(report, errors, tmp_path / "report")
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["prediction_source"] == "synthetic"
    assert written["aspect_detection"]["micro"]["tp"] == 2
    assert pd.read_csv(errors_path).empty


def test_sample_preparation_is_deterministic_and_omits_user_ids(tmp_path: Path) -> None:
    rows = [
        {"review_id": f"r{i}", "rating": 5, "title": f"t{i}", "asin": f"a{i}",
         "parent_asin": f"a{i}", "timestamp": "2024-01-01", "text": f"review {i}", "user_id": f"private-{i}"}
        for i in range(500)
    ]
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    reviews_path, gold_path = prepare_evaluation_sample(sample_path, tmp_path / "one")
    second_reviews, _ = prepare_evaluation_sample(sample_path, tmp_path / "two")
    reviews = pd.read_csv(reviews_path, dtype=str, keep_default_na=False)
    assert len(reviews) == 50
    assert reviews["review_id"].nunique() == 50
    assert "user_id" not in reviews.columns
    assert reviews["review_id"].tolist() == pd.read_csv(second_reviews, dtype=str)["review_id"].tolist()
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    assert set(gold["annotation_status"]) == {"unannotated"}
    assert len(gold) == 50