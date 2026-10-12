"""Streamlit views for the Review Insight Extractor dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard_data import (
    ASPECT_LABELS,
    SENTIMENT_LABELS,
    DashboardData,
    DashboardDataError,
    DashboardPaths,
    aspect_frequency,
    aspect_sentiment_matrix,
    aspect_sentiment_counts,
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


def _artifact_signature(paths: DashboardPaths) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in paths.__dict__.values():
        artifact = Path(path)
        try:
            stat = artifact.stat()
            signature.append((str(artifact), stat.st_size, stat.st_mtime_ns))
        except OSError:
            signature.append((str(artifact), -1, -1))
    return tuple(signature)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_dashboard_data(
    paths: DashboardPaths,
    signature: tuple[tuple[str, int, int], ...],
) -> DashboardData:
    del signature
    return load_dashboard_data(paths)


def _format_percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _show_issues(data: DashboardData) -> None:
    for issue in data.issues:
        st.warning(issue)
    if not data.available["results"]:
        st.info(
            "No aspect-result CSV is available. The dashboard is read-only and does not call Gemini; "
            "generate outputs through the documented Stage 4 batch workflow."
        )


def _sidebar(data: DashboardData) -> tuple[str, pd.DataFrame]:
    st.sidebar.markdown("### REVIEW SIGNALS")
    st.sidebar.caption("All Beauty · extraction workspace")
    view = st.sidebar.radio(
        "Workspace",
        ["Overview", "Explore findings", "Aspect matrix & priorities", "Evaluation & errors", "Data status"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown("#### Filters")
    results = data.results
    aspect_options = ["All", *[value for value in ASPECT_LABELS if value in set(results.get("aspect", []))]]
    sentiment_options = ["All", *[value for value in SENTIMENT_LABELS if value in set(results.get("sentiment", []))]]
    aspect = st.sidebar.selectbox("Aspect", aspect_options, disabled=results.empty)
    sentiment = st.sidebar.selectbox("Sentiment", sentiment_options, disabled=results.empty)

    asin_values = []
    if "asin" in results:
        asin_values = sorted(value for value in results["asin"].dropna().astype(str).unique() if value.strip())
    asin = st.sidebar.selectbox("Product ASIN", ["All", *asin_values], disabled=not asin_values)

    rating_min = rating_max = None
    if "rating" in results:
        ratings = pd.to_numeric(results["rating"], errors="coerce").dropna()
        unique_ratings = sorted(float(value) for value in ratings.unique())
        if len(unique_ratings) > 1:
            rating_min, rating_max = st.sidebar.select_slider(
                "Rating range",
                options=unique_ratings,
                value=(unique_ratings[0], unique_ratings[-1]),
                disabled=results.empty,
            )
        elif unique_ratings:
            rating_min = rating_max = unique_ratings[0]

    statuses = sorted(results["extraction_status"].dropna().astype(str).unique()) if "extraction_status" in results else []
    status = st.sidebar.selectbox("Extraction status", ["All", *statuses], disabled=not statuses)
    search = st.sidebar.text_input("Search reviews and quotes", placeholder="Text, quote, or review ID")

    filtered = filter_findings(
        results,
        aspect=aspect,
        sentiment=sentiment,
        asin=asin,
        rating_min=rating_min,
        rating_max=rating_max,
        extraction_status=status,
        search=search,
    )
    st.sidebar.caption(f"{len(filtered):,} aspect finding rows · {review_count(filtered):,} unique reviews")
    return view, filtered


def _metric(value: int | str | None) -> str:
    return "—" if value is None else f"{value:,}" if isinstance(value, int) else value


def _overview(data: DashboardData, findings: pd.DataFrame) -> None:
    summary = pipeline_summary(data)
    st.markdown("## Pipeline overview")
    st.caption("Review-level checkpoint totals are distinct reviews. Aspect records count finding rows.")
    columns = st.columns(6)
    labels = [
        ("Source reviews", summary["sample_reviews"]),
        ("Attempted reviews", summary["attempted_reviews"]),
        ("Successful reviews", summary["successful_reviews"]),
        ("Failed reviews", summary["failed_reviews"]),
        ("Recorded attempts", summary["recorded_attempts"]),
        ("Filtered aspect rows", len(findings)),
    ]
    for column, (label, value) in zip(columns, labels):
        column.metric(label, _metric(value))
    st.caption(f"Checkpoint state: {summary['pipeline_progress']}")
    if summary["recorded_attempts"] is not None:
        st.caption("Recorded attempts sum each review's current checkpoint attempt_count; lifetime attempts across restarts are not available.")
    attempted = summary["attempted_reviews"]
    population = summary["sample_reviews"]
    if isinstance(attempted, int) and isinstance(population, int) and population:
        st.progress(min(attempted / population, 1.0), text=f"{attempted:,} / {population:,} source reviews checkpointed")

    st.markdown("### Aspect and sentiment signals")
    left, right = st.columns(2)
    aspect_counts = aspect_frequency(findings)
    sentiment_counts = sentiment_distribution(findings)
    with left:
        st.markdown("#### Aspect frequency")
        if aspect_counts.empty:
            st.info("No extracted aspect findings are available for the current filters.")
        else:
            st.bar_chart(aspect_counts.set_index("aspect"), height=260, width="stretch")
    with right:
        st.markdown("#### Sentiment distribution")
        if findings.empty:
            st.info("Sentiment counts will appear when aspect-result rows are available.")
        else:
            st.bar_chart(sentiment_counts.set_index("sentiment"), height=260, width="stretch")

    if not findings.empty:
        st.markdown("#### Aspect by sentiment")
        st.bar_chart(aspect_sentiment_counts(findings).set_index("aspect"), height=300, width="stretch")

    st.markdown("### Evaluation snapshot")
    report = data.evaluation_report
    if report is None:
        st.caption("Manual-gold evaluation metrics are unavailable until a Stage 5 report exists.")
    else:
        if report.get("prediction_source") != "live":
            st.warning(f"Evaluation source is {report.get('prediction_source', 'unknown')!r}; not a live-model score.")
        aspect_metrics = report.get("aspect_detection", {}).get("micro", {})
        sentiment_metrics = report.get("sentiment_classification", {})
        eval_columns = st.columns(4)
        eval_columns[0].metric("Gold-labeled reviews", _metric(report.get("quality_population_reviews")))
        eval_columns[1].metric("Aspect micro F1", _format_percent(aspect_metrics.get("f1")))
        eval_columns[2].metric("Aspect precision", _format_percent(aspect_metrics.get("precision")))
        eval_columns[3].metric("Matched sentiment accuracy", _format_percent(sentiment_metrics.get("accuracy")))
        st.caption("Evaluation metrics use the human-labeled subset, not all pipeline reviews.")


def _finding_table(findings: pd.DataFrame) -> None:
    if findings.empty:
        st.info("No finding rows match the current filters, or extraction outputs have not been generated.")
        return
    st.dataframe(
        findings,
        hide_index=True,
        width="stretch",
        height=440,
        column_config={
            "supporting_quote": st.column_config.TextColumn("Supporting quote", width="large"),
            "review_text": st.column_config.TextColumn("Source review", width="large"),
            "review_id": st.column_config.TextColumn("Review ID", width="medium"),
        },
    )


def _explore(findings: pd.DataFrame) -> None:
    st.markdown("## Explore findings")
    st.caption("One row represents one extracted aspect finding. Repeated review IDs are expected for multi-aspect reviews.")
    first, second, third = st.columns(3)
    first.metric("Finding rows", f"{len(findings):,}")
    second.metric("Unique reviews", f"{review_count(findings):,}")
    third.metric("Distinct aspects", f"{findings['aspect'].nunique():,}" if not findings.empty else "0")
    st.download_button(
        "Export filtered findings CSV",
        data=findings_csv_bytes(findings),
        file_name="filtered_aspect_findings.csv",
        mime="text/csv",
        disabled=findings.empty,
    )
    _finding_table(findings)


def _aspect_matrix_and_priorities(findings: pd.DataFrame) -> None:
    st.markdown("## Aspect matrix and complaint priorities")
    st.caption(
        "Sentiment proportions use unique review/aspect/sentiment pairs. Negative-review frequency counts distinct reviews, not finding rows."
    )
    analysis_findings = findings
    if "quote_valid" in findings:
        analysis_findings = findings.loc[findings["quote_valid"].fillna(False).eq(True)]
        unverified_count = len(findings) - len(analysis_findings)
        if unverified_count:
            st.warning(f"{unverified_count} finding rows lack verified source quotes and are excluded from the matrix and priority score.")
    matrix = aspect_sentiment_matrix(analysis_findings)
    if matrix.empty:
        st.info("Aspect sentiment proportions and complaint priorities require finding rows for the current filters.")
        return

    st.markdown("### Aspect-level sentiment matrix")
    st.dataframe(
        matrix,
        hide_index=True,
        width="stretch",
        column_config={
            "positive_proportion": st.column_config.NumberColumn("Positive share", format="percent"),
            "neutral_proportion": st.column_config.NumberColumn("Neutral share", format="percent"),
            "negative_proportion": st.column_config.NumberColumn("Negative share", format="percent"),
        },
    )
    share_chart = matrix.set_index("aspect")[["positive_proportion", "neutral_proportion", "negative_proportion"]]
    st.bar_chart(share_chart, height=300, width="stretch")

    st.markdown("### Complaint prioritization")
    st.caption(
        "Score = weighted mean of (negative-review count / highest aspect count) and negative share. "
        "It is a ranking aid, not a severity or causal estimate."
    )
    weights = st.columns(2)
    frequency_weight = weights[0].slider("Negative-review frequency weight", 0, 100, 60, 5)
    negative_share_weight = weights[1].slider("Negative sentiment share weight", 0, 100, 40, 5)
    st.info("Validated severity is unavailable: the current extraction schema and Stage 4 output do not define severity.")
    if frequency_weight + negative_share_weight == 0:
        st.warning("Set at least one score weight above zero to rank aspects.")
        return

    priorities = complaint_prioritization(
        analysis_findings,
        frequency_weight=frequency_weight,
        negative_share_weight=negative_share_weight,
    )
    st.dataframe(
        priorities,
        hide_index=True,
        width="stretch",
        column_config={
            "negative_proportion": st.column_config.NumberColumn("Negative share", format="percent"),
            "normalized_frequency": st.column_config.NumberColumn("Relative negative-review count", format="percent"),
            "priority_score": st.column_config.NumberColumn("Priority score", format="%.1f"),
        },
    )

    st.markdown("### Representative source-review evidence")
    evidence = representative_evidence(analysis_findings)
    if evidence.empty:
        st.info("No source evidence is available for the current selection.")
    else:
        st.dataframe(
            evidence,
            hide_index=True,
            width="stretch",
            column_config={
                "supporting_quote": st.column_config.TextColumn("Exact extracted quote", width="large"),
                "review_text": st.column_config.TextColumn("Source review", width="large"),
            },
        )

    report = build_insight_report(priorities)
    st.markdown("### Evidence-backed insight report")
    report_columns = st.columns(3)
    for column, title, key in zip(
        report_columns,
        ["Observed facts", "Generated interpretation", "Proposed recommendation"],
        ["observed_facts", "generated_interpretations", "proposed_recommendations"],
    ):
        with column:
            st.markdown(f"#### {title}")
            for item in report[key]:
                st.write(item)
    st.caption(report["method_note"])
    st.download_button(
        "Download insight report",
        data=insight_report_markdown(report, evidence),
        file_name="evidence_backed_insights.md",
        mime="text/markdown",
    )


def _show_evaluation(data: DashboardData) -> None:
    st.markdown("## Evaluation and error analysis")
    report = data.evaluation_report
    if report is None:
        st.info(
            "No Stage 5 evaluation report is available. Human labels and provenance-matched predictions "
            "are required before model-quality metrics can be shown."
        )
        if not data.evaluation_errors.empty:
            st.markdown("### Evaluation errors")
            st.dataframe(data.evaluation_errors, hide_index=True, width="stretch")
        return

    source = report.get("prediction_source", "unknown")
    if source != "live":
        st.warning(f"Prediction source is {source!r}; these values must not be presented as live model performance.")
    st.caption("Manually annotated evaluation subset · separate from pipeline-wide extraction counts")
    gold_counts = report.get("gold_annotation_status_counts", {})
    evaluation_size = report.get("quality_population_reviews")
    st.metric("Reviews in quality evaluation", _metric(evaluation_size))
    st.caption(f"Gold annotation statuses: {gold_counts or 'not reported'}")

    aspect = report.get("aspect_detection", {}).get("micro", {})
    sentiment = report.get("sentiment_classification", {})
    quote = report.get("quote_quality", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric("Aspect micro F1", _format_percent(aspect.get("f1")))
    metric_columns[1].metric("Aspect precision", _format_percent(aspect.get("precision")))
    metric_columns[2].metric("Aspect recall", _format_percent(aspect.get("recall")))
    metric_columns[3].metric("Sentiment accuracy", _format_percent(sentiment.get("accuracy")))
    st.caption(
        f"Aspect TP {aspect.get('tp', '—')} · FP {aspect.get('fp', '—')} · FN {aspect.get('fn', '—')}"
        f" · Sentiment denominator {sentiment.get('denominator', '—')}"
    )

    quote_columns = st.columns(3)
    quote_columns[0].metric("Exact valid quotes", f"{quote.get('exact_substring_valid', '—')} / {quote.get('denominator', '—')}")
    quote_columns[1].metric("Invalid quotes", _metric(quote.get("invalid_quote_count")))
    quote_columns[2].metric("Extraction failure rate", _format_percent(report.get("operational_reliability", {}).get("extraction_failure_rate")))

    per_aspect = report.get("aspect_detection", {}).get("per_aspect", {})
    if per_aspect:
        frame = pd.DataFrame.from_dict(per_aspect, orient="index").rename_axis("aspect").reset_index()
        st.markdown("### Per-aspect metrics")
        st.dataframe(frame, hide_index=True, width="stretch")
    confusion = sentiment.get("confusion_matrix_gold_rows_predicted_columns")
    if confusion:
        st.markdown("### Sentiment confusion matrix")
        st.dataframe(pd.DataFrame.from_dict(confusion, orient="index").rename_axis("gold sentiment"), width="stretch")
    baseline = report.get("baseline_comparison")
    if isinstance(baseline, dict):
        st.markdown("### Baseline comparison")
        st.json(baseline)
    else:
        st.info("No baseline comparison is present in the Stage 5 report; no comparator is inferred.")
    if data.evaluation_errors.empty:
        st.info("No review-level evaluation error records are available.")
    else:
        st.markdown("### Review-level error examples")
        st.dataframe(data.evaluation_errors, hide_index=True, width="stretch")


def _data_status(data: DashboardData, paths: DashboardPaths) -> None:
    st.markdown("## Data status")
    st.caption("Artifact paths are resolved from the repository location, not the shell's current directory.")
    quality = data_quality_summary(data)
    quality_columns = st.columns(5)
    quality_columns[0].metric("Source reviews", f"{quality['source_reviews']:,}")
    quality_columns[1].metric("Valid finding rows", f"{quality['valid_finding_rows']:,}")
    quality_columns[2].metric("Invalid result rows", f"{quality['invalid_result_rows']:,}")
    quote_mismatches = quality["invalid_exact_quotes"]
    quality_columns[3].metric("Invalid exact quotes", "—" if quote_mismatches is None else f"{quote_mismatches:,}")
    quality_columns[4].metric("Checkpoint/result mismatches", f"{len(quality['checkpoint_result_mismatches']):,}")
    rows = [
        ("Review sample", paths.reviews, data.available["reviews"], len(data.reviews)),
        ("Aspect results", paths.results, data.available["results"], len(data.results)),
        ("Review checkpoints", paths.checkpoints, data.available["checkpoints"], len(data.checkpoints)),
        ("Failure log", paths.failures, data.available["failures"], len(data.failures)),
        ("Evaluation report", paths.evaluation_report, data.available["evaluation"], None),
        ("Evaluation errors", paths.evaluation_errors, data.available["evaluation_errors"], len(data.evaluation_errors)),
    ]
    st.dataframe(
        pd.DataFrame(
            [{"Artifact": name, "Path": str(path), "Present": present, "Rows": count} for name, path, present, count in rows]
        ),
        hide_index=True,
        width="stretch",
    )
    if data.invalid_results.empty:
        st.success("No invalid aspect-result rows detected.")
    else:
        st.warning(f"{len(data.invalid_results)} invalid result rows were excluded from charts.")
        st.dataframe(data.invalid_results, hide_index=True, width="stretch")
    if data.issues:
        st.markdown("### Loading notes")
        for issue in data.issues:
            st.write(f"- {issue}")
    if quality["checkpoint_result_mismatches"]:
        st.markdown("### Checkpoint/result count mismatches")
        st.dataframe(pd.DataFrame(quality["checkpoint_result_mismatches"]), hide_index=True, width="stretch")
    if not data.failures.empty:
        st.markdown("### Failure log categories")
        category_column = "error_category" if "error_category" in data.failures else "final_status"
        category_counts = data.failures[category_column].fillna("unknown").value_counts().rename_axis("category").reset_index(name="log entries")
        st.dataframe(category_counts, hide_index=True, width="stretch")
        st.caption("Failure log entries may be historical; current failed-review counts come from checkpoints.")
    st.caption("This dashboard does not load .env, request credentials, or run extraction.")


def run_dashboard() -> None:
    """Render the read-only dashboard and its data-driven filters."""

    st.set_page_config(page_title="Review Insight Extractor", page_icon="◈", layout="wide")
    st.markdown(
        """
        <style>
        :root { --pine: #245448; --coral: #bd5848; --gold: #c59639; --ink: #202a27; --muted: #687570; }
        .stApp { background: #f5f7f4; color: var(--ink); }
        [data-testid="stSidebar"] { background: #eaf0eb; border-right: 1px solid #d6e0d9; }
        h1, h2, h3 { color: var(--ink); }
        h1 { font-family: Georgia, serif; font-weight: 500; }
        [data-testid="stMetric"] { background: white; border: 1px solid #dce5de; border-radius: 6px; padding: 12px 14px; }
        [data-testid="stMetricValue"] { color: var(--pine); }
        .block-container { padding-top: 1.5rem; max-width: 1440px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    paths = DashboardPaths()
    try:
        data = _cached_dashboard_data(paths, _artifact_signature(paths))
    except DashboardDataError as exc:
        st.error(f"Dashboard artifact validation failed: {exc}")
        st.stop()
    st.markdown("# Review signals")
    st.caption("Amazon Reviews 2023 · All Beauty · aspect-level extraction")
    view, findings = _sidebar(data)
    _show_issues(data)

    if view == "Overview":
        _overview(data, findings)
    elif view == "Explore findings":
        _explore(findings)
    elif view == "Aspect matrix & priorities":
        _aspect_matrix_and_priorities(findings)
    elif view == "Evaluation & errors":
        _show_evaluation(data)
    else:
        _data_status(data, paths)