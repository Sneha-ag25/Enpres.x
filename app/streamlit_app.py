"""Enpres.x Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import altair as alt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_baseline import SELECTED_BUILDINGS  # noqa: E402
from forecast_24h import ForecastUnavailable, generate_24h_forecast  # noqa: E402
from phase2_extensions import (  # noqa: E402
    AUDIT_MIN_OBSERVED_HOURS,
    AUDIT_MIN_STREAK_HOURS,
    add_meter_quality_flags,
    build_audit_ranking,
    build_retrofit_scenarios,
    summarize_meter_quality,
)


MODEL_PATH = ROOT / "models" / "panther_baseline.joblib"
METRICS_PATH = ROOT / "reports" / "baseline_metrics.json"
INTERVAL_METRICS_PATH = ROOT / "reports" / "prediction_interval_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "baseline_test_predictions_with_intervals.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "panther_electricity.csv"
FUTURE_WEATHER_PATH = ROOT / "data" / "processed" / "future_weather.csv"


st.set_page_config(
    page_title="Enpres.x | Building Energy Forecasts",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main-title { font-size: 2.3rem; font-weight: 750; margin-bottom: 0.1rem; }
    .subtitle { color: #52606d; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .section-title { margin-top: 1.2rem; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading saved test predictions...")
def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])
    predictions = predictions.sort_values(["building_id", "timestamp"])
    return predictions


@st.cache_data(show_spinner="Checking meter data quality...")
def load_quality_flags() -> pd.DataFrame:
    usecols = ["timestamp", "building_id", "electricity"]
    pieces = []
    for chunk in pd.read_csv(
        PROCESSED_PATH,
        usecols=usecols,
        parse_dates=["timestamp"],
        chunksize=200_000,
        low_memory=True,
    ):
        selected = chunk[chunk["building_id"].isin(SELECTED_BUILDINGS)]
        if not selected.empty:
            pieces.append(selected)

    data = pd.concat(pieces, ignore_index=True)
    return add_meter_quality_flags(data)


@st.cache_data(show_spinner="Summarizing observation coverage...")
def load_coverage() -> pd.DataFrame:
    return summarize_meter_quality(load_quality_flags())


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_interval_metrics() -> dict:
    return json.loads(INTERVAL_METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_model_artifact() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading forecast history...")
def load_forecast_history(building_id: str) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "building_id",
        "electricity",
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "yearbuilt",
        "eui",
        "airTemperature",
        "cloudCoverage",
        "dewTemperature",
        "precipDepth1HR",
        "precipDepth6HR",
        "seaLvlPressure",
        "windDirection",
        "windSpeed",
    ]
    pieces = []
    for chunk in pd.read_csv(
        PROCESSED_PATH,
        usecols=usecols,
        parse_dates=["timestamp"],
        chunksize=200_000,
        low_memory=True,
    ):
        selected = chunk[chunk["building_id"] == building_id]
        if not selected.empty:
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True).sort_values("timestamp")


@st.cache_data
def load_future_weather() -> pd.DataFrame:
    if not FUTURE_WEATHER_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FUTURE_WEATHER_PATH, parse_dates=["timestamp"])


@st.cache_data(show_spinner="Ranking potential audit candidates...")
def load_audit_ranking() -> pd.DataFrame:
    return build_audit_ranking(load_predictions())


def metric_value(frame: pd.DataFrame, prediction_column: str, actual_column: str = "electricity") -> float:
    valid = frame[[actual_column, prediction_column]].dropna()
    if valid.empty:
        return float("nan")
    return float((valid[actual_column] - valid[prediction_column]).abs().mean())


def rmse_value(frame: pd.DataFrame, prediction_column: str, actual_column: str = "electricity") -> float:
    valid = frame[[actual_column, prediction_column]].dropna()
    if valid.empty:
        return float("nan")
    errors = valid[actual_column] - valid[prediction_column]
    return float((errors.pow(2).mean()) ** 0.5)


def main() -> None:
    try:
        predictions = load_predictions()
        quality_flags = load_quality_flags()
        coverage = load_coverage()
        audit_ranking = load_audit_ranking()
        saved_metrics = load_metrics()
        interval_metrics = load_interval_metrics()
        artifact = load_model_artifact()
    except Exception as exc:  # pragma: no cover - user-facing app error state
        st.error(f"The dashboard could not load the saved project artifacts: {exc}")
        return

    st.markdown('<div class="main-title">⚡ Enpres.x</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Building electricity consumption forecasts for the CL-03 hackathon</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Dashboard controls")
        selected_building = st.selectbox(
            "Choose a building",
            options=SELECTED_BUILDINGS,
            format_func=lambda value: value.replace("Panther_", "").replace("_", " ").title(),
        )
        st.caption("The selector contains the five buildings used to train the saved baseline model.")
        chart_points = st.slider(
            "Test-period points to display",
            min_value=24,
            max_value=720,
            value=336,
            step=24,
            help="The chart shows the most recent held-out test observations for the selected building.",
        )
        with st.expander("Model details"):
            st.write("Saved model loaded successfully")
            st.write(f"Model: `{type(artifact['pipeline'].named_steps['model']).__name__}`")
            st.write(f"Training end: `{saved_metrics['train_end']}`")
            st.write(f"Validation end: `{saved_metrics['validation_end']}`")

    selected_predictions = predictions[predictions["building_id"] == selected_building].copy()
    selected_predictions = selected_predictions.sort_values("timestamp")
    selected_coverage = coverage[coverage["building_id"] == selected_building].iloc[0]

    st.subheader("Held-out test performance")
    overall_test = saved_metrics["metrics"]["test"]
    overall_interval_coverage = interval_metrics["test_period"]["coverage"]
    selected_test_mae = metric_value(selected_predictions, "model_prediction")
    selected_test_rmse = rmse_value(selected_predictions, "model_prediction")
    columns = st.columns(4)
    columns[0].metric("Overall test MAE", f"{overall_test['model']['mae']:.3f}")
    columns[1].metric("Overall test RMSE", f"{overall_test['model']['rmse']:.3f}")
    columns[2].metric("Selected building MAE", f"{selected_test_mae:.3f}")
    columns[3].metric("Selected building RMSE", f"{selected_test_rmse:.3f}")
    st.caption(
        "Overall metrics are the saved results across all five buildings. "
        f"The held-out test period is {saved_metrics['test_start']} through {saved_metrics['test_end']}."
    )

    st.subheader("Actual versus predicted consumption")
    st.subheader("Empirical 80% prediction interval")
    interval_columns = st.columns(2)
    interval_columns[0].metric("Overall test interval coverage", f"{overall_interval_coverage:.1%}")
    selected_interval_coverage = interval_metrics["coverage_by_building"].get(selected_building)
    if selected_interval_coverage is not None:
        interval_columns[1].metric("Selected building coverage", f"{selected_interval_coverage:.1%}")
    st.caption(
        "The interval is calibrated from validation-period absolute residuals. "
        "80% is the nominal target; held-out test coverage may differ."
    )

    chart_data = selected_predictions.tail(chart_points).copy()
    base = alt.Chart(chart_data).encode(
        x=alt.X("timestamp:T", title="Timestamp"),
    )
    interval = base.mark_area(opacity=0.22, color="#5b8ff9").encode(
        y=alt.Y("prediction_lower:Q", title="Electricity consumption"),
        y2="prediction_upper:Q",
    )
    actual = base.mark_line(color="#1f2937", strokeWidth=2).encode(
        y=alt.Y("electricity:Q"),
        tooltip=["timestamp:T", "electricity:Q"],
    )
    prediction = base.mark_line(color="#2563eb", strokeWidth=2).encode(
        y=alt.Y("model_prediction:Q"),
        tooltip=["timestamp:T", "model_prediction:Q"],
    )
    benchmark = base.mark_line(color="#f59e0b", strokeDash=[5, 4]).encode(
        y=alt.Y("same_hour_previous_day:Q"),
        tooltip=["timestamp:T", "same_hour_previous_day:Q"],
    )
    st.altair_chart((interval + actual + prediction + benchmark).properties(height=420), width="stretch")
    st.caption(
        "Chart uses saved held-out test predictions only. The shaded band is the empirical prediction interval. "
        "Blank electricity targets were excluded from interval coverage."
    )

    st.subheader("Data quality and observation coverage")
    warning_columns = st.columns(4)
    warning_columns[0].metric("Observed hours", f"{int(selected_coverage['observed_hours']):,}")
    warning_columns[1].metric("Coverage", f"{selected_coverage['coverage_pct']:.1f}%")
    warning_columns[2].metric(
        "Longest missing gap", f"{int(selected_coverage['longest_missing_gap_hours']):,} h"
    )
    warning_columns[3].metric(
        "Suspicious spikes", f"{int(selected_coverage['suspicious_spike_count']):,}"
    )

    if selected_coverage["missing_hours"] > 0:
        st.warning(
            f"{selected_building} has {int(selected_coverage['missing_hours']):,} missing electricity readings. "
            "Missing readings are not treated as zero and were excluded from model scoring."
        )
    if selected_coverage["longest_missing_gap_hours"] > 24:
        st.info(
            "This building contains a missing-data gap longer than 24 hours. "
            "Forecast quality should be interpreted together with the observation coverage."
        )
    if selected_coverage["suspicious_spike_count"] > 0:
        st.info(
            f"{selected_building} has {int(selected_coverage['suspicious_spike_count']):,} suspicious extreme "
            "spike(s) under the documented rolling-median/MAD screening rule. These are review flags, not confirmed faults."
        )

    coverage_display = coverage.copy()
    coverage_display["building_id"] = coverage_display["building_id"].str.replace(
        "Panther_", "", regex=False
    ).str.replace("_", " ", regex=False).str.title()
    st.dataframe(
        coverage_display.set_index("building_id").rename(
            columns={
                "total_hours": "Total hours",
                "observed_hours": "Observed hours",
                "missing_hours": "Missing hours",
                "coverage_pct": "Coverage (%)",
                "longest_missing_gap_hours": "Longest gap (hours)",
                "long_missing_gap_count": "Long-gap runs",
                "long_missing_gap_hours": "Long-gap hours",
                "suspicious_spike_count": "Suspicious spikes",
                "confirmed_faulty_count": "Confirmed faulty",
            }
        ).style.format({"Coverage (%)": "{:.1f}"}),
        width="stretch",
    )
    with st.expander("Review flagged meter rows"):
        selected_flags = quality_flags[
            (quality_flags["building_id"] == selected_building)
            & (quality_flags["missing_reading"] | quality_flags["suspicious_spike"])
        ].copy()
        if selected_flags.empty:
            st.caption("No missing readings or suspicious spikes were flagged for this building.")
        else:
            st.dataframe(
                selected_flags[
                    [
                        "timestamp",
                        "electricity",
                        "flag_reason",
                        "missing_gap_hours",
                        "spike_baseline",
                        "confirmed_faulty_reading",
                    ]
                ].tail(100),
                width="stretch",
            )
    st.caption(
        "Missing readings are not treated as zero. A suspicious spike is a transparent screening flag; "
        "the available data contains no maintenance labels, so confirmed faulty readings remain zero."
    )

    st.subheader("Potential audit candidates")
    st.caption(
        f"Screening uses saved held-out predictions only. A candidate needs at least "
        f"{AUDIT_MIN_OBSERVED_HOURS} observed test hours and a positive residual streak of at least "
        f"{AUDIT_MIN_STREAK_HOURS} hours. Residual = observed electricity minus model prediction."
    )
    eligible_audits = audit_ranking[audit_ranking["audit_candidate"]].copy()
    if eligible_audits.empty:
        st.info("No building meets the minimum observed-data and sustained-residual thresholds.")
    else:
        audit_display = audit_ranking.copy()
        audit_display["building_id"] = audit_display["building_id"].str.replace(
            "Panther_", "", regex=False
        ).str.replace("_", " ", regex=False).str.title()
        st.dataframe(
            audit_display[
                [
                    "rank",
                    "building_id",
                    "observed_hours",
                    "positive_residual_share_pct",
                    "mean_positive_residual",
                    "sustained_positive_hours",
                    "longest_positive_streak_hours",
                    "audit_score",
                    "audit_candidate",
                ]
            ].rename(
                columns={
                    "rank": "Rank",
                    "building_id": "Building",
                    "observed_hours": "Observed hours",
                    "positive_residual_share_pct": "Positive residual (%)",
                    "mean_positive_residual": "Mean positive residual",
                    "sustained_positive_hours": "Sustained positive hours",
                    "longest_positive_streak_hours": "Longest positive streak",
                    "audit_score": "Screening score",
                    "audit_candidate": "Eligible",
                }
            ).style.format(
                {
                    "Positive residual (%)": "{:.1f}",
                    "Mean positive residual": "{:.2f}",
                    "Screening score": "{:.1f}",
                }
            ),
            width="stretch",
        )
        st.caption(
            "This is a prioritization aid for investigation, not evidence of energy waste. High residuals can reflect "
            "weather mismatch, occupancy, model error, or data quality issues and should be audited separately."
        )

    st.subheader("Illustrative retrofit scenarios")
    try:
        retrofit = build_retrofit_scenarios(selected_predictions)
        retrofit_display = retrofit.rename(
            columns={
                "scenario": "Scenario",
                "assumed_reduction_pct": "Assumed reduction (%)",
                "observed_baseline_hours": "Observed baseline hours",
                "baseline_energy": "Baseline energy",
                "estimated_energy_reduction": "Estimated reduction",
                "scenario_energy": "Scenario energy",
                "baseline_average_hourly": "Baseline average hourly",
                "scenario_average_hourly": "Scenario average hourly",
            }
        )
        st.dataframe(
            retrofit_display.style.format(
                {
                    "Assumed reduction (%)": "{:.0f}",
                    "Baseline energy": "{:.2f}",
                    "Estimated reduction": "{:.2f}",
                    "Scenario energy": "{:.2f}",
                    "Baseline average hourly": "{:.2f}",
                    "Scenario average hourly": "{:.2f}",
                }
            ),
            width="stretch",
        )
        st.caption(
            "Each row is a separate hypothetical reduction applied to the selected building's observed held-out test-period "
            "consumption: estimated reduction = observed baseline energy × assumed percentage. These are illustrative scenario "
            "estimates, not measured or guaranteed savings; no financial savings are assumed."
        )
    except ValueError as exc:
        st.info(f"Retrofit scenarios are unavailable: {exc}")

    st.subheader("Next-day forecast status")
    future_weather = load_future_weather()
    if future_weather.empty:
        st.info(
            "A verified live 24-hour forecast is currently unavailable. The saved baseline requires exactly 24 "
            "complete future weather rows, but no future weather input file is present. The prepared dataset ends "
            f"on {saved_metrics['test_end']}; historical test predictions are not presented as live future forecasts."
        )
        st.caption(
            "If future weather becomes available, place a 24-row data/processed/future_weather.csv file with "
            "timestamp plus the eight weather columns used by the model. No electricity column is needed."
        )
    else:
        try:
            history = load_forecast_history(selected_building)
            forecast = generate_24h_forecast(
                history,
                selected_building,
                future_weather,
                artifact["pipeline"],
                interval_half_width=interval_metrics["residual_quantile"][
                    "absolute_residual_half_width"
                ],
            )
            st.success(
                f"Forecast generated for {selected_building}: "
                f"{forecast['timestamp'].min()} through {forecast['timestamp'].max()}"
            )
            forecast_chart = alt.Chart(forecast).encode(
                x=alt.X("timestamp:T", title="Forecast timestamp")
            )
            forecast_interval = forecast_chart.mark_area(opacity=0.22, color="#5b8ff9").encode(
                y=alt.Y("prediction_lower:Q", title="Predicted electricity"),
                y2="prediction_upper:Q",
            )
            forecast_line = forecast_chart.mark_line(color="#2563eb", strokeWidth=2).encode(
                y=alt.Y("prediction:Q"),
                tooltip=["timestamp:T", "prediction:Q", "prediction_lower:Q", "prediction_upper:Q"],
            )
            st.altair_chart(
                (forecast_interval + forecast_line).properties(height=320), width="stretch"
            )
            st.caption(
                "These are model predictions, not observed consumption. The shaded bounds reuse the validation-calibrated "
                "empirical interval and are not guarantees."
            )
        except ForecastUnavailable as exc:
            st.warning(f"The 24-hour forecast is unavailable: {exc}")
    st.caption(
        "This dashboard reports predictions from the existing trained model and does not retrain or alter the model."
    )


if __name__ == "__main__":
    main()
