"""Evaluate the fixed Panther model on an unseen site without refitting.

The evaluation uses raw electricity, metadata, and weather files only to build
features for one non-Panther site. The saved Panther pipeline is never fitted
on the unseen site's electricity targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from train_baseline import FEATURE_COLUMNS, WEATHER_COLUMNS, build_features
except ModuleNotFoundError:  # Supports importing as ``src.evaluate_unseen_site``.
    from src.train_baseline import FEATURE_COLUMNS, WEATHER_COLUMNS, build_features


WORKSPACE = Path(__file__).resolve().parents[1]
RAW_PATH = WORKSPACE / "data" / "raw"
MODEL_PATH = WORKSPACE / "models" / "panther_baseline.joblib"
REPORT_PATH = WORKSPACE / "reports" / "unseen_site_evaluation.md"
SITE = "Gator"

METADATA_COLUMNS = [
    "building_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "yearbuilt",
    "eui",
]


def read_site_frame(site: str = SITE) -> tuple[pd.DataFrame, dict]:
    """Build one unseen site's long frame from untouched raw datasets."""
    metadata = pd.read_csv(RAW_PATH / "metadata.csv", low_memory=False)
    with (RAW_PATH / "electricity_cleaned.csv").open(encoding="utf-8") as source:
        electricity_columns = source.readline().strip().split(",")

    site_buildings = [column for column in electricity_columns if column.startswith(f"{site}_")]
    if not site_buildings:
        raise ValueError(f"No electricity columns were found for site {site}")
    site_metadata = metadata[metadata["building_id"].isin(site_buildings)].copy()
    missing_metadata = sorted(set(site_buildings) - set(site_metadata["building_id"]))
    if missing_metadata:
        raise ValueError(f"Metadata is missing unseen-site buildings: {missing_metadata[:5]}")

    electricity = pd.read_csv(
        RAW_PATH / "electricity_cleaned.csv",
        usecols=["timestamp", *site_buildings],
        parse_dates=["timestamp"],
        low_memory=False,
    )
    long = electricity.melt(
        id_vars=["timestamp"],
        var_name="building_id",
        value_name="electricity",
    )
    long = long.merge(site_metadata[METADATA_COLUMNS], on="building_id", how="left")

    weather = pd.read_csv(
        RAW_PATH / "weather.csv",
        usecols=["timestamp", "site_id", *WEATHER_COLUMNS],
        parse_dates=["timestamp"],
        low_memory=False,
    )
    site_weather = weather[weather["site_id"] == site].drop(columns=["site_id"])
    if site_weather.empty:
        raise ValueError(f"No weather rows were found for site {site}")
    long = long.merge(site_weather, on="timestamp", how="left")
    long = long.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    source_summary = {
        "site": site,
        "building_count": len(site_buildings),
        "electricity_rows": int(len(electricity)),
        "electricity_start": str(electricity["timestamp"].min()),
        "electricity_end": str(electricity["timestamp"].max()),
        "electricity_missing_values": int(long["electricity"].isna().sum()),
        "electricity_missing_rate": float(long["electricity"].isna().mean()),
        "weather_rows": int(len(site_weather)),
        "weather_start": str(site_weather["timestamp"].min()),
        "weather_end": str(site_weather["timestamp"].max()),
        "weather_duplicate_timestamps": int(site_weather["timestamp"].duplicated().sum()),
        "weather_missing_by_column": {
            column: int(site_weather[column].isna().sum()) for column in WEATHER_COLUMNS
        },
        "weather_complete_rows": int(site_weather[WEATHER_COLUMNS].notna().all(axis=1).sum()),
        "metadata_missing_yearbuilt": int(site_metadata["yearbuilt"].isna().sum()),
        "metadata_missing_eui": int(site_metadata["eui"].isna().sum()),
        "metadata_missing_sqm": int(site_metadata["sqm"].isna().sum()),
    }
    return long, source_summary


def evaluate_fixed_model(
    featured: pd.DataFrame,
    pipeline,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate the fixed pipeline on strict, common eligible rows."""
    weather_complete = featured[WEATHER_COLUMNS].notna().all(axis=1)
    required_past_features = featured[["lag_1", "lag_24", "rolling_24_mean"]].notna().all(axis=1)
    observed_target = featured["electricity"].notna()
    eligible = observed_target & required_past_features & weather_complete
    if not eligible.any():
        raise ValueError("The unseen site has no rows with observed targets, past lags, and complete weather")

    eligible_frame = featured.loc[eligible].copy()
    model_prediction = pipeline.predict(eligible_frame[FEATURE_COLUMNS])
    benchmark_prediction = eligible_frame["lag_24"].to_numpy()
    actual = eligible_frame["electricity"].to_numpy()
    predictions = eligible_frame[["timestamp", "building_id", "electricity", "lag_24"]].copy()
    predictions["model_prediction"] = model_prediction
    predictions["same_hour_previous_day"] = benchmark_prediction
    metrics = {
        "eligible_rows": int(len(eligible_frame)),
        "eligible_buildings": int(eligible_frame["building_id"].nunique()),
        "eligible_start": str(eligible_frame["timestamp"].min()),
        "eligible_end": str(eligible_frame["timestamp"].max()),
        "observed_target_rows": int(observed_target.sum()),
        "past_feature_rows": int(required_past_features.sum()),
        "complete_weather_rows": int(weather_complete.sum()),
        "model": {
            "mae": float(mean_absolute_error(actual, model_prediction)),
            "rmse": float(np.sqrt(mean_squared_error(actual, model_prediction))),
        },
        "benchmark": {
            "mae": float(mean_absolute_error(actual, benchmark_prediction)),
            "rmse": float(np.sqrt(mean_squared_error(actual, benchmark_prediction))),
        },
    }
    return predictions, metrics


def format_report(source: dict, metrics: dict) -> str:
    weather_missing = ", ".join(
        f"{column}: {count:,}" for column, count in source["weather_missing_by_column"].items() if count
    ) or "none"
    model = metrics["model"]
    benchmark = metrics["benchmark"]
    model_mae_delta = model["mae"] - benchmark["mae"]
    model_rmse_delta = model["rmse"] - benchmark["rmse"]
    return f"""# Unseen-Site Evaluation

## Scope and safeguards

The saved Panther-trained model was evaluated without refitting, feature selection, or tuning on the `{source['site']}` site's electricity targets. Raw electricity, metadata, and weather files were read but not modified. The comparison uses the same eligible rows for the model and the same-hour-previous-day benchmark.

The existing training-fitted preprocessing was reused. Missing numeric metadata values are accepted by the saved pipeline's training-fitted median imputer; unknown categorical building IDs are handled by the saved encoder's `handle_unknown=ignore` behavior. No missing electricity targets were filled with zero.

## Site and source availability

- Site selected: **{source['site']}**
- Buildings with electricity columns and metadata: **{source['building_count']}**
- Electricity rows: **{source['electricity_rows']:,}** across `{source['electricity_start']}` through `{source['electricity_end']}`
- Missing electricity values: **{source['electricity_missing_values']:,} ({source['electricity_missing_rate']:.2%})**
- Weather rows: **{source['weather_rows']:,}** across `{source['weather_start']}` through `{source['weather_end']}`
- Duplicate weather timestamps: **{source['weather_duplicate_timestamps']}**
- Weather rows complete across all eight model weather fields: **{source['weather_complete_rows']:,}**
- Weather missing-value counts: **{weather_missing}**
- Metadata missing `yearbuilt`: **{source['metadata_missing_yearbuilt']}** buildings
- Metadata missing `eui`: **{source['metadata_missing_eui']}** buildings
- Metadata missing `sqm`: **{source['metadata_missing_sqm']}** buildings

Because weather completeness is not universal, the evaluation conservatively retained only rows with all eight weather fields present. The saved preprocessing still handles missing numeric static metadata using its original training-fitted imputer.

## Eligible evaluation rows

- Observed electricity targets: **{metrics['observed_target_rows']:,}**
- Rows with all required past features: **{metrics['past_feature_rows']:,}**
- Rows with complete weather: **{metrics['complete_weather_rows']:,}**
- Final common eligible rows: **{metrics['eligible_rows']:,}** across **{metrics['eligible_buildings']}** buildings
- Evaluation period represented: `{metrics['eligible_start']}` through `{metrics['eligible_end']}`

Eligibility required an observed electricity target, valid `lag_1`, `lag_24`, and `rolling_24_mean`, plus complete values for all eight weather fields. The benchmark uses the observed `lag_24` value on exactly these rows.

## Results

| Method | MAE | RMSE | Rows |
|---|---:|---:|---:|
| Fixed Panther model | {model['mae']:.4f} | {model['rmse']:.4f} | {metrics['eligible_rows']:,} |
| Same-hour previous-day benchmark | {benchmark['mae']:.4f} | {benchmark['rmse']:.4f} | {metrics['eligible_rows']:,} |

Model minus benchmark: **MAE {model_mae_delta:+.4f}**, **RMSE {model_rmse_delta:+.4f}**. Negative values would indicate lower error for the model; positive values indicate the benchmark was better on this unseen-site slice.

## Interpretation and limitations

This is one cross-site evaluation of the fixed Panther model on Gator. It does not prove universal generalization to other sites. The unseen site's static `yearbuilt` and `eui` metadata are missing for all evaluated buildings and therefore rely on the saved Panther training medians; this is a material domain-shift limitation. Weather completeness also reduced the evaluation sample. The results should be treated as evidence about this specific site and eligible period only.

The dashboard remains Panther-focused; this report does not add the unseen site to the dashboard or alter the saved model.
"""


def main() -> None:
    frame, source_summary = read_site_frame()
    featured = build_features(frame)
    artifact = joblib.load(MODEL_PATH)
    _, metrics = evaluate_fixed_model(featured, artifact["pipeline"])
    REPORT_PATH.write_text(format_report(source_summary, metrics), encoding="utf-8")
    print(json.dumps({"source": source_summary, "metrics": metrics}, indent=2))
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
