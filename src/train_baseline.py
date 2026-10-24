"""Train and evaluate a small, leakage-safe Panther forecasting baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_PATH = WORKSPACE / "data" / "processed" / "panther_electricity.csv"
MODEL_PATH = WORKSPACE / "models" / "panther_baseline.joblib"
METRICS_PATH = WORKSPACE / "reports" / "baseline_metrics.json"
REPORT_PATH = WORKSPACE / "reports" / "baseline_evaluation.md"
PREDICTIONS_PATH = WORKSPACE / "reports" / "baseline_test_predictions.csv"

SELECTED_BUILDINGS = [
    "Panther_lodging_Kirk",
    "Panther_office_Clementine",
    "Panther_education_Misty",
    "Panther_education_Cleopatra",
    "Panther_education_Enriqueta",
]

WEATHER_COLUMNS = [
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "precipDepth1HR",
    "precipDepth6HR",
    "seaLvlPressure",
    "windDirection",
    "windSpeed",
]

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "precipDepth1HR",
    "precipDepth6HR",
    "seaLvlPressure",
    "windDirection",
    "windSpeed",
    "sqm",
    "yearbuilt",
    "eui",
    "lag_1",
    "lag_24",
    "rolling_24_mean",
]
CATEGORICAL_FEATURES = ["building_id", "primaryspaceusage", "sub_primaryspaceusage"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
READ_COLUMNS = [
    "timestamp",
    "building_id",
    "electricity",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "yearbuilt",
    "eui",
    *WEATHER_COLUMNS,
]


def load_selected_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read only the selected buildings and columns from the prepared CSV."""
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=READ_COLUMNS,
        parse_dates=["timestamp"],
        chunksize=200_000,
        low_memory=True,
    ):
        selected = chunk[chunk["building_id"].isin(SELECTED_BUILDINGS)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("None of the selected buildings were found in the prepared dataset")
    data = pd.concat(parts, ignore_index=True)
    found = set(data["building_id"])
    missing_buildings = sorted(set(SELECTED_BUILDINGS) - found)
    if missing_buildings:
        raise ValueError(f"Selected buildings missing from dataset: {missing_buildings}")
    return data.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and past-only target features without looking forward."""
    result = data.copy()
    result["electricity"] = pd.to_numeric(result["electricity"], errors="coerce")
    for column in [*WEATHER_COLUMNS, "sqm", "yearbuilt", "eui"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    timestamp = result["timestamp"]
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.day_of_week
    result["month"] = timestamp.dt.month
    result["day_of_year"] = timestamp.dt.day_of_year
    result["is_weekend"] = (timestamp.dt.day_of_week >= 5).astype(int)

    grouped_target = result.groupby("building_id", sort=False)["electricity"]
    result["lag_1"] = grouped_target.shift(1)
    result["lag_24"] = grouped_target.shift(24)
    result["rolling_24_mean"] = grouped_target.transform(
        lambda values: values.shift(1).rolling(24, min_periods=12).mean()
    )
    return result


def chronological_split(data: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.Series(data["timestamp"].drop_duplicates().sort_values().to_numpy())
    train_end = timestamps.iloc[int(len(timestamps) * 0.70) - 1]
    validation_end = timestamps.iloc[int(len(timestamps) * 0.85) - 1]
    return train_end, validation_end


def make_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def main() -> None:
    data = build_features(load_selected_data())
    train_end, validation_end = chronological_split(data)

    data["split"] = np.select(
        [
            data["timestamp"] <= train_end,
            data["timestamp"] <= validation_end,
        ],
        ["train", "validation"],
        default="test",
    )

    observed_target = data["electricity"].notna()
    usable_lag = data["lag_24"].notna()
    train_mask = (data["split"] == "train") & observed_target & usable_lag
    validation_mask = (data["split"] == "validation") & observed_target & usable_lag
    test_mask = (data["split"] == "test") & observed_target & usable_lag

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("The selected buildings do not provide usable train and test rows")

    pipeline = make_pipeline()
    pipeline.fit(data.loc[train_mask, FEATURE_COLUMNS], data.loc[train_mask, "electricity"])

    predictions = {}
    for split_name, mask in [("validation", validation_mask), ("test", test_mask)]:
        actual = data.loc[mask, "electricity"]
        model_prediction = pipeline.predict(data.loc[mask, FEATURE_COLUMNS])
        benchmark_prediction = data.loc[mask, "lag_24"].to_numpy()
        predictions[split_name] = {
            "model": metrics(actual, model_prediction),
            "benchmark": metrics(actual, benchmark_prediction),
            "rows": int(mask.sum()),
        }
        if split_name == "test":
            test_predictions = data.loc[mask, ["timestamp", "building_id", "electricity"]].copy()
            test_predictions["model_prediction"] = model_prediction
            test_predictions["same_hour_previous_day"] = benchmark_prediction
            test_predictions.to_csv(PREDICTIONS_PATH, index=False)

    artifact = {
        "pipeline": pipeline,
        "selected_buildings": SELECTED_BUILDINGS,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_end": str(train_end),
        "validation_end": str(validation_end),
        "benchmark": "lag_24: observed same building, same hour on the previous day",
    }
    joblib.dump(artifact, MODEL_PATH)

    report = {
        "selected_buildings": SELECTED_BUILDINGS,
        "source_rows_for_selected_buildings": int(len(data)),
        "source_timestamps": int(data["timestamp"].nunique()),
        "train_end": str(train_end),
        "validation_end": str(validation_end),
        "test_start": str(data.loc[data["split"] == "test", "timestamp"].min()),
        "test_end": str(data.loc[data["split"] == "test", "timestamp"].max()),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "missing_target_rows_selected": int(data["electricity"].isna().sum()),
        "missing_target_rate_selected": float(data["electricity"].isna().mean()),
        "metrics": predictions,
        "target_missing_policy": "Missing electricity targets were excluded; no zero filling or target interpolation was used.",
        "feature_leakage_policy": "Lag and rolling features use only prior timestamps within the same building. The pipeline is fitted on train rows only.",
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    test = predictions["test"]
    report_lines = [
        "# Enpres.x Baseline Evaluation",
        "",
        "## Scope",
        "",
        f"Selected buildings: {', '.join(SELECTED_BUILDINGS)}",
        "",
        f"Rows loaded for selected buildings: **{len(data):,}**",
        f"Missing electricity targets: **{report['missing_target_rows_selected']:,} ({report['missing_target_rate_selected']:.2%})**",
        "",
        "Missing electricity targets were excluded from training and scoring. They were never replaced with zero or interpolated.",
        "",
        "## Chronological split",
        "",
        f"- Training: through `{train_end}` (**{train_mask.sum():,}** usable rows)",
        f"- Validation: after training through `{validation_end}` (**{validation_mask.sum():,}** usable rows)",
        f"- Test: after validation through `{report['test_end']}` (**{test_mask.sum():,}** usable rows)",
        "",
        "Usable rows require an observed target and an observed same-building value 24 hours earlier. Weather feature gaps were handled by a training-fitted median imputer.",
        "",
        "## Test results",
        "",
        "Both methods were evaluated on exactly the same held-out test rows.",
        "",
        "| Method | MAE | RMSE | Rows |",
        "|---|---:|---:|---:|",
        f"| Model | {test['model']['mae']:.4f} | {test['model']['rmse']:.4f} | {test['rows']:,} |",
        f"| Same-hour previous-day benchmark | {test['benchmark']['mae']:.4f} | {test['benchmark']['rmse']:.4f} | {test['rows']:,} |",
        "",
        "## Validation results",
        "",
        "| Method | MAE | RMSE | Rows |",
        "|---|---:|---:|---:|",
        f"| Model | {predictions['validation']['model']['mae']:.4f} | {predictions['validation']['model']['rmse']:.4f} | {predictions['validation']['rows']:,} |",
        f"| Same-hour previous-day benchmark | {predictions['validation']['benchmark']['mae']:.4f} | {predictions['validation']['benchmark']['rmse']:.4f} | {predictions['validation']['rows']:,} |",
        "",
        "## Saved outputs",
        "",
        "- `models/panther_baseline.joblib`",
        "- `reports/baseline_metrics.json`",
        "- `reports/baseline_test_predictions.csv`",
        "",
        "The saved model is a scikit-learn pipeline containing training-fitted preprocessing and a histogram gradient-boosting regressor. The benchmark is not a trained model; it uses the observed lag-24 value.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"MODEL={MODEL_PATH}")
    print(f"METRICS={METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"PREDICTIONS={PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
