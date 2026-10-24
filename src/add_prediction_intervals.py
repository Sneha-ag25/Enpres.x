"""Add empirical prediction intervals using validation-period residuals."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_baseline import (
    DATA_PATH,
    FEATURE_COLUMNS,
    SELECTED_BUILDINGS,
    build_features,
    load_selected_data,
)


WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_PATH = WORKSPACE / "models" / "panther_baseline.joblib"
BASELINE_METRICS_PATH = WORKSPACE / "reports" / "baseline_metrics.json"
BASELINE_TEST_PATH = WORKSPACE / "reports" / "baseline_test_predictions.csv"
INTERVAL_TEST_PATH = WORKSPACE / "reports" / "baseline_test_predictions_with_intervals.csv"
INTERVAL_METRICS_PATH = WORKSPACE / "reports" / "prediction_interval_metrics.json"
INTERVAL_REPORT_PATH = WORKSPACE / "reports" / "prediction_intervals.md"

NOMINAL_COVERAGE = 0.80


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    observed = actual.notna() & lower.notna() & upper.notna()
    if not observed.any():
        return float("nan")
    inside = (actual[observed] >= lower[observed]) & (actual[observed] <= upper[observed])
    return float(inside.mean())


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    baseline_metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))

    # Recreate the already-used feature frame; this does not retrain the model.
    featured = build_features(load_selected_data(DATA_PATH))
    train_end = pd.Timestamp(artifact["train_end"])
    validation_end = pd.Timestamp(artifact["validation_end"])
    observed_target = featured["electricity"].notna()
    usable_lag = featured["lag_24"].notna()
    validation_mask = (
        (featured["timestamp"] > train_end)
        & (featured["timestamp"] <= validation_end)
        & observed_target
        & usable_lag
    )
    if validation_mask.sum() == 0:
        raise RuntimeError("No usable validation rows were available for residual calibration")

    validation_actual = featured.loc[validation_mask, "electricity"]
    validation_prediction = artifact["pipeline"].predict(
        featured.loc[validation_mask, FEATURE_COLUMNS]
    )
    validation_residuals = validation_actual.to_numpy() - validation_prediction
    absolute_residuals = np.abs(validation_residuals)
    residual_quantile = float(np.quantile(absolute_residuals, NOMINAL_COVERAGE))

    test_predictions = pd.read_csv(BASELINE_TEST_PATH, parse_dates=["timestamp"])
    observed_test = test_predictions["electricity"].notna()
    if not observed_test.all():
        raise RuntimeError("Test predictions contain missing actual values; refusing to score intervals on them")

    # Clip only the lower bound because electricity consumption cannot be negative.
    test_predictions["prediction_lower"] = np.maximum(
        0.0, test_predictions["model_prediction"] - residual_quantile
    )
    test_predictions["prediction_upper"] = (
        test_predictions["model_prediction"] + residual_quantile
    )

    test_coverage = coverage(
        test_predictions["electricity"],
        test_predictions["prediction_lower"],
        test_predictions["prediction_upper"],
    )
    building_coverage = {}
    building_counts = {}
    for building_id, group in test_predictions.groupby("building_id", sort=True):
        building_coverage[building_id] = coverage(
            group["electricity"], group["prediction_lower"], group["prediction_upper"]
        )
        building_counts[building_id] = int(len(group))

    test_predictions.to_csv(INTERVAL_TEST_PATH, index=False)

    interval_metrics = {
        "method": "symmetric absolute-residual interval",
        "nominal_coverage": NOMINAL_COVERAGE,
        "calibration_period": {
            "start_exclusive": str(train_end),
            "end_inclusive": str(validation_end),
            "rows": int(validation_mask.sum()),
        },
        "residual_quantile": {
            "quantile": NOMINAL_COVERAGE,
            "absolute_residual_half_width": residual_quantile,
            "validation_absolute_residual_count": int(len(absolute_residuals)),
        },
        "test_period": {
            "start": baseline_metrics["test_start"],
            "end": baseline_metrics["test_end"],
            "observed_rows_scored": int(observed_test.sum()),
            "coverage": test_coverage,
        },
        "coverage_by_building": building_coverage,
        "rows_by_building": building_counts,
        "bounds_policy": "lower = max(0, model_prediction - half_width); upper = model_prediction + half_width",
        "test_actuals_used_for_interval_construction": False,
        "note": "80% is nominal validation-calibrated coverage; realized test coverage may differ.",
    }
    INTERVAL_METRICS_PATH.write_text(json.dumps(interval_metrics, indent=2), encoding="utf-8")

    report_lines = [
        "# Enpres.x Empirical 80% Prediction Intervals",
        "",
        "## Method",
        "",
        "The interval uses a symmetric absolute-residual quantile calibrated only on the validation period.",
        "For each prediction, the bounds are:",
        "",
        "```text",
        "lower = max(0, model_prediction - q80_absolute_validation_residual)",
        "upper = model_prediction + q80_absolute_validation_residual",
        "```",
        "",
        f"- Nominal target coverage: **{NOMINAL_COVERAGE:.0%}**",
        f"- Validation rows used for calibration: **{validation_mask.sum():,}**",
        f"- Validation period: after `{train_end}` through `{validation_end}`",
        f"- Calibrated half-width: **{residual_quantile:.4f}** consumption units",
        "",
        "The test actual values were not used to construct or tune the interval. The lower bound was clipped at zero because electricity consumption cannot be negative.",
        "",
        "## Held-out test coverage",
        "",
        f"- Test period: `{baseline_metrics['test_start']}` through `{baseline_metrics['test_end']}`",
        f"- Observed test rows scored: **{observed_test.sum():,}**",
        f"- Overall empirical coverage: **{test_coverage:.2%}**",
        "",
        "80% is the nominal target based on validation calibration. Actual held-out test coverage can be higher or lower because the test period may have a different error distribution.",
        "",
        "## Coverage by selected building",
        "",
        "| Building | Test rows | Coverage |",
        "|---|---:|---:|",
    ]
    for building_id in SELECTED_BUILDINGS:
        report_lines.append(
            f"| {building_id} | {building_counts.get(building_id, 0):,} | {building_coverage.get(building_id, float('nan')):.2%} |"
        )
    report_lines.extend(
        [
            "",
            "## Saved outputs",
            "",
            "- `reports/baseline_test_predictions_with_intervals.csv`",
            "- `reports/prediction_interval_metrics.json`",
            "",
            "The original `reports/baseline_test_predictions.csv` and baseline evaluation report were preserved unchanged.",
            "",
        ]
    )
    INTERVAL_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(interval_metrics, indent=2))
    print(f"INTERVAL_PREDICTIONS={INTERVAL_TEST_PATH}")
    print(f"INTERVAL_METRICS={INTERVAL_METRICS_PATH}")
    print(f"INTERVAL_REPORT={INTERVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
