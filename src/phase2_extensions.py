"""Small, transparent Phase 2 analyses for the Enpres.x dashboard.

These helpers are deliberately descriptive rather than corrective: they flag
rows for review, but never rewrite electricity readings or claim that a flag is
proof of meter failure or energy waste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LONG_GAP_THRESHOLD_HOURS = 24
SPIKE_WINDOW_HOURS = 24
SPIKE_MULTIPLIER = 3.0
SPIKE_MAD_MULTIPLIER = 6.0
AUDIT_MIN_OBSERVED_HOURS = 168
AUDIT_MIN_STREAK_HOURS = 6
RETROFIT_SCENARIOS = (0.10, 0.08)


def _median_absolute_deviation(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def add_meter_quality_flags(
    data: pd.DataFrame,
    *,
    long_gap_threshold_hours: int = LONG_GAP_THRESHOLD_HOURS,
    spike_window_hours: int = SPIKE_WINDOW_HOURS,
) -> pd.DataFrame:
    """Return a copy with descriptive missing-gap and spike flags.

    A suspicious spike is an observed value above three times the centered
    24-hour rolling median and more than six rolling MADs above that median.
    This is a screening rule only; without maintenance or validation labels,
    no row is marked as a confirmed faulty meter reading.
    """
    required = {"timestamp", "building_id", "electricity"}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError("Meter-quality data is missing columns: " + ", ".join(missing_columns))
    if long_gap_threshold_hours < 1 or spike_window_hours < 3:
        raise ValueError("Gap and spike windows must be positive and practical")

    result = data[["timestamp", "building_id", "electricity"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["electricity"] = pd.to_numeric(result["electricity"], errors="coerce")
    result = result.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    result["missing_reading"] = result["electricity"].isna()
    observed_break = (~result["missing_reading"]).groupby(result["building_id"]).cumsum()
    gap_lengths = result["missing_reading"].groupby(
        [result["building_id"], observed_break], sort=False
    ).transform("sum")
    result["missing_gap_hours"] = gap_lengths.where(result["missing_reading"], 0).astype(int)
    result["long_missing_gap"] = result["missing_reading"] & (
        result["missing_gap_hours"] >= long_gap_threshold_hours
    )
    result["long_missing_gap_start"] = result["long_missing_gap"] & ~result.groupby(
        "building_id"
    )["long_missing_gap"].shift(fill_value=False)

    rolling_median = result.groupby("building_id", sort=False)["electricity"].transform(
        lambda values: values.rolling(
            spike_window_hours, min_periods=max(6, spike_window_hours // 2), center=True
        ).median()
    )
    rolling_mad = result.groupby("building_id", sort=False)["electricity"].transform(
        lambda values: values.rolling(
            spike_window_hours,
            min_periods=max(6, spike_window_hours // 2),
            center=True,
        ).apply(_median_absolute_deviation, raw=True)
    )
    result["spike_baseline"] = rolling_median
    result["suspicious_spike"] = (
        result["electricity"].notna()
        & rolling_median.notna()
        & (rolling_median > 0)
        & (result["electricity"] > SPIKE_MULTIPLIER * rolling_median)
        & (
            result["electricity"] - rolling_median
            > SPIKE_MAD_MULTIPLIER * rolling_mad.fillna(0)
        )
    )
    result["confirmed_faulty_reading"] = False
    result["flag_reason"] = np.select(
        [result["long_missing_gap"], result["missing_reading"], result["suspicious_spike"]],
        ["long missing gap", "missing reading", "suspicious spike"],
        default="observed without Phase 2 flag",
    )
    return result


def summarize_meter_quality(flags: pd.DataFrame) -> pd.DataFrame:
    """Summarize quality flags by building without changing source readings."""
    rows = []
    for building_id, group in flags.groupby("building_id", sort=True):
        rows.append(
            {
                "building_id": building_id,
                "total_hours": int(len(group)),
                "observed_hours": int(group["electricity"].notna().sum()),
                "missing_hours": int(group["missing_reading"].sum()),
                "coverage_pct": float(group["electricity"].notna().mean() * 100),
                "longest_missing_gap_hours": int(group["missing_gap_hours"].max()),
                "long_missing_gap_count": int(group["long_missing_gap_start"].sum()),
                "long_missing_gap_hours": int(group["long_missing_gap"].sum()),
                "suspicious_spike_count": int(group["suspicious_spike"].sum()),
                "confirmed_faulty_count": int(group["confirmed_faulty_reading"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _positive_streak_lengths(positive: pd.Series) -> pd.Series:
    run_id = positive.ne(positive.shift(fill_value=False)).cumsum()
    run_length = positive.groupby(run_id, sort=False).transform("sum")
    return run_length.where(positive, 0).astype(int)


def build_audit_ranking(
    predictions: pd.DataFrame,
    *,
    min_observed_hours: int = AUDIT_MIN_OBSERVED_HOURS,
    min_streak_hours: int = AUDIT_MIN_STREAK_HOURS,
) -> pd.DataFrame:
    """Rank possible audit candidates from sustained positive test residuals.

    Residual is observed electricity minus the saved model prediction. The
    score is ``sustained_positive_hours * mean_positive_residual`` where a
    sustained run is at least six consecutive positive-residual hours. The
    ranking is a screening aid, not proof of energy waste.
    """
    required = {"building_id", "electricity", "model_prediction"}
    missing_columns = sorted(required - set(predictions.columns))
    if missing_columns:
        raise ValueError("Audit ranking is missing columns: " + ", ".join(missing_columns))
    if min_observed_hours < 1 or min_streak_hours < 1:
        raise ValueError("Audit thresholds must be positive")

    data = predictions[["building_id", "timestamp", "electricity", "model_prediction"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["electricity"] = pd.to_numeric(data["electricity"], errors="coerce")
    data["model_prediction"] = pd.to_numeric(data["model_prediction"], errors="coerce")
    data = data.dropna(subset=["building_id", "electricity", "model_prediction"])
    data = data.sort_values(["building_id", "timestamp"])
    data["residual"] = data["electricity"] - data["model_prediction"]
    data["positive_residual"] = data["residual"] > 0
    data["positive_streak_hours"] = data.groupby("building_id", sort=False)[
        "positive_residual"
    ].transform(_positive_streak_lengths)

    rows = []
    for building_id, group in data.groupby("building_id", sort=True):
        positive = group[group["positive_residual"]]
        sustained = group[group["positive_streak_hours"] >= min_streak_hours]
        observed_hours = int(len(group))
        positive_hours = int(len(positive))
        mean_positive = float(positive["residual"].mean()) if not positive.empty else 0.0
        sustained_hours = int(len(sustained))
        max_streak = int(group["positive_streak_hours"].max()) if not group.empty else 0
        eligible = observed_hours >= min_observed_hours and sustained_hours > 0
        rows.append(
            {
                "building_id": building_id,
                "observed_hours": observed_hours,
                "positive_residual_hours": positive_hours,
                "positive_residual_share_pct": positive_hours / observed_hours * 100,
                "mean_positive_residual": mean_positive,
                "sustained_positive_hours": sustained_hours,
                "longest_positive_streak_hours": max_streak,
                "audit_score": sustained_hours * mean_positive,
                "audit_candidate": eligible,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["rank"] = pd.NA
    eligible_index = result["audit_candidate"]
    ranked = result.loc[eligible_index].sort_values(
        ["audit_score", "sustained_positive_hours", "mean_positive_residual"],
        ascending=False,
    )
    result.loc[ranked.index, "rank"] = range(1, len(ranked) + 1)
    return result.sort_values(["audit_candidate", "rank"], ascending=[False, True]).reset_index(
        drop=True
    )


def build_retrofit_scenarios(
    observed_frame: pd.DataFrame,
    *,
    scenarios: tuple[float, ...] = RETROFIT_SCENARIOS,
    minimum_baseline_hours: int = 24,
) -> pd.DataFrame:
    """Calculate separate illustrative reductions from observed consumption."""
    if "electricity" not in observed_frame.columns:
        raise ValueError("Retrofit scenarios require an electricity column")
    baseline = pd.to_numeric(observed_frame["electricity"], errors="coerce").dropna()
    if len(baseline) < minimum_baseline_hours:
        raise ValueError(
            f"Only {len(baseline)} observed hours are available; at least "
            f"{minimum_baseline_hours} are required for a scenario baseline"
        )
    if any(rate <= 0 or rate >= 1 for rate in scenarios):
        raise ValueError("Scenario reductions must be between 0% and 100%")

    baseline_total = float(baseline.sum())
    baseline_average = float(baseline.mean())
    rows = []
    for rate in scenarios:
        rows.append(
            {
                "scenario": f"Illustrative {rate:.0%} reduction",
                "assumed_reduction_pct": rate * 100,
                "observed_baseline_hours": int(len(baseline)),
                "baseline_energy": baseline_total,
                "estimated_energy_reduction": baseline_total * rate,
                "scenario_energy": baseline_total * (1 - rate),
                "baseline_average_hourly": baseline_average,
                "scenario_average_hourly": baseline_average * (1 - rate),
            }
        )
    return pd.DataFrame(rows)
