"""Validated recursive 24-hour forecast helper.

The helper intentionally requires a complete, externally supplied future-weather
frame. It never fills future weather values and never uses future actual electricity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from train_baseline import FEATURE_COLUMNS, WEATHER_COLUMNS
except ModuleNotFoundError:  # Supports importing as ``src.forecast_24h`` in tests.
    from src.train_baseline import FEATURE_COLUMNS, WEATHER_COLUMNS


FORECAST_HORIZON = 24
STATIC_COLUMNS = [
    "building_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "yearbuilt",
    "eui",
]


class ForecastUnavailable(ValueError):
    """Raised when the inputs cannot support a trustworthy forecast."""


def validate_future_weather(
    future_weather: pd.DataFrame, expected_timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """Require exactly 24 hourly weather rows with no missing weather values."""
    required = {"timestamp", *WEATHER_COLUMNS}
    missing_columns = sorted(required - set(future_weather.columns))
    if missing_columns:
        raise ForecastUnavailable(
            "Future weather is missing required columns: " + ", ".join(missing_columns)
        )

    weather = future_weather[["timestamp", *WEATHER_COLUMNS]].copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    if weather["timestamp"].isna().any():
        raise ForecastUnavailable("Future weather contains invalid timestamps")
    weather = weather.sort_values("timestamp").reset_index(drop=True)
    if not weather["timestamp"].is_unique:
        raise ForecastUnavailable("Future weather contains duplicate timestamps")
    if not weather["timestamp"].equals(pd.Series(expected_timestamps, name="timestamp")):
        raise ForecastUnavailable(
            "Future weather must contain exactly the next 24 hourly timestamps"
        )

    for column in WEATHER_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    if weather[WEATHER_COLUMNS].isna().any().any():
        raise ForecastUnavailable("Future weather contains missing values")
    return weather


def generate_24h_forecast(
    history: pd.DataFrame,
    building_id: str,
    future_weather: pd.DataFrame,
    model,
    interval_half_width: float | None = None,
) -> pd.DataFrame:
    """Generate 24 recursive predictions from observed history and future weather.

    The model is expected to be the saved scikit-learn pipeline. The last 48
    historical electricity readings must be observed and hourly so that lag-1,
    lag-24, and the past-only rolling mean are all available without imputation.
    """
    required_history = {"timestamp", "building_id", "electricity", *STATIC_COLUMNS[1:]}
    missing_columns = sorted(required_history - set(history.columns))
    if missing_columns:
        raise ForecastUnavailable(
            "History is missing required columns: " + ", ".join(missing_columns)
        )

    building_history = history[history["building_id"] == building_id].copy()
    if building_history.empty:
        raise ForecastUnavailable(f"No history is available for {building_id}")
    building_history["timestamp"] = pd.to_datetime(
        building_history["timestamp"], errors="coerce"
    )
    building_history["electricity"] = pd.to_numeric(
        building_history["electricity"], errors="coerce"
    )
    building_history = building_history.sort_values("timestamp").reset_index(drop=True)
    origin = building_history["timestamp"].max()
    if pd.isna(origin):
        raise ForecastUnavailable("History has no valid forecast origin timestamp")

    recent = building_history.tail(48).reset_index(drop=True)
    if len(recent) != 48:
        raise ForecastUnavailable("At least 48 historical hourly rows are required")
    expected_recent = pd.date_range(origin - pd.Timedelta(hours=47), origin, freq="h")
    if not recent["timestamp"].equals(pd.Series(expected_recent, name="timestamp")):
        raise ForecastUnavailable("The last 48 history timestamps are not hourly and contiguous")
    if recent["electricity"].isna().any():
        raise ForecastUnavailable(
            "The last 48 historical electricity readings include missing values"
        )

    future_timestamps = pd.date_range(
        origin + pd.Timedelta(hours=1), periods=FORECAST_HORIZON, freq="h"
    )
    weather = validate_future_weather(future_weather, future_timestamps)
    static = building_history.iloc[-1]
    values = {
        timestamp: float(value)
        for timestamp, value in zip(
            building_history["timestamp"], building_history["electricity"]
        )
        if pd.notna(value)
    }

    forecast_rows = []
    for weather_row in weather.itertuples(index=False):
        timestamp = weather_row.timestamp
        lag_1 = values.get(timestamp - pd.Timedelta(hours=1))
        lag_24 = values.get(timestamp - pd.Timedelta(hours=24))
        rolling_values = [
            values.get(timestamp - pd.Timedelta(hours=hours))
            for hours in range(1, 25)
        ]
        if lag_1 is None or lag_24 is None or any(value is None for value in rolling_values):
            raise ForecastUnavailable(
                "Required past electricity lags are unavailable at "
                f"{timestamp}"
            )

        row = {column: static[column] for column in STATIC_COLUMNS}
        row.update(
            {
                "timestamp": timestamp,
                "hour": timestamp.hour,
                "day_of_week": timestamp.day_of_week,
                "month": timestamp.month,
                "day_of_year": timestamp.day_of_year,
                "is_weekend": int(timestamp.day_of_week >= 5),
                "lag_1": lag_1,
                "lag_24": lag_24,
                "rolling_24_mean": float(np.mean(rolling_values)),
            }
        )
        row.update({column: getattr(weather_row, column) for column in WEATHER_COLUMNS})
        feature_frame = pd.DataFrame([row])[FEATURE_COLUMNS]
        prediction = float(model.predict(feature_frame)[0])
        prediction = max(0.0, prediction)
        values[timestamp] = prediction
        output = {
            "forecast_origin": origin,
            "timestamp": timestamp,
            "building_id": building_id,
            "prediction": prediction,
        }
        if interval_half_width is not None:
            output["prediction_lower"] = max(0.0, prediction - interval_half_width)
            output["prediction_upper"] = prediction + interval_half_width
        forecast_rows.append(output)

    return pd.DataFrame(forecast_rows)
