import unittest

import numpy as np
import pandas as pd

from src.forecast_24h import ForecastUnavailable, generate_24h_forecast
from src.train_baseline import WEATHER_COLUMNS


class LagOneModel:
    """Small deterministic model used only to test recursive lag handling."""

    def predict(self, frame):
        return frame["lag_1"].to_numpy() + 1.0


class Forecast24HourTests(unittest.TestCase):
    def setUp(self):
        origin = pd.Timestamp("2017-12-31 23:00:00")
        history_timestamps = pd.date_range(origin - pd.Timedelta(hours=71), origin, freq="h")
        self.history = pd.DataFrame(
            {
                "timestamp": history_timestamps,
                "building_id": "Panther_lodging_Kirk",
                "electricity": np.arange(72, dtype=float),
                "primaryspaceusage": "Lodging/residential",
                "sub_primaryspaceusage": "Residence Hall",
                "sqm": 1000.0,
                "yearbuilt": 2000.0,
                "eui": 100.0,
            }
        )
        for column in WEATHER_COLUMNS:
            self.history[column] = 1.0
        future_timestamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
        self.future_weather = pd.DataFrame({"timestamp": future_timestamps})
        for column in WEATHER_COLUMNS:
            self.future_weather[column] = 2.0

    def test_recursive_forecast_has_exactly_24_hours(self):
        result = generate_24h_forecast(
            self.history,
            "Panther_lodging_Kirk",
            self.future_weather,
            LagOneModel(),
            interval_half_width=5.0,
        )
        self.assertEqual(len(result), 24)
        self.assertEqual(result["timestamp"].nunique(), 24)
        self.assertEqual(result["timestamp"].iloc[0], pd.Timestamp("2018-01-01 00:00:00"))
        self.assertEqual(result["timestamp"].iloc[-1], pd.Timestamp("2018-01-01 23:00:00"))
        self.assertEqual(result["prediction"].tolist(), list(np.arange(72, 96, dtype=float)))
        self.assertTrue((result["prediction_lower"] <= result["prediction"]).all())
        self.assertTrue((result["prediction"] <= result["prediction_upper"]).all())

    def test_missing_future_weather_is_rejected(self):
        incomplete = self.future_weather.drop(columns=["windSpeed"])
        with self.assertRaises(ForecastUnavailable):
            generate_24h_forecast(
                self.history,
                "Panther_lodging_Kirk",
                incomplete,
                LagOneModel(),
            )


if __name__ == "__main__":
    unittest.main()
