import unittest

import numpy as np
import pandas as pd

from src.evaluate_unseen_site import evaluate_fixed_model
from src.train_baseline import FEATURE_COLUMNS, WEATHER_COLUMNS


class FixedPipeline:
    def predict(self, frame):
        return frame["lag_24"].to_numpy() + 1.0


class UnseenSiteEvaluationTests(unittest.TestCase):
    def test_model_and_benchmark_use_the_same_strict_rows(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="h"),
                "building_id": ["Gator_A"] * 4,
                "electricity": [10.0, 20.0, np.nan, 41.0],
                "lag_1": [10.0, 11.0, 21.0, 30.0],
                "lag_24": [10.0, 20.0, 30.0, np.nan],
                "rolling_24_mean": [10.0, 10.5, 14.0, 20.0],
            }
        )
        for column in WEATHER_COLUMNS:
            frame[column] = 1.0
        for column in FEATURE_COLUMNS:
            if column not in frame:
                frame[column] = 1.0

        predictions, metrics = evaluate_fixed_model(frame, FixedPipeline())

        self.assertEqual(len(predictions), 2)
        self.assertEqual(metrics["eligible_rows"], 2)
        self.assertEqual(metrics["model"]["mae"], 1.0)
        self.assertEqual(metrics["benchmark"]["mae"], 0.0)
        self.assertEqual(predictions["timestamp"].tolist(), frame["timestamp"].iloc[:2].tolist())


if __name__ == "__main__":
    unittest.main()
