import unittest

import numpy as np
import pandas as pd

from src.phase2_extensions import (
    add_meter_quality_flags,
    build_audit_ranking,
    build_retrofit_scenarios,
    summarize_meter_quality,
)


class Phase2ExtensionTests(unittest.TestCase):
    def test_meter_flags_distinguish_missing_gaps_and_spikes(self):
        timestamps = pd.date_range("2020-01-01", periods=60, freq="h")
        electricity = np.full(60, 10.0)
        electricity[5:30] = np.nan
        electricity[45] = 100.0
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "building_id": "B1",
                "electricity": electricity,
            }
        )

        flags = add_meter_quality_flags(data, long_gap_threshold_hours=24)
        summary = summarize_meter_quality(flags).iloc[0]

        self.assertEqual(int(summary["missing_hours"]), 25)
        self.assertEqual(int(summary["long_missing_gap_count"]), 1)
        self.assertEqual(int(summary["long_missing_gap_hours"]), 25)
        self.assertGreaterEqual(int(summary["suspicious_spike_count"]), 1)
        self.assertEqual(int(summary["confirmed_faulty_count"]), 0)

    def test_audit_ranking_requires_observed_threshold_and_sustained_residuals(self):
        timestamps = pd.date_range("2020-01-01", periods=12, freq="h")
        predictions = pd.DataFrame(
            {
                "timestamp": list(timestamps) * 2,
                "building_id": ["eligible"] * 12 + ["short"] * 12,
                "electricity": [12.0] * 8 + [8.0] * 4 + [12.0] + [8.0] * 11,
                "model_prediction": [10.0] * 8 + [8.0] * 4 + [10.0] + [8.0] * 11,
            }
        )

        ranking = build_audit_ranking(
            predictions, min_observed_hours=10, min_streak_hours=3
        )
        eligible = ranking.set_index("building_id").loc["eligible"]
        short = ranking.set_index("building_id").loc["short"]

        self.assertTrue(bool(eligible["audit_candidate"]))
        self.assertEqual(int(eligible["sustained_positive_hours"]), 8)
        self.assertFalse(bool(short["audit_candidate"]))

    def test_retrofit_scenarios_are_separate_calculations(self):
        scenarios = build_retrofit_scenarios(pd.DataFrame({"electricity": [10.0] * 24}))

        self.assertEqual(scenarios["assumed_reduction_pct"].tolist(), [10.0, 8.0])
        self.assertEqual(scenarios["estimated_energy_reduction"].tolist(), [24.0, 19.2])
        self.assertEqual(scenarios["scenario_energy"].tolist(), [216.0, 220.8])


if __name__ == "__main__":
    unittest.main()
