"""Reusable prediction helper for the saved Enpres.x baseline."""

from pathlib import Path

import joblib
import pandas as pd

from train_baseline import build_features, load_selected_data


WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_PATH = WORKSPACE / "models" / "panther_baseline.joblib"


def predict_frame(frame: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Return model predictions for a prepared-data frame with history columns."""
    artifact = joblib.load(model_path)
    featured = build_features(frame)
    featured["prediction"] = artifact["pipeline"].predict(
        featured[artifact["feature_columns"]]
    )
    return featured


def predict_selected_dataset(model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Load the five baseline buildings and return their model predictions."""
    return predict_frame(load_selected_data(), model_path=model_path)
