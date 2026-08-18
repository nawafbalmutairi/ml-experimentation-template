from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_template.config import Config, default_config_path, load_config
from ml_template.data_processor import load_dataset, validate_columns


def load_pipeline(path: Path) -> Pipeline:
    if not path.is_file():
        raise FileNotFoundError(f"Trained pipeline not found: {path}. Run training first.")
    pipeline: Pipeline = joblib.load(path)
    return pipeline


def positive_class_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray | None:
    """Probability of the positive class, or None when the model cannot produce one.

    The positive class is the highest label, matching the column `predict_proba` returns second.
    Only binary classification yields a score: regression already predicts the value, and a single
    number cannot rank more than two classes.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(features)
    if probabilities.shape[1] != 2:
        return None
    return np.asarray(probabilities[:, 1])


def predict(
    pipeline: Pipeline, frame: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Add a `prediction` column, and a `score` column whenever the model can rank.

    Targeting and prioritisation need the score rather than the label: a hard yes/no cannot say
    who to call first, and thresholding at 0.5 is a decision the caller should own.
    """
    validate_columns(frame, feature_columns)
    features = frame[list(feature_columns)]
    predicted = frame.assign(prediction=pipeline.predict(features))

    scores = positive_class_scores(pipeline, features)
    return predicted if scores is None else predicted.assign(score=scores)


def run_prediction(config: Config, input_path: Path, output_path: Path) -> pd.DataFrame:
    pipeline = load_pipeline(config.model.output_path)
    frame = load_dataset(input_path, config.data.separator)
    predicted = predict(pipeline, frame, config.features.all_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(output_path, index=False)
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with the trained model pipeline.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--input", type=Path, required=True, help="CSV file with feature columns.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV file.")
    args = parser.parse_args()

    config = load_config(args.config)
    predicted = run_prediction(config, args.input, args.output)
    print(f"Wrote {len(predicted)} predictions to {args.output}")


if __name__ == "__main__":
    main()
