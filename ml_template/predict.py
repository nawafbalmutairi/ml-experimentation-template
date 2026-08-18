from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_template.config import Config, default_config_path, load_config
from ml_template.data_processor import load_dataset, validate_columns


def load_pipeline(path: Path) -> Pipeline:
    if not path.is_file():
        raise FileNotFoundError(f"Trained pipeline not found: {path}. Run training first.")
    pipeline: Pipeline = joblib.load(path)
    return pipeline


def predict(
    pipeline: Pipeline, frame: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    validate_columns(frame, feature_columns)
    predictions = pipeline.predict(frame[list(feature_columns)])
    return frame.assign(prediction=predictions)


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
