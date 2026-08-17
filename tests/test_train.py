from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from src.config import Config
from src.data_processor import save_dataset
from src.predict import load_pipeline, predict, run_prediction
from src.train import run_training


def train_on(config: Config, frame: pd.DataFrame) -> dict[str, float]:
    save_dataset(frame, config.data.raw_path)
    return run_training(config)


def test_training_reports_the_configured_metrics(
    config: Config, training_frame: pd.DataFrame
) -> None:
    metrics = train_on(config, training_frame)

    assert list(metrics) == config.evaluation.metrics
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_training_learns_a_signal(config: Config, training_frame: pd.DataFrame) -> None:
    metrics = train_on(config, training_frame)

    assert metrics["accuracy"] > 0.8


def test_training_writes_processed_data_and_metrics(
    config: Config, training_frame: pd.DataFrame
) -> None:
    metrics = train_on(config, training_frame)

    assert config.data.processed_path.is_file()

    result_files = list(config.evaluation.results_dir.glob("metrics-*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["metrics"] == metrics
    assert payload["model_type"] == config.model.type
    assert payload["train_rows"] + payload["test_rows"] == len(training_frame)


def test_saved_pipeline_predicts_after_reloading(
    config: Config, training_frame: pd.DataFrame
) -> None:
    train_on(config, training_frame)
    assert config.model.output_path.is_file()

    pipeline = load_pipeline(config.model.output_path)
    unseen = training_frame.drop(columns=[config.data.target]).head(5)
    predicted = predict(pipeline, unseen, config.features.all_features)

    assert len(predicted) == 5
    assert set(predicted["prediction"]).issubset({0, 1})


def test_prediction_writes_a_csv_with_the_original_columns(
    config: Config, training_frame: pd.DataFrame
) -> None:
    train_on(config, training_frame)
    input_path = config.data.raw_path.parent / "inference.csv"
    output_path = config.data.raw_path.parent / "predictions.csv"
    save_dataset(training_frame.drop(columns=[config.data.target]).head(10), input_path)

    run_prediction(config, input_path, output_path)

    written = pd.read_csv(output_path)
    assert len(written) == 10
    assert "prediction" in written.columns


def test_regression_training_reports_regression_metrics(
    config: Config, training_frame: pd.DataFrame
) -> None:
    regression_config = replace(
        config,
        data=replace(config.data, target="monthly_charges"),
        features=replace(config.features, numerical=["tenure_months"]),
        model=replace(config.model, task="regression"),
        evaluation=replace(config.evaluation, metrics=["mae", "rmse", "r2"]),
    )

    metrics = train_on(regression_config, training_frame)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert "accuracy" not in metrics
