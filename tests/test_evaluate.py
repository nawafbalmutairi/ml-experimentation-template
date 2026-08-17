from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.evaluate import (
    classification_metrics,
    evaluate_model,
    regression_metrics,
    save_metrics,
)
from src.feature_engineer import split_features_and_target
from src.train import build_pipeline
from tests.conftest import TARGET


def test_classification_metrics_on_perfect_predictions() -> None:
    target = pd.Series([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])

    metrics = classification_metrics(target, predictions, np.array([0.1, 0.9, 0.2, 0.8]))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_omit_roc_auc_without_probabilities() -> None:
    target = pd.Series([0, 1])

    metrics = classification_metrics(target, np.array([0, 1]), None)

    assert "roc_auc" not in metrics


def test_regression_metrics_measure_error() -> None:
    target = pd.Series([1.0, 2.0, 3.0])
    predictions = np.array([1.0, 2.0, 5.0])

    metrics = regression_metrics(target, predictions)

    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(4.0 / 3.0))
    assert metrics["r2"] < 1.0


def test_save_metrics_writes_a_json_file_at_the_returned_path(config: Config) -> None:
    metrics = {"accuracy": 0.75}

    written_path = save_metrics(metrics, config.evaluation, context={"task": "classification"})

    assert written_path.parent == config.evaluation.results_dir
    payload = json.loads(written_path.read_text(encoding="utf-8"))
    assert payload["metrics"] == metrics
    assert payload["task"] == "classification"


def test_evaluate_model_rejects_metrics_that_do_not_apply(
    config: Config, training_frame: pd.DataFrame
) -> None:
    features, target = split_features_and_target(
        training_frame, config.features.all_features, TARGET
    )
    pipeline = build_pipeline(config)
    pipeline.fit(features, target)

    with pytest.raises(ValueError, match="mae"):
        evaluate_model(pipeline, features, target, config.model.task, ["accuracy", "mae"])
