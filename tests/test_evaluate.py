from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml_template.config import Config
from ml_template.evaluate import (
    classification_metrics,
    evaluate_model,
    regression_metrics,
    save_metrics,
)
from ml_template.feature_engineer import split_features_and_target
from ml_template.train import build_pipeline
from tests.conftest import TARGET


def test_classification_metrics_on_perfect_predictions() -> None:
    target = pd.Series([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])

    metrics = classification_metrics(target, predictions, np.array([0.1, 0.9, 0.2, 0.8]))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_average_precision_tracks_the_positive_rate_on_random_scores() -> None:
    """Unlike ROC-AUC, the PR-AUC baseline is the positive rate, which is why it is reported."""
    target = pd.Series([1] + [0] * 9)
    tied_scores = np.full(10, 0.5)

    metrics = classification_metrics(target, np.zeros(10, dtype=int), tied_scores)

    assert metrics["average_precision"] == pytest.approx(0.1)
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_classification_metrics_distinguish_precision_from_recall() -> None:
    """Two false positives and one false negative, so the two metrics cannot coincide."""
    target = pd.Series([0, 0, 1, 1, 1])
    predictions = np.array([1, 1, 1, 1, 0])

    metrics = classification_metrics(target, predictions, None)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(4 / 7)
    assert metrics["accuracy"] == pytest.approx(0.4)


def test_classification_metrics_score_the_higher_label_as_positive() -> None:
    target = pd.Series(["no", "no", "yes", "yes", "yes"])
    predictions = np.array(["yes", "yes", "yes", "yes", "no"])

    metrics = classification_metrics(target, predictions, None)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(0.4)


def test_classification_metrics_handle_string_labels_for_roc_auc() -> None:
    target = pd.Series(["no", "yes", "no", "yes"])
    predictions = np.array(["no", "yes", "no", "yes"])

    metrics = classification_metrics(target, predictions, np.array([0.1, 0.9, 0.2, 0.8]))

    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_average_over_classes_when_not_binary() -> None:
    """Class supports are unequal, so macro averaging differs from weighted and micro."""
    target = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    predictions = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0])

    metrics = classification_metrics(target, predictions, None)

    per_class_f1 = (2 / 3, 6 / 7, 10 / 11)
    assert metrics["f1"] == pytest.approx(sum(per_class_f1) / 3)
    assert metrics["precision"] == pytest.approx((2 / 3 + 3 / 4 + 1.0) / 3)
    assert metrics["recall"] == pytest.approx((2 / 3 + 1.0 + 5 / 6) / 3)
    assert metrics["accuracy"] == pytest.approx(10 / 12)
    assert "roc_auc" not in metrics


def test_binary_labels_use_numeric_order_not_text_order() -> None:
    """Text order would sort {2, 10} as [10, 2] and score 2 as the positive class."""
    target = pd.Series([2, 10, 2, 10])
    predictions = np.array([2, 10, 2, 10])

    metrics = classification_metrics(target, predictions, np.array([0.1, 0.9, 0.2, 0.8]))

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_omit_ranking_metrics_without_probabilities() -> None:
    target = pd.Series([0, 1])

    metrics = classification_metrics(target, np.array([0, 1]), None)

    assert "roc_auc" not in metrics
    assert "average_precision" not in metrics


def test_metrics_use_every_class_the_model_knows_not_only_those_in_the_split() -> None:
    """The split holds two of three classes; scoring it as a binary problem would be wrong."""
    target = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 2])

    metrics = classification_metrics(target, predictions, None, labels=[0, 1, 2])

    assert metrics["precision"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert metrics["recall"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_classification_metrics_reject_a_split_holding_a_single_class() -> None:
    """All-negative test rows and all-negative predictions used to report a perfect 1.0."""
    target = pd.Series([0] * 40)

    with pytest.raises(ValueError, match="single class"):
        classification_metrics(target, np.zeros(40, dtype=int), None, labels=[0, 1])


def test_evaluate_model_refuses_a_degenerate_test_split(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """A model that learned nothing must not land a perfect score in experiments/results."""
    features, target = split_features_and_target(
        training_frame, config.features.all_features, TARGET
    )
    pipeline = build_pipeline(config)
    pipeline.fit(features, target)
    negatives = target == 0

    with pytest.raises(ValueError, match="single class"):
        evaluate_model(
            pipeline,
            features[negatives],
            target[negatives],
            config.model.task,
            ["precision", "recall", "f1"],
        )


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
