from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ml_template.config import CLASSIFICATION, REGRESSION, EvaluationConfig
from ml_template.predict import positive_class_scores


def evaluate_model(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    task: str,
    metric_names: Sequence[str],
) -> dict[str, float]:
    predictions = pipeline.predict(features)

    if task == CLASSIFICATION:
        probabilities = positive_class_scores(pipeline, features)
        available = classification_metrics(
            target, predictions, probabilities, labels=list(pipeline[-1].classes_)
        )
    elif task == REGRESSION:
        available = regression_metrics(target, predictions)
    else:
        raise ValueError(f"Unsupported task: {task}")

    unknown = [name for name in metric_names if name not in available]
    if unknown:
        raise ValueError(
            f"Unsupported metrics for task '{task}': {unknown}. Available: {sorted(available)}"
        )
    return {name: available[name] for name in metric_names}


def classification_metrics(
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Metrics for any label type.

    `labels` is the class set the model was fitted on, and it decides both the averaging mode and
    which class counts as positive. Inferring it from `target` instead lets the test split rewrite
    the contract: a split that happens to hold one class would be scored as a one-class problem,
    and a model predicting nothing but that class would report a perfect precision and recall.
    It defaults to the classes present in the target and the predictions, for callers scoring two
    plain arrays without a fitted estimator to hand.

    Binary targets are compared against the highest label, which matches the column
    `predict_proba` returns second and the class scikit-learn treats as positive. Labels are
    ordered numerically, not textually: text order would make 2 the positive class in a {2, 10}
    target. Targets with more than two classes are averaged with `macro`, weighting every class
    equally.
    """
    observed = set(pd.unique(target))
    if len(observed) < 2:
        raise ValueError(
            f"The evaluated split holds a single class {sorted(observed, key=repr)}, so precision, "
            "recall and F1 on it measure nothing: a model that predicts only that class scores a "
            "perfect 1.0. Enlarge split.test_size, or re-split the data."
        )

    known = _known_labels(target, predictions) if labels is None else sorted(labels)
    is_binary = len(known) == 2
    average = "binary" if is_binary else "macro"

    if is_binary:
        actual = (target == known[-1]).astype(int)
        predicted = (np.asarray(predictions) == known[-1]).astype(int)
        metric_labels: list[Any] = [0, 1]
    else:
        actual, predicted = target, predictions
        metric_labels = known

    scoring = {"labels": metric_labels, "average": average, "zero_division": 0}
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(actual, predicted, **scoring)),
        "recall": float(recall_score(actual, predicted, **scoring)),
        "f1": float(f1_score(actual, predicted, **scoring)),
    }
    if probabilities is not None and is_binary:
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
        # Precision-recall AUC. On imbalanced targets this is far more informative than ROC-AUC,
        # whose baseline is 0.5 regardless of how rare the positive class is.
        metrics["average_precision"] = float(average_precision_score(actual, probabilities))
    return metrics


def _known_labels(target: pd.Series, predictions: np.ndarray) -> list[Any]:
    """Every class either side names, so one that is only ever predicted still counts."""
    return sorted(set(pd.unique(target)) | set(pd.unique(np.asarray(predictions))))


def regression_metrics(target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
        "r2": float(r2_score(target, predictions)),
    }


def save_metrics(
    metrics: dict[str, float],
    config: EvaluationConfig,
    context: dict[str, Any],
) -> Path:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    output_path = config.results_dir / f"metrics-{timestamp:%Y%m%dT%H%M%SZ}.json"
    payload = {"timestamp": timestamp.isoformat(), **context, "metrics": metrics}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
