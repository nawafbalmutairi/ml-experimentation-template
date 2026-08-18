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


def evaluate_model(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    task: str,
    metric_names: Sequence[str],
) -> dict[str, float]:
    predictions = pipeline.predict(features)

    if task == CLASSIFICATION:
        probabilities = _positive_class_probabilities(pipeline, features)
        available = classification_metrics(target, predictions, probabilities)
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
) -> dict[str, float]:
    """Metrics for any label type.

    Binary targets are compared against the highest label, which matches the column
    `predict_proba` returns second and the class scikit-learn treats as positive. Labels are
    ordered numerically, not textually: text order would make 2 the positive class in a {2, 10}
    target. Targets with more than two classes are averaged with `macro`, weighting every class
    equally.
    """
    labels = sorted(pd.unique(target))
    is_binary = len(labels) == 2
    average = "binary" if is_binary else "macro"

    if is_binary:
        actual = (target == labels[-1]).astype(int)
        predicted = (np.asarray(predictions) == labels[-1]).astype(int)
    else:
        actual, predicted = target, predictions

    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(actual, predicted, average=average, zero_division=0)),
        "recall": float(recall_score(actual, predicted, average=average, zero_division=0)),
        "f1": float(f1_score(actual, predicted, average=average, zero_division=0)),
    }
    if probabilities is not None and is_binary:
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
        # Precision-recall AUC. On imbalanced targets this is far more informative than ROC-AUC,
        # whose baseline is 0.5 regardless of how rare the positive class is.
        metrics["average_precision"] = float(average_precision_score(actual, probabilities))
    return metrics


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


def _positive_class_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(features)
    if probabilities.shape[1] != 2:
        return None
    return np.asarray(probabilities[:, 1])
