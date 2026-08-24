from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from ml_template.config import CLASSIFICATION, REGRESSION, ModelConfig

MODEL_FACTORIES: dict[tuple[str, str], Callable[..., BaseEstimator]] = {
    ("xgboost", CLASSIFICATION): XGBClassifier,
    ("xgboost", REGRESSION): XGBRegressor,
    # A linear reference point to beat before a booster is justified. It accepts raw class
    # labels, so it stays out of LABEL_ENCODED_MODEL_TYPES below.
    ("logistic_regression", CLASSIFICATION): LogisticRegression,
}

# Model types that reject raw class labels and need them encoded as 0..n-1.
LABEL_ENCODED_MODEL_TYPES = frozenset({"xgboost"})


def create_model(config: ModelConfig, seed: int) -> BaseEstimator:
    factory = MODEL_FACTORIES.get((config.type, config.task))
    if factory is None:
        supported = sorted(f"{model_type}/{task}" for model_type, task in MODEL_FACTORIES)
        raise ValueError(
            f"Unsupported model '{config.type}' for task '{config.task}'. Supported: {supported}"
        )
    return factory(random_state=seed, **config.params)
