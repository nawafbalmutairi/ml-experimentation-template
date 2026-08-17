from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor

from src.config import CLASSIFICATION, REGRESSION, ModelConfig

MODEL_FACTORIES: dict[tuple[str, str], Callable[..., BaseEstimator]] = {
    ("xgboost", CLASSIFICATION): XGBClassifier,
    ("xgboost", REGRESSION): XGBRegressor,
}


def create_model(config: ModelConfig, seed: int) -> BaseEstimator:
    factory = MODEL_FACTORIES.get((config.type, config.task))
    if factory is None:
        supported = sorted(f"{model_type}/{task}" for model_type, task in MODEL_FACTORIES)
        raise ValueError(
            f"Unsupported model '{config.type}' for task '{config.task}'. Supported: {supported}"
        )
    return factory(random_state=seed, **config.params)
