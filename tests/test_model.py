from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from xgboost import XGBClassifier, XGBRegressor

from ml_template.config import Config, ModelConfig
from ml_template.model import create_model


def model_config(task: str, tmp_path: Path, **params: object) -> ModelConfig:
    return ModelConfig(
        task=task,
        type="xgboost",
        output_path=tmp_path / "pipeline.joblib",
        params=dict(params),
    )


def test_create_model_returns_classifier_for_classification(tmp_path: Path) -> None:
    model = create_model(model_config("classification", tmp_path), seed=42)

    assert isinstance(model, XGBClassifier)


def test_create_model_returns_regressor_for_regression(tmp_path: Path) -> None:
    model = create_model(model_config("regression", tmp_path), seed=42)

    assert isinstance(model, XGBRegressor)


def test_create_model_applies_configured_hyperparameters(tmp_path: Path) -> None:
    model = create_model(
        model_config("classification", tmp_path, n_estimators=7, max_depth=3), seed=99
    )

    assert model.get_params()["n_estimators"] == 7
    assert model.get_params()["max_depth"] == 3
    assert model.get_params()["random_state"] == 99


def test_create_model_rejects_unknown_model_type(config: Config) -> None:
    unknown = replace(config.model, type="not-a-model")

    with pytest.raises(ValueError, match="not-a-model"):
        create_model(unknown, seed=42)
