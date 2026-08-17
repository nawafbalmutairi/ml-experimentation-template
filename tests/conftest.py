from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_template.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    FeatureConfig,
    ModelConfig,
    SplitConfig,
    ValidationConfig,
)

NUMERICAL_FEATURES = ["tenure_months", "monthly_charges"]
CATEGORICAL_FEATURES = ["contract_type"]
TARGET = "churned"


@pytest.fixture
def training_frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    size = 120
    tenure = generator.integers(1, 72, size=size)
    charges = generator.uniform(20, 120, size=size)
    contract = generator.choice(["month-to-month", "one-year"], size=size)
    churned = ((contract == "month-to-month") & (tenure < 40)).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure.astype(float),
            "monthly_charges": charges,
            "contract_type": contract,
            TARGET: churned,
        }
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        seed=42,
        data=DataConfig(
            raw_path=tmp_path / "raw.csv",
            processed_path=tmp_path / "processed.csv",
            target=TARGET,
        ),
        features=FeatureConfig(
            numerical=list(NUMERICAL_FEATURES),
            categorical=list(CATEGORICAL_FEATURES),
        ),
        split=SplitConfig(test_size=0.25),
        model=ModelConfig(
            task="classification",
            type="xgboost",
            output_path=tmp_path / "models" / "pipeline.joblib",
            params={"n_estimators": 10, "max_depth": 2},
        ),
        evaluation=EvaluationConfig(
            results_dir=tmp_path / "results",
            metrics=["accuracy", "f1", "roc_auc"],
        ),
        validation=ValidationConfig(
            min_rows=10,
            max_missing_fraction=0.2,
            max_duplicate_fraction=0.05,
            min_class_fraction=0.05,
            report_dir=tmp_path / "data_quality",
            value_ranges={},
        ),
    )
