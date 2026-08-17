from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

CLASSIFICATION = "classification"
REGRESSION = "regression"
SUPPORTED_TASKS = (CLASSIFICATION, REGRESSION)


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_path: Path
    target: str


@dataclass(frozen=True)
class FeatureConfig:
    numerical: list[str]
    categorical: list[str]

    @property
    def all_features(self) -> list[str]:
        return [*self.numerical, *self.categorical]


@dataclass(frozen=True)
class SplitConfig:
    test_size: float


@dataclass(frozen=True)
class ModelConfig:
    task: str
    type: str
    output_path: Path
    params: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    results_dir: Path
    metrics: list[str]


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig


def default_config_path() -> Path:
    configured = os.environ.get("ML_CONFIG_PATH")
    return resolve_path(configured) if configured else DEFAULT_CONFIG_PATH


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path | None = None) -> Config:
    config_path = path or default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> Config:
    data = _section(raw, "data")
    features = _section(raw, "features")
    split = _section(raw, "split")
    model = _section(raw, "model")
    evaluation = _section(raw, "evaluation")

    task = str(model["task"])
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {list(SUPPORTED_TASKS)}")

    test_size = float(split["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"split.test_size must be between 0 and 1, got {test_size}")

    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(
            raw_path=resolve_path(data["raw_path"]),
            processed_path=resolve_path(data["processed_path"]),
            target=str(data["target"]),
        ),
        features=FeatureConfig(
            numerical=list(features["numerical"]),
            categorical=list(features["categorical"]),
        ),
        split=SplitConfig(test_size=test_size),
        model=ModelConfig(
            task=task,
            type=str(model["type"]),
            output_path=resolve_path(model["output_path"]),
            params=dict(model["params"]),
        ),
        evaluation=EvaluationConfig(
            results_dir=resolve_path(evaluation["results_dir"]),
            metrics=list(evaluation["metrics"]),
        ),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in config")
    return section
