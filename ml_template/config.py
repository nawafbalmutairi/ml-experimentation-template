from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT_ENV = "ML_PROJECT_ROOT"
CONFIG_PATH_ENV = "ML_CONFIG_PATH"

CLASSIFICATION = "classification"
REGRESSION = "regression"
SUPPORTED_TASKS = (CLASSIFICATION, REGRESSION)

RANDOM_SPLIT = "random"
TEMPORAL_SPLIT = "temporal"
SUPPORTED_SPLIT_STRATEGIES = (RANDOM_SPLIT, TEMPORAL_SPLIT)


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_path: Path
    target: str
    separator: str = ","
    target_mapping: dict[Any, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureConfig:
    numerical: list[str]
    categorical: list[str]
    # Columns naming a group whose members tend to share an outcome - a household, a ticket, an
    # order. Encoded by the group's target rate rather than used raw. See ml_template/group_encoder.
    group_keys: list[str] = field(default_factory=list)

    @property
    def all_features(self) -> list[str]:
        return [*self.numerical, *self.categorical, *self.group_keys]


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    strategy: str = RANDOM_SPLIT


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
class ValueRange:
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ValidationConfig:
    min_rows: int
    max_missing_fraction: float
    max_duplicate_fraction: float
    min_class_fraction: float
    report_dir: Path
    value_ranges: dict[str, ValueRange]


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    validation: ValidationConfig


def project_root() -> Path:
    """Directory that relative paths in the config resolve against.

    The working directory, so a copied project always resolves its own data regardless of where
    the package itself is installed. Deriving this from the package location breaks as soon as the
    package is installed non-editable, when it would point inside site-packages. Set
    ML_PROJECT_ROOT to run from somewhere other than the project directory.
    """
    configured = os.environ.get(PROJECT_ROOT_ENV)
    return Path(configured).resolve() if configured else Path.cwd()


def default_config_path() -> Path:
    configured = os.environ.get(CONFIG_PATH_ENV)
    return resolve_path(configured) if configured else project_root() / "config" / "config.yaml"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


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
    validation = _section(raw, "validation")

    task = str(model["task"])
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {list(SUPPORTED_TASKS)}")

    test_size = float(split["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"split.test_size must be between 0 and 1, got {test_size}")

    strategy = str(split.get("strategy", RANDOM_SPLIT))
    if strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported split.strategy '{strategy}'. "
            f"Supported: {list(SUPPORTED_SPLIT_STRATEGIES)}"
        )

    target = str(data["target"])
    feature_config = _build_feature_config(features, target)

    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(
            raw_path=resolve_path(data["raw_path"]),
            processed_path=resolve_path(data["processed_path"]),
            target=target,
            separator=str(data.get("separator", ",")),
            target_mapping=_build_target_mapping(data.get("target_mapping") or {}),
        ),
        features=feature_config,
        split=SplitConfig(test_size=test_size, strategy=strategy),
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
        validation=_build_validation_config(validation, feature_config.numerical),
    )


def _build_validation_config(
    validation: dict[str, Any],
    numerical_features: list[str],
) -> ValidationConfig:
    min_rows = int(validation["min_rows"])
    if min_rows < 1:
        raise ValueError(f"validation.min_rows must be at least 1, got {min_rows}")

    return ValidationConfig(
        min_rows=min_rows,
        max_missing_fraction=_fraction(validation, "max_missing_fraction"),
        max_duplicate_fraction=_fraction(validation, "max_duplicate_fraction"),
        min_class_fraction=_fraction(validation, "min_class_fraction"),
        report_dir=resolve_path(validation["report_dir"]),
        value_ranges=_build_value_ranges(validation.get("value_ranges") or {}, numerical_features),
    )


def _build_feature_config(features: dict[str, Any], target: str) -> FeatureConfig:
    numerical = list(features["numerical"])
    categorical = list(features["categorical"])
    group_keys = list(features.get("group_keys") or [])

    declared = numerical + categorical + group_keys
    if target in declared:
        raise ValueError(
            f"data.target '{target}' is also listed under features, which would train the model "
            "on the answer. Remove it from features.numerical, categorical and group_keys."
        )
    duplicates = sorted({name for name in declared if declared.count(name) > 1})
    if duplicates:
        raise ValueError(f"Columns declared under more than one feature kind: {duplicates}")

    return FeatureConfig(numerical=numerical, categorical=categorical, group_keys=group_keys)


def _build_target_mapping(raw_mapping: dict[Any, Any]) -> dict[Any, int]:
    """Keys keep the type YAML gave them, so they can match the labels in the column.

    Coercing them to strings would leave a numeric target's 1 and 2 unmatchable by '1' and '2',
    and `encode_target` would report the mapping as having no entry for labels the config
    visibly maps.
    """
    boolean_keys = [label for label in raw_mapping if isinstance(label, bool)]
    if boolean_keys:
        raise ValueError(
            "data.target_mapping keys must be quoted: YAML reads yes, no, true and false as "
            "booleans, so write 'yes': 1 rather than yes: 1"
        )
    return {label: int(code) for label, code in raw_mapping.items()}


def _fraction(validation: dict[str, Any], name: str) -> float:
    value = float(validation[name])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"validation.{name} must be between 0 and 1, got {value}")
    return value


def _build_value_ranges(
    raw_ranges: dict[str, Any],
    numerical_features: list[str],
) -> dict[str, ValueRange]:
    """Bounds only apply to numeric columns, so a key naming anything else is a typo.

    Accepting it would leave the bound silently switched off — configuration that looks present
    and does nothing.
    """
    unknown = sorted(set(raw_ranges) - set(numerical_features))
    if unknown:
        raise ValueError(
            f"validation.value_ranges names columns that are not numerical features: {unknown}. "
            f"Available: {numerical_features}"
        )

    ranges = {}
    for column, bounds in raw_ranges.items():
        if not isinstance(bounds, dict):
            raise ValueError(f"validation.value_ranges.{column} must be a mapping of min/max")
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"validation.value_ranges.{column} has min greater than max")
        ranges[column] = ValueRange(
            minimum=None if minimum is None else float(minimum),
            maximum=None if maximum is None else float(maximum),
        )
    return ranges


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in config")
    return section
